import logging
from itertools import combinations

from tqdm import tqdm
import numpy as np
from mldaikon.config import config
from mldaikon.invariant.base_cls import (
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    Relation,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace

def calculate_hypo_value(value) -> str:
    if isinstance(value, (int, float)):
        hypo_value = f"{value:.7f}"
    elif isinstance(value, (list)):
        hypo_value = f"{np.linalg.norm(value, ord=1):.7f}"  # l1-norm
    elif isinstance(value, str):
        hypo_value = value
    else:
        hypo_value = ""   # TODO: how to represent None, 
    return hypo_value
    
    
class VarPeriodicChangeRelation(Relation):
    def __init__(self,):
        self.hypothesis = None
        
    def count_num_juistification(self, count: int):
        # TODO: modify this based on the histo_log
        return count > 1

    def infer(self, trace: Trace) -> list[Invariant]:
        """Infer Invariants for the VariableChangeRelation."""

        logger = logging.getLogger(__name__)
        ## 1. Pre-scanning: Collecting variable instances and their values from the trace
        # get identifiers of the variables, those variables can be used to query the actual values
        var_insts = trace.get_var_insts()
        if len(var_insts) == 0:
            logger.warning("No variables found in the trace.")
            return []
        
        if not self.hypothesis:
            self.hypothesis: dict[str, dict[str, dict[str, Hypothesis]]] = {}
        ## 2.Counting: count the number of each value of every variable attribute
        
        for var_id, attrs in var_insts.items():
            for attr_name, attr_insts in attrs.items():
                for attr_inst in attr_insts:
                    hypo_value = calculate_hypo_value(attr_inst.value)
                    group_names = var_id.var_name + '.' + attr_name + '.' + hypo_value
                    example = Example()
                    example.add_group(group_names, attr_inst.traces)
                    # TODO: discuss with Yuxuan how to classify positive and negative examples
                    if self.hypothesis[var_id.var_name][attr_name][hypo_value] in self.hypothesis:
                        hypo = self.hypothesis[var_id.var_name][attr_name][hypo_value]
                        self.hypothesis[var_id.var_name][attr_name][hypo_value].positive_examples.add_example(example)
                    else:
                        self.hypothesis[var_id.var_name][attr_name][hypo_value] = Hypothesis(
                            Invariant(
                                relation=VarPeriodicChangeRelation(),
                                param_selectors=[],
                            ),
                            positive_examples=ExampleList({group_names}),
                            negative_examples=ExampleList({group_names}),
                        )
                        self.hypothesis[var_id.var_name][attr_name][hypo_value].positive_examples.add_example(example)
                        self.hypothesis[var_id.var_name][attr_name][hypo_value].negative_examples.add_example(example)
                    
                        
        # tmp_positive = {key: value for key, value in self.hypothesis.items() if value > 1}
        # logger.debug(f"histogram detail: {tmp_positive}")
        
        return list([hypo.invariant for var_attrs in self.hypothesis.values()
                     for attr_values in var_attrs.values()
                     for hypo in attr_values.values()])