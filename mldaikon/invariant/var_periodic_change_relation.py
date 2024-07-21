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
    GroupedPreconditions,
    PT,
    PreconditionClause
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
        hypo_value = "None"   # TODO: how to represent None, 
    return hypo_value
    
    
class VarPeriodicChangeRelation(Relation):
        
    use_varType = False
    
    
    def count_num_juistification(hypothesis, count: int):
        # TODO: modify this based on the histo_log
        return count > 1

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants for the VariableChangeRelation."""
        
        logger = logging.getLogger(__name__)
        ## 1. Pre-scanning: Collecting variable instances and their values from the trace
        # get identifiers of the variables, those variables can be used to query the actual values
        var_insts = trace.get_var_insts()
        if len(var_insts) == 0:
            logger.warning("No variables found in the trace.")
            return []
        
        hypothesis: dict[str, dict[str, dict[str, Hypothesis]]] = {}
        ## 2.Counting: count the number of each value of every variable attribute
        
        for var_id, attrs in var_insts.items():
            for attr_name, attr_insts in attrs.items():
                for attr_inst in attr_insts:
                    hypo_value = calculate_hypo_value(attr_inst.value)
                    var_key = var_id.var_name
                    group_names = var_key + '.' + attr_name + '.' + hypo_value
                    if VarPeriodicChangeRelation.use_varType:
                        var_key = var_id.var_type
                        group_names = "var"
                    
                    example = Example()
                    example.add_group(group_names, attr_inst.traces)
                    # TODO: discuss with Yuxuan how to classify positive and negative examples
                    # Ensure all intermediate keys are properly initialized
                    if var_key not in hypothesis:
                        hypothesis[var_key] = {}
                    if attr_name not in hypothesis[var_key]:
                        hypothesis[var_key][attr_name] = {}
                    if hypo_value not in hypothesis[var_key][attr_name]:
                        hypo = Hypothesis(
                            Invariant(
                                relation=VarPeriodicChangeRelation,
                                params=[],
                                precondition=None,
                            ),
                            positive_examples=ExampleList({group_names}),
                            negative_examples=ExampleList({group_names}),
                        )
                        hypothesis[var_key][attr_name][hypo_value] = hypo
                        hypothesis[var_key][attr_name][hypo_value].negative_examples.add_example(example)
                    else:
                        hypothesis[var_key][attr_name][hypo_value].positive_examples.add_example(example)
                        
                    
        # var type: mark repeated variable as positive and mark thje rest as negative
        # var name: 
        # tmp_positive = {key: value for key, value in hypothesis.items() if value > 1}
        # logger.debug(f"histogram detail: {tmp_positive}")
        
        for var_name in hypothesis:
            for attr_name in hypothesis[var_name]:
                for hypo_value in hypothesis[var_name][attr_name]:
                    hypo = hypothesis[var_name][attr_name][hypo_value]
                    hypo.invariant.precondition = find_precondition(hypo)
                    hypo.invariant.text_description = f"Var Change Relation of {var_id.var_name + '.' + attr_name + '.' + hypo_value}.",
                    
        return list([hypothesis[var_name][attr_name][hypo_value].invariant
                     for var_name in hypothesis
                     for attr_name in hypothesis[var_name]
                     for hypo_value in hypothesis[var_name][attr_name]
                     if hypothesis[var_name][attr_name][hypo_value].invariant.precondition is not None
                     ])
        