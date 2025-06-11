from traincheck.invariant import Invariant, read_inv_file
from tqdm import tqdm
from traincheck.invariant.base_cls import (
    APIParam,
    Arguments,
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    Hypothesis,
    Invariant,
    Param,
    Relation,
    VarNameParam,
    VarTypeParam,
    calc_likelihood,
    construct_api_param,
    construct_var_param_from_var_change,
    is_signature_empty,
)
import json
from traincheck.trace import MDNONEJSONEncoder

def check(invariants: str):
    invs = read_inv_file(invariants)
    param_to_invs : dict[Param, list[Invariant]] = {}
    print(len(invs))
    for inv in invs:
        assert (
            inv.precondition is not None
        ), "Invariant precondition is None. It should at least be 'Unconditional' or an empty list. Please check the invariant file and the inference process."
        params = inv.relation.get_mapping_key(inv)
        for param in params:
            if param not in param_to_invs:
                param_to_invs[param] = []
            param_to_invs[param].append(inv)

    # with open("./test.txt", "w") as f:     
    #     for param, invs_ in param_to_invs.items():
    #         if isinstance(param, APIParam):
    #             f.write(param.api_full_name)
    #         elif isinstance(param, VarNameParam):
    #             f.write(param.var_name)
    #         elif isinstance(param, VarTypeParam):
    #             f.write(param.var_type)
    #         for inv in invs_:
    #             f.write(json.dumps(inv.to_dict(), cls=MDNONEJSONEncoder))
    #             f.write("\n")

    

                


if __name__ == "__main__":
    check("/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/invariants.json")