from typing import Type

from traincheck.invariant.consistency_relation import ConsistencyRelation
from traincheck.invariant.consistency_transient_vars import (
    ConsistentInputOutputRelation,
    ConsistentOutputRelation,
    ThresholdRelation,
)
from traincheck.invariant.contain_relation import APIContainRelation
from traincheck.invariant.cover_relation import FunctionCoverRelation
from traincheck.invariant.DistinctArgumentRelation import DistinctArgumentRelation
from traincheck.invariant.lead_relation import FunctionLeadRelation

# from traincheck.invariant.var_periodic_change_relation import VarPeriodicChangeRelation

relation_pool: list[Type] = [
    APIContainRelation,
    ConsistencyRelation,
    ConsistentOutputRelation,
    ConsistentInputOutputRelation,
    #    VarPeriodicChangeRelation,
    FunctionCoverRelation,
    FunctionLeadRelation,
    DistinctArgumentRelation,
    ThresholdRelation,
]
