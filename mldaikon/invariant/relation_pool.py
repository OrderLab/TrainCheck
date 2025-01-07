from typing import Type

from mldaikon.invariant.consistency_relation import ConsistencyRelation
from mldaikon.invariant.consistency_transient_vars import (
    ConsistentInputOutputRelation,
    ConsistentOutputRelation,
    ThresholdRelation,
)
from mldaikon.invariant.contain_relation import APIContainRelation
from mldaikon.invariant.cover_relation import FunctionCoverRelation
from mldaikon.invariant.DistinctArgumentRelation import DistinctArgumentRelation
from mldaikon.invariant.lead_relation import FunctionLeadRelation
from mldaikon.invariant.var_periodic_change_relation import VarPeriodicChangeRelation

relation_pool: list[Type] = [
    APIContainRelation,
    ConsistencyRelation,
    ConsistentOutputRelation,
    ConsistentInputOutputRelation,
    VarPeriodicChangeRelation,
    FunctionCoverRelation,
    FunctionLeadRelation,
    DistinctArgumentRelation,
    ThresholdRelation,
]
