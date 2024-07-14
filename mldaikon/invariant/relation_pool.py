from mldaikon.invariant.consistency_relation import ConsistencyRelation
from mldaikon.invariant.contain_relation import APIContainRelation
from mldaikon.invariant.var_periodic_change_relation import VarPeriodicChangeRelation

relation_pool = [
    APIContainRelation,
    ConsistencyRelation,
    VarPeriodicChangeRelation,
]
