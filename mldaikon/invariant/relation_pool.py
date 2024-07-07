from typing import Type

from mldaikon.invariant.consistency_relation import ConsistencyRelation
from mldaikon.invariant.contain_relation import APIContainRelation

relation_pool: list[Type] = [
    APIContainRelation,
    ConsistencyRelation,
]
