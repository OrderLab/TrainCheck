from typing import Type

from mldaikon.invariant.consistency_relation import ConsistencyRelation
from mldaikon.invariant.contain_relation import APIContainRelation
from mldaikon.invariant.cover_relation import FunctionCoverRelation
relation_pool: list[Type] = [
    # APIContainRelation,
    # ConsistencyRelation,
    FunctionCoverRelation,
]
