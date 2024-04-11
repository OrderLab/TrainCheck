
from .base_cls import Relation, Invariant
from .contain_relation import APIContainRelation

relation_pool: list[Relation] = [
    APIContainRelation,
]
