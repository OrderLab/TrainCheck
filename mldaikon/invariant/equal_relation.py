import logging
import time
from mldaikon.invariant.base_cls import (
    CheckerResult,
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    Relation,
    VarTypeParam,
)
from mldaikon.trace.trace import Liveness, Trace


class EqualRelation(Relation):

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants for the EqualRelation."""

        logger = logging.getLogger(__name__)
        import pdb; pdb.set_trace()
        var_insts = trace.get_var_insts()