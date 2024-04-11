import abc

from trace import Trace

class Invariant:
    def __init__(self):
        pass

class Relation(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        # TODO: indentify common attributes of relations and initialize them here
        pass

    @staticmethod
    @abc.abstractmethod
    def infer(self, trace) -> list[Invariant]:
        """Given a trace, should return a boolean value indicating
        whether the relation holds or not.

        args:
            trace: str
                A trace to infer the relation on.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, value_group: list) -> bool:
        """Given a group of values, should return a boolean value
        indicating whether the relation holds or not.

        args:
            value_group: list
                A list of values to evaluate the relation on. The length of the list
                should be equal to the number of variables in the relation.
        """
        pass

class Invariant:
    def __init__(self, relation: Relation, params: list, precondition: list):
    # def __init__(self, relation: Relation, params: list[Predicate], precondition: Predicate):
        self.relation = relation
        self.params = params
        self.precondition = precondition

    def verify(self, trace) -> bool:
        """Given a trace, should return a boolean value indicating
        whether the invariant holds or not.

        args:
            trace: str
                A trace to verify the invariant on.
        """
        relevant_trace = trace.filter(self.precondition)
        groups = relevant_trace.group(self.params)
        for g in groups:
            if not self.relation.evaluate(g):
                return False
        return True
    
class Hypothesis:
    def __init__(self, invariant: Invariant, positive_examples: list[Trace], negative_examples: list[Trace]):
        self.invariant = invariant
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
    
    def calc_positive_negative_examples(self, trace: Trace):
        return self.invariant.verify(trace)