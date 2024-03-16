from abc import ABC, abstractmethod

from src.instrumentor.variable import VariableInstance

from .utils import diffStates


class Invariant(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def check(self, variable_instance):
        pass


class SingleInvariantConstant(Invariant):
    def __init__(self, variable_instance: VariableInstance):
        self.variable_instance = variable_instance
        self.has_analyzed = False
        self.invariant_properties: dict[str, dict[str, object]] = {
            "same": {}
        }  # properties that are always the same

    def check(self):
        """Check all values, attrs, etc. of variable_instance across all its states"""
        ## 1. find all properties that are always the same

        states = self.variable_instance.values
        state_diffs = []
        for i in range(1, len(states)):
            state_diffs.append(diffStates(states[i - 1], states[i]))

        # find properties that are always the same
        properties = list(states[0]["properties"].keys())
        for prop in properties:
            same = True
            for diff in state_diffs:
                if prop in diff:
                    same = False
                    break
            if same:
                self.invariant_properties["same"][prop] = states[0]["properties"][prop]
        self.has_analyzed = True

    def get_invariant_properties(self):
        if not self.has_analyzed:
            self.check()
        return self.invariant_properties


class MultiInvariantConsistency(Invariant):
    def __init__(self, list_variable_instances: list):
        self.variable_instances = list_variable_instances

    def check(self):
        """Check consistency of all values, attrs, etc. of variable_instances across all their states"""
        pass
