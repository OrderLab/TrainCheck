from abc import ABC, abstractmethod

from src.instrumentor.variable import VariableInstance

from .utils import diffStates


class Invariant(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def analyze(self, variable_instance):
        pass


class SingleInvariantConstant(Invariant):
    def __init__(self, variable_instance: VariableInstance):
        self.variable_instance = variable_instance
        self.has_analyzed = False
        self.invariant_properties: dict[str, dict[str, object]] = {
            "same": {}
        }  # properties that are always the same

    def analyze(self):
        """analyze all values, attrs, etc. of variable_instance across all its states"""
        ## 1. find all properties that are always the same

        states = self.variable_instance.get_values()
        state_diffs = []
        for i in range(1, len(states)):
            state_diffs.append(diffStates(states[i - 1], states[i]))

        # find properties that are always the same
        properties = list(
            states[0].keys()
        )  # FIXME: this is not reliable if we change the state structure, need to add a get_properties method to VariableInstance
        for prop in properties:
            same = True
            for diff in state_diffs:
                if prop in diff:
                    same = False
                    break
            if same:
                self.invariant_properties["same"][prop] = states[0][prop]
        self.has_analyzed = True

    def get_invariant_properties(self):
        if not self.has_analyzed:
            self.analyze()
        return self.invariant_properties


class MultiInvariantConsistency(Invariant):
    def __init__(self, list_variable_instances: list):
        self.variable_instances = list_variable_instances
        self.has_analyzed = False
        self.invariant_properties: dict[str, list[str]] = {"consistent": []}
        self.pre_conditions: dict[str, list[str]] = {"consistent": []}

        # assert: all variable_instances have the same type
        self.variable_type = self.variable_instances[0].type
        for i, var in enumerate(self.variable_instances):
            assert (
                var.type == self.variable_type
            ), f"All variable_instances must have the same type, encountering 0: {self.variable_type}, {i}: {var.type}."

    def find_preconditions(self):
        """Preconditions are based on the meta_vars of the variable_instances,
        and are used to filter out the states that are not relevant for the invariant check
        during runtime.

        A few types of preconditions:
        - consistent: find the meta_vars that are always the same across all states & all variable_instances.
        - contiguous (?) TODO
        """
        # find all properties that are always the same
        meta_vars = [var.meta_vars for var in self.variable_instances]

        # for each state, find the consistent properties
        consistent_meta_var_names = None
        for i in range(len(meta_vars[0])):
            # we need to take intersection of all properties of all variable_instances
            _consistent_meta_vars = set()
            for var_meta_vars in meta_vars:
                properties = list(var_meta_vars[i].items())
                # handle unhashable values by repr
                properties = set([(pv[0], repr(pv[1])) for pv in properties])
                _consistent_meta_vars = (
                    _consistent_meta_vars.intersection(properties)
                    if len(_consistent_meta_vars) > 0
                    else properties
                )

            _consistent_meta_var_names = set([pv[0] for pv in _consistent_meta_vars])
            if consistent_meta_var_names is None:
                consistent_meta_var_names = _consistent_meta_var_names
            else:
                consistent_meta_var_names = consistent_meta_var_names.intersection(
                    _consistent_meta_var_names
                )

        self.pre_conditions["consistent"] = list(consistent_meta_var_names)

        return self.pre_conditions

    def analyze(self):
        """analyze consistency of all values, attrs, etc. of variable_instances across all their states"""
        ## 1. find all properties that are always the same

        var_states = [var.get_values() for var in self.variable_instances]
        num_states = [len(states) for states in var_states]
        if len(set(num_states)) != 1:
            # TODO: perhaps we can return no invariants instead of raising an error
            raise ValueError(
                "All variable_instances must have the same number of states."
            )
        num_states = num_states[0]

        # for each state, find the consistent properties
        consistent_property_names = None
        for i in range(num_states):
            # we need to take intersection of all properties of all variable_instances
            _consistent_properties = set()
            for var_state in var_states:
                properties = list(var_state[i].items())
                # handle unhashable values by repr
                properties = set([(pv[0], repr(pv[1])) for pv in properties])
                _consistent_properties = (
                    _consistent_properties.intersection(properties)
                    if len(_consistent_properties) > 0
                    else properties
                )

            _consistent_property_names = set([pv[0] for pv in _consistent_properties])
            if consistent_property_names is None:
                consistent_property_names = _consistent_property_names
            else:
                consistent_property_names = consistent_property_names.intersection(
                    _consistent_property_names
                )

        self.invariant_properties["consistent"] = list(consistent_property_names)
        self.has_analyzed = True

    def get_invariant_properties(self):
        if not self.has_analyzed:
            self.analyze()
        return self.invariant_properties
