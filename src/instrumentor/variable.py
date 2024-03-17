def flattenStateDict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flattenStateDict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class VariableInstance:
    def __init__(
        self, name: str, type: type, values: list[dict], meta_vars: list[dict]
    ):
        self.name = name
        self.type = type
        self.values = values
        self.flatten_values = [flattenStateDict(v) for v in values]
        self.meta_vars = meta_vars  # assumed to be a list of flattened dicts, each dict is associated with a state

    def get_values(self, flatten: bool = True):
        """Get all values of the variable_instance across all its states, by default flattened.
        The reason for flattening is that the invariant finder currently
        :param flatten: whether to return the values as a list of flattened dicts or as a list of dicts
        """
        if flatten:
            return self.flatten_values
        return self.values
