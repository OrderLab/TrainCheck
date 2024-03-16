class VariableInstance:
    def __init__(self, name: str, type: type, values=None):
        self.name = name
        self.type = type
        self.values = values
