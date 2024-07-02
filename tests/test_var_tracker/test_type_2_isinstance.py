import ast
import inspect

import astor


def type_handle_mldaikon_proxy(x):
    if hasattr(x, "is_ml_daikon_proxied_obj"):
        return type(x._obj)
    return type(x)


class TypeToIsInstanceTransformer(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)

        # Check if the call is type(xxx)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "type"
            and len(node.args) == 1
        ):
            # Replace type(xxx) with type_handle_mldaikon_proxy(xxx)
            new_node = ast.Call(
                func=ast.Name(id="type_handle_mldaikon_proxy", ctx=ast.Load()),
                args=node.args,
                keywords=[],
            )
            return ast.copy_location(new_node, node)
        return node


def adapt_func_for_proxy(func):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    transformer = TypeToIsInstanceTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    new_code = astor.to_source(new_tree)

    # Define a new dictionary to execute the transformed code
    new_locals = {}
    exec(new_code, func.__globals__, new_locals)

    # Return the transformed function object
    return new_locals[func.__name__]


# Example usage
def example_function():
    a = 42
    import pdb

    pdb.set_trace()
    if type(a) is int:  # noqa
        print("a is an int")


# Transform the example function
example_function = adapt_func_for_proxy(example_function)

# Call the transformed function
example_function()
