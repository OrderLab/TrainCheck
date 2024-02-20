"""
Methods for reading and instrumenting source files.
"""

def instrument_source(source: str) -> str:
    """
    Instruments the given source code and returns the instrumented source code.
    """
    # XXX: This is a dummy impleentation. Replace this with the actual implementation.
    # TODO: please look into this https://github.com/harshitandro/Python-Instrumentation for possible implementation
    
    return source

def instrument_file(path: str) -> str:
    """
    Instruments the given file and returns the instrumented source code.
    """


    with open(path, "r") as file:
        source = file.read()
    # instrument the source code
    instrumented_source = instrument_source(source)
    return instrumented_source

