import ast

def instrumentor_checker(
    source: str, 
    trace_folders: str,
    invariants: str,
) -> str:
    # trace_folders = "/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/traincheck_84911_trace"
    root = ast.parse(source)
    main_func = None
    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func = node
            break

    if main_func:
        if not any(isinstance(n, ast.Import) and any(alias.name == "subprocess" for alias in n.names) for n in root.body):
            root.body.insert(0, ast.parse("import subprocess").body[0])

        injected_code = f"""
command = [
    "traincheck-check-period",
    "--trace-folders", "{trace_folders}",
    "--invariants", "{invariants}",
]
process = subprocess.Popen(command)
"""

        instr_statements = ast.parse(injected_code).body
        main_func.body = instr_statements + main_func.body
        main_func.body.append(ast.parse("process.terminate()").body[0])
        main_func.body.append(ast.parse("process.wait()").body[0])

    else:
        injected_code = f"""
command = [
    "traincheck-check-period",
    "--trace-folders", "{trace_folders}",
    "--invariants", "{invariants}",
]
process = subprocess.Popen(command)
"""

        instr_statements = ast.parse(injected_code).body
        root.body = instr_statements + root.body
        root.body.insert(0, ast.parse("import subprocess").body[0])
        root.body.append(ast.parse("process.terminate()").body[0])
        root.body.append(ast.parse("process.wait()").body[0])

    return ast.unparse(root)