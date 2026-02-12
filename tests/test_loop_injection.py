import ast
import unittest

from traincheck.instrumentor.source_file import InsertTracerVisitor


class TestLoopInjection(unittest.TestCase):
    def test_inject_training_loop(self):
        source = """
import torch
def train():
    for i in range(10):
        data = get_data()
        optimizer.step()
        """

        visitor = InsertTracerVisitor(
            modules_to_instr=["torch"],
            scan_proxy_in_args=False,
            use_full_instr=False,
            funcs_to_instr=None,
            API_dump_stack_trace=False,
            sampling_interval=1,
            warm_up_steps=0,
        )

        tree = ast.parse(source)
        visitor.visit(tree)

        new_source = ast.unparse(tree)
        new_source = ast.unparse(tree)

        self.assertIn("import start_step", new_source)
        self.assertIn("start_step()", new_source)
        self.assertIn("from traincheck.instrumentor.control", new_source)

        self.assertIn("start_step", new_source)

    def test_inject_eval_loop(self):
        source = """
import torch
def test():
    for i in range(10):
        print(i)
        """

        visitor = InsertTracerVisitor(
            modules_to_instr=["torch"],
            scan_proxy_in_args=False,
            use_full_instr=False,
            funcs_to_instr=None,
            API_dump_stack_trace=False,
            sampling_interval=1,
            warm_up_steps=0,
        )

        tree = ast.parse(source)
        visitor.visit(tree)

        new_source = ast.unparse(tree)
        new_source = ast.unparse(tree)

        # Should detect "test" function name and inject start_eval_step
        self.assertIn("import start_eval_step", new_source)
        self.assertIn("start_eval_step()", new_source)

    def test_no_inject_irrelevant_loop(self):
        source = """
def check_thing():
    for i in range(10):
        print(i)
        """

        visitor = InsertTracerVisitor(
            modules_to_instr=["torch"],
            scan_proxy_in_args=False,
            use_full_instr=False,
            funcs_to_instr=None,
            API_dump_stack_trace=False,
            sampling_interval=1,
            warm_up_steps=0,
        )

        tree = ast.parse(source)
        visitor.visit(tree)

        new_source = ast.unparse(tree)

        self.assertNotIn("start_step", new_source)
        self.assertNotIn("start_eval_step", new_source)


if __name__ == "__main__":
    unittest.main()
