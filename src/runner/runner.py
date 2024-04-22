import subprocess
import sys


class ProgramRunner(object):
    def __init__(self, source_code: str, parent_folder_path: str):
        self.source_code = source_code
        self._tmp_file_name = "_temp.py"  # TODO: generate a random file name
        self.parent_folder_path = parent_folder_path
        self.python = sys.executable

    def run(self) -> str:
        """
        Runs the program and returns the trace of the program.
        """
        # create a temp file and write the source code to it
        with open(self._tmp_file_name, "w") as file:
            file.write(self.source_code)

        # run the program
        process = subprocess.Popen(
            ["python", self._tmp_file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # env={"PYTHONPATH": self.parent_folder_path},
        )
        out, err = process.communicate()

        if process.returncode != 0:
            # dump the output to the console
            print(out.decode("utf-8"))
            raise Exception(err.decode("utf-8"))
        # print(out, err)

        trace_str = out.decode("utf-8")

        # XXX: This is a dummy implementation. Replace this with the actual implementation.
        return trace_str
