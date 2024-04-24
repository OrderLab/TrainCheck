import os
import subprocess
import sys

from mldaikon.config.config import TMP_FILE_PREFIX


class ProgramRunner(object):
    def __init__(
        self,
        source_code: str,
        py_script_path: str,
        sh_script_path: str | None = None,
        dry_run: bool = False,
    ):
        self.python = (
            sys.executable
        )  # use the same python executable that is running this script
        self.dry_run = dry_run
        self._tmp_sh_script_path: str | None
        self._tmp_py_script_path: str

        # create temp files to write the source code to
        py_script_path = os.path.abspath(py_script_path)
        py_script_name = os.path.basename(py_script_path)
        _tmp_py_script_name = f"{TMP_FILE_PREFIX}{py_script_name}"
        self._tmp_py_script_path = os.path.join(
            os.path.dirname(py_script_path), _tmp_py_script_name
        )
        # write the source code to the temp file
        with open(self._tmp_py_script_path, "w") as file:
            file.write(source_code)
        if sh_script_path is None:
            self._tmp_sh_script_path = None
        else:
            sh_script_path = os.path.abspath(sh_script_path)
            sh_script_name = os.path.basename(sh_script_path)
            _tmp_sh_script_name = f"{TMP_FILE_PREFIX}{sh_script_name}"
            self._tmp_sh_script_path = os.path.join(
                os.path.dirname(sh_script_path), _tmp_sh_script_name
            )

            # modify the sh script to run the temp python script
            with open(sh_script_path, "r") as file:
                sh_script = file.read()
            sh_script = sh_script.replace(py_script_name, _tmp_py_script_name)
            with open(self._tmp_sh_script_path, "w") as file:
                file.write(sh_script)

    def run(self) -> str:
        """
        Runs the program and returns the trace of the program.
        """

        if self.dry_run:
            return "Dry run. Program not executed."

        if self._tmp_sh_script_path is not None:
            # change to the directory of the sh script
            current_dir = os.getcwd()
            os.chdir(os.path.dirname(self._tmp_sh_script_path))
            process = subprocess.Popen(
                ["bash", self._tmp_sh_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # change back to the original directory
            os.chdir(current_dir)
        else:
            process = subprocess.Popen(
                [self.python, self._tmp_py_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        out, err = process.communicate()

        if process.returncode != 0:
            # dump the output to the console
            print(out.decode("utf-8"))
            raise Exception(err.decode("utf-8"))
        # print(out, err)

        program_output = out.decode("utf-8")

        # XXX: This is a dummy implementation. Replace this with the actual implementation.
        return program_output
