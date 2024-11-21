import logging
import os
import subprocess
import sys

from mldaikon.config.config import TMP_FILE_PREFIX


def program_print(program_output: str):
    # print the program output in blue color
    print("\033[94m" + program_output + "\033[0m")


class ProgramRunner(object):
    def __init__(
        self,
        source_code: str,
        py_script_path: str,
        sh_script_path: str | None = None,
        dry_run: bool = False,
        profiling: bool = False,
        output_dir: str = "",
    ):
        self.python = (
            sys.executable
        )  # use the same python executable that is running this script
        self.dry_run = dry_run
        self._tmp_sh_script_path: str | None
        self._tmp_py_script_path: str
        self.output_dir = output_dir
        self.profiling = profiling

        output_dir = os.path.abspath(output_dir) if output_dir else ""
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create temp files to write the source code to
        py_script_path = os.path.abspath(py_script_path)
        self.original_py_script_path = py_script_path

        py_script_name = os.path.basename(py_script_path)
        _tmp_py_script_name = f"{TMP_FILE_PREFIX}{py_script_name}"
        self._tmp_py_script_path = os.path.join(self.output_dir, _tmp_py_script_name)

        # write the source code also to the output directory (for debugging)
        with open(self._tmp_py_script_path, "w") as file:
            file.write(source_code)

        # write the modified py script to the original location as well
        original_py_parent_dir = os.path.dirname(py_script_path)
        with open(
            os.path.join(original_py_parent_dir, _tmp_py_script_name), "w"
        ) as file:
            file.write(source_code)

        if sh_script_path is None:
            self._tmp_sh_script_path = None
        else:
            sh_script_path = os.path.abspath(sh_script_path)
            sh_script_name = os.path.basename(sh_script_path)
            _tmp_sh_script_name = f"{TMP_FILE_PREFIX}{sh_script_name}"
            self._tmp_sh_script_path = os.path.join(
                self.output_dir, _tmp_sh_script_name
            )

            # modify the sh script to run the temp python script
            with open(sh_script_path, "r") as file:
                sh_script = file.read()
            assert (
                py_script_name in sh_script
            ), f"{py_script_name} not found in {sh_script} at {sh_script_path}"
            sh_script = sh_script.replace(py_script_name, _tmp_py_script_name)

            # write the sh script also to the output directory (for debugging)
            with open(self._tmp_sh_script_path, "w") as file:
                file.write(sh_script)

    def run(self) -> tuple[str, int]:
        """
        Runs the program and returns the output and execution status of the program.
        """

        if self.dry_run:
            return "Dry run. Program not executed.", 0

        # prepare env: set the PYTHONPATH to the directory of the original python script
        os.environ["PYTHONPATH"] = os.path.dirname(self.original_py_script_path)

        if self._tmp_sh_script_path is not None:
            if self.profiling == "True":
                raise ValueError("Profiling is not supported with shell scripts.")
            # change to the directory of the sh script
            current_dir = os.getcwd()
            os.chdir(os.path.dirname(self._tmp_sh_script_path))
            process = subprocess.Popen(
                ["bash", self._tmp_sh_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ,
            )
            # change back to the original directory
            os.chdir(current_dir)
        else:
            if self.profiling == "True":
                profile_script_name = self._tmp_py_script_path.split("/")[-1].split(
                    "."
                )[0]
                assert profile_script_name.startswith(TMP_FILE_PREFIX)
                profile_script_name = profile_script_name[len(TMP_FILE_PREFIX) :]
                profile_output_path = os.path.join(
                    os.path.dirname(self._tmp_py_script_path),
                    f"{profile_script_name}.prof",
                )
                print("Profiling the program...")
                print(
                    f"Profiling the program and saving the result to {profile_output_path}"
                )
                cmdline = (
                    " ".join(
                        [
                            self.python,
                            "-m cProfile",
                            "-o",
                            f"{profile_output_path}",
                            self._tmp_py_script_path,
                        ]
                    ),
                )
                print(f"Running command: {cmdline}")
                # flush the stdout buffer
                sys.stdout.flush()
                process = subprocess.Popen(
                    cmdline,
                    shell=True,
                    env=os.environ,
                )
                # save the profiling result
                process.wait()
                return_code = process.returncode
                program_output = f"Profiling result saved to {profile_output_path}"

            else:
                process = subprocess.Popen(
                    [self.python, "-u", self._tmp_py_script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=os.environ,
                )

        out_lines = []  # STDERR is redirected to STDOUT
        assert process.stdout is not None
        with process.stdout as out:
            logging.info("Running the program... below is the output:")
            for line_out in out:
                decoded_line_out = line_out.decode("utf-8").strip("\n")
                program_print(decoded_line_out)
                out_lines.append(decoded_line_out)
            _, _ = process.communicate()
        program_output = "\n".join(out_lines)
        return_code = process.poll()
        assert return_code is not None

        # write the program output to a file
        with open(os.path.join(self.output_dir, "program_output.txt"), "w") as file:
            file.write(program_output)
            file.write(f"\n\nProgram exited with code {return_code}")

        return program_output, return_code
