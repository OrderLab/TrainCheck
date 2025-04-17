import logging
import os
import signal
import subprocess
import sys

from traincheck.config.config import RUNNER_DEFAULT_ENV, TMP_FILE_PREFIX


def program_print(program_output: str):
    # print the program output in blue color
    print("\033[94m" + program_output + "\033[0m")


RUNNING_PROCESSES = None
KILLING_PROCESS = (
    False  # True indicates that SIGTERM has been sent to the running process
)


# make sure process is killed when the program is exited
def kill_running_process():
    global RUNNING_PROCESSES
    global KILLING_PROCESS
    if RUNNING_PROCESSES is None or KILLING_PROCESS:
        return

    KILLING_PROCESS = True
    print("Killing the running process...")
    for running_process in RUNNING_PROCESSES:
        os.killpg(
            os.getpgid(running_process.pid), signal.SIGTERM
        )  # send SIGTERM to the process group NOTE: the signal will be delivered here again


ORIGINAL_SIGINT_HANDLER = signal.getsignal(signal.SIGINT)
ORIGINAL_SIGTERM_HANDLER = signal.getsignal(signal.SIGTERM)


def handle_SIGINT(signum, frame):
    global KILLING_PROCESS

    print("Received SIGINT")
    if KILLING_PROCESS:
        exit(130)
        return
    kill_running_process()
    if callable(ORIGINAL_SIGINT_HANDLER):
        ORIGINAL_SIGINT_HANDLER(signum, frame)


def handle_SIGTERM(signum, frame):
    global KILLING_PROCESS

    print("Received SIGTERM")
    if KILLING_PROCESS:
        exit(143)
        return
    kill_running_process()
    if callable(ORIGINAL_SIGTERM_HANDLER):
        ORIGINAL_SIGTERM_HANDLER(signum, frame)


curr_excepthook = sys.excepthook


def kill_running_process_on_except(typ, value, tb):
    kill_running_process()
    curr_excepthook(typ, value, tb)


def register_hook_closing_program():
    signal.signal(signal.SIGTERM, handle_SIGTERM)
    signal.signal(signal.SIGINT, handle_SIGINT)
    sys.excepthook = kill_running_process_on_except


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

        if self._tmp_sh_script_path is None:
            self.cmd = [self.python, "-u", self._tmp_py_script_path]
        else:
            self.cmd = ["bash", self._tmp_sh_script_path]

    def run_py_spy_profiling(self, pgid: int, output_dir: str):
        py_spy_cmd = [
            "py-spy",
            "record",
            "--pid",
            str(pgid),
            "--native",
            "--subprocesses",
            "--output",
            os.path.join(output_dir, "py_spy_profile.svg"),
        ]
        py_spy_process = subprocess.Popen(py_spy_cmd)
        return py_spy_process

    def run(self) -> tuple[str, int]:
        """
        Runs the program and returns the output and execution status of the program.
        """

        global RUNNING_PROCESSES

        register_hook_closing_program()

        if self.dry_run:
            return "Dry run. Program not executed.", 0

        # prepare env: set the PYTHONPATH to the directory of the original python script
        os.environ["PYTHONPATH"] = os.path.dirname(self.original_py_script_path)

        current_dir = os.getcwd()
        if self._tmp_sh_script_path is not None:
            os.chdir(os.path.dirname(self._tmp_sh_script_path))
        else:
            os.chdir(os.path.dirname(self._tmp_py_script_path))

        env_vars = RUNNER_DEFAULT_ENV.copy()
        env_vars.update(os.environ)

        process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env_vars,
        )
        # change back to the original directory
        os.chdir(current_dir)

        RUNNING_PROCESSES = [process]

        if self.profiling:
            logging.info(
                "Starting py-spy profiling on process with PID: %d", process.pid
            )
            py_spy_process = self.run_py_spy_profiling(process.pid, self.output_dir)
            RUNNING_PROCESSES.append(py_spy_process)

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

        # join the py-spy process if it was started
        if self.profiling:
            logging.info("Waiting for py-spy process to finish...")
            py_spy_process.wait()

        return program_output, return_code
