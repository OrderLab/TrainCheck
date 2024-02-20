import subprocess


class ProgramRunner(object):
    def __init__(self, source_code: str):
        self.source_code = source_code
        self._tmp_file_name = "_temp.py" # TODO: generate a random file name

    def run(self):
        """
        Runs the program and returns the trace of the program.
        """
        # create a temp file and write the source code to it
        with open(self._tmp_file_name, "w") as file:
            file.write(self.source_code)
        
        # run the program
        process = subprocess.Popen(["python3", self._tmp_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        print(out, err)

        # TODO: save the program to a temporary file and run it

        # XXX: This is a dummy implementation. Replace this with the actual implementation.
        return "dummy trace"