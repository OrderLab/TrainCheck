# Run Synoptic on API Trace

## Installation

1. go to the `baseline/synoptic` folder:

    ```bash
    pwd
    **/ml-daikon/baseline/synoptic
    ```

2. install dependencies:

    ```bash
    $sudo apt install graphviz
    $sudo apt install ant
    ```

3. check java runtime version, make sure you're in java8:

    ```bash
    $java -version
    $javac -version
    openjdk version "1.8.0_422"
    OpenJDK Runtime Environment (build 1.8.0_422-8u422-b05-1~22.04-b05)
    OpenJDK 64-Bit Server VM (build 25.422-b05, mixed mode)
    javac 1.8.0_422
    ```

4. build Synoptic from the command line with ant. Assuming that you checked out the code into synoptic/, there will be a top-level synoptic/build.xml file for ant to build all of the projects in the repository.

    ```bash
    ant synoptic # build synoptic and all of its dependencies.
    ant invarimint # build invarimint and all of its dependencies.
    ```

5. make sure you can run `synoptic` by:

    ```bash
    synoptic --help
    ```

## Run Trace

Ideally the synoptic/invarimint could support arbitrary type of trace. However, to save the effort to write extremely complicated regex matching. We provide a trace_processing script to get rid of unnecessary information and only remain function name and type. (We should keep process_id and thread_id but currently it does not have to scale out to multithreading conditions)

```bash
# Example usage
python trace_processing.py trace_API_1160540_139622061057856.log -o processed_trace.log
```

Then run `synoptic.sh` to do invariant inference. If you install graphviz by default your dot package would be under `-d /usr/bin/dot`

```bash
bash synoptic.sh -o output/test_trace -d /usr/bin/dot trace.log --dumpInvariants --outputInvariantsToFile=True
```

Up to this step you could obtain `output/test_trace.invariants.txt`. We provide another tool `invariant_processing.py` to convert the `pre` and `post` invariant to the existing invariants we have (e.g. containrelations). It's currently deprecated and might be required in the future.
