This folder contains some of the example pipelines that the tool should be able to instrument and analyze. 
Certain configs (e.g. number of epochs & training iterations) has been reduced to enable the tool to finish in a reasonable amount of time.
The pipelines also have some manual instrumentations that cannot be automated for now. So the team should aim 
to automate all the manual instrumentations, and make tracing efficient.

To use the pipelines here, run the following command:
```shell
python3 main.py --path <path-to-example-pipeline>
```

