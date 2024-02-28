# ML-DAIKON

## Usage
```shell
python3 main.py --help
usage: main.py [-h] --path PATH [--only-instrument] [--print_instr]

Invariant Finder for ML Pipelines in Python

options:
  -h, --help         show this help message and exit
  --path PATH        Path to the main file of the pipeline to be analyzed
  --only-instrument  Only instrument and dump the modified file
  --print_instr      print the log related to instrumentation
```

## Example
```shell
python3 main.py --path mnist.py
```

## Output
```shell
> head -5 invariants.json
"torch._C._get_privateuse1_backend_name": [],
"torch.random._seed_custom_device": [
    "{\"uuid\": \"7971ee35-2c99-4338-be11-a976f8ece67c\", \"thread_id\": 140084976235584, \"process_id\": 1884965, \"type\": \"function_call (pre)\", \"function\": \"torch._C._get_privateuse1_backend_name\"}",
    "{\"uuid\": \"7971ee35-2c99-4338-be11-a976f8ece67c\", \"thread_id\": 140084976235584, \"process_id\": 1884965, \"type\": \"function_call (post)\", \"function\": \"torch._C._get_privateuse1_backend_name\"}"
],
"torch.random.manual_seed": [
    "{\"uuid\": \"e6d0653e-3972-4734-b898-6e31a40a3d7c\", \"thread_id\": 140081023039040, \"process_id\": 1884965, \"type\": \"function_call (post)\", \"function\": \"torch._C._get_privateuse1_backend_name\"}",
```