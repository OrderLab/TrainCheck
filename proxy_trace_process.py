import argparse
import json
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="proxy_log.json")

args = parser.parse_args()

parent_folder = "/".join(args.input.split("/")[:-1]) + "/" if "/" in args.input else ""

traces = []
with open(args.input, "r") as f:
    for idx, line in enumerate(f):
        try:
            traces.append(json.loads(line))
        except Exception:
            print(f"{idx}: {line}")
            # raise

inf = 1000000
nan = 1234567

str_to_be_loaded = None
# processed_traces = []
# for trace in traces:
#     loaded = json.loads(trace["meta_vars"])
#     trace["type"] = "state_change"
#     if loaded == "":
#         trace["meta_vars"] = {}
#     else:
#         try:
#             trace["meta_vars"] = json.loads(
#                 loaded.strip('"')
#                 .replace("'", '"')
#                 .replace("None", "null")
#                 .replace("True", "true")
#                 .replace("False", "false")
#                 .replace("NaN", f"{nan}")
#                 .replace("-inf", f"-{inf}")
#                 .replace("inf", f"{inf}")
#             )
#         except:
#             str_to_be_loaded = trace["meta_vars"]
#             print(type(loaded))
#             print(loaded)
#             print(trace)
#             raise


buggy_value = None

for trace in tqdm(traces):
    trace["attributes"].pop("T", None)
    trace["attributes"].pop("mT", None)
    trace["attributes"].pop("H", None)
    trace["attributes"].pop("mH", None)

    # try:
    #     trace['attributes']['value'] = json.loads(trace['value'].replace("'", "\"").replace("None", "null").replace("True", "true").replace("False", "false").replace("(", "[").replace(")", "]").replace(",]", "]"))
    # except:
    #     print(trace['value'])
    #     buggy_value = trace['value']
    #     buggy_trace = trace
    #     raise

    for attr in trace["attributes"]:
        if isinstance(trace["attributes"][attr], dict):
            trace["attributes"][attr] = [hash(str(trace["attributes"][attr]))] * sum(
                trace["attributes"][attr]["shape"]
            )

    # trace.pop('value')


# split traces according to unique meta_vars schema
meta_vars_schema_traces: dict[str, list[dict]] = {}
for trace in traces:
    key = str(trace["meta_vars"].keys()) + str(trace["attributes"].keys())
    if key not in meta_vars_schema_traces:
        meta_vars_schema_traces[key] = []
    meta_vars_schema_traces[key].append(trace)

# # let's dump this to a file
# with open("proxy_trace_processed.json", "w") as f:
#     for trace in traces:
#         f.write(json.dumps(trace) + "\n")
processed_trace_folder = parent_folder + "processed_proxy_traces/"
if not os.path.exists(processed_trace_folder):
    os.makedirs(processed_trace_folder)
for i, schema in enumerate(meta_vars_schema_traces):
    print(f"Schema {i}: {schema}")
    print(f"Number of traces: {len(meta_vars_schema_traces[schema])}")
    with open(f"{processed_trace_folder}proxy_trace_processed_{i}.json", "w") as f:
        for trace in meta_vars_schema_traces[schema]:
            f.write(json.dumps(trace) + "\n")
