# Description: Extract the native functions from native_functions.yaml and filter the functions that intended to change the value.

from collections import defaultdict
import re

func_dict = defaultdict(list)
filtered_func_name = set()


def is_same_ret_type(info_list):
    """Check the return type. Should be all the same Tensors"""
    same_type = info_list[0]["return_type"]

    # If there are multiple return types, we can filter it
    if ", " in same_type:
        return False

    if same_type != () and "Tensor" not in same_type:
        return False

    for info in info_list:
        if info["return_type"] != same_type:
            return False
    return True


def judge_info(args_dict):
    tensor_count = 0
    for arg_name, arg_type in args_dict.items():
        if "Tensor" in arg_type:
            tensor_count += 1
    return tensor_count == 1


def parse_args(args_part):
    args = args_part.split(", ")
    parsed_args = {}
    default_args = {}
    for arg in args:
        arg_match = re.match(r"(.+) ([^=]+)=?(.*)?", arg)
        if arg_match:
            arg_type = arg_match.group(1)
            arg_name = arg_match.group(2)
            default_value = arg_match.group(3)
            parsed_args[arg_name] = arg_type
            if default_value:
                default_args[arg_name] = default_value
    return parsed_args, default_args


if __name__ == "__main__":
    with open("./native_functions.yaml", "r") as f:
        while True:
            lines = f.readline()
            if not lines:
                break

            match = re.match(r"- func: (\w+)(?:\.(\w+))?\((.*)\) -> (.+)", lines)

            if match:
                func_name = match.group(1)
                overload_name = match.group(2)
                args_part = match.group(3)
                return_type = match.group(4)

                parsed_args, default_args = parse_args(args_part)

                func_entry = {
                    "overload_name": overload_name,
                    "args": parsed_args,
                    "return_type": return_type,
                    "default_args": default_args,
                }

                # Find tags. If "tags" is pointwise, it means that it does some operations, so we can filter it.
                while True:
                    next_line = f.readline().strip()
                    if not next_line or next_line.startswith("- func:"):
                        if next_line.startswith("- func:"):
                            f.seek(f.tell() - len(next_line) - 1)
                        break

                    tag_match = re.match(r"tags: (.+)", next_line)
                    if tag_match:
                        func_entry["tags"] = tag_match.group(1)
                        break

                func_dict[func_name].append(func_entry)

    open("func_dict.json", "w").write(str(func_dict))

    for func_name, info_list in func_dict.items():
        # The return type should be the same
        if not is_same_ret_type(info_list):
            continue

        # If the tags is pointwise, we can filter it
        if info_list[0].get("tags") == "pointwise":
            continue

        decisions = []
        for info in info_list:
            decisions.append(judge_info(info["args"]))

        true_count = sum(decisions)

        if true_count > len(decisions) / 2:
            filtered_func_name.add(func_name)

    open("filtered_func_name.json", "w").write(str(filtered_func_name))
