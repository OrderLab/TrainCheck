import argparse
import json

keys_to_ignore = ["_id", "time", "time_ns", "_millis", "_str", "idx"]


def recursive_count_dict_elements(dict1):
    """
    Count the number of elements in a dictionary, including nested dictionaries.
    """
    count = 0
    for key, val in dict1.items():
        if isinstance(val, dict):
            count += recursive_count_dict_elements(val)
        else:
            count += 1
    return count


def diff_dicts(dict1, dict2):
    """
    Compare two dictionaries and return the number of differing keys
    and the differences in a dictionary format.
    """
    differing_keys = 0
    differences = {}

    # Compare key-value pairs
    for key in dict1:
        # skip if key ends with _id or time
        if any([key.endswith(k) for k in keys_to_ignore]):
            continue
        if key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                # Recursively compare sub-dictionaries
                sub_differing_keys, sub_differences = diff_dicts(dict1[key], dict2[key])
                differing_keys += sub_differing_keys
                if sub_differences:
                    differences = {
                        **differences,
                        **{f"{key}.{k}": v for k, v in sub_differences.items()},
                    }
            elif dict1[key] != dict2[key]:
                differing_keys += 1
                differences[key] = (dict1[key], dict2[key])
        else:
            if isinstance(dict1[key], dict):
                differing_keys += recursive_count_dict_elements(dict1[key])
            else:
                differing_keys += 1
            differences[key] = (dict1[key], None)

    # Check for any additional keys in dict2
    for key in dict2:
        if key not in dict1:
            if isinstance(dict2[key], dict):
                differing_keys += recursive_count_dict_elements(dict2[key])
            else:
                differing_keys += 1
            differences[key] = (None, dict2[key])

    return differing_keys, differences


def find_best_match(dict1, list2, selected_indices):
    """
    Find the best matching dictionary from list2.
    This ensures that no dictionary is selected more than once.
    """
    best_match = None
    fewest_differences = float("inf")
    best_diff = None
    best_match_index = -1

    for idx in range(len(list2)):
        if idx in selected_indices:
            continue

        dict2 = list2[idx]
        # if dict1 contains key text_description, then dict2 should also contain the key, and the value should be the same
        if "text_description" in dict1:
            if (
                "text_description" not in dict2
                or dict1["text_description"] != dict2["text_description"]
            ):
                continue
        differing_keys, differences = diff_dicts(dict1, dict2)
        if differing_keys < fewest_differences:
            best_match = dict2
            best_match_index = idx
            fewest_differences = differing_keys
            best_diff = differences

        # Break early if we find an exact match
        if fewest_differences == 0:
            break

    if best_match_index != -1:
        selected_indices.add(best_match_index)

    return fewest_differences, best_match, best_diff, best_match_index, selected_indices


def delete_common_keys_values(dict1, dict2):
    """
    Delete common keys-values from two dictionaries
    """
    dict_1 = {}
    dict_2 = {}
    for key in dict1:
        if any([key.endswith(k) for k in keys_to_ignore]):
            continue
        if (key not in dict2) or dict1[key] != dict2[key]:
            if (
                isinstance(dict1[key], dict)
                and key in dict2
                and isinstance(dict2[key], dict)
            ):
                # Recursively delete common keys-values in nested dictionaries
                dict1[key], dict2[key] = delete_common_keys_values(
                    dict1[key], dict2[key]
                )
            # if it is not a long string, then add it to the dictionary
            if not isinstance(dict1[key], str) or len(dict1[key]) < 25:
                dict_1[key] = dict1[key]
    for key in dict2:
        if any([key.endswith(k) for k in keys_to_ignore]):
            continue
        if (key not in dict1) or dict1[key] != dict2[key]:
            if (
                isinstance(dict2[key], dict)
                and key in dict1
                and isinstance(dict1[key], dict)
            ):
                # Recursively delete common keys-values in nested dictionaries
                dict1[key], dict2[key] = delete_common_keys_values(
                    dict1[key], dict2[key]
                )
            if not isinstance(dict2[key], str) or len(dict2[key]) < 25:
                dict_2[key] = dict2[key]
    return dict_1, dict_2


def invalidate(val1):
    """
    Check if the values are empty dictionaries or nested empty dictionaries
    """

    if isinstance(val1, dict):
        for key, val in val1.items():
            if not invalidate(val):
                return False
        return True
    return not bool(val1)


def diff_lists_of_dicts(list1, list2, output_file=None):
    """
    Compare two lists of dictionaries, preserving the forward matching flow and
    outputting diff-like results using +/- and line numbers.
    """
    used_indices_list2 = set()
    difference_threshold = 4
    selected_indices = set()
    # Process list1
    for i, dict1 in enumerate(list1):
        best_differing_keys, best_match, differences, match_index, selected_indices = (
            find_best_match(dict1, list2, selected_indices)
        )
        # if best_differing_keys != 0:
        #     print(
        #         f"Comparing Line {i+1} with Line {match_index+1} (fewest differing keys: {best_differing_keys})"
        #     )
        #     print(differences)
        #     if output_file:
        #         with open(output_file, "a") as f:
        #             f.write(f"Comparing Line {i+1} with Line {match_index+1} (fewest differing keys: {best_differing_keys})\n")

        if match_index != -1:
            used_indices_list2.add(match_index)
            # last_match_index = match_index

            if best_differing_keys == 0:
                # No differences, skip this line (like in diff output)
                continue
            elif best_differing_keys <= difference_threshold:
                # Show modification with - and +
                for key, (val1, val2) in differences.items():
                    if val1 is not None and val2 is not None:
                        if isinstance(val1, dict) and isinstance(val2, dict):
                            # Only print the differences key values in the nested dictionary
                            val1, val2 = delete_common_keys_values(val1, val2)
                            if invalidate(val1) and invalidate(val2):
                                continue
                        print(f"- Line {i+1}: {key}={val1}")
                        print(f"+ Line {match_index+1}: {key}={val2}")
                        if output_file:
                            with open(output_file, "a") as f:
                                f.write(f"- Line {i+1}: {key}={val1}\n")
                                f.write(f"+ Line {match_index+1}: {key}={val2}\n")
                    elif val1 is not None:
                        print(f"- Line {i+1}: {key}={val1}")
                        if output_file:
                            with open(output_file, "a") as f:
                                f.write(f"- Line {i+1}: {key}={val1}\n")
                    else:
                        print(f"+ Line {match_index+1}: {key}={val2}")
                        if output_file:
                            with open(output_file, "a") as f:
                                f.write(f"+ Line {match_index+1}: {key}={val2}\n")

            else:
                # Completely different, treat as deletion and insertion
                print(f"- Line {i+1}: {dict1}")
                print(f"+ Line {match_index+1}: {best_match}")
                if output_file:
                    with open(output_file, "a") as f:
                        f.write(f"- Line {i+1}: {dict1}\n")
                        f.write(f"+ Line {match_index+1}: {best_match}\n")
        else:
            # If no match found, treat it as a deletion
            print(f"- Line {i+1}: {dict1}")
            if output_file:
                with open(output_file, "a") as f:
                    f.write(f"- Line {i+1}: {dict1}\n")

    # Process any remaining unmatched lines in list2 (insertions)
    for j, dict2 in enumerate(list2):
        if j not in used_indices_list2:
            print(f"+ Line {j+1}: {dict2}")
            if output_file:
                with open(output_file, "a") as f:
                    f.write(f"+ Line {j+1}: {dict2}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the invariants of two runs to tell the difference"
    )
    parser.add_argument(
        "-f",
        "--inv_files",
        required=True,
        nargs=2,
        type=str,
        help="directories containing the invariants of the first and second run",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="diff_invs.json",
        type=str,
        help="output file to store the difference",
    )
    args = parser.parse_args()
    inv_files = args.inv_files
    output_file = args.output_file
    invs1 = []
    invs2 = []
    with open(inv_files[0], "r") as f:
        for line in f:
            invs1.append(json.loads(line))
    with open(inv_files[1], "r") as f:
        for line in f:
            invs2.append(json.loads(line))

    # create new file if output file is provided
    if output_file:
        with open(output_file, "w") as f:
            f.write("Differences between the invariants of two runs\n")

    diff_lists_of_dicts(invs1, invs2, output_file)

    # if output file is provided, write the differences to the file
    if output_file:
        print(f"Differences written to {output_file}")
    else:
        print("Differences not written to any file since output file is not provided")
