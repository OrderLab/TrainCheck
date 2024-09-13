import argparse
import json

keys_to_ignore = ["_id", "time", "time_ns"]
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
            if dict1[key] != dict2[key]:
                differing_keys += 1
                differences[key] = (dict1[key], dict2[key])
        else:
            differing_keys += 1
            differences[key] = (dict1[key], None)

    # Check for any additional keys in dict2
    for key in dict2:
        if key not in dict1:
            differing_keys += 1
            differences[key] = (None, dict2[key])

    return differing_keys, differences

def find_best_match_forward(dict1, list2, start_index):
    """
    Find the best matching dictionary from list2, starting at start_index.
    This ensures that no backward matching is allowed.
    """
    best_match = None
    fewest_differences = float('inf')
    best_diff = None
    best_match_index = -1

    for idx in range(start_index, len(list2)):
        dict2 = list2[idx]
        differing_keys, differences = diff_dicts(dict1, dict2)
        if differing_keys < fewest_differences:
            best_match = dict2
            best_match_index = idx
            fewest_differences = differing_keys
            best_diff = differences

        # Break early if we find an exact match
        if fewest_differences == 0:
            break

    return fewest_differences, best_match, best_diff, best_match_index

def delete_common_keys_values(dict1, dict2):
    """
    Delete common keys-values from two dictionaries
    """
    dict_1 = {}
    dict_2 = {}
    for key in dict1:
        if any([key.endswith(k) for k in keys_to_ignore]):
            continue
        if key not in dict2 or dict1[key] != dict2[key]:
            if isinstance(dict1[key], dict) and key in dict2 and isinstance(dict2[key], dict):
                # Recursively delete common keys-values in nested dictionaries
                dict1[key], dict2[key] = delete_common_keys_values(dict1[key], dict2[key])
            # if it is not a long string, then add it to the dictionary
            if not isinstance(dict1[key], str) or len(dict1[key]) < 25:
                dict_1[key] = dict1[key]
    for key in dict2:
        if any([key.endswith(k) for k in keys_to_ignore]):
            continue
        if key not in dict1 or dict1[key] != dict2[key]:
            if isinstance(dict2[key], dict) and key in dict1 and isinstance(dict1[key], dict):
                # Recursively delete common keys-values in nested dictionaries
                dict1[key], dict2[key] = delete_common_keys_values(dict1[key], dict2[key])
            if not isinstance(dict2[key], str) or len(dict2[key]) < 25:
                dict_2[key] = dict2[key]
    return dict_1, dict_2

def diff_lists_of_dicts(list1, list2):
    """
    Compare two lists of dictionaries, preserving the forward matching flow and
    outputting diff-like results using +/- and line numbers.
    """
    used_indices_list2 = set()
    last_match_index = -1
    difference_threshold = 2

    # Process list1
    for i, dict1 in enumerate(list1):
        best_differing_keys, best_match, differences, match_index = find_best_match_forward(dict1, list2, last_match_index + 1)
        if best_differing_keys != 0:
            print(f"Comparing Line {i+1} with Line {match_index+1} (fewest differing keys: {best_differing_keys})")

        if match_index != -1:
            used_indices_list2.add(match_index)
            last_match_index = match_index

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
                        print(f"- Line {i+1}: {key}={val1}")
                        print(f"+ Line {match_index+1}: {key}={val2}")
                    elif val1 is not None:
                        print(f"- Line {i+1}: {key}={val1}")
                    else:
                        print(f"+ Line {match_index+1}: {key}={val2}")
            else:
                # Completely different, treat as deletion and insertion
                print(f"- Line {i+1}: {dict1}")
                print(f"+ Line {match_index+1}: {best_match}")
        else:
            # If no match found, treat it as a deletion
            print(f"- Line {i+1}: {dict1}")

    # Process any remaining unmatched lines in list2 (insertions)
    for j, dict2 in enumerate(list2):
        if j not in used_indices_list2:
            print(f"+ Line {j+1}: {dict2}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a single trace of two runs to tell the difference"
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
            
    
    diff_lists_of_dicts(invs1, invs2)
    
    # if output file is provided, write the differences to the file
    if output_file:
        with open(output_file, "w") as f:
            for inv in invs1:
                f.write(json.dumps(inv))
                f.write("\n")
            for inv in invs2:
                f.write(json.dumps(inv))
                f.write("\n")
        print(f"Differences written to {output_file}")
    else:
        print("Differences not written to any file since output file is not provided")
