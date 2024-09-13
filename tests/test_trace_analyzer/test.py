def diff_dicts(dict1, dict2):
    """
    Compare two dictionaries and return the number of differing keys
    and the differences in a dictionary format.
    """
    differing_keys = 0
    differences = {}

    # Compare key-value pairs
    for key in dict1:
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

# Example usage:
list1 = [
    {"name": "Alice", "age": 30, "city": "NYC"},
    {"name": "Bob", "age": 25, "city": "LA"},
    {"name": "Charlie", "age": 35, "city": "Chicago"}
]

list2 = [
    {"name": "Alice", "age": 30, "city": "NYC"},
    {"name": "Dave", "age": 40, "city": "Seattle"},  # New insertion
    {"name": "Bob", "age": 26, "city": "LA"},  # Slightly modified (age changed)
    {"name": "Charlie", "age": 36, "city": "Boston"}  # Moved line, slightly modified
]

diff_lists_of_dicts(list1, list2)
