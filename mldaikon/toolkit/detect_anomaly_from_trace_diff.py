# This script is used to detect abnormaly in between two diff files of API traces.
# Ideally, in order to pinpoint potential abnormalies caused by bugs and resolved by fixes, we need to acquire three execution traces:
# 1. trace of the original program (before the bug)
# 2. trace of the buggy program (with the bug)
# 3. trace of the fixed program (after the bug)
# From this we could generate two diff files:
# diff_file_1: trace difference between the original program and the buggy program
# diff_file_2: trace difference between the original program and the fixed program (ccontrol group, to eliminate the false positives)
# Then we could use this script to detect abnormalies in the diff files.


import argparse
import json
import re

from traincheck.toolkit.analyze_trace import diff_dicts


def read_diff_file(diff_file):
    file_1_insersion = []
    file_1_deletion = []
    file_1_edits = []
    with open(diff_file, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("++"):

                file_1_insersion.append(lines[i])

            elif lines[i].startswith("--"):
                file_1_deletion.append(lines[i])
            else:
                # multi-line edits
                if lines[i].startswith("+"):

                    file_1_edits.append(lines[i])
                elif lines[i].startswith("-"):
                    file_1_edits.append(lines[i])
                elif file_1_edits:
                    file_1_edits[-1] += lines[i]
    file_1_info = {
        "insertion": file_1_insersion,
        "deletion": file_1_deletion,
        "edit": file_1_edits,
    }
    return file_1_info


def calculate_string_distance(str1, str2):
    """
    Calculate the similarity of two strings using the Levenshtein distance
    with dynamic programming for better performance.
    """
    len_str1 = len(str1)
    len_str2 = len(str2)

    # Create a matrix to store distances
    dp = [[0 for _ in range(len_str2 + 1)] for _ in range(len_str1 + 1)]

    # Initialize the matrix
    for i in range(len_str1 + 1):
        dp[i][0] = i
    for j in range(len_str2 + 1):
        dp[0][j] = j

    # Compute the distances
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,
            )  # Substitution

    return dp[len_str1][len_str2]


def group_similar_patterns(section, name):
    """
    Group similar patterns of edition.

    Parameters:
    - section (list of tuples): A list where each element is a tuple containing a pattern.

    Returns:
    - list of sets: A list where each set contains grouped similar patterns.
    """
    patterns = {}
    pattern_allocated = {}
    for i in range(len(section)):
        if i in pattern_allocated:
            continue
        content_1 = section[i].split(": ", 1)[1]
        content_1 = re.sub(r"\d+", "0", content_1).strip()

        if name == "edit":
            if content_1 not in patterns:
                patterns[content_1] = [(i, section[i])]
            else:
                patterns[content_1].append((i, section[i]))
        else:

            # jsonize the content
            # print("content_1: ", content_1)
            content_1_str = str(content_1)
            # print("content_1_str: ", content_1_str)

            content_1 = json.loads(content_1)
            for j in range(i + 1, len(section)):
                if j in pattern_allocated:
                    continue
                patterns[content_1_str] = [(i, section[i])]
                content_2 = section[j].split(": ", 1)[1]
                content_2 = re.sub(r"\d+", "0", content_2).strip()
                content_2 = json.loads(content_2)
                diff_keys, difference = diff_dicts(content_1, content_2)
                if diff_keys <= 3:
                    patterns[content_1_str] = [(j, section[j])]
                    pattern_allocated[j] = True
        # assert pattern_1.startswith("+") or pattern_1.startswith("-"), "The pattern should start with '+' or '-' but got {}".format(pattern_1)
        # for j in range(i+1, len(section)):
        #     pattern_2 = section[j][0]
        #     assert pattern_2.startswith("+") or pattern_2.startswith("-"), "The pattern should start with '+' or '-' but got {}".format(pattern_2)
        #     if pattern_1 != pattern_2:
        #         continue

        #     # ignore the line number, only read the content of the patterns

        #     content_2 = section[j].split(":")[1]
        #     # ignore the pure numbers in the patterns, replace them with a placeholder

        #     content_2 = re.sub(r"\d+", "NUM", content_2)

        # calculate the distance of the patterns

        # distance = calculate_string_distance(content_1, content_2)
        # print("distance: ", distance)

        # # if distance is below a certain threshold, group them together
        # if distance < 3:
        #     patterns[-1].add(section[j])
        #     pattern_allocated[j] = True

        # Or just use a dict

    return patterns

    # first read the signature of the patterns at the front, only same signature patterns are grouped together
    # ignore the line number, only read the content of the patterns
    # ignore the pure numbers in the patterns, replace them with a placeholder
    # then calculate the similarity of the patterns (two strings) using the Levenshtein distance
    # group the patterns when similarity is above a certain threshold (e.g. 0.9)


def find_anomaly(file_1_info, file_2_info, section):
    """
    Find the abnormalies deletion, insertion, and edition section that occurs in file_2_info but not in file_1_info
    Procedures as follows:
    1) try to group similar patterns of deletion, insertion, and edition in file_1_info and file_2_info
    2) find the groups that only appear in file_2_info but not in file_1_info, rank them with the highest frequency
    """

    file_1_section = file_1_info[section]
    file_2_section = file_2_info[section]
    file_1_group = group_similar_patterns(file_1_section, section)
    file_2_group = group_similar_patterns(file_2_section, section)
    file_1_group_keys = set(file_1_group.keys())
    file_2_group_keys = set(file_2_group.keys())
    # import pdb; pdb.set_trace()
    anomalies = file_2_group_keys - file_1_group_keys
    return list(anomalies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the two diff files to find potential abnormalies useful for investigating bugs and design invariant checks"
    )
    parser.add_argument(
        "-f",
        "--inv_files",
        required=True,
        nargs=2,
        type=str,
        help="directories containing the invariants of the first (original-bug) and second diff files (original-fixed)",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="trace_abnormalies.json",
        type=str,
        help="output file to store the abnormalies",
    )
    args = parser.parse_args()
    inv_files = args.inv_files
    output_file = args.output_file
    file_1_info = read_diff_file(inv_files[0])
    file_2_info = read_diff_file(inv_files[1])

    anomalies_edit = find_anomaly(file_1_info, file_2_info, "edit")
    anomalies_insertion = find_anomaly(file_1_info, file_2_info, "insertion")
    anomalies_deletion = find_anomaly(file_1_info, file_2_info, "deletion")
    anomalies = {
        "edit": anomalies_edit,
        "insertion": anomalies_insertion,
        "deletion": anomalies_deletion,
    }
    with open(output_file, "w") as f:
        json.dump(anomalies, f, indent=4)
    print("The abnormalies are stored in {}".format(output_file))
