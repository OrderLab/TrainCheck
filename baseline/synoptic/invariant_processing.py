import re
from collections import defaultdict
import argparse

# Sample input sequence
# input_sequence = """
# torch.nn.modules.module.Module.eval(post) AlwaysFollowedBy(t) torch.cuda.is_available(pre)
# torch.cuda._nvml_based_avail(pre) AlwaysFollowedBy(t) torch.nn.modules.module.Module.eval(pre) 
# torch.nn.modules.module.Module.eval(post) AlwaysFollowedBy(t) torch.cuda._nvml_based_avail(post)
# torch.nn.modules.module.Module.eval(post) AlwaysFollowedBy(t) torch.cuda.is_available(post)
# """
# Set up argument parser
parser = argparse.ArgumentParser(description='Process input and output file paths.')
parser.add_argument('input_file', type=str, help='Path to the input file')
parser.add_argument('output_file', type=str, help='Path to the output file')
args = parser.parse_args()

# Load from input file
with open(args.input_file) as f:
    input_sequence = f.read()

# Define regex to match the format: function_name(context) relation_type(t) function_name(context)
pattern = re.compile(r"([\w\.]+)\((\w+)\)\s+(AlwaysFollowedBy|NeverFollowedBy)\(t\)\s+([\w\.]+)\((\w+)\)")

# Parse the sequence and categorize by relation type
relations = defaultdict(list)
for line in input_sequence.strip().splitlines():
    match = pattern.search(line)
    if match:
        funcA, contextA, relation, funcB, contextB = match.groups()
        relations[relation].append((funcA, contextA, funcB, contextB))

# Define function to extract FunctionContainedBy with hashdict optimization
def extract_function_contained_by():
    contained_by = []
    pre_dict = {}
    post_dict = {}

    # Populate dictionaries with pre and post relations for efficient lookup
    for funcA, contextA, funcB, contextB in relations['AlwaysFollowedBy']:
        if contextA == 'pre' and contextB == 'pre':
            pre_dict[(funcA, funcB)] = (contextA, contextB)
        elif contextA == 'post' and contextB == 'post':
            post_dict[(funcA, funcB)] = (contextA, contextB)

    # Find matches where pre conditions are reversed in post
    for (funcA, funcB) in pre_dict:
        if (funcB, funcA) in post_dict:
            contained_by.append((funcA, funcB))
            
    return contained_by

def extract_function_followed_by():
    followed_by = []
    for funcA, contextA, funcB, contextB in relations['AlwaysFollowedBy']:
        if contextA == 'pre' and contextB == 'post':
            followed_by.append((funcA, funcB))
    return followed_by

# Extract and display the invariants
contained_by_invariants = extract_function_contained_by()
followed_by_invariants = extract_function_followed_by()

# Write the invariants to a file
with open(args.output_file, 'w') as f:
    f.write("FunctionContainedBy Invariants:\n")
    for funcA, funcB in contained_by_invariants:
        f.write(f"{funcA} AlwaysContainedBy(t) {funcB}\n")

    f.write("\nFunctionFollowedBy Invariants:\n")
    for funcA, funcB in followed_by_invariants:
        f.write(f"{funcA} AlwaysFollowedBy(t) {funcB}\n")
