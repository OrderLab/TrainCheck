# Description: Extracts the argument types from the trace file
# Example: python -m mldaikon.pre_instrument.type_annotation -f ./mldaikon_run_PT84911_torch_2.2.2_2024-08-25_23-46-28/trace_*.log

import argparse
import json
from collections import defaultdict


def type_annotation_from_trace(trace_path: str):
    """Extract type information from a trace file."""

    combined_results = defaultdict(lambda: defaultdict(set))

    elements_to_extract = ['function', "args", "return_type"]

    with open(trace_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            extracted_data = {key: item[key] for key in elements_to_extract if key in item}

            for key, value in extracted_data['args'].items():
                combined_results[extracted_data['function']][key].add(value)

            if 'return_type' in extracted_data:
                combined_results[extracted_data['function']]['return_type'].add(extracted_data['return_type'])

    final_result = {func: {k: list(v) for k, v in args.items()} for func, args in combined_results.items()}

    print(f'Extracted {len(final_result)} functions from {trace_path}')

    return final_result


def type_annotation_from_traces(trace_paths: list[str], output_path: str = 'output.json'):
    """Extract type information from a list of trace files."""

    func_to_annotations = {}

    for trace_path in trace_paths:
        func_to_annotations.update(type_annotation_from_trace(trace_path))

    print(f'Extracted {len(func_to_annotations)} functions from {len(trace_paths)} trace files')

    with open(output_path, 'w') as file:
        json.dump(func_to_annotations, file, indent=4)

    return func_to_annotations


if __name__ == '__main__':
    # pattern = r'/home/beijie/repos/daikon/type_anno/trace_API_*.log'
    parser = argparse.ArgumentParser(description='Process trace files for type annotation.')
    parser.add_argument('-f', '--files', nargs='+', required=True, help='List of trace files to process')
    parser.add_argument('-o', '--output', help='Output file path')

    args = parser.parse_args()

    if args.output:
        type_annotation_from_traces(args.files, args.output)
    else:
        type_annotation_from_traces(args.files)
