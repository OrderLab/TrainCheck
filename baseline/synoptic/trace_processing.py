import json
import argparse

def process_trace(trace_input_file, trace_output_file):
    with open(trace_input_file, 'r') as infile, open(trace_output_file, 'w') as outfile:
        for line in infile:
            trace = json.loads(line)
            function_name = trace['function']
            function_type = trace['type']
            outfile.write(f"{function_name}{function_type.split()[-1]}\n")

if __name__ == '__main__':
    # Example usage
    # python trace_processing.py trace_API_1160540_139622061057856.log -o trace.log
    parser = argparse.ArgumentParser(description='Process trace files.')
    parser.add_argument('input_file', type=str, help='The input trace file')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='The output file to write processed traces')

    args = parser.parse_args()

    process_trace(args.input_file, args.output_file)