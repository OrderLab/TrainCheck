import json
import logging
import os
from collections import namedtuple

import pandas as pd
import torch  # TODO: hardcoded for variable type, need to remove it by encoding var type in trace
import tqdm

from .api_invariant import APIInvariantConstantEvents
from .event import Event
from .variable_invariant import (
    NaryVariableInvariantConsistency,
    UnaryVariableInvariantConstant,
    VariableInstance,
)

process_and_thread = namedtuple("process_and_thread", ["pid", "tid"])


class Trace:
    """
    A trace is a sequence of events that occurred during the execution of a program.
    """

    def __init__(self, events: list[Event]) -> None:
        self.events = events
        self.event_per_pt: dict[process_and_thread, list[Event]] = {}  # pt: pid, tid
        for event in events:
            pt = process_and_thread(
                event.event_dict["process_id"], event.event_dict["thread_id"]
            )  # TODO: refactor trace to dump 'pid' and 'tid' instead of 'pid' and 'tid'
            if pt in self.event_per_pt:
                self.event_per_pt[pt].append(event)
            else:
                self.event_per_pt[pt] = [event]

        self.invariant_properties = None
        self.has_analyzed = False

    def get_analyzed_results(self):
        if not self.has_analyzed:
            self.analyze()
        return self.invariant_properties

    def get_trace_grouped_by_pt(self):
        return self.event_per_pt

    def analyze(self):
        """
        Current Status:
        - This is specifically implemented for PyTorch-FORUM84911. The analysis tries to find events that always occur during function calls.
        """
        logger = logging.getLogger(__name__)
        # invariant_events_during_function_calls = {}
        # pbar = tqdm.tqdm(total=len(self.event_per_pt))
        # for pt in self.event_per_pt:
        #     result = self.analyze_local(pt.pid, pt.tid)

        #     # dump this result for debugging
        #     def default(o):
        #         if isinstance(o, set):
        #             return list(o)
        #         if isinstance(o, Event):
        #             return o.get_event()
        #         return o

        #     # dump the invariants
        #     with open(f"invariants_{pt.pid}_{pt.tid}.json", "w") as f:
        #         json.dump(result, f, indent=4, default=default)

        #     for k, v in result.items():
        #         if k in invariant_events_during_function_calls:
        #             invariant_events_during_function_calls[k] = (
        #                 invariant_events_during_function_calls[k].intersection(v)
        #             )
        #         else:
        #             invariant_events_during_function_calls[k] = v
        #     pbar.update(1)
        # pbar.close()

        # return invariant_events_during_function_calls

        # Two types of analysis TBD

        logger.info("Analyzing the trace for API invariants")
        ## #1. API Invariant Analysis -- Do analysis per pt
        api_invariants = {}
        for pt in tqdm.tqdm(self.event_per_pt):
            api_invariants[pt] = APIInvariantConstantEvents(
                self.event_per_pt[pt]
            ).get_invariant_properties()

        api_invariants["merged"] = {}
        # merge the results from all the pts
        for pt in api_invariants:
            for k, v in api_invariants[pt].items():
                if k in api_invariants["merged"]:
                    api_invariants["merged"][k] = api_invariants["merged"][
                        k
                    ].intersection(v)
                else:
                    api_invariants["merged"][k] = v

        ## #2. Variable Invariant Analysis
        logger.info("Analyzing the trace for variable invariants")
        # Pre-processing: Create VariableInstances from the events for all variables observed
        logger.info(
            "Pre-processing: Create VariableInstances from the events for all variables observed"
        )
        var_state_changes = {pt: {} for pt in self.event_per_pt}
        for pt in tqdm.tqdm(self.event_per_pt):
            traces_df = pd.DataFrame([event.get_event_dict() for event in pt])
            traces_state_change = traces_df[(traces_df["type"] == "state_change")]
            init_state = traces_state_change[(traces_df["type"] == "state_dump")]
            for var in traces_state_change["variable"].unique():
                # find the initial state
                for param in init_state.iloc[0].state:
                    if param["name"] == var:
                        initial_state = param
                        break
                var_state_changes[pt][var] = {
                    "initial": initial_state,
                    "changes": traces_state_change[
                        traces_state_change["variable"] == var
                    ],
                }

        def construct_states(initial_state, changes):
            # change "param" to "value" for consistency
            if "param" in initial_state:
                initial_state["value"] = initial_state["param"]
                del initial_state["param"]

            states = [initial_state]
            for i, trace in changes.iterrows():
                state = states[-1].copy()
                if "properties" in trace.change:
                    state["properties"] = trace.change["properties"]["new"]
                if "value" in trace.change:
                    state["value"] = trace.change["value"]["new"]
                states.append(state)
            return states

        var_instances = {pt: {} for pt in self.event_per_pt}
        for pt in tqdm.tqdm(var_state_changes):
            for var in var_state_changes[pt]:
                states = construct_states(
                    var_state_changes[pt][var]["initial_state"],
                    var_state_changes[pt][var]["changes"],
                )
                var_instances[pt][var] = VariableInstance(
                    var,
                    torch.nn.Parameter,  # TODO: hardcoded for variable type, need to remove it by encoding var type in trace
                    states,
                    len(states)
                    * [
                        var_state_changes[pt][var]["changes"].meta_vars.iloc[0]
                    ],  # TODO: change this to the actual meta_vars after we support tracking variable changes
                )
        logger.info(
            "Pre-processing: Create VariableInstances from the events for all variables observed -- Done"
        )
        ## #2.1 Unary Analysis
        logger.info("Analyzing the trace for unary invariants")
        unary_invariants = {}
        for pt in tqdm.tqdm(var_instances):
            unary_invariants[pt] = {
                var: UnaryVariableInvariantConstant(
                    var_instances[pt][var]
                ).get_invariant_properties()
                for var in var_instances[pt]
            }

        ## #2.2 Cross-Process Analysis
        logger.info(
            "Analyzing the trace for n-ary invariants (cross process consistency check)"
        )
        nary_invariants = {}
        for var_name in tqdm.tqdm(var_instances[var_instances.keys()[0]]):
            nary_invariants[var_name] = NaryVariableInvariantConsistency(
                [
                    var_instances[pt][var_name]
                    for pt in var_instances
                    if var_name in var_instances[pt]
                ]
            )

        self.invariant_properties = {
            "api_invariants": api_invariants,
            "unary_invariants": unary_invariants,
            "nary_invariants": nary_invariants,
        }
        self.has_analyzed = True

        logger.info("Analysis of the trace is complete")
        return self.invariant_properties

    def dump(self):
        with open("trace.txt", "w") as f:
            for event in self.events:
                f.write(event.get_event() + "\n")
        pass


class TraceAnalyzer:
    def __init__(self, traces: list[Trace]) -> None:
        self.traces = traces
        pass

    def analyze(self):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Invariant Finder for ML Pipelines in Python"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the trace file(s) to be analyzed",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # if the path points to a specific file, read the trace from that file
    if args.path.endswith(".log"):
        logger.info(f"Reading trace from {args.path}")
        trace_lines = []
        with open(args.path, "r") as f:
            trace_lines = [
                Event(line.split(":trace:")[-1].strip())
                for line in f.readlines()
                if line.startswith("INFO:trace:") or line.startswith("ERROR:trace:")
            ]
    else:
        trace_paths = [
            os.path.join(args.path, f)
            for f in os.listdir(args.path)
            if f.endswith("_trace.log")
        ]
        logger.info(f"Reading traces from {trace_paths}")
        trace_lines = []
        for trace_path in trace_paths:
            with open(trace_path, "r") as f:
                lines = f.readlines()
                for trace in tqdm.tqdm(lines):
                    # heuristics to skip non-json lines
                    if not trace.startswith("{"):
                        continue
                    try:
                        trace_lines.append(
                            Event(trace)
                        )  # Event will parse the trace using json
                    except:
                        print(trace)
                        raise
    trace = Trace(trace_lines)
    invariants = trace.get_analyzed_results()

    def default(o):
        if isinstance(o, set):
            return list(o)
        if isinstance(o, Event):
            return o.get_event()
        return o

    # dump the invariants
    logger.info("Dumping the invariants to invariants.json")
    with open("invariants.json", "w") as f:
        json.dump(invariants, f, indent=4, default=default)

    # event1 = Event(
    #     '{"process_id": 1, "thread_id": 1, "uuid": 1, "type": "function_call (pre)", "function": "foo"}'
    # )
    # event2 = Event(
    #     '{"process_id": 1, "thread_id": 1, "uuid": 2, "type": "function_call (pre)", "function": "foo"}'
    # )

    # print(set([event1]).intersection(set([event2])))
