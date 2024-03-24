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
    UnaryVariableInvariantAlwaysDifferent,
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
        self.event_per_pt: dict[str, list[Event]] = {}  # pt: pid, tid
        for event in events:
            pt = f"{event.event_dict['process_id']}_{event.event_dict['thread_id']}"
            if pt in self.event_per_pt:
                self.event_per_pt[pt].append(event)
            else:
                self.event_per_pt[pt] = [event]

        self.invariant_properties = None
        self.has_api_analyzed = False
        self.has_var_inv_analyzed = False

    def get_analyzed_results(self, analyze_api_invariants, analyze_variable_invariants):
        return self.analyze(analyze_api_invariants, analyze_variable_invariants)

    def get_trace_grouped_by_pt(self):
        return self.event_per_pt

    def analyze(self, analyze_api_invariants, analyze_variable_invariants):
        """
        Current Status:
        - This is specifically implemented for PyTorch-FORUM84911. The analysis tries to find events that always occur during function calls.
        """
        logger = logging.getLogger(__name__)

        if self.has_api_analyzed:
            logger.info("API Invariant Analysis has already been done")
            api_invariants = self.invariant_properties["api_invariants"]
        elif not analyze_api_invariants:
            api_invariants = {}
            logger.info("Skipping API Invariant Analysis")
            self.has_api_analyzed = False
        else:
            logger.info("Analyzing the trace for API invariants")
            ## #1. API Invariant Analysis -- Do analysis per pt
            api_invariants = {}
            for pt in tqdm.tqdm(self.event_per_pt):
                api_invariants[pt] = APIInvariantConstantEvents(
                    self.event_per_pt[pt]
                ).get_invariant_properties()

            api_invariants["merged"] = {"constant_events_during_api_calls": {}}
            # merge the results from all the pts
            for pt in api_invariants:
                for k, v in api_invariants[pt][
                    "constant_events_during_api_calls"
                ].items():  # HACK: change the 'constant_events_during_api_calls' to generic key
                    if (
                        k
                        in api_invariants["merged"]["constant_events_during_api_calls"]
                    ):
                        api_invariants["merged"]["constant_events_during_api_calls"][
                            k
                        ] = list(
                            set(
                                api_invariants["merged"][
                                    "constant_events_during_api_calls"
                                ][k]
                            ).intersection(set(v))
                        )
                    else:
                        api_invariants["merged"]["constant_events_during_api_calls"][
                            k
                        ] = v

            # order the merged invariants by length of v
            api_invariants["merged"]["constant_events_during_api_calls"] = dict(
                sorted(
                    api_invariants["merged"].items(),
                    key=lambda item: len(item[1]),
                    reverse=True,
                )
            )

            self.has_api_analyzed = True

        if self.has_var_inv_analyzed:
            logger.info("Variable Invariant Analysis has already been done")
            unary_invariants = self.invariant_properties["unary_invariants"]
            nary_invariants_with_precond = self.invariant_properties[
                "nary_invariants_with_precond"
            ]
        elif not analyze_variable_invariants:
            self.invariant_properties = {
                "api_invariants": api_invariants,
                "unary_invariants": {},
                "nary_invariants_with_precond": {},
            }
            logger.info("Analysis of the trace is complete")
            return self.invariant_properties

        else:
            """The implementation of this part is specifically for DS-1801 where we need to track state updates of model weights
            across different workers. The below code assumes there will be a state_dump event in each separate trace file, which
            actually is not the case. We need to change the code to handle the case where there are multiple state_dump events in a trace file.
            Also, when child processes are spawned, the state_dump event is not captured in the trace file. We need to fix this as well.
            Perhaps by capturing parent child relationship among processes and threads, we can fix this issue.
            """

            ## #2. Variable Invariant Analysis
            logger.info("Analyzing the trace for variable invariants")
            # Pre-processing: Create VariableInstances from the events for all variables observed
            logger.info(
                "Pre-processing: Create VariableInstances from the events for all variables observed"
            )

            var_state_changes = {pt: {} for pt in self.event_per_pt}
            for pt in tqdm.tqdm(self.event_per_pt):
                traces_df = pd.DataFrame(
                    [event.get_event_dict() for event in self.event_per_pt[pt]]
                )  # iterate using per-process because variables are shared across threads
                traces_state_change = traces_df[(traces_df["type"] == "state_change")]
                init_state = traces_df[(traces_df["type"] == "state_dump")]

                # HACK: for mnist, if there are no state_dump and state_change events, skip the process
                if len(init_state) == 0 and len(traces_state_change) == 0:
                    logger.info(
                        f"Skipping process {pt} as there are no state_dump and state_change events"
                    )
                    continue
                elif len(init_state) != 1:
                    logger.error(
                        f"Multiple ({len(init_state)}, {len(traces_state_change)}) state_dump events found for process {pt}, aborting"
                    )
                    logger.error(f"State dump events: {init_state}")
                    logger.error(f"State change events: {traces_state_change}")
                    continue
                assert (
                    len(init_state) == 1
                ), "There should be only one state_dump event"  # FIXME: one trace has multiple, (reproduce with events_per_pt on the 8 traces)
                for var in tqdm.tqdm(
                    traces_state_change["name"].unique(),
                    desc=f"Collecting raw traces for process {pt}",
                ):  # TODO: this is tracer specific naming, need to change it soon as our design is kinda flying around
                    # find the initial state
                    for param in init_state.iloc[
                        0
                    ].state:  # FIXME: add a check for whether we only have one state_dump event
                        if param["name"] == var:
                            initial_state = param
                            break
                    var_state_changes[pt][var] = {
                        "initial_state": initial_state,
                        "changes": traces_state_change[
                            traces_state_change["name"]
                            == var  # TODO: this is tracer specific naming, need to change it soon as our design is kinda flying around
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
                for var in tqdm.tqdm(
                    var_state_changes[pt],
                    desc=f"Constructing states for process and thread {pt}",
                ):
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
                    | UnaryVariableInvariantAlwaysDifferent(
                        var_instances[pt][var]
                    ).get_invariant_properties()
                    for var in tqdm.tqdm(
                        var_instances[pt],
                        desc=f"Analyzing unary invariants for process and thread {pt}",
                    )
                }

            ## #2.2 Cross-Process Analysis
            logger.info(
                "Analyzing the trace for n-ary invariants (cross process consistency check)"
            )
            nary_invariants_with_precond = {}
            for var_name in tqdm.tqdm(var_instances[list(var_instances.keys())[0]]):
                nary_invariants_with_precond[var_name] = (
                    NaryVariableInvariantConsistency(
                        [
                            var_instances[pt][var_name]
                            for pt in tqdm.tqdm(
                                var_instances,
                                desc=f"Analyzing n-ary invariants for variable {var_name}",
                            )
                            if var_name in var_instances[pt]
                        ]
                    ).get_invariant_properties_with_precond()
                )

            self.invariant_properties = {
                "api_invariants": api_invariants,
                "unary_invariants": unary_invariants,
                "nary_invariants_with_precond": nary_invariants_with_precond,
            }
            self.has_var_inv_analyzed = True

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
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip API invariant analysis",
    )
    parser.add_argument(
        "--skip-variable",
        action="store_true",
        help="Skip variable invariant analysis",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # if the path points to a specific file, read the trace from that file
    if args.path.endswith(".log"):
        logger.info(f"Reading trace from {args.path}")
        trace_lines = []
        # with open(args.path, "r") as f:
        #     trace_lines = [
        #         Event(
        #             line
        #         )  # FIXME: the prefix can sometimes be logger dependent, need to fix this
        #         for line in tqdm.tqdm(f.readlines())
        #         if line.startswith("{")
        #     ]
        with open(args.path, "r") as f:
            log = f.read()
            # ad-hoc preprocessing step to convert trace into a list of events
            trace_lines = [
                Event(line.split(":trace:")[-1].strip())
                for line in log.split("\n")
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
            pid, tid = os.path.basename(trace_path).split("_")[0:2]
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
    invariants = trace.get_analyzed_results(
        analyze_api_invariants=not args.skip_api,
        analyze_variable_invariants=not args.skip_variable,
    )

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
