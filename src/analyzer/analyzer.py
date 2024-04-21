import json
from collections import namedtuple

import tqdm


class Event:
    """
    An event is a significant occurrence during the execution of a program.
    e.g. API Invocations, Model Parameter Changes, etc.
    """

    def __init__(self, event: str):
        self.event: str = event
        self.event_dict: dict = json.loads(event)

    def __repr__(self) -> str:
        return self.event

    def __hash__(self):
        self_dict = json.loads(
            self.event
        )  # making a copy of the event_dict as we are going to pop some keys

        # comment out the following lines because now traces are separate for each pid and tid, so events are expected to have same pid and tid.
        # self_dict.pop("process_id", None)
        # self_dict.pop("thread_id", None)

        self_dict.pop("uuid", None)

        return hash(json.dumps(self_dict, sort_keys=True))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_event(self):
        return self.event

    def get_event_dict(self):
        return self.event_dict

    def set_event(self, event: str):
        self.event = event


process_and_thread = namedtuple("process_and_thread", ["pid", "tid"])


class Trace:
    """
    A trace is a sequence of events that occurred during the execution of a program.
    """

    def __init__(self, events: "list[Event]") -> None:
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

    def analyze(self):
        """
        Current Status:
        - This is specifically implemented for PyTorch-FORUM84911. The analysis tries to find events that always occur during function calls.
        """

        invariant_events_during_function_calls = {}
        pbar = tqdm.tqdm(total=len(self.event_per_pt))
        for pt in self.event_per_pt:
            result = self.analyze_local(pt.pid, pt.tid)

            # dump this result for debugging
            def default(o):
                if isinstance(o, set):
                    return list(o)
                if isinstance(o, Event):
                    return o.get_event()
                return o

            # dump the invariants
            with open(f"invariants_{pt.pid}_{pt.tid}.json", "w") as f:
                json.dump(result, f, indent=4, default=default)

            for k, v in result.items():
                if k in invariant_events_during_function_calls:
                    invariant_events_during_function_calls[k] = (
                        invariant_events_during_function_calls[k].intersection(v)
                    )
                else:
                    invariant_events_during_function_calls[k] = v
            pbar.update(1)
        pbar.close()

        return invariant_events_during_function_calls

    def analyze_local(self, pid, tid):
        """
        Analyze the trace for a specific thread in a specific process.
        Assumes that within a thread, the events are ordered by their occurrence (True in Python).
        """

        pt = process_and_thread(pid, tid)
        if pt not in self.event_per_pt:
            return {}

        function_call_pres = 0
        function_call_posts = 0
        state_variable_changes = 0
        exception_events = 0

        invariant_events_during_function_calls = {}
        stack_current_function_calls = []

        local_events: list[Event] = self.event_per_pt[pt]

        pbar = tqdm.tqdm(total=len(local_events))
        for i, event in enumerate(local_events):

            event_dict: dict = event.get_event_dict()

            assert (
                event_dict["process_id"] == pid and event_dict["thread_id"] == tid
            ), f"Event pid and tid should match the input pid and tid, i: {i}, tid: {tid}, pid: {pid}, uuid: {event_dict['uuid']}"

            if event_dict["type"] == "function_call (pre)":
                function_call_pres += 1

                # add the function and its index to the stack
                stack_current_function_calls.append(
                    (event_dict["function"], event_dict["uuid"], i)
                )

            elif (
                event_dict["type"] == "function_call (post)"
                or event_dict["type"] == "function_call (post) (exception)"
            ):
                # TODO: handle the exception case separately. Currently, we are treating it as a normal function call post event in the analysis.
                function_call_posts += 1

                assert (
                    len(stack_current_function_calls) > 0
                ), "There should be a function call pre event before a function call post event"
                assert (
                    stack_current_function_calls[-1][0] == event_dict["function"]
                ), f"The function call pre and post events should match, i: {i}, tid: {tid}, pid: {pid}, uuid: {event_dict['uuid']}, stack_call pre: {stack_current_function_calls[-1][0]}, curr post: {event_dict['function']}"

                # pop the function from the stack
                _, _, i_pre = stack_current_function_calls.pop()

                # collect events between the function call pre and post events
                events = local_events[i_pre + 1 : i]

                if event_dict["function"] in invariant_events_during_function_calls:
                    invariant_events_during_function_calls[event_dict["function"]] = (
                        invariant_events_during_function_calls[
                            event_dict["function"]
                        ].intersection(set(events))
                    )
                else:
                    invariant_events_during_function_calls[event_dict["function"]] = (
                        set(events)
                    )

            elif event_dict["type"] == "state_variable_change":
                state_variable_changes += 1
            elif event_dict["type"] == "exception":
                exception_events += 1

            pbar.update(1)
        pbar.close()

        print(
            f"function_call_pres: {function_call_pres}, function_call_posts: {function_call_posts}, state_variable_changes: {state_variable_changes}, exception_events: {exception_events}"
        )
        return invariant_events_during_function_calls

    def dump(self):
        with open("trace.txt", "w") as f:
            for event in self.events:
                f.write(event.get_event() + "\n")
        pass


class TraceAnalyzer:
    def __init__(self, traces: "list[Trace]") -> None:
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
        help="Path to the trace file to be analyzed",
    )
    args = parser.parse_args()

    # read the trace from a file
    trace_lines = []
    with open(args.path, "r") as f:
        trace_lines = [
            Event(line.split(":trace:")[-1].strip())
            for line in f.readlines()
            if line.startswith("INFO:trace:") or line.startswith("ERROR:trace:")
        ]

    # create a trace object
    trace = Trace(trace_lines)
    result = trace.analyze()

    def default(o):
        if isinstance(o, set):
            return list(o)
        if isinstance(o, Event):
            return o.get_event()
        return o

    # dump the invariants
    with open("invariants.json", "w") as f:
        json.dump(result, f, indent=4, default=default)

    # event1 = Event(
    #     '{"process_id": 1, "thread_id": 1, "uuid": 1, "type": "function_call (pre)", "function": "foo"}'
    # )
    # event2 = Event(
    #     '{"process_id": 1, "thread_id": 1, "uuid": 2, "type": "function_call (pre)", "function": "foo"}'
    # )

    # print(set([event1]).intersection(set([event2])))
