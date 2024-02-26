import json


class Event:
    """
    An event is a significant occurrence during the execution of a program.
    e.g. API Invocations, Model Parameter Changes, etc.
    """

    def __init__(self, event: str):
        self.event = event

    def __repr__(self) -> str:
        return self.event

    def __hash__(self):
        self_dict = json.loads(self.event)
        self_dict.pop("process_id", None)
        self_dict.pop("thread_id", None)
        self_dict.pop("uuid", None)
        return hash(json.dumps(self_dict, sort_keys=True))
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def get_event(self):
        return self.event

    def set_event(self, event: str):
        self.event = event


class Trace:
    """
    A trace is a sequence of events that occurred during the execution of a program.
    """

    def __init__(self, events: list[Event]) -> None:
        self.events = events

    def analyze(self):
        """
        Current Status:
        - This is specifically implemented for PyTorch-FORUM84911. The analysis tries to find events that always occur during function calls.
        """
        unique_processes = set()
        unique_threads = set()
        for event in self.events:
            event_dict = json.loads(event.get_event())
            unique_threads.add(event_dict["thread_id"])
            unique_processes.add(event_dict["process_id"])

        print(
            f"Num of processes: {len(unique_processes)}, Process Ids: {unique_processes}"
        )
        print(f"Num of threads: {len(unique_threads)}, Thread Ids: {unique_threads}")

        invariant_events_during_function_calls = {}
        for pid in unique_processes:
            for tid in unique_threads:
                result = self.analyze_local(tid, pid)
                # dump this result for debugging
                def default(o):
                    if isinstance(o, set):
                        return list(o)
                    if isinstance(o, Event):
                        return o.get_event()
                    return o

                # dump the invariants
                with open(f"invariants_{pid}_{tid}.json", "w") as f:
                    json.dump(result, f, indent=4, default=default)


                for k, v in result.items():
                    if k in invariant_events_during_function_calls:
                        invariant_events_during_function_calls[k] = (
                            invariant_events_during_function_calls[k].intersection(v)
                        )
                    else:
                        invariant_events_during_function_calls[k] = v

        return invariant_events_during_function_calls

    def analyze_local(self, thread_id, process_id):
        """
        Analyze the trace for a specific thread in a specific process.
        Assumes that within a thread, the events are ordered by their occurrence (True in Python).
        """
        function_call_pres = 0
        function_call_posts = 0
        state_variable_changes = 0
        exception_events = 0

        invariant_events_during_function_calls = {}
        stack_current_function_calls = []
        for i, event in enumerate(self.events):
            # each event is a json string
            event_dict = json.loads(event.get_event())
            if (
                event_dict["thread_id"] != thread_id
                or event_dict["process_id"] != process_id
            ):
                continue

            # if event_dict["function"] == "set_num_threads" or event_dict["function"] == "_get_tracing_state":
            #     # TODO: This is a hack to ignore set_num_threads leading to assertion errors in the current implementation. We should handle this properly later.
            #     continue

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
                ), f"The function call pre and post events should match, i: {i}, thread_id: {thread_id}, process_id: {process_id}, uuid: {event_dict['uuid']}, stack_call pre: {stack_current_function_calls[-1][0]}, curr post: {event_dict['function']}"

                # pop the function from the stack
                _, _, i_pre = stack_current_function_calls.pop()

                # collect events between the function call pre and post events
                events = self.events[i_pre + 1 : i]

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
    def __init__(self, traces: list[Trace]) -> None:
        self.traces = traces
        pass

    def analyze(self):
        pass


if __name__ == "__main__":
    # read the trace from a file
    trace_lines = []
    with open("log.txt", "r") as f:
        trace_lines = [
            Event(l.split(":trace:")[-1].strip())
            for l in f.readlines()
            if l.startswith("INFO:trace:") or l.startswith("ERROR:trace:")
        ]

    # create a trace object
    trace = Trace(trace_lines)
    trace.analyze()

    # event1 = Event(
    #     '{"process_id": 1, "thread_id": 1, "uuid": 1, "type": "function_call (pre)", "function": "foo"}'
    # )
    # event2 = Event(
    #     '{"process_id": 1, "thread_id": 1, "uuid": 2, "type": "function_call (pre)", "function": "foo"}'
    # )

    # print(set([event1]).intersection(set([event2])))