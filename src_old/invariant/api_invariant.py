from abc import ABC, abstractmethod

import tqdm

from .event import Event


class APIInvariant(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def analyze(self):
        pass


class APIInvariantConstantEvents(APIInvariant):
    invariant_type = "constant_events_in_api"

    def __init__(self, events: list[Event]):
        self.events: list[Event] = (
            events  # should this be per-thread? I think so because we would have a lot of Invariant Implementation and we cannot duplicate the trace splitting logic everywhere
        )
        self.has_analyzed = False
        self.invariant_properties: dict[str, dict[str, object]] = {
            "constant_events_during_api_calls": {}
        }  # properties that are always the same

        # TODO: assert checks at **init time** to ensure that the events are at least from the same process & thread

    def get_invariant_properties(self):
        if not self.has_analyzed:
            self.analyze()
        return self.invariant_properties

    def analyze(self):
        """Find events that always happen between the API entry and exit"""
        invariant_events_during_function_calls = {}
        stack_current_function_calls = []

        pid, tid = (
            self.events[0].event_dict["process_id"],
            self.events[0].event_dict["thread_id"],
        )  # all events should be from the same process and thread

        pbar = tqdm.tqdm(total=len(self.events))
        for i, event in enumerate(self.events):

            event_dict: dict = event.get_event_dict()

            assert (
                event_dict["process_id"] == pid and event_dict["thread_id"] == tid
            ), f"Event pid and tid should match the input pid and tid, i: {i}, tid: {tid}, pid: {pid}, uuid: {event_dict['uuid']}"

            if event_dict["type"] == "function_call (pre)":
                # add the function and its index to the stack
                stack_current_function_calls.append(
                    (event_dict["function"], event_dict["uuid"], i)
                )

            elif (
                event_dict["type"] == "function_call (post)"
                or event_dict["type"] == "function_call (post) (exception)"
            ):
                # TODO: handle the exception case separately. Currently, we are treating it as a normal function call post event in the analysis.
                assert (
                    len(stack_current_function_calls) > 0
                ), "There should be a function call pre event before a function call post event"
                assert (
                    stack_current_function_calls[-1][0] == event_dict["function"]
                ), f"The function call pre and post events should match, i: {i}, tid: {tid}, pid: {pid}, uuid: {event_dict['uuid']}, stack_call pre: {stack_current_function_calls[-1][0]}, curr post: {event_dict['function']}"

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
            pbar.update(1)
        pbar.close()

        self.has_analyzed = True
        self.invariant_properties["constant_events_during_api_calls"] = (
            invariant_events_during_function_calls
        )

        return self.invariant_properties
