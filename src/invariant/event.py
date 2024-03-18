import json


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
