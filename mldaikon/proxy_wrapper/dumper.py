import json
import time


class json_dumper:

    def __init__(self, json_file_path):
        self.json_file = open(json_file_path, "a")

    def dump_json(
        self, process_id, thread_id, meta_vars, variable_name, var_properties_changed
    ):
        data = {
            "process_id": process_id,
            "thread_id": thread_id,
            "timestamp": time.time(),
            "meta_vars": json.dumps(str(meta_vars)),
            "variable_name": variable_name,
            "var_properties_changed": var_properties_changed,
        }
        print(data)
        json_data = json.dumps(data)

        self.json_file.write(json_data + "\n")

    def close(self):
        self.json_file.close()

    def create_instance(self):
        return json_dumper(self.json_file.name)
