proxy_log_dir = "proxy_log.json"  # FIXME: ad-hoc
disable_proxy_class = False  # Ziming: Currently disable proxy_class in default
proxy_update_limit = 15
debug_mode = False
meta_var_black_list = [
    "pre_process",
    "post_process",
    "__name__",
    "__file__",
    "__loader__",
    "__doc__",
    "logdir",
    "log_level",
    "is_root",
    "var_name",
    "mode",
    "process_id",
    "thread_id",
    "dumped_frame_array",
    "func_call_id",
]
tensor_attribute_black_list = [
    "T",
    "mT",
    "H",
    "mH",
    "volatile",
    "output_nr",
    "version",
    "_backward_hooks",
    "_backward_hooks",
    "_version",
    "real",
]
attribute_black_list = tensor_attribute_black_list

profiling = False

import os
# ad-hoc: should be the super super foler of this file
mldaikon_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(mldaikon_folder)
exclude_file_names = []
for root, dirs, files in os.walk(mldaikon_folder):
    for file in files:
        if file.endswith('.py'):
            exclude_file_names.append(os.path.join(root, file))
            
print(exclude_file_names)