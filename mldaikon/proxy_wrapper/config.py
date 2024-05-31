proxy_log_dir = "proxy_log.json"  # FIXME: ad-hoc
disable_proxy_class = False  # Ziming: Currently disable proxy_class in default
proxy_update_limit = 10
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
]
attribute_black_list = tensor_attribute_black_list

profiling = False
