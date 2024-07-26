import os

proxy_log_dir = "proxy_log.json"  # FIXME: ad-hoc
disable_proxy_class = False  # Ziming: This feature is deprecated, proxy trace would work only when you manually add Proxy()
proxy_update_limit = 5
debug_mode = False

delta_dump_config = {
    "delta_dump": False,  # only dump the changed part of the object (if this is set to be False, we would dump the whole object no matter what values delta_dump_meta_var and delta_dump_attribute are)
    "delta_dump_meta_var": True,  # only dump the changed part of the meta_var
    "delta_dump_attributes": True,  # only dump the changed part of the attribute
}
tensor_dump_format = {
    "dump_tensor_version": False,  # only dump the _version attribute of tensor
    "dump_tensor_hash": True,  # dump the hash of the tensor
    "dump_tensor_statistics": False,  # dump the statistics of tensor {min, max, mean, shape}
}

dump_info_config = {
    "dump_call_return": False,  # dump the return value of the function call
    "dump_iter": False,  # dump the variable states from iterator (this would usually generated from e.g. enumerate(self._blocks) function)
    "dump_update_only": False,  # only dump the updated part of the proxied object
    "filter_by_tensor_version": False,  # only dump the tensor when the version is changed
}

auto_observer_config = {
    "enable_auto_observer": True,  # automatically add observer to the function
    "only_dump_when_change": True,  # only dump the variable when it is changed
    "enable_auto_observer_depth": 3,  # the depth of the function call that we want to observe
    "observe_up_to_depth": False,  # observe up to the depth of the function call, if False, only observe the function call at the depth
    "neglect_hidden_func": True,  # neglect the hidden function (function that starts with '_')
    "neglect_hidden_module": True,  # neglect the hidden module (module that starts with '_')
    "observe_then_unproxy": True,  # observe the function call and then unproxy the arguments
}

enable_C_level_observer = False  # enable the observer at the C level (This would potentially lead to a lot of overhead since we need to observe and dump all proxied object at the C level function call, try to use auto observer with proper depth could reduce the overhead)

primitive_types = {
    int,
    float,
    str,
    bool,
}  # the primitive types that we want to filter out

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
    "mldaikon_folder",
    "enable_auto_observer_depth",
    "neglect_hidden_func",
    "neglect_hidden_module",
    "observe_then_unproxy",
    "observe_up_to_depth",
    "log_file",
    "log_dir",
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

# ad-hoc: should be the super super foler of this file
mldaikon_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(mldaikon_folder)
exclude_file_names = []
for root, dirs, files in os.walk(mldaikon_folder):
    for file in files:
        if file.endswith(".py"):
            exclude_file_names.append(os.path.join(root, file))
