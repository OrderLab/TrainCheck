import types

proxy_log_dir = "proxy_log.json"  # FIXME: ad-hoc
debug_mode = False

tensor_dump_format = {
    "dump_tensor_hash": True,  # dump the hash of the tensor
    "dump_tensor_stats": False,  # dump the statistics of tensor {min, max, mean, shape}
    "dump_tensor_full": False,  # dump the full tensor
}

dump_info_config = {
    "dump_call_return": False,  # dump the return value of the function call
    "dump_iter": False,  # dump the variable states from iterator (this would usually generated from e.g. enumerate(self._blocks) function)
    "dump_update_only": False,  # only dump the updated part of the proxied object
    "filter_by_tensor_version": False,  # only dump the tensor when the version is changed
}

auto_observer_config = {
    "enable_auto_observer": True,  # automatically add observer to the function
    "enable_auto_observer_depth": 3,  # the depth of the function call that we want to observe
    "observe_up_to_depth": False,  # observe up to the depth of the function call, if False, only observe the function call at the depth
    "neglect_hidden_func": True,  # neglect the hidden function (function that starts with '_')
    "neglect_hidden_module": True,  # neglect the hidden module (module that starts with '_')
    "observe_then_unproxy": True,  # observe the function call and then unproxy the arguments
}

enable_C_level_observer = False  # enable the observer at the C level (This would potentially lead to a lot of overhead since we need to observe and dump all proxied object at the C level function call, try to use auto observer with proper depth could reduce the overhead)

primitive_types = {
    types.NoneType,
    int,
    float,
    str,
    bool,
}  # the primitive types that we want to filter out

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
