import glob
import os

proxy_log_dir = "proxy_log.json"  # FIXME: ad-hoc
disable_proxy_class = False  # Ziming: This feature is deprecated, proxy trace would work only when you manually add Proxy()
proxy_update_limit = 0
debug_mode = False

dump_tensor_version = False  # only dump the _version attribute of tensor
dump_tensor_statistics = False  # dump the statistics of tensor {min, max, mean, shape}
dump_call_return = False  # dump the return value of the function call
dump_iter = False  # dump the variable states from iterator (this would usually generated from e.g. enumerate(self._blocks) function)
dump_update_only = False  # only dump the updated part of the proxied object
filter_by_tensor_version = False  # only dump the tensor when the version is changed

enable_auto_observer = False  # automatically add observer to the function
enable_auto_observer_depth = 3  # the depth of the function call that we want to observe
observe_up_to_depth = False  # observe up to the depth of the function call, if False, only observe the function call at the depth
neglect_hidden_func = (
    True  # neglect the hidden function (function that starts with '_')
)
neglect_hidden_module = True  # neglect the hidden module (module that starts with '_')
observe_then_unproxy = False  # observe the function call and then unproxy the arguments

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

print(exclude_file_names)

if enable_auto_observer:
    print("auto observer enabled with observing depth: ", enable_auto_observer_depth)
    if observe_up_to_depth:
        print("observe up to the depth of the function call")
    else:
        print("observe only the function call at the depth")
    from mldaikon.static_analyzer.call_graph_parser import call_graph_parser

    log_files = glob.glob(
        os.path.join(mldaikon_folder, "static_analyzer", "func_level", "*.log")
    )
    for log_file in log_files:
        call_graph_parser(
            log_file,
            depth=enable_auto_observer_depth,
            observe_up_to_depth=observe_up_to_depth,
            neglect_hidden_func=neglect_hidden_func,
            neglect_hidden_module=neglect_hidden_module,
            observe_then_unproxy=observe_then_unproxy,
        )
