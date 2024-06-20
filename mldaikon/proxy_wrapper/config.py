import os

proxy_log_dir = "proxy_log.json"  # FIXME: ad-hoc
disable_proxy_class = False  # Ziming: This feature is deprecated, proxy trace would work only when you manually add Proxy()
proxy_update_limit = 0
debug_mode = False

dump_tensor_version = False  # only dump the _version attribute of tensor
dump_tensor_statistics = False  # dump the statistics of tensor {min, max, mean, shape}
dump_call_return = False  # dump the return value of the function call
filter_by_tensor_version = True  # only dump the tensor when the version is changed

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
