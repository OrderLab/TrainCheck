TMP_FILE_PREFIX = "_ml_daikon_"
MODULES_TO_INSTRUMENT = ["torch"]
INCLUDED_WRAP_LIST = ["Net", "DataParallel"]  # FIXME: Net & DataParallel seem ad-hoc
proxy_log_dir = "proxy_log.json"  # FIXME: ad-hoc
disable_proxy_class = False  # Ziming: Currently disable proxy_class in default
proxy_update_limit = 0
debug_mode = False
meta_var_black_list = [
    'pre_process',
    'post_process',
    '__name__',
    '__file__',
    '__loader__',
    '__doc__',
    'logdir',
    'log_level',
    'is_root',
    'var_name',
    'dumped_frame_array',
]
profiling = False
LIVENESS_OVERLAP_THRESHOLD = 0.01  # 1%
PROP_ATTR_PATTERNS = [  ## Attributes that are properties (i.e. they won't be the targets of invariants, but can be precondition or postcondition)
    "^is_.*$",  # e.g., is_cuda, is_contiguous
    "^has_.*$",  # e.g., has_names, has_storage
    "^can_.*$",  # e.g., can_cast, can_slice
]
PROP_ATTR_TYPES = [bool]
CONST_CLAUSE_NUM_VALUES_THRESHOLD = 10  # FIXME: ad-hoc
