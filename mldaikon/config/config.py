import polars as pl

# tracer + instrumentor configs
TMP_FILE_PREFIX = "_ml_daikon_"
INSTR_MODULES_TO_INSTR = ["torch"]
INSTR_MODULES_TO_SKIP = [
    "torch.fx",
    "torch._sources",  # FIXME: cannot handle this module, instrumenting it will lead to exceptions: TypeError: module, class, method, function, traceback, frame, or code object was expected, got builtin_function_or_method
]
WRAP_WITHOUT_DUMP = [
    "torch._C",
    "torch._jit",
    "torch.jit",
    "torch._tensor_str",
    # "torch.is_grad_enabled",
    # "torch.autograd.grad_mode",
    # "torch._VariableFunctionsClass",
    # "torch.get_default_dtype",
]

# consistency relation configs
SKIP_INIT_VALUE_TYPES_KEY_WORDS = (
    [  ## Types whose initialization values should not be considered
        "tensor",
        "module",
        "parameter",
    ]
)
LIVENESS_OVERLAP_THRESHOLD = 0.01  # 1%
POSITIVE_EXAMPLES_THRESHOLD = 2  # in ConsistencyRelation, we need to see at least two positive examples on one pair of variables to add a hypothesis for their types


# trace configs
VAR_ATTR_PREFIX = "attributes."
INCOMPLETE_FUNC_CALL_SECONDS_TO_OUTERMOST_POST = 0.001  # only truncate the incomplete function call if it is no earlier than 1ms to the outermost function call's post event
PROP_ATTR_TYPES = [
    bool,
    pl.Boolean,
]
PROP_ATTR_PATTERNS = [  ## Attributes that are properties (i.e. they won't be the targets of invariants, but can be precondition or postcondition)
    "^is_.*$",  # e.g., is_cuda, is_contiguous
    "^has_.*$",  # e.g., has_names, has_storage
    "^can_.*$",  # e.g., can_cast, can_slice
]

# precondition inference configs
ENABLE_PRECOND_SAMPLING = True  # whether to enable sampling of positive and negative examples for precondition inference, can be overridden by the command line argument
PRECOND_SAMPLING_THRESHOLD = 10000  # the number of samples to take for precondition inference, if the number of samples is larger than this threshold, we will sample this number of samples
NOT_USE_AS_CLAUSE_FIELDS = [
    "func_call_id",
    "process_id",
    "thread_id",
    "time",
    "type",
    "mode",
    "stack_trace",
    "obj_id",
]
CONST_CLAUSE_NUM_VALUES_THRESHOLD = 1  # FIXME: ad-hoc
VAR_INV_TYPE = (
    "type"  # how to describe the variable in the invariant, can be "type" or "name"
)

# Ziming: added config for e2e pipeline
API_LOG_DIR = "."

META_VARS_FORBID_LIST = [
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

ENABLE_COND_DUMP = False