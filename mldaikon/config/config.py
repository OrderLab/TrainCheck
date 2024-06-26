import polars as pl

TMP_FILE_PREFIX = "_ml_daikon_"
INSTR_MODULES_TO_INSTRUMENT = ["torch"]
INSTR_MODULES_TO_SKIP = [
    "torch.fx",
    # "torch.jit",
    # "torch._jit",
    # "torch._C",
    "torch._sources",  # FIXME: cannot handle this module, instrumenting it will lead to exceptions: TypeError: module, class, method, function, traceback, frame, or code object was expected, got builtin_function_or_method
]
INCLUDED_WRAP_LIST = ["Net", "DataParallel"]  # FIXME: Net & DataParallel seem ad-hoc
LIVENESS_OVERLAP_THRESHOLD = 0.01  # 1%
PROP_ATTR_PATTERNS = [  ## Attributes that are properties (i.e. they won't be the targets of invariants, but can be precondition or postcondition)
    "^is_.*$",  # e.g., is_cuda, is_contiguous
    "^has_.*$",  # e.g., has_names, has_storage
    "^can_.*$",  # e.g., can_cast, can_slice
]
PROP_ATTR_TYPES = [
    bool,
    pl.Boolean,
]

SKIP_INIT_VALUE_TYPES_KEY_WORDS = [  ## Types that should be skipped for initialization
    "tensor",
    "module",
    "parameter",
]

NOT_USE_AS_CLAUSE_FIELDS = ["func_call_id", "process_id", "thread_id", "time", "type"]

VAR_ATTR_PREFIX = "attributes."

CONST_CLAUSE_NUM_VALUES_THRESHOLD = 1  # FIXME: ad-hoc
INCOMPLETE_FUNC_CALL_SECONDS_TO_OUTERMOST_POST = 0.001  # only truncate the incomplete function call if it is no earlier than 1ms to the outermost function call's post event
