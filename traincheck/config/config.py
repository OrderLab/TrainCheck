import json
import os

import polars as pl
import yaml

# trace dumper configs:
BUFFER_SIZE = 1000  # number of events to buffer before dumping
FLUSH_INTERVAL = 0.5  # seconds

# runner configs
RUNNER_DEFAULT_ENV = {
    "PYTORCH_JIT": "0",
}

# tracer + instrumentor configs
TMP_FILE_PREFIX = "_traincheck_"
INSTR_OPTS_FILE = "instr_opts.json"
INSTR_MODULES_TO_INSTR = ["torch"]
INSTR_MODULES_TO_SKIP = [
    "torch.fx",
    "torch._dynamo",
    "torch._sources",  # FIXME: cannot handle this module, instrumenting it will lead to exceptions: TypeError: module, class, method, function, traceback, frame, or code object was expected, got builtin_function_or_method
    # "torch.autocast",
    # "torch.amp",
    # "torch.matmul",
    # "torch._VariableFunctionsClass.matmul",
    # "torch._tensor._convert",
]

# SKIP_INSTR_APIS specifies the APIs that should not be instrumented at all, often out of the consideration of performance and the fact that those APIs are not interesting for the analysis
# caveat: not instrumenting an API means that no additional handling of proxied objects will be done, so the APIs here must not be any native APIs.
SKIP_INSTR_APIS = [
    "torch.nn.modules.module.Module._call_impl",
    "torch.nn.modules.module.Module._wrapped_impl",
    "torch.nn.modules.module.Module._wrapped_call_impl",
]
WRAP_WITHOUT_DUMP = [
    "torch._C",
    "torch._jit",
    "torch.jit",
    "torch._tensor_str",
    "torch.overrides.handle_torch_function",
    "torch._ops.OpOverloadPacket",
    "torch._ops._OpNamespace",
    # added after doing initial evaluation
    "torch.nn.modules.module.Module._apply",
    "torch.autograd",
    "torch.cuda._is_compiled",
    "torch.cuda.is_available",
    "torch.overrides",
    "torch._ops.OpOverload",
    "torch.utils.data._utils",
    "torch.get_default_dtype",
    # "torch.autocast",
    # "torch.amp",
    # "torch.is_grad_enabled",
    # "torch.autograd.grad_mode",
    # "torch._VariableFunctionsClass",
    # "torch.get_default_dtype",
]
WRAP_WITHOUT_DUMP_WHITELIST = [
    "torch._C.TensorBase",
]

ANALYSIS_SKIP_FUNC_NAMES = [
    "cuda.is_available",
    "torch.get_default_dtype",
    "torch._VariableFunctionsClass",
    "torch.nn.modules.module.Module._call_impl",
    "torch.nn.modules.module.Module._wrapped_impl",
    "torch.nn.modules.module.Module._wrapped_call_impl",
    "torch.cuda._is_compiled",
    "torch.is_grad_enabled",
    "torch._ops.OpOverload",
    "torch.nn.modules.module.Module",
    "torch.utils.data._utils",
    "torch.distributed.utils._to_kwargs",
    "torch.distributed.utils._recursive_to",
    "torch.nn.parallel.scatter_gather._is_namedtuple",
    "torch.nn._reduction",
    "torch.nn.functional",
    "torch._tensor.Tensor.backward",  # no autograd invariants
    "torch.autograd",  # no autograd invariants
    "torch.optim.optimizer.Optimizer._optimizer_step_code",
    "torch.optim.optimizer._get_value",
    "torch.overrides",
    "._",  # skip all private functions (they can only be the contained, but not containing functions)
]

INSTR_OPTS = None  # TODO: set defaults for this variable

# var dumper related error-backoff configs
TYPE_ERR_THRESHOLD = 3
RECURSION_ERR_THRESHOLD = 5


class InstrOpt:
    def __init__(
        self,
        func_instr_opts: dict[str, dict[str, bool | dict]],
        model_tracker_style: str | None,
        disable_proxy_dumping: bool,
    ):
        assert model_tracker_style in [
            "sampler",
            "proxy",
            None,
        ], "model_tracker_style should be one of ['sampler', 'proxy', None]"

        self.funcs_instr_opts: dict[str, dict[str, bool | dict]] = func_instr_opts
        self.model_tracker_style = model_tracker_style
        self.disable_proxy_dumping = disable_proxy_dumping

    def to_json(self, indent=True) -> str:
        if indent:
            return json.dumps(
                {
                    "funcs_instr_opts": self.funcs_instr_opts,
                    "model_tracker_style": self.model_tracker_style,
                    "disable_proxy_dumping": self.disable_proxy_dumping,
                },
                indent=4,
            )
        else:
            return json.dumps(
                {
                    "funcs_instr_opts": self.funcs_instr_opts,
                    "model_tracker_style": self.model_tracker_style,
                    "disable_proxy_dumping": self.disable_proxy_dumping,
                },
            )

    @staticmethod
    def from_json(instr_opt_json_str: str):
        instr_opt_dict = yaml.safe_load(instr_opt_json_str)
        return InstrOpt(
            instr_opt_dict["funcs_instr_opts"],
            instr_opt_dict["model_tracker_style"],
            instr_opt_dict["disable_proxy_dumping"],
        )


def load_instr_opts():
    global INSTR_OPTS
    if INSTR_OPTS is None and os.path.exists(INSTR_OPTS_FILE):
        with open(INSTR_OPTS_FILE, "r") as f:
            INSTR_OPTS = InstrOpt.from_json(f.read())
    return INSTR_OPTS


def should_disable_proxy_dumping() -> bool:
    return INSTR_OPTS is not None and INSTR_OPTS.disable_proxy_dumping


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
PROP_ATTR_TYPES = [  # we can't use this due to difficulties in determining the type of the variable (pd.Object == pl.Boolean --> True??)
    bool,
    pl.Boolean,
]
PROP_ATTR_PATTERNS = [  ## Attributes that are properties (i.e. they won't be the targets of invariants, but can be precondition or postcondition)
    "^is_.*$",  # e.g., is_cuda, is_contiguous
    "^has_.*$",  # e.g., has_names, has_storage
    "^can_.*$",  # e.g., can_cast, can_slice
]

# precondition inference configs
MAX_PRECOND_DEPTH = 8  # the maximum depth of the precondition inference
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
CONST_CLAUSE_STR_NUM_VALUES_THRESHOLD = 4  # FIXME: ad-hoc
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
    "recurse",
    "var_name",
    "mode",
    "process_id",
    "thread_id",
    "dumped_frame_array",
    "func_call_id",
    "traincheck_folder",
    "enable_auto_observer_depth",
    "neglect_hidden_func",
    "neglect_hidden_module",
    "observe_then_unproxy",
    "observe_up_to_depth",
    "log_file",
    "log_dir",
]

# question: can we use optimizer state to get step? That sounds more robust
# TRAIN_STEP_NAMES = [
#     "iter",
#     "iteration",
#     "step",
#     "batch_id",
# ]


INSTR_DESCRIPTORS = False

ALL_STAGE_NAMES = {
    "init",
    "training",
    "evaluation",
    "inference",
    "testing",
    "checkpointing",
    "preprocessing",
    "postprocessing",
}
