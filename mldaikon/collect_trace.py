import argparse
import datetime
import logging
import os

import yaml

import mldaikon.config.config as config
import mldaikon.instrumentor as instrumentor
import mldaikon.proxy_wrapper.proxy_config as proxy_config
import mldaikon.runner as runner
from mldaikon.config.config import InstrOpt
from mldaikon.invariant.base_cls import (
    APIParam,
    Arguments,
    InputOutputParam,
    Invariant,
    VarNameParam,
    VarTypeParam,
    read_inv_file,
)
from mldaikon.invariant.consistency_relation import ConsistencyRelation
from mldaikon.invariant.contain_relation import VAR_GROUP_NAME, APIContainRelation


def get_list_of_funcs_from_invariants(invariants: list[Invariant]) -> list[str]:
    """
    Get a list of functions from the invariants
    """
    funcs = set()
    for inv in invariants:
        for param in inv.params:
            if isinstance(param, APIParam):
                funcs.add(param.api_full_name)
    return sorted(list(funcs))


def get_per_func_instr_opts(
    invariants: list[Invariant],
) -> dict[str, dict[str, bool | dict]]:
    """
    Get per function instrumentation options
    """

    # TODO: for APIContainRelation that describes a variable, if the precondition is not unconditional on the variable and the API belongs to a class, then all class methods should be instrumented with `scan_proxy_in_args` set to True

    func_instr_opts: dict[str, dict[str, bool | dict]] = {}
    for inv in invariants:
        for param in inv.params:
            if isinstance(param, APIParam) or isinstance(param, InputOutputParam):
                func_name = (
                    param.api_full_name
                    if isinstance(param, APIParam)
                    else param.api_name
                )
                assert isinstance(func_name, str)
                if func_name not in func_instr_opts:
                    func_instr_opts[func_name] = {
                        "scan_proxy_in_args": False,
                        "dump_args": False,
                        "dump_ret": False,  # not really used for now
                    }

            if isinstance(param, APIParam):
                func_name = param.api_full_name

                def merge(a: dict, b: dict, path=[]):
                    for key in b:
                        if key in a:
                            if isinstance(a[key], dict) and isinstance(b[key], dict):
                                merge(a[key], b[key], path + [str(key)])
                            elif a[key] != b[key]:
                                if a[key] is None:
                                    a[key] = b[key]
                                elif b[key] is None:
                                    pass
                                else:
                                    raise Exception(
                                        "Conflict at "
                                        + ".".join(path + [str(key)])
                                        + f" {a[key]} != {b[key]}"
                                    )
                        else:
                            a[key] = b[key]
                    return a

                if isinstance(param.arguments, Arguments):
                    func_instr_opts[func_name]["dump_args"] = True
                    arg_instr_opt = param.arguments.to_instr_opts_dict()
                    if "dump_args_config" in func_instr_opts[func_name]:
                        existing_config = func_instr_opts[func_name]["dump_args_config"]
                        assert isinstance(existing_config, dict)
                        func_instr_opts[func_name]["dump_args_config"] = merge(
                            existing_config,
                            arg_instr_opt,
                        )
                    else:
                        func_instr_opts[func_name]["dump_args_config"] = arg_instr_opt

            if isinstance(param, InputOutputParam):
                func_name = param.api_name
                assert isinstance(func_name, str)
                func_instr_opts[func_name]["dump_args"] = True
                func_instr_opts[func_name]["dump_ret"] = True
                # TODO: convert the arguments to instr_opts_dict (currently not possible as the index indicates the index of the argument/ret value among other tensors not all arguments)
                logger.warning(
                    "Currently not supporting fine-grained dumping of arguments and return values for InputOutputParam"
                )

        if inv.relation == APIContainRelation:
            assert isinstance(inv.params[0], APIParam)
            assert inv.precondition is not None
            if (
                isinstance(inv.params[1], (VarNameParam, VarTypeParam))
                and VAR_GROUP_NAME in inv.precondition.get_group_names()
                and not inv.precondition.get_group(VAR_GROUP_NAME).is_unconditional()
            ):
                # if the APIContain invariant describes a variable, and the precondition is not unconditional on the variable, then scan the arguments of the function
                func_instr_opts[inv.params[0].api_full_name][
                    "scan_proxy_in_args"
                ] = True
            else:
                func_instr_opts[inv.params[0].api_full_name][
                    "scan_proxy_in_args"
                ] = False

    return func_instr_opts


def get_model_tracker_instr_opts(invariants: list[Invariant]) -> str | None:
    """
    Get model tracker instrumentation options
    """

    tracker_type = None
    for inv in invariants:
        if inv.relation == APIContainRelation:
            for param in inv.params:
                if isinstance(param, (VarNameParam, VarTypeParam)):
                    tracker_type = "proxy"
                    break
        if tracker_type is None and inv.relation == ConsistencyRelation:
            tracker_type = "sampler"

        if tracker_type == "proxy":
            break
    return tracker_type


def dump_env(output_dir: str):
    with open(os.path.join(output_dir, "env_dump.txt"), "w") as f:
        f.write("Arguments:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write("\n")
        f.write("Environment Variables:\n")
        for key, value in os.environ.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Python Version:\n")
        f.write(f"{os.popen('python --version').read()}\n")
        f.write("\n")
        f.write("Library Versions:\n")
        f.write(
            f"{os.popen('conda list').read()}\n"
        )  # FIXME: conda list here doesn't work in OSX, >>> import os; >>> os.popen('conda list').read(); /bin/sh: conda: command not found


def get_default_output_folder(args: argparse.Namespace) -> str:
    """Get the default output directory for the trace collection
    Note that the output is only the folder name, not an absolute path
    """
    pyfile_basename = os.path.basename(args.pyscript).split(".")[0]
    # get also the versions of the modules specified in `-t`
    modules = args.modules_to_instr
    modules_and_versions = []
    for module in modules:
        try:
            # this may not work if the module is not installed (e.g. only used locally)
            version = (
                os.popen(f"pip show {module} | grep Version")
                .read()
                .strip()
                .split(": ")[1]
            )
        except Exception as e:
            logger.warning(f"Could not get version of module {module}: {e}")
            version = "unknown"
        modules_and_versions.append(f"{module}_{version}")
    # sort the modules and versions
    modules_and_versions.sort()
    output_folder = f"mldaikon_run_{pyfile_basename}_{'_'.join(modules_and_versions)}_{START_TIME.strftime('%Y-%m-%d_%H-%M-%S')}"
    return output_folder


if __name__ == "__main__":
    # First parse the deciding arguments.
    use_config_args_parser = argparse.ArgumentParser(add_help=False)
    use_config_args_parser.add_argument(
        "--use-config",
        required=False,
        action="store_true",
        help="Use the configuration file to set the arguments, additionally provided arguments will complement the configuration file but not override it",
    )
    use_config_args, _ = use_config_args_parser.parse_known_args()
    use_config = use_config_args.use_config
    cmd_args_required = not use_config

    parser = argparse.ArgumentParser(
        description="Invariant Finder for ML Pipelines in Python",
        parents=[use_config_args_parser],
    )

    ## general configs
    parser.add_argument(
        "--config",
        type=str,
        required=use_config,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-p",
        "--pyscript",
        type=str,
        required=cmd_args_required,
        help="Path to the main file of the pipeline to be analyzed",
    )
    parser.add_argument(
        "-s",
        "--shscript",
        type=str,
        required=False,
        help="""Path to the shell script that runs the python script. 
        If not provided, the python script will be run directly.""",
    )
    parser.add_argument(
        "-a",
        "--copy-all-files",
        action="store_true",
        help="""Copy all files in pyscript's folder to the result directory, 
        this is necessary if you have relative paths in your code""",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="""Directory to store the output files, if not provided, it will be 
        defaulted to mldaikon_run_{pyscript_name}_{timestamp}""",
    )
    parser.add_argument(
        "--only-instr",
        action="store_true",
        help="Only instrument and dump the modified file",
    )
    parser.add_argument(
        "--instr-descriptors",
        action="store_true",
        help="""Instrument functions that can only be accessed through descriptors, 
        Set this to true if you want to instrument built-in types like torch.Tensor, 
        at the cost of larger (5x) instrumentation overhead and more interference with the program""",
    )
    parser.add_argument(
        "--profiling",
        default=False,
        action="store_true",
        help="Profile the instrumented program using py-spy, sudo may be required",
    )
    parser.add_argument(
        "-d",
        "--debug-mode",
        action="store_true",
        help="Enable debug mode for the program",
    )

    ## instrumentor configs
    parser.add_argument(
        "-t",
        "--modules-to-instr",
        nargs="*",
        help="Modules to be instrumented",
        default=config.INSTR_MODULES_TO_INSTR,
    )
    parser.add_argument(
        "--disable-scan-proxy-in-args",
        action="store_true",
        help="NOT Scan the arguments of the function for proxy objects, this will enable the infer engine to understand the relationship between the proxy objects and the functions. Overriden to False if Proxy-based variable tracking is not needed",
    )
    parser.add_argument(
        "--API-dump-stack-trace",
        action="store_true",
        help="Dump the stack trace for API calls",
    )
    parser.add_argument(
        "-i",
        "--invariants",
        nargs="*",
        help="Invariant files produced by the inference engine. If provided, we will only collect traces for APIs and variables that are related to the invariants. This can be used to speed up the trace collection. HAS TO BE USED WITHOUT `--use-full-instr` for the optimization to work properly.",
        default=None,
    )
    parser.add_argument(
        "--use-full-instr",
        action="store_true",
        help="Use full instrumentation for the instrumentor, if not set, the instrumentor may not dump traces for certain APIs in modules deemed not important (e.g. jit in torch)",
    )
    parser.add_argument(
        "--cond-dump",
        action="store_true",
        help="Dump the conditions for the APIs conditionally, currently only dumps an API if meta_var has changed since the last dump",
    )

    ## variable tracker configs
    parser.add_argument(
        "--models-to-track",
        nargs="*",
        help="Models to be tracked through instrumentation",
    )
    parser.add_argument(
        "--model-tracker-style",
        type=str,
        choices=["sampler", "proxy"],
        default="proxy",
    )
    parser.add_argument(
        "--proxy-update-limit",
        type=float,
        default=proxy_config.proxy_update_limit,
        help="The threshold for updating the proxy object",
    )
    parser.add_argument(
        "--tensor-dump-format",
        choices=["hash", "stats", "full"],
        type=str,
        default="hash",
        help="The format for dumping tensors. Choose from 'hash'(default), 'stats' or 'full'.",
    )
    parser.add_argument(
        "--delta-dump",
        type=bool,
        default=proxy_config.delta_dump_config["delta_dump"],
        help="Only dump the changed part of the object",
    )
    parser.add_argument(
        "--delta-dump-meta-var",
        type=bool,
        default=proxy_config.delta_dump_config["delta_dump_meta_var"],
        help="Only dump the changed part of the meta_var",
    )
    parser.add_argument(
        "--delta-dump-attributes",
        type=bool,
        default=proxy_config.delta_dump_config["delta_dump_attributes"],
        help="Only dump the changed part of the attribute",
    )
    parser.add_argument(
        "--enable-C-level-observer",
        type=bool,
        default=proxy_config.enable_C_level_observer,
        help="Enable the observer at the C level",
    )

    args = parser.parse_args()

    # read the configuration file
    if not use_config and args.config:
        raise ValueError("Configuration file provided without --use-config flag")
    if use_config:
        config_file = args.config
        with open(config_file, "r") as f:
            config_args = yaml.safe_load(f)
        for key, value in config_args.items():
            if not hasattr(args, key):
                raise ValueError(f"Invalid configuration key: {key}")
            setattr(args, key, value)

    # set up logging
    if args.debug_mode:
        logging.basicConfig(level=logging.DEBUG)
        os.environ["ML_DAIKON_DEBUG"] = "1"
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    START_TIME = datetime.datetime.now()

    output_dir = args.output_dir
    if not output_dir:
        output_dir = get_default_output_folder(args)
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dump_env(output_dir)

    # set up adjusted proxy_config
    proxy_basic_config: dict[str, int | bool | str] = {}
    for configs in [
        "proxy_update_limit",
        "debug_mode",
        "enable_C_level_observer",
    ]:
        if getattr(proxy_config, configs) != getattr(args, configs):
            proxy_basic_config[configs] = getattr(args, configs)
            print(f"Setting {configs} to {getattr(args, configs)}")
    proxy_log_output_dir = os.path.join(output_dir, "proxy_log.json")
    proxy_basic_config["proxy_log_dir"] = proxy_log_output_dir

    # set up tensor_dump_format
    tensor_dump_format: dict[str, int | bool] = {}
    if args.tensor_dump_format != "hash":
        tensor_dump_format = proxy_config.tensor_dump_format  # type: ignore
        # set all to False
        for key in tensor_dump_format:
            tensor_dump_format[key] = False
        # set the chosen one to True
        tensor_dump_format[f"dump_tensor_{args.tensor_dump_format}"] = True

    # set up delta_dump_config
    delta_dump_config: dict[str, int | bool] = {}
    for configs in ["delta_dump", "delta_dump_meta_var", "delta_dump_attributes"]:
        if proxy_config.delta_dump_config[configs] != getattr(args, configs):
            delta_dump_config[configs] = getattr(args, configs)
            print(f"Setting {configs} to {getattr(args, configs)}")

    auto_observer_config = proxy_config.auto_observer_config
    # call into the instrumentor
    adjusted_proxy_config: list[dict] = [
        auto_observer_config,  # Ziming: add auto_observer_config for proxy_wrapper
        proxy_basic_config,  # Ziming: add proxy_basic_config for proxy_wrapper
        tensor_dump_format,  # Ziming: add tensor_dump_format for proxy_wrapper
        delta_dump_config,  # Ziming: add delta_dump_config for proxy_wrapper
    ]

    # if args.disable_scan_proxy_in_args:
    scan_proxy_in_args = not args.disable_scan_proxy_in_args

    # if no proxy tracking specified in the arguments, disable the scan_proxy_in_args
    if not args.models_to_track or args.model_tracker_style != "proxy":
        scan_proxy_in_args = False

    if args.invariants:
        # selective instrumentation if invariants are provided, only funcs_to_instr will be instrumented with trace collection
        invariants = read_inv_file(args.invariants)
        instr_opts = InstrOpt(
            func_instr_opts=get_per_func_instr_opts(invariants),
            model_tracker_style=get_model_tracker_instr_opts(invariants),
            is_instr_selective=True,
        )
        models_to_track = (
            args.models_to_track if instr_opts.model_tracker_style else None
        )
        if models_to_track is None:
            logger.warning(
                """Model tracker is needed as per the invariants, 
but which to track is not specified in the arguments/md-config file, 
disabling model tracking."""
            )
            instr_opts.model_tracker_style = None

        with open(os.path.join(output_dir, config.INSTR_OPTS_FILE), "w") as f:
            f.write(instr_opts.to_json())
        source_code = instrumentor.instrument_file(
            path=args.pyscript,
            modules_to_instr=args.modules_to_instr,
            scan_proxy_in_args=scan_proxy_in_args,
            use_full_instr=args.use_full_instr,
            funcs_to_instr=None,
            models_to_track=models_to_track,
            model_tracker_style=instr_opts.model_tracker_style,
            adjusted_proxy_config=adjusted_proxy_config,  # type: ignore
            API_dump_stack_trace=args.API_dump_stack_trace,
            cond_dump=args.cond_dump,
            output_dir=output_dir,
            instr_descriptors=args.instr_descriptors,
        )
    else:
        source_code = instrumentor.instrument_file(
            path=args.pyscript,
            modules_to_instr=args.modules_to_instr,
            scan_proxy_in_args=scan_proxy_in_args,
            use_full_instr=args.use_full_instr,
            funcs_to_instr=None,
            models_to_track=args.models_to_track,
            model_tracker_style=args.model_tracker_style,
            adjusted_proxy_config=adjusted_proxy_config,  # type: ignore
            API_dump_stack_trace=args.API_dump_stack_trace,
            cond_dump=args.cond_dump,
            output_dir=output_dir,
            instr_descriptors=args.instr_descriptors,
        )

    if args.copy_all_files:
        # copy all files in the same directory as the pyscript to the output directory
        parent_dir = os.path.dirname(args.pyscript)
        if parent_dir == "":
            parent_dir = "."
        for file in os.listdir(parent_dir):
            os.system(f"cp -r {os.path.join(parent_dir, file)} {output_dir}")

    # call into the program runner
    program_runner = runner.ProgramRunner(
        source_code,
        args.pyscript,
        args.shscript,
        dry_run=args.only_instr,
        profiling=args.profiling,
        output_dir=output_dir,
    )

    try:
        program_output, return_code = program_runner.run()
    except Exception as e:
        print(f"An error occurred: {e}")

    if return_code != 0:
        logging.error(f"Program exited with code {return_code}, skipping analysis.")

    logger.info("Trace collection done.")
