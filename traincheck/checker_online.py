import argparse
import datetime
import html
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict

from traincheck.config import config
from traincheck.invariant import read_inv_file
from traincheck.invariant.base_cls import APIParam, Invariant, Param, VarTypeParam
from traincheck.onlinechecker.streamhandler_filesystem import run_stream_monitor
from traincheck.onlinechecker.utils import Checker_data
from traincheck.trace import MDNONEJSONEncoder
from traincheck.trace.types import VarInstId

OBSERVER = None
KILLING_PROCESS = (
    False  # True indicates that SIGTERM has been sent to the running process
)
NUM_VIOLATIONS = 0
FAILED_INV: dict[Invariant, int] = {}
TOTAL_INVARIANTS = 0
RELATION_TOTALS: dict[str, int] = {}
REPORT_CONFIG: dict[str, object] = {}
LAST_REPORT_TS = 0.0
LAST_REPORT_STATE = (-1, -1)
WANDB_RUN = None
MLFLOW_ACTIVE = False

ORIGINAL_SIGINT_HANDLER = signal.getsignal(signal.SIGINT)
ORIGINAL_SIGTERM_HANDLER = signal.getsignal(signal.SIGTERM)


def handle_SIGINT(signum, frame):
    global KILLING_PROCESS

    print("Received SIGINT")
    if KILLING_PROCESS:
        exit(130)
        return
    KILLING_PROCESS = True
    try:
        stop_checker()
    except Exception as e:
        print(f"Error when stopping checker: {e}")
    # if callable(ORIGINAL_SIGINT_HANDLER):
    #     ORIGINAL_SIGINT_HANDLER(signum, frame)
    exit(130)


def handle_SIGTERM(signum, frame):
    global KILLING_PROCESS

    print("Received SIGTERM")
    if KILLING_PROCESS:
        exit(143)
        return
    KILLING_PROCESS = True
    try:
        stop_checker()
    except Exception as e:
        print(f"Error when stopping checker: {e}")
    if callable(ORIGINAL_SIGTERM_HANDLER):
        ORIGINAL_SIGTERM_HANDLER(signum, frame)
    else:
        exit(143)


curr_excepthook = sys.excepthook


def kill_running_process_on_except(typ, value, tb):
    stop_checker()
    curr_excepthook(typ, value, tb)


def register_hook_closing_program():
    signal.signal(signal.SIGTERM, handle_SIGTERM)
    signal.signal(signal.SIGINT, handle_SIGINT)
    sys.excepthook = kill_running_process_on_except


def sort_inv_file(invariants):
    """Sort the invariants by their parameters. Also collect the needed data for online checking.
    Return:
        param_to_invs: dict[Param, list[Invariant]]
        vartype_to_invs: dict[str, dict[str, list[Invariant]]]
        needed_data: (set[str], set[str], set[str])
    """
    logger = logging.getLogger(__name__)
    logger.info("Reading invariants from file: %s", invariants)

    invs = read_inv_file(invariants)
    logger.info("Total %d invariants read from file: %s", len(invs), invariants)
    logger.info("Sorting invariants by parameters")

    param_to_invs: dict[Param, list[Invariant]] = {}
    vartype_to_invs: dict[str, dict[str, list[Invariant]]] = {}
    needed_vars = set()
    needed_apis = set()
    _get_api_args_map_to_check = set()
    for inv in invs:
        assert (
            inv.precondition is not None
        ), "Invariant precondition is None. It should at least be 'Unconditional' or an empty list. Please check the invariant file and the inference process."
        params = inv._get_identifying_params()
        needed_var, needed_api, needed_args_api = (
            inv._get_information_sources_to_check()
        )
        if needed_var is not None:
            needed_vars.update(needed_var)
        if needed_api is not None:
            needed_apis.update(needed_api)
        if needed_args_api is not None:
            _get_api_args_map_to_check.update(needed_args_api)
        for param in params:
            if isinstance(param, VarTypeParam):
                if param.var_type not in vartype_to_invs:
                    vartype_to_invs[param.var_type] = {}
                if param.attr_name not in vartype_to_invs[param.var_type]:
                    vartype_to_invs[param.var_type][param.attr_name] = []
                vartype_to_invs[param.var_type][param.attr_name].append(inv)
            else:
                if param not in param_to_invs:
                    param_to_invs[param] = []
                param_to_invs[param].append(inv)
    logger.info("Sorting done.")
    needed_data = (needed_vars, needed_apis, _get_api_args_map_to_check)
    return invs, param_to_invs, vartype_to_invs, needed_data


def _format_invariant_label(invariant: Invariant) -> str:
    if invariant.text_description:
        return invariant.text_description
    params = ", ".join(str(param) for param in invariant.params)
    return f"{invariant.relation.__name__}({params})"


def _build_online_html_report(report_data: dict) -> str:
    def esc(value: str) -> str:
        return html.escape(value, quote=True)

    relation_rows = []
    for relation_name in sorted(report_data["relation_totals"].keys()):
        total = report_data["relation_totals"][relation_name]
        violated = report_data["relation_violations"].get(relation_name, 0)
        relation_rows.append(
            "<tr>"
            f"<td>{esc(relation_name)}</td>"
            f"<td>{total}</td>"
            f"<td>{violated}</td>"
            "</tr>"
        )

    violated_items = []
    for entry in report_data["top_violations"]:
        violated_items.append(
            "<li>"
            f"<span class=\"inv-label\">{esc(entry['label'])}</span>"
            f"<span class=\"inv-detail\">{esc(entry['relation'])}</span>"
            f"<span class=\"inv-count\">{entry['count']}</span>"
            "</li>"
        )
    violated_list = "".join(violated_items) or "<li>None</li>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TrainCheck Online Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6fb;
      --panel: #ffffff;
      --text: #182033;
      --muted: #6c778c;
      --accent: #2f6fed;
      --failed: #e24c4b;
      --passed: #2fb679;
      --border: #e2e6f0;
    }}
    body {{
      margin: 0;
      font-family: "Satoshi", "Avenir Next", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #fefefe, #f4f6fb 45%, #edf0f7);
      color: var(--text);
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 24px 60px;
    }}
    header {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 28px;
    }}
    header h1 {{
      margin: 0;
      font-size: 32px;
      letter-spacing: -0.02em;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 14px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin: 18px 0 26px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 10px 30px rgba(24, 32, 51, 0.08);
    }}
    .card .label {{
      color: var(--muted);
      font-size: 13px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .card .value {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .panel {{
      background: var(--panel);
      border-radius: 16px;
      padding: 24px;
      border: 1px solid var(--border);
      box-shadow: 0 15px 30px rgba(24, 32, 51, 0.08);
      margin-bottom: 22px;
    }}
    .panel h2 {{
      margin: 0 0 4px;
      font-size: 22px;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 12px;
    }}
    .table th, .table td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid var(--border);
    }}
    .table th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .inv-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .inv-list li {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 8px;
      background: #fbfcff;
    }}
    .inv-label {{
      display: block;
      font-weight: 600;
    }}
    .inv-detail {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
    }}
    .inv-count {{
      font-size: 16px;
      font-weight: 700;
      color: var(--failed);
    }}
    footer {{
      margin-top: 28px;
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>TrainCheck Online Report</h1>
        <div class="subtle">Generated {esc(report_data['generated_at'])}</div>
      </div>
      <div class="subtle">Output: {esc(report_data['output_dir'])}</div>
    </header>

    <div class="cards">
      <div class="card">
        <div class="label">Total Invariants</div>
        <div class="value">{report_data['total_invariants']}</div>
      </div>
      <div class="card">
        <div class="label">Violations</div>
        <div class="value">{report_data['total_violations']}</div>
      </div>
      <div class="card">
        <div class="label">Violated Invariants</div>
        <div class="value">{report_data['violated_invariants']}</div>
      </div>
    </div>

    <section class="panel">
      <h2>Top Violated Invariants</h2>
      <ul class="inv-list">{violated_list}</ul>
    </section>

    <section class="panel">
      <h2>Relation Breakdown</h2>
      <table class="table">
        <thead>
          <tr><th>Relation</th><th>Total</th><th>Violated</th></tr>
        </thead>
        <tbody>
          {''.join(relation_rows)}
        </tbody>
      </table>
    </section>

    <footer>Generated by TrainCheck online checker.</footer>
  </div>
</body>
</html>
"""


def _build_report_data() -> dict:
    relation_violations: dict[str, int] = defaultdict(int)
    for inv in FAILED_INV:
        relation_violations[inv.relation.__name__] += 1

    top_pairs = sorted(
        ((count, inv) for inv, count in FAILED_INV.items()),
        key=lambda item: item[0],
        reverse=True,
    )[:10]
    top_violations = [
        {
            "label": _format_invariant_label(inv),
            "relation": inv.relation.__name__,
            "count": count,
        }
        for count, inv in top_pairs
    ]

    return {
        "generated_at": REPORT_CONFIG.get("generated_at", ""),
        "output_dir": REPORT_CONFIG.get("output_dir", ""),
        "total_invariants": TOTAL_INVARIANTS,
        "total_violations": NUM_VIOLATIONS,
        "violated_invariants": len(FAILED_INV),
        "relation_totals": dict(RELATION_TOTALS),
        "relation_violations": dict(relation_violations),
        "top_violations": top_violations,
    }


def _write_html_report(report_data: dict, output_dir: str) -> str:
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(_build_online_html_report(report_data))
    return report_path


def _maybe_emit_report(force: bool = False):
    global LAST_REPORT_TS
    global LAST_REPORT_STATE

    output_dir = REPORT_CONFIG.get("output_dir")
    if not isinstance(output_dir, str):
        return

    report_state = (NUM_VIOLATIONS, len(FAILED_INV))
    now = time.monotonic()
    interval_value = REPORT_CONFIG.get("report_interval_seconds", 0.0)
    interval = (
        float(interval_value) if isinstance(interval_value, (int, float)) else 0.0
    )

    if not force:
        if report_state == LAST_REPORT_STATE:
            if interval <= 0 or now - LAST_REPORT_TS < interval:
                return
        else:
            # state changed; update immediately
            pass

    report_data = _build_report_data()
    report_path = None
    if not REPORT_CONFIG.get("no_html_report", False):
        report_path = _write_html_report(report_data, output_dir)

    report_args = REPORT_CONFIG.get("args")
    if REPORT_CONFIG.get("report_wandb", False) and isinstance(
        report_args, argparse.Namespace
    ):
        _log_wandb(report_data, report_path, report_args)

    if REPORT_CONFIG.get("report_mlflow", False) and isinstance(
        report_args, argparse.Namespace
    ):
        _log_mlflow(report_data, report_path, report_args)

    LAST_REPORT_TS = now
    LAST_REPORT_STATE = report_state


def _log_wandb(report_data: dict, report_path: str | None, args: argparse.Namespace):
    global WANDB_RUN
    try:
        import wandb
    except ImportError:
        logging.getLogger(__name__).warning(
            "Weights & Biases is not installed. Skipping wandb logging."
        )
        return

    if WANDB_RUN is None:
        WANDB_RUN = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=args.wandb_tags,
            job_type="online_checker",
        )

    wandb.log(
        {
            "invariants/total": report_data["total_invariants"],
            "invariants/violated_unique": report_data["violated_invariants"],
            "violations/total": report_data["total_violations"],
        }
    )

    table = wandb.Table(columns=["relation", "total", "violated"])
    for relation_name, total in report_data["relation_totals"].items():
        table.add_data(
            relation_name,
            total,
            report_data["relation_violations"].get(relation_name, 0),
        )
    wandb.log({"relation_breakdown": table})

    if report_path:
        try:
            with open(report_path, "r") as f:
                wandb.log({"checker_report": wandb.Html(f.read())})
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to attach HTML report to wandb run."
            )


def _finish_wandb():
    global WANDB_RUN
    if WANDB_RUN is not None:
        WANDB_RUN.finish()
        WANDB_RUN = None


def _log_mlflow(report_data: dict, report_path: str | None, args: argparse.Namespace):
    global MLFLOW_ACTIVE
    try:
        import mlflow
    except ImportError:
        logging.getLogger(__name__).warning(
            "MLflow is not installed. Skipping MLflow logging."
        )
        return

    if args.mlflow_experiment:
        mlflow.set_experiment(args.mlflow_experiment)

    if not MLFLOW_ACTIVE:
        mlflow.start_run(run_name=args.mlflow_run_name)
        MLFLOW_ACTIVE = True

    mlflow.log_metric("invariants_total", report_data["total_invariants"])
    mlflow.log_metric("invariants_violated_unique", report_data["violated_invariants"])
    mlflow.log_metric("violations_total", report_data["total_violations"])
    if report_path:
        mlflow.log_artifact(report_path)


def _finish_mlflow():
    global MLFLOW_ACTIVE
    if MLFLOW_ACTIVE:
        try:
            import mlflow
        except ImportError:
            MLFLOW_ACTIVE = False
            return
        mlflow.end_run()
        MLFLOW_ACTIVE = False


def get_violated_pair_hash(trace_pair):
    from traincheck.invariant.base_cls import make_hashable

    h1 = hash(make_hashable(trace_pair[0]))
    h2 = hash(make_hashable(trace_pair[1]))
    return tuple(sorted((h1, h2), reverse=True))


def check(
    invariants, traces, trace_folders, output_dir: str, check_relation_first: bool
):
    global OBSERVER
    global NUM_VIOLATIONS
    global FAILED_INV
    global TOTAL_INVARIANTS
    global RELATION_TOTALS

    register_hook_closing_program()

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info("Starting online checker")

    invs, param_to_invs, vartype_to_invs, needed_data = sort_inv_file(invariants)
    TOTAL_INVARIANTS = len(invs)
    RELATION_TOTALS = defaultdict(int)
    for inv in invs:
        RELATION_TOTALS[inv.relation.__name__] += 1
    checker_data = Checker_data(needed_data)
    OBSERVER = run_stream_monitor(traces, trace_folders, checker_data)

    output_file = os.path.join(output_dir, "failed.log")
    violated_pairs = dict[Invariant, set[tuple[int, int]]]()

    _maybe_emit_report(force=True)

    while True:
        trace_record = checker_data.check_queue.get()
        if checker_data.check_queue.empty():
            logger.debug("Check queue is empty")
        if trace_record is None:
            continue

        with checker_data.cond:
            while True:
                if trace_record["time"] > checker_data.min_read_time:
                    logger.debug("Wait for the different trace file to catch up")
                    checker_data.cond.wait()
                    logger.debug("Woke up from wait")
                else:
                    break

        if "var_name" in trace_record and trace_record["var_name"] is not None:
            varid = VarInstId(
                trace_record["process_id"],
                trace_record["var_name"],
                trace_record["var_type"],
            )
            if varid.var_type in vartype_to_invs:
                for attr_name, invs in vartype_to_invs[varid.var_type].items():
                    attr_name = config.VAR_ATTR_PREFIX + attr_name
                    if (
                        attr_name in trace_record
                        and trace_record[attr_name] is not None
                    ):
                        for inv in invs:
                            try:
                                result = inv.online_check(
                                    trace_record, checker_data, check_relation_first
                                )
                                if not result.check_passed:
                                    violated_pair = get_violated_pair_hash(result.trace)
                                    if inv not in violated_pairs:
                                        violated_pairs[inv] = set()
                                    if violated_pair not in violated_pairs[inv]:
                                        violated_pairs[inv].add(violated_pair)
                                    else:
                                        continue
                                    if inv not in FAILED_INV:
                                        FAILED_INV[inv] = 0
                                    FAILED_INV[inv] += 1
                                    NUM_VIOLATIONS += 1
                                    result.set_id_and_detection_time(
                                        NUM_VIOLATIONS, time.monotonic_ns()
                                    )
                                    logger.error(
                                        f"Violated id {NUM_VIOLATIONS}:\nInvariant {inv} violated near time {trace_record['time']}"
                                    )
                                    with open(output_file, "a") as f:
                                        json.dump(
                                            result.to_dict(),
                                            f,
                                            indent=4,
                                            cls=MDNONEJSONEncoder,
                                        )
                                        f.write("\n")
                                    _maybe_emit_report()
                            except Exception as e:
                                logger.error(
                                    f"Error when checking invariant {inv.text_description} with trace {trace_record}: {e}"
                                )

        elif (
            "func_call_id" in trace_record and trace_record["func_call_id"] is not None
        ):
            apiparam = APIParam(trace_record["function"])
            if apiparam in param_to_invs:
                for inv in param_to_invs[apiparam]:
                    try:
                        result = inv.online_check(
                            trace_record, checker_data, check_relation_first
                        )
                        if not result.check_passed:
                            if inv not in FAILED_INV:
                                FAILED_INV[inv] = 0
                            FAILED_INV[inv] += 1
                            NUM_VIOLATIONS += 1
                            result.set_id_and_detection_time(
                                NUM_VIOLATIONS, time.monotonic_ns()
                            )
                            logger.error(
                                f"Violated id {NUM_VIOLATIONS}:\nInvariant {inv} violated near time {trace_record['time']}"
                            )
                            with open(output_file, "a") as f:
                                json.dump(
                                    result.to_dict(), f, indent=4, cls=MDNONEJSONEncoder
                                )
                                f.write("\n")
                            _maybe_emit_report()
                    except Exception as e:
                        logger.error(
                            f"Error when checking invariant {inv.text_description} with trace {trace_record}: {e}"
                        )

        _maybe_emit_report()


def stop_checker():
    global OBSERVER
    if OBSERVER is None:
        return

    OBSERVER.stop()
    OBSERVER.join()

    global NUM_VIOLATIONS
    global FAILED_INV

    logger = logging.getLogger(__name__)
    logger.info("Checker stopped")
    logger.info(f"Total {NUM_VIOLATIONS} violations found")
    logger.info(f"Total {len(FAILED_INV)} invariants violated:")
    # for inv, count in failed_inv.items():
    #     logger.info(f"Invariant {inv} violated {count} times")
    logger.info("Violations are stored")

    _maybe_emit_report(force=True)
    _finish_wandb()
    _finish_mlflow()


def main():
    parser = argparse.ArgumentParser(
        description="(Online) Invariant Checker for ML Pipelines in Python"
    )
    parser.add_argument(
        "-t",
        "--traces",
        nargs="+",
        required=False,
        help="Traces files to infer invariants on",
    )
    parser.add_argument(
        "-f",
        "--trace-folders",
        nargs="+",
        help='Folders containing traces files to infer invariants on. Trace files should start with "trace_" or "proxy_log.json"',
    )
    parser.add_argument(
        "-i",
        "--invariants",
        nargs="+",
        required=True,
        help="Invariants files to check on traces",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--check-relation-first",
        action="store_true",
        help="""Check the relation first, otherwise, the precondition will be checked first. 
            Enabling this flag will make the checker slower, but enables the checker to catch 
            the cases where the invariant still holds even if the precondition is not satisfied, 
            which opens opportunity for precondition refinement. Note that the precondition 
            refinement algorithm is not implemented yet.""",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output folder to store the results, defaulted to traincheck_checker_results_{timestamp}/",
    )
    parser.add_argument(
        "--no-html-report",
        action="store_true",
        help="Disable generating the standalone HTML report.",
    )
    parser.add_argument(
        "--report-wandb",
        action="store_true",
        help="Log checker summary and HTML report to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team/user).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Weights & Biases group name.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Weights & Biases tags.",
    )
    parser.add_argument(
        "--report-mlflow",
        action="store_true",
        help="Log checker summary and HTML report to MLflow.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="MLflow run name.",
    )
    parser.add_argument(
        "--report-interval-seconds",
        type=float,
        default=10.0,
        help="How often to refresh the online report when no new violations are found.",
    )

    args = parser.parse_args()

    # check if either traces or trace folders are provided
    if args.traces is None and args.trace_folders is None:
        # print help message if neither traces nor trace folders are provided
        parser.print_help()
        parser.error(
            "Please provide either traces or trace folders to infer invariants"
        )

    if args.invariants is None:
        parser.print_help()
        parser.error("Please provide exactly one invariant file to check")

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ## DEBUG
    time_now = f"{time_now}_relation_first_{args.check_relation_first}"
    # set logging to a file
    logging.basicConfig(
        filename=f"traincheck_onlinechecker_{time_now}.log",
        level=log_level,
    )

    logger = logging.getLogger(__name__)
    # log all the arguments
    logger.info("Checker started with Arguments:")
    for arg, val in vars(args).items():
        logger.info("%s: %s", arg, val)

    if not args.output_dir:
        args.output_dir = f"traincheck_onlinechecker_results_{time_now}"
    os.makedirs(args.output_dir, exist_ok=True)

    # copy the invariants to the output folder
    for inv_file in args.invariants:
        os.system(f"cp {inv_file} {args.output_dir}/invariants.json")

    REPORT_CONFIG.update(
        {
            "output_dir": args.output_dir,
            "generated_at": time_now,
            "no_html_report": args.no_html_report,
            "report_wandb": args.report_wandb,
            "report_mlflow": args.report_mlflow,
            "report_interval_seconds": args.report_interval_seconds,
            "args": args,
        }
    )

    check(
        args.invariants,
        args.traces,
        args.trace_folders,
        args.output_dir,
        args.check_relation_first,
    )


if __name__ == "__main__":
    main()
