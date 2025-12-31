import argparse
import datetime
import html
import json
import logging
import os
from collections import defaultdict

from tqdm import tqdm

from traincheck.invariant import CheckerResult, Invariant, read_inv_file
from traincheck.trace import MDNONEJSONEncoder, Trace, select_trace_implementation
from traincheck.utils import register_custom_excepthook

register_custom_excepthook()


def parse_checker_results(file_name: str):
    with open(file_name, "r") as f:
        lines = f.readlines()

    all_results: list[dict] = []
    current_res_str = ""
    for line in lines:
        if line.startswith("{") and current_res_str:
            all_results.append(json.loads(current_res_str))
            current_res_str = ""
        current_res_str += line

    if current_res_str:
        all_results.append(json.loads(current_res_str))
    return all_results


def check_engine(
    trace: Trace, invariants: list[Invariant], check_relation_first: bool
) -> list[CheckerResult]:
    logger = logging.getLogger(__name__)
    results = []
    for inv in tqdm(
        invariants, desc="Checking invariants", unit="invariant", leave=False
    ):
        assert (
            inv.precondition is not None
        ), "Invariant precondition is None. It should at least be 'Unconditional' or an empty list. Please check the invariant file and the inference process."
        logger.info("=====================================")
        res = inv.check(trace, check_relation_first)
        res.calc_and_set_time_precentage(trace.get_start_time(), trace.get_end_time())
        logger.info("Invariant %s on trace %s: %s", inv, trace, res)
        results.append(res)
    return results


def _format_invariant_label(invariant: Invariant) -> str:
    if invariant.text_description:
        return invariant.text_description
    params = ", ".join(str(param) for param in invariant.params)
    return f"{invariant.relation.__name__}({params})"


def _summarize_results(results: list[CheckerResult]) -> dict[str, int]:
    failed = sum(1 for res in results if not res.check_passed)
    not_triggered = sum(1 for res in results if res.triggered is False)
    passed = sum(1 for res in results if res.check_passed and res.triggered)
    total = len(results)
    triggered = total - not_triggered
    return {
        "total": total,
        "failed": failed,
        "passed": passed,
        "not_triggered": not_triggered,
        "triggered": triggered,
    }


def _relation_breakdown(results: list[CheckerResult]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"failed": 0, "passed": 0, "not_triggered": 0}
    )
    for res in results:
        relation_name = res.invariant.relation.__name__
        if not res.check_passed:
            counts[relation_name]["failed"] += 1
        elif res.triggered:
            counts[relation_name]["passed"] += 1
        else:
            counts[relation_name]["not_triggered"] += 1
    return dict(counts)


def _render_bar_segment(width_pct: float, class_name: str) -> str:
    width_pct = max(0.0, min(100.0, width_pct))
    return f'<span class="{class_name}" style="width: {width_pct:.2f}%"></span>'


def _build_html_report(report_data: dict) -> str:
    def esc(value: str) -> str:
        return html.escape(value, quote=True)

    def percent(part: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return (part / total) * 100.0

    overall = report_data["overall"]
    trace_sections = []
    for trace in report_data["traces"]:
        total = trace["summary"]["total"]
        failed = trace["summary"]["failed"]
        passed = trace["summary"]["passed"]
        not_triggered = trace["summary"]["not_triggered"]

        bar = (
            _render_bar_segment(percent(failed, total), "bar-failed")
            + _render_bar_segment(percent(passed, total), "bar-passed")
            + _render_bar_segment(percent(not_triggered, total), "bar-not-triggered")
        )

        failed_list_items = []
        for failed_item in trace["failed_invariants"][:10]:
            label = esc(failed_item["label"])
            detail = esc(failed_item["relation"])
            failed_list_items.append(
                f'<li><span class="inv-label">{label}</span>'
                f'<span class="inv-detail">{detail}</span></li>'
            )
        failed_list_html = "".join(failed_list_items) or "<li>None</li>"

        relation_rows = []
        for relation_name, rel_counts in sorted(trace["relations"].items()):
            relation_rows.append(
                "<tr>"
                f"<td>{esc(relation_name)}</td>"
                f"<td>{rel_counts['failed']}</td>"
                f"<td>{rel_counts['passed']}</td>"
                f"<td>{rel_counts['not_triggered']}</td>"
                "</tr>"
            )
        relation_table = "\n".join(relation_rows) or ""

        trace_sections.append(
            f"""
            <section class="panel">
              <div class="panel-header">
                <div>
                  <h2>{esc(trace['name'])}</h2>
                  <div class="subtle">Trace summary</div>
                </div>
                <div class="stat-inline">
                  <span>Failed</span><strong>{failed}</strong>
                  <span>Passed</span><strong>{passed}</strong>
                  <span>Not Triggered</span><strong>{not_triggered}</strong>
                </div>
              </div>
              <div class="bar">{bar}</div>
              <div class="legend">
                <span class="legend-item failed">Failed</span>
                <span class="legend-item passed">Passed</span>
                <span class="legend-item not-triggered">Not Triggered</span>
              </div>
              <div class="grid-two">
                <div>
                  <h3>Failed invariants (top 10)</h3>
                  <ul class="inv-list">{failed_list_html}</ul>
                </div>
                <div>
                  <h3>Relation breakdown</h3>
                  <table class="table">
                    <thead>
                      <tr><th>Relation</th><th>Failed</th><th>Passed</th><th>Not Triggered</th></tr>
                    </thead>
                    <tbody>
                      {relation_table}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
            """
        )

    overall_relation_rows = []
    for relation_name, rel_counts in sorted(report_data["relations"].items()):
        overall_relation_rows.append(
            "<tr>"
            f"<td>{esc(relation_name)}</td>"
            f"<td>{rel_counts['failed']}</td>"
            f"<td>{rel_counts['passed']}</td>"
            f"<td>{rel_counts['not_triggered']}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TrainCheck Report</title>
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
      --not-triggered: #f2b233;
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
    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}
    .panel h2 {{
      margin: 0 0 4px;
      font-size: 22px;
    }}
    .stat-inline {{
      display: grid;
      grid-auto-flow: column;
      gap: 8px;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }}
    .stat-inline strong {{
      color: var(--text);
      font-size: 16px;
    }}
    .bar {{
      display: flex;
      height: 14px;
      border-radius: 999px;
      overflow: hidden;
      background: #e8ecf5;
      margin: 16px 0 10px;
    }}
    .bar-failed {{ background: var(--failed); }}
    .bar-passed {{ background: var(--passed); }}
    .bar-not-triggered {{ background: var(--not-triggered); }}
    .legend {{
      display: flex;
      gap: 16px;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 18px;
    }}
    .legend-item::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 6px;
    }}
    .legend-item.failed::before {{ background: var(--failed); }}
    .legend-item.passed::before {{ background: var(--passed); }}
    .legend-item.not-triggered::before {{ background: var(--not-triggered); }}
    .grid-two {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
    }}
    .inv-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .inv-list li {{
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
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
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
        <h1>TrainCheck Report</h1>
        <div class="subtle">Generated {esc(report_data['generated_at'])}</div>
      </div>
      <div class="subtle">Output: {esc(report_data['output_dir'])}</div>
    </header>

    <div class="cards">
      <div class="card">
        <div class="label">Total Invariants</div>
        <div class="value">{overall['total_invariants']}</div>
      </div>
      <div class="card">
      <div class="label">Failed Checks</div>
      <div class="value">{overall['failed']}</div>
      </div>
      <div class="card">
      <div class="label">Passed Checks</div>
      <div class="value">{overall['passed']}</div>
      </div>
      <div class="card">
      <div class="label">Not Triggered</div>
      <div class="value">{overall['not_triggered']}</div>
      </div>
    </div>

    <section class="panel">
      <div class="panel-header">
        <div>
          <h2>Relation breakdown (overall)</h2>
          <div class="subtle">Grouped by invariant relation type</div>
        </div>
      </div>
      <table class="table">
        <thead>
          <tr><th>Relation</th><th>Failed</th><th>Passed</th><th>Not Triggered</th></tr>
        </thead>
        <tbody>
          {"".join(overall_relation_rows)}
        </tbody>
      </table>
    </section>

    {"".join(trace_sections)}

    <footer>Generated by TrainCheck checker.</footer>
  </div>
</body>
</html>
"""


def _write_html_report(report_data: dict, output_dir: str) -> str:
    html_report = _build_html_report(report_data)
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html_report)
    return report_path


def _log_wandb(report_data: dict, report_path: str | None, args: argparse.Namespace):
    try:
        import wandb
    except ImportError:
        logging.getLogger(__name__).warning(
            "Weights & Biases is not installed. Skipping wandb logging."
        )
        return

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        job_type="checker",
    )
    overall = report_data["overall"]
    wandb.log(
        {
            "invariants/total": overall["total_invariants"],
            "checks/total": overall["total_checks"],
            "checks/failed": overall["failed"],
            "checks/passed": overall["passed"],
            "checks/not_triggered": overall["not_triggered"],
        }
    )

    table = wandb.Table(
        columns=[
            "trace",
            "total",
            "failed",
            "passed",
            "not_triggered",
            "triggered",
        ]
    )
    for trace in report_data["traces"]:
        summary = trace["summary"]
        table.add_data(
            trace["name"],
            summary["total"],
            summary["failed"],
            summary["passed"],
            summary["not_triggered"],
            summary["triggered"],
        )
    wandb.log({"trace_summary": table})

    if report_path:
        try:
            with open(report_path, "r") as f:
                wandb.log({"checker_report": wandb.Html(f.read())})
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to attach HTML report to wandb run."
            )
    run.finish()


def _log_mlflow(report_data: dict, report_path: str | None, args: argparse.Namespace):
    try:
        import mlflow
    except ImportError:
        logging.getLogger(__name__).warning(
            "MLflow is not installed. Skipping MLflow logging."
        )
        return

    if args.mlflow_experiment:
        mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        overall = report_data["overall"]
        mlflow.log_metric("invariants_total", overall["total_invariants"])
        mlflow.log_metric("checks_total", overall["total_checks"])
        mlflow.log_metric("checks_failed", overall["failed"])
        mlflow.log_metric("checks_passed", overall["passed"])
        mlflow.log_metric("checks_not_triggered", overall["not_triggered"])
        for trace in report_data["traces"]:
            summary = trace["summary"]
            name = trace["name"].replace("/", "_")
            mlflow.log_metric(f"{name}_failed", summary["failed"])
            mlflow.log_metric(f"{name}_passed", summary["passed"])
            mlflow.log_metric(f"{name}_not_triggered", summary["not_triggered"])
            mlflow.log_metric(f"{name}_total", summary["total"])
        if report_path:
            mlflow.log_artifact(report_path)


def main():
    parser = argparse.ArgumentParser(
        description="(Offline) Invariant Checker for ML Pipelines in Python"
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
        "-b",
        "--backend",
        type=str,
        choices=["pandas", "polars", "dict"],
        default="pandas",
        help="Specify the backend to use for Trace",
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

    args = parser.parse_args()
    _, read_trace_file = select_trace_implementation(args.backend)
    # read the invariants

    # check if either traces or trace folders are provided
    if args.traces is None and args.trace_folders is None:
        # print help message if neither traces nor trace folders are provided
        parser.print_help()
        parser.error(
            "Please provide either traces or trace folders to infer invariants"
        )

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ## DEBUG
    time_now = f"{time_now}_relation_first_{args.check_relation_first}"
    # set logging to a file
    logging.basicConfig(
        filename=f"traincheck_checker_{time_now}.log",
        level=log_level,
    )

    logger = logging.getLogger(__name__)

    # log all the arguments
    logger.info("Checker started with Arguments:")
    for arg, val in vars(args).items():
        logger.info("%s: %s", arg, val)

    # create the output folder if not exists
    if not args.output_dir:
        args.output_dir = f"traincheck_checker_results_{time_now}"
    os.makedirs(args.output_dir, exist_ok=True)

    # copy the invariants to the output folder
    for inv_file in args.invariants:
        os.system(f"cp {inv_file} {args.output_dir}/invariants.json")

    logger.info("Reading invariants from %s", "\n".join(args.invariants))
    invs = read_inv_file(args.invariants)

    traces = []
    trace_parent_folders = []
    if args.traces is not None:
        logger.info("Reading traces from %s", "\n".join(args.traces))
        trace_parent_folders = [os.path.basename(os.path.commonpath(args.traces))]
        traces.append(read_trace_file(args.traces))
    if args.trace_folders is not None:
        for trace_folder in args.trace_folders:
            # file discovery
            trace_files = [
                f"{trace_folder}/{file}"
                for file in os.listdir(trace_folder)
                if file.startswith("trace_") or file.startswith("proxy_log.json")
            ]
            trace_parent_folder = os.path.basename(trace_folder)
            if trace_parent_folder in trace_parent_folders:
                logger.warning(
                    f"Found duplicate trace folder name {trace_folder}, breaking tie by adding _1 to the name"
                )
                while trace_parent_folder in trace_parent_folders:
                    trace_parent_folder += "_1"
            trace_parent_folders.append(trace_parent_folder)
            logger.info("Reading traces from %s", "\n".join(trace_files))
            traces.append(read_trace_file(trace_files))

    logger.addHandler(logging.StreamHandler())
    report_traces: list[dict] = []
    overall_relation_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"failed": 0, "passed": 0, "not_triggered": 0}
    )
    overall_summary = {
        "total_invariants": len(invs),
        "total_checks": 0,
        "failed": 0,
        "passed": 0,
        "not_triggered": 0,
        "triggered": 0,
    }

    for trace, trace_parent_folder in zip(traces, trace_parent_folders):
        results_per_trace = check_engine(trace, invs, args.check_relation_first)
        results_per_trace_failed = [
            res for res in results_per_trace if not res.check_passed
        ]
        results_per_trace_not_triggered = [
            res for res in results_per_trace if res.triggered is False
        ]

        logger.info("Checking finished. %d invariants checked", len(results_per_trace))
        logger.info(
            "Total failed invariants: %d/%d",
            len(results_per_trace_failed),
            len(results_per_trace),
        )
        logger.info(
            "Total passed invariants: %d/%d",
            len(results_per_trace) - len(results_per_trace_failed),
            len(results_per_trace),
        )
        logger.info(
            "Total invariants that are not triggered: %d/%d",
            len(results_per_trace_not_triggered),
            len(results_per_trace),
        )

        # mkdir for the trace parent folder in the output folder
        os.makedirs(os.path.join(args.output_dir, trace_parent_folder), exist_ok=True)

        # dump the results to a file
        with open(
            os.path.join(args.output_dir, trace_parent_folder, "failed.log"),
            "w",
        ) as f:
            for res in results_per_trace:
                if not res.check_passed:
                    json.dump(res.to_dict(), f, indent=4, cls=MDNONEJSONEncoder)
                    f.write("\n")

        with open(
            os.path.join(args.output_dir, trace_parent_folder, "not_triggered.log"),
            "w",
        ) as f:
            for res in results_per_trace:
                if not res.triggered:
                    json.dump(res.to_dict(), f, indent=4, cls=MDNONEJSONEncoder)
                    f.write("\n")

        with open(
            os.path.join(args.output_dir, trace_parent_folder, "passed.log"),
            "w",
        ) as f:
            for res in results_per_trace:
                if res.check_passed and res.triggered:
                    json.dump(res.to_dict(), f, indent=4, cls=MDNONEJSONEncoder)
                    f.write("\n")

        trace_summary = _summarize_results(results_per_trace)
        relation_counts = _relation_breakdown(results_per_trace)
        for relation_name, rel_counts in relation_counts.items():
            overall_relation_counts[relation_name]["failed"] += rel_counts["failed"]
            overall_relation_counts[relation_name]["passed"] += rel_counts["passed"]
            overall_relation_counts[relation_name]["not_triggered"] += rel_counts[
                "not_triggered"
            ]

        overall_summary["total_checks"] += trace_summary["total"]
        overall_summary["failed"] += trace_summary["failed"]
        overall_summary["passed"] += trace_summary["passed"]
        overall_summary["not_triggered"] += trace_summary["not_triggered"]
        overall_summary["triggered"] += trace_summary["triggered"]

        failed_invariants = [
            {
                "label": _format_invariant_label(res.invariant),
                "relation": res.invariant.relation.__name__,
            }
            for res in results_per_trace_failed
        ]

        report_traces.append(
            {
                "name": trace_parent_folder,
                "summary": trace_summary,
                "relations": relation_counts,
                "failed_invariants": failed_invariants,
            }
        )

    report_data = {
        "generated_at": time_now,
        "output_dir": args.output_dir,
        "overall": overall_summary,
        "relations": dict(overall_relation_counts),
        "traces": report_traces,
    }

    report_path = None
    if not args.no_html_report:
        report_path = _write_html_report(report_data, args.output_dir)
        logger.info("HTML report written to %s", report_path)

    if args.report_wandb:
        _log_wandb(report_data, report_path, args)

    if args.report_mlflow:
        _log_mlflow(report_data, report_path, args)


if __name__ == "__main__":
    main()
