"""Collection defaults and compatibility contracts for model tracker selection."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

import traincheck.collect_trace as collect_trace
import traincheck.config.config as config
import traincheck.instrumentor.proxy_wrapper.proxy_config as proxy_config
from tools.fresh_validation.check_coverage import observe
from traincheck.config.config import InstrOpt
from traincheck.instrumentor import dumper, tracer
from traincheck.instrumentor.proxy_wrapper import subclass
from traincheck.instrumentor.proxy_wrapper.proxy_registry import VarRegistry
from traincheck.invariant.base_cls import (
    APIParam,
    GroupedPreconditions,
    Invariant,
    Preconditions,
    UnconditionalPrecondition,
    VarTypeParam,
)
from traincheck.invariant.consistency_relation import ConsistencyRelation
from traincheck.invariant.contain_relation import (
    APIContainRelation,
    cache_events_scanner,
)
from traincheck.trace.trace_pandas import TracePandas
from traincheck.utils import typename


def parameter_rule() -> Invariant:
    return Invariant(
        APIContainRelation,
        [APIParam("torch.optim.SGD.step"), VarTypeParam("torch.nn.Parameter", "data")],
        GroupedPreconditions(
            {"parent_func_call_pre": Preconditions([UnconditionalPrecondition()])}
        ),
    )


def freeze_parameter(parameter: torch.nn.Parameter) -> None:
    parameter.requires_grad = False


class ModelTrackerSelectionTests(unittest.TestCase):
    def collect(
        self,
        arguments: list[str],
        *,
        yaml_options: str | None = None,
        invariants: list[Invariant] | None = None,
    ) -> tuple[dict[str, object], str, InstrOpt | None]:
        """Generate real instrumented source without running a training process."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            script = directory / "training.py"
            script.write_text(
                "import torch\nmodel = torch.nn.Linear(2, 1)\n"
                "optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n",
                encoding="utf-8",
            )
            output = directory / "trace"
            argv = [
                "traincheck-collect",
                "--pyscript",
                str(script),
                "--output-dir",
                str(output),
                "--only-instr",
                *arguments,
            ]
            if yaml_options is not None:
                configuration = directory / "traincheck.yml"
                configuration.write_text(yaml_options, encoding="utf-8")
                argv.extend(["--use-config", "--config", str(configuration)])
            if invariants is not None:
                argv.extend(["--invariants", "reference.json"])

            with (
                patch.object(sys, "argv", argv),
                patch.object(collect_trace, "dump_env"),
                patch.object(collect_trace, "read_inv_file", return_value=invariants),
                patch.object(
                    proxy_config,
                    "auto_observer_config",
                    copy.deepcopy(proxy_config.auto_observer_config),
                ),
                patch.object(
                    collect_trace.instrumentor,
                    "instrument_file",
                    wraps=collect_trace.instrumentor.instrument_file,
                ) as instrument,
                patch.object(collect_trace.runner, "ProgramRunner") as runner,
            ):
                runner.return_value.run.return_value = ("", 0)
                collect_trace.main()
                assert instrument.call_args is not None
                assert runner.call_args is not None
                options: dict[str, object] = dict(instrument.call_args.kwargs)
                source: object = runner.call_args.args[0]
                assert isinstance(source, str)
                compile(source, str(script), "exec")

            saved_path = output / "instr_opts.json"
            saved_options = (
                InstrOpt.from_json(saved_path.read_text(encoding="utf-8"))
                if saved_path.exists()
                else None
            )
            return options, source, saved_options

    def test_default_tracks_parameters_with_subclass(self) -> None:
        options, source, saved = self.collect(["--models-to-track", "model"])
        self.assertEqual(options["model_tracker_style"], "subclass")
        self.assertEqual(options["models_to_track"], ["model"])
        self.assertIn("proxy_parameter(model,", source)
        self.assertNotIn("model = Proxy(model,", source)
        self.assertIsNone(saved)

    def test_explicit_cli_modes_remain_available(self) -> None:
        statements = {
            "proxy": "model = Proxy(model,",
            "subclass": "proxy_parameter(model,",
            "sampler": "model_sampler = VarSampler(model,",
        }
        for style, statement in statements.items():
            with self.subTest(style=style):
                options, source, _ = self.collect(
                    ["--models-to-track", "model", "--model-tracker-style", style]
                )
                self.assertEqual(options["model_tracker_style"], style)
                self.assertIn(statement, source)

    def test_explicit_proxy_yaml_preserves_legacy_selection(self) -> None:
        options, source, _ = self.collect(
            [], yaml_options="models_to_track: [model]\nmodel_tracker_style: proxy\n"
        )
        self.assertEqual(options["model_tracker_style"], "proxy")
        self.assertIn("model = Proxy(model,", source)

    def test_yaml_without_style_uses_subclass(self) -> None:
        options, source, _ = self.collect([], yaml_options="models_to_track: [model]\n")
        self.assertEqual(options["model_tracker_style"], "subclass")
        self.assertIn("proxy_parameter(model,", source)

    def test_default_does_not_activate_unselected_models(self) -> None:
        options, source, _ = self.collect([])
        self.assertIsNone(options["models_to_track"])
        self.assertNotIn("proxy_parameter(model,", source)
        self.assertNotIn("model = Proxy(model,", source)
        self.assertNotIn("model_sampler = VarSampler(model,", source)

    def test_selective_parameter_rules_preserve_selected_tracker(self) -> None:
        for explicit_style in (None, "proxy", "subclass"):
            with self.subTest(explicit_style=explicit_style):
                arguments = ["--models-to-track", "model"]
                if explicit_style is not None:
                    arguments.extend(["--model-tracker-style", explicit_style])
                options, _, saved = self.collect(
                    arguments, invariants=[parameter_rule()]
                )
                expected = "subclass" if explicit_style is None else explicit_style
                self.assertEqual(options["model_tracker_style"], expected)
                assert saved is not None
                self.assertEqual(saved.model_tracker_style, expected)
                self.assertTrue(saved.disable_proxy_dumping)
                self.assertEqual(
                    saved.funcs_instr_opts["torch.optim.SGD.step"][
                        "var_types_to_track"
                    ],
                    {"torch.nn.Parameter": {"dump_unchanged": False}},
                )

    def test_sampler_cannot_replace_immediate_variable_tracking(self) -> None:
        self.assertEqual(
            collect_trace.get_model_tracker_instr_opts([parameter_rule()], "sampler"),
            "subclass",
        )

    def test_consistency_only_rules_keep_sampler_policy(self) -> None:
        invariant = Invariant(
            ConsistencyRelation,
            [
                VarTypeParam("torch.nn.Parameter", "data"),
                VarTypeParam("torch.nn.Parameter", "grad"),
            ],
            None,
        )
        for style in ("subclass", "proxy", "sampler"):
            with self.subTest(style=style):
                self.assertEqual(
                    collect_trace.get_model_tracker_instr_opts([invariant], style),
                    "sampler",
                )

    def test_selective_api_only_rules_disable_model_tracking(self) -> None:
        invariant = Invariant(
            APIContainRelation,
            [APIParam("torch.optim.SGD.step"), APIParam("torch.add")],
            GroupedPreconditions(
                {"parent_func_call_pre": Preconditions([UnconditionalPrecondition()])}
            ),
        )
        options, source, saved = self.collect(
            ["--models-to-track", "model"], invariants=[invariant]
        )
        self.assertIsNone(options["model_tracker_style"])
        self.assertIsNone(options["models_to_track"])
        self.assertNotIn("proxy_parameter(model,", source)
        assert saved is not None
        self.assertIsNone(saved.model_tracker_style)

    def test_saved_instrumentation_styles_round_trip_without_migration(self) -> None:
        for style in ("proxy", "subclass", "sampler", None):
            with self.subTest(style=style):
                original = InstrOpt({}, style, True)
                restored = InstrOpt.from_json(original.to_json())
                self.assertEqual(restored.model_tracker_style, style)
                self.assertEqual(restored.to_json(), original.to_json())

    def test_saved_instrumentation_style_remains_required(self) -> None:
        incomplete = {"funcs_instr_opts": {}, "disable_proxy_dumping": True}
        with self.assertRaises(KeyError):
            InstrOpt.from_json(json.dumps(incomplete))

    def test_subclass_parameter_rule_activates_and_checks_real_transition(self) -> None:
        records: list[dict[str, object]] = []

        def capture(record: dict[str, object]) -> None:
            # Retain production timestamping, metadata conversion, and JSON
            # serialization; substitute only the final file/buffer destination.
            payload: object = json.loads(dumper.serialize(record))
            assert isinstance(payload, dict)
            validated: dict[str, object] = {}
            for key, value in payload.items():
                assert isinstance(key, str)
                validated[key] = value
            records.append(validated)

        destination = SimpleNamespace(add_trace=capture)
        with (
            patch.object(subclass, "get_global_registry", return_value=VarRegistry()),
            patch.object(
                dumper,
                "THREAD_DATA",
                SimpleNamespace(
                    api_trace_buffer=destination, var_trace_buffer=destination
                ),
            ),
            patch.object(config, "DISABLE_WRAPPER", False),
            patch.object(config, "INSTR_OPTS", None),
        ):
            model = torch.nn.Module()
            model.register_parameter("weight", torch.nn.Parameter(torch.ones(2)))
            subclass.proxy_parameter(model, parent_name="model")
            wrapped = tracer.wrapper(
                freeze_parameter,
                is_bound_method=False,
                scan_proxy_in_args=True,
                dump_stack_trace=False,
                dump_args=False,
                dump_ret=False,
                handle_proxy=False,
            )
            wrapped(model.weight)

        for expected_post, should_pass in ((False, True), (True, False)):
            with self.subTest(expected_post=expected_post):
                invariant = Invariant(
                    APIContainRelation,
                    [
                        APIParam(typename(freeze_parameter)),
                        VarTypeParam(
                            "torch.nn.Parameter",
                            "requires_grad",
                            pre_value=True,
                            post_value=expected_post,
                        ),
                    ],
                    GroupedPreconditions(
                        {
                            "parent_func_call_pre": Preconditions(
                                [UnconditionalPrecondition()]
                            )
                        }
                    ),
                    "freezing a parameter changes requires_grad",
                )
                cache_events_scanner.clear()
                self.addCleanup(cache_events_scanner.clear)
                report = observe(
                    TracePandas(
                        pd.json_normalize(records), truncate_incomplete_func_calls=False
                    ),
                    invariant,
                )
                self.assertEqual(report["status"], "completed", report)
                self.assertTrue(report["active"])
                self.assertFalse(report["dormant"])
                self.assertTrue(report["fully_validated_active"])
                self.assertEqual(report["decisions_completed"], 1)
                self.assertEqual(report["check_passed"], should_pass)


if __name__ == "__main__":
    unittest.main()
