"""Negative metadata contracts for the installed-release validator."""

import importlib.metadata
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from tools.release.verify_installed import console_entries, verify_installed


class InstalledReleaseValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.prefix = Path(self.temporary_directory.name).resolve()
        metadata = self.prefix / "traincheck-0.2.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: traincheck\nVersion: 0.2.0\n"
        )
        self.metadata = metadata
        self.distribution = importlib.metadata.Distribution.at(metadata)

    def test_missing_console_command_is_rejected(self) -> None:
        (self.metadata / "entry_points.txt").write_text(
            "[console_scripts]\ntraincheck-infer = traincheck.infer_engine:main\n"
        )
        with self.assertRaisesRegex(ValueError, "Unexpected console command set"):
            console_entries(self.distribution)

    def test_missing_entrypoint_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected console command set"):
            console_entries(self.distribution)

    def test_duplicate_command_is_rejected(self) -> None:
        entry = importlib.metadata.EntryPoint(
            name="traincheck-infer",
            value="traincheck.infer_engine:main",
            group="console_scripts",
        )
        # Distribution providers can expose repeated entries even when the
        # filesystem metadata parser rejects duplicate INI keys itself.
        with patch.object(
            type(self.distribution),
            "entry_points",
            new_callable=PropertyMock,
            return_value=importlib.metadata.EntryPoints([entry, entry]),
        ):
            with self.assertRaisesRegex(ValueError, "Duplicate console entry point"):
                console_entries(self.distribution)

    def test_wrong_installed_version_is_rejected(self) -> None:
        with patch(
            "tools.release.verify_installed.importlib.metadata.distribution",
            return_value=self.distribution,
        ):
            with self.assertRaisesRegex(ValueError, "Expected traincheck 9.9.9"):
                verify_installed("9.9.9")

    def test_package_outside_environment_is_rejected(self) -> None:
        package = self.prefix / "traincheck"
        package.mkdir()
        (package / "__init__.py").write_text("")
        with patch(
            "tools.release.verify_installed.importlib.metadata.distribution",
            return_value=self.distribution,
        ):
            with self.assertRaisesRegex(ValueError, "outside the selected Python"):
                verify_installed("0.2.0")


if __name__ == "__main__":
    unittest.main()
