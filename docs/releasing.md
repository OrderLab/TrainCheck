# Releasing TrainCheck

Release from a reviewed commit that includes the version in `pyproject.toml`.
The Git tag must be `v` followed by that exact version. Keep tags immutable and
preserve configured commit/tag signing. Publication and merging require the
maintainer authorization applicable to the task.

## Build and inspect

Use Python 3.10 or newer and an isolated build environment with `build` and
`twine`. From a clean checkout:

```sh
python -m build --outdir /absolute/path/to/release-dist
python -m twine check --strict /absolute/path/to/release-dist/*
```

The default build creates the source distribution first, then builds the wheel
from it. Inspect both archives: include the two required
`traincheck/static_analyzer/func_level/*_func_level.log` tables and new production
modules; exclude traces, checkpoints, credentials, and local experiment outputs.
The source distribution also includes test fixtures and supporting tools.

Install the wheel into a separate Python environment, leave the source checkout,
and run the installed-release validator with that environment's Python:

```sh
python /absolute/path/to/checkout/tools/release/verify_installed.py --expected-version VERSION
python -m pip check
traincheck-collect --help
traincheck-infer --help
traincheck-check --help
traincheck-onlinecheck --help
```

The validator checks installed metadata and import location, then loads each
console entrypoint in a fresh isolated interpreter. It intentionally rejects
source/editable installs. Validate an installation rebuilt from the source
distribution too. Run relevant tests and configured checks; state the actual
Python/framework versions and any dependency/environment limitations.

Record source commit, artifact filenames, SHA-256 hashes, build/runtime versions,
commands, test results, and known limits in the agreed experiment evidence
directory. Publish exactly the inspected files. Version/metadata or included-file
changes require rebuilding; do not silently replace previously published bytes.

## GitHub and PyPI

After the reviewed release commit is ready, create a signed annotated tag and
verify it before pushing. Create the GitHub release using the version's notes and
attach the inspected wheel, source archive, and checksums.

For local PyPI uploads, configure a project-scoped token through the normal Twine
credential mechanism, such as the maintainer's private `~/.pypirc` or an OS
keyring. Never put credentials in this repository, command arguments, logs, or
chat. A private configuration uses a `[pypi]` section with
`username = __token__` and its token in `password`.

```sh
python -m twine upload --non-interactive /absolute/path/to/release-dist/*
```

Do not retry an uncertain upload blindly. Query
`https://pypi.org/pypi/traincheck/VERSION/json` and compare both filenames and
SHA-256 digests with the recorded artifacts. Confirm the GitHub tag points to the
intended commit and attached asset hashes match. A successful local build or
GitHub release alone does not establish PyPI publication.
