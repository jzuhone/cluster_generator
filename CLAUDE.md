# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`cluster_generator` is a Python scientific library for generating equilibrium galaxy cluster models (profiles, particles, initial conditions) for (M)HD simulations. Single package, installed with pip.

## Build

- Install editable: `pip install -e .` — `setup.py` (not `pyproject.toml`) defines dependencies, packages, and the Cython `ext_modules`. `pyproject.toml` only holds build-system, ruff, pytest, and setuptools_scm config.
- The package contains **compiled Cython extensions** in `cluster_generator/opt/` (`cython_utils.pyx`, `structures.pyx`). After editing any `.pyx` file you must recompile: run `./clean.sh` (removes `*.so`, `*.pyc`, `build/`, `dist/`, `egg-info`) then `pip install -e .`. A stale `.so` is silently used otherwise — see the `/rebuild` skill.

## Testing

Tests live in `cluster_generator/tests/`. `conftest.py` defines custom pytest options: `--answer_dir`, `--answer_store`, `--tmp`.

- The suite uses **answer testing**: generated models/particles are compared field-by-field against HDF5 files in `answers/`. `--answer_dir` is required (a fixture calls `os.path.abspath` on it).
- Default local run (compare against stored answers — regression check):
  `pytest cluster_generator --answer_dir=./answers`
- Regenerate/overwrite stored answers (this is what CI does — do NOT use it to "check" changes): add `--answer_store`.
- Doctests run alongside unit tests via `--doctest-modules`.
- `conftest.py` enforces test ordering: `test_model.py::test_model_build` runs first to build a shared `base_model.h5` reused by later tests; doctests run last.
- Markers: `slow`, `highmem`. Deselect with `-m "not slow"`.

## Lint / format

Tooling is **ruff** (it replaces black/isort/docformatter — ignore the README badges that say otherwise). isort rules run via ruff's `I` selector.

- Run everything: `pre-commit run --all-files`
- Or directly: `ruff format .` and `ruff check --fix .`
- Line length is **110**. Docstrings are **NumPy style** (numpydoc).

## Gotchas

- Config is loaded **at import time** from `cluster_generator/bin/config.yaml` into `cgparams` (in `utils.py`); a missing or corrupt file raises on import. The YAML uses custom tags `!unyt` and `!lambda` (the latter `eval`s strings).
- `numpy` is hard-pinned to `>=2.0,<3.0`.
- Generated `*.c`/`*.so` are gitignored artifacts, not source.

## Git workflow

Create a feature branch and open a PR for changes rather than committing to `master`.
