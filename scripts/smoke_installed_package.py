#!/usr/bin/env python3
"""Install a built py.nozzle artifact into a fresh venv and smoke test it."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

DIST_NAME = "nozzle-io"


def fail(message: str) -> None:
    print(f"smoke_error: {message}")
    raise SystemExit(1)


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def artifact_for(dist_dir: Path, kind: str) -> Path:
    pattern = "*.whl" if kind == "wheel" else "*.tar.gz"
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        fail(f"expected exactly one {kind} in {dist_dir}, found {len(matches)}")
    return matches[0].resolve()


def selected_artifact(dist_dir: Path, kind: str, artifact: Path | None) -> Path:
    if artifact is None:
        return artifact_for(dist_dir, kind)
    resolved = artifact.resolve()
    if not resolved.is_file():
        fail(f"artifact does not exist: {resolved}")
    if kind == "wheel" and resolved.suffix != ".whl":
        fail(f"expected wheel artifact, got {resolved.name}")
    if kind == "sdist" and not resolved.name.endswith(".tar.gz"):
        fail(f"expected sdist artifact, got {resolved.name}")
    print(f"selected_artifact={resolved}")
    return resolved


def smoke_code(expected_version: Optional[str]) -> str:
    expected_line = f"expected = {expected_version!r}" if expected_version else "expected = None"
    return f'''
from importlib.metadata import distribution, version
from pathlib import Path
import os
import nozzle
{expected_line}
metadata_version = version("{DIST_NAME}")
dist_name = distribution("{DIST_NAME}").metadata["Name"]
import_origin = Path(nozzle.__file__).resolve()
forbidden_root = Path(os.environ["NOZZLE_FORBIDDEN_IMPORT_ROOT"]).resolve()
print("distribution_name=" + dist_name)
print("metadata_version=" + metadata_version)
print("import_version=" + nozzle.__version__)
print("import_origin=" + str(import_origin))
assert dist_name == "{DIST_NAME}"
assert metadata_version == nozzle.__version__
assert not import_origin.is_relative_to(forbidden_root), (
    f"imported nozzle from checkout: {{import_origin}} under {{forbidden_root}}"
)
if expected is not None:
    assert metadata_version == expected
'''


def copy_installed_tests(tests_dir: Path, smoke_root: Path) -> Path:
    if not tests_dir.is_dir():
        fail(f"installed tests dir does not exist: {tests_dir}")
    destination = smoke_root / "tests"
    copied = 0
    for source in sorted(tests_dir.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        relative = source.relative_to(tests_dir)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"installed_test_file={target}")
        copied += 1
    if copied == 0:
        fail(f"no installed test files found in {tests_dir}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", default="dist", type=Path)
    parser.add_argument("--artifact", type=Path, help="Explicit wheel or sdist path to smoke from a mixed artifact tree.")
    parser.add_argument("--kind", choices=["wheel", "sdist"], required=True)
    parser.add_argument("--expected-version")
    parser.add_argument(
        "--tests-dir",
        type=Path,
        help=(
            "Deprecated alias for --installed-tests-dir. Tests are still copied "
            "into the smoke temp directory before pytest runs."
        ),
    )
    parser.add_argument(
        "--installed-tests-dir",
        type=Path,
        help="Directory containing backend-free installed-artifact tests to copy into the smoke temp directory.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.tests_dir is not None and args.installed_tests_dir is not None:
        fail("use only one of --tests-dir or --installed-tests-dir")
    if args.tests_dir is not None:
        print("smoke_warning=--tests-dir is deprecated; use --installed-tests-dir")
    tests_arg = args.installed_tests_dir if args.installed_tests_dir is not None else args.tests_dir
    dist_dir = args.dist_dir if args.dist_dir.is_absolute() else repo_root / args.dist_dir
    artifact = selected_artifact(dist_dir, args.kind, args.artifact)
    smoke_root = Path(tempfile.mkdtemp(prefix=f"py-nozzle-{args.kind}-smoke."))
    print(f"smoke_cwd={smoke_root}")
    try:
        artifact_copy = smoke_root / "artifacts" / artifact.name
        artifact_copy.parent.mkdir(parents=True)
        shutil.copy2(artifact, artifact_copy)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["NOZZLE_FORBIDDEN_IMPORT_ROOT"] = str(repo_root.resolve())
        venv_dir = smoke_root / "venv"
        run([sys.executable, "-m", "venv", str(venv_dir)], smoke_root, env)
        py = venv_python(venv_dir)
        run([str(py), "-m", "pip", "install", "--upgrade", "pip"], smoke_root, env)
        run([str(py), "-m", "pip", "install", str(artifact_copy)], smoke_root, env)
        run([str(py), "-m", "pip", "show", DIST_NAME], smoke_root, env)
        run([str(py), "-c", smoke_code(args.expected_version)], smoke_root, env)
        if tests_arg is not None:
            tests_dir = tests_arg if tests_arg.is_absolute() else repo_root / tests_arg
            installed_tests = copy_installed_tests(tests_dir, smoke_root)
            print(f"installed_tests_cwd={smoke_root}")
            run([str(py), "-m", "pip", "install", "pytest"], smoke_root, env)
            run([str(py), "-m", "pytest", str(installed_tests), "-v", "-s"], smoke_root, env)
        print(f"smoke_ok={args.kind} artifact={artifact.name} cwd={smoke_root}")
    finally:
        shutil.rmtree(smoke_root, ignore_errors=True)


if __name__ == "__main__":
    main()
