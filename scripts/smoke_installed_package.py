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


def smoke_code(expected_version: Optional[str]) -> str:
    expected_line = f"expected = {expected_version!r}" if expected_version else "expected = None"
    return f'''
from importlib.metadata import distribution, version
import nozzle
{expected_line}
metadata_version = version("{DIST_NAME}")
dist_name = distribution("{DIST_NAME}").metadata["Name"]
print("distribution_name=" + dist_name)
print("metadata_version=" + metadata_version)
print("import_version=" + nozzle.__version__)
assert dist_name == "{DIST_NAME}"
assert metadata_version == nozzle.__version__
if expected is not None:
    assert metadata_version == expected
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", default="dist", type=Path)
    parser.add_argument("--kind", choices=["wheel", "sdist"], required=True)
    parser.add_argument("--expected-version")
    parser.add_argument("--tests-dir", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dist_dir = args.dist_dir if args.dist_dir.is_absolute() else repo_root / args.dist_dir
    artifact = artifact_for(dist_dir, args.kind)
    smoke_root = Path(tempfile.mkdtemp(prefix=f"py-nozzle-{args.kind}-smoke."))
    try:
        artifact_copy = smoke_root / "artifacts" / artifact.name
        artifact_copy.parent.mkdir(parents=True)
        shutil.copy2(artifact, artifact_copy)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        venv_dir = smoke_root / "venv"
        run([sys.executable, "-m", "venv", str(venv_dir)], smoke_root, env)
        py = venv_python(venv_dir)
        run([str(py), "-m", "pip", "install", "--upgrade", "pip"], smoke_root, env)
        run([str(py), "-m", "pip", "install", str(artifact_copy)], smoke_root, env)
        run([str(py), "-m", "pip", "show", DIST_NAME], smoke_root, env)
        run([str(py), "-c", smoke_code(args.expected_version)], smoke_root, env)
        if args.tests_dir is not None:
            tests_dir = args.tests_dir if args.tests_dir.is_absolute() else repo_root / args.tests_dir
            run([str(py), "-m", "pip", "install", "pytest"], smoke_root, env)
            run([str(py), "-m", "pytest", str(tests_dir), "-v"], smoke_root, env)
        print(f"smoke_ok={args.kind} artifact={artifact.name} cwd={smoke_root}")
    finally:
        shutil.rmtree(smoke_root, ignore_errors=True)


if __name__ == "__main__":
    main()
