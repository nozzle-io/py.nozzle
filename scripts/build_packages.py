#!/usr/bin/env python3
"""Build py.nozzle wheel and/or sdist artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", action="store_true", help="build a wheel")
    parser.add_argument("--sdist", action="store_true", help="build an sdist")
    parser.add_argument("--clean", action="store_true", help="remove the output directory first")
    parser.add_argument("--out-dir", default="dist", type=Path)
    args = parser.parse_args()

    if not args.wheel and not args.sdist:
        args.wheel = True
        args.sdist = True

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wheel:
        run([sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir)], repo_root)
    if args.sdist:
        run([sys.executable, "-m", "build", "--sdist", "--outdir", str(out_dir)], repo_root)

    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            print(f"dist_file={path.name}")


if __name__ == "__main__":
    main()
