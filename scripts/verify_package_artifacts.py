#!/usr/bin/env python3
"""Verify py.nozzle package artifact tags and sdist contents."""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

DIST_PREFIX = "nozzle_io"
EXPECTED_PYTHON_TAG = "cp312"
EXPECTED_ABI_TAG = "abi3"
REQUIRED_SDIST_PATHS = [
    "libs/nozzle/CMakeLists.txt",
    "libs/nozzle/LICENSE",
    "libs/nozzle/include/nozzle/nozzle.hpp",
    "libs/nozzle/include/nozzle/nozzle_c.h",
    "libs/nozzle/src/common/device.cpp",
    "libs/nozzle/src/c_api/nozzle_c.cpp",
    "libs/nozzle/libs/plog/include/plog/Log.h",
]
FORBIDDEN_COMPONENTS = {".git", "build", "_skbuild", "__pycache__", ".pytest_cache"}


def fail(message: str) -> None:
    print(f"artifact_error: {message}")
    raise SystemExit(1)


def wheel_files(dist_dir: Path) -> list[Path]:
    return sorted(dist_dir.glob("*.whl"))


def sdist_files(dist_dir: Path) -> list[Path]:
    return sorted(dist_dir.glob("*.tar.gz"))


def parse_wheel(path: Path) -> tuple[str, str, str, str, str]:
    if not path.name.endswith(".whl"):
        fail(f"not a wheel: {path.name}")
    stem = path.name[:-4]
    parts = stem.split("-")
    if len(parts) < 5:
        fail(f"invalid wheel filename: {path.name}")
    distribution = parts[0]
    version = parts[1]
    python_tag, abi_tag, platform_tag = parts[-3:]
    return distribution, version, python_tag, abi_tag, platform_tag


def classify_platform(platform_name: str, platform_tag: str) -> str:
    if platform_name == "linux":
        if platform_tag.startswith(("manylinux", "musllinux")):
            return "publishable-manylinux"
        if platform_tag.startswith("linux_"):
            return "ci-only-raw-linux"
        fail(f"unexpected Linux wheel platform tag: {platform_tag}")
    if platform_name == "macos":
        if platform_tag.startswith("macosx_"):
            return "publishable-github-release"
        fail(f"unexpected macOS wheel platform tag: {platform_tag}")
    if platform_name == "windows":
        if platform_tag.startswith("win_"):
            return "publishable-github-release"
        fail(f"unexpected Windows wheel platform tag: {platform_tag}")
    fail(f"unsupported platform: {platform_name}")


def verify_wheels(dist_dir: Path, platform_name: str) -> None:
    wheels = wheel_files(dist_dir)
    if not wheels:
        fail(f"no wheels found in {dist_dir}")
    for wheel in wheels:
        distribution, version, python_tag, abi_tag, platform_tag = parse_wheel(wheel)
        if distribution != DIST_PREFIX:
            fail(f"unexpected wheel distribution {distribution}: {wheel.name}")
        if python_tag != EXPECTED_PYTHON_TAG:
            fail(f"unexpected wheel python tag {python_tag}: {wheel.name}")
        if abi_tag != EXPECTED_ABI_TAG:
            fail(f"expected {EXPECTED_ABI_TAG} ABI tag, got {abi_tag}: {wheel.name}")
        policy = classify_platform(platform_name, platform_tag)
        print(
            "wheel=" + wheel.name
            + f" distribution={distribution} version={version} python_tag={python_tag}"
            + f" abi_tag={abi_tag} platform_tag={platform_tag} publish_policy={policy}"
        )


def archive_rel_paths(path: Path) -> list[str]:
    rels: set[str] = set()
    with tarfile.open(path, "r:gz") as tf:
        for member in tf.getmembers():
            parts = Path(member.name).parts
            if any(part in FORBIDDEN_COMPONENTS for part in parts):
                fail(f"forbidden sdist member {member.name}")
            if len(parts) > 1:
                rels.add("/".join(parts[1:]))
    return sorted(rels)


def verify_sdist(dist_dir: Path) -> None:
    sdists = sdist_files(dist_dir)
    if not sdists:
        fail(f"no sdists found in {dist_dir}")
    for sdist in sdists:
        if not sdist.name.startswith(f"{DIST_PREFIX}-"):
            fail(f"unexpected sdist distribution name: {sdist.name}")
        rels = archive_rel_paths(sdist)
        rel_set = set(rels)
        missing = [required for required in REQUIRED_SDIST_PATHS if required not in rel_set]
        if missing:
            fail(f"sdist {sdist.name} missing required vendored files: {', '.join(missing)}")
        print(f"sdist={sdist.name} vendored_nozzle=yes junk_absent=yes")
        for rel in rels:
            print(f"sdist_member={rel}")
        for required in REQUIRED_SDIST_PATHS:
            print(f"sdist_required={required}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", default="dist", type=Path)
    parser.add_argument("--platform", required=True, choices=["macos", "windows", "linux"])
    parser.add_argument("--skip-wheel", action="store_true")
    parser.add_argument("--require-sdist", action="store_true")
    args = parser.parse_args()

    dist_dir = args.dist_dir.resolve()
    if not dist_dir.is_dir():
        fail(f"dist dir does not exist: {dist_dir}")
    if not args.skip_wheel:
        verify_wheels(dist_dir, args.platform)
    if args.require_sdist:
        verify_sdist(dist_dir)


if __name__ == "__main__":
    main()
