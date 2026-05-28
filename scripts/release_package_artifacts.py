#!/usr/bin/env python3
"""Validate and publish versioned py.nozzle GitHub Release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from verify_package_artifacts import (
    DIST_PREFIX,
    EXPECTED_ABI_TAG,
    EXPECTED_PYTHON_TAG,
    REQUIRED_SDIST_PATHS,
    archive_rel_paths,
    parse_wheel,
)

SEMVER_TAG_RE = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
PUBLISH_POLICY = "publishable-github-release"
LINUX_REJECT_POLICY = "ci-only-raw-linux"
LINUX_REJECT_REASON = "raw-linux-not-public"
EXPECTED_PLATFORM_TAGS = {
    "macos_wheel": "macosx_14_0_universal2",
    "windows_wheel": "win_amd64",
}
EXPECTED_JOBS = {
    "macos_wheel": "release-validate-macos",
    "windows_wheel": "release-validate-windows",
    "sdist": "release-validate-linux-sdist",
}
REQUIRED_CLASSES = tuple(EXPECTED_JOBS)
SCHEMA_VERSION = 1


def fail(message: str) -> None:
    print(f"release_error: {message}")
    raise SystemExit(1)


def run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def release_version(tag: str) -> str:
    match = SEMVER_TAG_RE.fullmatch(tag)
    if not match:
        fail(f"release tag must be exact vX.Y.Z semver: {tag}")
    return tag[1:]


def pyproject_version(root: Path) -> str:
    in_project = False
    for raw_line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[project]":
            in_project = True
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = False
        if in_project and line.startswith("version"):
            match = re.fullmatch(r'version\s*=\s*"([^"]+)"', line)
            if not match:
                fail("pyproject.toml project.version is malformed")
            return match.group(1)
    fail("pyproject.toml missing project.version")


def validate_tag_matches_pyproject(tag: str, root: Path) -> str:
    version = release_version(tag)
    project_version = pyproject_version(root)
    print(f"release_tag={tag}")
    print(f"pyproject_version={project_version}")
    if project_version != version:
        fail(f"tag version {version} does not match pyproject version {project_version}")
    print("release_version_match=yes")
    return version


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files(root: Path) -> list[Path]:
    files = sorted([*root.rglob("*.whl"), *root.rglob("*.tar.gz")])
    if not files:
        fail(f"no package artifacts found under {root}")
    return files


def find_by_filename(root: Path, filename: str) -> Path:
    matches = [path for path in artifact_files(root) if path.name == filename]
    if len(matches) != 1:
        fail(f"expected exactly one artifact named {filename}, found {len(matches)}")
    return matches[0]


def wheel_record(path: Path) -> dict[str, str]:
    distribution, version, python_tag, abi_tag, platform_tag = parse_wheel(path)
    return {
        "filename": path.name,
        "distribution": distribution,
        "version": version,
        "python_tag": python_tag,
        "abi_tag": abi_tag,
        "platform_tag": platform_tag,
    }


def validate_common_wheel(record: dict[str, str], expected_version: str) -> None:
    if record["distribution"] != DIST_PREFIX:
        fail(f"unexpected wheel distribution {record['distribution']}: {record['filename']}")
    if record["version"] != expected_version:
        fail(f"unexpected wheel version {record['version']}: {record['filename']}")
    if record["python_tag"] != EXPECTED_PYTHON_TAG:
        fail(f"unexpected wheel python tag {record['python_tag']}: {record['filename']}")
    if record["abi_tag"] != EXPECTED_ABI_TAG:
        fail(f"unexpected wheel ABI tag {record['abi_tag']}: {record['filename']}")


def classify_artifacts(root: Path, version: str) -> tuple[dict[str, Path], list[dict[str, str]]]:
    selected: dict[str, Path] = {}
    rejected: list[dict[str, str]] = []
    for artifact in artifact_files(root):
        if artifact.suffix == ".whl":
            record = wheel_record(artifact)
            if record["distribution"] == DIST_PREFIX and record["version"] == version:
                if record["python_tag"] == EXPECTED_PYTHON_TAG and record["abi_tag"] == EXPECTED_ABI_TAG:
                    if record["platform_tag"] == EXPECTED_PLATFORM_TAGS["macos_wheel"]:
                        add_selected(selected, "macos_wheel", artifact)
                    elif record["platform_tag"] == EXPECTED_PLATFORM_TAGS["windows_wheel"]:
                        add_selected(selected, "windows_wheel", artifact)
                    elif record["platform_tag"].startswith("linux_"):
                        item = dict(record)
                        item.update({"publish_policy": LINUX_REJECT_POLICY, "reason": LINUX_REJECT_REASON})
                        rejected.append(item)
                        print(
                            f"release_asset_rejected={artifact.name} "
                            f"publish_policy={LINUX_REJECT_POLICY} reason={LINUX_REJECT_REASON}"
                        )
                    else:
                        fail(f"unexpected wheel platform tag for release: {record['platform_tag']} in {artifact.name}")
                else:
                    fail(f"unexpected wheel tags for release: {artifact.name}")
            else:
                fail(f"unexpected wheel identity for release: {artifact.name}")
        elif artifact.name == f"{DIST_PREFIX}-{version}.tar.gz":
            add_selected(selected, "sdist", artifact)
        elif artifact.name.endswith(".tar.gz"):
            fail(f"unexpected sdist for release: {artifact.name}")
    return selected, rejected


def add_selected(selected: dict[str, Path], artifact_class: str, path: Path) -> None:
    if artifact_class in selected:
        fail(f"duplicate {artifact_class} release artifact: {selected[artifact_class].name}, {path.name}")
    selected[artifact_class] = path


def validate_selected_set(selected: dict[str, Path]) -> None:
    missing = [artifact_class for artifact_class in REQUIRED_CLASSES if artifact_class not in selected]
    if missing:
        fail("missing release artifacts: " + ", ".join(missing))


def validate_artifact_class(path: Path, artifact_class: str, version: str) -> dict[str, str]:
    if artifact_class in EXPECTED_PLATFORM_TAGS:
        record = wheel_record(path)
        validate_common_wheel(record, version)
        expected_platform = EXPECTED_PLATFORM_TAGS[artifact_class]
        if record["platform_tag"] != expected_platform:
            fail(f"expected {artifact_class} platform tag {expected_platform}, got {record['platform_tag']}")
        print(
            f"release_asset_allowed={path.name} artifact_class={artifact_class} "
            f"distribution={record['distribution']} version={record['version']} "
            f"python_tag={record['python_tag']} abi_tag={record['abi_tag']} "
            f"platform_tag={record['platform_tag']} publish_policy={PUBLISH_POLICY}"
        )
        return record
    if artifact_class == "sdist":
        expected_name = f"{DIST_PREFIX}-{version}.tar.gz"
        if path.name != expected_name:
            fail(f"expected sdist {expected_name}, got {path.name}")
        rels = set(archive_rel_paths(path))
        missing = [required for required in REQUIRED_SDIST_PATHS if required not in rels]
        if missing:
            fail(f"sdist {path.name} missing required vendored files: {', '.join(missing)}")
        print(f"release_asset_allowed={path.name} artifact_class=sdist publish_policy={PUBLISH_POLICY}")
        return {
            "filename": path.name,
            "distribution": DIST_PREFIX,
            "version": version,
            "python_tag": "",
            "abi_tag": "",
            "platform_tag": "sdist",
        }
    fail(f"unknown artifact class: {artifact_class}")


def smoke_kind(artifact_class: str) -> str:
    return "sdist" if artifact_class == "sdist" else "wheel"


def run_smoke(path: Path, artifact_class: str, version: str, tests_dir: Path | None) -> None:
    cmd = [
        sys.executable,
        "scripts/smoke_installed_package.py",
        "--kind",
        smoke_kind(artifact_class),
        "--artifact",
        str(path),
        "--expected-version",
        version,
    ]
    if tests_dir is not None:
        cmd.extend(["--installed-tests-dir", str(tests_dir)])
    run(cmd, repo_root())


def write_manifest(
    path: Path,
    job: str,
    artifact_class: str,
    artifact: Path,
    record: dict[str, str],
    version: str,
    rejected: list[dict[str, str]],
) -> None:
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job": job,
        "artifact_class": artifact_class,
        "filename": artifact.name,
        "sha256": sha256(artifact),
        "distribution": record["distribution"],
        "version": record["version"],
        "python_tag": record["python_tag"],
        "abi_tag": record["abi_tag"],
        "platform_tag": record["platform_tag"],
        "publish_policy": PUBLISH_POLICY,
        "expected_version": version,
        "smoke_result": "passed",
        "rejected_artifacts": rejected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"release_manifest={path} artifact_class={artifact_class} filename={artifact.name} sha256={manifest['sha256']}")


def validate_native(args: argparse.Namespace) -> None:
    root = repo_root()
    version = validate_tag_matches_pyproject(args.tag, root)
    artifact_root = args.artifact_root.resolve()
    selected, rejected = classify_artifacts(artifact_root, version)
    validate_selected_set(selected)
    artifact = selected[args.artifact_class]
    record = validate_artifact_class(artifact, args.artifact_class, version)
    run_smoke(artifact, args.artifact_class, version, args.installed_tests_dir)
    write_manifest(args.manifest_out, args.job, args.artifact_class, artifact, record, version, rejected)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"failed to read manifest {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"manifest root must be object: {path}")
    required = {
        "schema_version",
        "job",
        "artifact_class",
        "filename",
        "sha256",
        "distribution",
        "version",
        "python_tag",
        "abi_tag",
        "platform_tag",
        "publish_policy",
        "expected_version",
        "smoke_result",
        "rejected_artifacts",
    }
    missing = sorted(required - set(data))
    extra = sorted(set(data) - required)
    if missing:
        fail(f"manifest {path} missing fields: {', '.join(missing)}")
    if extra:
        fail(f"manifest {path} has unknown fields: {', '.join(extra)}")
    if data["schema_version"] != SCHEMA_VERSION:
        fail(f"manifest {path} has unsupported schema_version {data['schema_version']}")
    for field in required - {"schema_version", "rejected_artifacts"}:
        if not isinstance(data[field], str):
            fail(f"manifest {path} field {field} must be string")
    if not isinstance(data["rejected_artifacts"], list):
        fail(f"manifest {path} rejected_artifacts must be list")
    return data


def load_manifests(manifest_root: Path, version: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(manifest_root.rglob("*.json")):
        data = load_manifest(path)
        artifact_class = data["artifact_class"]
        if artifact_class not in REQUIRED_CLASSES:
            fail(f"manifest {path} has unknown artifact_class {artifact_class}")
        if artifact_class in result:
            fail(f"duplicate manifest for artifact_class {artifact_class}")
        if data["job"] != EXPECTED_JOBS[artifact_class]:
            fail(f"manifest {path} has unexpected job {data['job']} for {artifact_class}")
        if data["version"] != version or data["expected_version"] != version:
            fail(f"manifest {path} version mismatch for release version {version}")
        if data["distribution"] != DIST_PREFIX:
            fail(f"manifest {path} distribution mismatch: {data['distribution']}")
        if data["publish_policy"] != PUBLISH_POLICY:
            fail(f"manifest {path} publish_policy mismatch: {data['publish_policy']}")
        if data["smoke_result"] != "passed":
            fail(f"manifest {path} smoke_result is not passed")
        if artifact_class in EXPECTED_PLATFORM_TAGS:
            if data["python_tag"] != EXPECTED_PYTHON_TAG or data["abi_tag"] != EXPECTED_ABI_TAG:
                fail(f"manifest {path} wheel tag mismatch")
            if data["platform_tag"] != EXPECTED_PLATFORM_TAGS[artifact_class]:
                fail(f"manifest {path} platform_tag mismatch: {data['platform_tag']}")
        elif data["platform_tag"] != "sdist":
            fail(f"manifest {path} sdist platform_tag mismatch: {data['platform_tag']}")
        result[artifact_class] = data
    missing = [artifact_class for artifact_class in REQUIRED_CLASSES if artifact_class not in result]
    if missing:
        fail("missing validation manifests: " + ", ".join(missing))
    filenames = [data["filename"] for data in result.values()]
    if len(filenames) != len(set(filenames)):
        fail("duplicate upload asset names in manifests")
    return result


def verify_rejected_linux(manifests: dict[str, dict[str, Any]]) -> None:
    rejected: list[dict[str, Any]] = []
    for data in manifests.values():
        rejected.extend(data["rejected_artifacts"])
    linux = [item for item in rejected if isinstance(item, dict) and item.get("publish_policy") == LINUX_REJECT_POLICY]
    if not linux:
        fail("no rejected raw Linux wheel evidence found in validation manifests")
    for item in linux:
        if item.get("reason") != LINUX_REJECT_REASON:
            fail("raw Linux rejection record missing expected reason")
        filename = item.get("filename")
        print(f"release_asset_rejected={filename} publish_policy={LINUX_REJECT_POLICY} reason={LINUX_REJECT_REASON}")


def verify_manifests_against_artifacts(artifact_root: Path, manifests: dict[str, dict[str, Any]]) -> list[Path]:
    upload_paths: list[Path] = []
    for artifact_class in REQUIRED_CLASSES:
        data = manifests[artifact_class]
        artifact = find_by_filename(artifact_root, data["filename"])
        digest = sha256(artifact)
        if digest != data["sha256"]:
            fail(f"sha256 mismatch for {artifact.name}: artifact={digest} manifest={data['sha256']}")
        print(f"release_manifest_verified={artifact.name} artifact_class={artifact_class} sha256={digest}")
        upload_paths.append(artifact)
    return upload_paths


def gh_json(args: list[str]) -> Any | None:
    completed = subprocess.run(["gh", *args], cwd=repo_root(), text=True, capture_output=True)
    if completed.returncode != 0:
        return None
    return json.loads(completed.stdout)


def preflight_release(tag: str, upload_paths: list[Path], dry_run: bool) -> bool:
    release = gh_json(["release", "view", tag, "--json", "tagName,assets"])
    exists = release is not None
    print(f"release_exists={'yes' if exists else 'no'}")
    existing_assets: set[str] = set()
    if exists:
        if release.get("tagName") != tag:
            fail(f"existing release tag mismatch: {release.get('tagName')} != {tag}")
        assets = release.get("assets")
        if not isinstance(assets, list):
            fail("existing release assets payload is malformed")
        for asset in assets:
            if isinstance(asset, dict) and isinstance(asset.get("name"), str):
                existing_assets.add(asset["name"])
    for path in upload_paths:
        if path.name in existing_assets:
            fail(f"release asset already exists: {path.name}")
        print(f"release_asset_preflight={path.name} status=absent")
    if dry_run:
        print("upload_skipped=true")
    return exists


def publish(args: argparse.Namespace) -> None:
    root = repo_root()
    version = validate_tag_matches_pyproject(args.tag, root)
    print(f"release_dry_run={'true' if args.dry_run else 'false'}")
    artifact_root = args.artifact_root.resolve()
    selected, _ = classify_artifacts(artifact_root, version)
    validate_selected_set(selected)
    manifests = load_manifests(args.manifest_root.resolve(), version)
    verify_rejected_linux(manifests)
    upload_paths = verify_manifests_against_artifacts(artifact_root, manifests)
    exists = preflight_release(args.tag, upload_paths, args.dry_run)
    if args.dry_run:
        return
    if not exists:
        run(["gh", "release", "create", args.tag, "--title", args.tag, "--notes", f"py.nozzle {args.tag}"], root)
    for path in upload_paths:
        print(f"release_asset_upload={path.name}")
        run(["gh", "release", "upload", args.tag, str(path)], root)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-native")
    validate.add_argument("--artifact-root", required=True, type=Path)
    validate.add_argument("--artifact-class", required=True, choices=REQUIRED_CLASSES)
    validate.add_argument("--tag", required=True)
    validate.add_argument("--job", required=True)
    validate.add_argument("--manifest-out", required=True, type=Path)
    validate.add_argument("--installed-tests-dir", type=Path, default=Path("tests/installed"))
    validate.set_defaults(func=validate_native)

    publish_parser = sub.add_parser("publish")
    publish_parser.add_argument("--artifact-root", required=True, type=Path)
    publish_parser.add_argument("--manifest-root", required=True, type=Path)
    publish_parser.add_argument("--tag", required=True)
    publish_parser.add_argument("--dry-run", action="store_true")
    publish_parser.set_defaults(func=publish)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
