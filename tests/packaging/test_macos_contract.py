from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_package_artifacts as verifier


class MacosContractTests(unittest.TestCase):
    def test_native_minimum_not_sdk(self):
        load = "Load command 8\n cmd LC_BUILD_VERSION\n platform 1\n minos 14.0\n sdk 15.5\n"
        self.assertEqual(verifier.macos_minimum_from_otool(load), (14, 0, 0))
        self.assertEqual(verifier.macos_minimum_from_otool(load.replace("14.0", "15.0")), (15, 0, 0))

    def test_legacy_command(self):
        self.assertEqual(verifier.macos_minimum_from_otool(
            "Load command 8\n cmd LC_VERSION_MIN_MACOSX\n version 14.0.0\n sdk 15.5\n"
        ), (14, 0, 0))

    def test_missing_duplicate_wrong_platform_and_malformed_fail(self):
        load = "Load command 8\n cmd LC_BUILD_VERSION\n platform 1\n minos 14.0\n"
        for invalid in ("", load + load, load.replace("platform 1", "platform 2"),
                        load.replace("minos 14.0", "minos unknown")):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                verifier.macos_minimum_from_otool(invalid)

    def test_runner_tag_not_accepted(self):
        with self.assertRaises(SystemExit):
            verifier.classify_platform("macos", "macosx_15_0_universal2")

    def test_each_native_slice_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "native.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("nozzle/_nozzle.so", b"test fixture")
            for minimum in ("14.0", "15.0"):
                def inspect(command, **kwargs):
                    if command[0] == "lipo":
                        output = "Architectures in the fat file: native are: x86_64 arm64"
                    else:
                        target = minimum if command[3] == "x86_64" else "14.0"
                        output = f"Load command 8\n cmd LC_BUILD_VERSION\n platform 1\n minos {target}\n sdk 15.5\n"
                    return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
                with patch.object(verifier.subprocess, "run", side_effect=inspect) as run:
                    if minimum == "14.0":
                        verifier.verify_macos_universal2_binaries(wheel, ["nozzle/_nozzle.so"])
                    else:
                        with self.assertRaises(SystemExit):
                            verifier.verify_macos_universal2_binaries(wheel, ["nozzle/_nozzle.so"])
                    self.assertEqual([call.args[0][3] for call in run.call_args_list[1:]],
                                     ["arm64", "x86_64"])

    def test_internal_metadata_cannot_be_retagged(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "nozzle_io-0.1.0-cp312-abi3-macosx_14_0_universal2.whl"
            for tag in ("cp312-abi3-macosx_14_0_universal2", "cp312-abi3-macosx_15_0_universal2"):
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("nozzle_io-0.1.0.dist-info/WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
                if "14_0" in tag:
                    verifier.verify_macos_wheel_metadata(wheel)
                else:
                    with self.assertRaises(SystemExit):
                        verifier.verify_macos_wheel_metadata(wheel)
