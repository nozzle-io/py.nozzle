import os
import subprocess
import time
import uuid

import numpy as np
import nozzle
import pytest


@pytest.mark.parametrize("layout", ["rgba", "rgba_strided_readonly", "rgb"])
def test_python_sender_native_storage(layout):
    peer = os.environ.get("NOZZLE_NATIVE_PEER")
    if not peer:
        pytest.skip("native peer is not configured for this job")
    name = "py_native_" + uuid.uuid4().hex[:16]
    sender = nozzle.Sender.create(name, "python_native_interop")
    pixels = np.empty((3, 65, 4), dtype=np.uint8)
    for y in range(3):
        for x in range(65):
            pixels[y, x] = [190 + y, 20 + x, 40 + y, 149]
    command = [peer, name]
    if layout == "rgba_strided_readonly":
        backing = np.empty((3, 130, 4), dtype=np.uint8)
        backing[:, ::2, :] = pixels
        pixels = backing[:, ::2, :]
        pixels.flags.writeable = False
    elif layout == "rgb":
        pixels = pixels[:, :, :3].copy()
        command = [peer, "--receive-rgb", name]
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        try:
            deadline = time.monotonic() + 8
            while process.poll() is None and time.monotonic() < deadline:
                sender.publish_array(pixels)
                time.sleep(0.02)
            output, _ = process.communicate(timeout=2)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    print(output)
    assert process.returncode == 0, output
    assert "native_storage_verified=195_pixels" in output


@pytest.mark.parametrize("access", [
    "get_array", "frame_dlpack_direct", "lock_pixels", "locked_dlpack_direct",
])
def test_native_sender_python_receiver(access):
    peer = os.environ.get("NOZZLE_NATIVE_PEER")
    if not peer:
        pytest.skip("native peer is not configured for this job")
    name = "native_py_" + uuid.uuid4().hex[:16]
    with subprocess.Popen(
        [peer, "--send", name], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        try:
            receiver = None
            frame = None
            last_error = None
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    if receiver is None:
                        receiver = nozzle.Receiver.create(name, "native_python_interop")
                    candidate = receiver.acquire_frame(timeout_ms=100)
                    if candidate.valid():
                        frame = candidate
                        break
                except RuntimeError as error:
                    last_error = str(error)
                time.sleep(0.01)
            if frame is None:
                raise AssertionError(f"native sender frame missing: {last_error}")
            info = frame.info()
            storage_format = info.format
            assert info.semantic_format == nozzle.TextureFormat.RGBA8_UNORM
            assert (info.width, info.height) == (65, 3)
            assert storage_format in (
                nozzle.TextureFormat.RGBA8_UNORM, nozzle.TextureFormat.BGRA8_UNORM,
            )
            if access == "get_array":
                actual = frame.get_array()
                frame.release()
            elif access == "frame_dlpack_direct":
                # Existing direct-call behavior; not a standard DLPack consumer test.
                actual = np.asarray(frame.__dlpack__()).copy()
                frame.release()
            else:
                locked = frame.lock_pixels()
                assert locked.format == storage_format
                view = (locked.to_ndarray() if access == "lock_pixels"
                        else locked.__dlpack__())
                actual = np.asarray(view).copy()
                del view, locked
            output, _ = process.communicate(input="done\n", timeout=3)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    print(output)
    print(f"python_access={access} storage={storage_format} first_pixel={actual[0, 0].tolist()}")
    assert process.returncode == 0, output
    assert "native_sender_published=195_pixels" in output
    expected = np.empty((3, 65, 4), dtype=np.uint8)
    for y in range(3):
        for x in range(65):
            expected[y, x] = [190 + y, 20 + x, 40 + y, 149]
    if access in ("lock_pixels", "locked_dlpack_direct"):
        if storage_format == nozzle.TextureFormat.BGRA8_UNORM:
            expected = expected[:, :, [2, 1, 0, 3]]
    np.testing.assert_array_equal(actual, expected)
