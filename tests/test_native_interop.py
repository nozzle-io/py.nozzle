import os
import subprocess
import time
import uuid

import numpy as np
import nozzle
import pytest


def test_python_sender_native_storage():
    peer = os.environ.get("NOZZLE_NATIVE_PEER")
    if not peer:
        pytest.skip("native peer is not configured for this job")
    name = "py_native_" + uuid.uuid4().hex[:16]
    sender = nozzle.Sender.create(name, "python_native_interop")
    pixels = np.empty((3, 65, 4), dtype=np.uint8)
    for y in range(3):
        for x in range(65):
            pixels[y, x] = [190 + y, 20 + x, 40 + y, 149]
    with subprocess.Popen(
        [peer, name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
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
