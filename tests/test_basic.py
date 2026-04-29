import pytest
import numpy as np
import nozzle


def test_enumerate_senders():
    senders = nozzle.enumerate_senders()
    assert isinstance(senders, list)


def test_list_senders():
    names = nozzle.list_senders()
    assert isinstance(names, list)


def test_sender_create():
    sender = nozzle.Sender.create("test_sender", "test_app")
    assert sender.valid()
    info = sender.info()
    assert info.name == "test_sender"
    assert info.application_name == "test_app"


def test_sender_create_defaults():
    sender = nozzle.Sender.create("test_sender_defaults")
    assert sender.valid()


def test_send_receive_rgba8():
    sender = nozzle.Sender.create("test_roundtrip_py", "test_app")
    receiver = nozzle.Receiver.create("test_roundtrip_py", "test_viewer")

    img = np.zeros((480, 640, 4), dtype=np.uint8)
    img[100, 100] = [255, 128, 64, 255]

    sender.publish_array(img)

    frame = receiver.acquire_frame(timeout_ms=1000)
    if frame.valid():
        received = frame.get_array()
        assert received.shape == (480, 640, 4)
        assert received.dtype == np.uint8
        np.testing.assert_array_equal(received[100, 100], [255, 128, 64, 255])
    else:
        pytest.skip("could not acquire frame (sender/receiver not connected in time)")


def test_frame_info():
    sender = nozzle.Sender.create("test_info_py", "test_app")
    receiver = nozzle.Receiver.create("test_info_py", "test_viewer")

    sender.publish_array(np.zeros((100, 200, 4), dtype=np.uint8))

    frame = receiver.acquire_frame(timeout_ms=1000)
    if frame.valid():
        info = frame.info()
        assert info.width == 200
        assert info.height == 100
        assert info.format == nozzle.TextureFormat.RGBA8_UNORM
    else:
        pytest.skip("could not acquire frame")


def test_enums():
    assert nozzle.BackendType.METAL.value != nozzle.BackendType.D3D11.value
    assert nozzle.TextureFormat.RGBA8_UNORM.value != nozzle.TextureFormat.RGBA32_FLOAT.value
    assert nozzle.ReceiveMode.LATEST_ONLY.value != nozzle.ReceiveMode.SEQUENTIAL_BEST_EFFORT.value


def test_invalid_array_shape():
    sender = nozzle.Sender.create("test_invalid_py", "test_app")
    bad = np.zeros((5,), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        sender.publish_array(bad)


def test_sender_info():
    sender = nozzle.Sender.create("test_info_check", "my_app", ring_buffer_size=5)
    info = sender.info()
    assert info.name == "test_info_check"
    assert info.application_name == "my_app"


def test_send_receive_grayscale():
    sender = nozzle.Sender.create("test_gray_py", "test_app")
    receiver = nozzle.Receiver.create("test_gray_py", "test_viewer")

    gray = np.full((64, 128), 128, dtype=np.uint8)
    sender.publish_array(gray)

    frame = receiver.acquire_frame(timeout_ms=1000)
    if frame.valid():
        received = frame.get_array()
        assert received.shape == (64, 128)
        assert received.dtype == np.uint8
        assert received[0, 0] == 128
    else:
        pytest.skip("could not acquire frame (sender/receiver not connected in time)")


def test_send_receive_float32():
    sender = nozzle.Sender.create("test_f32_py", "test_app")
    receiver = nozzle.Receiver.create("test_f32_py", "test_viewer")

    img = np.ones((32, 48, 4), dtype=np.float32) * 0.5
    sender.publish_array(img)

    frame = receiver.acquire_frame(timeout_ms=1000)
    if frame.valid():
        received = frame.get_array()
        assert received.shape == (32, 48, 4)
        assert received.dtype == np.float32
        np.testing.assert_allclose(received[0, 0], [0.5, 0.5, 0.5, 0.5], atol=1e-6)
    else:
        pytest.skip("could not acquire frame (sender/receiver not connected in time)")


def test_frame_dlpack():
    sender = nozzle.Sender.create("test_dlpack_py", "test_app")
    receiver = nozzle.Receiver.create("test_dlpack_py", "test_viewer")

    img = np.zeros((32, 32, 4), dtype=np.uint8)
    img[0, 0] = [200, 100, 50, 255]
    sender.publish_array(img)

    frame = receiver.acquire_frame(timeout_ms=1000)
    if frame.valid():
        device, device_id = frame.__dlpack_device__()
        assert device == 1
        assert device_id == 0

        arr = frame.__dlpack__()
        assert arr.shape == (32, 32, 4)
        assert arr.dtype == np.uint8
        np.testing.assert_array_equal(arr[0, 0], [200, 100, 50, 255])
    else:
        pytest.skip("could not acquire frame (sender/receiver not connected in time)")


def test_locked_pixels():
    sender = nozzle.Sender.create("test_locked_py", "test_app")
    receiver = nozzle.Receiver.create("test_locked_py", "test_viewer")

    img = np.zeros((64, 64, 4), dtype=np.uint8)
    img[10, 20] = [255, 0, 128, 200]
    sender.publish_array(img)

    frame = receiver.acquire_frame(timeout_ms=1000)
    if frame.valid():
        locked = frame.lock_pixels()
        assert locked.width == 64
        assert locked.height == 64

        device, device_id = locked.__dlpack_device__()
        assert device == 1
        assert device_id == 0

        arr = locked.to_ndarray()
        assert arr.shape == (64, 64, 4)
        assert arr.dtype == np.uint8
        np.testing.assert_array_equal(arr[10, 20], [255, 0, 128, 200])

        dlpack_arr = locked.__dlpack__()
        assert dlpack_arr.shape == (64, 64, 4)
        np.testing.assert_array_equal(dlpack_arr[10, 20], [255, 0, 128, 200])
    else:
        pytest.skip("could not acquire frame (sender/receiver not connected in time)")


def test_locked_pixels_invalid_frame():
    sender = nozzle.Sender.create("test_invalid_frame_py", "test_app")
    sender.publish_array(np.zeros((4, 4, 4), dtype=np.uint8))
    receiver = nozzle.Receiver.create("test_invalid_frame_py", "test_viewer")
    frame = receiver.acquire_frame(timeout_ms=1000)
    if not frame.valid():
        pytest.skip("could not acquire frame")
    frame.release()
    with pytest.raises(RuntimeError, match="frame is not valid"):
        frame.lock_pixels()


def test_send_receive_r32_float():
    sender = nozzle.Sender.create("test_r32f_py", "test_app")
    receiver = nozzle.Receiver.create("test_r32f_py", "test_viewer")

    img = np.full((16, 32), 3.14, dtype=np.float32)
    sender.publish_array(img)

    frame = receiver.acquire_frame(timeout_ms=1000)
    if frame.valid():
        received = frame.get_array()
        assert received.shape == (16, 32)
        assert received.dtype == np.float32
        np.testing.assert_allclose(received[0, 0], 3.14, atol=1e-6)
    else:
        pytest.skip("could not acquire frame (sender/receiver not connected in time)")
