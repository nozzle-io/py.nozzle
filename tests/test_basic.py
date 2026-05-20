import pytest
import numpy as np
import nozzle


_DEFAULT_APPLICATION_NAME = object()


def create_sender_or_skip(name, application_name=_DEFAULT_APPLICATION_NAME, **kwargs):
    try:
        if application_name is _DEFAULT_APPLICATION_NAME:
            return nozzle.Sender.create(name, **kwargs)
        return nozzle.Sender.create(name, application_name, **kwargs)
    except RuntimeError as error:
        message = str(error)
        if "failed to create default device" in message or "No backend available" in message:
            pytest.skip(f"default nozzle backend unavailable on this runner: {message}")
        raise


def wait_for_frame(receiver, timeout_ms=2000):
    frame = receiver.acquire_frame(timeout_ms=timeout_ms)
    if not frame.valid():
        pytest.skip("could not acquire frame (sender/receiver not connected in time)")
    return frame


def publish_until_array(sender, receiver, array, expected_at, expected_value, timeout_attempts=5):
    for attempt in range(timeout_attempts):
        sender.publish_array(array)
        frame = wait_for_frame(receiver)
        received = frame.get_array()
        try:
            np.testing.assert_array_equal(received[expected_at], expected_value)
            return received, frame
        except AssertionError:
            if attempt == timeout_attempts - 1:
                raise
            frame.release()


def publish_until_allclose(sender, receiver, array, expected_at, expected_value, timeout_attempts=5):
    for attempt in range(timeout_attempts):
        sender.publish_array(array)
        frame = wait_for_frame(receiver)
        received = frame.get_array()
        try:
            np.testing.assert_allclose(received[expected_at], expected_value, atol=1e-6)
            return received, frame
        except AssertionError:
            if attempt == timeout_attempts - 1:
                raise
            frame.release()


def test_enumerate_senders():
    senders = nozzle.enumerate_senders()
    assert isinstance(senders, list)


def test_list_senders():
    names = nozzle.list_senders()
    assert isinstance(names, list)


def test_sender_create():
    sender = create_sender_or_skip("test_sender", "test_app")
    assert sender.valid()
    info = sender.info()
    assert info.name == "test_sender"
    assert info.application_name == "test_app"


def test_sender_create_defaults():
    sender = create_sender_or_skip("test_sender_defaults")
    assert sender.valid()


def test_send_receive_rgba8():
    sender = create_sender_or_skip("test_roundtrip_py", "test_app")
    receiver = nozzle.Receiver.create("test_roundtrip_py", "test_viewer")

    img = np.zeros((480, 640, 4), dtype=np.uint8)
    img[100, 100] = [255, 128, 64, 255]

    received, frame = publish_until_array(sender, receiver, img, (100, 100), [255, 128, 64, 255])
    assert received.shape == (480, 640, 4)
    assert received.dtype == np.uint8
    frame.release()


def test_frame_info():
    sender = create_sender_or_skip("test_info_py", "test_app")
    receiver = nozzle.Receiver.create("test_info_py", "test_viewer")

    sender.publish_array(np.zeros((100, 200, 4), dtype=np.uint8))

    frame = wait_for_frame(receiver)
    info = frame.info()
    assert info.width == 200
    assert info.height == 100
    assert info.format == nozzle.TextureFormat.RGBA8_UNORM
    assert hasattr(info, 'semantic_format'), "FrameInfo missing semantic_format attribute"
    assert info.semantic_format == nozzle.TextureFormat.RGBA8_UNORM
    frame.release()


def test_frame_info_semantic_format_field():
    sender = create_sender_or_skip("test_semantic_fmt", "test_app")
    receiver = nozzle.Receiver.create("test_semantic_fmt", "test_viewer")

    sender.publish_array(np.zeros((64, 128, 4), dtype=np.uint8))

    frame = wait_for_frame(receiver)
    info = frame.info()
    assert hasattr(info, 'semantic_format'), "FrameInfo missing semantic_format attribute"
    assert info.semantic_format == nozzle.TextureFormat.RGBA8_UNORM
    assert info.format == info.semantic_format
    frame.release()


def test_connected_sender_info_semantic_format():
    sender = create_sender_or_skip("test_conn_semantic", "test_app")
    receiver = nozzle.Receiver.create("test_conn_semantic", "test_viewer")
    sender.publish_array(np.zeros((32, 32, 4), dtype=np.uint8))

    frame = wait_for_frame(receiver)
    info = receiver.connected_info()
    assert hasattr(info, 'semantic_format'), "ConnectedSenderInfo missing semantic_format attribute"
    assert info.semantic_format == nozzle.TextureFormat.RGBA8_UNORM
    frame.release()


def test_enums():
    assert nozzle.BackendType.METAL.value != nozzle.BackendType.D3D11.value
    assert nozzle.TextureFormat.RGBA8_UNORM.value != nozzle.TextureFormat.RGBA32_FLOAT.value
    assert nozzle.ReceiveMode.LATEST_ONLY.value != nozzle.ReceiveMode.SEQUENTIAL_BEST_EFFORT.value


def test_invalid_array_shape():
    sender = create_sender_or_skip("test_invalid_py", "test_app")
    bad = np.zeros((5,), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        sender.publish_array(bad)


def test_sender_info():
    sender = create_sender_or_skip("test_info_check", "my_app", ring_buffer_size=5)
    info = sender.info()
    assert info.name == "test_info_check"
    assert info.application_name == "my_app"


def test_send_receive_grayscale():
    sender = create_sender_or_skip("test_gray_py", "test_app")
    receiver = nozzle.Receiver.create("test_gray_py", "test_viewer")

    gray = np.full((64, 128), 128, dtype=np.uint8)

    received, frame = publish_until_array(sender, receiver, gray, (0, 0), 128)
    assert received.shape == (64, 128)
    assert received.dtype == np.uint8
    frame.release()


def test_send_receive_float32():
    sender = create_sender_or_skip("test_f32_py", "test_app")
    receiver = nozzle.Receiver.create("test_f32_py", "test_viewer")

    img = np.ones((32, 48, 4), dtype=np.float32) * 0.5

    received, frame = publish_until_allclose(sender, receiver, img, (0, 0), [0.5, 0.5, 0.5, 0.5])
    assert received.shape == (32, 48, 4)
    assert received.dtype == np.float32
    frame.release()


def test_frame_dlpack():
    sender = create_sender_or_skip("test_dlpack_py", "test_app")
    receiver = nozzle.Receiver.create("test_dlpack_py", "test_viewer")

    img = np.zeros((32, 32, 4), dtype=np.uint8)
    img[0, 0] = [200, 100, 50, 255]

    received, frame = publish_until_array(sender, receiver, img, (0, 0), [200, 100, 50, 255])
    device, device_id = frame.__dlpack_device__()
    assert device == 1
    assert device_id == 0

    arr = frame.__dlpack__()
    assert arr.shape == (32, 32, 4)
    assert arr.dtype == np.uint8
    np.testing.assert_array_equal(arr[0, 0], [200, 100, 50, 255])
    np.testing.assert_array_equal(received[0, 0], [200, 100, 50, 255])
    frame.release()


def test_locked_pixels():
    sender = create_sender_or_skip("test_locked_py", "test_app")
    receiver = nozzle.Receiver.create("test_locked_py", "test_viewer")

    img = np.zeros((64, 64, 4), dtype=np.uint8)
    img[10, 20] = [255, 0, 128, 200]

    _, frame = publish_until_array(sender, receiver, img, (10, 20), [255, 0, 128, 200])
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


def test_locked_pixels_invalid_frame():
    sender = create_sender_or_skip("test_invalid_frame_py", "test_app")
    sender.publish_array(np.zeros((4, 4, 4), dtype=np.uint8))
    receiver = nozzle.Receiver.create("test_invalid_frame_py", "test_viewer")
    frame = wait_for_frame(receiver)
    frame.release()
    with pytest.raises(RuntimeError, match="frame is not valid"):
        frame.lock_pixels()


def test_send_receive_r32_float():
    sender = create_sender_or_skip("test_r32f_py", "test_app")
    receiver = nozzle.Receiver.create("test_r32f_py", "test_viewer")

    img = np.full((16, 32), 3.14, dtype=np.float32)

    received, frame = publish_until_allclose(sender, receiver, img, (0, 0), 3.14)
    assert received.shape == (16, 32)
    assert received.dtype == np.float32
    frame.release()
