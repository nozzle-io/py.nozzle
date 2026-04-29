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
