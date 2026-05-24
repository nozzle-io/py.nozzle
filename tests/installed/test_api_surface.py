import nozzle


def test_expected_public_exports_exist():
    expected_exports = [
        "BackendType",
        "TextureFormat",
        "TransferMode",
        "ReceiveMode",
        "FrameInfo",
        "SenderInfo",
        "ConnectedSenderInfo",
        "Frame",
        "LockedPixels",
        "Sender",
        "Receiver",
        "Device",
        "enumerate_senders",
        "list_senders",
    ]
    for name in expected_exports:
        assert hasattr(nozzle, name), f"missing public export: {name}"


def test_enum_surface_is_available_without_backend():
    assert nozzle.BackendType.METAL.value != nozzle.BackendType.D3D11.value
    assert nozzle.TextureFormat.RGBA8_UNORM.value != nozzle.TextureFormat.RGBA32_FLOAT.value
    assert nozzle.TransferMode.ZERO_COPY_SHARED_TEXTURE.value != nozzle.TransferMode.CPU_COPY.value
    assert nozzle.ReceiveMode.LATEST_ONLY.value != nozzle.ReceiveMode.SEQUENTIAL_BEST_EFFORT.value


def test_class_method_surface_is_available_without_instantiation():
    assert hasattr(nozzle.Sender, "create")
    assert hasattr(nozzle.Sender, "valid")
    assert hasattr(nozzle.Sender, "info")
    assert hasattr(nozzle.Sender, "publish_array")
    assert hasattr(nozzle.Receiver, "create")
    assert hasattr(nozzle.Receiver, "valid")
    assert hasattr(nozzle.Receiver, "is_connected")
    assert hasattr(nozzle.Receiver, "connected_info")
    assert hasattr(nozzle.Receiver, "sender_metadata")
    assert hasattr(nozzle.Receiver, "acquire_frame")
    assert hasattr(nozzle.Device, "default_device")
