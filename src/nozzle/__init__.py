from nozzle._nozzle import *
from nozzle._nozzle import __version__

__all__ = [
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
]


def list_senders():
    infos = enumerate_senders()
    return [info.name for info in infos]
