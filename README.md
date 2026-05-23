# py.nozzle

> This codebase is currently in its AI-slob prototyping phase: the code runs on momentum, vibes, and plausible intent.
> Proper debugging will be introduced once demand graduates from hypothetical to measurable.

Python bindings for [nozzle](https://github.com/nozzle-io/nozzle) — cross-platform inter-process GPU texture sharing.

Built with [nanobind](https://github.com/wjakob/nanobind) and [scikit-build-core](https://github.com/scientific-python/scikit-build-core). NumPy arrays are zero-copy via DLPack.

## Disclaimer / Notice

This library is currently a work in progress and contains many incomplete features and unverified implementations.
Although it may appear usable at first glance, it may not function correctly.

Please use it with the understanding that no guarantees are made regarding its behavior, and perform debugging, validation, and review as needed.
If you encounter problems, please do not become angry; instead, contributions in the form of Issues or Pull Requests would be greatly appreciated.

## Requirements

- Python 3.12+
- CMake 3.20+
- C++17 compiler
- macOS 12+ / Windows 10+ / Linux (glibc 2.31+)

## Install

```bash
pip install nozzle-io
```

The distribution name is `nozzle-io` because the `nozzle` name on PyPI is already occupied by an unrelated project. The import package remains `nozzle`:

```python
import nozzle
```

Do not install another distribution that provides a top-level `nozzle` package into the same environment.

### From Source

```bash
git clone --recurse-submodules https://github.com/nozzle-io/py.nozzle.git
cd py.nozzle
pip install .
```

## Usage

### Sender

```python
import nozzle
import numpy as np

sender = nozzle.Sender.create("my_output", "MyApp")

img = np.zeros((480, 640, 4), dtype=np.uint8)
img[100, 100] = [255, 128, 64, 255]
sender.publish_array(img)
```

### Receiver

```python
receiver = nozzle.Receiver.create("my_output", "MyViewer")

frame = receiver.acquire_frame(timeout_ms=1000)
if frame.valid():
    info = frame.info()
    print(f"{info.width}x{info.height}, format={info.format}")

    arr = frame.get_array()
    print(arr.shape, arr.dtype)
```

### Discovery

```python
senders = nozzle.enumerate_senders()
for s in senders:
    print(f"{s.name} ({s.application_name}) — {s.backend}")

names = nozzle.list_senders()
```

### Pixel Formats

#### Receiver Format Mapping

All nozzle texture formats map to NumPy dtypes when receiving frames:

| Nozzle Format | NumPy dtype | Channels | Bytes/Pixel |
|---|---|---|---|
| R8 UNORM | uint8 | 1 | 1 |
| RG8 UNORM | uint8 | 2 | 2 |
| RGBA8 UNORM / BGRA8 UNORM / RGBA8 SRGB / BGRA8 SRGB | uint8 | 4 | 4 |
| R16 UNORM | uint16 | 1 | 2 |
| RG16 UNORM | uint16 | 2 | 4 |
| RGBA16 UNORM | uint16 | 4 | 8 |
| R16 Float | uint16 | 1 | 2 |
| RG16 Float | uint16 | 2 | 4 |
| RGBA16 Float | uint16 | 4 | 8 |
| R32 Float | float32 | 1 | 4 |
| RG32 Float | float32 | 2 | 8 |
| RGBA32 Float | float32 | 4 | 16 |
| R32 Uint | uint32 | 1 | 4 |
| RGBA32 Uint | uint32 | 4 | 16 |

#### Sender Format Auto-detection

When sending NumPy arrays via `publish_array()`, the following dtype and channel combinations are auto-detected:

| NumPy dtype | Channels | Nozzle Format |
|---|---|---|
| uint8 | 1 | R8 UNORM |
| uint8 | 2 | RG8 UNORM |
| uint8 | 3 | RGBA8 UNORM (3 channels padded) |
| uint8 | 4 | RGBA8 UNORM |
| float32 | 1 | R32 Float |
| float32 | 4 | RGBA32 Float |
| uint16 | 4 | RGBA16 Float |

Other dtype and channel combinations are not auto-detected for sending. Use the C API directly for those formats.

### DLPack Support

Frames implement `__dlpack__` and `__dlpack_device__` for zero-copy interop with PyTorch, JAX, etc.

```python
import torch
frame = receiver.acquire_frame(timeout_ms=1000)
if frame.valid():
    tensor = torch.from_dlpack(frame)
```

## API Reference

### `Sender`

| Method | Description |
|--------|-------------|
| `Sender.create(name, application_name, ring_buffer_size=3)` | Create a sender |
| `sender.publish_array(array)` | Publish a NumPy array as a frame |
| `sender.info()` | Get sender metadata |
| `sender.valid()` | Check if sender is initialized |

### `Receiver`

| Method | Description |
|--------|-------------|
| `Receiver.create(name, application_name)` | Create a receiver |
| `receiver.acquire_frame(timeout_ms=0)` | Acquire latest frame |
| `receiver.valid()` | Check if receiver is initialized |

### `Frame`

| Method | Description |
|--------|-------------|
| `frame.info()` | Get frame metadata (width, height, format, frame_index) |
| `frame.get_array()` | Copy frame data to NumPy array |
| `frame.lock_pixels()` | Zero-copy pixel access via `LockedPixels` |
| `frame.valid()` | Check if frame holds valid data |
| `frame.release()` | Explicitly release frame |
| `frame.__dlpack__()` | DLPack export |

## Architecture

```
py.nozzle
├── src/nozzle.cpp     nanobind C++ extension wrapping nozzle C ABI
├── src/nozzle/        Python package (__init__.py)
├── libs/nozzle/       git submodule (nozzle static library)
└── tests/             pytest test suite
```

The extension calls through the C++ API (`nozzle.hpp`).

## License

MIT

Third-party dependencies:

- [nozzle](https://github.com/nozzle-io/nozzle) — MIT
- [nanobind](https://github.com/wjakob/nanobind) — BSD-3-Clause
