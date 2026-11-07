# py.nozzle

> This codebase is currently in its AI-slob prototyping phase: the code runs on momentum, vibes, and plausible intent.
> Proper debugging will be introduced once demand graduates from hypothetical to measurable.

Python bindings for [nozzle](https://github.com/nozzle-io/nozzle) — cross-platform inter-process GPU texture sharing.

Built with [nanobind](https://github.com/wjakob/nanobind) and [scikit-build-core](https://github.com/scientific-python/scikit-build-core). Frame arrays can be copied or accessed as raw mapped views.

## Disclaimer / Notice

This library is currently a work in progress and contains many incomplete features and unverified implementations.
Although it may appear usable at first glance, it may not function correctly.

Please use it with the understanding that no guarantees are made regarding its behavior, and perform debugging, validation, and review as needed.
If you encounter problems, please do not become angry; instead, contributions in the form of Issues or Pull Requests would be greatly appreciated.

## Requirements

- Python 3.12+
- CMake 3.26+
- C++17 compiler
- macOS 14+ for release wheels / Windows 10+ / Linux (glibc 2.31+)

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

The CMake default `CMAKE_OSX_DEPLOYMENT_TARGET=12.0` is a legacy fallback for
source builds when no target is supplied. It is not the Python distribution's
supported minimum and does not demonstrate macOS 12 runtime compatibility.
Release wheel CI explicitly supplies `MACOSX_DEPLOYMENT_TARGET=14.0`, which
sets both the native compilation minimum and wheel platform tag. A local
source build with different settings is not a conforming release wheel.

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

#### Copying and Raw Pixel Access

`frame.get_array()` returns a packed, top-left-origin copy. BGRA8 storage is
converted to RGBA channel order, including sRGB formats without changing their
transfer function. RGBA16 Float is widened to float32; other listed formats
retain their element representation. Float32 channels are not swapped.

`frame.lock_pixels()` transfers the frame into a `LockedPixels` owner.
`locked.to_ndarray()` is a read-only raw view: its channel order is the actual
`locked.format`, and its strides retain mapped row padding. In particular,
BGRA8 yields `[B, G, R, A]`, not `[R, G, B, A]`. The view retains the mapping
owner. No in-place swap or hidden normalization copy is performed. Mapping can
still involve backend staging/readback. Read-only access prevents writes through
the view; it does not freeze producer updates to shared storage. Use
`get_array()` for a retained copy rather than retaining a live mapped view.

`frame.info().format` describes storage; `semantic_format` describes the
published meaning. A logical RGBA8 frame can have BGRA8 storage on Metal.

Raw mapped views use these NumPy representations (half-float entries expose
uint16 bit patterns, not numerical uint16 values):

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

Four-channel uint8 input is logical RGBA; three-channel input is RGB with
opaque alpha added. `publish_array()` converts to the actual mapped storage
order. Read-only CPU NumPy arrays are accepted; strided arrays are converted
to a contiguous temporary by nanobind before copying. Unsupported storage
conversions fail rather than reinterpreting the input bytes or reducing
precision silently.

### NumPy API Migration

`Frame` and `LockedPixels` do not implement DLPack. The former `__dlpack__`
and `__dlpack_device__` methods were removed because they advertised a protocol
that the array-returning implementation did not implement
([#215](https://github.com/nozzle-io/nozzle-dev/issues/215)). A real exporter is
deferred; do not pass these objects to `numpy.from_dlpack`, PyTorch or JAX.

Replace direct `frame.__dlpack__()` calls with `frame.get_array()` for logical
channel-order copies. Note that `get_array()` widens RGBA16 Float to float32,
whereas the removed method returned uint16 half bits. For raw storage and half
bits, create `locked = frame.lock_pixels()`, use `locked.to_ndarray()` and
consult `locked.format`.
Replace direct `locked.__dlpack__()` calls with `locked.to_ndarray()`; this
view is now read-only. Use `.copy()` when an independently writable array is
needed. These paths do not promise GPU zero-copy interoperability.

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
| `frame.lock_pixels()` | Transfer frame ownership to `LockedPixels` for read-only raw mapped access |
| `frame.valid()` | Check if frame holds valid data |
| `frame.release()` | Explicitly release frame |

## Architecture

### macOS release wheels

Published wheels retain the `cp312-abi3-macosx_14_0_universal2` contract,
independent of the GitHub runner OS. The packaging job sets
`MACOSX_DEPLOYMENT_TARGET=14.0` for both scikit-build-core wheel tagging and
native compilation. Both the wheel metadata and each Mach-O slice must match
that policy. A macOS 15 smoke run does not establish macOS 14 runtime testing
or execution of the Intel slice. Developer builds can select another target;
they are not automatically accepted as release artifacts.

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
