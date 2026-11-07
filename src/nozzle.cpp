#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nozzle/nozzle.hpp>
#include <nozzle/pixel_access.hpp>
#include <nozzle/discovery.hpp>
#include <nozzle/channel_swizzle.hpp>
#include <nozzle/format_convert.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
namespace nz = nozzle;

struct format_info {
    uint32_t channels;
    nb::dlpack::dtype dtype;
    size_t element_bytes;
};

static format_info get_format_info(nz::texture_format fmt) {
    switch (fmt) {
        case nz::texture_format::r8_unorm:      return {1, nb::dtype<uint8_t>(),  1};
        case nz::texture_format::rg8_unorm:     return {2, nb::dtype<uint8_t>(),  1};
        case nz::texture_format::rgba8_unorm:   return {4, nb::dtype<uint8_t>(),  1};
        case nz::texture_format::bgra8_unorm:   return {4, nb::dtype<uint8_t>(),  1};
        case nz::texture_format::rgba8_srgb:    return {4, nb::dtype<uint8_t>(),  1};
        case nz::texture_format::bgra8_srgb:    return {4, nb::dtype<uint8_t>(),  1};
        case nz::texture_format::r16_unorm:     return {1, nb::dtype<uint16_t>(), 2};
        case nz::texture_format::rg16_unorm:    return {2, nb::dtype<uint16_t>(), 2};
        case nz::texture_format::rgba16_unorm:  return {4, nb::dtype<uint16_t>(), 2};
        case nz::texture_format::r16_float:     return {1, nb::dtype<uint16_t>(), 2};
        case nz::texture_format::rg16_float:    return {2, nb::dtype<uint16_t>(), 2};
        case nz::texture_format::rgba16_float:  return {4, nb::dtype<uint16_t>(), 2};
        case nz::texture_format::r32_float:     return {1, nb::dtype<float>(),    4};
        case nz::texture_format::rg32_float:    return {2, nb::dtype<float>(),    4};
        case nz::texture_format::rgba32_float:  return {4, nb::dtype<float>(),    4};
        case nz::texture_format::r32_uint:      return {1, nb::dtype<uint32_t>(), 4};
        case nz::texture_format::rgba32_uint:   return {4, nb::dtype<uint32_t>(), 4};
        default:                                 return {0, nb::dtype<uint8_t>(),  0};
    }
}

static nz::texture_format numpy_dtype_to_format(nb::dlpack::dtype dt, uint32_t channels) {
    if (dt == nb::dtype<uint8_t>() && channels == 4) return nz::texture_format::rgba8_unorm;
    if (dt == nb::dtype<uint8_t>() && channels == 3) return nz::texture_format::rgba8_unorm;
    if (dt == nb::dtype<uint8_t>() && channels == 2) return nz::texture_format::rg8_unorm;
    if (dt == nb::dtype<uint8_t>() && channels == 1) return nz::texture_format::r8_unorm;
    if (dt == nb::dtype<float>() && channels == 4)   return nz::texture_format::rgba32_float;
    if (dt == nb::dtype<float>() && channels == 1)   return nz::texture_format::r32_float;
    if (dt == nb::dtype<uint16_t>() && channels == 4) return nz::texture_format::rgba16_float;
    return nz::texture_format::unknown;
}

static bool is_bgra8(nz::texture_format format) {
    return format == nz::texture_format::bgra8_unorm ||
           format == nz::texture_format::bgra8_srgb;
}

static nb::object pixels_to_numpy(const nz::mapped_pixels &px, bool widen_rgba_half = false) {
    auto fi = get_format_info(px.format);
    if (fi.channels == 0) {
        throw std::runtime_error("unsupported texture format");
    }

    const bool widen = widen_rgba_half && px.format == nz::texture_format::rgba16_float;
    if (widen) {
        fi.dtype = nb::dtype<float>();
        fi.element_bytes = sizeof(float);
    }
    const size_t row_bytes = static_cast<size_t>(px.width) * fi.channels * fi.element_bytes;
    if (px.height == 0 || row_bytes == 0 ||
        row_bytes > std::numeric_limits<size_t>::max() / px.height) {
        throw std::runtime_error("invalid mapped pixel dimensions");
    }
    auto buf = std::make_unique<std::vector<uint8_t>>(row_bytes * px.height);
    if (widen) {
        auto converted = nz::widen_half_to_float(
            px.data, buf->data(), px.width, px.height,
            px.row_stride_bytes, static_cast<int64_t>(row_bytes), fi.channels);
        if (!converted.ok()) {
            throw std::runtime_error(converted.error().message.c_str());
        }
    } else if (is_bgra8(px.format)) {
        auto converted = nz::swizzle_channels(
            px.data, buf->data(), px.width, px.height,
            px.row_stride_bytes, static_cast<int64_t>(row_bytes),
            px.format, nz::channel_permute{{2, 1, 0, 3}});
        if (!converted.ok()) {
            throw std::runtime_error(converted.error().message.c_str());
        }
    } else {
        for (uint32_t y = 0; y < px.height; ++y) {
            const auto *src = static_cast<const uint8_t *>(px.data) +
                static_cast<std::ptrdiff_t>(y) * px.row_stride_bytes;
            std::memcpy(buf->data() + static_cast<size_t>(y) * row_bytes, src, row_bytes);
        }
    }

    auto *data = buf->data();
    nb::capsule deleter(buf.get(), [](void *p) noexcept {
        delete static_cast<std::vector<uint8_t> *>(p);
    });
    buf.release();

    if (fi.channels > 1) {
        size_t shape[3] = {
            static_cast<size_t>(px.height),
            static_cast<size_t>(px.width),
            static_cast<size_t>(fi.channels)
        };
        nb::ndarray<nb::numpy> arr(
            data, 3, shape, deleter, nullptr, fi.dtype
        );
        return nb::cast(std::move(arr));
    } else {
        size_t shape[2] = {
            static_cast<size_t>(px.height),
            static_cast<size_t>(px.width)
        };
        nb::ndarray<nb::numpy> arr(
            data, 2, shape, deleter, nullptr, fi.dtype
        );
        return nb::cast(std::move(arr));
    }
}

struct LockedPixels {
    nz::frame frame_;
    void *data{nullptr};
    std::ptrdiff_t row_bytes{0};
    uint32_t width{0};
    uint32_t height{0};
    nz::texture_format format{nz::texture_format::unknown};
    bool locked{false};

    explicit LockedPixels(nz::frame frm) : frame_(std::move(frm)) {
        if (!frame_.valid()) {
            throw std::runtime_error("frame is not valid");
        }
        auto result = nz::lock_frame_pixels_with_origin(frame_, nz::texture_origin::top_left);
        if (!result.ok()) {
            throw std::runtime_error(result.error().message.c_str());
        }
        auto &px = result.value();
        data = px.data;
        row_bytes = px.row_stride_bytes;
        width = px.width;
        height = px.height;
        format = px.format;
        locked = true;
    }

    ~LockedPixels() {
        if (locked) {
            nz::unlock_frame_pixels(frame_);
            locked = false;
        }
    }

    LockedPixels(const LockedPixels &) = delete;
    LockedPixels &operator=(const LockedPixels &) = delete;
    LockedPixels(LockedPixels &&o) noexcept
        : frame_(std::move(o.frame_)), data(o.data), row_bytes(o.row_bytes),
          width(o.width), height(o.height), format(o.format), locked(o.locked) {
        o.locked = false;
        o.data = nullptr;
    }
};

NB_MODULE(_nozzle, m) {
    m.attr("__version__") = "0.1.0";

    nb::register_exception_translator([](const std::exception_ptr &exception, void *) {
        try {
            if (exception) {
                std::rethrow_exception(exception);
            }
        } catch (const std::runtime_error &error) {
            PyErr_SetString(PyExc_RuntimeError, error.what());
        }
    });

    nb::enum_<nz::backend_type>(m, "BackendType")
        .value("UNKNOWN", nz::backend_type::unknown)
        .value("D3D11", nz::backend_type::d3d11)
        .value("METAL", nz::backend_type::metal)
        .value("OPENGL", nz::backend_type::opengl)
        .value("DMA_BUF", nz::backend_type::dma_buf)
        ;

    nb::enum_<nz::texture_format>(m, "TextureFormat")
        .value("UNKNOWN", nz::texture_format::unknown)
        .value("R8_UNORM", nz::texture_format::r8_unorm)
        .value("RG8_UNORM", nz::texture_format::rg8_unorm)
        .value("RGBA8_UNORM", nz::texture_format::rgba8_unorm)
        .value("BGRA8_UNORM", nz::texture_format::bgra8_unorm)
        .value("RGBA8_SRGB", nz::texture_format::rgba8_srgb)
        .value("BGRA8_SRGB", nz::texture_format::bgra8_srgb)
        .value("R16_UNORM", nz::texture_format::r16_unorm)
        .value("RG16_UNORM", nz::texture_format::rg16_unorm)
        .value("RGBA16_UNORM", nz::texture_format::rgba16_unorm)
        .value("R16_FLOAT", nz::texture_format::r16_float)
        .value("RG16_FLOAT", nz::texture_format::rg16_float)
        .value("RGBA16_FLOAT", nz::texture_format::rgba16_float)
        .value("R32_FLOAT", nz::texture_format::r32_float)
        .value("RG32_FLOAT", nz::texture_format::rg32_float)
        .value("RGBA32_FLOAT", nz::texture_format::rgba32_float)
        .value("R32_UINT", nz::texture_format::r32_uint)
        .value("RGBA32_UINT", nz::texture_format::rgba32_uint)
        .value("DEPTH32_FLOAT", nz::texture_format::depth32_float)
        ;

    nb::enum_<nz::transfer_mode>(m, "TransferMode")
        .value("UNKNOWN", nz::transfer_mode::unknown)
        .value("ZERO_COPY_SHARED_TEXTURE", nz::transfer_mode::zero_copy_shared_texture)
        .value("GPU_COPY", nz::transfer_mode::gpu_copy)
        .value("CPU_COPY", nz::transfer_mode::cpu_copy)
        ;

    nb::enum_<nz::receive_mode>(m, "ReceiveMode")
        .value("LATEST_ONLY", nz::receive_mode::latest_only)
        .value("SEQUENTIAL_BEST_EFFORT", nz::receive_mode::sequential_best_effort)
        ;

    nb::class_<nz::frame_info>(m, "FrameInfo")
        .def_ro("frame_index", &nz::frame_info::frame_index)
        .def_ro("timestamp_ns", &nz::frame_info::timestamp_ns)
        .def_ro("width", &nz::frame_info::width)
        .def_ro("height", &nz::frame_info::height)
        .def_ro("format", &nz::frame_info::format)
        .def_ro("semantic_format", &nz::frame_info::semantic_format)
        .def_ro("transfer_mode", &nz::frame_info::transfer_mode_val)
        .def_ro("sync_mode", &nz::frame_info::sync_mode_val)
        .def_ro("dropped_frame_count", &nz::frame_info::dropped_frame_count)
        ;

    nb::class_<nz::sender_info>(m, "SenderInfo")
        .def_ro("name", &nz::sender_info::name)
        .def_ro("application_name", &nz::sender_info::application_name)
        .def_ro("id", &nz::sender_info::id)
        .def_ro("backend", &nz::sender_info::backend)
        ;

    nb::class_<nz::connected_sender_info>(m, "ConnectedSenderInfo")
        .def_ro("name", &nz::connected_sender_info::name)
        .def_ro("application_name", &nz::connected_sender_info::application_name)
        .def_ro("id", &nz::connected_sender_info::id)
        .def_ro("backend", &nz::connected_sender_info::backend)
        .def_ro("width", &nz::connected_sender_info::width)
        .def_ro("height", &nz::connected_sender_info::height)
        .def_ro("format", &nz::connected_sender_info::format)
        .def_ro("semantic_format", &nz::connected_sender_info::semantic_format)
        .def_ro("estimated_fps", &nz::connected_sender_info::estimated_fps)
        .def_ro("frame_counter", &nz::connected_sender_info::frame_counter)
        .def_ro("last_update_time_ns", &nz::connected_sender_info::last_update_time_ns)
        ;

    nb::class_<LockedPixels>(m, "LockedPixels")
        .def("to_ndarray", [](LockedPixels &lp) -> nb::ndarray<nb::numpy, nb::ro> {
            auto fi = get_format_info(lp.format);
            if (fi.channels == 0) {
                throw std::runtime_error("unsupported texture format");
            }

            nb::handle owner = nb::find(&lp);

            if (fi.channels > 1) {
                size_t shape[3] = {
                    static_cast<size_t>(lp.height),
                    static_cast<size_t>(lp.width),
                    static_cast<size_t>(fi.channels)
                };
                int64_t strides[3] = {
                    static_cast<int64_t>(lp.row_bytes / static_cast<std::ptrdiff_t>(fi.element_bytes)),
                    static_cast<int64_t>(fi.channels),
                    1
                };
                return nb::ndarray<nb::numpy, nb::ro>(
                    lp.data, 3, shape, owner, strides, fi.dtype
                );
            } else {
                size_t shape[2] = {
                    static_cast<size_t>(lp.height),
                    static_cast<size_t>(lp.width)
                };
                int64_t strides[2] = {
                    static_cast<int64_t>(lp.row_bytes / static_cast<std::ptrdiff_t>(fi.element_bytes)),
                    1
                };
                return nb::ndarray<nb::numpy, nb::ro>(
                    lp.data, 2, shape, owner, strides, fi.dtype
                );
            }
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("width", [](const LockedPixels &lp) { return lp.width; })
        .def_prop_ro("height", [](const LockedPixels &lp) { return lp.height; })
        .def_prop_ro("format", [](const LockedPixels &lp) { return lp.format; })
        ;

    nb::class_<nz::frame>(m, "Frame")
        .def("valid", &nz::frame::valid)
        .def("info", &nz::frame::info)
        .def("release", &nz::frame::release)
        .def("lock_pixels", [](nb::handle py_frm) -> LockedPixels* {
            auto *frm_ptr = nb::inst_ptr<nz::frame>(py_frm);
            auto locked = new LockedPixels(std::move(*frm_ptr));
            return locked;
        }, nb::rv_policy::take_ownership)
        .def("get_array", [](nz::frame &frm) -> nb::object {
            if (!frm.valid()) {
                throw std::runtime_error("frame is not valid");
            }
            auto result = nz::lock_frame_pixels_mapping_with_origin(frm, nz::texture_origin::top_left);
            if (!result.ok()) {
                throw std::runtime_error(result.error().message.c_str());
            }
            return pixels_to_numpy(result.value().pixels(), true);
        })
        ;

    nb::class_<nz::sender>(m, "Sender")
        .def_static("create", [](const std::string &name,
                                  const std::string &app_name,
                                  uint32_t ring_size) {
            nz::sender_desc desc{};
            desc.name = name;
            desc.application_name = app_name;
            desc.ring_buffer_size = ring_size;
            auto result = nz::sender::create(desc);
            if (!result.ok()) {
                throw std::runtime_error(result.error().message.c_str());
            }
            return std::move(result.value());
        }, "name"_a, "application_name"_a = "", "ring_buffer_size"_a = 3)
        .def("info", &nz::sender::info)
        .def("valid", &nz::sender::valid)
        .def("publish_array", [](nz::sender &snd,
                                  nb::ndarray<nb::numpy, nb::device::cpu, nb::c_contig, nb::ro> arr) {
            int ndim = arr.ndim();
            if (ndim < 2 || ndim > 3) {
                throw std::runtime_error("array must be 2D (H, W) or 3D (H, W, C)");
            }

            if (arr.shape(0) == 0 || arr.shape(1) == 0 ||
                arr.shape(0) > std::numeric_limits<uint32_t>::max() ||
                arr.shape(1) > std::numeric_limits<uint32_t>::max() ||
                (ndim == 3 && (arr.shape(2) == 0 || arr.shape(2) > 4))) {
                throw std::runtime_error("invalid array dimensions or channel count");
            }
            uint32_t height = static_cast<uint32_t>(arr.shape(0));
            uint32_t width = static_cast<uint32_t>(arr.shape(1));
            uint32_t channels = ndim == 3 ? static_cast<uint32_t>(arr.shape(2)) : 1;

            auto dt = arr.dtype();
            nz::texture_format fmt = numpy_dtype_to_format(dt, channels);
            if (fmt == nz::texture_format::unknown) {
                throw std::runtime_error(
                    "unsupported array dtype/channels combination: "
                    "expected uint8 (1/2/3/4ch), float32 (1/4ch), or uint16 (4ch)"
                );
            }

            nz::texture_desc tdesc{};
            tdesc.width = width;
            tdesc.height = height;
            tdesc.format = fmt;

            auto wf_result = snd.acquire_writable_frame(tdesc);
            if (!wf_result.ok()) {
                throw std::runtime_error(wf_result.error().message.c_str());
            }
            auto &wf = wf_result.value();
            // The mapping is destroyed first; discard then releases a reserved ring slot on errors.
            struct writable_guard {
                nz::sender &sender;
                nz::writable_frame &frame;
                ~writable_guard() {
                    if (frame.valid()) {
                        (void)sender.discard_frame(frame);
                    }
                }
            } guard{snd, wf};

            auto px_result = nz::lock_writable_pixels_mapping_with_origin(wf, nz::texture_origin::top_left);
            if (!px_result.ok()) {
                throw std::runtime_error(px_result.error().message.c_str());
            }
            const auto &px = px_result.value().pixels();
            const bool swap_rb = fmt == nz::texture_format::rgba8_unorm &&
                                 px.format == nz::texture_format::bgra8_unorm;
            if (px.width != width || px.height != height ||
                (px.format != fmt && !swap_rb)) {
                throw std::runtime_error("unsupported mapped storage conversion for input array");
            }
            const auto fi = get_format_info(fmt);
            const size_t src_row_bytes = static_cast<size_t>(width) * channels * fi.element_bytes;
            if (channels == 3) {
                for (uint32_t y = 0; y < height; ++y) {
                    const auto *src = static_cast<const uint8_t *>(arr.data()) +
                        static_cast<size_t>(y) * src_row_bytes;
                    auto *dst = static_cast<uint8_t *>(px.data) +
                        static_cast<std::ptrdiff_t>(y) * px.row_stride_bytes;
                    for (uint32_t x = 0; x < width; ++x) {
                        dst[4 * x] = src[3 * x + (swap_rb ? 2 : 0)];
                        dst[4 * x + 1] = src[3 * x + 1];
                        dst[4 * x + 2] = src[3 * x + (swap_rb ? 0 : 2)];
                        dst[4 * x + 3] = 255;
                    }
                }
            } else if (swap_rb) {
                auto converted = nz::swizzle_channels(
                    arr.data(), px.data, width, height,
                    static_cast<int64_t>(src_row_bytes), px.row_stride_bytes,
                    px.format, nz::channel_permute{{2, 1, 0, 3}});
                if (!converted.ok()) {
                    throw std::runtime_error(converted.error().message.c_str());
                }
            } else {
                for (uint32_t y = 0; y < height; ++y) {
                    const auto *src = static_cast<const uint8_t *>(arr.data()) +
                        static_cast<size_t>(y) * src_row_bytes;
                    auto *dst = static_cast<uint8_t *>(px.data) +
                        static_cast<std::ptrdiff_t>(y) * px.row_stride_bytes;
                    std::memcpy(dst, src, src_row_bytes);
                }
            }

            auto unlock_result = px_result.value().unlock_checked();
            if (!unlock_result.ok()) {
                throw std::runtime_error(unlock_result.error().message.c_str());
            }

            auto commit_result = snd.commit_frame(wf);
            if (!commit_result.ok()) {
                throw std::runtime_error(commit_result.error().message.c_str());
            }
        }, "array"_a)
        ;

    nb::class_<nz::receiver>(m, "Receiver")
        .def_static("create", [](const std::string &name,
                                  const std::string &app_name,
                                  nz::receive_mode mode) {
            nz::receiver_desc desc{};
            desc.name = name;
            desc.application_name = app_name;
            desc.receive_mode_val = mode;
            auto result = nz::receiver::create(desc);
            if (!result.ok()) {
                throw std::runtime_error(result.error().message.c_str());
            }
            return std::move(result.value());
        }, "name"_a, "application_name"_a = "", "receive_mode"_a = nz::receive_mode::latest_only)
        .def("valid", &nz::receiver::valid)
        .def("is_connected", &nz::receiver::is_connected)
        .def("connected_info", &nz::receiver::connected_info)
        .def("sender_metadata", &nz::receiver::sender_metadata)
        .def("acquire_frame", [](nz::receiver &rcv, uint64_t timeout_ms) {
            nz::acquire_desc desc{};
            desc.timeout_ms = timeout_ms;
            auto result = rcv.acquire_frame(desc);
            if (!result.ok()) {
                throw std::runtime_error(result.error().message.c_str());
            }
            return std::move(result.value());
        }, "timeout_ms"_a = 0)
        ;

    nb::class_<nz::device>(m, "Device")
        .def_static("default_device", []() {
            auto result = nz::device::default_device();
            if (!result.ok()) {
                throw std::runtime_error(result.error().message.c_str());
            }
            return std::move(result.value());
        })
        .def("valid", &nz::device::valid)
        ;

    m.def("enumerate_senders", []() {
        return nz::enumerate_senders();
    });
}
