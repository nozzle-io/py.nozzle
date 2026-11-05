#include <nozzle/nozzle.hpp>
#include <nozzle/pixel_access.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>

static int send_pattern(const char *name) {
    nozzle::sender_desc desc{};
    desc.name = name;
    auto created = nozzle::sender::create(desc);
    if (!created.ok()) {
        std::cerr << created.error().message << '\n';
        return 1;
    }
    auto sender = std::move(created.value());
    nozzle::texture_desc texture{};
    texture.width = 65;
    texture.height = 3;
    texture.format = nozzle::texture_format::rgba8_unorm;
    auto acquired = sender.acquire_writable_frame(texture);
    if (!acquired.ok()) {
        std::cerr << acquired.error().message << '\n';
        return 1;
    }
    auto &frame = acquired.value();
    auto locked = nozzle::lock_writable_pixels_mapping_with_origin(
        frame, nozzle::texture_origin::top_left);
    if (!locked.ok()) {
        std::cerr << locked.error().message << '\n';
        return 1;
    }
    const auto &pixels = locked.value().pixels();
    const bool bgra = pixels.format == nozzle::texture_format::bgra8_unorm;
    if (!bgra && pixels.format != nozzle::texture_format::rgba8_unorm) {
        std::cerr << "unexpected native sender storage format\n";
        return 1;
    }
    for (uint32_t y = 0; y < pixels.height; ++y) {
        auto *row = static_cast<uint8_t *>(pixels.data) +
            static_cast<std::ptrdiff_t>(y) * pixels.row_stride_bytes;
        for (uint32_t x = 0; x < pixels.width; ++x) {
            const uint8_t red = static_cast<uint8_t>(190 + y);
            const uint8_t blue = static_cast<uint8_t>(40 + y);
            row[4 * x] = bgra ? blue : red;
            row[4 * x + 1] = static_cast<uint8_t>(20 + x);
            row[4 * x + 2] = bgra ? red : blue;
            row[4 * x + 3] = 149;
        }
    }
    std::cout << "native_sender_storage=" << (bgra ? "BGRA8" : "RGBA8")
              << " row_stride=" << pixels.row_stride_bytes << '\n';
    auto unlocked = locked.value().unlock_checked();
    if (!unlocked.ok()) {
        std::cerr << unlocked.error().message << '\n';
        return 1;
    }
    auto committed = sender.commit_frame(frame);
    if (!committed.ok()) {
        std::cerr << committed.error().message << '\n';
        return 1;
    }
    std::cout << "native_sender_published=195_pixels" << std::endl;
    // Keep the published surface alive until the Python reader has released it.
    std::string acknowledgement;
    return std::getline(std::cin, acknowledgement) && acknowledgement == "done" ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 3 && std::string(argv[1]) == "--send") {
        return send_pattern(argv[2]);
    }
    if (argc != 2) {
        std::cerr << "usage: nozzle_python_native_peer [--send] sender_name\n";
        return 2;
    }
    nozzle::receiver_desc desc{};
    desc.name = argv[1];
    auto created = nozzle::receiver::create(desc);
    if (!created.ok()) {
        std::cerr << created.error().message << '\n';
        return 1;
    }
    auto receiver = std::move(created.value());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        nozzle::acquire_desc acquire{};
        acquire.timeout_ms = 100;
        auto received = receiver.acquire_frame(acquire);
        if (!received.ok()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        auto frame = std::move(received.value());
        const auto info = frame.info();
        if (info.width != 65 || info.height != 3 ||
            info.semantic_format != nozzle::texture_format::rgba8_unorm) {
            std::cerr << "unexpected dimensions or semantic format\n";
            return 1;
        }
        auto locked = nozzle::lock_frame_pixels_mapping_with_origin(
            frame, nozzle::texture_origin::top_left);
        if (!locked.ok()) {
            std::cerr << locked.error().message << '\n';
            return 1;
        }
        const auto &pixels = locked.value().pixels();
        const bool bgra = pixels.format == nozzle::texture_format::bgra8_unorm;
        if (!bgra && pixels.format != nozzle::texture_format::rgba8_unorm) {
            std::cerr << "unexpected storage format\n";
            return 1;
        }
        std::cout << "native_storage=" << (bgra ? "BGRA8" : "RGBA8")
                  << " row_stride=" << pixels.row_stride_bytes << '\n';
        for (uint32_t y = 0; y < pixels.height; ++y) {
            const auto *row = static_cast<const uint8_t *>(pixels.data) +
                static_cast<std::ptrdiff_t>(y) * pixels.row_stride_bytes;
            for (uint32_t x = 0; x < pixels.width; ++x) {
                const uint8_t red = static_cast<uint8_t>(190 + y);
                const uint8_t green = static_cast<uint8_t>(20 + x);
                const uint8_t blue = static_cast<uint8_t>(40 + y);
                const uint8_t expected[4] = {
                    bgra ? blue : red, green, bgra ? red : blue, 149};
                for (uint32_t channel = 0; channel < 4; ++channel) {
                    if (row[4 * x + channel] != expected[channel]) {
                        std::cerr << "native_storage_mismatch x=" << x << " y=" << y
                                  << " channel=" << channel
                                  << " actual=" << unsigned(row[4 * x + channel])
                                  << " expected=" << unsigned(expected[channel]) << '\n';
                        return 1;
                    }
                }
            }
        }
        std::cout << "native_storage_verified=195_pixels\n";
        return 0;
    }
    std::cerr << "native receiver timed out without a frame\n";
    return 1;
}
