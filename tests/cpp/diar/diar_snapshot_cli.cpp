// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Test-only two-process state checkpoint utility. Snapshot files are created
// mode 0600 and atomically renamed into place on POSIX hosts.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "diarizer.h"
#include "fe.h"

using namespace nemo_speech::asr;

namespace {

constexpr size_t kMaxStateBytes = size_t{1} << 30;

std::vector<uint8_t>
read_private_state(const std::string& path) {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int fd = open(path.c_str(), flags);
    if (fd < 0) throw std::runtime_error("cannot open snapshot file");
    struct stat info {};
    if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0 ||
        static_cast<uint64_t>(info.st_size) > kMaxStateBytes) {
        close(fd);
        throw std::runtime_error("snapshot file type or size is invalid");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(info.st_size));
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t n = read(fd, bytes.data() + offset, bytes.size() - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            close(fd);
            throw std::runtime_error("snapshot file read failed");
        }
        offset += static_cast<size_t>(n);
    }
    close(fd);
    return bytes;
}

void
write_private_state_atomic(const std::string& path, const std::vector<uint8_t>& bytes) {
    if (bytes.size() > kMaxStateBytes) throw std::runtime_error("snapshot exceeds 1 GiB");
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string temp = path + ".tmp." + std::to_string(static_cast<long long>(getpid())) +
                             "." + std::to_string(static_cast<unsigned long long>(nonce));
    int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = open(temp.c_str(), flags, S_IRUSR | S_IWUSR);
    if (fd < 0) throw std::runtime_error("cannot create private snapshot temp file");
    size_t offset = 0;
    bool ok = true;
    while (offset < bytes.size()) {
        const ssize_t n = write(fd, bytes.data() + offset, bytes.size() - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            ok = false;
            break;
        }
        offset += static_cast<size_t>(n);
    }
    if (ok && fsync(fd) != 0) ok = false;
    if (close(fd) != 0) ok = false;
    if (!ok || rename(temp.c_str(), path.c_str()) != 0) {
        unlink(temp.c_str());
        throw std::runtime_error("snapshot write or atomic replace failed");
    }
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc != 5 || (std::string(argv[3]) != "--save-state" &&
                      std::string(argv[3]) != "--load-state")) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <audio.wav> --save-state|--load-state <state.bin>\n",
                     argv[0]);
        return 2;
    }
    try {
        std::vector<float> audio;
        int sample_rate = 0;
        if (!read_wav_mono_16k(argv[2], audio, sample_rate) || sample_rate <= 0)
            throw std::runtime_error("could not read mono WAV input");

        auto diarizer = Diarizer::load(-1, argv[1]);
        auto stream = diarizer->streaming_diarize();
        if (std::string(argv[3]) == "--load-state") {
            const auto state = read_private_state(argv[4]);
            stream->import_state(state.data(), state.size());
        }
        constexpr size_t kSamplesPer160ms = 2560;
        for (size_t offset = 0; offset < audio.size(); offset += kSamplesPer160ms) {
            const size_t count = std::min(kSamplesPer160ms, audio.size() - offset);
            stream->push(audio.data() + offset, count, sample_rate);
        }

        if (std::string(argv[3]) == "--save-state") {
            // Snapshot before finish(): finish flushes right-context and audio
            // tails, which would make a subsequent process a new final run.
            const auto state = stream->export_state();
            write_private_state_atomic(argv[4], state);
            std::fprintf(stderr, "saved %zu private bytes at frame %lld\n", state.size(),
                         static_cast<long long>(stream->frame_count()));
            return 0;
        }

        stream->finish();
        for (const auto& segment : stream->segments())
            std::printf("speaker=%d start=%.3f end=%.3f\n", segment.speaker + 1, segment.t0,
                        segment.t1);
        std::fprintf(stderr, "continued session through frame %lld (%.3f s)\n",
                     static_cast<long long>(stream->frame_count()),
                     stream->frame_count() * stream->seconds_per_frame());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "snapshot utility: %s\n", e.what());
        return 1;
    }
}
