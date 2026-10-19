// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Test-only AOSC profile interventions. This format parser intentionally lives
// under tests/ and is not linked into ordinary inference or the public C ABI.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace native_matching {

struct AoscProfile {
    std::vector<uint8_t> header;  // Stable 113-byte NMSAOSC1 header.
    int speakers = 0;
    int dimension = 0;
    int frames = 0;
    std::vector<float> cache;       // row-major [frames, dimension]
    std::vector<float> predictions; // row-major [frames, speakers]
    std::vector<float> silence;
};

inline uint32_t read_u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
           (uint32_t(p[3]) << 24);
}

inline uint64_t read_u64(const uint8_t* p) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) value = (value << 8) | p[i];
    return value;
}

inline void append_u32(std::vector<uint8_t>& out, uint32_t value) {
    const size_t offset = out.size();
    out.resize(offset + 4);
    for (int i = 0; i < 4; ++i) out[offset + static_cast<size_t>(i)] = static_cast<uint8_t>(value >> (8 * i));
}

inline void append_u64(std::vector<uint8_t>& out, uint64_t value) {
    const size_t offset = out.size();
    out.resize(offset + 8);
    for (int i = 0; i < 8; ++i) out[offset + static_cast<size_t>(i)] = static_cast<uint8_t>(value >> (8 * i));
}

inline void append_f32(std::vector<uint8_t>& out, float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "AOSC profile requires float32");
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(out, bits);
}

inline uint32_t crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return crc ^ 0xffffffffu;
}

inline std::vector<float> read_vector(const uint8_t* data, size_t size, size_t& pos, size_t expected) {
    if (size - pos < 8) throw std::invalid_argument("truncated AOSC profile vector length");
    const uint64_t count = read_u64(data + pos);
    pos += 8;
    if (count != expected || count > (size - pos) / sizeof(float))
        throw std::invalid_argument("AOSC profile vector dimensions are inconsistent");
    std::vector<float> out(static_cast<size_t>(count));
    for (float& value : out) {
        const uint32_t bits = read_u32(data + pos);
        pos += 4;
        std::memcpy(&value, &bits, sizeof(value));
    }
    return out;
}

inline AoscProfile parse_profile(const std::vector<uint8_t>& bytes) {
    constexpr size_t kHeaderSize = 113;
    constexpr uint8_t kMagic[8] = {'N', 'M', 'S', 'A', 'O', 'S', 'C', '1'};
    if (bytes.size() < kHeaderSize || !std::equal(kMagic, kMagic + 8, bytes.begin()))
        throw std::invalid_argument("invalid AOSC profile magic or truncated header");
    if (read_u32(bytes.data() + 8) != 1 || read_u32(bytes.data() + 12) != 3)
        throw std::invalid_argument("unsupported AOSC profile version or architecture");

    AoscProfile profile;
    profile.speakers = static_cast<int>(read_u32(bytes.data() + 52));
    profile.dimension = static_cast<int>(read_u32(bytes.data() + 56));
    profile.frames = static_cast<int32_t>(read_u32(bytes.data() + 88));
    if (profile.speakers <= 0 || profile.speakers > 8 || profile.dimension <= 0 ||
        profile.frames <= 0)
        throw std::invalid_argument("invalid AOSC profile tensor dimensions");
    const uint64_t payload_size = read_u64(bytes.data() + 101);
    if (payload_size != bytes.size() - kHeaderSize ||
        read_u32(bytes.data() + 109) != crc32(bytes.data() + kHeaderSize, bytes.size() - kHeaderSize))
        throw std::invalid_argument("AOSC profile checksum or payload size is invalid");

    profile.header.assign(bytes.begin(), bytes.begin() + kHeaderSize);
    size_t pos = kHeaderSize;
    profile.cache = read_vector(bytes.data(), bytes.size(), pos,
                                static_cast<size_t>(profile.frames) * profile.dimension);
    profile.predictions = read_vector(bytes.data(), bytes.size(), pos,
                                      static_cast<size_t>(profile.frames) * profile.speakers);
    profile.silence = read_vector(bytes.data(), bytes.size(), pos,
                                  static_cast<size_t>(profile.dimension));
    if (pos != bytes.size()) throw std::invalid_argument("AOSC profile contains trailing data");
    return profile;
}

inline std::vector<uint8_t> encode_profile(const AoscProfile& profile) {
    constexpr size_t kHeaderSize = 113;
    if (profile.header.size() != kHeaderSize || profile.speakers <= 0 || profile.dimension <= 0 ||
        profile.frames <= 0 || profile.cache.size() != static_cast<size_t>(profile.frames) * profile.dimension ||
        profile.predictions.size() != static_cast<size_t>(profile.frames) * profile.speakers ||
        profile.silence.size() != static_cast<size_t>(profile.dimension))
        throw std::invalid_argument("invalid AOSC profile intervention shape");
    std::vector<uint8_t> payload;
    payload.reserve(24 + (profile.cache.size() + profile.predictions.size() + profile.silence.size()) * 4);
    auto write_vector = [&](const std::vector<float>& values) {
        append_u64(payload, values.size());
        for (float value : values) append_f32(payload, value);
    };
    write_vector(profile.cache);
    write_vector(profile.predictions);
    write_vector(profile.silence);

    std::vector<uint8_t> out = profile.header;
    out.resize(kHeaderSize);
    const uint64_t payload_size = payload.size();
    for (int i = 0; i < 8; ++i) out[101 + i] = static_cast<uint8_t>(payload_size >> (i * 8));
    const uint32_t checksum = crc32(payload.data(), payload.size());
    for (int i = 0; i < 4; ++i) out[109 + i] = static_cast<uint8_t>(checksum >> (i * 8));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

inline std::vector<int> speech_rows(const AoscProfile& profile, int channel) {
    if (channel < 0 || channel >= profile.speakers)
        throw std::invalid_argument("AOSC intervention channel is out of range");
    std::vector<int> rows;
    for (int row = 0; row < profile.frames; ++row) {
        const float* p = profile.predictions.data() + static_cast<size_t>(row) * profile.speakers;
        // AOSC's compressor treats each channel independently and admits a
        // row when that channel prediction is > 0.5. Do not collapse rows to
        // their argmax channel: overlapping rows can support multiple columns.
        if (p[channel] > 0.5f) rows.push_back(row);
    }
    return rows;
}

inline std::vector<int> exclusive_speech_rows(
    const AoscProfile& profile, int channel, int excluded_channel) {
    if (channel < 0 || channel >= profile.speakers || excluded_channel < 0 ||
        excluded_channel >= profile.speakers || channel == excluded_channel)
        throw std::invalid_argument("AOSC neutral-control channel is invalid");
    std::vector<int> rows;
    for (int row = 0; row < profile.frames; ++row) {
        const float* p = profile.predictions.data() + static_cast<size_t>(row) * profile.speakers;
        if (p[channel] > 0.5f && p[excluded_channel] <= 0.5f) rows.push_back(row);
    }
    return rows;
}

inline std::vector<int> evenly_spaced_rows(const std::vector<int>& rows, size_t count) {
    if (count > rows.size()) throw std::invalid_argument("not enough AOSC rows for matched intervention");
    if (count == 0) return {};
    std::vector<int> selected;
    selected.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t index = ((2 * i + 1) * rows.size()) / (2 * count);
        selected.push_back(rows[std::min(index, rows.size() - 1)]);
    }
    return selected;
}

inline AoscProfile replace_with_silence(const AoscProfile& source, const std::vector<int>& rows) {
    AoscProfile changed = source;
    for (int row : rows) {
        if (row < 0 || row >= changed.frames) throw std::invalid_argument("AOSC row is out of range");
        std::copy(changed.silence.begin(), changed.silence.end(),
                  changed.cache.begin() + static_cast<size_t>(row) * changed.dimension);
        std::fill_n(changed.predictions.begin() + static_cast<size_t>(row) * changed.speakers,
                    changed.speakers, 0.0f);
    }
    return changed;
}

inline std::vector<uint8_t> ablate_channel(const std::vector<uint8_t>& bytes, int channel) {
    const AoscProfile profile = parse_profile(bytes);
    return encode_profile(replace_with_silence(profile, speech_rows(profile, channel)));
}

inline std::vector<uint8_t> ablate_channels(const std::vector<uint8_t>& bytes, const std::vector<int>& channels) {
    const AoscProfile profile = parse_profile(bytes);
    std::vector<int> rows;
    for (int channel : channels) {
        const auto selected = speech_rows(profile, channel);
        rows.insert(rows.end(), selected.begin(), selected.end());
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return encode_profile(replace_with_silence(profile, rows));
}

inline std::vector<uint8_t> ablate_matched_rows(
    const std::vector<uint8_t>& bytes, int source_channel, size_t count) {
    const AoscProfile profile = parse_profile(bytes);
    return encode_profile(replace_with_silence(
        profile, evenly_spaced_rows(speech_rows(profile, source_channel), count)));
}

inline std::vector<uint8_t> ablate_matched_rows_excluding(
    const std::vector<uint8_t>& bytes, int source_channel, int excluded_channel, size_t count) {
    const AoscProfile profile = parse_profile(bytes);
    return encode_profile(replace_with_silence(
        profile, evenly_spaced_rows(exclusive_speech_rows(profile, source_channel, excluded_channel), count)));
}

inline std::vector<uint8_t> reverse_cache_row_order(const std::vector<uint8_t>& bytes) {
    AoscProfile profile = parse_profile(bytes);
    AoscProfile reversed = profile;
    for (int row = 0; row < profile.frames; ++row) {
        const int source_row = profile.frames - 1 - row;
        std::copy_n(profile.cache.begin() + static_cast<size_t>(source_row) * profile.dimension,
                    profile.dimension,
                    reversed.cache.begin() + static_cast<size_t>(row) * profile.dimension);
        std::copy_n(profile.predictions.begin() + static_cast<size_t>(source_row) * profile.speakers,
                    profile.speakers,
                    reversed.predictions.begin() + static_cast<size_t>(row) * profile.speakers);
    }
    return encode_profile(reversed);
}

}  // namespace native_matching
