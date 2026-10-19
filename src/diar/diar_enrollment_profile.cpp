// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "diar_enrollment_profile.h"

#include "diarizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

using namespace nemo_speech::asr;

namespace {
constexpr uint8_t kMagic[8] = {'N', 'M', 'S', 'A', 'O', 'S', 'C', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kArchitecture = 3;  // Sortformer V3.
constexpr size_t kMaxProfileBytes = 64 * 1024 * 1024;

class Writer {
   public:
    std::vector<uint8_t> bytes;

    void raw(const void* data, size_t size) {
        if (size > kMaxProfileBytes - bytes.size())
            throw std::invalid_argument("AOSC profile exceeds the 64 MiB limit");
        const auto* p = static_cast<const uint8_t*>(data);
        bytes.insert(bytes.end(), p, p + size);
    }
    void u8(uint8_t x) { bytes.push_back(x); }
    void u32(uint32_t x) {
        for (int shift = 0; shift < 32; shift += 8) u8(static_cast<uint8_t>(x >> shift));
    }
    void i32(int32_t x) { u32(static_cast<uint32_t>(x)); }
    void u64(uint64_t x) {
        for (int shift = 0; shift < 64; shift += 8) u8(static_cast<uint8_t>(x >> shift));
    }
    void i64(int64_t x) { u64(static_cast<uint64_t>(x)); }
    void f32(float x) {
        uint32_t bits;
        static_assert(sizeof(bits) == sizeof(x), "float32 is required");
        std::memcpy(&bits, &x, sizeof(bits));
        u32(bits);
    }
    void f32_vector(const std::vector<float>& values) {
        u64(values.size());
        for (float x : values) f32(x);
    }
};

class Reader {
   public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {
        if (!data || size > kMaxProfileBytes || size < 113)
            throw std::invalid_argument("AOSC profile size is invalid");
    }
    void raw(void* dest, size_t n) {
        require(n);
        std::memcpy(dest, data_ + pos_, n);
        pos_ += n;
    }
    uint8_t u8() {
        require(1);
        return data_[pos_++];
    }
    uint32_t u32() {
        uint32_t x = 0;
        for (int shift = 0; shift < 32; shift += 8) x |= uint32_t(u8()) << shift;
        return x;
    }
    int32_t i32() { return static_cast<int32_t>(u32()); }
    uint64_t u64() {
        uint64_t x = 0;
        for (int shift = 0; shift < 64; shift += 8) x |= uint64_t(u8()) << shift;
        return x;
    }
    int64_t i64() { return static_cast<int64_t>(u64()); }
    float f32() {
        const uint32_t bits = u32();
        float x;
        std::memcpy(&x, &bits, sizeof(x));
        return x;
    }
    std::vector<float> f32_vector(size_t max_elements) {
        const uint64_t count = u64();
        if (count > max_elements || count > remaining() / sizeof(float))
            throw std::invalid_argument("AOSC profile vector size is invalid");
        std::vector<float> out(static_cast<size_t>(count));
        for (float& x : out) x = f32();
        return out;
    }
    size_t remaining() const { return size_ - pos_; }
    size_t position() const { return pos_; }
    const uint8_t* current() const { return data_ + pos_; }
    bool eof() const { return pos_ == size_; }

   private:
    void require(size_t n) {
        if (n > remaining()) throw std::invalid_argument("AOSC profile is truncated");
    }
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
};

uint32_t
crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; i++) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; bit++)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return crc ^ 0xffffffffu;
}

bool
all_finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
}

struct Profile {
    std::array<uint8_t, 32> fingerprint{};
    uint32_t sample_rate = 0;
    uint32_t n_spk = 0;
    uint32_t emb_dim = 0;
    uint32_t output_factor = 0;
    DiarGeometry geometry;
    int32_t cache_frames = 0;
    int64_t n_sil_frames = 0;
    bool learned_silence = false;
    std::vector<float> cache;
    std::vector<float> preds;
    std::vector<float> mean_silence;
};

Profile
read_profile(const uint8_t* data, size_t size) {
    Reader in(data, size);
    uint8_t magic[sizeof(kMagic)];
    in.raw(magic, sizeof(magic));
    if (!std::equal(std::begin(magic), std::end(magic), std::begin(kMagic)))
        throw std::invalid_argument("AOSC profile magic identifier is invalid");
    if (in.u32() != kVersion || in.u32() != kArchitecture)
        throw std::invalid_argument("AOSC profile version or architecture is unsupported");
    Profile p;
    in.raw(p.fingerprint.data(), p.fingerprint.size());
    p.sample_rate = in.u32();
    p.n_spk = in.u32();
    p.emb_dim = in.u32();
    p.output_factor = in.u32();
    p.geometry.spkcache_len = in.i32();
    p.geometry.fifo_len = in.i32();
    p.geometry.chunk_len = in.i32();
    p.geometry.spkcache_update_period = in.i32();
    p.geometry.chunk_left_context = in.i32();
    p.geometry.chunk_right_context = in.i32();
    p.cache_frames = in.i32();
    p.n_sil_frames = in.i64();
    const uint8_t learned = in.u8();
    if (learned > 1) throw std::invalid_argument("AOSC profile silence flag is invalid");
    p.learned_silence = learned != 0;
    if (p.sample_rate == 0 || p.n_spk == 0 || p.n_spk > 8 || p.emb_dim == 0 ||
        p.emb_dim > 8192 || p.output_factor == 0 || p.cache_frames <= 0 ||
        static_cast<uint64_t>(p.cache_frames) * p.emb_dim > kMaxProfileBytes / sizeof(float) ||
        static_cast<uint64_t>(p.cache_frames) * p.n_spk > kMaxProfileBytes / sizeof(float) ||
        p.n_sil_frames < 0)
        throw std::invalid_argument("AOSC profile header dimensions are invalid");
    const uint64_t payload_size = in.u64();
    const uint32_t checksum = in.u32();
    if (payload_size != in.remaining() || crc32(in.current(), in.remaining()) != checksum)
        throw std::invalid_argument("AOSC profile payload length or checksum is invalid");
    Reader payload(in.current(), in.remaining());
    const size_t cache_max = static_cast<size_t>(p.cache_frames) * p.emb_dim;
    const size_t preds_max = static_cast<size_t>(p.cache_frames) * p.n_spk;
    p.cache = payload.f32_vector(cache_max);
    p.preds = payload.f32_vector(preds_max);
    p.mean_silence = payload.f32_vector(p.emb_dim);
    if (!payload.eof()) throw std::invalid_argument("AOSC profile has trailing payload bytes");
    if (p.sample_rate == 0 || p.n_spk == 0 || p.emb_dim == 0 || p.output_factor == 0 ||
        p.cache_frames <= 0 || p.n_sil_frames < 0 ||
        p.cache.size() != cache_max || p.preds.size() != preds_max ||
        p.mean_silence.size() != p.emb_dim || !all_finite(p.cache) || !all_finite(p.preds) ||
        !all_finite(p.mean_silence))
        throw std::invalid_argument("AOSC profile state dimensions or values are invalid");
    for (float value : p.preds)
        if (value < 0.f || value > 1.f)
            throw std::invalid_argument("AOSC profile predictions are outside [0, 1]");
    return p;
}

bool
same_geometry(const DiarGeometry& a, const DiarGeometry& b) {
    return a.spkcache_len == b.spkcache_len && a.fifo_len == b.fifo_len &&
           a.chunk_len == b.chunk_len && a.spkcache_update_period == b.spkcache_update_period &&
           a.chunk_left_context == b.chunk_left_context &&
           a.chunk_right_context == b.chunk_right_context;
}

}  // namespace

std::vector<uint8_t>
DiarAoscProfileCodec::encode(const DiarizationStream& wrapper) {
    std::lock_guard<std::mutex> lock(wrapper.operation_mutex_);
    if (!wrapper.model_->cfg().is_v3())
        throw std::invalid_argument("AOSC enrollment profile currently supports Sortformer V3 only");
    const auto& stream = *wrapper.stream_;
    const auto& state = stream.state_;
    if (state.spk_frames_ <= 0 || state.spk_frames_ != stream.geo_.spkcache_len ||
        !state.spkcache_preds_valid())
        throw std::invalid_argument(
            "AOSC profile is not ready: the cache has not completed its first compression; "
            "enroll enough audio to exceed FIFO plus speaker-cache capacity");
    if (state.spkcache_.size() != static_cast<size_t>(state.spk_frames_) * state.emb_dim_ ||
        state.spkcache_preds_.size() != static_cast<size_t>(state.spk_frames_) * state.n_spk_ ||
        state.mean_sil_emb_.size() != static_cast<size_t>(state.emb_dim_) ||
        state.n_sil_frames_ < 0 || !all_finite(state.spkcache_) || !all_finite(state.spkcache_preds_) ||
        !all_finite(state.mean_sil_emb_))
        throw std::invalid_argument("AOSC profile state dimensions or values are invalid");
    for (float value : state.spkcache_preds_)
        if (value < 0.f || value > 1.f)
            throw std::invalid_argument("AOSC profile predictions are outside [0, 1]");

    Writer payload;
    payload.f32_vector(state.spkcache_);
    payload.f32_vector(state.spkcache_preds_);
    payload.f32_vector(state.mean_sil_emb_);

    const auto& cfg = wrapper.model_->cfg();
    Writer out;
    out.raw(kMagic, sizeof(kMagic));
    out.u32(kVersion);
    out.u32(kArchitecture);
    out.raw(wrapper.model_->model_fingerprint().data(), wrapper.model_->model_fingerprint().size());
    out.u32(static_cast<uint32_t>(cfg.sample_rate));
    out.u32(static_cast<uint32_t>(cfg.num_speakers));
    out.u32(static_cast<uint32_t>(cfg.encoder.d_model));
    out.u32(static_cast<uint32_t>(cfg.output_subsampling_factor));
    out.i32(stream.geo_.spkcache_len);
    out.i32(stream.geo_.fifo_len);
    out.i32(stream.geo_.chunk_len);
    out.i32(stream.geo_.spkcache_update_period);
    out.i32(stream.geo_.chunk_left_context);
    out.i32(stream.geo_.chunk_right_context);
    out.i32(state.spk_frames_);
    out.i64(state.n_sil_frames_);
    out.u8(state.use_learned_silence_ ? 1 : 0);
    out.u64(payload.bytes.size());
    out.u32(crc32(payload.bytes.data(), payload.bytes.size()));
    out.raw(payload.bytes.data(), payload.bytes.size());
    return out.bytes;
}

void
DiarAoscProfileCodec::decode(DiarizationStream& wrapper, const uint8_t* data, size_t size) {
    std::lock_guard<std::mutex> lock(wrapper.operation_mutex_);
    const auto& current = *wrapper.stream_;
    if (wrapper.state_imported_ || wrapper.input_sample_rate_ != 0 || wrapper.resampler_ ||
        current.finished_ || current.n_frames() != 0 || !current.audio_buf_.empty() ||
        current.audio_base_ != 0 || !current.mel_buf_.empty() || current.mel_base_ != 0 ||
        current.mel_consumed_ != 0 || current.provisional_frames_ != 0 ||
        !current.probs_.empty() || current.probs_base_ != 0 || !current.frozen_segs_.empty() ||
        current.state_.spk_frames_ != 0 || current.state_.fifo_frames_ != 0 ||
        current.birth_gate_.frame_ != 0)
        throw std::invalid_argument("AOSC profile import requires a fresh stream");
    const Profile p = read_profile(data, size);
    const auto& cfg = wrapper.model_->cfg();
    const auto& stream = *wrapper.stream_;
    const auto& expected_fingerprint = wrapper.model_->model_fingerprint();
    if (p.fingerprint != expected_fingerprint || p.sample_rate != static_cast<uint32_t>(cfg.sample_rate) ||
        p.n_spk != static_cast<uint32_t>(cfg.num_speakers) ||
        p.emb_dim != static_cast<uint32_t>(cfg.encoder.d_model) ||
        p.output_factor != static_cast<uint32_t>(cfg.output_subsampling_factor) ||
        !same_geometry(p.geometry, stream.geo_))
        throw std::invalid_argument("AOSC profile model or streaming geometry is incompatible");
    if (p.learned_silence != stream.state_.use_learned_silence_ ||
        p.cache_frames != stream.geo_.spkcache_len)
        throw std::invalid_argument("AOSC profile state is incompatible with this model");

    auto& state = wrapper.stream_->state_;
    state.spk_frames_ = p.cache_frames;
    state.spkcache_ = p.cache;
    state.spkcache_preds_ = p.preds;
    state.mean_sil_emb_ = p.mean_silence;
    state.n_sil_frames_ = p.n_sil_frames;
    wrapper.state_imported_ = true;
}

AoscProfileSummary
DiarAoscProfileCodec::summarize(const std::vector<uint8_t>& bytes) {
    const Profile p = read_profile(bytes.data(), bytes.size());
    AoscProfileSummary summary;
    summary.speaker_channels = static_cast<int>(p.n_spk);
    summary.embedding_dim = static_cast<int>(p.emb_dim);
    summary.cache_frames = p.cache_frames;
    summary.channel_centroid_frames.assign(p.n_spk, 0);
    for (int f = 0; f < p.cache_frames; f++) {
        const float* row = p.preds.data() + static_cast<size_t>(f) * p.n_spk;
        int best = 0;
        for (int s = 1; s < static_cast<int>(p.n_spk); s++)
            if (row[s] > row[best]) best = s;
        // AOSC only treats >0.5 predictions as speech when selecting cache
        // rows. This same model-defined mask excludes its silence sentinels.
        if (row[best] > 0.5f) summary.channel_centroid_frames[best]++;
    }
    return summary;
}

std::vector<float>
DiarAoscProfileCodec::cosine_scores(
    const std::vector<uint8_t>& bytes, const float* embeddings, int n_frames, int embedding_dim) {
    const Profile p = read_profile(bytes.data(), bytes.size());
    if ((!embeddings && n_frames > 0) || n_frames < 0 || embedding_dim != static_cast<int>(p.emb_dim))
        throw std::invalid_argument("AOSC cosine probe embedding shape is invalid");
    const int n_spk = static_cast<int>(p.n_spk);
    const int dim = static_cast<int>(p.emb_dim);
    std::vector<double> sums(static_cast<size_t>(n_spk) * dim, 0.0);
    std::vector<int> counts(n_spk, 0);
    for (int f = 0; f < p.cache_frames; f++) {
        const float* row = p.preds.data() + static_cast<size_t>(f) * n_spk;
        int best = 0;
        for (int s = 1; s < n_spk; s++)
            if (row[s] > row[best]) best = s;
        if (row[best] <= 0.5f) continue;
        counts[best]++;
        const float* emb = p.cache.data() + static_cast<size_t>(f) * dim;
        double* sum = sums.data() + static_cast<size_t>(best) * dim;
        for (int d = 0; d < dim; d++) sum[d] += emb[d];
    }

    std::vector<double> norms(n_spk, 0.0);
    for (int s = 0; s < n_spk; s++) {
        if (counts[s] == 0) continue;
        double* sum = sums.data() + static_cast<size_t>(s) * dim;
        for (int d = 0; d < dim; d++) {
            sum[d] /= counts[s];
            norms[s] += sum[d] * sum[d];
        }
        norms[s] = std::sqrt(norms[s]);
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> out(static_cast<size_t>(n_frames) * n_spk, nan);
    for (int f = 0; f < n_frames; f++) {
        const float* emb = embeddings + static_cast<size_t>(f) * dim;
        double emb_norm = 0.0;
        for (int d = 0; d < dim; d++) emb_norm += static_cast<double>(emb[d]) * emb[d];
        emb_norm = std::sqrt(emb_norm);
        if (!std::isfinite(emb_norm) || emb_norm <= 1e-12) continue;
        for (int s = 0; s < n_spk; s++) {
            if (counts[s] == 0 || norms[s] <= 1e-12) continue;
            const double* centroid = sums.data() + static_cast<size_t>(s) * dim;
            double dot = 0.0;
            for (int d = 0; d < dim; d++) dot += centroid[d] * emb[d];
            out[static_cast<size_t>(f) * n_spk + s] =
                static_cast<float>(dot / (norms[s] * emb_norm));
        }
    }
    return out;
}
