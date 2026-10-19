// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "diar_snapshot.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "audio_resampler.h"
#include "diar_pipeline.h"
#include "diarizer.h"

namespace nemo_speech::asr {
namespace {

constexpr uint8_t kMagic[] = {'N', 'M', 'S', 'D', 'S', 'T', '0', '1'};
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kArchitectureVersion = 3;  // Nemotron 3 / Sortformer V3.
constexpr uint64_t kMaxSnapshotBytes = 1024ULL * 1024ULL * 1024ULL;
constexpr uint64_t kMaxArrayElements = kMaxSnapshotBytes / sizeof(float);
static_assert(std::numeric_limits<float>::is_iec559, "snapshot format requires IEEE-754 float32");
static_assert(std::numeric_limits<double>::is_iec559, "snapshot format requires IEEE-754 float64");

class Writer {
   public:
    std::vector<uint8_t> bytes;

    void u8(uint8_t v) { bytes.push_back(v); }
    void u32(uint32_t v) {
        for (unsigned i = 0; i < 4; ++i) u8(static_cast<uint8_t>(v >> (8 * i)));
    }
    void u64(uint64_t v) {
        for (unsigned i = 0; i < 8; ++i) u8(static_cast<uint8_t>(v >> (8 * i)));
    }
    void i32(int32_t v) { u32(static_cast<uint32_t>(v)); }
    void i64(int64_t v) { u64(static_cast<uint64_t>(v)); }
    void f32(float v) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(v), "float32 required");
        std::memcpy(&bits, &v, sizeof(bits));
        u32(bits);
    }
    void f64(double v) {
        uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(v), "float64 required");
        std::memcpy(&bits, &v, sizeof(bits));
        u64(bits);
    }
    void raw(const uint8_t* p, size_t n) {
        if (n) bytes.insert(bytes.end(), p, p + n);
    }
    void f32_vector(const std::vector<float>& v) {
        u64(v.size());
        for (float x : v) f32(x);
    }
    void i32_vector(const std::vector<int>& v) {
        u64(v.size());
        for (int x : v) i32(x);
    }
    void i64_vector(const std::vector<int64_t>& v) {
        u64(v.size());
        for (int64_t x : v) i64(x);
    }
    void u8_vector(const std::vector<uint8_t>& v) {
        u64(v.size());
        raw(v.data(), v.size());
    }
};

class Reader {
   public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    size_t remaining() const { return size_ - offset_; }
    const uint8_t* current() const { return data_ + offset_; }
    void require_end() const {
        if (remaining() != 0)
            throw std::invalid_argument("snapshot contains trailing data");
    }
    uint8_t u8() {
        require(1);
        return data_[offset_++];
    }
    uint32_t u32() {
        uint32_t v = 0;
        for (unsigned i = 0; i < 4; ++i) v |= static_cast<uint32_t>(u8()) << (8 * i);
        return v;
    }
    uint64_t u64() {
        uint64_t v = 0;
        for (unsigned i = 0; i < 8; ++i) v |= static_cast<uint64_t>(u8()) << (8 * i);
        return v;
    }
    int32_t i32() { return static_cast<int32_t>(u32()); }
    int64_t i64() { return static_cast<int64_t>(u64()); }
    float f32() {
        const uint32_t bits = u32();
        float v = 0.f;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    double f64() {
        const uint64_t bits = u64();
        double v = 0.0;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    void raw(uint8_t* out, size_t n) {
        require(n);
        if (n) std::memcpy(out, data_ + offset_, n);
        offset_ += n;
    }
    std::vector<float> f32_vector(uint64_t max_count = kMaxArrayElements) {
        const uint64_t count = vector_count(max_count, sizeof(float));
        std::vector<float> v(static_cast<size_t>(count));
        for (float& x : v) x = f32();
        return v;
    }
    std::vector<int> i32_vector(uint64_t max_count = kMaxArrayElements) {
        const uint64_t count = vector_count(max_count, sizeof(int32_t));
        std::vector<int> v(static_cast<size_t>(count));
        for (int& x : v) x = i32();
        return v;
    }
    std::vector<int64_t> i64_vector(uint64_t max_count = kMaxArrayElements) {
        const uint64_t count = vector_count(max_count, sizeof(int64_t));
        std::vector<int64_t> v(static_cast<size_t>(count));
        for (int64_t& x : v) x = i64();
        return v;
    }
    std::vector<uint8_t> u8_vector(uint64_t max_count = kMaxArrayElements) {
        const uint64_t count = vector_count(max_count, sizeof(uint8_t));
        std::vector<uint8_t> v(static_cast<size_t>(count));
        raw(v.data(), v.size());
        return v;
    }

   private:
    void require(size_t n) const {
        if (n > remaining())
            throw std::invalid_argument("snapshot is truncated");
    }
    uint64_t vector_count(uint64_t max_count, size_t element_size) {
        const uint64_t count = u64();
        if (count > max_count || count > std::numeric_limits<size_t>::max() ||
            count > remaining() / element_size)
            throw std::invalid_argument("snapshot array length is invalid");
        return count;
    }

    const uint8_t* data_;
    size_t size_;
    size_t offset_ = 0;
};

uint32_t
crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xffffffffU;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return ~crc;
}

// Self-contained SHA-256 for hashing the actual GGUF bytes. This avoids an
// optional platform crypto dependency and hashes the weights plus metadata.
class Sha256 {
   public:
    void update(const uint8_t* data, size_t size) {
        total_bytes_ += size;
        while (size) {
            const size_t take = std::min(size, block_.size() - used_);
            std::memcpy(block_.data() + used_, data, take);
            used_ += take;
            data += take;
            size -= take;
            if (used_ == block_.size()) {
                transform(block_.data());
                used_ = 0;
            }
        }
    }
    std::array<uint8_t, 32> finish() {
        const uint64_t bit_count = total_bytes_ * 8;
        block_[used_++] = 0x80;
        if (used_ > 56) {
            std::fill(block_.begin() + used_, block_.end(), 0);
            transform(block_.data());
            used_ = 0;
        }
        std::fill(block_.begin() + used_, block_.begin() + 56, 0);
        for (unsigned i = 0; i < 8; ++i)
            block_[63 - i] = static_cast<uint8_t>(bit_count >> (8 * i));
        transform(block_.data());
        std::array<uint8_t, 32> digest{};
        for (size_t i = 0; i < state_.size(); ++i)
            for (unsigned j = 0; j < 4; ++j)
                digest[i * 4 + j] = static_cast<uint8_t>(state_[i] >> (24 - 8 * j));
        return digest;
    }

   private:
    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
    void transform(const uint8_t* block) {
        static constexpr uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
                   (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(block[i * 4 + 2]) << 8) | block[i * 4 + 3];
        for (unsigned i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (unsigned i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<uint8_t, 64> block_{};
    size_t used_ = 0;
    uint64_t total_bytes_ = 0;
};

void
write_geometry(Writer& out, const DiarGeometry& geo) {
    out.i32(geo.spkcache_len);
    out.i32(geo.fifo_len);
    out.i32(geo.chunk_len);
    out.i32(geo.spkcache_update_period);
    out.i32(geo.chunk_left_context);
    out.i32(geo.chunk_right_context);
}

bool
same_geometry(const DiarGeometry& a, const DiarGeometry& b) {
    return a.spkcache_len == b.spkcache_len && a.fifo_len == b.fifo_len &&
           a.chunk_len == b.chunk_len && a.spkcache_update_period == b.spkcache_update_period &&
           a.chunk_left_context == b.chunk_left_context &&
           a.chunk_right_context == b.chunk_right_context;
}

void
write_resampler(Writer& out, const audio::AudioResampler::State& s) {
    out.i32(s.input_rate);
    out.i32(s.output_rate);
    out.u64(s.rate_gcd);
    out.u64(s.phase_count);
    out.u8(s.exact_phase_table ? 1 : 0);
    out.f32_vector(s.coefficients);
    out.f32_vector(s.input);
    out.u64(s.input_start);
    out.u64(s.total_input);
    out.u64(s.next_output);
    out.u64(s.source_index);
    out.u64(s.source_remainder);
    out.u8(s.finished ? 1 : 0);
}

audio::AudioResampler::State
read_resampler(Reader& in) {
    audio::AudioResampler::State s;
    s.input_rate = in.i32();
    s.output_rate = in.i32();
    s.rate_gcd = in.u64();
    const uint64_t phases = in.u64();
    if (phases > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("snapshot resampler phase table is too large");
    s.phase_count = static_cast<size_t>(phases);
    const uint8_t exact = in.u8();
    if (exact > 1) throw std::invalid_argument("snapshot resampler flag is invalid");
    s.exact_phase_table = exact != 0;
    s.coefficients = in.f32_vector(2048ULL * 32ULL);
    s.input = in.f32_vector();
    s.input_start = in.u64();
    s.total_input = in.u64();
    s.next_output = in.u64();
    s.source_index = in.u64();
    s.source_remainder = in.u64();
    const uint8_t finished = in.u8();
    if (finished > 1) throw std::invalid_argument("snapshot resampler finished flag is invalid");
    s.finished = finished != 0;
    return s;
}

}  // namespace

class DiarSnapshotAccess {
   public:
    static void write_stream_payload(Writer& out, const DiarizationStream& wrapper);
    static void validate_aosc(const AoscState& state);
    static void validate_birth_gate(const ChannelBirthGate& gate);
    static void validate_stream_state(
        const DiarizationStream& wrapper, const DiarStream& stream, int input_sample_rate,
        bool has_resampler, const audio::AudioResampler::State* resampler_state);
    static bool is_fresh(const DiarizationStream& wrapper);
    static void read_stream_payload(
        Reader& in, DiarizationStream& destination,
        std::unique_ptr<audio::AudioResampler>& restored_resampler, int& restored_input_rate);
};

void
DiarSnapshotAccess::write_stream_payload(Writer& out, const DiarizationStream& wrapper) {
    const auto& stream = *wrapper.stream_;
    out.i32(wrapper.input_sample_rate_);
    out.u8(wrapper.resampler_ ? 1 : 0);
    if (wrapper.resampler_) write_resampler(out, wrapper.resampler_->export_state());

    const auto& aosc = stream.state_;
    out.i32(aosc.spk_frames_);
    out.f32_vector(aosc.spkcache_);
    out.f32_vector(aosc.spkcache_preds_);
    out.i32(aosc.fifo_frames_);
    out.f32_vector(aosc.fifo_);
    out.f32_vector(aosc.mean_sil_emb_);
    out.i64(aosc.n_sil_frames_);
    out.u8(aosc.use_learned_silence_ ? 1 : 0);

    const auto& gate = stream.birth_gate_;
    out.i64(gate.frame_);
    out.u8_vector(gate.established_);
    out.i32_vector(gate.clean_frames_);
    out.i32_vector(gate.fading_frames_);
    out.i64_vector(gate.last_win_);
    out.f32_vector(gate.raw_ring_);

    out.u64(stream.audio_base_);
    out.f32_vector(stream.audio_buf_);
    out.i64(stream.mel_base_);
    out.f32_vector(stream.mel_buf_);
    out.i64(stream.mel_consumed_);
    out.i64(stream.provisional_frames_);
    out.u8(stream.finished_ ? 1 : 0);
    out.f32_vector(stream.probs_);
    out.i64(stream.probs_base_);
    out.u64(stream.frozen_segs_.size());
    for (const auto& segment : stream.frozen_segs_) {
        out.f64(segment.t0);
        out.f64(segment.t1);
        out.i32(segment.speaker);
    }
    out.i64(stream.compact_trigger_frames_);
    out.i64(stream.compact_retain_frames_);
    out.i64(stream.hold_from_);
}

static bool
all_finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
}

static void
validate_probability_vector(const std::vector<float>& values, const char* name) {
    for (float x : values)
        if (!std::isfinite(x) || x < 0.f || x > 1.f)
            throw std::invalid_argument(std::string("snapshot ") + name + " has invalid values");
}

void
DiarSnapshotAccess::validate_aosc(const AoscState& state) {
    const size_t expected_cache = static_cast<size_t>(state.spk_frames_) * state.emb_dim_;
    const size_t expected_fifo = static_cast<size_t>(state.fifo_frames_) * state.emb_dim_;
    if (state.spk_frames_ < 0 || state.spk_frames_ > state.geo_.spkcache_len ||
        state.fifo_frames_ < 0 || state.fifo_frames_ > state.geo_.fifo_len ||
        state.spkcache_.size() != expected_cache || state.fifo_.size() != expected_fifo ||
        state.mean_sil_emb_.size() != static_cast<size_t>(state.emb_dim_) ||
        state.n_sil_frames_ < 0 ||
        (!state.spkcache_preds_.empty() &&
         state.spkcache_preds_.size() != static_cast<size_t>(state.spk_frames_) * state.n_spk_))
        throw std::invalid_argument("snapshot AOSC dimensions are invalid");
    if (!all_finite(state.spkcache_) || !all_finite(state.fifo_) ||
        !all_finite(state.mean_sil_emb_))
        throw std::invalid_argument("snapshot AOSC embeddings have invalid values");
    validate_probability_vector(state.spkcache_preds_, "AOSC predictions");
}

void
DiarSnapshotAccess::validate_birth_gate(const ChannelBirthGate& gate) {
    const size_t n = static_cast<size_t>(gate.n_spk_);
    const size_t ring_values = gate.raw_ring_.size();
    if (gate.frame_ < 0 || gate.established_.size() != n || gate.clean_frames_.size() != n ||
        gate.fading_frames_.size() != n || gate.last_win_.size() != n ||
        ring_values > static_cast<size_t>(ChannelBirthGate::revision_frames) * n ||
        ring_values % n != 0)
        throw std::invalid_argument("snapshot channel gate dimensions are invalid");
    for (uint8_t v : gate.established_)
        if (v > 1) throw std::invalid_argument("snapshot channel gate flags are invalid");
    for (int v : gate.clean_frames_)
        if (v < 0 || v > gate.frame_)
            throw std::invalid_argument("snapshot channel gate counters are invalid");
    for (int v : gate.fading_frames_)
        if (v < 0 || v > gate.frame_)
            throw std::invalid_argument("snapshot channel gate counters are invalid");
    const int64_t never_won = std::numeric_limits<int64_t>::min() / 2;
    for (int64_t v : gate.last_win_)
        if (v != never_won && (v < 0 || v > gate.frame_))
            throw std::invalid_argument("snapshot channel gate timestamps are invalid");
    validate_probability_vector(gate.raw_ring_, "channel gate history");
}

void
DiarSnapshotAccess::validate_stream_state(
    const DiarizationStream& wrapper, const DiarStream& stream, int input_sample_rate,
    bool has_resampler, const audio::AudioResampler::State* resampler_state) {
    const int model_rate = wrapper.model_->cfg().sample_rate;
    if (input_sample_rate != 0 && !audio::supported_input_sample_rate(input_sample_rate))
        throw std::invalid_argument("snapshot input sample rate is invalid");
    if (has_resampler) {
        if (!resampler_state || stream.finished_ || input_sample_rate == 0 ||
            input_sample_rate == model_rate || resampler_state->input_rate != input_sample_rate ||
            resampler_state->output_rate != model_rate || resampler_state->finished)
            throw std::invalid_argument("snapshot resampler does not match stream state");
    } else if (!stream.finished_ && input_sample_rate != 0 && input_sample_rate != model_rate) {
        throw std::invalid_argument("snapshot is missing the active resampler");
    }
    if (stream.finished_ && has_resampler)
        throw std::invalid_argument("finished snapshot must not retain a resampler");

    const int n_spk = stream.n_spk_;
    if (stream.audio_base_ > std::numeric_limits<uint64_t>::max() - stream.audio_buf_.size() ||
        stream.mel_base_ < 0 || stream.mel_consumed_ < 0 ||
        stream.mel_buf_.size() % static_cast<size_t>(stream.n_mels_) != 0 ||
        stream.mel_buf_.size() / static_cast<size_t>(stream.n_mels_) >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - stream.mel_base_) ||
        stream.mel_base_ > stream.mel_consumed_ ||
        stream.mel_produced() < stream.mel_consumed_ || stream.provisional_frames_ < 0 ||
        (stream.finished_ && stream.provisional_frames_ != 0) ||
        stream.probs_base_ < 0 || stream.probs_.size() % static_cast<size_t>(n_spk) != 0 ||
        stream.compact_retain_frames_ <= 0 ||
        stream.compact_trigger_frames_ <= stream.compact_retain_frames_ ||
        (stream.hold_from_ != std::numeric_limits<int64_t>::max() && stream.hold_from_ < 0))
        throw std::invalid_argument("snapshot stream counters or buffer shapes are invalid");
    const uint64_t retained_frames = stream.probs_.size() / static_cast<size_t>(n_spk);
    if (retained_frames > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        static_cast<uint64_t>(stream.provisional_frames_) > retained_frames ||
        stream.probs_base_ > std::numeric_limits<int64_t>::max() -
                                  static_cast<int64_t>(retained_frames))
        throw std::invalid_argument("snapshot probability timeline is invalid");
    if (!all_finite(stream.audio_buf_) || !all_finite(stream.mel_buf_))
        throw std::invalid_argument("snapshot audio or mel buffer has invalid values");
    validate_probability_vector(stream.probs_, "frame probabilities");
    for (const auto& segment : stream.frozen_segs_)
        if (!std::isfinite(segment.t0) || !std::isfinite(segment.t1) || segment.t0 < 0.0 ||
            segment.t1 < segment.t0 || segment.speaker < 0 || segment.speaker >= n_spk)
            throw std::invalid_argument("snapshot frozen segments are invalid");
    validate_aosc(stream.state_);
    if (stream.state_.use_learned_silence_ !=
        !wrapper.model_->model().learnable_silence_embedding().empty())
        throw std::invalid_argument("snapshot AOSC silence mode is incompatible");
    validate_birth_gate(stream.birth_gate_);
}

bool
DiarSnapshotAccess::is_fresh(const DiarizationStream& wrapper) {
    const auto& s = *wrapper.stream_;
    return !wrapper.state_imported_ && wrapper.input_sample_rate_ == 0 && !wrapper.resampler_ && !s.finished_ &&
           s.n_frames() == 0 && s.audio_buf_.empty() && s.audio_base_ == 0 &&
           s.mel_buf_.empty() && s.mel_base_ == 0 && s.mel_consumed_ == 0 &&
           s.provisional_frames_ == 0 && s.probs_.empty() && s.probs_base_ == 0 &&
           s.frozen_segs_.empty() && s.state_.spk_frames_ == 0 && s.state_.fifo_frames_ == 0 &&
           s.birth_gate_.frame_ == 0;
}

void
DiarSnapshotAccess::read_stream_payload(
    Reader& in, DiarizationStream& destination,
    std::unique_ptr<audio::AudioResampler>& restored_resampler, int& restored_input_rate) {
    restored_input_rate = in.i32();
    const uint8_t resampler_flag = in.u8();
    if (resampler_flag > 1) throw std::invalid_argument("snapshot resampler flag is invalid");
    audio::AudioResampler::State resampler_state;
    const bool has_resampler = resampler_flag != 0;
    if (has_resampler) {
        resampler_state = read_resampler(in);
        restored_resampler = std::make_unique<audio::AudioResampler>(
            resampler_state.input_rate, resampler_state.output_rate);
        restored_resampler->import_state(resampler_state);
    }

    // Construct a detached candidate and populate it completely before the
    // destination's state pointer is changed.
    DiarStream candidate(*destination.model_, destination.stream_->geo_);
    auto& aosc = candidate.state_;
    aosc.spk_frames_ = in.i32();
    aosc.spkcache_ = in.f32_vector();
    aosc.spkcache_preds_ = in.f32_vector();
    aosc.fifo_frames_ = in.i32();
    aosc.fifo_ = in.f32_vector();
    aosc.mean_sil_emb_ = in.f32_vector();
    aosc.n_sil_frames_ = in.i64();
    const uint8_t learned = in.u8();
    if (learned > 1) throw std::invalid_argument("snapshot AOSC silence flag is invalid");
    if ((learned != 0) != aosc.use_learned_silence_)
        throw std::invalid_argument("snapshot AOSC silence mode is incompatible");

    auto& gate = candidate.birth_gate_;
    gate.frame_ = in.i64();
    gate.established_ = in.u8_vector(8);
    gate.clean_frames_ = in.i32_vector(8);
    gate.fading_frames_ = in.i32_vector(8);
    gate.last_win_ = in.i64_vector(8);
    gate.raw_ring_ = in.f32_vector(
        static_cast<uint64_t>(ChannelBirthGate::revision_frames) * gate.n_spk_);

    const uint64_t audio_base = in.u64();
    if (audio_base > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("snapshot audio offset is too large");
    candidate.audio_base_ = static_cast<size_t>(audio_base);
    candidate.audio_buf_ = in.f32_vector();
    candidate.mel_base_ = in.i64();
    candidate.mel_buf_ = in.f32_vector();
    candidate.mel_consumed_ = in.i64();
    candidate.provisional_frames_ = in.i64();
    const uint8_t finished = in.u8();
    if (finished > 1) throw std::invalid_argument("snapshot finished flag is invalid");
    candidate.finished_ = finished != 0;
    candidate.probs_ = in.f32_vector();
    candidate.probs_base_ = in.i64();
    const uint64_t segment_count = in.u64();
    if (segment_count > kMaxArrayElements || segment_count > in.remaining() / 20)
        throw std::invalid_argument("snapshot frozen segment count is invalid");
    candidate.frozen_segs_.resize(static_cast<size_t>(segment_count));
    for (auto& segment : candidate.frozen_segs_) {
        segment.t0 = in.f64();
        segment.t1 = in.f64();
        segment.speaker = in.i32();
    }
    candidate.compact_trigger_frames_ = in.i64();
    candidate.compact_retain_frames_ = in.i64();
    candidate.hold_from_ = in.i64();
    in.require_end();

    validate_stream_state(
        destination, candidate, restored_input_rate, has_resampler,
        has_resampler ? &resampler_state : nullptr);

    // Allocate before committing so an allocation failure cannot leave a
    // half-restored destination stream.
    auto restored_stream = std::make_unique<DiarStream>(std::move(candidate));
    // Commit operations are moves of fully validated candidates.
    destination.input_sample_rate_ = restored_input_rate;
    destination.state_imported_ = true;
    destination.resampled_.clear();  // transient output from the previous push
    destination.resampler_ = std::move(restored_resampler);
    destination.stream_ = std::move(restored_stream);
}

std::array<uint8_t, 32>
DiarStreamSnapshotCodec::fingerprint_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read model file for snapshot fingerprint");
    Sha256 hash;
    std::array<uint8_t, 1 << 16> buffer{};
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        const auto count = input.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<size_t>(count));
    }
    if (!input.eof()) throw std::runtime_error("failed reading model for snapshot fingerprint");
    return hash.finish();
}

std::vector<uint8_t>
DiarStreamSnapshotCodec::encode(const DiarizationStream& wrapper) {
    Writer payload;
    DiarSnapshotAccess::write_stream_payload(payload, wrapper);
    if (payload.bytes.size() > kMaxSnapshotBytes)
        throw std::invalid_argument("stream state exceeds the 1 GiB snapshot limit");

    const auto& cfg = wrapper.model_->cfg();
    Writer out;
    out.raw(kMagic, sizeof(kMagic));
    out.u32(kFormatVersion);
    out.u32(kArchitectureVersion);
    out.raw(wrapper.model_->model_fingerprint().data(), wrapper.model_->model_fingerprint().size());
    out.u32(static_cast<uint32_t>(cfg.sample_rate));
    out.u32(static_cast<uint32_t>(cfg.num_speakers));
    out.u32(static_cast<uint32_t>(cfg.n_mels));
    out.u32(static_cast<uint32_t>(cfg.n_fft));
    out.u32(static_cast<uint32_t>(wrapper.model_->fe().hop_length()));
    out.u32(static_cast<uint32_t>(cfg.output_subsampling_factor));
    write_geometry(out, wrapper.stream_->geo_);
    out.u64(payload.bytes.size());
    out.u32(crc32(payload.bytes.data(), payload.bytes.size()));
    out.raw(payload.bytes.data(), payload.bytes.size());
    if (out.bytes.size() > kMaxSnapshotBytes)
        throw std::invalid_argument("stream state exceeds the 1 GiB snapshot limit");
    return out.bytes;
}

void
DiarStreamSnapshotCodec::decode(DiarizationStream& wrapper, const uint8_t* data, size_t size) {
    if (!data) throw std::invalid_argument("snapshot data must not be NULL");
    if (size > kMaxSnapshotBytes || size < 108)
        throw std::invalid_argument("snapshot size is invalid");
    if (!DiarSnapshotAccess::is_fresh(wrapper))
        throw std::invalid_argument("snapshot import requires a fresh stream");

    Reader in(data, size);
    uint8_t magic[sizeof(kMagic)];
    in.raw(magic, sizeof(magic));
    if (!std::equal(std::begin(magic), std::end(magic), std::begin(kMagic)))
        throw std::invalid_argument("snapshot magic identifier is invalid");
    const uint32_t version = in.u32();
    if (version != kFormatVersion)
        throw std::invalid_argument("unsupported snapshot format version");
    const uint32_t architecture = in.u32();
    if (architecture != kArchitectureVersion)
        throw std::invalid_argument("snapshot model architecture is incompatible");
    std::array<uint8_t, 32> fingerprint{};
    in.raw(fingerprint.data(), fingerprint.size());
    if (fingerprint != wrapper.model_->model_fingerprint())
        throw std::invalid_argument("snapshot model fingerprint does not match loaded weights");
    const auto& cfg = wrapper.model_->cfg();
    if (in.u32() != static_cast<uint32_t>(cfg.sample_rate) ||
        in.u32() != static_cast<uint32_t>(cfg.num_speakers) ||
        in.u32() != static_cast<uint32_t>(cfg.n_mels) ||
        in.u32() != static_cast<uint32_t>(cfg.n_fft) ||
        in.u32() != static_cast<uint32_t>(wrapper.model_->fe().hop_length()) ||
        in.u32() != static_cast<uint32_t>(cfg.output_subsampling_factor))
        throw std::invalid_argument("snapshot streaming frame geometry is incompatible");
    DiarGeometry snapshot_geo;
    snapshot_geo.spkcache_len = in.i32();
    snapshot_geo.fifo_len = in.i32();
    snapshot_geo.chunk_len = in.i32();
    snapshot_geo.spkcache_update_period = in.i32();
    snapshot_geo.chunk_left_context = in.i32();
    snapshot_geo.chunk_right_context = in.i32();
    if (!same_geometry(snapshot_geo, wrapper.stream_->geo_))
        throw std::invalid_argument("snapshot streaming configuration is incompatible");
    const uint64_t payload_size = in.u64();
    const uint32_t checksum = in.u32();
    if (payload_size != in.remaining())
        throw std::invalid_argument("snapshot payload length is invalid or truncated");
    if (crc32(in.current(), in.remaining()) != checksum)
        throw std::invalid_argument("snapshot payload checksum mismatch");

    Reader payload(in.current(), in.remaining());
    std::unique_ptr<audio::AudioResampler> restored_resampler;
    int restored_input_rate = 0;
    DiarSnapshotAccess::read_stream_payload(payload, wrapper, restored_resampler, restored_input_rate);
}

}  // namespace nemo_speech::asr
