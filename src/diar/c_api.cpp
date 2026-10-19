// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Maps the nemo_speech_diar_* C ABI onto the Sortformer runtime. Exceptions
// become shared C ABI status codes and thread-local error text.
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>

#include "c_api_internal.h"
#include "diarizer.h"
#include "nemo_speech/diar.h"

namespace asr_core = nemo_speech::asr;
using nemo_speech::c_api::set_last_error;

struct nemo_speech_diar_model {
    std::shared_ptr<asr_core::Diarizer> impl;
};
// One diarization job: a live stream (impl set) or a finished offline result
// (offline set). Core stream state retains the model resources it needs.
struct nemo_speech_diar_stream {
    std::unique_ptr<asr_core::DiarizationStream> impl;
    std::optional<asr_core::DiarizationResult> offline;
};

namespace {

#define HAS_FIELD(ptr, type, member) \
    ((ptr) != nullptr && (ptr)->size >= offsetof(type, member) + sizeof(((type*)nullptr)->member))

template <class F>
nemo_speech_asr_status
guard(F&& f) {
    try {
        return f();
    }
    catch (const std::bad_alloc&) {
        set_last_error("out of memory");
        return NEMO_SPEECH_ASR_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::invalid_argument& e) {
        set_last_error(e.what());
        return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
    }
    catch (const std::exception& e) {
        set_last_error(e.what());
        return NEMO_SPEECH_ASR_ERROR_RUNTIME;
    }
    catch (...) {
        set_last_error("unknown error");
        return NEMO_SPEECH_ASR_ERROR_RUNTIME;
    }
}

// Unify the two job flavors for the result accessors. `probs` covers frames
// [base, n_frames): long-lived streams compact old raw probabilities into
// frozen segments, so base can be > 0 there (offline jobs always have base 0).
struct DiarResultView {
    const float* probs;
    int64_t base;
    int64_t n_frames;
    int n_spk;
    double sec_per_frame;
};
DiarResultView
diar_view(const nemo_speech_diar_stream* s) {
    if (s->impl)
        return {
            s->impl->frame_probabilities().data(), s->impl->frame_probs_start(),
            s->impl->frame_count(), s->impl->num_speakers(), s->impl->seconds_per_frame()};
    const auto& offline = *s->offline;
    return {
        offline.frame_probabilities.data(), offline.frame_probs_start, offline.frame_count,
        offline.num_speakers, offline.seconds_per_frame};
}

}  // namespace

extern "C" {

nemo_speech_asr_status
nemo_speech_diar_create(const nemo_speech_diar_model_config* cfg, nemo_speech_diar_model** out) {
    return guard([&] {
        if (!out)
            throw std::invalid_argument("out must not be NULL");
        *out = nullptr;
        if (!cfg || !HAS_FIELD(cfg, nemo_speech_diar_model_config, model_path) ||
            !cfg->model_path || !*cfg->model_path)
            throw std::invalid_argument("nemo_speech_diar_model_config.model_path is required");

        asr_core::DiarGeometry geo;
        if (HAS_FIELD(cfg, nemo_speech_diar_model_config, preset) && cfg->preset && *cfg->preset)
            geo = asr_core::DiarGeometry::preset(cfg->preset);
        if (HAS_FIELD(cfg, nemo_speech_diar_model_config, chunk_frames) && cfg->chunk_frames > 0)
            geo.chunk_len = cfg->chunk_frames;
        if (HAS_FIELD(cfg, nemo_speech_diar_model_config, right_context_frames) &&
            cfg->right_context_frames > 0)
            geo.chunk_right_context = cfg->right_context_frames;
        if (HAS_FIELD(cfg, nemo_speech_diar_model_config, left_context_frames) &&
            cfg->left_context_frames >= 0)
            geo.chunk_left_context = cfg->left_context_frames;
        if (HAS_FIELD(cfg, nemo_speech_diar_model_config, fifo_frames) && cfg->fifo_frames > 0)
            geo.fifo_len = cfg->fifo_frames;
        if (HAS_FIELD(cfg, nemo_speech_diar_model_config, spkcache_frames) &&
            cfg->spkcache_frames > 0)
            geo.spkcache_len = cfg->spkcache_frames;
        if (HAS_FIELD(cfg, nemo_speech_diar_model_config, update_period_frames) &&
            cfg->update_period_frames > 0)
            geo.spkcache_update_period = cfg->update_period_frames;

        const int gpu = HAS_FIELD(cfg, nemo_speech_diar_model_config, gpu) ? cfg->gpu : 0;
        auto handle = std::make_unique<nemo_speech_diar_model>();
        handle->impl = asr_core::Diarizer::load(gpu, cfg->model_path, geo);
        *out = handle.release();
        return NEMO_SPEECH_ASR_OK;
    });
}

void
nemo_speech_diar_destroy(nemo_speech_diar_model* model) {
    delete model;
}

int32_t
nemo_speech_diar_num_speakers(const nemo_speech_diar_model* model) {
    return model ? model->impl->num_speakers() : 0;
}

double
nemo_speech_diar_seconds_per_frame(const nemo_speech_diar_model* model) {
    if (!model)
        return 0.0;
    return model->impl->seconds_per_frame();
}

nemo_speech_asr_status
nemo_speech_diar_stream_open(nemo_speech_diar_model* model, nemo_speech_diar_stream** out) {
    return guard([&] {
        if (!model || !out)
            throw std::invalid_argument("model and out must not be NULL");
        *out = nullptr;  // never leave a stale pointer on failure
        auto s = std::make_unique<nemo_speech_diar_stream>();
        s->impl = model->impl->streaming_diarize();
        *out = s.release();
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_diar_stream_push_f32(
    nemo_speech_diar_stream* stream, const float* samples, size_t n_samples, int32_t sample_rate) {
    return guard([&] {
        if (!stream || !stream->impl)
            throw std::invalid_argument("not a live diarization stream");
        if (!samples && n_samples > 0)
            throw std::invalid_argument("samples must not be NULL");
        stream->impl->push(samples, n_samples, sample_rate);
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_diar_stream_finish(nemo_speech_diar_stream* stream) {
    return guard([&] {
        if (!stream || !stream->impl)
            throw std::invalid_argument("not a live diarization stream");
        stream->impl->finish();
        return NEMO_SPEECH_ASR_OK;
    });
}

void
nemo_speech_diar_stream_close(nemo_speech_diar_stream* stream) {
    delete stream;
}

nemo_speech_asr_status
nemo_speech_diar_stream_state_size(const nemo_speech_diar_stream* stream, size_t* size_bytes) {
    return guard([&] {
        if (!stream || !stream->impl || !size_bytes)
            throw std::invalid_argument("live stream and size_bytes must not be NULL");
        *size_bytes = 0;
        *size_bytes = stream->impl->export_state().size();
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_diar_stream_save_state(
    const nemo_speech_diar_stream* stream, void* out, size_t capacity_bytes, size_t* written) {
    return guard([&] {
        if (written) *written = 0;
        if (!stream || !stream->impl || !out || !written)
            throw std::invalid_argument("live stream, out and written must not be NULL");
        const auto snapshot = stream->impl->export_state();
        if (capacity_bytes < snapshot.size())
            throw std::invalid_argument(
                "capacity_bytes too small (need " + std::to_string(snapshot.size()) + ")");
        std::memcpy(out, snapshot.data(), snapshot.size());
        *written = snapshot.size();
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_diar_stream_load_state(
    nemo_speech_diar_stream* stream, const void* data, size_t size_bytes) {
    return guard([&] {
        if (!stream || !stream->impl || !data)
            throw std::invalid_argument("live stream and data must not be NULL");
        stream->impl->import_state(static_cast<const uint8_t*>(data), size_bytes);
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_diar_offline_f32(
    nemo_speech_diar_model* model, const float* samples, size_t n_samples, int32_t sample_rate,
    nemo_speech_diar_stream** out) {
    return guard([&] {
        if (!model || !out)
            throw std::invalid_argument("model and out must not be NULL");
        *out = nullptr;  // never leave a stale pointer on failure
        if (!samples && n_samples > 0)
            throw std::invalid_argument("samples must not be NULL");
        auto s = std::make_unique<nemo_speech_diar_stream>();
        s->offline = model->impl->diarize(
            samples, n_samples, sample_rate, asr_core::DiarizationMode::Offline);
        *out = s.release();
        return NEMO_SPEECH_ASR_OK;
    });
}

int64_t
nemo_speech_diar_frame_count(const nemo_speech_diar_stream* stream) {
    return stream ? diar_view(stream).n_frames : 0;
}

int64_t
nemo_speech_diar_frame_probs_start(const nemo_speech_diar_stream* stream) {
    return stream ? diar_view(stream).base : 0;
}

nemo_speech_asr_status
nemo_speech_diar_frame_probs(
    const nemo_speech_diar_stream* stream, float* out, size_t capacity_floats) {
    return guard([&] {
        if (!stream || !out)
            throw std::invalid_argument("stream and out must not be NULL");
        const auto v = diar_view(stream);
        const size_t need = static_cast<size_t>(v.n_frames - v.base) * v.n_spk;
        if (capacity_floats < need)
            throw std::invalid_argument(
                "capacity_floats too small (need " + std::to_string(need) + ")");
        std::memcpy(out, v.probs, need * sizeof(float));
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_diar_segments(
    const nemo_speech_diar_stream* stream, const nemo_speech_diar_segmentation_config* cfg,
    nemo_speech_diar_segment* out, size_t capacity, size_t* count) {
    return guard([&] {
        if (!stream || !count)
            throw std::invalid_argument("stream and count must not be NULL");
        asr_core::DiarSegmentationCfg sc;  // library defaults
        if (cfg) {
            if (HAS_FIELD(cfg, nemo_speech_diar_segmentation_config, onset) && cfg->onset > 0)
                sc.onset = cfg->onset;
            if (HAS_FIELD(cfg, nemo_speech_diar_segmentation_config, offset) && cfg->offset > 0)
                sc.offset = cfg->offset;
            if (HAS_FIELD(cfg, nemo_speech_diar_segmentation_config, pad_onset_sec) &&
                cfg->pad_onset_sec > 0)
                sc.pad_onset = cfg->pad_onset_sec;
            if (HAS_FIELD(cfg, nemo_speech_diar_segmentation_config, pad_offset_sec) &&
                cfg->pad_offset_sec > 0)
                sc.pad_offset = cfg->pad_offset_sec;
            if (HAS_FIELD(cfg, nemo_speech_diar_segmentation_config, min_gap_sec) &&
                cfg->min_gap_sec > 0)
                sc.min_duration_off = cfg->min_gap_sec;
            if (HAS_FIELD(cfg, nemo_speech_diar_segmentation_config, min_duration_sec) &&
                cfg->min_duration_sec > 0)
                sc.min_duration_on = cfg->min_duration_sec;
        }
        // Live streams go through the core stream so compacted prefixes
        // (frozen segments) are included; offline jobs segment the full probs.
        const auto v = diar_view(stream);
        const auto segs = stream->impl ? stream->impl->segments(sc)
                                       : asr_core::diar_segments_from_probs(
                                             v.probs, v.n_frames, v.n_spk, v.sec_per_frame, sc);
        *count = segs.size();
        if (!out)
            return NEMO_SPEECH_ASR_OK;  // size query
        if (capacity < segs.size())
            throw std::invalid_argument(
                "capacity too small (need " + std::to_string(segs.size()) + ")");
        for (size_t i = 0; i < segs.size(); i++) {
            out[i].start_time = segs[i].t0;
            out[i].end_time = segs[i].t1;
            out[i].speaker = segs[i].speaker + 1;  // 1-based, like WordInfo.speaker_tag
        }
        return NEMO_SPEECH_ASR_OK;
    });
}

#ifdef NEMO_SPEECH_DIAR_C_API_STANDALONE
const char*
nemo_speech_asr_last_error(void) {
    return nemo_speech::c_api::last_error();
}

const char*
nemo_speech_asr_version(void) {
    return nemo_speech::c_api::version();
}
#endif

}  // extern "C"
