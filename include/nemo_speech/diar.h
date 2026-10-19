// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// nemo-speech: stable C ABI for standalone speaker diarization.
//
// The Sortformer diarizer as its own pipeline: audio in, per-frame speaker
// probabilities and speaker segments out - no ASR involved. For word-level
// speaker tags on transcripts, integrate the diarization C ABI with your
// application; this project ships no ASR API.
//
// The ABI uses opaque handles, append-only size-prefixed structs, and does not
// pass exceptions across the boundary. The status type and established
// nemo_speech_asr_last_error() symbol retain their historical names for ABI
// compatibility; no ASR implementation or ASR API is included.
#ifndef NEMO_SPEECH_DIAR_H
#define NEMO_SPEECH_DIAR_H

#include "nemo_speech/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nemo_speech_diar_model nemo_speech_diar_model;
// One diarization job: either a live stream (nemo_speech_diar_stream_open + push +
// finish) or a completed offline result (nemo_speech_diar_offline_f32). The result
// accessors below work on both.
typedef struct nemo_speech_diar_stream nemo_speech_diar_stream;

typedef struct nemo_speech_diar_model_config {
    size_t size;
    const char* model_path;  // Nemotron 3 / Sortformer V3 GGUF (required)
    int32_t gpu;             // GPU device index; -1 = CPU
    // Streaming geometry preset: "v3-streaming" or "v3-offline". NULL/""
    // selects the model-specific low-latency default. Every preset uses the streaming state machine; the
    // stateless full-attention path is nemo_speech_diar_offline_f32.
    const char* preset;
    // Individual geometry overrides in coarse 80 ms encoder frames, applied on top
    // of the preset (<= 0 = keep preset value; left context: < 0 keeps it, 0
    // is a valid explicit value). Validated at create: bad combinations fail
    // with INVALID_ARGUMENT rather than degrading.
    int32_t chunk_frames;
    int32_t right_context_frames;
    int32_t left_context_frames;
    int32_t fifo_frames;
    int32_t spkcache_frames;
    int32_t update_period_frames;
} nemo_speech_diar_model_config;

// Segment postprocessing (NeMo ts_vad semantics: onset/offset hysteresis ->
// pad segment edges -> fill short gaps -> drop short segments). Zero/negative
// fields keep the library defaults, which are NeMo's callhome-tuned values
// for this checkpoint; they are dataset-sensitive (clean read speech wants a
// lower onset and larger pads).
typedef struct nemo_speech_diar_segmentation_config {
    size_t size;
    float onset;              // enter speech above this probability
    float offset;             // leave speech below this probability
    double pad_onset_sec;     // widen each segment start
    double pad_offset_sec;    // widen each segment end
    double min_gap_sec;       // fill silence gaps shorter than this
    double min_duration_sec;  // drop segments shorter than this
} nemo_speech_diar_segmentation_config;

typedef struct nemo_speech_diar_segment {
    double start_time;  // seconds
    double end_time;    // seconds
    int32_t speaker;    // 1-based speaker slot
} nemo_speech_diar_segment;

// ---- Model ----

NEMO_SPEECH_ASR_API nemo_speech_asr_status
nemo_speech_diar_create(const nemo_speech_diar_model_config* cfg, nemo_speech_diar_model** out);
NEMO_SPEECH_ASR_API void nemo_speech_diar_destroy(nemo_speech_diar_model* model);

// Model capacity and native V3 output cadence (8 speakers/10 ms for the
// supported Nemotron 3 checkpoint).
NEMO_SPEECH_ASR_API int32_t nemo_speech_diar_num_speakers(const nemo_speech_diar_model* model);
NEMO_SPEECH_ASR_API double nemo_speech_diar_seconds_per_frame(const nemo_speech_diar_model* model);

// ---- Streaming ----
// The model handle must outlive every stream opened from it. Streams are
// single-threaded by contract; independent streams may run on different
// threads (compute serializes internally).

NEMO_SPEECH_ASR_API nemo_speech_asr_status
nemo_speech_diar_stream_open(nemo_speech_diar_model* model, nemo_speech_diar_stream** out);
// Buffer mono float32 audio; diarization advances as whole chunks
// (chunk + right context) become available. Rates from 8-96 kHz are resampled
// to the model rate; 0 means model rate. The rate cannot change within a
// stream. Pushes after finish are
// silent no-ops.
NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_diar_stream_push_f32(
    nemo_speech_diar_stream* stream, const float* samples, size_t n_samples, int32_t sample_rate);
// No more audio: flush the resampler tail (if any) and label the remaining
// audio tail.
NEMO_SPEECH_ASR_API nemo_speech_asr_status
nemo_speech_diar_stream_finish(nemo_speech_diar_stream* stream);
NEMO_SPEECH_ASR_API void nemo_speech_diar_stream_close(nemo_speech_diar_stream* stream);

// ---- Streaming state checkpoint (format version 1) ----------------------
// Snapshots contain voice-derived state and should be handled as sensitive
// data. The size query and save call may run only between stream operations;
// the implementation serializes them with push/finish on the same handle.
NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_diar_stream_state_size(
    const nemo_speech_diar_stream* stream, size_t* size_bytes);
// Two-call export: query size with out=NULL/capacity=0, allocate that many
// bytes, then save. *written receives the exact snapshot size on success.
NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_diar_stream_save_state(
    const nemo_speech_diar_stream* stream, void* out, size_t capacity_bytes, size_t* written);
// Import only into a fresh stream returned by stream_open. A failed import
// leaves that stream untouched; importing into an active stream is rejected.
NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_diar_stream_load_state(
    nemo_speech_diar_stream* stream, const void* data, size_t size_bytes);

// ---- Offline (stateless) ----
// One full-attention pass over the whole file with no streaming state (NeMo
// streaming_mode=False). `sample_rate` follows the push contract above
// (8-96 kHz resampled, 0 = model rate). Bounded by the encoder's positional
// limit (model-dependent); longer audio fails with
// INVALID_ARGUMENT - use a stream. On success *out is a finished job handle
// for the accessors below (release with nemo_speech_diar_stream_close).
NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_diar_offline_f32(
    nemo_speech_diar_model* model, const float* samples, size_t n_samples, int32_t sample_rate,
    nemo_speech_diar_stream** out);

// ---- Results (valid on a live stream at any point; complete after finish) --

// Number of labeled native-cadence frames so far. Query the duration with
// nemo_speech_diar_seconds_per_frame().
NEMO_SPEECH_ASR_API int64_t nemo_speech_diar_frame_count(const nemo_speech_diar_stream* stream);
// First frame whose raw probabilities are still retained. 0 for offline jobs
// and streams shorter than the compaction horizon (~20 min); long-lived
// streams bound their memory by converting older probabilities into final
// segments (still reported by nemo_speech_diar_segments), after which this advances.
NEMO_SPEECH_ASR_API int64_t
nemo_speech_diar_frame_probs_start(const nemo_speech_diar_stream* stream);
// Copy the retained per-frame speaker probabilities (frame-major,
// (frame_count - frame_probs_start) * num_speakers floats, covering frames
// [frame_probs_start, frame_count)). `capacity_floats` must be at least that
// (else INVALID_ARGUMENT).
NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_diar_frame_probs(
    const nemo_speech_diar_stream* stream, float* out, size_t capacity_floats);
// Speaker segments under `cfg` (NULL = library defaults). Two-call pattern:
// pass out=NULL to receive the count, then call again with a buffer of at
// least *count segments. Segments are sorted by start time. On streams past
// the compaction horizon, segments before frame_probs_start were finalized
// with the library-default config; `cfg` applies to the retained tail.
NEMO_SPEECH_ASR_API nemo_speech_asr_status nemo_speech_diar_segments(
    const nemo_speech_diar_stream* stream, const nemo_speech_diar_segmentation_config* cfg,
    nemo_speech_diar_segment* out, size_t capacity, size_t* count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NEMO_SPEECH_DIAR_H
