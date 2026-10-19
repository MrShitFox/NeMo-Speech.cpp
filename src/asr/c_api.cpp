// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Maps the nemo_speech_asr_* C ABI onto the C++ recognizer. Exceptions become status
// codes and thread-local errors; no C++ types cross the boundary.
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "c_api_internal.h"
#include "nemo_speech/asr.h"
#include "recognizer.h"
#include "types.h"

namespace asr_core = nemo_speech::asr;
using nemo_speech::c_api::set_last_error;

struct nemo_speech_asr_recognizer {
    std::unique_ptr<asr_core::Recognizer> impl;
};
struct nemo_speech_asr_stream {
    std::unique_ptr<asr_core::RecognitionStream> impl;
    bool finished = false;
    std::optional<asr_core::Result> pending_final;
};
struct nemo_speech_asr_result {
    asr_core::Result value;
};
namespace {

// A field is present iff the caller's struct `size` covers it (append-only ABI:
// a caller compiled against an older header sends a smaller size; missing tail
// fields keep their library defaults).
#define HAS_FIELD(ptr, type, member) \
    ((ptr) != nullptr && (ptr)->size >= offsetof(type, member) + sizeof(((type*)nullptr)->member))

std::string
str_or_empty(const char* s) {
    return s ? std::string(s) : std::string();
}

asr_core::RecognizerConfig
to_config(const nemo_speech_asr_recognizer_config* c) {
    asr_core::RecognizerConfig cfg;  // library defaults
    if (!c)
        return cfg;
    if (HAS_FIELD(c, nemo_speech_asr_recognizer_config, backend) && c->backend) {
        const auto* b = c->backend;
        if (HAS_FIELD(b, nemo_speech_asr_backend_config, gpu))
            cfg.backend.gpu = b->gpu;
    }
    if (HAS_FIELD(c, nemo_speech_asr_recognizer_config, model) && c->model) {
        const auto* m = c->model;
        if (HAS_FIELD(m, nemo_speech_asr_model_config, path))
            cfg.model.path = str_or_empty(m->path);
        if (HAS_FIELD(m, nemo_speech_asr_model_config, name))
            cfg.model.name = str_or_empty(m->name);
    }
    if (HAS_FIELD(c, nemo_speech_asr_recognizer_config, streaming) && c->streaming) {
        const auto* e = c->streaming;
        if (HAS_FIELD(e, nemo_speech_asr_streaming_config, chunk_size))
            cfg.streaming.chunk_size = e->chunk_size;
        if (HAS_FIELD(e, nemo_speech_asr_streaming_config, ctc_left_padding))
            cfg.streaming.ctc_left_padding = e->ctc_left_padding;
        if (HAS_FIELD(e, nemo_speech_asr_streaming_config, ctc_right_padding))
            cfg.streaming.ctc_right_padding = e->ctc_right_padding;
        if (HAS_FIELD(e, nemo_speech_asr_streaming_config, rnnt_right_context))
            cfg.streaming.rnnt_right_context = e->rnnt_right_context;
    }
    if (HAS_FIELD(c, nemo_speech_asr_recognizer_config, decoder) && c->decoder) {
        const auto* d = c->decoder;
        if (HAS_FIELD(d, nemo_speech_asr_decoder_config, kind))
            cfg.decoder.kind = (d->kind == NEMO_SPEECH_ASR_DECODER_FLASHLIGHT)
                                   ? asr_core::DecoderConfig::Kind::Flashlight
                                   : asr_core::DecoderConfig::Kind::Greedy;
        if (HAS_FIELD(d, nemo_speech_asr_decoder_config, flashlight_lm))
            cfg.decoder.lm_path = str_or_empty(d->flashlight_lm);
        if (HAS_FIELD(d, nemo_speech_asr_decoder_config, flashlight_lexicon))
            cfg.decoder.lexicon_path = str_or_empty(d->flashlight_lexicon);
        if (HAS_FIELD(d, nemo_speech_asr_decoder_config, flashlight_tokenizer))
            cfg.decoder.tokenizer_path = str_or_empty(d->flashlight_tokenizer);
        if (HAS_FIELD(d, nemo_speech_asr_decoder_config, beam_size) && d->beam_size > 0)
            cfg.decoder.beam_size = d->beam_size;
        if (HAS_FIELD(d, nemo_speech_asr_decoder_config, beam_size_token) && d->beam_size_token > 0)
            cfg.decoder.beam_size_token = d->beam_size_token;
        if (HAS_FIELD(d, nemo_speech_asr_decoder_config, beam_threshold) && d->beam_threshold > 0)
            cfg.decoder.beam_threshold = d->beam_threshold;
        if (HAS_FIELD(d, nemo_speech_asr_decoder_config, lm_weight))
            cfg.decoder.lm_weight = d->lm_weight;
        if (HAS_FIELD(d, nemo_speech_asr_decoder_config, word_insertion_score))
            cfg.decoder.word_insertion_score = d->word_insertion_score;
        if (HAS_FIELD(d, nemo_speech_asr_decoder_config, max_boost) && d->max_boost > 0)
            cfg.decoder.max_boost = d->max_boost;
        if (!cfg.decoder.lm_path.empty())
            cfg.decoder.kind = asr_core::DecoderConfig::Kind::Flashlight;
    }
    if (HAS_FIELD(c, nemo_speech_asr_recognizer_config, vad) && c->vad) {
        const auto* v = c->vad;
        if (HAS_FIELD(v, nemo_speech_asr_vad_config, model_path))
            cfg.vad.model_path = str_or_empty(v->model_path);
        if (HAS_FIELD(v, nemo_speech_asr_vad_config, enable_masking))
            cfg.vad.masker.mask_enable = v->enable_masking;
        if (HAS_FIELD(v, nemo_speech_asr_vad_config, onset) && v->onset > 0)
            cfg.vad.masker.onset = v->onset;
        if (HAS_FIELD(v, nemo_speech_asr_vad_config, offset) && v->offset > 0)
            cfg.vad.masker.offset = v->offset;
    }
    if (HAS_FIELD(c, nemo_speech_asr_recognizer_config, endpointing) && c->endpointing) {
        const auto* e = c->endpointing;
        if (HAS_FIELD(e, nemo_speech_asr_endpointing_config, enable))
            cfg.endpointing.enable = e->enable;
        if (HAS_FIELD(e, nemo_speech_asr_endpointing_config, vad_based))
            cfg.endpointing.vad_based = e->vad_based;
        if (HAS_FIELD(e, nemo_speech_asr_endpointing_config, stop_history_eou_ms) &&
            e->stop_history_eou_ms > 0)
            cfg.endpointing.stop_history_eou_ms = e->stop_history_eou_ms;
    }
    if (HAS_FIELD(c, nemo_speech_asr_recognizer_config, postproc) && c->postproc) {
        const auto* p = c->postproc;
        if (HAS_FIELD(p, nemo_speech_asr_postproc_config, profanity_list_path))
            cfg.postproc.profanity_list_path = str_or_empty(p->profanity_list_path);
        if (HAS_FIELD(p, nemo_speech_asr_postproc_config, itn_model_dir))
            cfg.postproc.itn_model_dir = str_or_empty(p->itn_model_dir);
        if (HAS_FIELD(p, nemo_speech_asr_postproc_config, pnc_model_path))
            cfg.postproc.pnc_model_path = str_or_empty(p->pnc_model_path);
    }
    if (HAS_FIELD(c, nemo_speech_asr_recognizer_config, diar) && c->diar) {
        const auto* d = c->diar;
        if (HAS_FIELD(d, nemo_speech_asr_diar_config, model_path))
            cfg.diar.model_path = str_or_empty(d->model_path);
        if (HAS_FIELD(d, nemo_speech_asr_diar_config, chunk_frames) && d->chunk_frames > 0)
            cfg.diar.geometry.chunk_len = d->chunk_frames;
        if (HAS_FIELD(d, nemo_speech_asr_diar_config, right_context_frames) &&
            d->right_context_frames > 0)
            cfg.diar.geometry.chunk_right_context = d->right_context_frames;
        if (HAS_FIELD(d, nemo_speech_asr_diar_config, left_context_frames) &&
            d->left_context_frames >= 0)
            cfg.diar.geometry.chunk_left_context = d->left_context_frames;
        if (HAS_FIELD(d, nemo_speech_asr_diar_config, fifo_frames) && d->fifo_frames > 0)
            cfg.diar.geometry.fifo_len = d->fifo_frames;
        if (HAS_FIELD(d, nemo_speech_asr_diar_config, spkcache_frames) && d->spkcache_frames > 0)
            cfg.diar.geometry.spkcache_len = d->spkcache_frames;
        if (HAS_FIELD(d, nemo_speech_asr_diar_config, update_period_frames) &&
            d->update_period_frames > 0)
            cfg.diar.geometry.spkcache_update_period = d->update_period_frames;
    }
    if (HAS_FIELD(c, nemo_speech_asr_recognizer_config, batching) && c->batching) {
        const auto* b = c->batching;
        if (HAS_FIELD(b, nemo_speech_asr_batching_config, enable))
            cfg.batching.enabled = b->enable;
        if (HAS_FIELD(b, nemo_speech_asr_batching_config, max_batch_size) && b->max_batch_size > 0)
            cfg.batching.max_batch_size = b->max_batch_size;
        if (HAS_FIELD(b, nemo_speech_asr_batching_config, max_queue_delay_us) &&
            b->max_queue_delay_us >= 0)
            cfg.batching.max_queue_delay_us = b->max_queue_delay_us;
        if (HAS_FIELD(b, nemo_speech_asr_batching_config, max_queue_depth) &&
            b->max_queue_depth > 0)
            cfg.batching.max_queue_depth = b->max_queue_depth;
        if (HAS_FIELD(b, nemo_speech_asr_batching_config, ingress_cohort_delay_us) &&
            b->ingress_cohort_delay_us >= 0)
            cfg.batching.ingress_cohort_delay_us = b->ingress_cohort_delay_us;
        if (HAS_FIELD(b, nemo_speech_asr_batching_config, state_arena_slots) &&
            b->state_arena_slots > 0)
            cfg.batching.state_arena_slots = b->state_arena_slots;
    }
    return cfg;
}

asr_core::AsrRequestOptions
to_options(const nemo_speech_asr_recognition_options* o) {
    asr_core::AsrRequestOptions opts;
    if (!o)
        return opts;
    if (HAS_FIELD(o, nemo_speech_asr_recognition_options, max_alternatives) &&
        o->max_alternatives > 1)
        opts.max_alternatives = o->max_alternatives;
    if (HAS_FIELD(o, nemo_speech_asr_recognition_options, enable_word_time_offsets))
        opts.enable_word_time_offsets = o->enable_word_time_offsets;
    if (HAS_FIELD(o, nemo_speech_asr_recognition_options, enable_automatic_punctuation))
        opts.enable_automatic_punctuation = o->enable_automatic_punctuation;
    if (HAS_FIELD(o, nemo_speech_asr_recognition_options, verbatim_transcripts))
        opts.verbatim_transcripts = o->verbatim_transcripts;
    if (HAS_FIELD(o, nemo_speech_asr_recognition_options, profanity_filter))
        opts.profanity_filter = o->profanity_filter;
    if (HAS_FIELD(o, nemo_speech_asr_recognition_options, stop_history_eou_ms) &&
        o->stop_history_eou_ms > 0)
        opts.stop_history_eou_ms = static_cast<float>(o->stop_history_eou_ms);
    if (HAS_FIELD(o, nemo_speech_asr_recognition_options, enable_speaker_diarization))
        opts.enable_speaker_diarization = o->enable_speaker_diarization;
    if (HAS_FIELD(o, nemo_speech_asr_recognition_options, max_speaker_count) &&
        o->max_speaker_count > 0)
        opts.max_speaker_count = o->max_speaker_count;
    if (HAS_FIELD(o, nemo_speech_asr_recognition_options, speech_context_count) &&
        o->speech_contexts) {
        for (size_t i = 0; i < o->speech_context_count; ++i) {
            const nemo_speech_asr_speech_context& sc = o->speech_contexts[i];
            asr_core::AsrRequestOptions::Boost b;
            b.boost = sc.boost;
            if (sc.phrases) {
                for (size_t j = 0; j < sc.phrase_count; ++j)
                    b.phrases.push_back(str_or_empty(sc.phrases[j]));
            }
            opts.speech_contexts.push_back(std::move(b));
        }
    }
    return opts;
}

std::string
lang_of(const nemo_speech_asr_recognition_options* o) {
    return (HAS_FIELD(o, nemo_speech_asr_recognition_options, language_code))
               ? str_or_empty(o->language_code)
               : std::string();
}

// Converts exceptions to status codes and last-error text.
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

}  // namespace

extern "C" {

nemo_speech_asr_recognition_options
nemo_speech_asr_recognition_options_default(void) {
    nemo_speech_asr_recognition_options o;
    std::memset(&o, 0, sizeof(o));
    o.size = sizeof(o);
    o.max_alternatives = 1;  // 1-best
    return o;
}

nemo_speech_asr_status
nemo_speech_asr_create(
    const nemo_speech_asr_recognizer_config* cfg, nemo_speech_asr_recognizer** out) {
    if (!out)
        return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return guard([&] {
        asr_core::RecognizerConfig rc = to_config(cfg);
        if (rc.model.path.empty()) {
            set_last_error("recognizer config: model.path is required");
            return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
        }
        auto h = std::make_unique<nemo_speech_asr_recognizer>();
        h->impl = std::make_unique<asr_core::Recognizer>(std::move(rc));
        *out = h.release();
        return NEMO_SPEECH_ASR_OK;
    });
}

void
nemo_speech_asr_destroy(nemo_speech_asr_recognizer* recognizer) {
    delete recognizer;
}

nemo_speech_asr_status
nemo_speech_asr_recognize_f32(
    nemo_speech_asr_recognizer* recognizer, const nemo_speech_asr_recognition_options* options,
    const float* samples, size_t n_samples, int32_t sample_rate, nemo_speech_asr_result** out) {
    if (!recognizer || !out)
        return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return guard([&] {
        if (!samples || n_samples == 0) {
            set_last_error("empty audio");
            return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
        }
        auto r = std::make_unique<nemo_speech_asr_result>();
        r->value = recognizer->impl->recognize(
            samples, n_samples, to_options(options), lang_of(options), sample_rate);
        *out = r.release();
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_asr_streaming_recognize(
    nemo_speech_asr_recognizer* recognizer, const nemo_speech_asr_recognition_options* options,
    nemo_speech_asr_stream** out) {
    if (!recognizer || !out)
        return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return guard([&] {
        auto h = std::make_unique<nemo_speech_asr_stream>();
        h->impl = recognizer->impl->streaming_recognize(to_options(options), lang_of(options));
        *out = h.release();
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_asr_stream_push_f32(
    nemo_speech_asr_stream* stream, const float* samples, size_t n_samples, int32_t sample_rate) {
    if (!stream)
        return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
    return guard([&] {
        if (samples == nullptr && n_samples > 0) {
            set_last_error("audio samples must not be null when n_samples is nonzero");
            return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
        }
        if (samples && n_samples)
            stream->impl->push(samples, n_samples, sample_rate);
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_asr_stream_force_endpoint(nemo_speech_asr_stream* stream) {
    if (!stream)
        return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
    return guard([&] {
        stream->impl->force_endpoint();
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_asr_stream_finish(nemo_speech_asr_stream* stream) {
    if (!stream)
        return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
    return guard([&] {
        stream->pending_final = stream->impl->finish();
        stream->finished = true;
        return NEMO_SPEECH_ASR_OK;
    });
}

nemo_speech_asr_status
nemo_speech_asr_stream_next(nemo_speech_asr_stream* stream, nemo_speech_asr_result** out) {
    if (!stream || !out)
        return NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return guard([&] {
        std::optional<asr_core::Result> r;
        if (stream->finished) {
            r = std::move(stream->pending_final);
            stream->pending_final.reset();
        } else {
            r = stream->impl->next();
        }
        if (r) {
            auto h = std::make_unique<nemo_speech_asr_result>();
            h->value = std::move(*r);
            *out = h.release();
        }
        return NEMO_SPEECH_ASR_OK;
    });
}

void
nemo_speech_asr_stream_close(nemo_speech_asr_stream* stream) {
    delete stream;
}

namespace {
// Returns alternative `alt` (best-first), or nullptr when out of range.
const asr_core::Alternative*
alt_at(const nemo_speech_asr_result* r, size_t alt) {
    return (r && alt < r->value.alternatives.size()) ? &r->value.alternatives[alt] : nullptr;
}
}  // namespace

bool
nemo_speech_asr_result_is_final(const nemo_speech_asr_result* result) {
    return result ? result->value.is_final : false;
}
float
nemo_speech_asr_result_audio_processed(const nemo_speech_asr_result* result) {
    return result ? result->value.audio_processed : 0.0f;
}
int32_t
nemo_speech_asr_result_channel_tag(const nemo_speech_asr_result* result) {
    return result ? result->value.channel_tag : 0;
}
size_t
nemo_speech_asr_result_alternative_count(const nemo_speech_asr_result* result) {
    return result ? result->value.alternatives.size() : 0;
}
const char*
nemo_speech_asr_result_transcript(const nemo_speech_asr_result* result, size_t alt) {
    const auto* a = alt_at(result, alt);
    return a ? a->transcript.c_str() : "";
}
float
nemo_speech_asr_result_confidence(const nemo_speech_asr_result* result, size_t alt) {
    const auto* a = alt_at(result, alt);
    return a ? a->confidence : 0.0f;
}
size_t
nemo_speech_asr_result_word_count(const nemo_speech_asr_result* result, size_t alt) {
    const auto* a = alt_at(result, alt);
    return a ? a->words.size() : 0;
}
const char*
nemo_speech_asr_result_word_text(const nemo_speech_asr_result* result, size_t alt, size_t i) {
    const auto* a = alt_at(result, alt);
    return (a && i < a->words.size()) ? a->words[i].word.c_str() : "";
}
int32_t
nemo_speech_asr_result_word_speaker_tag(
    const nemo_speech_asr_result* result, size_t alt, size_t i) {
    const auto* a = alt_at(result, alt);
    return (a && i < a->words.size()) ? a->words[i].speaker_tag : 0;
}
int32_t
nemo_speech_asr_result_word_start_time(const nemo_speech_asr_result* result, size_t alt, size_t i) {
    const auto* a = alt_at(result, alt);
    return (a && i < a->words.size()) ? a->words[i].start_time : 0;
}
int32_t
nemo_speech_asr_result_word_end_time(const nemo_speech_asr_result* result, size_t alt, size_t i) {
    const auto* a = alt_at(result, alt);
    return (a && i < a->words.size()) ? a->words[i].end_time : 0;
}
float
nemo_speech_asr_result_word_confidence(const nemo_speech_asr_result* result, size_t alt, size_t i) {
    const auto* a = alt_at(result, alt);
    return (a && i < a->words.size()) ? a->words[i].confidence : 0.0f;
}
size_t
nemo_speech_asr_result_language_count(const nemo_speech_asr_result* result, size_t alt) {
    const auto* a = alt_at(result, alt);
    return a ? a->language_codes.size() : 0;
}
const char*
nemo_speech_asr_result_language_code(const nemo_speech_asr_result* result, size_t alt, size_t i) {
    const auto* a = alt_at(result, alt);
    return (a && i < a->language_codes.size()) ? a->language_codes[i].c_str() : "";
}
void
nemo_speech_asr_result_destroy(nemo_speech_asr_result* result) {
    delete result;
}

const char*
nemo_speech_asr_last_error(void) {
    return nemo_speech::c_api::last_error();
}

const char*
nemo_speech_asr_version(void) {
    return nemo_speech::c_api::version();
}

}  // extern "C"
