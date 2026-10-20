// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "diarizer.h"

#include <stdexcept>
#include <utility>

#include "audio_resampler.h"
#include "diar_snapshot.h"
#include "runtime.h"

namespace nemo_speech::asr {
namespace {

ggml_runtime::Params
backend_params(int gpu, const std::string& gpu_backend_name) {
    ggml_runtime::Params params;
    params.use_gpu = gpu >= 0;
    params.gpu_device_idx = gpu < 0 ? 0 : gpu;
    params.gpu_backend_name = gpu_backend_name;
    params.pe_bin_path = const_cast<char*>("");
    return params;
}

struct DiarizerResources {
    ggml_runtime::BackendManager backend;
    DiarModel model;

    DiarizerResources(
        int gpu, const std::string& model_path, const BatchingConfig& batching,
        const std::string& gpu_backend_name)
        : backend(backend_params(gpu, gpu_backend_name)), model(backend, model_path, batching) {}
};

}  // namespace

DiarizationStream::DiarizationStream(std::shared_ptr<DiarModel> model, const DiarGeometry& geometry)
    : model_(std::move(model)) {
    if (!model_)
        throw std::invalid_argument("diarization model must not be null");
    stream_ = std::make_unique<DiarStream>(*model_, geometry);
}

void
DiarizationStream::push(const float* samples, size_t n_samples, int sample_rate) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    if (!samples && n_samples > 0)
        throw std::invalid_argument("samples must not be null");
    const int model_rate = model_->cfg().sample_rate;
    const int rate = sample_rate > 0 ? sample_rate : model_rate;
    if (sample_rate < 0 || !audio::supported_input_sample_rate(rate))
        throw std::invalid_argument("input sample rate must be between 8000 and 96000 Hz");
    if (input_sample_rate_ == 0) {
        input_sample_rate_ = rate;
        if (rate != model_rate)
            resampler_ = std::make_unique<audio::AudioResampler>(rate, model_rate);
    } else if (rate != input_sample_rate_) {
        throw std::invalid_argument("input sample rate cannot change within a stream");
    }
    if (n_samples == 0)
        return;
    if (!resampler_) {
        stream_->feed_audio(samples, n_samples);
        return;
    }
    resampled_.clear();
    resampler_->process(samples, n_samples, &resampled_);
    if (!resampled_.empty())
        stream_->feed_audio(resampled_.data(), resampled_.size());
}

void
DiarizationStream::finish() {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    if (resampler_) {
        resampled_.clear();
        resampler_->finish(&resampled_);
        resampler_.reset();
        if (!resampled_.empty())
            stream_->feed_audio(resampled_.data(), resampled_.size());
    }
    stream_->finish();
}

void
DiarizationStream::flush_available(int64_t target_frame) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    stream_->flush_available(target_frame);
}

void
DiarizationStream::set_compaction(int64_t trigger_frames, int64_t retain_frames) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    stream_->set_compaction(trigger_frames, retain_frames);
}

std::vector<uint8_t>
DiarizationStream::export_state() const {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    return DiarStreamSnapshotCodec::encode(*this);
}

void
DiarizationStream::import_state(const uint8_t* data, size_t size) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    DiarStreamSnapshotCodec::decode(*this, data, size);
}

void
DiarizationStream::set_research_embedding_observer(DiarStream::ResearchEmbeddingObserver observer) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    stream_->set_research_embedding_observer(std::move(observer));
}

void
DiarizationStream::set_research_feature_observer(DiarStream::ResearchFeatureObserver observer) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    stream_->set_research_feature_observer(std::move(observer));
}

void
DiarizationStream::set_research_aosc_selection_observer(
    DiarStream::ResearchAoscSelectionObserver observer) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    stream_->set_research_aosc_selection_observer(std::move(observer));
}

void
DiarizationStream::set_research_aosc_compression_observer(
    DiarStream::ResearchAoscCompressionObserver observer) {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    stream_->set_research_aosc_compression_observer(std::move(observer));
}

int64_t
DiarizationStream::frame_count() const {
    return stream_->n_frames();
}

int64_t
DiarizationStream::frame_probs_start() const {
    return stream_->frame_probs_base();
}

const std::vector<float>&
DiarizationStream::frame_probabilities() const {
    return stream_->frame_probs();
}

std::array<int64_t, 3>
DiarizationStream::last_prediction_tensor_shape() const {
    return stream_->last_prediction_tensor_shape();
}

int
DiarizationStream::num_speakers() const {
    return model_->cfg().num_speakers;
}

double
DiarizationStream::seconds_per_frame() const {
    return stream_->seconds_per_frame();
}

std::vector<DiarSegment>
DiarizationStream::segments(const DiarSegmentationCfg& config) const {
    return stream_->segments(config);
}

std::shared_ptr<Diarizer>
Diarizer::load(
    int gpu, const std::string& model_path, DiarGeometry geometry, BatchingConfig batching,
    std::string gpu_backend_name) {
    auto resources = std::make_shared<DiarizerResources>(gpu, model_path, batching, gpu_backend_name);
    // Streams retain this alias, which keeps both the model and its backend alive.
    auto* model_pointer = &resources->model;
    auto model = std::shared_ptr<DiarModel>(std::move(resources), model_pointer);
    return std::make_shared<Diarizer>(std::move(model), std::move(geometry));
}

Diarizer::Diarizer(std::shared_ptr<DiarModel> model, DiarGeometry geometry)
    : model_(std::move(model)), geometry_(std::move(geometry)) {
    if (!model_)
        throw std::invalid_argument("diarization model must not be null");
    geometry_ = model_->resolved_geometry(geometry_);
    geometry_.validate(
        model_->cfg().num_speakers, model_->cfg().scoring.sil_frames_per_spk,
        model_->cfg().encoder.pos_emb_max_len);
}

DiarizationResult
Diarizer::diarize(
    const float* samples, size_t n_samples, int sample_rate, DiarizationMode mode,
    const DiarSegmentationCfg& segmentation) const {
    if (!samples && n_samples > 0)
        throw std::invalid_argument("samples must not be null");
    const int model_rate = this->sample_rate();
    const int input_rate = sample_rate > 0 ? sample_rate : model_rate;
    if (sample_rate < 0 || !audio::supported_input_sample_rate(input_rate))
        throw std::invalid_argument("input sample rate must be between 8000 and 96000 Hz");

    std::vector<float> converted;
    const float* audio_samples = samples;
    size_t audio_size = n_samples;
    if (input_rate != model_rate) {
        converted = audio::resample_audio(samples, n_samples, input_rate, model_rate);
        audio_samples = converted.data();
        audio_size = converted.size();
    }

    DiarizationResult result;
    result.audio_duration = input_rate > 0 ? static_cast<double>(n_samples) / input_rate : 0.0;
    if (mode == DiarizationMode::Offline) {
        int64_t frames = 0;
        const auto probabilities = model_->diarize_offline(audio_samples, audio_size, &frames);
        const auto& config = model_->cfg();
        const double seconds_per_frame = config.seconds_per_output_frame();
        result.segments = diar_segments_from_probs(
            probabilities.data(), frames, config.num_speakers, seconds_per_frame, segmentation);
        result.frame_probabilities = probabilities;
        result.frame_count = frames;
        result.num_speakers = config.num_speakers;
        result.seconds_per_frame = seconds_per_frame;
    } else {
        auto stream = streaming_diarize();
        stream->push(audio_samples, audio_size, model_rate);
        stream->finish();
        result.segments = stream->segments(segmentation);
        result.frame_probabilities = stream->frame_probabilities();
        result.frame_probs_start = stream->frame_probs_start();
        result.frame_count = stream->frame_count();
        result.num_speakers = stream->num_speakers();
        result.seconds_per_frame = stream->seconds_per_frame();
    }
    return result;
}

std::unique_ptr<DiarizationStream>
Diarizer::streaming_diarize() const {
    return std::make_unique<DiarizationStream>(model_, geometry_);
}

int
Diarizer::sample_rate() const {
    return model_->cfg().sample_rate;
}

int
Diarizer::num_speakers() const {
    return model_->cfg().num_speakers;
}

int
Diarizer::embedding_dim() const {
    return model_->cfg().encoder.d_model;
}

const std::array<uint8_t, 32>&
Diarizer::model_fingerprint() const {
    return model_->model_fingerprint();
}

double
Diarizer::seconds_per_frame() const {
    return model_->cfg().seconds_per_output_frame();
}

BatchMetrics
Diarizer::batch_metrics() const {
    return model_->batch_metrics();
}

void
Diarizer::dump_schedule(std::ostream& out, const std::string& session_label) const {
    model_->model().session()->dump_schedule(out, session_label);
}

}  // namespace nemo_speech::asr
