// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "audio_resampler.h"
#include "diar_pipeline.h"

namespace nemo_speech::asr {

class DiarStreamSnapshotCodec;
class DiarSnapshotAccess;
class DiarAoscProfileCodec;

enum class DiarizationMode { Streaming, Offline };

struct DiarizationResult {
    std::vector<DiarSegment> segments;
    std::vector<float> frame_probabilities;
    int64_t frame_probs_start = 0;
    int64_t frame_count = 0;
    int num_speakers = 0;
    double seconds_per_frame = 0.0;
    double audio_duration = 0.0;
};

class DiarizationStream {
   public:
    DiarizationStream(std::shared_ptr<DiarModel> model, const DiarGeometry& geometry);

    void push(const float* samples, size_t n_samples, int sample_rate = 0);
    void finish();
    void flush_available(int64_t target_frame);
    // Internal long-stream memory control, exposed at the C++ streaming layer
    // so tests and embedders can checkpoint custom compaction geometry.
    void set_compaction(int64_t trigger_frames, int64_t retain_frames);
    std::vector<uint8_t> export_state() const;
    void import_state(const uint8_t* data, size_t size);
    // Internal research hook; this is not part of the public C ABI.
    void set_research_embedding_observer(DiarStream::ResearchEmbeddingObserver observer);
    void set_research_feature_observer(DiarStream::ResearchFeatureObserver observer);
    void set_research_aosc_selection_observer(
        DiarStream::ResearchAoscSelectionObserver observer);
    int64_t frame_count() const;
    int64_t frame_probs_start() const;
    const std::vector<float>& frame_probabilities() const;
    int num_speakers() const;
    double seconds_per_frame() const;
    std::vector<DiarSegment> segments(const DiarSegmentationCfg& config = {}) const;

   private:
    friend class DiarStreamSnapshotCodec;
    friend class DiarSnapshotAccess;
    friend class DiarAoscProfileCodec;

    std::shared_ptr<DiarModel> model_;
    std::unique_ptr<DiarStream> stream_;
    std::unique_ptr<audio::AudioResampler> resampler_;
    std::vector<float> resampled_;
    int input_sample_rate_ = 0;
    bool state_imported_ = false;
    mutable std::mutex operation_mutex_;
};

// Standalone diarization interface that owns model lifetime, input-rate
// conversion, streaming/offline execution, and segmentation.
class Diarizer {
   public:
    static std::shared_ptr<Diarizer> load(
        int gpu, const std::string& model_path, DiarGeometry geometry = {},
        BatchingConfig batching = {});

    Diarizer(std::shared_ptr<DiarModel> model, DiarGeometry geometry);

    DiarizationResult diarize(
        const float* samples, size_t n_samples, int sample_rate,
        DiarizationMode mode = DiarizationMode::Streaming,
        const DiarSegmentationCfg& segmentation = {}) const;
    std::unique_ptr<DiarizationStream> streaming_diarize() const;

    int sample_rate() const;
    int num_speakers() const;
    int embedding_dim() const;
    const std::array<uint8_t, 32>& model_fingerprint() const;
    double seconds_per_frame() const;
    BatchMetrics batch_metrics() const;

   private:
    std::shared_ptr<DiarModel> model_;
    DiarGeometry geometry_;
};

}  // namespace nemo_speech::asr
