// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nemo_speech::asr {

class DiarizationStream;

struct AoscProfileSummary {
    int speaker_channels = 0;
    int embedding_dim = 0;
    int cache_frames = 0;
    std::vector<int> channel_centroid_frames;
};

// Experimental AOSC-only serialization used by the Task 05 probe. It carries
// only compressed speaker-cache state and the silence profile required by the
// cache updater; it deliberately excludes FIFO and stream/audio history.
class DiarAoscProfileCodec {
   public:
    static std::vector<uint8_t> encode(const DiarizationStream& stream);
    static void decode(DiarizationStream& stream, const uint8_t* data, size_t size);

    // Frame-major cosine values: n_frames x speaker_channels. A channel with
    // no selected speech rows in the enrollment cache is represented by NaN.
    static std::vector<float> cosine_scores(
        const std::vector<uint8_t>& profile, const float* embeddings, int n_frames, int embedding_dim);
    static AoscProfileSummary summarize(const std::vector<uint8_t>& profile);
};

}  // namespace nemo_speech::asr
