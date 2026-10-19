// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nemo_speech::asr {

class DiarizationStream;

// Internal codec for versioned, portable streaming checkpoints. It is a friend
// of the state-owning classes so no runtime objects are exposed in the format.
class DiarStreamSnapshotCodec {
   public:
    static std::array<uint8_t, 32> fingerprint_file(const std::string& path);
    static std::vector<uint8_t> encode(const DiarizationStream& stream);
    static void decode(DiarizationStream& stream, const uint8_t* data, size_t size);
};

}  // namespace nemo_speech::asr
