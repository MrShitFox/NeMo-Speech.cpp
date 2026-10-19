// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "aosc_profile_intervention.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

native_matching::AoscProfile fixture() {
    native_matching::AoscProfile profile;
    profile.header.assign(113, 0);
    const uint8_t magic[8] = {'N', 'M', 'S', 'A', 'O', 'S', 'C', '1'};
    std::copy(magic, magic + 8, profile.header.begin());
    auto put32 = [&](size_t offset, uint32_t value) {
        for (int i = 0; i < 4; ++i) profile.header[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    };
    put32(8, 1);        // profile version
    put32(12, 3);       // Sortformer V3
    put32(48, 16000);   // sample rate
    put32(52, 2);       // speaker channels
    put32(56, 3);       // feature dimension
    put32(60, 8);       // output factor
    put32(64, 264);     // cache geometry
    put32(68, 80);
    put32(72, 13);
    put32(76, 40);
    put32(80, 0);
    put32(84, 1);
    put32(88, 4);       // actual test rows
    profile.speakers = 2;
    profile.dimension = 3;
    profile.frames = 4;
    profile.cache = {10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42};
    profile.predictions = {0.9f, 0.1f, 0.8f, 0.2f, 0.1f, 0.95f, 0.0f, 0.0f};
    profile.silence = {-1, -2, -3};
    return profile;
}

bool row_is_zero(const native_matching::AoscProfile& profile, int row) {
    const float* values = profile.predictions.data() + static_cast<size_t>(row) * profile.speakers;
    for (int i = 0; i < profile.speakers; ++i)
        if (values[i] != 0.0f) return false;
    return true;
}

}  // namespace

int main() {
    try {
        using namespace native_matching;
        const AoscProfile source = fixture();
        const std::vector<uint8_t> encoded = encode_profile(source);
        const AoscProfile parsed = parse_profile(encoded);
        require(parsed.cache == source.cache, "profile cache round-trip changed features");
        require(parsed.predictions == source.predictions, "profile round-trip changed prediction metadata");
        require(speech_rows(parsed, 0) == std::vector<int>({0, 1}), "channel 0 row selection is wrong");
        require(speech_rows(parsed, 1) == std::vector<int>({2}), "channel 1 row selection is wrong");
        require(exclusive_speech_rows(parsed, 0, 1) == std::vector<int>({0, 1}),
                "neutral rows unexpectedly contain the excluded target channel");
        require(exclusive_speech_rows(parsed, 1, 0) == std::vector<int>({2}),
                "reverse neutral row selection is wrong");

        const auto full_a = parse_profile(ablate_channel(encoded, 0));
        require(full_a.frames == source.frames && full_a.dimension == source.dimension &&
                    full_a.speakers == source.speakers,
                "full ablation changed cache tensor shape");
        require(std::equal(source.silence.begin(), source.silence.end(), full_a.cache.begin()),
                "target feature row was not replaced with the stored silence profile");
        require(row_is_zero(full_a, 0) && row_is_zero(full_a, 1),
                "target prediction metadata was not cleared with the feature rows");
        require(!row_is_zero(full_a, 2) && full_a.cache[6] == source.cache[6],
                "full ablation modified a non-target channel row");
        require(full_a.cache[9] == source.cache[9] && row_is_zero(full_a, 3),
                "existing silence sentinel was not preserved");

        const auto partial_target = parse_profile(ablate_matched_rows(encoded, 0, 1));
        const auto partial_control = parse_profile(ablate_matched_rows_excluding(encoded, 1, 0, 1));
        int changed_target_rows = 0;
        int changed_control_rows = 0;
        for (int row = 0; row < source.frames; ++row) {
            if (row_is_zero(partial_target, row) != row_is_zero(source, row)) ++changed_target_rows;
            if (row_is_zero(partial_control, row) != row_is_zero(source, row)) ++changed_control_rows;
        }
        require(changed_target_rows == 1 && changed_control_rows == 1,
                "matched intervention and control changed different numbers of rows");
        require(partial_target.cache.size() == source.cache.size() &&
                    partial_target.predictions.size() == source.predictions.size(),
                "partial intervention changed tensor cardinality");

        const auto reverse_rows = parse_profile(reverse_cache_row_order(encoded));
        require(reverse_rows.cache[0] == source.cache[9] && reverse_rows.predictions[0] == source.predictions[6],
                "cache row reordering failed to keep acoustic features and prediction metadata paired");
        require(reverse_rows.cache.size() == source.cache.size() &&
                    reverse_rows.predictions.size() == source.predictions.size(),
                "cache row reordering changed tensor dimensions");

        auto corrupted = encoded;
        corrupted.back() ^= 1;
        bool rejected = false;
        try { (void)parse_profile(corrupted); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "checksum-corrupted AOSC intervention was accepted");
        std::cout << "AOSC profile intervention invariants passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_aosc_profile_intervention: " << e.what() << '\n';
        return 1;
    }
}
