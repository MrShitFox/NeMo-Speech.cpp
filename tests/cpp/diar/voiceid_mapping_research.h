// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace voiceid_research {

struct GlobalAssignment {
    std::vector<int> channel_for_identity;
    int64_t score = 0;
    int optimal_assignments = 0;  // Capped at 2; 2 means the optimum is tied.
};

inline bool validate_imported_identity_map(
    const std::vector<std::string>& identities,
    const std::map<std::string, int>& mapping,
    int channel_count) {
    if (channel_count <= 0 || mapping.size() != identities.size()) return false;
    std::vector<bool> used(static_cast<size_t>(channel_count), false);
    for (const auto& identity : identities) {
        const auto found = mapping.find(identity);
        if (found == mapping.end() || found->second < 0 || found->second >= channel_count ||
            used[static_cast<size_t>(found->second)])
            return false;
        used[static_cast<size_t>(found->second)] = true;
    }
    return true;
}

// Exhaustive bit-mask assignment is small and deterministic for the model's
// eight output channels. Rows are labeled enrollment identities and columns
// are native speaker channels. This helper only optimizes the diagnostic
// activity count; it does not claim an identity mapping is supported.
inline GlobalAssignment maximum_weight_assignment(
    const std::vector<std::vector<int64_t>>& activity) {
    const size_t identities = activity.size();
    if (identities == 0) return {};
    const size_t channels = activity.front().size();
    if (identities > channels || channels > 8)
        throw std::invalid_argument("global VoiceID assignment requires 1..8 identities and channels");
    for (const auto& row : activity)
        if (row.size() != channels)
            throw std::invalid_argument("global VoiceID activity matrix is ragged");

    GlobalAssignment result;
    result.channel_for_identity.assign(identities, -1);
    std::vector<int> current(identities, -1);
    std::function<void(size_t, unsigned, int64_t)> visit =
        [&](size_t identity, unsigned used, int64_t score) {
            if (identity == identities) {
                if (score > result.score || result.optimal_assignments == 0) {
                    result.score = score;
                    result.channel_for_identity = current;
                    result.optimal_assignments = 1;
                } else if (score == result.score) {
                    result.optimal_assignments = 2;
                }
                return;
            }
            for (size_t channel = 0; channel < channels; ++channel) {
                const unsigned bit = 1u << channel;
                if (used & bit) continue;
                current[identity] = static_cast<int>(channel);
                visit(identity + 1, used | bit, score + activity[identity][channel]);
            }
        };
    visit(0, 0, 0);
    return result;
}

// Independent support guard for a proposed assignment. It deliberately uses
// the same existing enrollment evidence requirements as strict mapping:
// unique top-1 evidence for at least half of active frames and mean channel
// probability >= 0.5. A matching algorithm cannot make an unsupported row
// look valid by assigning it an unused channel.
inline std::vector<bool> assignment_supported(
    const GlobalAssignment& assignment,
    const std::vector<std::vector<int64_t>>& activity,
    const std::vector<std::vector<double>>& mean_probability,
    const std::vector<int64_t>& active_frames,
    double minimum_mean_probability = 0.5) {
    if (activity.size() != assignment.channel_for_identity.size() ||
        mean_probability.size() != activity.size() || active_frames.size() != activity.size())
        throw std::invalid_argument("global VoiceID evidence dimensions disagree");
    std::vector<bool> supported(activity.size(), false);
    for (size_t identity = 0; identity < activity.size(); ++identity) {
        const int channel = assignment.channel_for_identity[identity];
        if (channel < 0 || static_cast<size_t>(channel) >= activity[identity].size() ||
            mean_probability[identity].size() != activity[identity].size() || active_frames[identity] <= 0)
            continue;
        const int64_t count = activity[identity][static_cast<size_t>(channel)];
        supported[identity] =
            count * 2 >= active_frames[identity] &&
            mean_probability[identity][static_cast<size_t>(channel)] >= minimum_mean_probability;
    }
    return supported;
}

}  // namespace voiceid_research
