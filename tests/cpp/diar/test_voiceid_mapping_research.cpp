// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "voiceid_mapping_research.h"

#include <stdexcept>
#include <vector>

using voiceid_research::assignment_supported;
using voiceid_research::maximum_weight_assignment;
using voiceid_research::validate_imported_identity_map;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    {
        const auto result = maximum_weight_assignment({{10, 0}, {6, 8}});
        require(result.channel_for_identity == std::vector<int>({0, 1}), "valid global assignment");
        require(result.score == 18 && result.optimal_assignments == 1, "unique assignment score");
        const auto supported = assignment_supported(
            result, {{10, 0}, {6, 8}}, {{0.9, 0.01}, {0.55, 0.8}}, {10, 14});
        require(supported == std::vector<bool>({true, true}), "independent evidence supports assignment");
    }
    {
        const auto collision = maximum_weight_assignment({{10, 0}, {9, 0}});
        const auto supported = assignment_supported(
            collision, {{10, 0}, {9, 0}}, {{0.9, 0.01}, {0.8, 0.02}}, {10, 9});
        require(collision.channel_for_identity[0] != collision.channel_for_identity[1], "matching is one-to-one");
        require(supported[0] && !supported[1], "forced unsupported assignment abstains");
    }
    {
        const auto tie = maximum_weight_assignment({{4, 4}, {4, 4}});
        require(tie.optimal_assignments == 2, "tied assignment remains ambiguous");
        const auto weak = maximum_weight_assignment({{3, 1}, {2, 1}});
        const auto supported = assignment_supported(
            weak, {{3, 1}, {2, 1}}, {{0.4, 0.2}, {0.3, 0.2}}, {4, 3});
        require(!supported[0] && !supported[1], "weak evidence is rejected");
    }
    {
        const std::vector<std::string> identities{"alice", "bob"};
        require(validate_imported_identity_map(
                    identities, {{"alice", 1}, {"bob", 0}}, 8),
                "complete unique imported identity map is valid");
        require(!validate_imported_identity_map(
                    identities, {{"alice", 1}, {"bob", 1}}, 8),
                "imported identity map rejects channel collisions");
        require(!validate_imported_identity_map(
                    identities, {{"alice", 1}}, 8),
                "imported identity map rejects missing speakers");
        require(!validate_imported_identity_map(
                    identities, {{"alice", 1}, {"bob", 8}}, 8),
                "imported identity map rejects out-of-range channels");
    }
}
