#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDimension = 512;
constexpr std::size_t kQueries = 12000;

void normalize(std::vector<float>& value) {
    double norm2 = 0.0;
    for (float x : value) norm2 += static_cast<double>(x) * x;
    const float inverse = static_cast<float>(1.0 / std::sqrt(std::max(norm2, 1e-24)));
    for (float& x : value) x *= inverse;
}

}  // namespace

int main() {
    std::mt19937 generator(20261020);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> center(kDimension), scale(kDimension), raw_queries(kQueries * kDimension);
    for (std::size_t d = 0; d < kDimension; ++d) {
        center[d] = normal(generator) * 0.1f;
        scale[d] = 0.5f + std::abs(normal(generator));
    }
    for (float& x : raw_queries) x = normal(generator);

    constexpr std::size_t registry_sizes[] = {1, 2, 4, 8, 16, 64, 256};
    std::cout << "{\"schema\":\"nemo-speech.voiceid-task09c-matching-benchmark.v1\","
              << "\"dimension\":" << kDimension << ",\"query_vectors\":" << kQueries
              << ",\"compiler_mode\":\"-O3 -DNDEBUG -march=native\",\"registry_rows\":[";
    bool first = true;
    volatile double checksum = 0.0;
    for (std::size_t registry_size : registry_sizes) {
        std::vector<std::vector<float>> prototypes(registry_size, std::vector<float>(kDimension));
        for (auto& prototype : prototypes) {
            for (float& x : prototype) x = normal(generator);
            normalize(prototype);
        }
        std::vector<float> query(kDimension);
        double score_sum = 0.0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t q = 0; q < kQueries; ++q) {
            const float* source = raw_queries.data() + q * kDimension;
            double norm2 = 0.0;
            for (std::size_t d = 0; d < kDimension; ++d) {
                query[d] = std::clamp((source[d] - center[d]) / scale[d], -8.0f, 8.0f);
                norm2 += static_cast<double>(query[d]) * query[d];
            }
            const float inverse = static_cast<float>(1.0 / std::sqrt(std::max(norm2, 1e-24)));
            for (float& x : query) x *= inverse;
            for (const auto& prototype : prototypes) {
                double score = 0.0;
                for (std::size_t d = 0; d < kDimension; ++d)
                    score += static_cast<double>(query[d]) * prototype[d];
                score_sum += score;
            }
        }
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        checksum += score_sum;
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"registry_size\":" << registry_size
                  << ",\"query_latency_us\":" << std::setprecision(8) << (seconds * 1e6 / kQueries)
                  << ",\"queries_per_second\":" << (kQueries / seconds)
                  << ",\"multiply_accumulate_terms_per_query\":" << registry_size * kDimension
                  << ",\"prototype_raw_bytes\":" << registry_size * kDimension * sizeof(float) << '}';
    }
    std::cout << "],\"checksum\":" << static_cast<double>(checksum) << "}\n";
    return 0;
}
