// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Research-only open-set VoiceID experiments using native Nemotron V3 AOSC.

#include "audio_file.h"
#include "aosc_profile_intervention.h"
#include "diar_enrollment_profile.h"
#include "diarizer.h"
#include "runtime.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace nemo_speech::asr;
using json = nlohmann::json;
namespace fs = std::filesystem;
using clock_type = std::chrono::steady_clock;

namespace {

struct Options {
    std::string manifest;
    std::string model = "models/Nemotron-3-Diarization.q8_0.gguf";
    std::string cache_dir;
    std::string trace_dir;
    std::string profile_dir;
    std::string output_jsonl;
    std::string preflight_wav;
    std::string preflight_out;
    std::string device = "vulkan";
    int gpu_device = 0;
    int threads = 4;
    int concurrency = 4;
    int enroll_gap_ms = 250;
    bool resume = true;
    bool unknown_traffic_only = false;
    std::set<std::string> trial_ids;
};

double elapsed_ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

std::string fnv64_hex(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : value) hash = (hash ^ c) * 1099511628211ULL;
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

std::string hex_u64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

uint64_t peak_rss_bytes() {
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024;
#endif
}

std::string gpu_memory_used_mib() {
    FILE* pipe = popen(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!pipe) return {};
    char buffer[64]{};
    std::string value;
    if (fgets(buffer, sizeof(buffer), pipe)) value = buffer;
    pclose(pipe);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

Options parse_args(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--manifest") out.manifest = next("--manifest");
        else if (arg == "--model") out.model = next("--model");
        else if (arg == "--cache-dir") out.cache_dir = next("--cache-dir");
        else if (arg == "--trace-dir") out.trace_dir = next("--trace-dir");
        else if (arg == "--profile-dir") out.profile_dir = next("--profile-dir");
        else if (arg == "--output-jsonl") out.output_jsonl = next("--output-jsonl");
        else if (arg == "--preflight-wav") out.preflight_wav = next("--preflight-wav");
        else if (arg == "--preflight-out") out.preflight_out = next("--preflight-out");
        else if (arg == "--device") out.device = next("--device");
        else if (arg == "--gpu-device") out.gpu_device = std::stoi(next("--gpu-device"));
        else if (arg == "--threads") out.threads = std::stoi(next("--threads"));
        else if (arg == "--concurrency") out.concurrency = std::stoi(next("--concurrency"));
        else if (arg == "--enroll-gap-ms") out.enroll_gap_ms = std::stoi(next("--enroll-gap-ms"));
        else if (arg == "--trial-id") out.trial_ids.insert(next("--trial-id"));
        else if (arg == "--unknown-traffic-only") out.unknown_traffic_only = true;
        else if (arg == "--no-resume") out.resume = false;
        else if (arg == "--help" || arg == "-h") {
            throw std::runtime_error(
                "usage: diar_vctk_voiceid_probe --manifest FILE --cache-dir DIR --trace-dir DIR "
                "--profile-dir DIR --output-jsonl FILE [--device vulkan --gpu-device 0 "
                "--concurrency 4 --threads 4] [--trial-id ID] [--unknown-traffic-only] [--no-resume]\n"
                "       diar_vctk_voiceid_probe --preflight-wav WAV --preflight-out JSON "
                "[--device cpu|vulkan]");
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (out.device != "cpu" && out.device != "vulkan")
        throw std::runtime_error("--device must be cpu or vulkan");
    if (out.gpu_device < 0 || out.threads <= 0 || out.concurrency <= 0 || out.concurrency > 32 ||
        out.enroll_gap_ms < 0)
        throw std::runtime_error("invalid GPU device, thread, concurrency, or enrollment gap value");
    if (out.device == "cpu" && out.gpu_device != 0)
        throw std::runtime_error("--gpu-device is valid only with --device vulkan");
    if (out.preflight_wav.empty() &&
        (out.manifest.empty() || out.cache_dir.empty() || out.trace_dir.empty() ||
         out.profile_dir.empty() || out.output_jsonl.empty()))
        throw std::runtime_error("benchmark mode requires --manifest, --cache-dir, --trace-dir, "
                                 "--profile-dir, and --output-jsonl");
    if (!out.preflight_wav.empty() && out.preflight_out.empty())
        throw std::runtime_error("preflight mode requires --preflight-out");
    return out;
}

json safe_number(double value) {
    return std::isfinite(value) ? json(value) : json(nullptr);
}

std::string sanitized(std::string value) {
    for (char& c : value)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) c = '_';
    if (value.empty()) value = "unnamed";
    return value;
}

fs::path cache_path(const fs::path& root, const std::string& relative) {
    const fs::path child(relative);
    if (child.is_absolute()) throw std::runtime_error("manifest asset path must be relative to cache-dir");
    const fs::path full = (root / child).lexically_normal();
    const fs::path normalized_root = root.lexically_normal();
    const auto mismatch = std::mismatch(
        normalized_root.begin(), normalized_root.end(), full.begin(), full.end());
    if (mismatch.first != normalized_root.end())
        throw std::runtime_error("manifest asset path escapes cache-dir");
    return full;
}

std::vector<uint8_t> read_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) throw std::runtime_error("cannot size " + path.string());
    std::vector<uint8_t> result(static_cast<size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!result.empty()) input.read(reinterpret_cast<char*>(result.data()), result.size());
    if (!input) throw std::runtime_error("short read from " + path.string());
    return result;
}

json array_of_ints(const std::vector<int>& values) {
    json out = json::array();
    for (int value : values) out.push_back(value);
    return out;
}

json profile_row_summary(const std::vector<uint8_t>& bytes) {
    const native_matching::AoscProfile profile = native_matching::parse_profile(bytes);
    std::vector<int> positive(profile.speakers, 0);
    std::vector<int> argmax(profile.speakers, 0);
    int overlapping_rows = 0;
    for (int f = 0; f < profile.frames; ++f) {
        const float* row = profile.predictions.data() + static_cast<size_t>(f) * profile.speakers;
        int best = 0;
        int positive_count = 0;
        for (int channel = 0; channel < profile.speakers; ++channel) {
            if (row[channel] > row[best]) best = channel;
            if (row[channel] > 0.5f) {
                ++positive[channel];
                ++positive_count;
            }
        }
        if (row[best] > 0.5f) ++argmax[best];
        if (positive_count > 1) ++overlapping_rows;
    }
    return {
        {"cache_frames", profile.frames},
        {"speaker_channels", profile.speakers},
        {"embedding_dim", profile.dimension},
        {"positive_rows_by_channel_gt_0_5", array_of_ints(positive)},
        {"argmax_rows_by_channel_gt_0_5", array_of_ints(argmax)},
        {"rows_with_multiple_positive_channel_metadata", overlapping_rows},
        {"crc32", native_matching::read_u32(bytes.data() + 109)},
    };
}

void write_u32_le(std::ostream& output, uint32_t value) {
    for (int i = 0; i < 4; ++i) output.put(static_cast<char>(value >> (i * 8)));
}
void write_u64_le(std::ostream& output, uint64_t value) {
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(value >> (i * 8)));
}
void write_f64_le(std::ostream& output, double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u64_le(output, bits);
}

uint64_t write_trace(
    const fs::path& trace_dir, const std::string& name, const std::vector<float>& probabilities,
    int64_t frames, int channels, double frame_period, std::string& relative_path) {
    if (frames < 0 || channels <= 0 || probabilities.size() != static_cast<size_t>(frames) * channels)
        throw std::runtime_error("probability trace does not match its declared frame/channel shape");
    fs::create_directories(trace_dir);
    const std::string filename = sanitized(name) + ".nmsp8f1";
    const fs::path output_path = trace_dir / filename;
    const fs::path temporary = output_path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create probability trace " + temporary.string());
    output.write("NMSP8F1\0", 8);
    write_u32_le(output, 1);
    write_u32_le(output, static_cast<uint32_t>(channels));
    write_u64_le(output, static_cast<uint64_t>(frames));
    write_f64_le(output, frame_period);
    uint64_t hash = 1469598103934665603ULL;
    for (float value : probabilities) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int i = 0; i < 4; ++i) {
            const uint8_t byte = static_cast<uint8_t>(bits >> (i * 8));
            output.put(static_cast<char>(byte));
            hash = (hash ^ byte) * 1099511628211ULL;
        }
    }
    output.flush();
    if (!output) throw std::runtime_error("failed writing probability trace " + temporary.string());
    output.close();
    fs::rename(temporary, output_path);
    relative_path = filename;
    return hash;
}

void write_private_profile(const fs::path& path, const std::vector<uint8_t>& bytes) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot create private profile " + temporary.string());
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        output.flush();
        if (!output) throw std::runtime_error("failed writing private profile " + temporary.string());
    }
    if (chmod(temporary.c_str(), S_IRUSR | S_IWUSR) != 0)
        throw std::runtime_error("could not set private profile permissions");
    fs::rename(temporary, path);
}

struct AudioAsset {
    nemo_speech::audio::AudioFile audio;
    std::vector<uint8_t> mask;
    std::string audio_file;
    std::string mask_file;
};

AudioAsset load_asset(const json& item, const fs::path& cache_dir) {
    AudioAsset out;
    out.audio_file = item.at("audio_file").get<std::string>();
    out.mask_file = item.at("vad_mask_file").get<std::string>();
    out.audio = nemo_speech::audio::load_wav_file(cache_path(cache_dir, out.audio_file).string());
    out.mask = read_bytes(cache_path(cache_dir, out.mask_file));
    if (out.audio.sample_rate != 16000 || out.audio.source_channels != 1)
        throw std::runtime_error("prepared WAV is not mono 16 kHz: " + out.audio_file);
    if (out.mask.size() != (out.audio.samples.size() + 159) / 160)
        throw std::runtime_error("10 ms VAD mask length disagrees with WAV samples: " + out.audio_file);
    return out;
}

struct InferenceOutput {
    std::vector<float> probabilities;
    int64_t frames = 0;
    int64_t base = 0;
    int channels = 0;
    double frame_period = 0.0;
    std::array<int64_t, 3> tensor_shape{{0, 0, 0}};
    double import_ms = 0.0;
    double inference_ms = 0.0;
};

InferenceOutput run_with_profile(
    const std::shared_ptr<Diarizer>& diarizer, const std::vector<uint8_t>& profile,
    const nemo_speech::audio::AudioFile& input) {
    auto stream = diarizer->streaming_diarize();
    const auto import_start = clock_type::now();
    DiarAoscProfileCodec::decode(*stream, profile.data(), profile.size());
    const double import_ms = elapsed_ms(import_start);
    const auto inference_start = clock_type::now();
    stream->push(input.samples.data(), input.samples.size(), input.sample_rate);
    stream->finish();
    const double inference_ms = elapsed_ms(inference_start);
    InferenceOutput out;
    out.probabilities = stream->frame_probabilities();
    out.frames = stream->frame_count();
    out.base = stream->frame_probs_start();
    out.channels = stream->num_speakers();
    out.frame_period = stream->seconds_per_frame();
    out.tensor_shape = stream->last_prediction_tensor_shape();
    out.import_ms = import_ms;
    out.inference_ms = inference_ms;
    if (out.frames - out.base < 0 || out.probabilities.size() !=
            static_cast<size_t>(out.frames - out.base) * out.channels)
        throw std::runtime_error("stream probability vector is not frame-major [frames, channels]");
    return out;
}

struct FrameSummary {
    int64_t speech_frames = 0;
    int64_t activity_frames = 0;
    std::vector<int64_t> channel_top1_counts;
    std::vector<double> channel_mean;
    std::vector<double> channel_mean_margin;
    int dominant_channel = -1;
    bool top1_tie = false;
    double dominance = 0.0;
    double mean_margin = 0.0;
    int64_t last_frame = -1;
};

FrameSummary summarize_frames(
    const InferenceOutput& output, const std::vector<uint8_t>& mask, int64_t start_frame) {
    FrameSummary result;
    result.channel_top1_counts.assign(output.channels, 0);
    result.channel_mean.assign(output.channels, 0.0);
    result.channel_mean_margin.assign(output.channels, 0.0);
    std::vector<double> margin_sum(output.channels, 0.0);
    std::vector<int64_t> channel_frames(output.channels, 0);
    int64_t margin_count = 0;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] == 0) continue;
        ++result.speech_frames;
        const int64_t absolute = start_frame + static_cast<int64_t>(i);
        const int64_t frame = absolute - output.base;
        if (frame < 0 || frame >= output.frames - output.base) continue;
        result.last_frame = frame;
        const float* p = output.probabilities.data() + static_cast<size_t>(frame) * output.channels;
        int top1 = 0;
        int top2 = output.channels > 1 ? 1 : 0;
        if (output.channels > 1 && p[top2] > p[top1]) std::swap(top1, top2);
        for (int c = 2; c < output.channels; ++c) {
            if (p[c] > p[top1]) {
                top2 = top1;
                top1 = c;
            } else if (p[c] > p[top2]) {
                top2 = c;
            }
        }
        for (int c = 0; c < output.channels; ++c) {
            result.channel_mean[c] += p[c];
            ++channel_frames[c];
        }
        if (p[top1] >= 0.5f) {
            ++result.activity_frames;
            ++result.channel_top1_counts[top1];
            margin_sum[0] += p[top1] - p[top2];
            ++margin_count;
        }
    }
    for (int c = 0; c < output.channels; ++c)
        if (channel_frames[c] > 0) result.channel_mean[c] /= channel_frames[c];
    if (result.activity_frames > 0) {
        const auto best = std::max_element(result.channel_top1_counts.begin(), result.channel_top1_counts.end());
        const int best_channel = static_cast<int>(best - result.channel_top1_counts.begin());
        const int best_count = *best;
        const int ties = static_cast<int>(std::count(
            result.channel_top1_counts.begin(), result.channel_top1_counts.end(), best_count));
        result.top1_tie = ties != 1;
        result.dominance = static_cast<double>(best_count) / result.activity_frames;
        if (!result.top1_tie) result.dominant_channel = best_channel;
        result.mean_margin = margin_count ? margin_sum[0] / margin_count : 0.0;
    }
    return result;
}

json frame_summary_json(const FrameSummary& summary, double frame_period) {
    json means = json::array();
    for (double value : summary.channel_mean) means.push_back(safe_number(value));
    json counts = json::array();
    for (int64_t value : summary.channel_top1_counts) counts.push_back(value);
    return {
        {"vad_speech_frames", summary.speech_frames},
        {"vad_speech_duration_sec", summary.speech_frames * frame_period},
        {"activity_frames_max_channel_ge_0_5", summary.activity_frames},
        {"channel_mean_probability_over_vad_frames", means},
        {"top1_frame_counts_over_active_vad_frames", counts},
        {"dominant_channel", summary.dominant_channel < 0 ? json(nullptr) : json(summary.dominant_channel)},
        {"dominant_channel_share", safe_number(summary.dominance)},
        {"mean_top1_top2_margin_over_active_vad_frames", safe_number(summary.mean_margin)},
        {"top1_tie", summary.top1_tie},
    };
}

struct EnrollmentStats {
    int candidate_channel = -1;
    bool established = false;
    bool tied = false;
    int64_t speech_frames = 0;
    int64_t active_frames = 0;
    double dominance = 0.0;
    double mean_winner_probability = 0.0;
    std::vector<int64_t> top1_counts;
    std::vector<double> mean_probability;
};

json serialize_enrollment_stats(const EnrollmentStats& stats) {
    json counts = json::array();
    for (auto value : stats.top1_counts) counts.push_back(value);
    json means = json::array();
    for (auto value : stats.mean_probability) means.push_back(safe_number(value));
    return {
        {"candidate_channel", stats.candidate_channel < 0 ? json(nullptr) : json(stats.candidate_channel)},
        {"channel_established", stats.established},
        {"dominant_channel_tie", stats.tied},
        {"vad_speech_frames", stats.speech_frames},
        {"model_activity_frames", stats.active_frames},
        {"dominant_channel_share", stats.dominance},
        {"mean_probability_on_candidate_channel", safe_number(stats.mean_winner_probability)},
        {"top1_frames_by_channel", counts},
        {"mean_probability_by_channel", means},
    };
}

struct EnrollmentClipRef {
    std::string speaker;
    int64_t start_frame = 0;
    AudioAsset asset;
};

std::map<std::string, EnrollmentStats> analyze_enrollment(
    const std::vector<EnrollmentClipRef>& clips, const InferenceOutput& output,
    int channels, const std::vector<std::string>& enrolled_ids) {
    std::map<std::string, EnrollmentStats> result;
    std::map<std::string, std::vector<double>> sums;
    std::map<std::string, std::vector<int64_t>> counts;
    for (const std::string& id : enrolled_ids) {
        result[id].top1_counts.assign(channels, 0);
        result[id].mean_probability.assign(channels, 0.0);
        sums[id].assign(channels, 0.0);
        counts[id].assign(channels, 0);
    }
    for (const auto& clip : clips) {
        auto& stats = result[clip.speaker];
        for (size_t i = 0; i < clip.asset.mask.size(); ++i) {
            if (clip.asset.mask[i] == 0) continue;
            ++stats.speech_frames;
            const int64_t frame = clip.start_frame + static_cast<int64_t>(i) - output.base;
            if (frame < 0 || frame >= output.frames - output.base) continue;
            const float* p = output.probabilities.data() + static_cast<size_t>(frame) * channels;
            int winner = static_cast<int>(std::max_element(p, p + channels) - p);
            if (p[winner] >= 0.5f) {
                ++stats.active_frames;
                ++stats.top1_counts[winner];
            }
            for (int c = 0; c < channels; ++c) {
                sums[clip.speaker][c] += p[c];
                ++counts[clip.speaker][c];
            }
        }
    }
    for (const std::string& id : enrolled_ids) {
        auto& stats = result[id];
        for (int c = 0; c < channels; ++c)
            if (counts[id][c]) stats.mean_probability[c] = sums[id][c] / counts[id][c];
        if (stats.active_frames == 0) continue;
        const int best_count = *std::max_element(stats.top1_counts.begin(), stats.top1_counts.end());
        const int ties = static_cast<int>(std::count(stats.top1_counts.begin(), stats.top1_counts.end(), best_count));
        stats.tied = ties != 1;
        stats.dominance = static_cast<double>(best_count) / stats.active_frames;
        if (!stats.tied) {
            stats.candidate_channel = static_cast<int>(std::max_element(
                stats.top1_counts.begin(), stats.top1_counts.end()) - stats.top1_counts.begin());
            stats.mean_winner_probability = stats.mean_probability[stats.candidate_channel];
            stats.established = stats.dominance >= 0.5 && stats.mean_winner_probability >= 0.5;
        }
    }
    return result;
}

json identity_map_json(const std::map<std::string, int>& identity_to_channel) {
    json out = json::object();
    for (const auto& item : identity_to_channel) out[item.first] = item.second;
    return out;
}

std::string decide_category(
    bool known, const std::string& speaker, int channel,
    const std::map<std::string, int>& identity_to_channel, bool no_decision) {
    if (no_decision) return "no_decision";
    std::string predicted;
    for (const auto& item : identity_to_channel)
        if (item.second == channel) predicted = item.first;
    if (predicted.empty()) return known ? "known_rejected_as_unknown" : "unknown_correctly_rejected";
    if (!known) return "unknown_false_accept";
    return predicted == speaker ? "known_correct" : "known_wrong_enrolled_identity";
}

json query_output(
    const json& item, const std::shared_ptr<Diarizer>& diarizer,
    const std::vector<uint8_t>& profile, const std::map<std::string, int>& mapping,
    bool mapping_valid, const fs::path& cache_dir, const fs::path& trace_dir,
    const std::string& trace_name, double stream_start_sec = 0.0) {
    AudioAsset asset = load_asset(item, cache_dir);
    InferenceOutput inference = run_with_profile(diarizer, profile, asset.audio);
    FrameSummary summary = summarize_frames(
        inference, asset.mask, static_cast<int64_t>(std::llround(stream_start_sec / inference.frame_period)));
    const bool no_decision = summary.activity_frames == 0 || summary.dominant_channel < 0;
    const bool known = item.value("known", false);
    const std::string speaker = item.value("speaker_id", std::string());
    std::string category = mapping_valid
        ? decide_category(known, speaker, summary.dominant_channel, mapping, no_decision)
        : "not_evaluated_ambiguous_enrollment_mapping";
    std::string trace_file;
    const uint64_t trace_hash = write_trace(
        trace_dir, trace_name, inference.probabilities, inference.frames - inference.base,
        inference.channels, inference.frame_period, trace_file);
    json registered_means = json::object();
    double max_registered_mean = -1.0;
    for (const auto& entry : mapping) {
        const double mean = summary.channel_mean[entry.second];
        registered_means[entry.first] = mean;
        max_registered_mean = std::max(max_registered_mean, mean);
    }
    return {
        {"query_id", item.value("query_id", item.value("conversation_id", std::string("query")))},
        {"speaker_id", speaker.empty() ? json(nullptr) : json(speaker)},
        {"known", known},
        {"microphone", item.value("mic", std::string("mixed"))},
        {"duration_bucket", item.value("duration_bucket", std::string("conversation"))},
        {"source_utterance_id", item.value("utterance_id", std::string())},
        {"source_partition", item.value("source_partition", std::string())},
        {"query_evaluated", true},
        {"recording_duration_sec", item.value("duration_sec", static_cast<double>(asset.audio.samples.size()) / asset.audio.sample_rate)},
        {"actual_vad_speech_duration_sec", summary.speech_frames * inference.frame_period},
        {"activity_frames", summary.activity_frames},
        {"no_decision", no_decision},
        {"dominant_channel", summary.dominant_channel < 0 ? json(nullptr) : json(summary.dominant_channel)},
        {"predicted_identity", no_decision ? json(nullptr) :
             (std::find_if(mapping.begin(), mapping.end(), [&](const auto& entry) {
                  return entry.second == summary.dominant_channel;
              }) == mapping.end() ? json("UNKNOWN") :
              json(std::find_if(mapping.begin(), mapping.end(), [&](const auto& entry) {
                  return entry.second == summary.dominant_channel;
              })->first))},
        {"baseline_category", category},
        {"top1_dominance", summary.dominance},
        {"mean_top1_top2_margin", summary.mean_margin},
        {"channel_mean_probability_over_vad_frames", frame_summary_json(summary, inference.frame_period)["channel_mean_probability_over_vad_frames"]},
        {"registered_channel_means", registered_means},
        {"diagnostic_max_registered_channel_mean", safe_number(max_registered_mean)},
        {"frame_count", inference.frames - inference.base},
        {"num_speakers", inference.channels},
        {"seconds_per_output_frame", inference.frame_period},
        {"last_prediction_tensor_shape_channels_frames_batch", inference.tensor_shape},
        {"profile_import_ms", inference.import_ms},
        {"inference_ms", inference.inference_ms},
        {"audio_per_inference_second", inference.inference_ms > 0
             ? static_cast<double>(asset.audio.samples.size()) / asset.audio.sample_rate / (inference.inference_ms / 1000.0)
             : 0.0},
        {"trace_file", trace_file},
        {"trace_probability_hash_fnv1a64", hex_u64(trace_hash)},
        {"trace_layout", "little-endian float32 frame-major [frame, channel], NMSP8F1 v1"},
    };
}

struct EnrollmentBuild {
    std::vector<uint8_t> profile;
    fs::path profile_path;
    std::vector<EnrollmentClipRef> clips;
    std::map<std::string, EnrollmentStats> speaker_stats;
    std::map<std::string, int> mapping;
    bool mapping_valid = false;
    std::string mapping_error;
    json row_summary = nullptr;
    json retained_rows_by_identity = json::object();
    int64_t frames = 0;
    double audio_sec = 0.0;
    double speech_sec = 0.0;
    double build_ms = 0.0;
    std::string not_ready_reason;
    bool ready = false;
};

EnrollmentBuild build_profile(
    const json& trial, const std::shared_ptr<Diarizer>& diarizer,
    const fs::path& cache_dir, const fs::path& profile_dir, int gap_ms) {
    EnrollmentBuild result;
    const auto start = clock_type::now();
    const std::vector<std::string> ids = trial.at("enrolled_speaker_ids").get<std::vector<std::string>>();
    auto stream = diarizer->streaming_diarize();
    const std::vector<float> silence(static_cast<size_t>(gap_ms) * 16, 0.0f);
    int64_t cursor_samples = 0;
    for (const auto& item : trial.at("enrollment")) {
        AudioAsset asset = load_asset(item, cache_dir);
        if (asset.audio.sample_rate != diarizer->sample_rate())
            throw std::runtime_error("enrollment sample rate differs from loaded GGUF");
        const int64_t start_frame = cursor_samples / 160;
        stream->push(asset.audio.samples.data(), asset.audio.samples.size(), asset.audio.sample_rate);
        result.audio_sec += static_cast<double>(asset.audio.samples.size()) / asset.audio.sample_rate;
        result.speech_sec += static_cast<double>(std::count(asset.mask.begin(), asset.mask.end(), uint8_t{1})) * 0.01;
        cursor_samples += static_cast<int64_t>(asset.audio.samples.size());
        const size_t padding = static_cast<size_t>((160 - (cursor_samples % 160)) % 160);
        if (padding) {
            std::vector<float> pad(padding, 0.0f);
            stream->push(pad.data(), pad.size(), asset.audio.sample_rate);
            cursor_samples += static_cast<int64_t>(padding);
        }
        result.clips.push_back({item.at("speaker_id").get<std::string>(), start_frame, std::move(asset)});
        if (!silence.empty()) {
            stream->push(silence.data(), silence.size(), diarizer->sample_rate());
            cursor_samples += static_cast<int64_t>(silence.size());
            result.audio_sec += static_cast<double>(silence.size()) / diarizer->sample_rate();
        }
    }
    stream->finish();
    result.frames = stream->frame_count();
    result.speaker_stats = analyze_enrollment(
        result.clips, InferenceOutput{
            stream->frame_probabilities(), stream->frame_count(), stream->frame_probs_start(),
            stream->num_speakers(), stream->seconds_per_frame(), stream->last_prediction_tensor_shape()},
        stream->num_speakers(), ids);
    try {
        result.profile = DiarAoscProfileCodec::encode(*stream);
    } catch (const std::exception& error) {
        // AOSC readiness is a property of the native cache state. File-system
        // errors after successful encoding must fail the run, not masquerade
        // as a failed profile-construction attempt.
        result.not_ready_reason = error.what();
    }
    if (!result.profile.empty()) {
        result.profile_path = profile_dir / (sanitized(trial.at("trial_id").get<std::string>()) + ".nmsaosc");
        write_private_profile(result.profile_path, result.profile);
        // The evaluation context must come from the saved immutable bytes.
        result.profile = read_bytes(result.profile_path);
        result.row_summary = profile_row_summary(result.profile);
        result.ready = true;
    }
    std::map<int, std::vector<std::string>> channel_owners;
    bool all_established = true;
    for (const auto& id : ids) {
        const auto& stats = result.speaker_stats.at(id);
        if (!stats.established || stats.candidate_channel < 0) {
            all_established = false;
            continue;
        }
        channel_owners[stats.candidate_channel].push_back(id);
    }
    bool collision = false;
    for (const auto& owner : channel_owners) {
        if (owner.second.size() > 1) collision = true;
        else result.mapping[owner.second.front()] = owner.first;
    }
    result.mapping_valid = all_established && !collision && result.mapping.size() == ids.size();
    if (result.ready) {
        const auto& positive = result.row_summary.at("positive_rows_by_channel_gt_0_5");
        const auto& argmax = result.row_summary.at("argmax_rows_by_channel_gt_0_5");
        for (const auto& item : result.mapping) {
            result.retained_rows_by_identity[item.first] = {
                {"channel", item.second},
                {"positive_rows_gt_0_5_including_overlaps", positive.at(item.second)},
                {"argmax_rows_gt_0_5", argmax.at(item.second)},
            };
        }
    }
    if (!all_established) result.mapping_error = "one or more speakers did not establish a stable channel from VAD speech frames";
    else if (collision) result.mapping_error = "multiple enrolled identities selected the same native channel";
    result.build_ms = elapsed_ms(start);
    return result;
}

json distribution_from_bytes(const std::vector<uint8_t>& bytes) {
    return profile_row_summary(bytes);
}

json contamination_sequence(
    const json& trial, const EnrollmentBuild& enrollment,
    const std::shared_ptr<Diarizer>& diarizer, const fs::path& cache_dir,
    const fs::path& trace_dir, int gap_ms, bool probe_each_stage) {
    json out = {
        {"status", "skipped"},
        {"reason", "profile not ready or identity mapping is ambiguous"},
        {"immutable_enrollment_profile_preserved", true},
    };
    if (!enrollment.ready || !enrollment.mapping_valid) return out;
    const auto& ids = trial.at("unknown_traffic_query_ids");
    if (!ids.is_array() || ids.empty()) {
        out["reason"] = "no unknown traffic queries in the manifest";
        return out;
    }
    std::map<std::string, json> query_by_id;
    for (const auto& query : trial.at("queries")) query_by_id[query.at("query_id").get<std::string>()] = query;
    json known_query = nullptr;
    for (const auto& query : trial.at("queries")) {
        if (query.value("known", false) && query.value("mic", std::string("mic1")) == "mic1") {
            known_query = query;
            break;
        }
    }
    if (known_query.is_null()) {
        out["status"] = "partial";
        out["reason"] = "no known mic1 query was present";
        return out;
    }
    AudioAsset known_audio = load_asset(known_query, cache_dir);
    json independent_probe = nullptr;
    if (probe_each_stage) {
        independent_probe = query_output(
            known_query, diarizer, enrollment.profile, enrollment.mapping, enrollment.mapping_valid,
            cache_dir, trace_dir, trial.at("trial_id").get<std::string>() + "_known_return_fresh");
    }
    auto stream = diarizer->streaming_diarize();
    DiarAoscProfileCodec::decode(*stream, enrollment.profile.data(), enrollment.profile.size());
    json stages = json::array();
    int64_t cursor_samples = 0;
    std::vector<float> silence(static_cast<size_t>(std::max(1000, gap_ms)) * 16, 0.0f);
    std::set<std::string> unknown_seen;
    for (const auto& id_json : ids) {
        const std::string id = id_json.get<std::string>();
        auto found = query_by_id.find(id);
        if (found == query_by_id.end()) continue;
        const json& item = found->second;
        AudioAsset asset = load_asset(item, cache_dir);
        stream->push(asset.audio.samples.data(), asset.audio.samples.size(), asset.audio.sample_rate);
        cursor_samples += static_cast<int64_t>(asset.audio.samples.size());
        const size_t padding = static_cast<size_t>((160 - (cursor_samples % 160)) % 160);
        if (padding) {
            std::vector<float> pad(padding, 0.0f);
            stream->push(pad.data(), pad.size(), asset.audio.sample_rate);
            cursor_samples += static_cast<int64_t>(padding);
        }
        stream->push(silence.data(), silence.size(), diarizer->sample_rate());
        cursor_samples += static_cast<int64_t>(silence.size());
        unknown_seen.insert(item.at("speaker_id").get<std::string>());
        std::vector<uint8_t> current = DiarAoscProfileCodec::encode(*stream);
        json stage = {
            {"after_unknown_speakers", unknown_seen.size()},
            {"last_unknown_speaker_id", item.at("speaker_id")},
            {"last_unknown_query_id", id},
            {"cache_rows", distribution_from_bytes(current)},
            {"temporary_profile_changed", current != enrollment.profile},
        };
        if (probe_each_stage) {
            stage["known_return_from_fresh_import_of_accumulated_aosc"] = query_output(
                known_query, diarizer, current, enrollment.mapping, enrollment.mapping_valid,
                cache_dir, trace_dir,
                trial.at("trial_id").get<std::string>() + "_after_unknown_" +
                    std::to_string(unknown_seen.size()) + "_known_return");
        }
        stages.push_back(std::move(stage));
    }
    const int64_t known_start = cursor_samples / 160;
    stream->push(known_audio.audio.samples.data(), known_audio.audio.samples.size(), known_audio.audio.sample_rate);
    stream->finish();
    InferenceOutput inference{
        stream->frame_probabilities(), stream->frame_count(), stream->frame_probs_start(),
        stream->num_speakers(), stream->seconds_per_frame(), stream->last_prediction_tensor_shape()};
    FrameSummary summary = summarize_frames(inference, known_audio.mask, known_start);
    const bool no_decision = summary.activity_frames == 0 || summary.dominant_channel < 0;
    const std::string category = decide_category(
        true, known_query.at("speaker_id").get<std::string>(), summary.dominant_channel,
        enrollment.mapping, no_decision);
    std::string trace_file;
    write_trace(trace_dir, trial.at("trial_id").get<std::string>() + "_unknown_traffic_return",
                inference.probabilities, inference.frames - inference.base, inference.channels,
                inference.frame_period, trace_file);
    const std::vector<uint8_t> final_profile = DiarAoscProfileCodec::encode(*stream);
    out = {
        {"status", "complete"},
        {"unknown_speaker_count", unknown_seen.size()},
        {"unknown_query_order", ids},
        {"stages", stages},
        {"independent_fresh_session_probe", independent_probe},
        {"known_return_query_id", known_query.at("query_id")},
        {"known_return_category", category},
        {"known_return_dominant_channel", summary.dominant_channel < 0 ? json(nullptr) : json(summary.dominant_channel)},
        {"known_return_metrics", frame_summary_json(summary, inference.frame_period)},
        {"known_return_trace_file", trace_file},
        {"final_temporary_cache_rows", distribution_from_bytes(final_profile)},
        {"temporary_profile_changed", final_profile != enrollment.profile},
        {"immutable_enrollment_profile_preserved", true},
    };
    return out;
}

json run_conversation(
    const json& conversation, const std::shared_ptr<Diarizer>& diarizer,
    const EnrollmentBuild& enrollment, const fs::path& cache_dir, const fs::path& trace_dir,
    const std::string& trial_id) {
    if (!enrollment.ready || !enrollment.mapping_valid)
        return {{"conversation_id", conversation.value("conversation_id", std::string())},
                {"status", "not_evaluated_profile_or_mapping_invalid"}};
    AudioAsset asset = load_asset(conversation, cache_dir);
    InferenceOutput inference = run_with_profile(diarizer, enrollment.profile, asset.audio);
    std::string trace_file;
    const uint64_t hash = write_trace(
        trace_dir, trial_id + "_" + conversation.at("conversation_id").get<std::string>(),
        inference.probabilities, inference.frames - inference.base, inference.channels,
        inference.frame_period, trace_file);
    return {
        {"conversation_id", conversation.at("conversation_id")},
        {"scenario", conversation.value("scenario", std::string())},
        {"status", "complete"},
        {"frame_count", inference.frames - inference.base},
        {"num_speakers", inference.channels},
        {"seconds_per_output_frame", inference.frame_period},
        {"trace_file", trace_file},
        {"trace_probability_hash_fnv1a64", hex_u64(hash)},
        {"reference_tracks", conversation.value("reference_tracks", json::array())},
        {"sources", conversation.value("sources", json::array())},
        {"mapping", identity_map_json(enrollment.mapping)},
        {"mapping_valid", enrollment.mapping_valid},
        {"profile_import_ms", inference.import_ms},
        {"inference_ms", inference.inference_ms},
    };
}

json run_trial(
    const json& trial, const std::string& manifest_fingerprint,
    const std::shared_ptr<Diarizer>& diarizer, const fs::path& cache_dir,
    const fs::path& trace_dir, const fs::path& profile_dir, int concurrency, int enroll_gap_ms,
    const json& model_info, const std::string& selected_device, bool unknown_traffic_only) {
    const auto trial_start = clock_type::now();
    EnrollmentBuild enrollment = build_profile(trial, diarizer, cache_dir, profile_dir, enroll_gap_ms);
    json speaker_stats = json::object();
    for (const auto& entry : enrollment.speaker_stats)
        speaker_stats[entry.first] = serialize_enrollment_stats(entry.second);
    json result = {
        {"schema", "nemo-speech.voiceid-task08-result.v1"},
        {"manifest_fingerprint_fnv1a64", manifest_fingerprint},
        {"trial_id", trial.at("trial_id")},
        {"cohort_id", trial.at("cohort_id")},
        {"population", trial.at("population")},
        {"evaluation_partition", trial.at("evaluation_partition")},
        {"seed", trial.at("seed")},
        {"group_size", trial.at("group_size")},
        {"enrollment_budget_speech_sec_per_person", trial.at("enrollment_budget_speech_sec_per_person")},
        {"enrollment_order_variant", trial.at("enrollment_order_variant")},
        {"enrollment_order", trial.at("enrollment_order")},
        {"enrolled_speaker_ids", trial.at("enrolled_speaker_ids")},
        {"unknown_speaker_ids", trial.at("unknown_speaker_ids")},
        {"model", model_info},
        {"backend", { {"name", model_info.value("selected_gpu_backend", std::string("CPU"))}, {"device", selected_device} }},
        {"aosc_ready", enrollment.ready},
        {"aosc_not_ready_reason", enrollment.not_ready_reason.empty() ? json(nullptr) : json(enrollment.not_ready_reason)},
        {"profile_saved_and_reloaded", enrollment.ready},
        {"profile_size_bytes", enrollment.profile.size()},
        {"profile_crc32", enrollment.ready ? json(native_matching::read_u32(enrollment.profile.data() + 109)) : json(nullptr)},
        {"cache_row_summary", enrollment.row_summary},
        {"retained_rows_by_enrolled_identity", enrollment.retained_rows_by_identity},
        {"enrollment_speech_duration_sec", enrollment.speech_sec},
        {"enrollment_audio_duration_sec_with_gaps", enrollment.audio_sec},
        {"enrollment_frame_count", enrollment.frames},
        {"speaker_channel_establishment", speaker_stats},
        {"identity_to_channel", identity_map_json(enrollment.mapping)},
        {"identity_mapping_valid", enrollment.mapping_valid},
        {"identity_mapping_error", enrollment.mapping_error.empty() ? json(nullptr) : json(enrollment.mapping_error)},
        {"timing", {{"enrollment_construction_ms", enrollment.build_ms}}},
        {"queries", json::array()},
        {"conversations", json::array()},
        {"unknown_traffic", json(nullptr)},
        {"evaluation_mode", unknown_traffic_only ? "unknown_traffic_only" : "full_trial"},
    };

    if (enrollment.ready && enrollment.mapping_valid) {
        if (!unknown_traffic_only) {
            const json& queries = trial.at("queries");
            for (size_t batch_start = 0; batch_start < queries.size(); batch_start += concurrency) {
                const size_t batch_end = std::min(queries.size(), batch_start + static_cast<size_t>(concurrency));
                std::vector<std::future<json>> calls;
                calls.reserve(batch_end - batch_start);
                for (size_t i = batch_start; i < batch_end; ++i) {
                    const json query = queries[i];
                    calls.emplace_back(std::async(std::launch::async, [&, query] {
                        ScopedBatchCohort cohort(concurrency);
                        return query_output(
                            query, diarizer, enrollment.profile, enrollment.mapping, enrollment.mapping_valid,
                            cache_dir, trace_dir, trial.at("trial_id").get<std::string>() + "_" +
                                query.at("query_id").get<std::string>());
                    }));
                }
                for (auto& call : calls) result["queries"].push_back(call.get());
            }
            for (const auto& conversation : trial.value("conversations", json::array()))
                result["conversations"].push_back(run_conversation(
                    conversation, diarizer, enrollment, cache_dir, trace_dir,
                    trial.at("trial_id").get<std::string>()));
        }
        result["unknown_traffic"] = trial.value("run_unknown_traffic", false)
            ? contamination_sequence(
                  trial, enrollment, diarizer, cache_dir, trace_dir, enroll_gap_ms, unknown_traffic_only)
            : json{{"status", "not_scheduled"}, {"immutable_enrollment_profile_preserved", true}};
    } else {
        result["evaluation_status"] = enrollment.ready
            ? "skipped_ambiguous_enrollment_mapping" : "skipped_aosc_profile_not_ready";
        for (const auto& query : trial.at("queries")) {
            result["queries"].push_back({
                {"query_id", query.at("query_id")},
                {"speaker_id", query.at("speaker_id")},
                {"known", query.at("known")},
                {"microphone", query.value("mic", std::string())},
                {"duration_bucket", query.value("duration_bucket", std::string())},
                {"recording_duration_sec", query.value("duration_sec", 0.0)},
                {"actual_vad_speech_duration_sec", query.value("speech_duration_sec", 0.0)},
                {"query_evaluated", false},
                {"baseline_category", enrollment.ready
                     ? "not_evaluated_ambiguous_enrollment_mapping" : "not_evaluated_profile_not_ready"},
            });
        }
    }
    if (enrollment.ready) {
        const bool immutable = read_bytes(enrollment.profile_path) == enrollment.profile;
        result["profile_immutable_after_evaluation"] = immutable;
        if (!immutable) throw std::runtime_error("stored enrollment profile changed during evaluation");
    }
    result["timing"]["trial_end_to_end_ms"] = elapsed_ms(trial_start);
    result["host_peak_rss_bytes"] = peak_rss_bytes();
    const std::string gpu_memory = gpu_memory_used_mib();
    result["gpu_memory_used_mib_sample"] = gpu_memory.empty()
        ? json(nullptr) : safe_number(std::stod(gpu_memory));
    return result;
}

json model_info(const DiarModel& model) {
    const auto& cfg = model.cfg();
    std::ostringstream fingerprint;
    fingerprint << std::hex << std::setfill('0');
    for (uint8_t byte : model.model_fingerprint()) fingerprint << std::setw(2) << static_cast<int>(byte);
    return {
        {"gguf", "Nemotron-3-Diarization.q8_0.gguf"},
        {"model_fingerprint_sha256", fingerprint.str()},
        {"sortformer_num_speakers", cfg.num_speakers},
        {"cfg_num_speakers", cfg.num_speakers},
        {"output_subsampling_factor", cfg.output_subsampling_factor},
        {"window_stride_sec", cfg.window_stride},
        {"seconds_per_output_frame", cfg.seconds_per_output_frame()},
        {"encoder_d_model", cfg.encoder.d_model},
        {"trained_channel_capacity", cfg.num_speakers},
    };
}

void write_json_file(const fs::path& path, const json& value) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    {
        std::ofstream out(temporary, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create " + temporary.string());
        out << value.dump(2) << '\n';
        out.flush();
        if (!out) throw std::runtime_error("failed writing " + temporary.string());
    }
    fs::rename(temporary, path);
}

void preflight(
    const Options& options, DiarModel& model, const std::string& backend_name,
    const std::string& device) {
    AudioAsset unused;
    nemo_speech::audio::AudioFile input = nemo_speech::audio::load_wav_file(options.preflight_wav);
    if (input.sample_rate != model.cfg().sample_rate || input.source_channels != 1)
        throw std::runtime_error("preflight WAV must be mono at the GGUF sample rate");
    const auto begin = clock_type::now();
    DiarStream stream(model, model.resolved_geometry(DiarGeometry::v3_streaming()));
    const size_t push = 16000 * 160 / 1000;
    for (size_t off = 0; off < input.samples.size(); off += push)
        stream.feed_audio(input.samples.data() + off, std::min(push, input.samples.size() - off));
    stream.finish();
    const double inference_ms = elapsed_ms(begin);
    const auto& cfg = model.cfg();
    const auto& probs = stream.frame_probs();
    const int64_t frames = stream.n_frames();
    const auto tensor_shape = stream.last_prediction_tensor_shape();
    if (tensor_shape[0] != cfg.num_speakers ||
        probs.size() != static_cast<size_t>(frames) * cfg.num_speakers)
        throw std::runtime_error("preflight output shape disagrees with loaded GGUF dimensions");
    std::string trace_name;
    const int channels = cfg.num_speakers;
    std::vector<float> trace = probs;
    const uint64_t trace_hash = write_trace(
        options.trace_dir.empty() ? fs::path("/tmp") : fs::path(options.trace_dir),
        "task08_geometry_preflight", trace, frames, channels, cfg.seconds_per_output_frame(), trace_name);
    json report = {
        {"schema", "nemo-speech.voiceid-task08-preflight.v1"},
        {"backend", { {"name", backend_name}, {"device", device} }},
        {"gguf_config_loaded_from_metadata", {
            {"sortformer.num_speakers", cfg.num_speakers},
            {"sortformer.output_subsampling_factor", cfg.output_subsampling_factor},
            {"sortformer.preprocessor.window_stride_sec", cfg.window_stride},
        }},
        {"runtime", {
            {"cfg.num_speakers", cfg.num_speakers},
            {"seconds_per_output_frame_sec", cfg.seconds_per_output_frame()},
            {"output_tensor_axis_order", "[channels, native_frames, batch]"},
            {"last_prediction_tensor_shape_channels_frames_batch", tensor_shape},
            {"cpp_probability_vector_shape_frames_channels", {frames, cfg.num_speakers}},
            {"cpp_probability_vector_layout", "frame-major; probabilities[frame * num_speakers + channel]"},
            {"top1_channel_comparison_index", "probabilities[frame * num_speakers + channel]"},
            {"frame_count", frames},
            {"probability_value_count", probs.size()},
            {"expected_value_count", frames * cfg.num_speakers},
        }},
        {"input", {
            {"wav", options.preflight_wav},
            {"duration_sec", static_cast<double>(input.samples.size()) / input.sample_rate},
            {"sample_rate_hz", input.sample_rate},
            {"channels", input.source_channels},
        }},
        {"binary_trace", {
            {"file", trace_name},
            {"layout", "NMSP8F1 v1; little-endian float32; explicit channel/frame axes"},
            {"probability_hash_fnv1a64", hex_u64(trace_hash)},
        }},
        {"export_layouts", {
            {"test_diar_streaming_dump_probs", {
                {"format", "headerless float32 array"},
                {"logical_shape", "[frame, channel]"},
                {"channels", cfg.num_speakers},
                {"values_per_frame", cfg.num_speakers},
            }},
            {"task08_binary_trace", {
                {"format", "NMSP8F1 v1 header plus little-endian float32 payload"},
                {"logical_shape", "[frame, channel]"},
                {"channels", cfg.num_speakers},
            }},
            {"json_research_exports", "per-frame arrays contain num_speakers values in channel order"},
            {"csv_probability_export", "none in the native C++ research harnesses; external CSV conversion must reshape by cfg.num_speakers"},
        }},
        {"inference_ms", inference_ms},
        {"observed_frames_per_input_second", static_cast<double>(frames) / (input.samples.size() / static_cast<double>(input.sample_rate))},
        {"reporting_only_reinterpretation", {
            {"task07_5_reported_channel_count", 4},
            {"task07_5_reported_ami_frames", 12002},
            {"correct_channel_count", cfg.num_speakers},
            {"correct_ami_frames", frames},
            {"runtime_geometry_bug", false},
        }},
    };
    if (!options.preflight_out.empty()) write_json_file(options.preflight_out, report);
    std::cout << report.dump(2) << '\n';
}

std::set<std::string> read_checkpoint(
    const fs::path& path, const std::string& manifest_fingerprint) {
    std::set<std::string> completed;
    if (!fs::exists(path)) return completed;
    const std::vector<uint8_t> bytes = read_bytes(path);
    const std::string contents(bytes.begin(), bytes.end());
    const size_t complete_end = contents.find_last_of('\n');
    if (complete_end == std::string::npos) {
        fs::resize_file(path, 0);
        return completed;
    }
    if (complete_end + 1 < contents.size()) fs::resize_file(path, complete_end + 1);
    std::istringstream input(contents.substr(0, complete_end));
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        json record = json::parse(line);
        if (record.value("manifest_fingerprint_fnv1a64", std::string()) != manifest_fingerprint)
            throw std::runtime_error("checkpoint belongs to a different manifest; use a new output-jsonl path");
        completed.insert(record.at("trial_id").get<std::string>());
    }
    return completed;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto process_start = clock_type::now();
        Options options = parse_args(argc, argv);
        ggml_runtime::Params params;
        params.use_gpu = options.device == "vulkan";
        params.gpu_device_idx = options.gpu_device;
        params.gpu_backend_name = params.use_gpu ? "Vulkan" : "";
        params.n_threads = options.threads;
        params.pe_bin_path = const_cast<char*>("");
        ggml_runtime::BackendManager backend(params);
        BatchingConfig batching;
        batching.enabled = options.preflight_wav.empty() && options.concurrency > 1;
        batching.max_batch_size = options.concurrency;
        batching.max_queue_delay_us = 20000;
        batching.ingress_cohort_delay_us = 20000;
        const auto model_start = clock_type::now();
        DiarModel model(backend, options.model, batching);
        const double model_init_ms = elapsed_ms(model_start);
        if (options.device == "vulkan" && backend.selected_gpu_backend_name() != "Vulkan")
            throw std::runtime_error("requested Vulkan backend was not selected; refusing CPU fallback");
        const std::string selected_device = options.device == "vulkan"
            ? backend.selected_gpu_device_name() : "CPU";
        if (selected_device.empty()) throw std::runtime_error("selected backend did not report a device");
        if (!options.preflight_wav.empty()) {
            preflight(options, model, options.device == "vulkan" ? "Vulkan" : "CPU", selected_device);
            return 0;
        }
        auto shared_model = std::shared_ptr<DiarModel>(&model, [](DiarModel*) {});
        const DiarGeometry geometry = model.resolved_geometry(DiarGeometry::v3_streaming());
        auto diarizer = std::make_shared<Diarizer>(shared_model, geometry);
        const fs::path cache_dir = fs::absolute(options.cache_dir);
        const fs::path trace_dir = fs::absolute(options.trace_dir);
        const fs::path profile_dir = fs::absolute(options.profile_dir);
        fs::create_directories(trace_dir);
        fs::create_directories(profile_dir);
        const std::string manifest_text = [&] {
            std::ifstream in(options.manifest);
            if (!in) throw std::runtime_error("cannot read manifest " + options.manifest);
            std::ostringstream contents;
            contents << in.rdbuf();
            return contents.str();
        }();
        const json manifest = json::parse(manifest_text);
        if (manifest.value("schema", std::string()) != "nemo-speech.voiceid-task08.v1")
            throw std::runtime_error("unsupported VCTK experiment manifest schema");
        const std::string fingerprint = fnv64_hex(manifest_text);
        const fs::path output_path = fs::absolute(options.output_jsonl);
        fs::create_directories(output_path.parent_path());
        std::set<std::string> completed = options.resume
            ? read_checkpoint(output_path, fingerprint) : std::set<std::string>();
        if (!options.resume) {
            if (fs::exists(output_path)) fs::remove(output_path);
            const fs::path run_log_path = output_path.string() + ".runs.jsonl";
            if (fs::exists(run_log_path)) fs::remove(run_log_path);
        }
        std::ofstream checkpoint(output_path, std::ios::app);
        if (!checkpoint) throw std::runtime_error("cannot append checkpoint " + output_path.string());
        json info = model_info(model);
        info["model_init_ms"] = model_init_ms;
        info["selected_gpu_backend"] = options.device == "vulkan"
            ? backend.selected_gpu_backend_name() : "CPU";
        info["selected_device"] = selected_device;
        info["threads"] = options.threads;
        info["concurrency"] = options.concurrency;
        info["batching_enabled"] = batching.enabled;
        info["max_batch_size"] = batching.max_batch_size;
        std::cout << "model loaded: " << info.dump() << '\n';
        int written = 0;
        int selected_trial_count = 0;
        std::set<std::string> matched_requested_trials;
        for (const auto& trial : manifest.at("trials")) {
            const std::string id = trial.at("trial_id").get<std::string>();
            if (!options.trial_ids.empty() && options.trial_ids.count(id) == 0) continue;
            ++selected_trial_count;
            if (!options.trial_ids.empty()) matched_requested_trials.insert(id);
            if (completed.count(id)) continue;
            json result = run_trial(
                trial, fingerprint, diarizer, cache_dir, trace_dir, profile_dir,
                options.concurrency, options.enroll_gap_ms, info, selected_device,
                options.unknown_traffic_only);
            checkpoint << result.dump() << '\n';
            checkpoint.flush();
            if (!checkpoint) throw std::runtime_error("failed to checkpoint completed trial " + id);
            completed.insert(id);
            ++written;
            std::cout << "completed " << id << " (AOSC "
                      << (result.at("aosc_ready").get<bool>() ? "ready" : "not ready")
                      << ", mapping " << (result.at("identity_mapping_valid").get<bool>() ? "valid" : "ambiguous")
                      << ")\n";
        }
        for (const auto& requested : options.trial_ids)
            if (!matched_requested_trials.count(requested))
                throw std::runtime_error("requested trial_id is not present in manifest: " + requested);
        if (written == 0 && options.trial_ids.empty())
            std::cout << "checkpoint already contains every manifest trial\n";
        const BatchMetrics metrics = model.batch_metrics();
        const std::string gpu_memory = gpu_memory_used_mib();
        std::cout << "run complete: new_trials=" << written
                  << " max_batch=" << metrics.max_observed_batch
                  << " batches=" << metrics.batches
                  << " requested_items=" << metrics.requested_items
                  << " peak_rss_bytes=" << peak_rss_bytes()
                  << " gpu_memory_used_mib=" << gpu_memory << '\n';
        const fs::path run_log_path = output_path.string() + ".runs.jsonl";
        std::ofstream run_log(run_log_path, std::ios::app);
        if (!run_log) throw std::runtime_error("cannot append run event " + run_log_path.string());
        run_log << json{
            {"schema", "nemo-speech.voiceid-task08-run-event.v1"},
            {"manifest_fingerprint_fnv1a64", fingerprint},
            {"backend", { {"name", options.device == "vulkan" ? "Vulkan" : "CPU"}, {"device", selected_device} }},
            {"model_init_ms", model_init_ms},
            {"process_elapsed_ms", elapsed_ms(process_start)},
            {"selected_trial_count", selected_trial_count},
            {"new_trials", written},
            {"batching_enabled", batching.enabled},
            {"max_batch_size", batching.max_batch_size},
            {"concurrency", options.concurrency},
            {"max_observed_batch", metrics.max_observed_batch},
            {"batches", metrics.batches},
            {"requested_items", metrics.requested_items},
            {"host_peak_rss_bytes", peak_rss_bytes()},
            {"gpu_memory_used_mib_sample", gpu_memory.empty() ? json(nullptr) : safe_number(std::stod(gpu_memory))},
        }.dump() << '\n';
        run_log.flush();
        if (!run_log) throw std::runtime_error("failed writing run event " + run_log_path.string());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "diar_vctk_voiceid_probe: " << error.what() << '\n';
        return 1;
    }
}
