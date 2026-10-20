// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Research-only open-set VoiceID experiments using native Nemotron V3 AOSC.

#include "audio_file.h"
#include "aosc_profile_intervention.h"
#include "diar_enrollment_profile.h"
#include "diarizer.h"
#include "runtime.h"
#include "voiceid_mapping_research.h"

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
    std::string profile_in;
    std::string preflight_wav;
    std::string preflight_out;
    std::string trial_ids_file;
    std::string device = "vulkan";
    int gpu_device = 0;
    int threads = 4;
    int concurrency = 4;
    int enroll_gap_ms = 250;
    bool resume = true;
    bool unknown_traffic_only = false;
    bool task09a_diagnostics = false;
    std::set<std::string> trial_ids;
    std::set<std::string> query_ids;
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
        else if (arg == "--profile-in") out.profile_in = next("--profile-in");
        else if (arg == "--preflight-wav") out.preflight_wav = next("--preflight-wav");
        else if (arg == "--preflight-out") out.preflight_out = next("--preflight-out");
        else if (arg == "--device") out.device = next("--device");
        else if (arg == "--gpu-device") out.gpu_device = std::stoi(next("--gpu-device"));
        else if (arg == "--threads") out.threads = std::stoi(next("--threads"));
        else if (arg == "--concurrency") out.concurrency = std::stoi(next("--concurrency"));
        else if (arg == "--enroll-gap-ms") out.enroll_gap_ms = std::stoi(next("--enroll-gap-ms"));
        else if (arg == "--trial-id") out.trial_ids.insert(next("--trial-id"));
        else if (arg == "--query-id") out.query_ids.insert(next("--query-id"));
        else if (arg == "--trial-ids-file") out.trial_ids_file = next("--trial-ids-file");
        else if (arg == "--unknown-traffic-only") out.unknown_traffic_only = true;
        else if (arg == "--task09a-diagnostics") out.task09a_diagnostics = true;
        else if (arg == "--no-resume") out.resume = false;
        else if (arg == "--help" || arg == "-h") {
            throw std::runtime_error(
                "usage: diar_vctk_voiceid_probe --manifest FILE --cache-dir DIR --trace-dir DIR "
                "--profile-dir DIR --output-jsonl FILE [--device vulkan --gpu-device 0 "
                "--concurrency 4 --threads 4] [--trial-id ID] [--unknown-traffic-only] "
                "[--query-id ID] [--profile-in FILE] [--task09a-diagnostics] "
                "[--trial-ids-file FILE] [--no-resume]\n"
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
    if (!out.trial_ids_file.empty()) {
        std::ifstream ids(out.trial_ids_file);
        if (!ids) throw std::runtime_error("cannot read trial ID file " + out.trial_ids_file);
        std::string id;
        while (std::getline(ids, id)) {
            while (!id.empty() && std::isspace(static_cast<unsigned char>(id.back()))) id.pop_back();
            const auto first = std::find_if_not(id.begin(), id.end(), [](unsigned char c) { return std::isspace(c); });
            id.erase(id.begin(), first);
            if (!id.empty()) out.trial_ids.insert(id);
        }
    }
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
    std::vector<DiarSegment> segments;
    std::vector<ResearchAoscCompressionEvent> compression_events;
};

InferenceOutput run_with_profile(
    const std::shared_ptr<Diarizer>& diarizer, const std::vector<uint8_t>& profile,
    const nemo_speech::audio::AudioFile& input, bool collect_aosc_compression = false) {
    auto stream = diarizer->streaming_diarize();
    std::vector<ResearchAoscCompressionEvent> compression_events;
    if (collect_aosc_compression) {
        stream->set_research_aosc_compression_observer(
            [&](const ResearchAoscCompressionEvent& event) { compression_events.push_back(event); });
    }
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
    out.segments = stream->segments();
    out.compression_events = std::move(compression_events);
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

json validate_imported_profile(
    const std::vector<uint8_t>& profile, const json& validation_clips,
    const std::map<std::string, int>& mapping, bool mapping_valid,
    const std::shared_ptr<Diarizer>& diarizer, const fs::path& cache_dir,
    double& inference_ms, bool& passed) {
    inference_ms = 0.0;
    passed = false;
    if (!validation_clips.is_array() || validation_clips.empty())
        return {{"status", "not_provided"}, {"passed", false}, {"clip_count", 0}};
    if (!mapping_valid)
        return {
            {"status", "not_evaluated_mapping_invalid"},
            {"passed", false},
            {"clip_count", validation_clips.size()},
        };

    json clip_results = json::array();
    json identity_results = json::object();
    std::map<std::string, bool> identity_pass;
    std::map<std::string, int> identity_clips;
    std::map<std::string, int64_t> identity_speech, identity_active;
    std::map<std::string, std::vector<int64_t>> identity_top1;
    for (const auto& item : validation_clips) {
        AudioAsset asset = load_asset(item, cache_dir);
        const std::string id = item.at("speaker_id").get<std::string>();
        const auto mapped = mapping.find(id);
        if (mapped == mapping.end()) {
            clip_results.push_back({
                {"speaker_id", id}, {"status", "not_evaluated_identity_unmapped"},
                {"utterance_id", item.value("utterance_id", std::string())},
            });
            identity_pass[id] = false;
            continue;
        }
        InferenceOutput inference = run_with_profile(diarizer, profile, asset.audio);
        inference_ms += inference.inference_ms;
        const FrameSummary summary = summarize_frames(inference, asset.mask, 0);
        const int expected = mapped->second;
        const int64_t expected_top1 = expected >= 0 && expected < inference.channels
            ? summary.channel_top1_counts[expected] : 0;
        const double expected_share = summary.activity_frames > 0
            ? static_cast<double>(expected_top1) / summary.activity_frames : 0.0;
        const double expected_mean = expected >= 0 && expected < inference.channels
            ? summary.channel_mean[expected] : 0.0;
        const bool clip_pass = summary.activity_frames > 0 && !summary.top1_tie &&
            summary.dominant_channel == expected && expected_share >= 0.5 && expected_mean >= 0.5;
        identity_pass[id] = identity_pass.count(id) ? identity_pass[id] && clip_pass : clip_pass;
        ++identity_clips[id];
        identity_speech[id] += summary.speech_frames;
        identity_active[id] += summary.activity_frames;
        if (identity_top1[id].empty()) identity_top1[id].assign(inference.channels, 0);
        for (int c = 0; c < inference.channels; ++c)
            identity_top1[id][c] += summary.channel_top1_counts[c];
        clip_results.push_back({
            {"speaker_id", id},
            {"utterance_id", item.value("utterance_id", std::string())},
            {"assigned_channel", expected},
            {"ground_truth_vad_speech_frames", summary.speech_frames},
            {"model_active_frames", summary.activity_frames},
            {"top1_frames_by_channel", summary.channel_top1_counts},
            {"mean_probability_by_channel", summary.channel_mean},
            {"assigned_channel_top1_share", expected_share},
            {"assigned_channel_mean_probability", expected_mean},
            {"assigned_channel_top1_margin_mean", summary.mean_margin},
            {"passed", clip_pass},
        });
    }
    bool every_identity_passed = !mapping.empty();
    for (const auto& entry : mapping) {
        const auto found = identity_pass.find(entry.first);
        const bool has_validation = identity_clips[entry.first] > 0;
        const bool id_pass = has_validation && found != identity_pass.end() && found->second;
        every_identity_passed = every_identity_passed && id_pass;
        json top1 = json::array();
        for (int64_t count : identity_top1[entry.first]) top1.push_back(count);
        identity_results[entry.first] = {
            {"assigned_channel", entry.second},
            {"validation_utterances", identity_clips[entry.first]},
            {"vad_speech_frames", identity_speech[entry.first]},
            {"active_frames", identity_active[entry.first]},
            {"top1_frames_by_channel", std::move(top1)},
            {"passed", id_pass},
        };
    }
    passed = every_identity_passed;
    return {
        {"status", "complete"},
        {"rule", "every held-out enrollment utterance has a unique dominant active channel equal to the assigned identity channel; assigned mean probability >= 0.5"},
        {"passed", passed},
        {"clip_count", validation_clips.size()},
        {"inference_ms", inference_ms},
        {"per_identity", std::move(identity_results)},
        {"clips", std::move(clip_results)},
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
    int64_t start_sample = 0;
    int enrollment_position = 0;
    std::string utterance_id;
    AudioAsset asset;
};

json enrollment_clip_diagnostics(
    const std::vector<EnrollmentClipRef>& clips, const InferenceOutput& output,
    const std::vector<std::string>& enrolled_ids, const std::vector<DiarSegment>& segments) {
    json per_clip = json::array();
    json per_identity = json::object();
    std::map<std::string, std::vector<int64_t>> speaker_positions;
    for (const auto& clip : clips) speaker_positions[clip.speaker].push_back(clip.enrollment_position);
    for (const auto& id : enrolled_ids) {
        std::sort(speaker_positions[id].begin(), speaker_positions[id].end());
        per_identity[id] = {
            {"utterance_count", speaker_positions[id].size()},
            {"early_vs_late_order_split", "first ceil(utterance_count/2) enrollment utterances vs remaining utterances"},
            {"early_vad_speech_frames", 0},
            {"late_vad_speech_frames", 0},
            {"early_active_top1_frames_by_channel", json::array({0, 0, 0, 0, 0, 0, 0, 0})},
            {"late_active_top1_frames_by_channel", json::array({0, 0, 0, 0, 0, 0, 0, 0})},
            {"clip_top1_channel_sequence", json::array()},
            {"clip_to_clip_dominant_channel_transitions", 0},
        };
    }
    std::map<std::string, int> identity_ordinal;
    for (const auto& id : enrolled_ids) identity_ordinal[id] = 0;
    std::map<std::string, int> last_clip_channel;
    for (const auto& clip : clips) {
        const int ordinal = identity_ordinal[clip.speaker]++;
        const int early_count = static_cast<int>((speaker_positions[clip.speaker].size() + 1) / 2);
        const bool early = ordinal < early_count;
        const int channels = output.channels;
        std::vector<int64_t> all_top1(channels, 0), active_top1(channels, 0);
        std::vector<double> probability_sums(channels, 0.0);
        std::vector<double> margins;
        int64_t speech_frames = 0, active_frames = 0, transitions = 0;
        int previous = -1;
        int64_t previous_frame = -2;
        for (size_t i = 0; i < clip.asset.mask.size(); ++i) {
            if (!clip.asset.mask[i]) {
                previous = -1;
                continue;
            }
            ++speech_frames;
            const int64_t frame = clip.start_frame + static_cast<int64_t>(i) - output.base;
            if (frame < 0 || frame >= output.frames - output.base) continue;
            const float* p = output.probabilities.data() + static_cast<size_t>(frame) * channels;
            int best = 0;
            int second = channels > 1 ? 1 : 0;
            if (channels > 1 && p[second] > p[best]) std::swap(best, second);
            for (int c = 2; c < channels; ++c) {
                if (p[c] > p[best]) {
                    second = best;
                    best = c;
                } else if (p[c] > p[second]) {
                    second = c;
                }
            }
            ++all_top1[best];
            for (int c = 0; c < channels; ++c) probability_sums[c] += p[c];
            margins.push_back(static_cast<double>(p[best] - p[second]));
            if (p[best] >= 0.5f) {
                ++active_frames;
                ++active_top1[best];
            }
            if (previous >= 0 && frame == previous_frame + 1 && previous != best) ++transitions;
            previous = best;
            previous_frame = frame;
        }
        std::vector<double> means(channels, 0.0);
        if (speech_frames > 0)
            for (int c = 0; c < channels; ++c) means[c] = probability_sums[c] / speech_frames;
        std::sort(margins.begin(), margins.end());
        const double mean_margin = margins.empty() ? 0.0 :
            std::accumulate(margins.begin(), margins.end(), 0.0) / margins.size();
        const double p10_margin = margins.empty() ? 0.0 : margins[(margins.size() - 1) / 10];
        const double median_margin = margins.empty() ? 0.0 : margins[(margins.size() - 1) / 2];
        const int64_t best_active_count = active_top1.empty()
            ? 0 : *std::max_element(active_top1.begin(), active_top1.end());
        const int best_active_ties = static_cast<int>(std::count(
            active_top1.begin(), active_top1.end(), best_active_count));
        const int dominant = active_frames && best_active_ties == 1
            ? static_cast<int>(std::max_element(active_top1.begin(), active_top1.end()) - active_top1.begin())
            : -1;
        auto& identity = per_identity[clip.speaker];
        identity[early ? "early_vad_speech_frames" : "late_vad_speech_frames"] =
            identity.at(early ? "early_vad_speech_frames" : "late_vad_speech_frames").get<int64_t>() + speech_frames;
        const char* channel_key = early ? "early_active_top1_frames_by_channel" : "late_active_top1_frames_by_channel";
        for (int c = 0; c < channels; ++c)
            identity[channel_key][c] = identity[channel_key][c].get<int64_t>() + active_top1[c];
        identity["clip_top1_channel_sequence"].push_back(
            { {"utterance_id", clip.utterance_id}, {"enrollment_position", clip.enrollment_position},
              {"dominant_active_channel", dominant < 0 ? json(nullptr) : json(dominant)} });
        if (dominant >= 0 && last_clip_channel.count(clip.speaker) &&
            last_clip_channel[clip.speaker] != dominant)
            identity["clip_to_clip_dominant_channel_transitions"] =
                identity.at("clip_to_clip_dominant_channel_transitions").get<int>() + 1;
        if (dominant >= 0) last_clip_channel[clip.speaker] = dominant;
        per_clip.push_back({
            {"speaker_id", clip.speaker},
            {"utterance_id", clip.utterance_id},
            {"enrollment_position", clip.enrollment_position},
            {"temporal_half", early ? "early" : "late"},
            {"utterance_audio_duration_sec",
                static_cast<double>(clip.asset.audio.samples.size()) / clip.asset.audio.sample_rate},
            {"ground_truth_vad_speech_frames", speech_frames},
            {"ground_truth_vad_speech_duration_sec", speech_frames * output.frame_period},
            {"model_active_frames_max_channel_ge_0_5", active_frames},
            {"top1_channel_counts_all_vad_frames", all_top1},
            {"top1_channel_counts_active_vad_frames", active_top1},
            {"mean_probability_by_channel_over_vad_frames", means},
            {"top1_top2_margin_mean", safe_number(mean_margin)},
            {"top1_top2_margin_p10", safe_number(p10_margin)},
            {"top1_top2_margin_median", safe_number(median_margin)},
            {"consecutive_vad_frame_top1_channel_transitions", transitions},
        });
    }

    json segment_count = json::array({0, 0, 0, 0, 0, 0, 0, 0});
    json segment_duration = json::array({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    json segment_vad_overlap = json::object();
    json segment_boundaries = json::array();
    for (const auto& id : enrolled_ids)
        segment_vad_overlap[id] = json::array({0, 0, 0, 0, 0, 0, 0, 0});
    for (const auto& segment : segments) {
        if (segment.speaker < 0 || segment.speaker >= output.channels) continue;
        segment_count[segment.speaker] = segment_count[segment.speaker].get<int>() + 1;
        segment_duration[segment.speaker] = segment_duration[segment.speaker].get<double>() + (segment.t1 - segment.t0);
        segment_boundaries.push_back({
            {"channel", segment.speaker},
            {"start_sec", segment.t0},
            {"end_sec", segment.t1},
        });
        const int64_t begin = static_cast<int64_t>(std::floor(segment.t0 / output.frame_period));
        const int64_t end = static_cast<int64_t>(std::ceil(segment.t1 / output.frame_period));
        for (const auto& clip : clips) {
            auto& counts = segment_vad_overlap[clip.speaker];
            for (size_t i = 0; i < clip.asset.mask.size(); ++i) {
                if (!clip.asset.mask[i]) continue;
                const int64_t frame = clip.start_frame + static_cast<int64_t>(i);
                if (frame >= begin && frame < end)
                    counts[segment.speaker] = counts[segment.speaker].get<int64_t>() + 1;
            }
        }
    }
    return {
        {"ground_truth_source", "Task 08 deterministic enrollment VAD masks; each clip has one enrolled speaker label"},
        {"clips", std::move(per_clip)},
        {"per_identity_temporal_activity", std::move(per_identity)},
        {"native_segmenter", {
            {"segments_by_channel", std::move(segment_count)},
            {"segment_duration_sec_by_channel", std::move(segment_duration)},
            {"vad_speech_frame_intersections_by_identity_and_channel", std::move(segment_vad_overlap)},
            {"segment_boundaries_sec", std::move(segment_boundaries)},
        }},
    };
}

std::string enrollment_source_identity(
    int64_t source_frame, const std::vector<EnrollmentClipRef>& clips) {
    if (source_frame < 0) return "silence_or_unavailable";
    const int64_t sample = source_frame * 1280;  // Eighty-millisecond AOSC encoder frame at 16 kHz.
    for (const auto& clip : clips) {
        const int64_t end_sample = clip.start_sample + static_cast<int64_t>(clip.asset.audio.samples.size());
        if (sample >= clip.start_sample && sample < end_sample) return clip.speaker;
    }
    return "inter_speaker_silence";
}

json compression_matrix_summary(
    const std::vector<int64_t>& source_frames, const std::vector<float>& predictions,
    int channels, const std::vector<EnrollmentClipRef>& clips) {
    std::vector<int64_t> positive(channels, 0), argmax(channels, 0);
    json by_source = json::object();
    for (const auto& clip : clips)
        if (!by_source.contains(clip.speaker))
            by_source[clip.speaker] = json{{"rows", 0}, {"positive_rows_by_prediction_channel", json::array({0,0,0,0,0,0,0,0})}};
    by_source["inter_speaker_silence"] = json{{"rows", 0}, {"positive_rows_by_prediction_channel", json::array({0,0,0,0,0,0,0,0})}};
    by_source["silence_or_unavailable"] = json{{"rows", 0}, {"positive_rows_by_prediction_channel", json::array({0,0,0,0,0,0,0,0})}};
    const size_t rows = source_frames.size();
    if (predictions.size() != rows * static_cast<size_t>(channels))
        throw std::runtime_error("AOSC compression observer produced a mismatched prediction matrix");
    for (size_t f = 0; f < rows; ++f) {
        const std::string source = enrollment_source_identity(source_frames[f], clips);
        by_source[source]["rows"] = by_source[source]["rows"].get<int64_t>() + 1;
        int best = 0;
        for (int c = 0; c < channels; ++c) {
            const float p = predictions[f * channels + c];
            if (p > predictions[f * channels + best]) best = c;
            if (p > 0.5f) {
                ++positive[c];
                by_source[source]["positive_rows_by_prediction_channel"][c] =
                    by_source[source]["positive_rows_by_prediction_channel"][c].get<int64_t>() + 1;
            }
        }
        if (predictions[f * channels + best] > 0.5f) ++argmax[best];
    }
    return {
        {"cache_rows", rows},
        {"positive_rows_by_channel_gt_0_5", positive},
        {"argmax_rows_by_channel_gt_0_5", argmax},
        {"source_identity_rows_and_channel_activity", std::move(by_source)},
    };
}

json compression_event_diagnostics(
    const std::vector<ResearchAoscCompressionEvent>& events,
    const std::vector<EnrollmentClipRef>& clips) {
    json out = json::array();
    size_t index = 0;
    for (const auto& event : events) {
        const json before = compression_matrix_summary(
            event.input_source_frames, event.input_predictions, event.speaker_channels, clips);
        const json after = compression_matrix_summary(
            event.output_source_frames, event.output_predictions, event.speaker_channels, clips);
        out.push_back({
            {"compression_index", index++},
            {"before", before},
            {"after", after},
            {"input_source_frame_ids", event.input_source_frames},
            {"retained_source_frame_ids", event.output_source_frames},
            {"retained_source_row_delta_by_identity", [&] {
                json deltas = json::object();
                for (const auto& clip : clips) {
                    const std::string& id = clip.speaker;
                    if (deltas.contains(id)) continue;
                    const int64_t before_rows = before["source_identity_rows_and_channel_activity"][id]["rows"].get<int64_t>();
                    const int64_t after_rows = after["source_identity_rows_and_channel_activity"][id]["rows"].get<int64_t>();
                    deltas[id] = after_rows - before_rows;
                }
                return deltas;
            }()},
        });
    }
    return out;
}

json query_compression_diagnostics(const std::vector<ResearchAoscCompressionEvent>& events) {
    json out = json::array();
    size_t index = 0;
    auto summary = [](const std::vector<int64_t>& source_frames,
                      const std::vector<float>& predictions, int channels) {
        std::vector<int64_t> positive(channels, 0), argmax(channels, 0);
        if (predictions.size() != source_frames.size() * static_cast<size_t>(channels))
            throw std::runtime_error("query AOSC compression observer matrix shape mismatch");
        for (size_t row = 0; row < source_frames.size(); ++row) {
            const float* values = predictions.data() + row * channels;
            int best = 0;
            for (int channel = 0; channel < channels; ++channel) {
                if (values[channel] > values[best]) best = channel;
                if (values[channel] > 0.5f) ++positive[channel];
            }
            if (values[best] > 0.5f) ++argmax[best];
        }
        return json{
            {"cache_rows", source_frames.size()},
            {"positive_rows_by_channel_gt_0_5", positive},
            {"argmax_rows_by_channel_gt_0_5", argmax},
            {"query_audio_source_frame_rows", std::count_if(source_frames.begin(), source_frames.end(), [](int64_t frame) { return frame >= 0; })},
            {"silence_or_unavailable_rows", std::count_if(source_frames.begin(), source_frames.end(), [](int64_t frame) { return frame < 0; })},
        };
    };
    for (const auto& event : events) {
        json new_rows = json::array();
        for (int64_t frame : event.output_source_frames)
            if (frame >= 0) new_rows.push_back(frame);
        out.push_back({
            {"compression_index", index++},
            {"before", summary(event.input_source_frames, event.input_predictions, event.speaker_channels)},
            {"after", summary(event.output_source_frames, event.output_predictions, event.speaker_channels)},
            {"input_source_frame_ids", event.input_source_frames},
            {"retained_source_frame_ids", event.output_source_frames},
            {"retained_query_audio_source_frame_ids", std::move(new_rows)},
        });
    }
    return out;
}

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
    bool mapping_valid, const std::map<std::string, int>& alternate_mapping,
    bool alternate_mapping_valid, const fs::path& cache_dir, const fs::path& trace_dir,
    const std::string& trace_name, double stream_start_sec = 0.0,
    bool collect_aosc_compression = false) {
    AudioAsset asset = load_asset(item, cache_dir);
    InferenceOutput inference = run_with_profile(diarizer, profile, asset.audio, collect_aosc_compression);
    FrameSummary summary = summarize_frames(
        inference, asset.mask, static_cast<int64_t>(std::llround(stream_start_sec / inference.frame_period)));
    const bool no_decision = summary.activity_frames == 0 || summary.dominant_channel < 0;
    const bool known = item.value("known", false);
    const std::string speaker = item.value("speaker_id", std::string());
    std::string category = mapping_valid
        ? decide_category(known, speaker, summary.dominant_channel, mapping, no_decision)
        : "not_evaluated_ambiguous_enrollment_mapping";
    const std::string alternate_category = alternate_mapping_valid
        ? decide_category(known, speaker, summary.dominant_channel, alternate_mapping, no_decision)
        : "not_evaluated_alternative_mapping_unsupported";
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
    auto predicted_for_channel = [&](const std::map<std::string, int>& candidate) -> json {
        const auto found = std::find_if(candidate.begin(), candidate.end(), [&](const auto& entry) {
            return entry.second == summary.dominant_channel;
        });
        return no_decision ? json(nullptr) : found == candidate.end() ? json("UNKNOWN") : json(found->first);
    };
    json native_segments = json::array();
    json native_segment_counts = json::array({0, 0, 0, 0, 0, 0, 0, 0});
    for (const auto& segment : inference.segments) {
        if (segment.speaker < 0 || segment.speaker >= inference.channels) continue;
        native_segment_counts[segment.speaker] =
            native_segment_counts[segment.speaker].get<int>() + 1;
        native_segments.push_back({
            {"channel", segment.speaker}, {"start_sec", segment.t0}, {"end_sec", segment.t1},
        });
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
        {"research_global_assignment_category", alternate_category},
        {"research_global_assignment_predicted_identity", predicted_for_channel(alternate_mapping)},
        {"top1_dominance", summary.dominance},
        {"mean_top1_top2_margin", summary.mean_margin},
        {"channel_mean_probability_over_vad_frames", frame_summary_json(summary, inference.frame_period)["channel_mean_probability_over_vad_frames"]},
        {"registered_channel_means", registered_means},
        {"diagnostic_max_registered_channel_mean", safe_number(max_registered_mean)},
        {"frame_count", inference.frames - inference.base},
        {"num_speakers", inference.channels},
        {"seconds_per_output_frame", inference.frame_period},
        {"native_segments_by_channel", std::move(native_segment_counts)},
        {"native_segment_boundaries", std::move(native_segments)},
        {"aosc_query_compression_events", query_compression_diagnostics(inference.compression_events)},
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
    std::map<std::string, int> research_global_mapping;
    bool mapping_valid = false;
    bool research_global_mapping_valid = false;
    std::string mapping_error;
    json row_summary = nullptr;
    json retained_rows_by_identity = json::object();
    json detailed_diagnostics = nullptr;
    json research_global_assignment = nullptr;
    int64_t frames = 0;
    double audio_sec = 0.0;
    double speech_sec = 0.0;
    double build_ms = 0.0;
    double research_global_assignment_ms = 0.0;
    std::string not_ready_reason;
    bool ready = false;
};

json global_assignment_diagnostics(
    const std::map<std::string, EnrollmentStats>& stats,
    const std::vector<std::string>& enrolled_ids) {
    std::vector<std::vector<int64_t>> activity;
    std::vector<std::vector<double>> means;
    std::vector<int64_t> active_frames;
    activity.reserve(enrolled_ids.size());
    means.reserve(enrolled_ids.size());
    for (const auto& id : enrolled_ids) {
        const auto& row = stats.at(id);
        activity.push_back(row.top1_counts);
        means.push_back(row.mean_probability);
        active_frames.push_back(row.active_frames);
    }
    const auto assignment = voiceid_research::maximum_weight_assignment(activity);
    const auto support = voiceid_research::assignment_supported(
        assignment, activity, means, active_frames, 0.5);
    json candidate = json::object();
    json evidence = json::object();
    bool all_supported = true;
    for (size_t i = 0; i < enrolled_ids.size(); ++i) {
        const auto& id = enrolled_ids[i];
        const int channel = assignment.channel_for_identity[i];
        candidate[id] = channel;
        const int64_t assigned_count = channel >= 0 ? activity[i][channel] : 0;
        const double share = active_frames[i] > 0
            ? static_cast<double>(assigned_count) / active_frames[i] : 0.0;
        const double mean = channel >= 0 ? means[i][channel] : 0.0;
        evidence[id] = {
            {"assigned_channel", channel},
            {"assigned_top1_active_frames", assigned_count},
            {"active_frames", active_frames[i]},
            {"assigned_top1_share", share},
            {"assigned_mean_probability", mean},
            {"independent_enrollment_evidence_supported", support[i]},
        };
        all_supported = all_supported && support[i];
    }
    const bool unique_optimum = assignment.optimal_assignments == 1;
    return {
        {"method", "maximum_weight_one_to_one_assignment_of_active_top1_frame_counts"},
        {"candidate_identity_to_channel", std::move(candidate)},
        {"total_assignment_weight", assignment.score},
        {"optimal_assignment_count_capped_at_2", assignment.optimal_assignments},
        {"unique_optimum", unique_optimum},
        {"independent_support_rule", "assigned channel has >= 50% of this identity's active top1 frames and mean probability >= 0.5; uses existing strict enrollment criteria"},
        {"evidence_by_identity", std::move(evidence)},
        {"mapping_accepted_for_research_comparison", unique_optimum && all_supported},
    };
}

EnrollmentBuild build_profile(
    const json& trial, const std::shared_ptr<Diarizer>& diarizer,
    const fs::path& cache_dir, const fs::path& profile_dir, int gap_ms,
    bool task09a_diagnostics) {
    EnrollmentBuild result;
    std::vector<ResearchAoscCompressionEvent> compression_events;
    const auto start = clock_type::now();
    const std::vector<std::string> ids = trial.at("enrolled_speaker_ids").get<std::vector<std::string>>();
    auto stream = diarizer->streaming_diarize();
    if (task09a_diagnostics) {
        stream->set_research_aosc_compression_observer(
            [&](const ResearchAoscCompressionEvent& event) { compression_events.push_back(event); });
    }
    const std::vector<float> silence(static_cast<size_t>(gap_ms) * 16, 0.0f);
    int64_t cursor_samples = 0;
    int enrollment_position = 0;
    const auto& enrollment_items = trial.at("enrollment");
    for (size_t clip_index = 0; clip_index < enrollment_items.size(); ++clip_index) {
        const auto& item = enrollment_items[clip_index];
        AudioAsset asset = load_asset(item, cache_dir);
        if (asset.audio.sample_rate != diarizer->sample_rate())
            throw std::runtime_error("enrollment sample rate differs from loaded GGUF");
        const int64_t start_sample = cursor_samples;
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
        result.clips.push_back({
            item.at("speaker_id").get<std::string>(), start_frame, start_sample, enrollment_position++,
            item.value("utterance_id", std::string()), std::move(asset)});
        bool append_gap = true;
        if (trial.value("task09a_inter_speaker_pause_only", false)) {
            append_gap = clip_index + 1 < enrollment_items.size() &&
                enrollment_items[clip_index + 1].value("speaker_id", std::string()) !=
                item.value("speaker_id", std::string());
        }
        if (append_gap && !silence.empty()) {
            stream->push(silence.data(), silence.size(), diarizer->sample_rate());
            cursor_samples += static_cast<int64_t>(silence.size());
            result.audio_sec += static_cast<double>(silence.size()) / diarizer->sample_rate();
        }
    }
    stream->finish();
    result.frames = stream->frame_count();
    const InferenceOutput enrollment_output{
        stream->frame_probabilities(), stream->frame_count(), stream->frame_probs_start(),
        stream->num_speakers(), stream->seconds_per_frame(), stream->last_prediction_tensor_shape(),
        0.0, 0.0, {}, {}};
    result.speaker_stats = analyze_enrollment(
        result.clips, enrollment_output, stream->num_speakers(), ids);
    if (task09a_diagnostics) {
        result.detailed_diagnostics = enrollment_clip_diagnostics(
            result.clips, enrollment_output, ids, stream->segments());
        result.detailed_diagnostics["aosc_compression_events"] =
            compression_event_diagnostics(compression_events, result.clips);
        result.detailed_diagnostics["compression_event_count"] = compression_events.size();
        result.detailed_diagnostics["compression_source_identity_lineage_version"] = 2;
        const auto assignment_start = clock_type::now();
        result.research_global_assignment = global_assignment_diagnostics(result.speaker_stats, ids);
        result.research_global_assignment_ms = elapsed_ms(assignment_start);
        result.research_global_mapping_valid =
            result.research_global_assignment.at("mapping_accepted_for_research_comparison").get<bool>();
        for (auto it = result.research_global_assignment.at("candidate_identity_to_channel").begin();
             it != result.research_global_assignment.at("candidate_identity_to_channel").end(); ++it)
            result.research_global_mapping[it.key()] = it.value().get<int>();
    }
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
            enrollment.research_global_mapping, false,
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
                enrollment.research_global_mapping, false,
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
        stream->num_speakers(), stream->seconds_per_frame(), stream->last_prediction_tensor_shape(),
        0.0, 0.0, {}, {}};
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
    const json& model_info, const std::string& selected_device, bool unknown_traffic_only,
    bool task09a_diagnostics, const std::vector<uint8_t>* profile_override,
    const std::set<std::string>& query_ids) {
    const auto trial_start = clock_type::now();
    const int trial_gap_ms = trial.value("task09a_enrollment_gap_ms", enroll_gap_ms);
    EnrollmentBuild enrollment;
    // Fixed-profile parity runs must measure the supplied immutable AOSC state.
    // Rebuilding the trial's enrollment first would add unrelated inference and
    // temporal state changes before the imported bytes replace that result.
    if (profile_override && trial.contains("task09a_identity_to_channel")) {
        enrollment.profile = *profile_override;
        enrollment.profile_path = profile_dir /
            (sanitized(trial.at("trial_id").get<std::string>()) + ".nmsaosc");
        enrollment.row_summary = profile_row_summary(enrollment.profile);
        enrollment.ready = true;
    } else {
        enrollment = build_profile(
            trial, diarizer, cache_dir, profile_dir, trial_gap_ms, task09a_diagnostics);
    }
    const json built_row_summary = profile_override ? json(nullptr) : enrollment.row_summary;
    const json built_crc = !profile_override && enrollment.ready
        ? json(native_matching::read_u32(enrollment.profile.data() + 109)) : json(nullptr);
    if (profile_override) {
        if (!enrollment.ready)
            throw std::runtime_error("profile override evaluation requires an AOSC-ready profile");
        enrollment.profile = *profile_override;
        enrollment.row_summary = profile_row_summary(enrollment.profile);
        write_private_profile(enrollment.profile_path, enrollment.profile);
        enrollment.profile = read_bytes(enrollment.profile_path);
        if (trial.contains("task09a_identity_to_channel")) {
            const json& map = trial.at("task09a_identity_to_channel");
            enrollment.mapping.clear();
            for (auto it = map.begin(); it != map.end(); ++it)
                enrollment.mapping[it.key()] = it.value().get<int>();
            const auto ids = trial.at("enrolled_speaker_ids").get<std::vector<std::string>>();
            enrollment.mapping_valid = voiceid_research::validate_imported_identity_map(
                ids, enrollment.mapping,
                enrollment.row_summary.at("speaker_channels").get<int>());
            enrollment.mapping_error = enrollment.mapping_valid
                ? std::string() : "provided imported-profile identity map is incomplete or non-unique";
        }
        const auto& positive = enrollment.row_summary.at("positive_rows_by_channel_gt_0_5");
        const auto& argmax = enrollment.row_summary.at("argmax_rows_by_channel_gt_0_5");
        enrollment.retained_rows_by_identity = json::object();
        for (const auto& item : enrollment.mapping)
            enrollment.retained_rows_by_identity[item.first] = {
                {"channel", item.second},
                {"positive_rows_gt_0_5_including_overlaps", positive.at(item.second)},
                {"argmax_rows_gt_0_5", argmax.at(item.second)},
            };
    }
    const json validation_clips = trial.value("validation_enrollment", json::array());
    double strict_validation_ms = 0.0, global_validation_ms = 0.0;
    double strict_validation_wall_ms = 0.0, global_validation_wall_ms = 0.0;
    bool strict_validation_passed = false, global_validation_passed = false;
    json strict_validation = {{"status", "not_provided"}, {"passed", false}};
    json global_validation = {{"status", "not_provided"}, {"passed", false}};
    if (enrollment.ready && validation_clips.is_array() && !validation_clips.empty()) {
        const auto strict_validation_start = clock_type::now();
        strict_validation = validate_imported_profile(
            enrollment.profile, validation_clips, enrollment.mapping, enrollment.mapping_valid,
            diarizer, cache_dir, strict_validation_ms, strict_validation_passed);
        strict_validation_wall_ms = elapsed_ms(strict_validation_start);
        if (enrollment.research_global_mapping_valid) {
            const auto global_validation_start = clock_type::now();
            global_validation = validate_imported_profile(
                enrollment.profile, validation_clips, enrollment.research_global_mapping, true,
                diarizer, cache_dir, global_validation_ms, global_validation_passed);
            global_validation_wall_ms = elapsed_ms(global_validation_start);
        } else {
            global_validation = {
                {"status", "not_evaluated_global_assignment_unsupported"},
                {"passed", false}, {"clip_count", validation_clips.size()},
            };
        }
    }
    json speaker_stats = json::object();
    for (const auto& entry : enrollment.speaker_stats)
        speaker_stats[entry.first] = serialize_enrollment_stats(entry.second);
    json result = {
        {"schema", trial.contains("task09a_candidate_group_id")
                ? "nemo-speech.voiceid-task09a-result.v1" : "nemo-speech.voiceid-task08-result.v1"},
        {"manifest_fingerprint_fnv1a64", manifest_fingerprint},
        {"trial_id", trial.at("trial_id")},
        {"cohort_id", trial.at("cohort_id")},
        {"population", trial.at("population")},
        {"evaluation_partition", trial.at("evaluation_partition")},
        {"seed", trial.at("seed")},
        {"group_size", trial.at("group_size")},
        {"enrollment_budget_speech_sec_per_person", trial.at("enrollment_budget_speech_sec_per_person")},
        {"task09a_candidate_group_id", trial.value("task09a_candidate_group_id", json(nullptr))},
        {"task09a_candidate_variant", trial.value("task09a_candidate_variant", json(nullptr))},
        {"task09a_experiment_family", trial.value("task09a_experiment_family", json(nullptr))},
        {"task09a_matched_comparison_id", trial.value("task09a_matched_comparison_id", json(nullptr))},
        {"task09a_candidate_attempt_limit", trial.value("task09a_candidate_attempt_limit", json(nullptr))},
        {"task09a_inter_speaker_pause_only", trial.value("task09a_inter_speaker_pause_only", false)},
        {"enrollment_inter_speaker_gap_ms", trial_gap_ms},
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
        {"profile_build_crc32_before_override", built_crc},
        {"profile_source", profile_override ? "external_immutable_aosc_import" : "fresh_profile_construction"},
        {"profile_import_bytes_fnv1a64", profile_override
             ? json(fnv64_hex(std::string(enrollment.profile.begin(), enrollment.profile.end())))
             : json(nullptr)},
        {"profile_build_cache_row_summary_before_override", built_row_summary},
        {"cache_row_summary", enrollment.row_summary},
        {"retained_rows_by_enrolled_identity", enrollment.retained_rows_by_identity},
        {"enrollment_speech_duration_sec", enrollment.speech_sec},
        {"enrollment_audio_duration_sec_with_gaps", enrollment.audio_sec},
        {"enrollment_frame_count", enrollment.frames},
        {"speaker_channel_establishment", speaker_stats},
        {"research_global_assignment", enrollment.research_global_assignment},
        {"task09a_enrollment_diagnostics", enrollment.detailed_diagnostics},
        {"identity_to_channel", identity_map_json(enrollment.mapping)},
        {"identity_mapping_valid", enrollment.mapping_valid},
        {"identity_mapping_error", enrollment.mapping_error.empty() ? json(nullptr) : json(enrollment.mapping_error)},
        {"enrollment_validation", {
            {"strict_mapping", strict_validation},
            {"research_global_assignment", global_validation},
            {"strict_verification_passed", strict_validation_passed},
            {"research_global_verification_passed", global_validation_passed},
            {"strict_end_to_end_success", enrollment.ready && enrollment.mapping_valid && strict_validation_passed},
            {"research_global_end_to_end_success", enrollment.ready && enrollment.research_global_mapping_valid && global_validation_passed},
        }},
        {"registration_outcome", !enrollment.ready ? "aosc_not_ready" :
            !enrollment.mapping_valid ? "aosc_ready_identity_map_invalid" :
            validation_clips.empty() ? "aosc_ready_identity_map_valid_verification_not_provided" :
            strict_validation_passed ? "aosc_ready_identity_map_valid_verification_passed" :
            "aosc_ready_identity_map_valid_verification_failed"},
        {"timing", {
            {"enrollment_construction_ms", enrollment.build_ms},
            {"research_global_assignment_calculation_ms", enrollment.research_global_assignment_ms},
            {"strict_additional_enrollment_validation_ms", strict_validation_ms},
            {"research_global_additional_enrollment_validation_ms", global_validation_ms},
            {"strict_additional_enrollment_validation_wall_ms", strict_validation_wall_ms},
            {"research_global_additional_enrollment_validation_wall_ms", global_validation_wall_ms},
        }},
        {"queries", json::array()},
        {"conversations", json::array()},
        {"unknown_traffic", json(nullptr)},
        {"evaluation_mode", unknown_traffic_only ? "unknown_traffic_only" : "full_trial"},
    };

    const bool global_mapping_for_queries = enrollment.ready && enrollment.research_global_mapping_valid &&
        global_validation_passed;
    if (enrollment.ready && (enrollment.mapping_valid || global_mapping_for_queries)) {
        if (!unknown_traffic_only) {
            std::vector<json> selected_queries;
            for (const auto& query : trial.at("queries"))
                if (query_ids.empty() || query_ids.count(query.at("query_id").get<std::string>()))
                    selected_queries.push_back(query);
            for (size_t batch_start = 0; batch_start < selected_queries.size(); batch_start += concurrency) {
                const size_t batch_end = std::min(selected_queries.size(), batch_start + static_cast<size_t>(concurrency));
                std::vector<std::future<json>> calls;
                calls.reserve(batch_end - batch_start);
                for (size_t i = batch_start; i < batch_end; ++i) {
                    const json query = selected_queries[i];
                    calls.emplace_back(std::async(std::launch::async, [&, query] {
                        ScopedBatchCohort cohort(concurrency);
                        return query_output(
                            query, diarizer, enrollment.profile, enrollment.mapping, enrollment.mapping_valid,
                            enrollment.research_global_mapping, global_mapping_for_queries,
                            cache_dir, trace_dir, trial.at("trial_id").get<std::string>() + "_" +
                                query.at("query_id").get<std::string>(), 0.0, task09a_diagnostics);
                    }));
                }
                for (auto& call : calls) result["queries"].push_back(call.get());
            }
            if (enrollment.mapping_valid)
                for (const auto& conversation : trial.value("conversations", json::array()))
                    result["conversations"].push_back(run_conversation(
                        conversation, diarizer, enrollment, cache_dir, trace_dir,
                        trial.at("trial_id").get<std::string>()));
        }
        result["unknown_traffic"] = enrollment.mapping_valid && trial.value("run_unknown_traffic", false)
            ? contamination_sequence(
                  trial, enrollment, diarizer, cache_dir, trace_dir, trial_gap_ms, unknown_traffic_only)
            : json{{"status", enrollment.mapping_valid ? "not_scheduled" : "not_evaluated_strict_mapping_invalid"},
                   {"immutable_enrollment_profile_preserved", true}};
    } else {
        result["evaluation_status"] = enrollment.ready
            ? "skipped_ambiguous_enrollment_mapping" : "skipped_aosc_profile_not_ready";
        for (const auto& query : trial.at("queries")) {
            if (!query_ids.empty() && !query_ids.count(query.at("query_id").get<std::string>())) continue;
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
                {"research_global_assignment_category", "not_evaluated_profile_or_supported_mapping_unavailable"},
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

void hydrate_task09a_asset_references(json& manifest) {
    if (!manifest.contains("task09a_asset_catalog")) return;
    const json& catalog = manifest.at("task09a_asset_catalog");
    auto hydrate_list = [&](json& list) {
        if (!list.is_array()) throw std::runtime_error("Task 09A asset reference list must be an array");
        for (auto& ref : list) {
            std::string key;
            json overrides = json::object();
            if (ref.is_string()) {
                key = ref.get<std::string>();
            } else if (ref.is_object() && ref.contains("asset_id")) {
                key = ref.at("asset_id").get<std::string>();
                overrides = ref;
            } else {
                throw std::runtime_error("Task 09A asset reference has no asset ID");
            }
            if (!catalog.contains(key)) throw std::runtime_error("Task 09A asset catalog is missing " + key);
            json expanded = catalog.at(key);
            for (auto it = overrides.begin(); it != overrides.end(); ++it)
                if (it.key() != "asset_id") expanded[it.key()] = it.value();
            ref = std::move(expanded);
        }
    };
    for (auto& trial : manifest.at("trials")) {
        hydrate_list(trial.at("enrollment"));
        hydrate_list(trial.at("validation_enrollment"));
        hydrate_list(trial.at("queries"));
    }
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
        const std::vector<uint8_t> profile_override = options.profile_in.empty()
            ? std::vector<uint8_t>{} : read_bytes(options.profile_in);
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
        json manifest = json::parse(manifest_text);
        const std::string manifest_schema = manifest.value("schema", std::string());
        if (manifest_schema != "nemo-speech.voiceid-task08.v1" &&
            manifest_schema != "nemo-speech.voiceid-task09a-enrollment.v1")
            throw std::runtime_error("unsupported VCTK experiment manifest schema");
        hydrate_task09a_asset_references(manifest);
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
        std::set<std::string> matched_requested_queries;
        for (const auto& trial : manifest.at("trials")) {
            const std::string id = trial.at("trial_id").get<std::string>();
            if (!options.trial_ids.empty() && options.trial_ids.count(id) == 0) continue;
            ++selected_trial_count;
            if (!options.trial_ids.empty()) matched_requested_trials.insert(id);
            if (completed.count(id)) continue;
            for (const auto& query : trial.at("queries")) {
                const std::string query_id = query.at("query_id").get<std::string>();
                if (options.query_ids.count(query_id)) matched_requested_queries.insert(query_id);
            }
            json result = run_trial(
                trial, fingerprint, diarizer, cache_dir, trace_dir, profile_dir,
                options.concurrency, options.enroll_gap_ms, info, selected_device,
                options.unknown_traffic_only, options.task09a_diagnostics,
                options.profile_in.empty() ? nullptr : &profile_override, options.query_ids);
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
        for (const auto& requested : options.query_ids)
            if (!matched_requested_queries.count(requested))
                throw std::runtime_error("requested query_id is not present in selected trial IDs: " + requested);
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
