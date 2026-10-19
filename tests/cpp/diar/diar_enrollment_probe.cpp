// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "audio_file.h"
#include "diar_enrollment_profile.h"
#include "diarizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <sys/stat.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace nemo_speech::asr;

namespace {

struct AudioSpec {
    std::string identity;
    std::string path;
    double requested_start = 0.0;
    double requested_end = -1.0;
    double actual_start = 0.0;
    double actual_end = 0.0;
    int sample_rate = 0;
    std::vector<float> samples;
};

struct EmbeddingFrame {
    int64_t coarse_frame = 0;
    std::vector<float> values;
};

struct ReferenceInterval {
    std::string recording;
    std::string identity;
    double start_sec = 0.0;
    double end_sec = 0.0;
};

struct MethodResult {
    std::string name;
    std::vector<std::string> tests_json;
    std::string unknown_then_known_json = "null";
    int known_count = 0;
    int known_correct = 0;
    int known_false_rejections = 0;
    int unknown_count = 0;
    int unknown_false_acceptances = 0;
    int speaker_id_swaps = 0;
    int uncertain = 0;
    int reference_known_turns = 0;
    int reference_known_turns_correct = 0;
    int reference_known_false_rejections = 0;
    int reference_unknown_turns = 0;
    int reference_unknown_false_acceptances = 0;
    int reference_diarization_misses = 0;
};

struct Options {
    std::string model;
    std::string report = "-";
    std::string profile_path;
    std::string profile_input;
    std::string snapshot_path;
    std::string rttm_path;
    int silence_ms = 1000;
    std::vector<std::string> enroll_specs;
    std::vector<std::string> identity_channel_specs;
    std::vector<std::string> test_specs;
};

[[noreturn]] void usage(const char* argv0, const std::string& error = {}) {
    if (!error.empty()) std::cerr << "error: " << error << "\n\n";
    std::cerr
        << "Usage: " << argv0 << " MODEL.gguf --enroll ID=FILE[@START:END]..."
        << " --test ID=FILE[@START:END]... [options]\n"
        << "  --silence-ms N       silence inserted between enrollments (default 1000)\n"
        << "  --profile FILE       write the AOSC-only profile (mode 0600 on POSIX)\n"
        << "  --load-profile FILE  evaluate a previously saved AOSC-only profile\n"
        << "  --identity-channel ID=N  map an enrolled label to a saved channel (repeatable)\n"
        << "  --snapshot FILE      write the full streaming snapshot (mode 0600 on POSIX)\n"
        << "  --rttm FILE          align output segments to this RTTM reference\n"
        << "  --report FILE        write JSON report; '-' writes to stdout (default)\n"
        << "  IDs on --test are ground-truth labels. A label absent from enrollment is unknown.\n"
        << "  Times are seconds into the WAV; intervals are start-inclusive/end-exclusive.\n";
    std::exit(error.empty() ? 0 : 2);
}

Options parse_args(int argc, char** argv) {
    if (argc < 2) usage(argv[0], "missing model path");
    Options out;
    out.model = argv[1];
    for (int i = 2; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&](const char* option) -> std::string {
            if (i + 1 >= argc) usage(argv[0], std::string("missing value for ") + option);
            return argv[++i];
        };
        if (arg == "--enroll") out.enroll_specs.push_back(next("--enroll"));
        else if (arg == "--test") out.test_specs.push_back(next("--test"));
        else if (arg == "--silence-ms") out.silence_ms = std::stoi(next("--silence-ms"));
        else if (arg == "--profile") out.profile_path = next("--profile");
        else if (arg == "--load-profile") out.profile_input = next("--load-profile");
        else if (arg == "--identity-channel") out.identity_channel_specs.push_back(next("--identity-channel"));
        else if (arg == "--snapshot") out.snapshot_path = next("--snapshot");
        else if (arg == "--rttm") out.rttm_path = next("--rttm");
        else if (arg == "--report") out.report = next("--report");
        else if (arg == "--help" || arg == "-h") usage(argv[0]);
        else usage(argv[0], "unknown option: " + arg);
    }
    if (out.enroll_specs.empty() && out.profile_input.empty())
        usage(argv[0], "provide --enroll or --load-profile");
    if (!out.enroll_specs.empty() && !out.profile_input.empty())
        usage(argv[0], "--enroll and --load-profile are mutually exclusive");
    if (!out.profile_input.empty() && out.identity_channel_specs.empty())
        usage(argv[0], "--load-profile requires one or more --identity-channel mappings");
    if (out.test_specs.empty()) usage(argv[0], "at least one --test is required");
    if (out.silence_ms < 0 || out.silence_ms > 60000)
        usage(argv[0], "--silence-ms must be between 0 and 60000");
    return out;
}

std::pair<std::string, std::string> split_identity_spec(const std::string& value) {
    const size_t equal = value.find('=');
    if (equal == std::string::npos || equal == 0 || equal + 1 == value.size())
        throw std::invalid_argument("audio specification must be ID=FILE[@START:END]: " + value);
    return {value.substr(0, equal), value.substr(equal + 1)};
}

AudioSpec load_spec(const std::string& spec) {
    AudioSpec out;
    const auto parts = split_identity_spec(spec);
    out.identity = parts.first;
    std::string location = parts.second;
    const size_t at = location.rfind('@');
    if (at != std::string::npos) {
        const std::string range = location.substr(at + 1);
        const size_t colon = range.find(':');
        if (colon == std::string::npos)
            throw std::invalid_argument("time interval must use @START:END: " + spec);
        out.path = location.substr(0, at);
        try {
            out.requested_start = std::stod(range.substr(0, colon));
            out.requested_end = range.substr(colon + 1).empty()
                                    ? -1.0
                                    : std::stod(range.substr(colon + 1));
        } catch (const std::exception&) {
            throw std::invalid_argument("invalid time interval in audio specification: " + spec);
        }
    } else {
        out.path = location;
    }
    const auto file = nemo_speech::audio::load_wav_file(out.path);
    if (out.requested_start < 0.0 || !std::isfinite(out.requested_start) ||
        (out.requested_end >= 0.0 &&
         (!std::isfinite(out.requested_end) || out.requested_end <= out.requested_start)))
        throw std::invalid_argument("invalid audio time interval: " + spec);
    const size_t begin = static_cast<size_t>(std::llround(out.requested_start * file.sample_rate));
    const size_t end = out.requested_end < 0.0
                           ? file.samples.size()
                           : static_cast<size_t>(std::llround(out.requested_end * file.sample_rate));
    if (begin >= end || end > file.samples.size())
        throw std::invalid_argument("audio time interval is outside the WAV: " + spec);
    out.sample_rate = file.sample_rate;
    out.actual_start = static_cast<double>(begin) / file.sample_rate;
    out.actual_end = static_cast<double>(end) / file.sample_rate;
    out.samples.assign(file.samples.begin() + begin, file.samples.begin() + end);
    return out;
}

std::vector<ReferenceInterval> load_rttm(const std::string& path) {
    std::vector<ReferenceInterval> refs;
    if (path.empty()) return refs;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open RTTM reference: " + path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        std::string type, recording, channel, start, duration, field5, field6, identity, field8, field9;
        if (!(row >> type >> recording >> channel >> start >> duration >> field5 >> field6 >> identity >> field8 >> field9))
            throw std::invalid_argument("malformed RTTM row: " + line);
        if (type != "SPEAKER") continue;
        const double t0 = std::stod(start);
        const double length = std::stod(duration);
        if (!std::isfinite(t0) || !std::isfinite(length) || t0 < 0.0 || length <= 0.0)
            throw std::invalid_argument("invalid RTTM time interval: " + line);
        refs.push_back({recording, identity, t0, t0 + length});
    }
    return refs;
}

std::vector<ReferenceInterval> references_for_test(
    const std::vector<ReferenceInterval>& refs, const AudioSpec& test) {
    const std::string stem = std::filesystem::path(test.path).stem().string();
    std::vector<ReferenceInterval> out;
    for (const auto& ref : refs) {
        if (ref.recording != stem || ref.end_sec <= test.actual_start || ref.start_sec >= test.actual_end)
            continue;
        ReferenceInterval clipped = ref;
        clipped.start_sec = std::max(ref.start_sec, test.actual_start) - test.actual_start;
        clipped.end_sec = std::min(ref.end_sec, test.actual_end) - test.actual_start;
        out.push_back(std::move(clipped));
    }
    return out;
}

std::string json_quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
                else out << static_cast<char>(c);
        }
    }
    out << '"';
    return out.str();
}

void json_number(std::ostringstream& out, double value) {
    if (!std::isfinite(value)) out << "null";
    else out << std::setprecision(9) << value;
}

void json_floats(std::ostringstream& out, const std::vector<float>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); i++) {
        if (i) out << ',';
        json_number(out, values[i]);
    }
    out << ']';
}

void json_ints(std::ostringstream& out, const std::vector<int>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); i++) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

std::string fingerprint_hex(const std::array<uint8_t, 32>& fingerprint) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint8_t byte : fingerprint) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

void write_private_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open output file: " + path);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) throw std::runtime_error("failed writing output file: " + path);
#if !defined(_WIN32)
    if (chmod(path.c_str(), S_IRUSR | S_IWUSR) != 0)
        throw std::runtime_error("cannot set mode 0600 on output file: " + path);
#endif
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open saved profile/snapshot: " + path);
    const auto end = in.tellg();
    if (end < 0 || static_cast<uint64_t>(end) > 1024ULL * 1024ULL * 1024ULL)
        throw std::runtime_error("saved profile/snapshot size is invalid: " + path);
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    in.seekg(0);
    if (!bytes.empty()) in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) throw std::runtime_error("failed reading saved profile/snapshot: " + path);
    return bytes;
}

void write_text_file(const std::string& path, const std::string& value) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open report file: " + path);
    out << value;
    out.close();
    if (!out) throw std::runtime_error("failed writing report file: " + path);
}

std::vector<float> copy_eval_probabilities(
    const DiarizationStream& stream, int64_t start_frame, int64_t wanted_frames, int n_spk) {
    const int64_t base = stream.frame_probs_start();
    const int64_t available = stream.frame_count() - base;
    const auto& all = stream.frame_probabilities();
    std::vector<float> result;
    if (start_frame < base || start_frame >= base + available) return result;
    const int64_t count = std::max<int64_t>(0, std::min(wanted_frames, base + available - start_frame));
    const size_t offset = static_cast<size_t>(start_frame - base) * n_spk;
    const size_t values = static_cast<size_t>(count) * n_spk;
    if (offset + values > all.size()) throw std::runtime_error("probability timeline shape mismatch");
    result.assign(all.begin() + offset, all.begin() + offset + values);
    return result;
}

std::vector<double> activity_sums(const std::vector<float>& probs, int n_spk) {
    std::vector<double> sums(n_spk, 0.0);
    for (size_t f = 0; f < probs.size() / n_spk; f++)
        for (int s = 0; s < n_spk; s++) sums[s] += probs[f * n_spk + s];
    return sums;
}

int argmax(const std::vector<double>& values) {
    int best = -1;
    for (size_t i = 0; i < values.size(); i++)
        if (std::isfinite(values[i]) && (best < 0 || values[i] > values[best])) best = static_cast<int>(i);
    return best;
}

std::vector<int> frame_winners(const std::vector<float>& probs, int n_spk) {
    std::vector<int> out;
    for (size_t f = 0; f < probs.size() / n_spk; f++) {
        int best = 0;
        for (int s = 1; s < n_spk; s++)
            if (probs[f * n_spk + s] > probs[f * n_spk + best]) best = s;
        out.push_back(best);
    }
    return out;
}

std::string audio_key(const AudioSpec& item) {
    return item.identity + "=" + item.path + "@" + std::to_string(item.actual_start) + ":" +
           std::to_string(item.actual_end);
}

void emit_matrix(std::ostringstream& out, const std::vector<float>& values, int n_frames, int n_spk) {
    out << '[';
    for (int f = 0; f < n_frames; f++) {
        if (f) out << ',';
        out << '[';
        for (int s = 0; s < n_spk; s++) {
            if (s) out << ',';
            json_number(out, values[static_cast<size_t>(f) * n_spk + s]);
        }
        out << ']';
    }
    out << ']';
}

std::vector<double> cosine_channel_means(const std::vector<float>& values, int n_frames, int n_spk) {
    std::vector<double> sums(n_spk, 0.0);
    std::vector<int> counts(n_spk, 0);
    for (int f = 0; f < n_frames; f++)
        for (int s = 0; s < n_spk; s++) {
            const float x = values[static_cast<size_t>(f) * n_spk + s];
            if (std::isfinite(x)) {
                sums[s] += x;
                counts[s]++;
            }
        }
    for (int s = 0; s < n_spk; s++)
        if (counts[s]) sums[s] /= counts[s];
        else sums[s] = std::numeric_limits<double>::quiet_NaN();
    return sums;
}

std::string make_mapping_json(const std::vector<std::string>& channel_identity) {
    std::ostringstream out;
    out << '[';
    for (size_t s = 0; s < channel_identity.size(); s++) {
        if (s) out << ',';
        out << "{\"channel\":" << s << ",\"identity\":";
        if (channel_identity[s].empty()) out << "null";
        else out << json_quote(channel_identity[s]);
        out << '}';
    }
    out << ']';
    return out.str();
}

std::string encode_enrollment_record(
    const AudioSpec& item, double stream_start_sec, int64_t start_frame, int64_t n_frames,
    const std::vector<float>& probs, int n_spk) {
    const auto sums = activity_sums(probs, n_spk);
    const int winner = argmax(sums);
    std::ostringstream out;
    out << "{\"identity\":" << json_quote(item.identity)
        << ",\"path\":" << json_quote(item.path)
        << ",\"clip_start_sec\":";
    json_number(out, item.actual_start);
    out << ",\"clip_end_sec\":";
    json_number(out, item.actual_end);
    out << ",\"duration_sec\":";
    json_number(out, item.actual_end - item.actual_start);
    out << ",\"stream_start_sec\":";
    json_number(out, stream_start_sec);
    out << ",\"frame_start\":" << start_frame << ",\"frame_count\":" << n_frames
        << ",\"dominant_activity_channel\":" << winner << ",\"channel_activity_sums\":";
    out << '[';
    for (size_t i = 0; i < sums.size(); i++) {
        if (i) out << ',';
        json_number(out, sums[i]);
    }
    out << "]}";
    return out.str();
}

MethodResult run_method(
    const std::string& name, const Options& options, DiarGeometry geometry,
    const std::vector<uint8_t>& full_snapshot, const std::vector<uint8_t>& aosc_profile,
    bool aosc_profile_ready, const std::vector<AudioSpec>& tests,
    const std::vector<ReferenceInterval>& references,
    const std::vector<std::string>& enrollment_ids,
    const std::vector<std::string>& identity_to_channel,
    int input_rate, double enrollment_duration_sec, int num_speakers, double frame_period) {
    MethodResult summary;
    summary.name = name;
    auto diarizer = Diarizer::load(-1, options.model, geometry);
    const int dim = diarizer->embedding_dim();
    std::vector<std::vector<float>> prior_probability_results;
    std::vector<std::string> prior_test_specs;

    for (const auto& test : tests) {
        std::string status = "ok";
        const std::string current_test_key = audio_key(test);
        const auto previous_test = std::find(prior_test_specs.begin(), prior_test_specs.end(), current_test_key);
        const int repeat_index = previous_test == prior_test_specs.end()
                                     ? -1
                                     : static_cast<int>(std::distance(prior_test_specs.begin(), previous_test));
        const bool identical_repeat = repeat_index >= 0;
        std::unique_ptr<DiarizationStream> stream;
        try {
            stream = diarizer->streaming_diarize();
            if (name == "A_full_snapshot") stream->import_state(full_snapshot.data(), full_snapshot.size());
            else if (name == "B_aosc_only") {
                if (!aosc_profile_ready) {
                    status = "not_available_aosc_profile_not_ready";
                    std::ostringstream out;
                    out << "{\"identity\":" << json_quote(test.identity)
                        << ",\"path\":" << json_quote(test.path)
                        << ",\"status\":" << json_quote(status) << '}';
                    summary.tests_json.push_back(out.str());
                    stream.reset();
                    continue;
                } else {
                    DiarAoscProfileCodec::decode(*stream, aosc_profile.data(), aosc_profile.size());
                }
            }
            const double prefix_seconds = name == "A_full_snapshot" ? enrollment_duration_sec : 0.0;
            std::vector<EmbeddingFrame> embeddings;
            if (aosc_profile_ready) {
                stream->set_research_embedding_observer(
                    [&](int64_t first, const float* frames, int count, int embedding_dim) {
                        for (int i = 0; i < count; i++) {
                            EmbeddingFrame frame;
                            frame.coarse_frame = first + i;
                            const float* row = frames + static_cast<size_t>(i) * embedding_dim;
                            frame.values.assign(row, row + embedding_dim);
                            embeddings.push_back(std::move(frame));
                        }
                    });
            }
            const int64_t prefix_frames = static_cast<int64_t>(std::llround(prefix_seconds / frame_period));
            const int64_t wanted_frames = static_cast<int64_t>(std::ceil(
                (test.actual_end - test.actual_start) / frame_period));
            stream->push(test.samples.data(), test.samples.size(), test.sample_rate);
            stream->finish();
            const auto eval_probs = copy_eval_probabilities(*stream, prefix_frames, wanted_frames, num_speakers);
            const int eval_frames = static_cast<int>(eval_probs.size() / num_speakers);
            const auto sums = activity_sums(eval_probs, num_speakers);
            const int channel = argmax(sums);
            const bool multi = test.identity == "MULTI";
            const bool known = !multi && std::find(enrollment_ids.begin(), enrollment_ids.end(), test.identity) !=
                               enrollment_ids.end();
            const std::string mapped_identity =
                channel >= 0 && channel < static_cast<int>(identity_to_channel.size())
                    ? identity_to_channel[channel]
                    : std::string();
            const bool maps_to_enrolled = !mapped_identity.empty();
            const bool correct = known && maps_to_enrolled && mapped_identity == test.identity;
            const bool false_rejection = known && !correct;
            const bool false_acceptance = !known && maps_to_enrolled;
            const bool swap = known && maps_to_enrolled && mapped_identity != test.identity;
            const bool uncertain = channel < 0 || !maps_to_enrolled;
            if (known && !identical_repeat) {
                summary.known_count++;
                summary.known_correct += correct;
                summary.known_false_rejections += false_rejection;
                summary.speaker_id_swaps += swap;
            } else if (!multi && !identical_repeat) {
                summary.unknown_count++;
                summary.unknown_false_acceptances += false_acceptance;
            }
            if (!multi && !identical_repeat) summary.uncertain += uncertain;

            std::vector<int> winners = frame_winners(eval_probs, num_speakers);
            const auto segments = diar_segments_from_probs(
                eval_probs.data(), eval_frames, num_speakers, frame_period, DiarSegmentationCfg{});
            const auto test_references = references_for_test(references, test);
            struct ReferenceObservation {
                ReferenceInterval interval;
                int channel = -1;
                double overlap = 0.0;
                std::string mapped_identity;
                bool known = false;
                bool diarization_found = false;
            };
            std::vector<ReferenceObservation> reference_observations;
            for (const auto& reference : test_references) {
                ReferenceObservation observation;
                observation.interval = reference;
                observation.known = std::find(
                    enrollment_ids.begin(), enrollment_ids.end(), reference.identity) != enrollment_ids.end();
                std::vector<double> overlap_by_channel(num_speakers, 0.0);
                for (const auto& segment : segments) {
                    const double overlap = std::max(
                        0.0, std::min(segment.t1, reference.end_sec) -
                                 std::max(segment.t0, reference.start_sec));
                    if (overlap > 0.0) overlap_by_channel[segment.speaker] += overlap;
                }
                observation.channel = argmax(overlap_by_channel);
                if (observation.channel >= 0) {
                    observation.overlap = overlap_by_channel[observation.channel];
                    observation.diarization_found = observation.overlap > 0.0;
                    if (observation.channel < static_cast<int>(identity_to_channel.size()))
                        observation.mapped_identity = identity_to_channel[observation.channel];
                }
                if (!identical_repeat && !observation.diarization_found) summary.reference_diarization_misses++;
                if (!identical_repeat && observation.known) {
                    summary.reference_known_turns++;
                    const bool correct_turn = observation.diarization_found &&
                                              observation.mapped_identity == reference.identity;
                    summary.reference_known_turns_correct += correct_turn;
                    summary.reference_known_false_rejections += !correct_turn;
                } else if (!identical_repeat) {
                    summary.reference_unknown_turns++;
                    summary.reference_unknown_false_acceptances +=
                        observation.diarization_found && !observation.mapped_identity.empty();
                }
                reference_observations.push_back(std::move(observation));
            }

            std::vector<EmbeddingFrame> speech_embeddings;
            for (auto& frame : embeddings) {
                const double t = (frame.coarse_frame + 0.5) * 0.08;
                if (t < prefix_seconds || t >= prefix_seconds + (test.actual_end - test.actual_start)) continue;
                // Keep embeddings whose matching 80 ms probability window has
                // at least one active channel. This only selects speech-like
                // frames for score analysis; it is not an identity threshold.
                const int64_t native = frame.coarse_frame * 8;
                const int64_t start = native - prefix_frames;
                double max_activity = 0.0;
                int n_activity = 0;
                for (int k = 0; k < 8; k++) {
                    const int64_t f = start + k;
                    if (f < 0 || f >= eval_frames) continue;
                    float best = 0.0f;
                    for (int s = 0; s < num_speakers; s++)
                        best = std::max(best, eval_probs[static_cast<size_t>(f) * num_speakers + s]);
                    max_activity += best;
                    n_activity++;
                }
                if (n_activity && max_activity / n_activity > 0.5) speech_embeddings.push_back(std::move(frame));
            }
            std::vector<float> cosine_values;
            std::vector<double> cosine_means(num_speakers, std::numeric_limits<double>::quiet_NaN());
            if (aosc_profile_ready && !speech_embeddings.empty()) {
                std::vector<float> flat;
                flat.reserve(speech_embeddings.size() * dim);
                for (const auto& f : speech_embeddings) flat.insert(flat.end(), f.values.begin(), f.values.end());
                cosine_values = DiarAoscProfileCodec::cosine_scores(
                    aosc_profile, flat.data(), static_cast<int>(speech_embeddings.size()), dim);
                cosine_means = cosine_channel_means(
                    cosine_values, static_cast<int>(speech_embeddings.size()), num_speakers);
            }
            const int cosine_winner = argmax(cosine_means);
            bool repeat_bit_identical = false;
            if (repeat_index >= 0)
                repeat_bit_identical = prior_probability_results[repeat_index] == eval_probs;
            prior_test_specs.push_back(current_test_key);
            prior_probability_results.push_back(eval_probs);

            std::ostringstream out;
            out << "{\"identity\":" << json_quote(test.identity)
                << ",\"path\":" << json_quote(test.path)
                << ",\"clip_start_sec\":";
            json_number(out, test.actual_start);
            out << ",\"clip_end_sec\":";
            json_number(out, test.actual_end);
            out << ",\"duration_sec\":";
            json_number(out, test.actual_end - test.actual_start);
            out << ",\"multi_speaker_input\":" << (multi ? "true" : "false")
                << ",\"expected_known\":";
            if (multi) out << "null";
            else out << (known ? "true" : "false");
            out << ",\"status\":" << json_quote(status)
                << ",\"evaluation_start_frame\":" << prefix_frames
                << ",\"frame_count\":" << eval_frames
                << ",\"frame_probabilities\":";
            emit_matrix(out, eval_probs, eval_frames, num_speakers);
            out << ",\"frame_argmax_channels\":";
            json_ints(out, winners);
            out << ",\"channel_activity_sums\":[";
            for (size_t i = 0; i < sums.size(); i++) {
                if (i) out << ',';
                json_number(out, sums[i]);
            }
            out << "] ,\"dominant_activity_channel\":" << channel
                << ",\"mapped_identity\":";
            if (maps_to_enrolled) out << json_quote(mapped_identity);
            else out << "null";
            out << ",\"channel_maps_to_enrolled_identity\":" << (maps_to_enrolled ? "true" : "false")
                << ",\"identity_match_under_channel_map\":";
            if (multi) out << "null";
            else if (known) out << (correct ? "true" : "false");
            else out << "null";
            out << ",\"unknown_channel_map_false_acceptance\":";
            if (multi) out << "null";
            else out << (false_acceptance ? "true" : "false");
            out << ",\"known_channel_map_false_rejection\":";
            if (multi) out << "null";
            else out << (false_rejection ? "true" : "false");
            out << ",\"speaker_id_swap\":";
            if (multi) out << "null";
            else out << (swap ? "true" : "false");
            out << ",\"identity_unmapped_or_uncertain\":";
            if (multi) out << "null";
            else out << (uncertain ? "true" : "false");
            out << ",\"segments\":[";
            for (size_t i = 0; i < segments.size(); i++) {
                if (i) out << ',';
                out << "{\"start_sec\":";
                json_number(out, segments[i].t0);
                out << ",\"end_sec\":";
                json_number(out, segments[i].t1);
                out << ",\"channel\":" << segments[i].speaker << '}';
            }
            out << "],\"reference_speaker_segments\":[";
            for (size_t i = 0; i < reference_observations.size(); i++) {
                if (i) out << ',';
                const auto& observation = reference_observations[i];
                out << "{\"identity\":" << json_quote(observation.interval.identity)
                    << ",\"start_sec\":";
                json_number(out, observation.interval.start_sec);
                out << ",\"end_sec\":";
                json_number(out, observation.interval.end_sec);
                out << ",\"known_identity\":" << (observation.known ? "true" : "false")
                    << ",\"diarization_segment_found\":"
                    << (observation.diarization_found ? "true" : "false")
                    << ",\"best_overlapping_channel\":" << observation.channel
                    << ",\"overlap_sec\":";
                json_number(out, observation.overlap);
                out << ",\"mapped_identity\":";
                if (observation.mapped_identity.empty()) out << "null";
                else out << json_quote(observation.mapped_identity);
                out << '}';
            }
            out << "],\"cosine_probe\":{\"speech_embedding_frames\":" << speech_embeddings.size()
                << ",\"score_channel_means\":";
            json_floats(out, std::vector<float>(cosine_means.begin(), cosine_means.end()));
            out << ",\"score_top_channel\":" << cosine_winner
                << ",\"raw_frame_channel_scores\":";
            emit_matrix(out, cosine_values, static_cast<int>(speech_embeddings.size()), num_speakers);
            out << "},\"repeat_of_test_index\":" << repeat_index
                << ",\"probabilities_bit_identical_to_repeat\":";
            if (repeat_index < 0) out << "null";
            else out << (repeat_bit_identical ? "true" : "false");
            out << '}';
            summary.tests_json.push_back(out.str());
        } catch (const std::exception& e) {
            std::ostringstream out;
            out << "{\"identity\":" << json_quote(test.identity)
                << ",\"path\":" << json_quote(test.path)
                << ",\"status\":\"error\",\"error\":" << json_quote(e.what()) << '}';
            summary.tests_json.push_back(out.str());
        }
        stream.reset();  // every case receives a pristine session/profile import.
    }

    const AudioSpec* unknown_test = nullptr;
    std::vector<const AudioSpec*> known_tests;
    std::set<std::string> sequence_test_ids;
    for (const auto& test : tests) {
        if (test.identity == "MULTI") continue;
        if (std::find(enrollment_ids.begin(), enrollment_ids.end(), test.identity) == enrollment_ids.end()) {
            if (!unknown_test) unknown_test = &test;
        } else if (sequence_test_ids.insert(test.identity).second) {
            known_tests.push_back(&test);
        }
    }
    if (unknown_test && !known_tests.empty()) {
        if (name == "B_aosc_only" && !aosc_profile_ready) {
            summary.unknown_then_known_json = "{\"status\":\"not_available_aosc_profile_not_ready\"}";
        } else {
            auto sequence_stream = diarizer->streaming_diarize();
            if (name == "A_full_snapshot") sequence_stream->import_state(full_snapshot.data(), full_snapshot.size());
            else if (name == "B_aosc_only")
                DiarAoscProfileCodec::decode(*sequence_stream, aosc_profile.data(), aosc_profile.size());
            const double prefix_seconds = name == "A_full_snapshot" ? enrollment_duration_sec : 0.0;
            double cursor = prefix_seconds;
            std::vector<const AudioSpec*> sequence{unknown_test};
            sequence.insert(sequence.end(), known_tests.begin(), known_tests.end());
            std::vector<double> starts;
            constexpr int sequence_silence_ms = 2000;
            std::vector<float> silence(
                static_cast<size_t>(std::llround(input_rate * sequence_silence_ms / 1000.0)), 0.0f);
            std::string post_unknown_status = "not_available";
            bool post_unknown_changed = false;
            std::vector<int> post_unknown_channel_rows;
            for (size_t i = 0; i < sequence.size(); i++) {
                starts.push_back(cursor);
                sequence_stream->push(sequence[i]->samples.data(), sequence[i]->samples.size(), sequence[i]->sample_rate);
                cursor += sequence[i]->samples.size() / static_cast<double>(input_rate);
                if (!silence.empty()) {
                    sequence_stream->push(silence.data(), silence.size(), input_rate);
                    cursor += silence.size() / static_cast<double>(input_rate);
                }
                if (i == 0) {
                    try {
                        const auto after_unknown = DiarAoscProfileCodec::encode(*sequence_stream);
                        post_unknown_status = "available";
                        post_unknown_changed = after_unknown != aosc_profile;
                        post_unknown_channel_rows =
                            DiarAoscProfileCodec::summarize(after_unknown).channel_centroid_frames;
                    } catch (const std::exception& e) {
                        post_unknown_status = std::string("unavailable: ") + e.what();
                    }
                }
            }
            sequence_stream->finish();
            std::ostringstream sequence_json;
            sequence_json << "{\"status\":\"ok\",\"order\":[";
            for (size_t i = 0; i < sequence.size(); i++) {
                if (i) sequence_json << ',';
                sequence_json << json_quote(sequence[i]->identity);
            }
            sequence_json << "],\"silence_after_each_clip_ms\":" << sequence_silence_ms
                          << ",\"post_unknown_aosc_profile_status\":"
                          << json_quote(post_unknown_status)
                          << ",\"post_unknown_aosc_cache_changed\":";
            if (post_unknown_status == "available") sequence_json << (post_unknown_changed ? "true" : "false");
            else sequence_json << "null";
            sequence_json << ",\"post_unknown_channel_speech_cache_rows\":";
            if (post_unknown_status == "available") json_ints(sequence_json, post_unknown_channel_rows);
            else sequence_json << "null";
            sequence_json << ",\"results\":[";
            for (size_t i = 0; i < sequence.size(); i++) {
                if (i) sequence_json << ',';
                const auto& item = *sequence[i];
                const int64_t start = static_cast<int64_t>(std::llround(starts[i] / frame_period));
                const int64_t count = static_cast<int64_t>(std::ceil(
                    (item.actual_end - item.actual_start) / frame_period));
                const auto probs = copy_eval_probabilities(*sequence_stream, start, count, num_speakers);
                const int channel = argmax(activity_sums(probs, num_speakers));
                const bool known = std::find(enrollment_ids.begin(), enrollment_ids.end(), item.identity) !=
                                   enrollment_ids.end();
                const std::string mapped = channel >= 0 && channel < static_cast<int>(identity_to_channel.size())
                                               ? identity_to_channel[channel]
                                               : std::string();
                sequence_json << "{\"identity\":" << json_quote(item.identity)
                              << ",\"expected_known\":" << (known ? "true" : "false")
                              << ",\"frame_start\":" << start
                              << ",\"frame_count\":" << probs.size() / num_speakers
                              << ",\"dominant_activity_channel\":" << channel
                              << ",\"mapped_identity\":";
                if (mapped.empty()) sequence_json << "null";
                else sequence_json << json_quote(mapped);
                sequence_json << ",\"identity_match_under_channel_map\":";
                if (known) sequence_json << (mapped == item.identity ? "true" : "false");
                else sequence_json << "null";
                sequence_json << ",\"frame_probabilities\":";
                emit_matrix(sequence_json, probs, static_cast<int>(probs.size() / num_speakers), num_speakers);
                sequence_json << '}';
            }
            sequence_json << "]}";
            summary.unknown_then_known_json = sequence_json.str();
            sequence_stream.reset();
        }
    }
    diarizer.reset();
    return summary;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        std::vector<AudioSpec> enrollments;
        std::vector<AudioSpec> tests;
        const std::vector<ReferenceInterval> references = load_rttm(options.rttm_path);
        for (const auto& spec : options.enroll_specs) enrollments.push_back(load_spec(spec));
        for (const auto& spec : options.test_specs) tests.push_back(load_spec(spec));
        if (!options.profile_input.empty()) {
            if (!enrollments.empty()) throw std::invalid_argument("--load-profile mode does not accept --enroll");
            const int input_rate = tests.front().sample_rate;
            for (const auto& item : tests)
                if (item.sample_rate != input_rate)
                    throw std::invalid_argument("all test WAVs must use the same sample rate");
            const std::vector<uint8_t> aosc_profile = read_binary_file(options.profile_input);
            const AoscProfileSummary profile_summary = DiarAoscProfileCodec::summarize(aosc_profile);
            auto model = Diarizer::load(-1, options.model);
            const int num_speakers = model->num_speakers();
            const int model_rate = model->sample_rate();
            const double frame_period = model->seconds_per_frame();
            const int dim = model->embedding_dim();
            const auto fingerprint = model->model_fingerprint();
            if (profile_summary.speaker_channels != num_speakers || profile_summary.embedding_dim != dim)
                throw std::invalid_argument("loaded AOSC profile dimensions do not match the model");
            auto validation_stream = model->streaming_diarize();
            DiarAoscProfileCodec::decode(*validation_stream, aosc_profile.data(), aosc_profile.size());
            validation_stream.reset();
            model.reset();

            std::vector<std::string> enrollment_ids;
            std::vector<std::string> identity_to_channel(num_speakers);
            std::set<int> mapped_channels;
            for (const auto& spec : options.identity_channel_specs) {
                const size_t equal = spec.find('=');
                if (equal == std::string::npos || equal == 0 || equal + 1 == spec.size())
                    throw std::invalid_argument("identity channel mapping must use ID=N: " + spec);
                const std::string id = spec.substr(0, equal);
                int channel = -1;
                try {
                    size_t parsed = 0;
                    channel = std::stoi(spec.substr(equal + 1), &parsed);
                    if (parsed != spec.size() - equal - 1) throw std::invalid_argument("trailing mapping text");
                } catch (const std::exception&) {
                    throw std::invalid_argument("invalid identity channel mapping: " + spec);
                }
                if (channel < 0 || channel >= num_speakers || !mapped_channels.insert(channel).second ||
                    std::find(enrollment_ids.begin(), enrollment_ids.end(), id) != enrollment_ids.end())
                    throw std::invalid_argument("duplicate or out-of-range identity channel mapping: " + spec);
                enrollment_ids.push_back(id);
                identity_to_channel[channel] = id;
            }
            std::vector<MethodResult> results;
            results.push_back(run_method(
                "B_aosc_only", options, {}, {}, aosc_profile, true, tests, references,
                enrollment_ids, identity_to_channel, input_rate, 0.0, num_speakers, frame_period));
            results.push_back(run_method(
                "C_fresh_session", options, {}, {}, aosc_profile, true, tests, references,
                enrollment_ids, identity_to_channel, input_rate, 0.0, num_speakers, frame_period));

            std::ostringstream json;
            json << "{\n  \"schema\":\"nemo-speech.aosc-enrollment-probe.v1\",\n"
                 << "  \"model\":{\"path\":" << json_quote(options.model)
                 << ",\"sha256\":" << json_quote(fingerprint_hex(fingerprint))
                 << ",\"sample_rate\":" << model_rate << ",\"speaker_channels\":" << num_speakers
                 << ",\"embedding_dim\":" << dim << ",\"frame_period_sec\":";
            json_number(json, frame_period);
            json << "},\n  \"experiment\":{\"mode\":\"load_existing_aosc_profile\",\"loaded_profile_path\":"
                 << json_quote(options.profile_input) << ",\"profile_loaded_in_new_process\":true"
                 << ",\"model_and_validation_session_destroyed_before_trials\":true"
                 << ",\"profile_bytes\":" << aosc_profile.size()
                 << ",\"cache_frames\":" << profile_summary.cache_frames
                 << ",\"channel_speech_cache_rows\":";
            json_ints(json, profile_summary.channel_centroid_frames);
            json << ",\"reference_rttm\":";
            if (options.rttm_path.empty()) json << "null";
            else json << json_quote(options.rttm_path);
            json << "},\n  \"enrollments\":[],\n  \"enrollment_identity_channel_map\":"
                 << make_mapping_json(identity_to_channel)
                 << ",\n  \"enrollment_raw_frame_probabilities\":{\"frame_start\":0,\"frame_count\":0,\"speaker_channels\":"
                 << num_speakers << ",\"values\":[]},\n  \"methods\":[";
            for (size_t i = 0; i < results.size(); i++) {
                if (i) json << ',';
                const auto& result = results[i];
                json << "{\"name\":" << json_quote(result.name)
                     << ",\"known_attempts\":" << result.known_count
                     << ",\"known_correct_channel_mappings\":" << result.known_correct
                     << ",\"known_false_rejections\":" << result.known_false_rejections
                     << ",\"unknown_attempts\":" << result.unknown_count
                     << ",\"unknown_channel_map_false_acceptances\":" << result.unknown_false_acceptances
                     << ",\"speaker_id_swaps\":" << result.speaker_id_swaps
                     << ",\"unmapped_or_uncertain\":" << result.uncertain
                     << ",\"reference_known_turns\":" << result.reference_known_turns
                     << ",\"reference_known_turns_correct\":" << result.reference_known_turns_correct
                     << ",\"reference_known_turn_false_rejections\":" << result.reference_known_false_rejections
                     << ",\"reference_unknown_turns\":" << result.reference_unknown_turns
                     << ",\"reference_unknown_turn_false_acceptances\":"
                     << result.reference_unknown_false_acceptances
                     << ",\"reference_diarization_segment_misses\":" << result.reference_diarization_misses
                     << ",\"unknown_then_known_sequence\":" << result.unknown_then_known_json
                     << ",\"tests\":[";
                for (size_t j = 0; j < result.tests_json.size(); j++) {
                    if (j) json << ',';
                    json << result.tests_json[j];
                }
                json << "]}";
            }
            json << "]\n}\n";
            if (options.report == "-") std::cout << json.str();
            else write_text_file(options.report, json.str());
            return 0;
        }
        std::set<std::string> identities;
        for (const auto& item : enrollments)
            if (!identities.insert(item.identity).second)
                throw std::invalid_argument("enrollment identity labels must be unique: " + item.identity);
        const int input_rate = enrollments.front().sample_rate;
        for (const auto& item : enrollments)
            if (item.sample_rate != input_rate)
                throw std::invalid_argument("all enrollment WAVs must use the same sample rate");
        for (const auto& item : tests)
            if (item.sample_rate != input_rate)
                throw std::invalid_argument("all test WAVs must use the enrollment sample rate");

        auto enrollment_model = Diarizer::load(-1, options.model);
        const int num_speakers = enrollment_model->num_speakers();
        const int model_rate = enrollment_model->sample_rate();
        const double frame_period = enrollment_model->seconds_per_frame();
        const auto fingerprint = enrollment_model->model_fingerprint();
        const int dim = enrollment_model->embedding_dim();
        auto enrollment_stream = enrollment_model->streaming_diarize();
        const size_t silence_samples = static_cast<size_t>(std::llround(input_rate * options.silence_ms / 1000.0));
        const std::vector<float> silence(silence_samples, 0.0f);
        double enrollment_duration_sec = 0.0;
        std::vector<double> enrollment_start_times;
        for (size_t i = 0; i < enrollments.size(); i++) {
            enrollment_start_times.push_back(enrollment_duration_sec);
            enrollment_stream->push(enrollments[i].samples.data(), enrollments[i].samples.size(), input_rate);
            enrollment_duration_sec += enrollments[i].samples.size() / static_cast<double>(input_rate);
            if (silence_samples) {
                enrollment_stream->push(silence.data(), silence.size(), input_rate);
                enrollment_duration_sec += silence.size() / static_cast<double>(input_rate);
            }
        }
        const int64_t enrollment_target = static_cast<int64_t>(std::ceil(enrollment_duration_sec / frame_period));
        enrollment_stream->flush_available(enrollment_target);
        const std::vector<uint8_t> full_snapshot = enrollment_stream->export_state();
        const int64_t enrollment_frames = enrollment_stream->frame_count();
        const int64_t enroll_probability_base = enrollment_stream->frame_probs_start();
        const std::vector<float> enroll_probabilities = enrollment_stream->frame_probabilities();

        std::vector<std::string> enrollment_ids;
        std::vector<std::string> enrollment_records;
        std::map<int, std::set<std::string>> channel_labels;
        for (size_t i = 0; i < enrollments.size(); i++) {
            const int64_t start = static_cast<int64_t>(std::llround(enrollment_start_times[i] / frame_period));
            const int64_t wanted = static_cast<int64_t>(std::ceil(
                (enrollments[i].samples.size() / static_cast<double>(input_rate)) / frame_period));
            const int64_t base = enroll_probability_base;
            const int64_t available_end = base + static_cast<int64_t>(enroll_probabilities.size()) / num_speakers;
            std::vector<float> clip_probs;
            if (start >= base && start < available_end) {
                const int64_t count = std::min(wanted, available_end - start);
                const size_t offset = static_cast<size_t>(start - base) * num_speakers;
                const size_t values = static_cast<size_t>(count) * num_speakers;
                clip_probs.assign(enroll_probabilities.begin() + offset, enroll_probabilities.begin() + offset + values);
            }
            const int channel = argmax(activity_sums(clip_probs, num_speakers));
            enrollment_ids.push_back(enrollments[i].identity);
            if (channel >= 0) channel_labels[channel].insert(enrollments[i].identity);
            enrollment_records.push_back(encode_enrollment_record(
                enrollments[i], enrollment_start_times[i], start,
                static_cast<int64_t>(clip_probs.size()) / num_speakers, clip_probs, num_speakers));
        }
        std::vector<std::string> identity_to_channel(num_speakers);
        for (const auto& entry : channel_labels)
            if (entry.second.size() == 1) identity_to_channel[entry.first] = *entry.second.begin();

        bool aosc_profile_ready = false;
        std::string aosc_profile_error;
        std::vector<uint8_t> aosc_profile;
        AoscProfileSummary profile_summary;
        try {
            aosc_profile = DiarAoscProfileCodec::encode(*enrollment_stream);
            profile_summary = DiarAoscProfileCodec::summarize(aosc_profile);
            aosc_profile_ready = true;
        } catch (const std::exception& e) {
            aosc_profile_error = e.what();
        }
        if (!options.profile_path.empty() && aosc_profile_ready)
            write_private_file(options.profile_path, aosc_profile);
        if (!options.snapshot_path.empty()) write_private_file(options.snapshot_path, full_snapshot);

        // The enrollment model and session are deliberately destroyed before
        // any evaluation model/session is created, exercising actual reload.
        enrollment_stream.reset();
        enrollment_model.reset();
        if (aosc_profile_ready && !options.profile_path.empty()) {
            aosc_profile = read_binary_file(options.profile_path);
            profile_summary = DiarAoscProfileCodec::summarize(aosc_profile);
        }
        std::vector<uint8_t> loaded_full_snapshot = full_snapshot;
        if (!options.snapshot_path.empty()) loaded_full_snapshot = read_binary_file(options.snapshot_path);

        std::vector<MethodResult> results;
        results.push_back(run_method(
            "A_full_snapshot", options, {}, loaded_full_snapshot, aosc_profile, aosc_profile_ready,
            tests, references, enrollment_ids, identity_to_channel,
            input_rate, enrollment_duration_sec, num_speakers, frame_period));
        results.push_back(run_method(
            "B_aosc_only", options, {}, loaded_full_snapshot, aosc_profile, aosc_profile_ready,
            tests, references, enrollment_ids, identity_to_channel,
            input_rate, enrollment_duration_sec, num_speakers, frame_period));
        results.push_back(run_method(
            "C_fresh_session", options, {}, loaded_full_snapshot, aosc_profile, aosc_profile_ready,
            tests, references, enrollment_ids, identity_to_channel,
            input_rate, enrollment_duration_sec, num_speakers, frame_period));

        std::ostringstream json;
        json << "{\n  \"schema\":\"nemo-speech.aosc-enrollment-probe.v1\",\n"
             << "  \"model\":{\"path\":" << json_quote(options.model)
             << ",\"sha256\":" << json_quote(fingerprint_hex(fingerprint))
             << ",\"sample_rate\":" << model_rate << ",\"speaker_channels\":" << num_speakers
             << ",\"embedding_dim\":" << dim << ",\"frame_period_sec\":";
        json_number(json, frame_period);
        json << "},\n  \"experiment\":{\"inserted_silence_ms\":" << options.silence_ms
             << ",\"reference_rttm\":";
        if (options.rttm_path.empty()) json << "null";
        else json << json_quote(options.rttm_path);
        json
             << ",\"enrollment_stream_duration_sec\":";
        json_number(json, enrollment_duration_sec);
        json << ",\"enrollment_frame_count\":" << enrollment_frames
             << ",\"full_snapshot_bytes\":" << full_snapshot.size()
             << ",\"model_and_stream_destroyed_before_evaluation\":true"
             << ",\"snapshot_reloaded_from_disk\":"
             << (!options.snapshot_path.empty() ? "true" : "false")
             << ",\"aosc_profile_reloaded_from_disk\":"
             << ((!options.profile_path.empty() && aosc_profile_ready) ? "true" : "false")
             << ",\"aosc_profile_status\":" << json_quote(aosc_profile_ready ? "ready" : "not_ready")
             << ",\"aosc_profile_error\":";
        if (aosc_profile_error.empty()) json << "null";
        else json << json_quote(aosc_profile_error);
        json << ",\"aosc_profile_bytes\":" << aosc_profile.size();
        if (aosc_profile_ready) {
            json << ",\"aosc_profile_summary\":{\"cache_frames\":" << profile_summary.cache_frames
                 << ",\"embedding_dim\":" << profile_summary.embedding_dim
                 << ",\"channel_speech_cache_rows\":";
            json_ints(json, profile_summary.channel_centroid_frames);
            json << '}';
        }
        json << "},\n  \"enrollments\":[";
        for (size_t i = 0; i < enrollment_records.size(); i++) {
            if (i) json << ',';
            json << enrollment_records[i];
        }
        json << "],\n  \"enrollment_identity_channel_map\":" << make_mapping_json(identity_to_channel)
             << ",\n  \"enrollment_raw_frame_probabilities\":{\"frame_start\":" << enroll_probability_base
             << ",\"frame_count\":" << enroll_probabilities.size() / num_speakers
             << ",\"speaker_channels\":" << num_speakers << ",\"values\":";
        emit_matrix(json, enroll_probabilities, static_cast<int>(enroll_probabilities.size() / num_speakers), num_speakers);
        json << "},\n  \"methods\":[";
        for (size_t i = 0; i < results.size(); i++) {
            if (i) json << ',';
            const auto& result = results[i];
            json << "{\"name\":" << json_quote(result.name)
                 << ",\"known_attempts\":" << result.known_count
                 << ",\"known_correct_channel_mappings\":" << result.known_correct
                 << ",\"known_false_rejections\":" << result.known_false_rejections
                 << ",\"unknown_attempts\":" << result.unknown_count
                 << ",\"unknown_channel_map_false_acceptances\":" << result.unknown_false_acceptances
                 << ",\"speaker_id_swaps\":" << result.speaker_id_swaps
                 << ",\"unmapped_or_uncertain\":" << result.uncertain
                 << ",\"reference_known_turns\":" << result.reference_known_turns
                 << ",\"reference_known_turns_correct\":" << result.reference_known_turns_correct
                 << ",\"reference_known_turn_false_rejections\":" << result.reference_known_false_rejections
                 << ",\"reference_unknown_turns\":" << result.reference_unknown_turns
                 << ",\"reference_unknown_turn_false_acceptances\":"
                 << result.reference_unknown_false_acceptances
                 << ",\"reference_diarization_segment_misses\":" << result.reference_diarization_misses
                 << ",\"unknown_then_known_sequence\":" << result.unknown_then_known_json
                 << ",\"tests\":[";
            for (size_t j = 0; j < result.tests_json.size(); j++) {
                if (j) json << ',';
                json << result.tests_json[j];
            }
            json << "]}";
        }
        json << "]\n}\n";
        if (options.report == "-") std::cout << json.str();
        else write_text_file(options.report, json.str());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "diar_enrollment_probe: " << e.what() << '\n';
        return 1;
    }
}
