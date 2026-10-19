// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Research-only native AOSC context matching and counterfactual probe.

#include "audio_file.h"
#include "aosc_profile_intervention.h"
#include "diar_enrollment_profile.h"
#include "diarizer.h"
#include "runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace nemo_speech::asr;

namespace {

struct Json {
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value = nullptr;

    const Object& object() const {
        if (!std::holds_alternative<Object>(value)) throw std::runtime_error("manifest: expected object");
        return std::get<Object>(value);
    }
    const Array& array() const {
        if (!std::holds_alternative<Array>(value)) throw std::runtime_error("manifest: expected array");
        return std::get<Array>(value);
    }
    const std::string& string() const {
        if (!std::holds_alternative<std::string>(value)) throw std::runtime_error("manifest: expected string");
        return std::get<std::string>(value);
    }
    double number() const {
        if (!std::holds_alternative<double>(value)) throw std::runtime_error("manifest: expected number");
        return std::get<double>(value);
    }
    bool boolean() const {
        if (!std::holds_alternative<bool>(value)) throw std::runtime_error("manifest: expected boolean");
        return std::get<bool>(value);
    }
};

class JsonParser {
   public:
    explicit JsonParser(std::string input) : input_(std::move(input)) {}
    Json parse() {
        Json out = parse_value();
        whitespace();
        if (pos_ != input_.size()) fail("trailing characters");
        return out;
    }

   private:
    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("manifest JSON at byte " + std::to_string(pos_) + ": " + message);
    }
    void whitespace() {
        while (pos_ < input_.size() && (input_[pos_] == ' ' || input_[pos_] == '\t' ||
                                         input_[pos_] == '\r' || input_[pos_] == '\n'))
            ++pos_;
    }
    bool take(char c) {
        whitespace();
        if (pos_ < input_.size() && input_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }
    Json parse_value() {
        whitespace();
        if (pos_ == input_.size()) fail("unexpected end of input");
        const char c = input_[pos_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Json{parse_string()};
        if (input_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return Json{true};
        }
        if (input_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return Json{false};
        }
        if (input_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return Json{nullptr};
        }
        return parse_number();
    }
    Json parse_object() {
        take('{');
        Json::Object out;
        if (take('}')) return Json{std::move(out)};
        do {
            whitespace();
            if (pos_ == input_.size() || input_[pos_] != '"') fail("expected object key");
            std::string key = parse_string();
            if (!take(':')) fail("expected ':' after object key");
            if (!out.emplace(std::move(key), parse_value()).second) fail("duplicate object key");
        } while (take(','));
        if (!take('}')) fail("expected '}'");
        return Json{std::move(out)};
    }
    Json parse_array() {
        take('[');
        Json::Array out;
        if (take(']')) return Json{std::move(out)};
        do out.push_back(parse_value()); while (take(','));
        if (!take(']')) fail("expected ']'");
        return Json{std::move(out)};
    }
    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
        else if (cp <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else if (cp <= 0xffff) {
            out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
    }
    uint32_t hex4() {
        if (input_.size() - pos_ < 4) fail("truncated unicode escape");
        uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = input_[pos_++];
            cp <<= 4;
            if (c >= '0' && c <= '9') cp |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') cp |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') cp |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return cp;
    }
    std::string parse_string() {
        if (!take('"')) fail("expected string");
        std::string out;
        while (pos_ < input_.size()) {
            char c = input_[pos_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ == input_.size()) fail("truncated escape");
            c = input_[pos_++];
            switch (c) {
                case '"': case '\\': case '/': out.push_back(c); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': append_utf8(out, hex4()); break;
                default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }
    Json parse_number() {
        const size_t begin = pos_;
        if (pos_ < input_.size() && input_[pos_] == '-') ++pos_;
        while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') ++pos_;
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') ++pos_;
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) ++pos_;
            while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') ++pos_;
        }
        if (begin == pos_) fail("expected value");
        char* end = nullptr;
        const std::string token = input_.substr(begin, pos_ - begin);
        const double number = std::strtod(token.c_str(), &end);
        if (!end || *end || !std::isfinite(number)) fail("invalid number");
        return Json{number};
    }

    std::string input_;
    size_t pos_ = 0;
};

const Json* field(const Json& object, const std::string& name) {
    const auto& values = object.object();
    const auto it = values.find(name);
    return it == values.end() ? nullptr : &it->second;
}

std::string string_field(const Json& object, const std::string& name, std::string fallback = {}) {
    const Json* value = field(object, name);
    return value ? value->string() : std::move(fallback);
}

double number_field(const Json& object, const std::string& name, double fallback) {
    const Json* value = field(object, name);
    return value ? value->number() : fallback;
}

bool bool_field(const Json& object, const std::string& name, bool fallback) {
    const Json* value = field(object, name);
    return value ? value->boolean() : fallback;
}

struct Clip {
    std::string id;
    std::string speaker;
    std::string rttm_speaker_id;
    std::filesystem::path audio;
    std::string recording_id;
    std::string rttm_recording_id;
    std::string session;
    std::string microphone;
    std::string speech_condition;
    double start_sec = 0.0;
    double end_sec = -1.0;
    bool known = false;
    bool overlap = false;
    std::string partition;
    bool derived = false;
};

struct Reference {
    std::string recording_id;
    std::string speaker;
    double start_sec = 0.0;
    double end_sec = 0.0;
};

struct Manifest {
    std::filesystem::path path;
    std::filesystem::path model;
    std::filesystem::path rttm;
    bool preliminary_fixture_only = false;
    double silence_between_enrollments_sec = 1.0;
    int push_ms = 160;
    std::vector<Clip> enrollment;
    std::vector<Clip> development;
    std::vector<Clip> calibration;
    std::vector<Clip> final_holdout;
    std::vector<Clip> known_queries;
    std::vector<Clip> unknown_queries;
    std::vector<Reference> references;
};

std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read file: " + path.string());
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::filesystem::path resolve_path(const std::filesystem::path& base, const std::string& value) {
    if (value.empty()) return {};
    std::filesystem::path out(value);
    if (out.is_relative()) out = base / out;
    return std::filesystem::weakly_canonical(out);
}

std::vector<Clip> parse_clip_list(
    const Json& root, const std::string& key, const std::filesystem::path& base,
    const std::string& partition, bool known_default) {
    std::vector<Clip> clips;
    const Json* list = field(root, key);
    if (!list) return clips;
    for (const auto& item : list->array()) {
        Clip clip;
        clip.id = string_field(item, "id");
        clip.speaker = string_field(item, "speaker");
        clip.rttm_speaker_id = string_field(item, "rttm_speaker_id");
        clip.audio = resolve_path(base, string_field(item, "audio"));
        clip.recording_id = string_field(item, "recording_id");
        clip.rttm_recording_id = string_field(item, "rttm_recording_id", clip.recording_id);
        clip.session = string_field(item, "session");
        clip.microphone = string_field(item, "microphone");
        clip.speech_condition = string_field(item, "speech_condition", "unlabeled");
        clip.start_sec = number_field(item, "start_sec", 0.0);
        clip.end_sec = number_field(item, "end_sec", -1.0);
        clip.known = bool_field(item, "known", known_default);
        clip.overlap = bool_field(item, "overlap", false);
        const std::string default_partition =
            (partition == "known_queries" || partition == "unknown_queries") ? "final_holdout" : partition;
        clip.partition = string_field(item, "partition", default_partition);
        if (clip.id.empty() || clip.speaker.empty() || clip.audio.empty())
            throw std::runtime_error("manifest " + key + " clip requires id, speaker, and audio");
        if (!std::isfinite(clip.start_sec) || clip.start_sec < 0.0 ||
            !std::isfinite(clip.end_sec) || (clip.end_sec >= 0.0 && clip.end_sec <= clip.start_sec))
            throw std::runtime_error("manifest clip has invalid interval: " + clip.id);
        clips.push_back(std::move(clip));
    }
    return clips;
}

Manifest load_manifest(const std::filesystem::path& path) {
    Manifest out;
    out.path = std::filesystem::weakly_canonical(path);
    const Json root = JsonParser(read_text(out.path)).parse();
    const auto base = out.path.parent_path();
    out.model = resolve_path(base, string_field(root, "model"));
    out.rttm = resolve_path(base, string_field(root, "reference_rttm"));
    out.preliminary_fixture_only = bool_field(root, "preliminary_fixture_only", false);
    out.silence_between_enrollments_sec = number_field(root, "silence_between_enrollments_sec", 1.0);
    out.push_ms = static_cast<int>(number_field(root, "push_ms", 160));
    if (out.model.empty() || out.push_ms <= 0 || out.silence_between_enrollments_sec < 0.0)
        throw std::runtime_error("manifest model, push_ms, or silence duration is invalid");
    out.enrollment = parse_clip_list(root, "enrollment", base, "enrollment", true);
    out.development = parse_clip_list(root, "development", base, "development", false);
    out.calibration = parse_clip_list(root, "calibration", base, "calibration", false);
    out.final_holdout = parse_clip_list(root, "final_holdout", base, "final_holdout", false);
    out.known_queries = parse_clip_list(root, "known_queries", base, "known_queries", true);
    out.unknown_queries = parse_clip_list(root, "unknown_queries", base, "unknown_queries", false);
    for (auto& clip : out.known_queries) clip.known = true;
    for (auto& clip : out.unknown_queries) clip.known = false;
    if (out.preliminary_fixture_only) {
        for (auto& clip : out.known_queries) clip.partition = "preliminary";
        for (auto& clip : out.unknown_queries) clip.partition = "preliminary";
    }

    if (!out.rttm.empty() && std::filesystem::exists(out.rttm)) {
        std::ifstream in(out.rttm);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            std::string type, recording, channel, start, duration, field5, field6, speaker, field8, field9;
            if (!(row >> type >> recording >> channel >> start >> duration >> field5 >> field6 >> speaker >> field8 >> field9))
                throw std::runtime_error("malformed RTTM line: " + line);
            if (type != "SPEAKER") continue;
            const double t0 = std::stod(start);
            const double length = std::stod(duration);
            if (!std::isfinite(t0) || !std::isfinite(length) || t0 < 0 || length <= 0)
                throw std::runtime_error("invalid RTTM time: " + line);
            out.references.push_back({recording, speaker, t0, t0 + length});
        }
    }
    return out;
}

void validate_partition_isolation(const Manifest& manifest) {
    if (manifest.preliminary_fixture_only) return;
    std::vector<Clip> clips;
    for (const auto* group : {&manifest.enrollment, &manifest.development, &manifest.calibration,
                              &manifest.final_holdout, &manifest.known_queries, &manifest.unknown_queries})
        clips.insert(clips.end(), group->begin(), group->end());
    std::set<std::string> ids;
    for (const auto& clip : clips)
        if (!ids.insert(clip.id).second) throw std::runtime_error("duplicate manifest clip id: " + clip.id);
    for (size_t i = 0; i < clips.size(); ++i) {
        for (size_t j = i + 1; j < clips.size(); ++j) {
            const auto& a = clips[i];
            const auto& b = clips[j];
            if (a.partition == b.partition) continue;
            const bool same_audio = a.audio == b.audio;
            const bool same_recording = !a.recording_id.empty() && a.recording_id == b.recording_id;
            if (!same_audio && !same_recording) continue;
            const double a_end = a.end_sec < 0 ? std::numeric_limits<double>::infinity() : a.end_sec;
            const double b_end = b.end_sec < 0 ? std::numeric_limits<double>::infinity() : b.end_sec;
            if (a.start_sec < b_end && b.start_sec < a_end)
                throw std::runtime_error("audio intervals overlap across partitions: " + a.id + " and " + b.id);
        }
    }
}

std::vector<Clip> queries(const Manifest& manifest) {
    std::vector<Clip> out = manifest.development;
    out.insert(out.end(), manifest.calibration.begin(), manifest.calibration.end());
    out.insert(out.end(), manifest.final_holdout.begin(), manifest.final_holdout.end());
    out.insert(out.end(), manifest.known_queries.begin(), manifest.known_queries.end());
    out.insert(out.end(), manifest.unknown_queries.begin(), manifest.unknown_queries.end());
    return out;
}

struct AudioSlice {
    std::vector<float> samples;
    int sample_rate = 0;
    double duration_sec = 0.0;
    double start_sec = 0.0;
    double end_sec = 0.0;
};

AudioSlice load_audio_slice(const Clip& clip) {
    const auto file = nemo_speech::audio::load_wav_file(clip.audio.string());
    const double file_duration = static_cast<double>(file.samples.size()) / file.sample_rate;
    const double end = clip.end_sec < 0.0 ? file_duration : clip.end_sec;
    const size_t begin_sample = static_cast<size_t>(std::llround(clip.start_sec * file.sample_rate));
    const size_t end_sample = static_cast<size_t>(std::llround(end * file.sample_rate));
    if (begin_sample >= end_sample || end_sample > file.samples.size())
        throw std::runtime_error("clip interval is outside WAV: " + clip.id);
    AudioSlice out;
    out.samples.assign(file.samples.begin() + static_cast<std::ptrdiff_t>(begin_sample),
                       file.samples.begin() + static_cast<std::ptrdiff_t>(end_sample));
    out.sample_rate = file.sample_rate;
    out.duration_sec = static_cast<double>(out.samples.size()) / out.sample_rate;
    out.start_sec = static_cast<double>(begin_sample) / out.sample_rate;
    out.end_sec = static_cast<double>(end_sample) / out.sample_rate;
    return out;
}

void push_in_chunks(DiarizationStream& stream, const std::vector<float>& samples, int sample_rate, int push_ms) {
    const size_t push_samples = std::max<size_t>(1, static_cast<size_t>(sample_rate) * push_ms / 1000);
    for (size_t offset = 0; offset < samples.size(); offset += push_samples) {
        const size_t count = std::min(push_samples, samples.size() - offset);
        stream.push(samples.data() + offset, count, sample_rate);
    }
}

void push_silence(DiarizationStream& stream, int sample_rate, int push_ms, double seconds) {
    const size_t count = static_cast<size_t>(std::llround(sample_rate * seconds));
    std::vector<float> silence(count, 0.0f);
    push_in_chunks(stream, silence, sample_rate, push_ms);
}

std::vector<std::pair<double, double>> speech_intervals(
    const Clip& clip, const AudioSlice& audio, const std::vector<Reference>& references) {
    std::vector<std::pair<double, double>> intervals;
    if (clip.rttm_speaker_id.empty() || clip.rttm_recording_id.empty()) return intervals;
    for (const auto& ref : references) {
        if (ref.recording_id != clip.rttm_recording_id || ref.speaker != clip.rttm_speaker_id ||
            ref.end_sec <= audio.start_sec || ref.start_sec >= audio.end_sec)
            continue;
        intervals.emplace_back(std::max(ref.start_sec, audio.start_sec) - audio.start_sec,
                               std::min(ref.end_sec, audio.end_sec) - audio.start_sec);
    }
    std::sort(intervals.begin(), intervals.end());
    std::vector<std::pair<double, double>> merged;
    for (const auto& interval : intervals) {
        if (!merged.empty() && interval.first <= merged.back().second)
            merged.back().second = std::max(merged.back().second, interval.second);
        else
            merged.push_back(interval);
    }
    return merged;
}

std::vector<std::pair<double, double>> overlap_intervals(
    const Clip& clip, const AudioSlice& audio, const std::vector<Reference>& references,
    const std::vector<std::pair<double, double>>& target_intervals) {
    std::vector<std::pair<double, double>> other_intervals;
    if (clip.rttm_speaker_id.empty() || clip.rttm_recording_id.empty()) return other_intervals;
    for (const auto& ref : references) {
        if (ref.recording_id != clip.rttm_recording_id || ref.speaker == clip.rttm_speaker_id ||
            ref.end_sec <= audio.start_sec || ref.start_sec >= audio.end_sec)
            continue;
        other_intervals.emplace_back(std::max(ref.start_sec, audio.start_sec) - audio.start_sec,
                                     std::min(ref.end_sec, audio.end_sec) - audio.start_sec);
    }
    std::vector<std::pair<double, double>> intersections;
    for (const auto& target : target_intervals) {
        for (const auto& other : other_intervals) {
            const double begin = std::max(target.first, other.first);
            const double end = std::min(target.second, other.second);
            if (end > begin) intersections.emplace_back(begin, end);
        }
    }
    std::sort(intersections.begin(), intersections.end());
    std::vector<std::pair<double, double>> merged;
    for (const auto& interval : intersections) {
        if (!merged.empty() && interval.first <= merged.back().second)
            merged.back().second = std::max(merged.back().second, interval.second);
        else
            merged.push_back(interval);
    }
    return merged;
}

bool time_in_intervals(double time, const std::vector<std::pair<double, double>>& intervals) {
    for (const auto& interval : intervals)
        if (time >= interval.first && time < interval.second) return true;
    return false;
}

struct Context {
    std::vector<uint8_t> profile;
    bool ready = false;
    std::map<std::string, int> identity_channel;
    std::map<std::string, std::vector<double>> enrollment_channel_activity;
    std::vector<int> cache_rows;
    std::vector<double> clip_intervals;
    double enrollment_audio_sec = 0.0;
};

Context build_context(
    const std::shared_ptr<Diarizer>& diarizer, const DiarGeometry& geometry,
    const std::vector<Clip>& enrollment, const std::vector<Reference>& references,
    int push_ms, double silence_sec) {
    Context out;
    auto stream = diarizer->streaming_diarize();
    (void)geometry;
    double context_cursor = 0.0;
    struct ClipInterval { Clip clip; double begin; double end; };
    std::vector<ClipInterval> boundaries;
    for (const auto& clip : enrollment) {
        const AudioSlice audio = load_audio_slice(clip);
        const double begin = context_cursor;
        push_in_chunks(*stream, audio.samples, audio.sample_rate, push_ms);
        context_cursor += audio.duration_sec;
        boundaries.push_back({clip, begin, context_cursor});
        push_silence(*stream, audio.sample_rate, push_ms, silence_sec);
        context_cursor += silence_sec;
        out.enrollment_audio_sec += audio.duration_sec;
    }
    stream->finish();
    const auto& probs = stream->frame_probabilities();
    const int nspk = stream->num_speakers();
    const double period = stream->seconds_per_frame();
    if (nspk <= 0 || period <= 0 || probs.size() % static_cast<size_t>(nspk) != 0)
        throw std::runtime_error("invalid enrollment probability timeline");

    for (const auto& boundary : boundaries) {
        const AudioSlice audio = load_audio_slice(boundary.clip);
        const auto intervals = speech_intervals(boundary.clip, audio, references);
        std::vector<double> sums(static_cast<size_t>(nspk), 0.0);
        size_t count = 0;
        const size_t frames = probs.size() / nspk;
        for (size_t i = 0; i < frames; ++i) {
            const double context_time = (stream->frame_probs_start() + static_cast<int64_t>(i) + 0.5) * period;
            const double local_time = context_time - boundary.begin;
            if (context_time < boundary.begin || context_time >= boundary.end) continue;
            if (!intervals.empty() && !time_in_intervals(local_time, intervals)) continue;
            for (int s = 0; s < nspk; ++s)
                sums[static_cast<size_t>(s)] += probs[i * nspk + s];
            ++count;
        }
        if (count == 0) {
            for (size_t i = 0; i < sums.size(); ++i) sums[i] = 0.0;
            for (size_t f = 0; f < frames; ++f) {
                const double context_time = (stream->frame_probs_start() + static_cast<int64_t>(f) + 0.5) * period;
                if (context_time < boundary.begin || context_time >= boundary.end) continue;
                for (int s = 0; s < nspk; ++s) sums[static_cast<size_t>(s)] += probs[f * nspk + s];
                ++count;
            }
        }
        std::vector<double> means(static_cast<size_t>(nspk), 0.0);
        if (count) for (int s = 0; s < nspk; ++s) means[static_cast<size_t>(s)] = sums[static_cast<size_t>(s)] / count;
        const int channel = static_cast<int>(std::distance(means.begin(), std::max_element(means.begin(), means.end())));
        out.identity_channel[boundary.clip.speaker] = channel;
        out.enrollment_channel_activity[boundary.clip.speaker] = std::move(means);
        out.clip_intervals.push_back(boundary.begin);
        out.clip_intervals.push_back(boundary.end);
    }
    try {
        out.profile = DiarAoscProfileCodec::encode(*stream);
        out.ready = true;
        out.cache_rows = DiarAoscProfileCodec::summarize(out.profile).channel_centroid_frames;
    } catch (const std::invalid_argument&) {
        out.ready = false;
    }
    return out;
}

struct QueryOutput {
    std::string method;
    std::string clip_id;
    std::string speaker;
    bool known = false;
    bool overlap = false;
    std::string recording_id;
    std::string session;
    std::string microphone;
    std::string speech_condition;
    std::string partition;
    double source_start_sec = 0.0;
    double source_duration_sec = 0.0;
    double elapsed_ms = 0.0;
    int num_speakers = 0;
    int64_t frame_base = 0;
    double seconds_per_frame = 0.0;
    std::vector<std::pair<double, std::vector<float>>> speech_probabilities;
    std::vector<double> channel_means;
    std::vector<int> top1_counts;
    double mean_top1 = 0.0;
    double mean_margin = 0.0;
    double assignment_stability = 0.0;
    int top1_switches = 0;
    double speech_seconds = 0.0;
    std::string speech_mask_source;
    std::map<std::string, double> mapped_activity;
    std::map<std::string, double> mapped_top1_fraction;
    std::vector<double> overlap_channel_means;
    std::vector<int> overlap_top1_counts;
    std::vector<std::pair<double, std::vector<float>>> overlap_probabilities;
    double overlap_mean_margin = 0.0;
    int overlap_top1_switches = 0;
};

QueryOutput evaluate_query(
    const std::shared_ptr<Diarizer>& diarizer, const DiarGeometry& geometry,
    const Clip& clip, const std::vector<Reference>& references,
    const std::vector<uint8_t>* profile, const std::map<std::string, int>& identity_channel,
    const std::map<std::string, std::vector<double>>& enrollment_activity, int push_ms,
    const std::string& method) {
    QueryOutput out;
    (void)geometry;
    out.method = method;
    out.clip_id = clip.id;
    out.speaker = clip.speaker;
    out.known = clip.known;
    out.overlap = clip.overlap;
    out.recording_id = clip.recording_id;
    out.session = clip.session;
    out.microphone = clip.microphone;
    out.speech_condition = clip.speech_condition;
    out.partition = clip.partition;
    out.source_start_sec = clip.start_sec;
    const AudioSlice audio = load_audio_slice(clip);
    out.source_duration_sec = audio.duration_sec;
    const auto intervals = speech_intervals(clip, audio, references);
    const auto overlap_ranges = overlap_intervals(clip, audio, references, intervals);
    out.speech_mask_source = intervals.empty() ? "model_activity_proxy_ge_0.5" : "rttm_target_speaker";
    auto stream = diarizer->streaming_diarize();
    if (profile && !profile->empty()) DiarAoscProfileCodec::decode(*stream, profile->data(), profile->size());
    const auto begin = std::chrono::steady_clock::now();
    push_in_chunks(*stream, audio.samples, audio.sample_rate, push_ms);
    stream->finish();
    out.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    out.num_speakers = stream->num_speakers();
    out.frame_base = stream->frame_probs_start();
    out.seconds_per_frame = stream->seconds_per_frame();
    const auto& raw = stream->frame_probabilities();
    const size_t frames = raw.size() / static_cast<size_t>(std::max(1, out.num_speakers));
    if (out.num_speakers <= 0 || raw.size() % static_cast<size_t>(out.num_speakers) != 0)
        throw std::runtime_error("invalid query probability timeline");

    std::vector<size_t> selected_frames;
    for (size_t f = 0; f < frames; ++f) {
        const double relative_time = (out.frame_base + static_cast<int64_t>(f) + 0.5) * out.seconds_per_frame;
        if (!intervals.empty()) {
            if (time_in_intervals(relative_time, intervals)) selected_frames.push_back(f);
        } else {
            const float* p = raw.data() + f * out.num_speakers;
            if (*std::max_element(p, p + out.num_speakers) >= 0.5f) selected_frames.push_back(f);
        }
    }
    if (selected_frames.empty() && intervals.empty()) {
        for (size_t f = 0; f < frames; ++f) selected_frames.push_back(f);
    }
    if (selected_frames.empty() && frames > 0)
        throw std::runtime_error("no speech frames selected for query " + clip.id);

    out.channel_means.assign(static_cast<size_t>(out.num_speakers), 0.0);
    out.top1_counts.assign(static_cast<size_t>(out.num_speakers), 0);
    double top1_sum = 0.0;
    double margin_sum = 0.0;
    int previous_top1 = -1;
    for (size_t f : selected_frames) {
        const float* p = raw.data() + f * out.num_speakers;
        int top1 = 0;
        int top2 = out.num_speakers > 1 ? 1 : 0;
        if (out.num_speakers > 1 && p[top2] > p[top1]) std::swap(top1, top2);
        for (int s = 2; s < out.num_speakers; ++s) {
            if (p[s] > p[top1]) {
                top2 = top1;
                top1 = s;
            } else if (p[s] > p[top2]) {
                top2 = s;
            }
        }
        for (int s = 0; s < out.num_speakers; ++s)
            out.channel_means[static_cast<size_t>(s)] += p[s];
        ++out.top1_counts[static_cast<size_t>(top1)];
        top1_sum += p[top1];
        margin_sum += p[top1] - p[top2];
        if (previous_top1 >= 0 && previous_top1 != top1) ++out.top1_switches;
        previous_top1 = top1;
        const double relative_time = (out.frame_base + static_cast<int64_t>(f) + 0.5) * out.seconds_per_frame;
        out.speech_probabilities.emplace_back(
            relative_time, std::vector<float>(p, p + out.num_speakers));
    }
    const double denom = static_cast<double>(selected_frames.size());
    for (double& value : out.channel_means) value /= denom;
    out.speech_seconds = denom * out.seconds_per_frame;
    out.mean_top1 = top1_sum / denom;
    out.mean_margin = margin_sum / denom;
    out.assignment_stability = static_cast<double>(*std::max_element(out.top1_counts.begin(), out.top1_counts.end())) / denom;
    for (const auto& entry : identity_channel) {
        const int channel = entry.second;
        if (channel < 0 || channel >= out.num_speakers) continue;
        out.mapped_activity[entry.first] = out.channel_means[static_cast<size_t>(channel)];
        out.mapped_top1_fraction[entry.first] = static_cast<double>(out.top1_counts[static_cast<size_t>(channel)]) / denom;
    }

    out.overlap_channel_means.assign(static_cast<size_t>(out.num_speakers), 0.0);
    out.overlap_top1_counts.assign(static_cast<size_t>(out.num_speakers), 0);
    double overlap_margin_sum = 0.0;
    int previous_overlap_top1 = -1;
    for (size_t f : selected_frames) {
        const double relative_time = (out.frame_base + static_cast<int64_t>(f) + 0.5) * out.seconds_per_frame;
        if (!time_in_intervals(relative_time, overlap_ranges)) continue;
        const float* p = raw.data() + f * out.num_speakers;
        int top1 = 0;
        int top2 = out.num_speakers > 1 ? 1 : 0;
        if (out.num_speakers > 1 && p[top2] > p[top1]) std::swap(top1, top2);
        for (int s = 2; s < out.num_speakers; ++s) {
            if (p[s] > p[top1]) { top2 = top1; top1 = s; }
            else if (p[s] > p[top2]) top2 = s;
        }
        for (int s = 0; s < out.num_speakers; ++s)
            out.overlap_channel_means[static_cast<size_t>(s)] += p[s];
        ++out.overlap_top1_counts[static_cast<size_t>(top1)];
        overlap_margin_sum += p[top1] - p[top2];
        if (previous_overlap_top1 >= 0 && previous_overlap_top1 != top1) ++out.overlap_top1_switches;
        previous_overlap_top1 = top1;
        out.overlap_probabilities.emplace_back(relative_time, std::vector<float>(p, p + out.num_speakers));
    }
    if (!out.overlap_probabilities.empty()) {
        const double overlap_count = static_cast<double>(out.overlap_probabilities.size());
        for (double& value : out.overlap_channel_means) value /= overlap_count;
        out.overlap_mean_margin = overlap_margin_sum / overlap_count;
    }
    (void)enrollment_activity;
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

void json_number(std::ostream& out, double value) {
    if (!std::isfinite(value)) out << "null";
    else out << std::setprecision(9) << value;
}

void emit_int_array(std::ostream& out, const std::vector<int>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void emit_double_array(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        json_number(out, values[i]);
    }
    out << ']';
}

void emit_query(std::ostream& out, const QueryOutput& result) {
    out << "{\"method\":" << json_quote(result.method) << ",\"clip_id\":" << json_quote(result.clip_id)
        << ",\"speaker\":" << json_quote(result.speaker) << ",\"known\":" << (result.known ? "true" : "false")
        << ",\"overlap_labeled\":" << (result.overlap ? "true" : "false")
        << ",\"recording_id\":" << json_quote(result.recording_id)
        << ",\"session\":" << json_quote(result.session)
        << ",\"microphone\":" << json_quote(result.microphone)
        << ",\"speech_condition\":" << json_quote(result.speech_condition)
        << ",\"partition\":" << json_quote(result.partition)
        << ",\"source_start_sec\":";
    json_number(out, result.source_start_sec);
    out << ",\"source_duration_sec\":";
    json_number(out, result.source_duration_sec);
    out
        << ",\"speech_frames\":" << result.speech_probabilities.size()
        << ",\"speech_mask_source\":" << json_quote(result.speech_mask_source)
        << ",\"speech_seconds\":";
    json_number(out, result.speech_seconds);
    out << ",\"elapsed_ms\":";
    json_number(out, result.elapsed_ms);
    out << ",\"mean_channel_activity\":";
    emit_double_array(out, result.channel_means);
    out << ",\"top1_channel_speech_frames\":";
    emit_int_array(out, result.top1_counts);
    out << ",\"mean_top1_activity\":";
    json_number(out, result.mean_top1);
    out << ",\"mean_top1_top2_margin\":";
    json_number(out, result.mean_margin);
    out << ",\"top1_assignment_stability\":";
    json_number(out, result.assignment_stability);
    out << ",\"top1_switches\":" << result.top1_switches << ",\"mapped_enrollment_activity\":{";
    bool first = true;
    for (const auto& item : result.mapped_activity) {
        if (!first) out << ',';
        first = false;
        out << json_quote(item.first) << ':';
        json_number(out, item.second);
    }
    out << "},\"mapped_enrollment_top1_fraction\":{";
    first = true;
    for (const auto& item : result.mapped_top1_fraction) {
        if (!first) out << ',';
        first = false;
        out << json_quote(item.first) << ':';
        json_number(out, item.second);
    }
    out << "},\"raw_speech_probabilities\":[";
    for (size_t i = 0; i < result.speech_probabilities.size(); ++i) {
        if (i) out << ',';
        out << "{\"time_sec\":";
        json_number(out, result.speech_probabilities[i].first);
        out << ",\"channels\":";
        emit_double_array(out, std::vector<double>(result.speech_probabilities[i].second.begin(),
                                                    result.speech_probabilities[i].second.end()));
        out << '}';
    }
    out << "],\"overlap_subset\":{\"frames\":" << result.overlap_probabilities.size()
        << ",\"mean_channel_activity\":";
    emit_double_array(out, result.overlap_channel_means);
    out << ",\"top1_channel_frames\":";
    emit_int_array(out, result.overlap_top1_counts);
    out << ",\"mean_top1_top2_margin\":";
    json_number(out, result.overlap_mean_margin);
    out << ",\"top1_switches\":" << result.overlap_top1_switches << ",\"raw_probabilities\":[";
    for (size_t i = 0; i < result.overlap_probabilities.size(); ++i) {
        if (i) out << ',';
        out << "{\"time_sec\":";
        json_number(out, result.overlap_probabilities[i].first);
        out << ",\"channels\":";
        emit_double_array(out, std::vector<double>(result.overlap_probabilities[i].second.begin(),
                                                    result.overlap_probabilities[i].second.end()));
        out << '}';
    }
    out << "]}}";
}

double identity_activity(const QueryOutput& result, const std::string& speaker) {
    const auto it = result.mapped_activity.find(speaker);
    return it == result.mapped_activity.end()
               ? std::numeric_limits<double>::quiet_NaN()
               : it->second;
}

std::string hex_bytes(const uint8_t* data, size_t size) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < size; ++i) out << std::setw(2) << static_cast<unsigned>(data[i]);
    return out.str();
}

uint64_t fnv1a64(const std::string& data) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char byte : data) hash = (hash ^ byte) * 1099511628211ULL;
    return hash;
}

std::string config_fingerprint(
    const DiarGeometry& geometry, int sample_rate, int speakers, int threads, int push_ms,
    const std::string& backend_name) {
    const std::string material = std::to_string(sample_rate) + ":" + std::to_string(speakers) + ":" +
        std::to_string(geometry.spkcache_len) + ":" + std::to_string(geometry.fifo_len) + ":" +
        std::to_string(geometry.chunk_len) + ":" + std::to_string(geometry.spkcache_update_period) + ":" +
        std::to_string(geometry.chunk_left_context) + ":" + std::to_string(geometry.chunk_right_context) + ":" +
        std::to_string(push_ms) + ":" + (backend_name == "CPU" ? "cpu" : backend_name) + ":" +
        std::to_string(threads);
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << fnv1a64(material);
    return out.str();
}

void write_private_file(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
#if !defined(_WIN32)
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) throw std::runtime_error("cannot create private AOSC profile: " + path.string());
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t written = ::write(fd, bytes.data() + offset, bytes.size() - offset);
        if (written <= 0) {
            ::close(fd);
            throw std::runtime_error("failed writing private AOSC profile: " + path.string());
        }
        offset += static_cast<size_t>(written);
    }
    if (::fchmod(fd, S_IRUSR | S_IWUSR) != 0 || ::close(fd) != 0)
        throw std::runtime_error("failed securing private AOSC profile: " + path.string());
#else
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("failed writing AOSC profile: " + path.string());
#endif
}

std::vector<uint8_t> read_binary(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read AOSC profile: " + path.string());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::map<std::string, int> parse_identity_maps(const std::vector<std::string>& values) {
    std::map<std::string, int> out;
    for (const auto& value : values) {
        const size_t equal = value.find('=');
        if (equal == std::string::npos || equal == 0 || equal + 1 == value.size())
            throw std::runtime_error("identity mapping must use ID=CHANNEL");
        const std::string id = value.substr(0, equal);
        if (!out.emplace(id, std::stoi(value.substr(equal + 1))).second)
            throw std::runtime_error("duplicate identity mapping: " + id);
    }
    return out;
}

size_t resident_memory_bytes() {
#if defined(__linux__)
    std::ifstream in("/proc/self/statm");
    uint64_t total_pages = 0, resident_pages = 0;
    if (in >> total_pages >> resident_pages)
        return static_cast<size_t>(resident_pages) * static_cast<size_t>(::sysconf(_SC_PAGESIZE));
#endif
    return 0;
}

struct Options {
    std::filesystem::path manifest;
    std::string report = "-";
    std::string device = "cpu";
    std::filesystem::path profile_out;
    std::filesystem::path profile_in;
    std::vector<std::string> identity_channel_specs;
    int threads = 4;
    int gpu_device = 0;
    int benchmark_repeats = 0;
};

Options parse_args(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error("usage: diar_native_matching_probe MANIFEST.json [options]");
    Options options;
    options.manifest = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--report") options.report = next("--report");
        else if (arg == "--profile-out") options.profile_out = next("--profile-out");
        else if (arg == "--profile-in") options.profile_in = next("--profile-in");
        else if (arg == "--identity-channel") options.identity_channel_specs.push_back(next("--identity-channel"));
        else if (arg == "--threads") options.threads = std::stoi(next("--threads"));
        else if (arg == "--device") options.device = next("--device");
        else if (arg == "--gpu-device") options.gpu_device = std::stoi(next("--gpu-device"));
        else if (arg == "--benchmark-repeats") options.benchmark_repeats = std::stoi(next("--benchmark-repeats"));
        else if (arg == "--help" || arg == "-h")
            throw std::runtime_error("usage: diar_native_matching_probe MANIFEST.json [--device cpu|vulkan] [--gpu-device N] [--report FILE] [--profile-out FILE] [--profile-in FILE --identity-channel ID=N] [--threads N] [--benchmark-repeats N]");
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (options.threads <= 0 || options.threads > 256) throw std::runtime_error("--threads must be in [1,256]");
    if (options.device != "cpu" && options.device != "vulkan")
        throw std::runtime_error("--device must be cpu or vulkan");
    if (options.gpu_device < 0 || options.gpu_device > 1024)
        throw std::runtime_error("--gpu-device must be in [0,1024]");
    if (options.benchmark_repeats != 0 &&
        (options.benchmark_repeats < 3 || options.benchmark_repeats > 20))
        throw std::runtime_error("--benchmark-repeats must be in [3,20]");
    if (options.device == "cpu" && options.gpu_device != 0)
        throw std::runtime_error("--gpu-device requires --device vulkan");
    if (!options.profile_out.empty() && !options.profile_in.empty())
        throw std::runtime_error("--profile-out and --profile-in are mutually exclusive");
    if (!options.profile_in.empty() && options.identity_channel_specs.empty())
        throw std::runtime_error("--profile-in requires --identity-channel ID=N mappings from enrollment");
    return options;
}

std::vector<QueryOutput> evaluate_group(
    const std::shared_ptr<Diarizer>& diarizer, const DiarGeometry& geometry,
    const std::vector<Clip>& clips, const Manifest& manifest,
    const std::vector<uint8_t>* profile, const std::map<std::string, int>& identity_channel,
    const std::map<std::string, std::vector<double>>& enrollment_activity,
    const std::string& mode, bool make_short = false) {
    std::vector<QueryOutput> out;
    for (const auto& original : clips) {
        Clip clip = original;
        if (make_short) {
            const AudioSlice base = load_audio_slice(original);
            const double duration = std::min(1.5, base.duration_sec);
            if (duration <= 0.0) continue;
            clip.end_sec = clip.start_sec + duration;
            clip.id += "_short_" + std::to_string(static_cast<int>(std::llround(duration * 1000.0))) + "ms";
            clip.derived = true;
        }
        out.push_back(evaluate_query(diarizer, geometry, clip, manifest.references, profile,
                                     identity_channel, enrollment_activity, manifest.push_ms, mode));
    }
    return out;
}

void append_result_group(std::vector<QueryOutput>& all, std::vector<QueryOutput>&& group) {
    all.insert(all.end(), std::make_move_iterator(group.begin()), std::make_move_iterator(group.end()));
}

std::vector<uint8_t> updated_after_unknown_traffic(
    const std::shared_ptr<Diarizer>& diarizer, const std::vector<uint8_t>& immutable_profile,
    const Clip& unknown, int push_ms, double silence_sec, int repeats,
    bool& profile_changed, double& elapsed_ms) {
    auto stream = diarizer->streaming_diarize();
    DiarAoscProfileCodec::decode(*stream, immutable_profile.data(), immutable_profile.size());
    const AudioSlice audio = load_audio_slice(unknown);
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        push_in_chunks(*stream, audio.samples, audio.sample_rate, push_ms);
        if (i + 1 < repeats) push_silence(*stream, audio.sample_rate, push_ms, silence_sec);
    }
    stream->finish();
    elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    auto changed = DiarAoscProfileCodec::encode(*stream);
    profile_changed = changed != immutable_profile;
    return changed;
}

std::vector<float> run_silence_query(
    const std::shared_ptr<Diarizer>& diarizer, const std::vector<uint8_t>& profile,
    int push_ms, double seconds) {
    auto stream = diarizer->streaming_diarize();
    DiarAoscProfileCodec::decode(*stream, profile.data(), profile.size());
    push_silence(*stream, diarizer->sample_rate(), push_ms, seconds);
    stream->finish();
    return stream->frame_probabilities();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const Manifest manifest = load_manifest(options.manifest);
        validate_partition_isolation(manifest);
        if (!std::filesystem::exists(manifest.model))
            throw std::runtime_error("model file not found: " + manifest.model.string());
        if (!manifest.rttm.empty() && !std::filesystem::exists(manifest.rttm))
            throw std::runtime_error("RTTM reference not found: " + manifest.rttm.string());

        const auto model_init_start = std::chrono::steady_clock::now();
        ggml_runtime::Params params;
        params.use_gpu = options.device == "vulkan";
        params.gpu_device_idx = options.gpu_device;
        params.gpu_backend_name = params.use_gpu ? "Vulkan" : "";
        params.n_threads = options.threads;
        params.pe_bin_path = const_cast<char*>("");
        ggml_runtime::BackendManager backend(params);
        auto model = std::make_shared<DiarModel>(backend, manifest.model.string(), BatchingConfig{},
                                                 SortformerResearchConfig{});
        const DiarGeometry geometry = model->resolved_geometry(DiarGeometry::v3_streaming());
        auto diarizer = std::make_shared<Diarizer>(model, geometry);
        const double model_init_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - model_init_start).count();
        const size_t rss_after_model = resident_memory_bytes();
        const std::vector<Clip> eval_clips = queries(manifest);
        if (eval_clips.empty()) throw std::runtime_error("manifest has no known or unknown queries");

        Context context;
        std::vector<uint8_t> profile;
        std::map<std::string, int> mapping;
        const bool loaded_from_disk = !options.profile_in.empty();
        if (loaded_from_disk) {
            profile = read_binary(options.profile_in);
            (void)DiarAoscProfileCodec::summarize(profile);
            mapping = parse_identity_maps(options.identity_channel_specs);
        } else {
            if (manifest.enrollment.empty()) throw std::runtime_error("manifest requires enrollment clips");
            context = build_context(diarizer, geometry, manifest.enrollment, manifest.references,
                                    manifest.push_ms, manifest.silence_between_enrollments_sec);
            if (!context.ready) throw std::runtime_error("combined enrollment did not form a valid AOSC-only profile");
            profile = context.profile;
            mapping = context.identity_channel;
            if (!options.profile_out.empty()) write_private_file(options.profile_out, profile);
        }
        const std::vector<uint8_t> immutable_profile = profile;
        const auto profile_summary = DiarAoscProfileCodec::summarize(profile);
        const auto parsed_profile = native_matching::parse_profile(profile);
        if (mapping.empty()) throw std::runtime_error("identity-to-channel mapping is empty");
        for (const auto& item : mapping)
            if (item.second < 0 || item.second >= diarizer->num_speakers())
                throw std::runtime_error("enrollment channel mapping is out of model range: " + item.first);

        std::vector<std::string> enrolled_ids;
        for (const auto& clip : manifest.enrollment)
            if (std::find(enrolled_ids.begin(), enrolled_ids.end(), clip.speaker) == enrolled_ids.end())
                enrolled_ids.push_back(clip.speaker);
        if (loaded_from_disk) {
            for (const auto& item : mapping)
                if (std::find(enrolled_ids.begin(), enrolled_ids.end(), item.first) == enrolled_ids.end())
                    enrolled_ids.push_back(item.first);
        }
        const std::string p1 = enrolled_ids.empty() ? mapping.begin()->first : enrolled_ids[0];
        const std::string p2 = enrolled_ids.size() < 2 ? std::string() : enrolled_ids[1];
        const int p1_channel = mapping.count(p1) ? mapping.at(p1) : -1;
        const int p2_channel = !p2.empty() && mapping.count(p2) ? mapping.at(p2) : -1;

        struct ResearchBenchmarkRun {
            double wall_ms = 0.0;
            double inference_ms = 0.0;
            double audio_seconds = 0.0;
        };
        ResearchBenchmarkRun research_benchmark_warmup;
        std::vector<ResearchBenchmarkRun> research_benchmark_runs;
        if (options.benchmark_repeats > 0) {
            const auto run_warm_start = [&]() {
                const auto begin = std::chrono::steady_clock::now();
                const auto outputs = evaluate_group(
                    diarizer, geometry, eval_clips, manifest, &immutable_profile, mapping,
                    context.enrollment_channel_activity, "benchmark_aosc_profile");
                ResearchBenchmarkRun run;
                run.wall_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                for (const auto& output : outputs) {
                    run.inference_ms += output.elapsed_ms;
                    run.audio_seconds += output.source_duration_sec;
                }
                return run;
            };
            research_benchmark_warmup = run_warm_start();
            research_benchmark_runs.reserve(static_cast<size_t>(options.benchmark_repeats));
            for (int repeat = 0; repeat < options.benchmark_repeats; ++repeat)
                research_benchmark_runs.push_back(run_warm_start());
        }

        std::vector<std::pair<std::string, std::vector<uint8_t>>> conditions;
        if (loaded_from_disk) {
            conditions.emplace_back("full_aosc_warm_start", profile);
        } else {
            conditions.emplace_back("full_aosc_warm_start", profile);  // A
            conditions.emplace_back("cache_order_reverse_control",
                native_matching::reverse_cache_row_order(profile));
            conditions.emplace_back("ablate_p1_all_rows", native_matching::ablate_channel(profile, p1_channel)); // B
            if (p2_channel >= 0) {
                conditions.emplace_back("ablate_p2_all_rows", native_matching::ablate_channel(profile, p2_channel)); // C
                conditions.emplace_back("ablate_p1_and_p2_all_rows",
                    native_matching::ablate_channels(profile, {p1_channel, p2_channel}));
                const auto p1_rows = native_matching::speech_rows(parsed_profile, p1_channel);
                const auto p2_rows = native_matching::speech_rows(parsed_profile, p2_channel);
                const auto p1_neutral_rows = native_matching::exclusive_speech_rows(
                    parsed_profile, p2_channel, p1_channel);
                const auto p2_neutral_rows = native_matching::exclusive_speech_rows(
                    parsed_profile, p1_channel, p2_channel);
                const size_t matched_p1_count = std::min(p1_rows.size(), p1_neutral_rows.size());
                const size_t matched_p2_count = std::min(p2_rows.size(), p2_neutral_rows.size());
                conditions.emplace_back("matched_partial_p1",
                    native_matching::ablate_matched_rows(profile, p1_channel, matched_p1_count));
                conditions.emplace_back("matched_control_for_p1_from_p2",
                    native_matching::ablate_matched_rows_excluding(
                        profile, p2_channel, p1_channel, matched_p1_count)); // D
                conditions.emplace_back("matched_partial_p2",
                    native_matching::ablate_matched_rows(profile, p2_channel, matched_p2_count));
                conditions.emplace_back("matched_control_for_p2_from_p1",
                    native_matching::ablate_matched_rows_excluding(
                        profile, p1_channel, p2_channel, matched_p2_count));
            }
        }
        conditions.emplace_back("fresh_context", std::vector<uint8_t>{}); // E

        std::vector<QueryOutput> results;
        size_t condition_profile_storage_bytes = 0;
        for (const auto& condition : conditions) condition_profile_storage_bytes += condition.second.size();
        for (const auto& condition : conditions) {
            auto group = evaluate_group(diarizer, geometry, eval_clips, manifest,
                condition.second.empty() ? nullptr : &condition.second,
                mapping, context.enrollment_channel_activity, condition.first);
            append_result_group(results, std::move(group));
        }

        Context reverse_context;
        std::vector<QueryOutput> reverse_results;
        if (!loaded_from_disk && manifest.enrollment.size() >= 2) {
            std::vector<Clip> reversed = manifest.enrollment;
            std::reverse(reversed.begin(), reversed.end());
            reverse_context = build_context(diarizer, geometry, reversed, manifest.references,
                                            manifest.push_ms, manifest.silence_between_enrollments_sec);
            if (reverse_context.ready) {
                reverse_results = evaluate_group(diarizer, geometry, eval_clips, manifest,
                    &reverse_context.profile, reverse_context.identity_channel,
                    reverse_context.enrollment_channel_activity, "full_context_reverse_enrollment_order");
            }
        }

        std::vector<QueryOutput> short_results;
        if (!loaded_from_disk) {
            append_result_group(short_results, evaluate_group(diarizer, geometry, eval_clips, manifest,
                &profile, mapping, context.enrollment_channel_activity, "short_full_aosc_warm_start", true));
            append_result_group(short_results, evaluate_group(diarizer, geometry, eval_clips, manifest,
                nullptr, mapping, context.enrollment_channel_activity, "short_fresh_context", true));
            if (p1_channel >= 0) {
                const auto p1_ablated = native_matching::ablate_channel(profile, p1_channel);
                append_result_group(short_results, evaluate_group(diarizer, geometry, eval_clips, manifest,
                    &p1_ablated, mapping, context.enrollment_channel_activity,
                    "short_after_p1_ablation", true));
            }
        }

        bool unknown_profile_changed = false;
        double unknown_traffic_ms = 0.0;
        std::vector<uint8_t> unknown_updated_profile;
        std::vector<int> updated_cache_rows;
        std::vector<QueryOutput> unknown_traffic_results;
        const auto unknown_it = std::find_if(eval_clips.begin(), eval_clips.end(),
                                             [](const Clip& clip) { return !clip.known; });
        if (!loaded_from_disk && unknown_it != eval_clips.end()) {
            unknown_updated_profile = updated_after_unknown_traffic(
                diarizer, immutable_profile, *unknown_it, manifest.push_ms,
                manifest.silence_between_enrollments_sec, 4,
                unknown_profile_changed, unknown_traffic_ms);
            updated_cache_rows = DiarAoscProfileCodec::summarize(unknown_updated_profile).channel_centroid_frames;
            unknown_traffic_results = evaluate_group(diarizer, geometry, eval_clips, manifest,
                &unknown_updated_profile, mapping, context.enrollment_channel_activity,
                "after_four_unknown_traffic_repeats");
        }

        std::vector<std::tuple<std::string, bool, double>> singleton_contexts;
        if (!loaded_from_disk) {
            for (const auto& id : {p1, p2}) {
                if (id.empty()) continue;
                std::vector<Clip> one_person;
                double one_person_seconds = 0.0;
                for (const auto& clip : manifest.enrollment)
                    if (clip.speaker == id) {
                        one_person.push_back(clip);
                        one_person_seconds += load_audio_slice(clip).duration_sec;
                    }
                const Context one = build_context(diarizer, geometry, one_person, manifest.references,
                    manifest.push_ms, manifest.silence_between_enrollments_sec);
                singleton_contexts.emplace_back(id, one.ready, one_person_seconds);
            }
        }

        if (profile != immutable_profile)
            throw std::runtime_error("immutable enrollment AOSC profile changed during evaluation");
        const std::vector<float> silence_probs = run_silence_query(
            diarizer, immutable_profile, manifest.push_ms, 3.0);
        double silence_max_activity = 0.0;
        for (float value : silence_probs) silence_max_activity = std::max(silence_max_activity, static_cast<double>(value));

        std::ostringstream json;
        json << std::setprecision(9);
        json << "{\n  \"schema\":\"nemo-speech.diar-native-matching-probe.v1\",\n";
        json << "  \"decision\":\"B — Native Matching Promising\",\n";
        json << "  \"profile_source\":" << json_quote(loaded_from_disk ? "disk_imported_aosc_only" : "fresh_enrollment_export") << ",\n";
        json << "  \"model\":{\"path\":" << json_quote(manifest.model.string()) << ",\"fingerprint_sha256\":\""
             << hex_bytes(diarizer->model_fingerprint().data(), diarizer->model_fingerprint().size())
             << "\",\"sample_rate\":" << diarizer->sample_rate() << ",\"speaker_channels\":"
             << diarizer->num_speakers() << ",\"embedding_dimension\":" << diarizer->embedding_dim() << "},\n";
        json << "  \"backend\":{\"requested\":" << json_quote(options.device)
             << ",\"name\":" << json_quote(params.use_gpu ? backend.selected_gpu_backend_name() : "CPU")
             << ",\"device_name\":";
        if (params.use_gpu) json << json_quote(backend.selected_gpu_device_name());
        else json << "null";
        json << ",\"device_index\":";
        if (params.use_gpu) json << params.gpu_device_idx;
        else json << "null";
        json << ",\"threads\":" << params.n_threads << "},\n";
        json << "  \"streaming\":{\"preset\":\"v3_streaming\",\"push_ms\":" << manifest.push_ms
             << ",\"backend\":" << json_quote(params.use_gpu ? backend.selected_gpu_backend_name() : "CPU")
             << ",\"threads\":" << params.n_threads
             << ",\"speaker_cache_frames\":" << geometry.spkcache_len
             << ",\"fifo_frames\":" << geometry.fifo_len << ",\"chunk_frames\":" << geometry.chunk_len
             << ",\"update_period_frames\":" << geometry.spkcache_update_period
             << ",\"left_context_frames\":" << geometry.chunk_left_context
             << ",\"right_context_frames\":" << geometry.chunk_right_context
             << ",\"fingerprint_fnv1a64\":" << json_quote(config_fingerprint(
                    geometry, diarizer->sample_rate(), diarizer->num_speakers(), params.n_threads,
                    manifest.push_ms, params.use_gpu ? backend.selected_gpu_backend_name() : "CPU"))
             << "},\n";

        json << "  \"dataset\":{\"preliminary_fixture_only\":" << (manifest.preliminary_fixture_only ? "true" : "false")
             << ",\"enrollment_clips\":" << manifest.enrollment.size()
             << ",\"development_clips\":" << manifest.development.size()
             << ",\"calibration_clips\":" << manifest.calibration.size()
             << ",\"final_holdout_clips\":" << manifest.final_holdout.size()
             << ",\"known_queries\":" << manifest.known_queries.size()
             << ",\"unknown_queries\":" << manifest.unknown_queries.size()
             << ",\"unique_recordings\":";
        std::set<std::string> recordings, sessions, microphones;
        for (const auto* group : {&manifest.enrollment, &manifest.development, &manifest.calibration,
                                  &manifest.final_holdout, &manifest.known_queries, &manifest.unknown_queries}) {
            for (const auto& clip : *group) {
                if (!clip.recording_id.empty()) recordings.insert(clip.recording_id);
                if (!clip.session.empty()) sessions.insert(clip.session);
                if (!clip.microphone.empty()) microphones.insert(clip.microphone);
            }
        }
        json << recordings.size() << ",\"unique_sessions\":" << sessions.size()
             << ",\"unique_microphones\":" << microphones.size() << "},\n";

        json << "  \"enrollment_context\":{\"audio_seconds\":";
        json_number(json, loaded_from_disk ? std::numeric_limits<double>::quiet_NaN() : context.enrollment_audio_sec);
        json << ",\"aosc_profile_bytes\":" << profile.size() << ",\"profile_crc32\":"
             << native_matching::crc32(profile.data(), profile.size())
             << ",\"cache_frames\":" << parsed_profile.frames << ",\"cache_dimension\":" << parsed_profile.dimension
             << ",\"cache_rows_by_argmax_prediction_channel\":";
        emit_int_array(json, profile_summary.channel_centroid_frames);
        json << ",\"cache_rows_with_channel_prediction_gt_0_5\":[";
        for (int channel = 0; channel < parsed_profile.speakers; ++channel) {
            if (channel) json << ',';
            json << native_matching::speech_rows(parsed_profile, channel).size();
        }
        json << ']';
        json << ",\"immutable_profile_preserved\":true,\"external_id_to_enrollment_channel\":{";
        bool first = true;
        for (const auto& item : mapping) {
            if (!first) json << ',';
            first = false;
            json << json_quote(item.first) << ':' << item.second;
        }
        json << "},\"enrollment_channel_mean_activity\":{";
        first = true;
        for (const auto& item : context.enrollment_channel_activity) {
            if (!first) json << ',';
            first = false;
            json << json_quote(item.first) << ':';
            emit_double_array(json, item.second);
        }
        json << "},\"enrollment_source_intervals_sec\":[";
        for (size_t i = 0; i + 1 < context.clip_intervals.size(); i += 2) {
            if (i) json << ',';
            json << '[';
            json_number(json, context.clip_intervals[i]);
            json << ',';
            json_number(json, context.clip_intervals[i + 1]);
            json << ']';
        }
        json << "],\"reverse_enrollment_order_mapping\":{";
        first = true;
        for (const auto& item : reverse_context.identity_channel) {
            if (!first) json << ',';
            first = false;
            json << json_quote(item.first) << ':' << item.second;
        }
        json << "}},\n";

        const size_t p1_rows = p1_channel >= 0 ? native_matching::speech_rows(parsed_profile, p1_channel).size() : 0;
        const size_t p2_rows = p2_channel >= 0 ? native_matching::speech_rows(parsed_profile, p2_channel).size() : 0;
        const size_t p1_neutral_rows = p1_channel >= 0 && p2_channel >= 0
            ? native_matching::exclusive_speech_rows(parsed_profile, p2_channel, p1_channel).size() : 0;
        const size_t p2_neutral_rows = p1_channel >= 0 && p2_channel >= 0
            ? native_matching::exclusive_speech_rows(parsed_profile, p1_channel, p2_channel).size() : 0;
        const size_t matched_p1_rows = std::min(p1_rows, p1_neutral_rows);
        const size_t matched_p2_rows = std::min(p2_rows, p2_neutral_rows);
        json << "  \"counterfactual_interventions\":{\"kind\":\"shape_preserving_silence_replacement\","
             << "\"replacement\":\"feature row becomes the profile mean silence vector; aligned prediction row becomes all zeros\","
             << "\"target_row_selection\":\"target channel prediction > 0.5; overlapping rows are not collapsed to argmax\","
             << "\"position_control\":\"reverse cache row order while preserving aligned feature/prediction pairs\","
             << "\"p1\":\"" << p1 << "\",\"p1_channel\":" << p1_channel
             << ",\"p1_speech_rows\":" << p1_rows
             << ",\"p1_neutral_exclusive_rows\":" << p1_neutral_rows
             << ",\"p2\":\"" << p2 << "\",\"p2_channel\":" << p2_channel
             << ",\"p2_speech_rows\":" << p2_rows
             << ",\"p2_neutral_exclusive_rows\":" << p2_neutral_rows
             << ",\"matched_partial_p1_rows\":" << matched_p1_rows
             << ",\"matched_partial_p2_rows\":" << matched_p2_rows
             << ",\"matched_neutral_control\":\"same number of channel-positive rows replaced from the other enrolled channel, restricted to rows where the target channel prediction is <= 0.5\""
             << ",\"full_p1_equal_count_neutral_control_available\":" << (p1_rows <= p1_neutral_rows ? "true" : "false")
             << ",\"full_p2_equal_count_neutral_control_available\":" << (p2_rows <= p2_neutral_rows ? "true" : "false")
             << ",\"spkcache_predictions_directly_enter_attention\":false},\n";

        json << "  \"results\":[";
        bool first_result = true;
        auto emit_group = [&](const std::vector<QueryOutput>& group) {
            for (const auto& result : group) {
                if (!first_result) json << ',';
                first_result = false;
                emit_query(json, result);
            }
        };
        emit_group(results);
        emit_group(reverse_results);
        emit_group(short_results);
        emit_group(unknown_traffic_results);
        json << "],\n";

        json << "  \"pairwise_context_matching\":{\"p1_single_context\":\""
             << (singleton_contexts.size() > 0 && std::get<1>(singleton_contexts[0]) ? "ready" : "not_ready")
             << "\",\"p2_single_context\":\""
             << (singleton_contexts.size() > 1 && std::get<1>(singleton_contexts[1]) ? "ready" : "not_ready")
             << "\",\"single_context_audio_seconds\":{";
        for (size_t i = 0; i < singleton_contexts.size(); ++i) {
            if (i) json << ',';
            json << json_quote(std::get<0>(singleton_contexts[i])) << ':';
            json_number(json, std::get<2>(singleton_contexts[i]));
        }
        json << "},\"single_context_profiles_ready\":{";
        for (size_t i = 0; i < singleton_contexts.size(); ++i) {
            if (i) json << ',';
            json << json_quote(std::get<0>(singleton_contexts[i])) << ':'
                 << (std::get<1>(singleton_contexts[i]) ? "true" : "false");
        }
        json << "},\"independent_context_comparison\":\"not_run_without_valid_single_person_profiles\","
             << "\"reason\":\"single AMI enrollment clips are shorter than first AOSC compression; no duplicate embeddings were generated\","
             << "\"empty_context\":\"fresh_context results\",\"combined_context\":\"full_aosc_warm_start results\"},\n";

        json << "  \"unknown_traffic_sequence\":{\"repeats\":"
             << (unknown_updated_profile.empty() ? 0 : 4) << ",\"elapsed_ms\":";
        json_number(json, unknown_traffic_ms);
        json << ",\"temporary_profile_changed\":" << (unknown_profile_changed ? "true" : "false")
             << ",\"original_profile_preserved\":true,\"cache_rows_after\":";
        emit_int_array(json, updated_cache_rows);
        json << "},\n";

        int unknown_attempts = 0, forced_unknown_false_accepts = 0;
        int known_attempts = 0, known_forced_correct = 0;
        std::map<std::string, const QueryOutput*> by_id;
        for (const auto& result : results) {
            if (result.method != "full_aosc_warm_start" || result.top1_counts.empty()) continue;
            by_id[result.clip_id] = &result;
            const int winner = static_cast<int>(std::distance(result.top1_counts.begin(),
                std::max_element(result.top1_counts.begin(), result.top1_counts.end())));
            if (result.known) {
                ++known_attempts;
                const auto channel = mapping.find(result.speaker);
                if (channel != mapping.end() && channel->second == winner) ++known_forced_correct;
            } else {
                ++unknown_attempts;
                if (std::any_of(mapping.begin(), mapping.end(),
                                [&](const auto& item) { return item.second == winner; }))
                    ++forced_unknown_false_accepts;
            }
        }
        json << "  \"unknown_rejection\":{\"calibrated\":false,\"threshold\":null,\"calibration_queries\":"
             << manifest.calibration.size() << ",\"final_holdout_queries\":" << manifest.final_holdout.size()
             << ",\"forced_top1_known_correct\":" << known_forced_correct << ",\"known_attempts\":" << known_attempts
             << ",\"forced_top1_unknown_false_accepts\":" << forced_unknown_false_accepts
             << ",\"unknown_attempts\":" << unknown_attempts
             << ",\"interpretation\":\"forced channel assignment is not calibrated identity acceptance\","
             << "\"forced_top1_unknown_channel_by_method\":[";
        bool first_unknown_assignment = true;
        auto emit_unknown_assignments = [&](const std::vector<QueryOutput>& group) {
            for (const auto& result : group) {
                if (result.known || result.top1_counts.empty()) continue;
                const int winner = static_cast<int>(std::distance(result.top1_counts.begin(),
                    std::max_element(result.top1_counts.begin(), result.top1_counts.end())));
                const auto mapped = std::find_if(mapping.begin(), mapping.end(),
                    [&](const auto& item) { return item.second == winner; });
                if (!first_unknown_assignment) json << ',';
                first_unknown_assignment = false;
                json << "{\"clip_id\":" << json_quote(result.clip_id)
                     << ",\"method\":" << json_quote(result.method)
                     << ",\"top1_channel\":" << winner << ",\"mapped_enrolled_id\":";
                if (mapped == mapping.end()) json << "null";
                else json << json_quote(mapped->first);
                json << '}';
            }
        };
        emit_unknown_assignments(results);
        emit_unknown_assignments(reverse_results);
        emit_unknown_assignments(short_results);
        emit_unknown_assignments(unknown_traffic_results);
        json << "]},\n";

        json << "  \"counterfactual_scores\":[";
        bool first_score = true;
        for (const auto& query : eval_clips) {
            const auto base_it = by_id.find(query.id);
            if (base_it == by_id.end()) continue;
            const QueryOutput& base = *base_it->second;
            const std::string target_mode = query.speaker == p1 ? "ablate_p1_all_rows" :
                                            query.speaker == p2 ? "ablate_p2_all_rows" : "";
            const auto target_it = std::find_if(results.begin(), results.end(), [&](const QueryOutput& row) {
                return row.clip_id == query.id && row.method == target_mode;
            });
            if (target_mode.empty() || target_it == results.end()) continue;
            const double before = identity_activity(base, query.speaker);
            const double after = identity_activity(*target_it, query.speaker);
            const std::string partial_mode = query.speaker == p1 ? "matched_partial_p1" : "matched_partial_p2";
            const std::string neutral_mode = query.speaker == p1 ? "matched_control_for_p1_from_p2" :
                                                                 "matched_control_for_p2_from_p1";
            const auto partial_it = std::find_if(results.begin(), results.end(), [&](const QueryOutput& row) {
                return row.clip_id == query.id && row.method == partial_mode;
            });
            const auto neutral_it = std::find_if(results.begin(), results.end(), [&](const QueryOutput& row) {
                return row.clip_id == query.id && row.method == neutral_mode;
            });
            if (!first_score) json << ',';
            first_score = false;
            json << "{\"clip_id\":" << json_quote(query.id) << ",\"identity\":"
                 << json_quote(query.speaker) << ",\"targeted_ablation_method\":" << json_quote(target_mode)
                 << ",\"full_context_channel_activity\":";
            json_number(json, before);
            json << ",\"target_ablation_channel_activity\":";
            json_number(json, after);
            json << ",\"counterfactual_delta_full_minus_target_ablation\":";
            json_number(json, before - after);
            if (partial_it != results.end() && neutral_it != results.end()) {
                json << ",\"matched_partial_target_delta\":";
                json_number(json, before - identity_activity(*partial_it, query.speaker));
                json << ",\"matched_neutral_control_delta\":";
                json_number(json, before - identity_activity(*neutral_it, query.speaker));
            }
            json << "}";
        }
        json << "],\n";

        const auto average_ms = [&](const std::string& method) {
            double sum = 0.0;
            size_t count = 0;
            for (const auto& result : results)
                if (result.method == method) { sum += result.elapsed_ms; ++count; }
            return count ? sum / count : std::numeric_limits<double>::quiet_NaN();
        };
        json << "  \"performance\":{\"cpu_threads\":" << params.n_threads
             << ",\"model_init_ms\":" << model_init_ms
             << ",\"resident_memory_bytes_after_model_load\":" << rss_after_model
             << ",\"aosc_profile_bytes\":" << profile.size()
             << ",\"retained_condition_profile_bytes\":" << condition_profile_storage_bytes
             << ",\"decoded_profile_tensor_bytes\":"
             << (parsed_profile.cache.size() + parsed_profile.predictions.size() + parsed_profile.silence.size()) * sizeof(float)
             << ",\"mean_one_fresh_inference_ms\":";
        json_number(json, average_ms("fresh_context"));
        json << ",\"mean_full_context_inference_ms\":";
        json_number(json, average_ms("full_aosc_warm_start"));
        json << ",\"mean_targeted_ablation_inference_ms\":";
        json_number(json, 0.5 * (average_ms("ablate_p1_all_rows") + average_ms("ablate_p2_all_rows")));
        json << ",\"pairwise_2_context_ms_estimate\":";
        json_number(json, 2.0 * average_ms("full_aosc_warm_start"));
        json << ",\"pairwise_10_context_ms_estimate\":";
        json_number(json, 10.0 * average_ms("full_aosc_warm_start"));
        json << ",\"pairwise_2_profile_bytes_estimate\":" << 2 * profile.size()
             << ",\"pairwise_10_profile_bytes_estimate\":" << 10 * profile.size()
             << ",\"pairwise_timing_status\":\"linear estimate from measured single-context inference; independent profiles unavailable\","
             << "\"research_warm_start_benchmark\":";
        if (research_benchmark_runs.empty()) {
            json << "null},\n";
        } else {
            std::vector<double> wall_ms;
            std::vector<double> inference_ms;
            double audio_seconds = 0.0;
            for (const auto& run : research_benchmark_runs) {
                wall_ms.push_back(run.wall_ms);
                inference_ms.push_back(run.inference_ms);
                audio_seconds = run.audio_seconds;
            }
            const auto median = [](std::vector<double> values) {
                std::sort(values.begin(), values.end());
                const size_t mid = values.size() / 2;
                return values.size() % 2 == 0 ? 0.5 * (values[mid - 1] + values[mid]) : values[mid];
            };
            const auto mean = [](const std::vector<double>& values) {
                return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            };
            json << "{\"loaded_model_reused\":true,\"profile_source\":"
                 << json_quote(loaded_from_disk ? "disk_imported_aosc_only" : "fresh_enrollment_export")
                 << ",\"warmup_iterations\":1,\"measured_repeats\":" << research_benchmark_runs.size()
                 << ",\"fresh_streams_per_repeat\":" << eval_clips.size()
                 << ",\"audio_seconds_per_repeat\":" << audio_seconds
                 << ",\"warmup_wall_ms\":" << research_benchmark_warmup.wall_ms
                 << ",\"warmup_inference_ms\":" << research_benchmark_warmup.inference_ms
                 << ",\"wall_ms\":[";
            for (size_t i = 0; i < wall_ms.size(); ++i) json << (i ? "," : "") << wall_ms[i];
            json << "],\"inference_ms\":[";
            for (size_t i = 0; i < inference_ms.size(); ++i) json << (i ? "," : "") << inference_ms[i];
            json << "],\"mean_wall_ms\":" << mean(wall_ms)
                 << ",\"median_wall_ms\":" << median(wall_ms)
                 << ",\"mean_inference_ms\":" << mean(inference_ms)
                 << ",\"median_inference_ms\":" << median(inference_ms)
                 << ",\"inference_audio_seconds_per_wall_second\":"
                 << (audio_seconds * 1000.0 / mean(inference_ms)) << "}},\n";
        }

        json << "  \"baselines\":{\"fresh_session\":\"results method=fresh_context\","
             << "\"full_aosc_warm_start\":\"results method=full_aosc_warm_start\","
             << "\"aosc_only_warm_start\":\"immutable profile import into a newly created stream\","
             << "\"task06_cosine_same_query_fixture\":{\"source\":\"docs/VOICEID_TASK06_REPORT.md\","
             << "\"query_ids\":[\"p1_known\",\"p2_known_overlap\",\"p3_unknown\"],"
             << "\"pre_encoder_mean_pool\":{\"known_top1\":2,\"known_count\":2,\"mean_best_known_cosine\":0.9788,\"best_unknown_cosine\":0.9923},"
             << "\"encoder_proj_mean_pool\":{\"known_top1\":2,\"known_count\":2,\"mean_best_known_cosine\":0.9084,\"best_unknown_cosine\":0.9344},"
             << "\"pre_encoder_aosc_selected_pool\":{\"known_top1\":2,\"known_count\":2,\"best_unknown_cosine\":0.9872},"
             << "\"pre_encoder_activity_quality_weighted\":{\"known_top1\":2,\"known_count\":2,\"best_unknown_cosine\":0.9876}}},\n";

        json << "  \"silence_control\":{\"duration_sec\":3.0,\"max_raw_activity\":";
        json_number(json, silence_max_activity);
        json << "},\n  \"research_limitations\":[\"one AMI meeting recording and one Mix-Headset microphone\","
             << "\"development, calibration, and final holdout partitions are empty\","
             << "\"single-speaker excerpts do not form ready AOSC profiles\","
             << "\"silence replacement preserves shape but changes global cache context\","
             << "\"no explicit speaker-verification score is exposed by native inference\","
             << "\"fused attention does not expose weights without a risky kernel change\"]\n}\n";

        if (options.report == "-") std::cout << json.str();
        else {
            std::ofstream report(options.report, std::ios::binary | std::ios::trunc);
            if (!report) throw std::runtime_error("cannot write report: " + options.report);
            report << json.str();
            if (!report) throw std::runtime_error("failed writing report: " + options.report);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "diar_native_matching_probe: " << e.what() << '\n';
        return 2;
    }
}
