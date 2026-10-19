// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Research-only internal-representation and unknown-rejection benchmark.

#include "audio_file.h"
#include "diar_enrollment_profile.h"
#include "diar_snapshot.h"
#include "diarizer.h"
#include "runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
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
#include <variant>
#include <vector>

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
        Json out = value();
        whitespace();
        if (pos_ != input_.size()) fail("trailing characters");
        return out;
    }

   private:
    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("manifest JSON at byte " + std::to_string(pos_) + ": " + message);
    }
    void whitespace() {
        while (pos_ < input_.size() &&
               (input_[pos_] == ' ' || input_[pos_] == '\t' || input_[pos_] == '\r' || input_[pos_] == '\n'))
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
    Json value() {
        whitespace();
        if (pos_ == input_.size()) fail("unexpected end of input");
        const char c = input_[pos_];
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') return Json{string()};
        if (c == 't' && input_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return Json{true};
        }
        if (c == 'f' && input_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return Json{false};
        }
        if (c == 'n' && input_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return Json{nullptr};
        }
        return number();
    }
    Json object() {
        take('{');
        Json::Object fields;
        if (take('}')) return Json{std::move(fields)};
        do {
            whitespace();
            if (pos_ == input_.size() || input_[pos_] != '"') fail("expected object key");
            std::string key = string();
            if (!take(':')) fail("expected ':' after object key");
            if (!fields.emplace(std::move(key), value()).second) fail("duplicate object key");
        } while (take(','));
        if (!take('}')) fail("expected '}'");
        return Json{std::move(fields)};
    }
    Json array() {
        take('[');
        Json::Array items;
        if (take(']')) return Json{std::move(items)};
        do items.push_back(value()); while (take(','));
        if (!take(']')) fail("expected ']'");
        return Json{std::move(items)};
    }
    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
        else if (cp <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
    }
    uint32_t hex4() {
        if (input_.size() - pos_ < 4) fail("truncated unicode escape");
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = input_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return value;
    }
    std::string string() {
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
    Json number() {
        const size_t start = pos_;
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
        if (start == pos_) fail("expected value");
        const std::string token = input_.substr(start, pos_ - start);
        char* end = nullptr;
        const double parsed = std::strtod(token.c_str(), &end);
        if (!end || *end || !std::isfinite(parsed)) fail("invalid number");
        return Json{parsed};
    }

    std::string input_;
    size_t pos_ = 0;
};

const Json* field(const Json& object, const std::string& name) {
    const auto& fields = object.object();
    auto it = fields.find(name);
    return it == fields.end() ? nullptr : &it->second;
}

std::string string_field(const Json& object, const std::string& name, const std::string& fallback = {}) {
    const Json* value = field(object, name);
    return value ? value->string() : fallback;
}

double number_field(const Json& object, const std::string& name, double fallback) {
    const Json* value = field(object, name);
    return value ? value->number() : fallback;
}

struct Clip {
    std::string id;
    std::string speaker;
    std::string path;
    std::string recording_id;
    std::string rttm_recording_id;
    std::string session;
    std::string device;
    std::string partition;
    double start_sec = 0.0;
    double end_sec = -1.0;
    bool known = false;
    bool enrollment = false;
    bool has_overlap_label = false;
    bool overlap = false;
};

struct ReferenceTurn {
    std::string recording;
    std::string speaker;
    double start_sec = 0.0;
    double end_sec = 0.0;
};

struct Manifest {
    std::filesystem::path path;
    std::filesystem::path directory;
    std::string model;
    std::filesystem::path reference_rttm;
    std::vector<ReferenceTurn> reference_turns;
    bool preliminary_only = false;
    std::vector<std::string> representations;
    std::vector<Clip> enrollment;
    std::vector<Clip> development;
    std::vector<Clip> calibration;
    std::vector<Clip> holdout;
    std::vector<Clip> preliminary;
    struct ContextProbe {
        std::string target;
        std::vector<std::string> after_known;
        std::vector<std::string> after_unknown;
        std::vector<std::string> after_unknown_long;
        bool reverse_enrollment_order = false;
        bool aosc_only_warm_start = false;
        bool full_snapshot_warm_start = false;
    };
    std::vector<ContextProbe> context_probes;
};

std::vector<ReferenceTurn> load_rttm(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open reference RTTM: " + path.string());
    std::vector<ReferenceTurn> turns;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream row(line);
        std::string type, recording, channel, unused1, unused2, speaker, unused3, unused4;
        double start = 0.0, duration = 0.0;
        if (!(row >> type)) continue;
        if (type != "SPEAKER" || !(row >> recording >> channel >> start >> duration >> unused1 >> unused2 >> speaker))
            throw std::runtime_error("invalid RTTM speaker row at line " + std::to_string(line_number));
        if (start < 0.0 || duration <= 0.0 || speaker.empty())
            throw std::runtime_error("invalid RTTM interval at line " + std::to_string(line_number));
        turns.push_back({recording, speaker, start, start + duration});
    }
    return turns;
}

Clip parse_clip(const Json& value, const std::filesystem::path& base, bool enrollment, const std::string& partition) {
    Clip clip;
    clip.id = string_field(value, "id");
    clip.speaker = string_field(value, "speaker");
    const std::string audio = string_field(value, "audio");
    if (clip.id.empty() || clip.speaker.empty() || audio.empty())
        throw std::runtime_error("manifest clip requires nonempty id, speaker, and audio");
    clip.path = (base / audio).lexically_normal().string();
    clip.recording_id = string_field(value, "recording_id", clip.path);
    clip.rttm_recording_id = string_field(value, "rttm_recording_id", clip.recording_id);
    clip.session = string_field(value, "session");
    clip.device = string_field(value, "device");
    clip.partition = partition;
    clip.start_sec = number_field(value, "start_sec", 0.0);
    clip.end_sec = number_field(value, "end_sec", -1.0);
    clip.enrollment = enrollment;
    clip.known = enrollment;
    if (!enrollment) {
        const Json* known = field(value, "known");
        if (!known) throw std::runtime_error("evaluation clip requires explicit boolean 'known': " + clip.id);
        clip.known = known->boolean();
    }
    if (const Json* overlap = field(value, "overlap")) {
        clip.has_overlap_label = true;
        clip.overlap = overlap->boolean();
    }
    if (clip.start_sec < 0.0 || (clip.end_sec >= 0.0 && clip.end_sec <= clip.start_sec))
        throw std::runtime_error("invalid audio interval for clip " + clip.id);
    return clip;
}

void parse_clip_array(const Json& root, const std::string& key, const std::filesystem::path& base,
                      bool enrollment, const std::string& partition, std::vector<Clip>& out) {
    const Json* array = field(root, key);
    if (!array) return;
    for (const auto& item : array->array()) out.push_back(parse_clip(item, base, enrollment, partition));
}

Manifest load_manifest(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open manifest: " + path);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const Json root = JsonParser(buffer.str()).parse();
    Manifest out;
    out.path = std::filesystem::absolute(path);
    out.directory = out.path.parent_path();
    out.model = string_field(root, "model");
    if (out.model.empty()) throw std::runtime_error("manifest requires model path");
    out.model = (out.directory / out.model).lexically_normal().string();
    const std::string reference_rttm = string_field(root, "reference_rttm");
    if (!reference_rttm.empty()) {
        out.reference_rttm = (out.directory / reference_rttm).lexically_normal();
        out.reference_turns = load_rttm(out.reference_rttm);
    }
    if (const Json* preliminary = field(root, "preliminary_fixture_only"))
        out.preliminary_only = preliminary->boolean();
    if (const Json* reps = field(root, "representations"))
        for (const auto& rep : reps->array()) out.representations.push_back(rep.string());
    parse_clip_array(root, "enrollment", out.directory, true, "enrollment", out.enrollment);
    parse_clip_array(root, "development", out.directory, false, "development", out.development);
    parse_clip_array(root, "calibration", out.directory, false, "calibration", out.calibration);
    parse_clip_array(root, "holdout", out.directory, false, "holdout", out.holdout);
    parse_clip_array(root, "preliminary", out.directory, false, "preliminary", out.preliminary);
    if (const Json* evaluation = field(root, "evaluation")) {
        for (const auto& item : evaluation->array()) {
            const std::string partition = string_field(item, "partition", "development");
            if (partition == "development") out.development.push_back(parse_clip(item, out.directory, false, partition));
            else if (partition == "calibration") out.calibration.push_back(parse_clip(item, out.directory, false, partition));
            else if (partition == "holdout") out.holdout.push_back(parse_clip(item, out.directory, false, partition));
            else if (partition == "preliminary") out.preliminary.push_back(parse_clip(item, out.directory, false, partition));
            else throw std::runtime_error("unknown evaluation partition '" + partition + "'");
        }
    }
    if (const Json* probes = field(root, "context_probes")) {
        for (const auto& item : probes->array()) {
            Manifest::ContextProbe probe;
            probe.target = string_field(item, "target");
            if (probe.target.empty()) throw std::runtime_error("context probe requires target id");
            auto ids = [&](const char* key, std::vector<std::string>& out) {
                if (const Json* array = field(item, key))
                    for (const auto& value : array->array()) out.push_back(value.string());
            };
            ids("after_known", probe.after_known);
            ids("after_unknown", probe.after_unknown);
            ids("after_unknown_long", probe.after_unknown_long);
            if (const Json* enabled = field(item, "reverse_enrollment_order")) probe.reverse_enrollment_order = enabled->boolean();
            if (const Json* enabled = field(item, "aosc_only_warm_start")) probe.aosc_only_warm_start = enabled->boolean();
            if (const Json* enabled = field(item, "full_snapshot_warm_start")) probe.full_snapshot_warm_start = enabled->boolean();
            out.context_probes.push_back(std::move(probe));
        }
    }
    if (out.enrollment.empty()) throw std::runtime_error("manifest requires at least one enrollment clip");
    std::set<std::string> ids;
    for (const auto* collection : {&out.enrollment, &out.development, &out.calibration, &out.holdout, &out.preliminary})
        for (const auto& clip : *collection)
            if (!ids.insert(clip.id).second) throw std::runtime_error("duplicate clip id: " + clip.id);
    return out;
}

void validate_manifest(const Manifest& manifest) {
    std::set<std::string> speakers;
    for (const auto& clip : manifest.enrollment) speakers.insert(clip.speaker);
    std::map<std::string, std::set<std::string>> enrollment_recordings;
    for (const auto& clip : manifest.enrollment)
        enrollment_recordings[clip.speaker].insert(clip.recording_id);
    std::set<std::string> all_enrollment_recordings;
    std::set<std::string> all_enrollment_paths;
    for (const auto& [speaker, recordings] : enrollment_recordings) {
        (void)speaker;
        all_enrollment_recordings.insert(recordings.begin(), recordings.end());
    }
    for (const auto& clip : manifest.enrollment) all_enrollment_paths.insert(clip.path);
    for (const auto* partition : {&manifest.development, &manifest.calibration, &manifest.holdout}) {
        for (const auto& clip : *partition) {
            if (!manifest.preliminary_only && all_enrollment_recordings.count(clip.recording_id))
                throw std::runtime_error("evaluation must use recordings independent from enrollment: " + clip.id);
            if (!manifest.preliminary_only && all_enrollment_paths.count(clip.path))
                throw std::runtime_error(
                    "evaluation audio file must be independent from enrollment: " + clip.id);
            if (clip.known && !speakers.count(clip.speaker))
                throw std::runtime_error("known evaluation identity has no enrollment: " + clip.id);
            if (!clip.known && speakers.count(clip.speaker))
                throw std::runtime_error("unknown evaluation label is also enrolled: " + clip.id);
            if (!manifest.preliminary_only && clip.known) {
                const auto& enrollment = enrollment_recordings.at(clip.speaker);
                if (enrollment.count(clip.recording_id))
                    throw std::runtime_error(
                        "known enrollment and evaluation must use independent recordings: " + clip.id);
            }
        }
    }
    if (!manifest.preliminary_only) {
        std::set<std::string> evaluation_recordings;
        std::set<std::string> evaluation_paths;
        for (const auto* partition : {&manifest.development, &manifest.calibration, &manifest.holdout}) {
            for (const auto& clip : *partition) {
                if (!evaluation_recordings.insert(clip.recording_id).second)
                    throw std::runtime_error(
                        "recording_id must be unique across development, calibration, and holdout: " +
                        clip.recording_id);
                if (!evaluation_paths.insert(clip.path).second)
                    throw std::runtime_error(
                        "audio file must be unique across development, calibration, and holdout: " +
                        clip.id);
            }
        }
    }
}

struct FeatureMatrix {
    int dim = 0;
    std::vector<int64_t> frame_ids;
    std::vector<float> values;
    size_t frames() const { return frame_ids.size(); }
};

using PrototypeSet = std::vector<std::vector<float>>;
using Profiles = std::map<std::string, PrototypeSet>;
double vector_norm(const float* values, int dim);

struct Extraction {
    std::map<std::string, FeatureMatrix> features;
    std::set<int64_t> aosc_selected_frames;
    std::vector<float> probabilities;
    int64_t probability_base = 0;
    int64_t frame_count = 0;
    int num_speakers = 0;
    std::vector<DiarSegment> segments;
    std::vector<uint8_t> snapshot;
    std::vector<uint8_t> aosc_profile;
    std::vector<uint8_t> pre_target_aosc_profile;
    bool aosc_profile_ready = false;
    bool pre_target_aosc_profile_ready = false;
    bool repeat_data_bit_identical = true;
    uint64_t feature_hash = 0;
    double elapsed_ms = 0.0;
    size_t runtime_feature_host_copy_bytes = 0;
    size_t benchmark_retention_copy_bytes = 0;
    size_t max_chunk_feature_bytes = 0;
    size_t max_chunk_runtime_output_bytes = 0;
};

struct DiarizationCounts {
    size_t clips_scored = 0;
    size_t reference_speaker_frames = 0;
    size_t scored_frames = 0;
    size_t missed_speaker_frames = 0;
    size_t false_alarm_speaker_frames = 0;
    size_t speaker_confusion_frames = 0;
    size_t reference_turns = 0;
    size_t turns_with_any_speech = 0;
    size_t turns_with_correct_speaker = 0;
};

void json_number(std::ostream& out, double number);

std::vector<int> maximum_weight_assignment(const std::vector<std::vector<size_t>>& weights) {
    const size_t rows = weights.size();
    const size_t real_columns = rows == 0 ? 0 : weights.front().size();
    const size_t columns = std::max(rows, real_columns);
    if (rows == 0) return {};
    std::vector<int64_t> u(rows + 1), v(columns + 1);
    std::vector<size_t> p(columns + 1), way(columns + 1);
    for (size_t i = 1; i <= rows; ++i) {
        p[0] = i;
        size_t j0 = 0;
        std::vector<int64_t> minv(columns + 1, std::numeric_limits<int64_t>::max() / 4);
        std::vector<uint8_t> used(columns + 1, 0);
        do {
            used[j0] = 1;
            const size_t i0 = p[j0];
            int64_t delta = std::numeric_limits<int64_t>::max() / 4;
            size_t j1 = 0;
            for (size_t j = 1; j <= columns; ++j) {
                if (used[j]) continue;
                const int64_t weight = j <= real_columns ? static_cast<int64_t>(weights[i0 - 1][j - 1]) : 0;
                const int64_t current = -weight - u[i0] - v[j];
                if (current < minv[j]) minv[j] = current, way[j] = j0;
                if (minv[j] < delta) delta = minv[j], j1 = j;
            }
            for (size_t j = 0; j <= columns; ++j) {
                if (used[j]) u[p[j]] += delta, v[j] -= delta;
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const size_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<int> assignment(rows, -1);
    for (size_t j = 1; j <= columns; ++j)
        if (p[j] != 0 && j <= real_columns) assignment[p[j] - 1] = static_cast<int>(j - 1);
    return assignment;
}

DiarizationCounts evaluate_diarization(
    const std::vector<std::pair<Clip, Extraction>>& clips,
    const std::vector<ReferenceTurn>& reference_turns, double frame_seconds, int overlap_filter = -1) {
    DiarizationCounts total;
    for (const auto& [clip, extraction] : clips) {
        if (overlap_filter >= 0 &&
            (!clip.has_overlap_label || clip.overlap != static_cast<bool>(overlap_filter)))
            continue;
        const size_t frames = static_cast<size_t>(std::max<int64_t>(0, extraction.frame_count));
        const int channels = std::max(0, extraction.num_speakers);
        if (frames == 0 || channels == 0) continue;
        const double duration = frames * frame_seconds;
        std::vector<std::string> speakers;
        std::vector<const ReferenceTurn*> turns;
        for (const auto& turn : reference_turns) {
            if (turn.recording != clip.rttm_recording_id || turn.end_sec <= clip.start_sec ||
                turn.start_sec >= clip.start_sec + duration)
                continue;
            turns.push_back(&turn);
            if (std::find(speakers.begin(), speakers.end(), turn.speaker) == speakers.end())
                speakers.push_back(turn.speaker);
        }
        if (turns.empty()) continue;
        std::sort(speakers.begin(), speakers.end());
        std::vector<std::vector<uint8_t>> reference(speakers.size(), std::vector<uint8_t>(frames, 0));
        std::vector<std::vector<uint8_t>> predicted(static_cast<size_t>(channels), std::vector<uint8_t>(frames, 0));
        struct TurnRange { size_t speaker; size_t begin; size_t end; };
        std::vector<TurnRange> turn_ranges;
        for (const auto* turn : turns) {
            const size_t speaker = static_cast<size_t>(std::distance(
                speakers.begin(), std::find(speakers.begin(), speakers.end(), turn->speaker)));
            const double begin_sec = std::max(0.0, turn->start_sec - clip.start_sec);
            const double end_sec = std::min(duration, turn->end_sec - clip.start_sec);
            const size_t begin = std::min(frames, static_cast<size_t>(std::floor(begin_sec / frame_seconds)));
            const size_t end = std::min(frames, static_cast<size_t>(std::ceil(end_sec / frame_seconds)));
            if (begin >= end) continue;
            std::fill(reference[speaker].begin() + static_cast<std::ptrdiff_t>(begin),
                      reference[speaker].begin() + static_cast<std::ptrdiff_t>(end), 1);
            turn_ranges.push_back({speaker, begin, end});
        }
        for (const auto& segment : extraction.segments) {
            if (segment.speaker < 0 || segment.speaker >= channels) continue;
            const double begin_sec = std::max(0.0, segment.t0);
            const double end_sec = std::min(duration, segment.t1);
            if (begin_sec >= end_sec) continue;
            const size_t begin = std::min(frames, static_cast<size_t>(std::floor(begin_sec / frame_seconds)));
            const size_t end = std::min(frames, static_cast<size_t>(std::ceil(end_sec / frame_seconds)));
            std::fill(predicted[static_cast<size_t>(segment.speaker)].begin() + static_cast<std::ptrdiff_t>(begin),
                      predicted[static_cast<size_t>(segment.speaker)].begin() + static_cast<std::ptrdiff_t>(end), 1);
        }
        std::vector<std::vector<size_t>> overlaps(static_cast<size_t>(channels), std::vector<size_t>(speakers.size(), 0));
        for (size_t frame = 0; frame < frames; ++frame)
            for (size_t speaker = 0; speaker < speakers.size(); ++speaker)
                if (reference[speaker][frame])
                    for (int channel = 0; channel < channels; ++channel)
                        if (predicted[static_cast<size_t>(channel)][frame])
                            ++overlaps[static_cast<size_t>(channel)][speaker];
        const auto assignment = maximum_weight_assignment(overlaps);
        size_t reference_frames = 0, misses = 0, false_alarms = 0, confusions = 0;
        for (size_t frame = 0; frame < frames; ++frame) {
            size_t ref_active = 0, pred_active = 0, correct = 0;
            for (size_t speaker = 0; speaker < speakers.size(); ++speaker) ref_active += reference[speaker][frame] != 0;
            for (int channel = 0; channel < channels; ++channel) {
                const bool active = predicted[static_cast<size_t>(channel)][frame] != 0;
                pred_active += active;
                const int speaker = assignment[static_cast<size_t>(channel)];
                if (active && speaker >= 0 && reference[static_cast<size_t>(speaker)][frame]) ++correct;
            }
            reference_frames += ref_active;
            misses += ref_active > pred_active ? ref_active - pred_active : 0;
            false_alarms += pred_active > ref_active ? pred_active - ref_active : 0;
            const size_t paired = std::min(ref_active, pred_active);
            confusions += paired > correct ? paired - correct : 0;
        }
        size_t any_turns = 0, correct_turns = 0;
        for (const auto& turn : turn_ranges) {
            bool any = false, correct_speaker = false;
            for (int channel = 0; channel < channels; ++channel) {
                const auto& activity = predicted[static_cast<size_t>(channel)];
                for (size_t frame = turn.begin; frame < turn.end; ++frame)
                    if (activity[frame]) {
                        any = true;
                        if (assignment[static_cast<size_t>(channel)] == static_cast<int>(turn.speaker))
                            correct_speaker = true;
                    }
            }
            any_turns += any;
            correct_turns += correct_speaker;
        }
        ++total.clips_scored;
        total.scored_frames += frames;
        total.reference_speaker_frames += reference_frames;
        total.missed_speaker_frames += misses;
        total.false_alarm_speaker_frames += false_alarms;
        total.speaker_confusion_frames += confusions;
        total.reference_turns += turn_ranges.size();
        total.turns_with_any_speech += any_turns;
        total.turns_with_correct_speaker += correct_turns;
    }
    return total;
}

void emit_diarization_counts(std::ostream& json, const DiarizationCounts& counts) {
    const size_t errors = counts.missed_speaker_frames + counts.false_alarm_speaker_frames +
                          counts.speaker_confusion_frames;
    json << "{\"clips_scored\":" << counts.clips_scored
         << ",\"scored_frames\":" << counts.scored_frames
         << ",\"reference_speaker_frames\":" << counts.reference_speaker_frames
         << ",\"missed_speaker_frames\":" << counts.missed_speaker_frames
         << ",\"false_alarm_speaker_frames\":" << counts.false_alarm_speaker_frames
         << ",\"speaker_confusion_frames\":" << counts.speaker_confusion_frames
         << ",\"der\":";
    json_number(json, counts.reference_speaker_frames
                          ? errors / static_cast<double>(counts.reference_speaker_frames)
                          : std::numeric_limits<double>::quiet_NaN());
    json << ",\"reference_turns\":" << counts.reference_turns
         << ",\"turns_with_any_speech\":" << counts.turns_with_any_speech
         << ",\"turns_with_correct_speaker\":" << counts.turns_with_correct_speaker << '}';
}

std::vector<float> clip_samples(const Clip& clip, int& sample_rate) {
    auto audio = nemo_speech::audio::load_wav_file(clip.path);
    sample_rate = audio.sample_rate;
    const size_t begin = std::min(
        audio.samples.size(), static_cast<size_t>(std::llround(clip.start_sec * sample_rate)));
    const size_t end = clip.end_sec < 0.0
                           ? audio.samples.size()
                           : std::min(audio.samples.size(), static_cast<size_t>(std::llround(clip.end_sec * sample_rate)));
    if (end <= begin) throw std::runtime_error("clip interval is empty: " + clip.id);
    return std::vector<float>(audio.samples.begin() + static_cast<std::ptrdiff_t>(begin),
                              audio.samples.begin() + static_cast<std::ptrdiff_t>(end));
}

void feed_160ms(DiarizationStream& stream, const std::vector<float>& samples, int sample_rate) {
    const size_t block = std::max<size_t>(1, static_cast<size_t>(sample_rate) * 160 / 1000);
    for (size_t begin = 0; begin < samples.size(); begin += block) {
        const size_t count = std::min(block, samples.size() - begin);
        stream.push(samples.data() + begin, count, sample_rate);
    }
}

Extraction extract_clip(const std::shared_ptr<DiarModel>& model, const Clip& clip, const DiarGeometry& geometry,
                        bool instrumentation, int repeats, bool retain_features = true) {
    int sample_rate = 0;
    const std::vector<float> samples = clip_samples(clip, sample_rate);
    Extraction first;
    double total_ms = 0.0;
    for (int repetition = 0; repetition < repeats; ++repetition) {
        uint64_t feature_hash = 1469598103934665603ULL;
        auto stream = std::make_unique<DiarizationStream>(model, geometry);
        if (instrumentation) {
            stream->set_research_feature_observer(
                [&](int64_t start, const std::vector<SortformerResearchFeatureView>& views) {
                    size_t chunk_runtime_output_bytes = 0;
                    for (const auto& view : views) {
                        for (unsigned char c : view.name) feature_hash = (feature_hash ^ c) * 1099511628211ULL;
                        feature_hash = (feature_hash ^ static_cast<uint64_t>(view.dim)) * 1099511628211ULL;
                        feature_hash = (feature_hash ^ static_cast<uint64_t>(view.frames)) * 1099511628211ULL;
                        for (size_t i = 0; i < static_cast<size_t>(view.frames) * view.dim; ++i) {
                            uint32_t bits = 0;
                            std::memcpy(&bits, view.values + i, sizeof(bits));
                            feature_hash = (feature_hash ^ bits) * 1099511628211ULL;
                        }
                        if (retain_features && repetition == 0) {
                            auto& matrix = first.features[view.name];
                            if (matrix.dim == 0) matrix.dim = view.dim;
                            if (matrix.dim != view.dim) throw std::runtime_error("feature dimension changed during run");
                            for (int f = 0; f < view.frames; ++f) {
                                matrix.frame_ids.push_back(start + f);
                                matrix.values.insert(matrix.values.end(), view.values + static_cast<size_t>(f) * view.dim,
                                                     view.values + static_cast<size_t>(f + 1) * view.dim);
                            }
                        }
                        if (repetition == 0 && view.name != "pre_encoder") {
                            first.runtime_feature_host_copy_bytes += view.extra_host_copy_bytes;
                            chunk_runtime_output_bytes += view.extra_host_copy_bytes;
                        }
                        if (retain_features && repetition == 0 && view.name != "pre_encoder")
                            first.benchmark_retention_copy_bytes +=
                                static_cast<size_t>(view.frames) * view.dim * sizeof(float);
                        if (repetition == 0)
                            first.max_chunk_feature_bytes = std::max(
                                first.max_chunk_feature_bytes, static_cast<size_t>(view.frames) * view.dim * sizeof(float));
                    }
                    if (repetition == 0)
                        first.max_chunk_runtime_output_bytes =
                            std::max(first.max_chunk_runtime_output_bytes, chunk_runtime_output_bytes);
                });
            if (retain_features)
                stream->set_research_aosc_selection_observer([&](const std::vector<int64_t>& source_frames) {
                    for (int64_t frame : source_frames)
                        if (frame >= 0) first.aosc_selected_frames.insert(frame);
                });
        }
        const auto begin = std::chrono::steady_clock::now();
        feed_160ms(*stream, samples, sample_rate);
        stream->finish();
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - begin).count();
        if (instrumentation) {
            if (repetition == 0) first.feature_hash = feature_hash;
            else if (first.feature_hash != feature_hash) first.repeat_data_bit_identical = false;
        }
        if (repetition == 0) {
            first.probabilities = stream->frame_probabilities();
            first.probability_base = stream->frame_probs_start();
            first.frame_count = stream->frame_count();
            first.num_speakers = stream->num_speakers();
            first.segments = stream->segments();
            first.snapshot = stream->export_state();
            try {
                first.aosc_profile = DiarAoscProfileCodec::encode(*stream);
                first.aosc_profile_ready = true;
            } catch (const std::invalid_argument&) {
                first.aosc_profile_ready = false;
            }
        } else {
            if (first.probabilities != stream->frame_probabilities() || first.frame_count != stream->frame_count())
                first.repeat_data_bit_identical = false;
        }
    }
    first.elapsed_ms = total_ms / repeats;
    return first;
}

std::vector<float> resample_clip(const Clip& clip, int target_rate) {
    int source_rate = 0;
    const auto samples = clip_samples(clip, source_rate);
    if (source_rate == target_rate) return samples;
    nemo_speech::audio::AudioResampler resampler(source_rate, target_rate);
    std::vector<float> output;
    resampler.process(samples.data(), samples.size(), &output);
    resampler.finish(&output);
    return output;
}

void push_silence(DiarizationStream& stream, int sample_rate, double seconds = 1.0) {
    std::vector<float> silence(static_cast<size_t>(std::llround(sample_rate * seconds)), 0.0f);
    feed_160ms(stream, silence, sample_rate);
}

size_t push_context_clip(DiarizationStream& stream, const Clip& clip, int model_sample_rate) {
    const auto samples = resample_clip(clip, model_sample_rate);
    feed_160ms(stream, samples, model_sample_rate);
    push_silence(stream, model_sample_rate);
    return samples.size() + static_cast<size_t>(model_sample_rate);
}

Extraction extract_target_after_context(const std::shared_ptr<DiarModel>& model, const Clip& target,
                                        const DiarGeometry& geometry, const std::vector<Clip>& contexts,
                                        const std::vector<uint8_t>* aosc_profile = nullptr,
                                        const std::vector<uint8_t>* full_snapshot = nullptr,
                                        size_t snapshot_prefix_samples = 0) {
    auto stream = std::make_unique<DiarizationStream>(model, geometry);
    if (aosc_profile && !aosc_profile->empty())
        DiarAoscProfileCodec::decode(*stream, aosc_profile->data(), aosc_profile->size());
    if (full_snapshot && !full_snapshot->empty())
        stream->import_state(full_snapshot->data(), full_snapshot->size());
    const int64_t snapshot_native_frames = stream->frame_count();
    size_t context_samples = 0;
    for (const auto& context : contexts)
        context_samples += push_context_clip(*stream, context, model->cfg().sample_rate);
    const int sample_rate = model->cfg().sample_rate;
    const size_t encoder_hop_samples = static_cast<size_t>(std::llround(
        sample_rate * model->cfg().window_stride * model->cfg().encoder.subsampling_factor));
    if (encoder_hop_samples == 0) throw std::runtime_error("invalid encoder hop for context probe");
    const size_t prefix_samples = snapshot_prefix_samples + context_samples;
    const size_t alignment_padding = (encoder_hop_samples - prefix_samples % encoder_hop_samples) %
                                     encoder_hop_samples;
    if (alignment_padding) {
        push_silence(*stream, sample_rate, alignment_padding / static_cast<double>(sample_rate));
        context_samples += alignment_padding;
    }
    Extraction out;
    try {
        out.pre_target_aosc_profile = DiarAoscProfileCodec::encode(*stream);
        out.pre_target_aosc_profile_ready = true;
    } catch (const std::invalid_argument&) {
        out.pre_target_aosc_profile_ready = false;
    }
    const int64_t target_feature_start =
        snapshot_prefix_samples
            ? static_cast<int64_t>((snapshot_prefix_samples + context_samples) / encoder_hop_samples)
            : snapshot_native_frames / std::max(1, model->cfg().upsample_factor) +
                  static_cast<int64_t>(context_samples / encoder_hop_samples);
    const auto samples = resample_clip(target, sample_rate);
    const int64_t target_feature_end =
        target_feature_start + static_cast<int64_t>((samples.size() + encoder_hop_samples - 1) /
                                                    encoder_hop_samples);

    stream->set_research_feature_observer(
        [&](int64_t start, const std::vector<SortformerResearchFeatureView>& views) {
            for (const auto& view : views) {
                auto& matrix = out.features[view.name];
                if (matrix.dim == 0) matrix.dim = view.dim;
                if (matrix.dim != view.dim) throw std::runtime_error("context probe feature dimension mismatch");
                for (int f = 0; f < view.frames; ++f) {
                    const int64_t frame = start + f;
                    if (frame < target_feature_start || frame >= target_feature_end) continue;
                    matrix.frame_ids.push_back(frame);
                    matrix.values.insert(matrix.values.end(), view.values + static_cast<size_t>(f) * view.dim,
                                         view.values + static_cast<size_t>(f + 1) * view.dim);
                }
                if (view.name != "pre_encoder")
                    out.runtime_feature_host_copy_bytes += view.extra_host_copy_bytes;
            }
        });
    stream->set_research_aosc_selection_observer([&](const std::vector<int64_t>& source_frames) {
        for (int64_t frame : source_frames)
            if (frame >= 0) out.aosc_selected_frames.insert(frame);
    });
    const auto begin = std::chrono::steady_clock::now();
    feed_160ms(*stream, samples, sample_rate);
    stream->finish();
    out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    out.probabilities = stream->frame_probabilities();
    out.probability_base = stream->frame_probs_start();
    out.frame_count = stream->frame_count();
    out.num_speakers = stream->num_speakers();
    out.segments = stream->segments();
    try {
        out.aosc_profile = DiarAoscProfileCodec::encode(*stream);
        out.aosc_profile_ready = true;
    } catch (const std::invalid_argument&) {
        out.aosc_profile_ready = false;
    }
    return out;
}

struct WarmArtifacts {
    std::vector<uint8_t> aosc_profile;
    std::vector<uint8_t> full_snapshot;
    bool profile_ready = false;
    std::map<std::string, std::map<std::string, FeatureMatrix>> selected_by_speaker;
    size_t selected_source_frames = 0;
    size_t total_input_samples = 0;
};

WarmArtifacts build_warm_artifacts(const std::shared_ptr<DiarModel>& model, const DiarGeometry& geometry,
                                   const std::vector<Clip>& enrollment) {
    WarmArtifacts out;
    const int sample_rate = model->cfg().sample_rate;
    DiarizationStream stream(model, geometry);
    std::string current_speaker;
    std::map<std::string, FeatureMatrix> all_features;
    std::map<int64_t, std::string> source_owner;
    std::set<int64_t> selected;
    stream.set_research_feature_observer(
        [&](int64_t start, const std::vector<SortformerResearchFeatureView>& views) {
            for (const auto& view : views) {
                auto& matrix = all_features[view.name];
                if (matrix.dim == 0) matrix.dim = view.dim;
                for (int f = 0; f < view.frames; ++f) {
                    const int64_t frame = start + f;
                    matrix.frame_ids.push_back(frame);
                    matrix.values.insert(matrix.values.end(), view.values + static_cast<size_t>(f) * view.dim,
                                         view.values + static_cast<size_t>(f + 1) * view.dim);
                    if (!current_speaker.empty()) source_owner[frame] = current_speaker;
                }
            }
        });
    stream.set_research_aosc_selection_observer([&](const std::vector<int64_t>& source_frames) {
        for (int64_t frame : source_frames)
            if (frame >= 0) selected.insert(frame);
    });
    for (const auto& clip : enrollment) {
        current_speaker = clip.speaker;
        const auto samples = resample_clip(clip, sample_rate);
        feed_160ms(stream, samples, sample_rate);
        push_silence(stream, sample_rate);
        out.total_input_samples += samples.size() + static_cast<size_t>(sample_rate);
    }
    try {
        out.aosc_profile = DiarAoscProfileCodec::encode(stream);
        out.profile_ready = true;
    } catch (const std::invalid_argument&) {
        out.profile_ready = false;
    }
    out.full_snapshot = stream.export_state();
    out.selected_source_frames = selected.size();
    for (int64_t frame : selected) {
        const auto owner = source_owner.find(frame);
        if (owner == source_owner.end()) continue;
        auto& per_representation = out.selected_by_speaker[owner->second];
        for (const auto& [name, source] : all_features) {
            const auto pos = std::lower_bound(source.frame_ids.begin(), source.frame_ids.end(), frame);
            if (pos == source.frame_ids.end() || *pos != frame) continue;
            const size_t index = static_cast<size_t>(pos - source.frame_ids.begin());
            auto& destination = per_representation[name];
            if (destination.dim == 0) destination.dim = source.dim;
            destination.frame_ids.push_back(frame);
            destination.values.insert(destination.values.end(),
                source.values.begin() + static_cast<std::ptrdiff_t>(index * source.dim),
                source.values.begin() + static_cast<std::ptrdiff_t>((index + 1) * source.dim));
        }
    }
    return out;
}

Profiles make_aosc_selected_profiles(const WarmArtifacts& artifacts, const std::string& representation) {
    Profiles profiles;
    for (const auto& [speaker, features] : artifacts.selected_by_speaker) {
        const auto it = features.find(representation);
        if (it == features.end() || it->second.frames() == 0) continue;
        std::vector<float> mean(static_cast<size_t>(it->second.dim), 0.0f);
        for (size_t row = 0; row < it->second.frames(); ++row)
            for (int d = 0; d < it->second.dim; ++d)
                mean[static_cast<size_t>(d)] += it->second.values[row * it->second.dim + d];
        for (float& value : mean) value = static_cast<float>(value / it->second.frames());
        const double norm = vector_norm(mean.data(), static_cast<int>(mean.size()));
        if (norm > 1e-12) for (float& value : mean) value = static_cast<float>(value / norm);
        profiles.emplace(speaker + std::string(1, '\x1f') + "aosc_selected", PrototypeSet{std::move(mean)});
    }
    return profiles;
}

std::pair<double, size_t> context_cosine(const FeatureMatrix& baseline, const FeatureMatrix& context) {
    const size_t count = std::min(baseline.frames(), context.frames());
    double sum = 0.0;
    size_t valid = 0;
    for (size_t row = 0; row < count; ++row) {
        const float* a = baseline.values.data() + row * baseline.dim;
        const float* b = context.values.data() + row * context.dim;
        if (baseline.dim != context.dim) continue;
        const double na = vector_norm(a, baseline.dim), nb = vector_norm(b, context.dim);
        if (na <= 1e-12 || nb <= 1e-12) continue;
        double dot = 0.0;
        for (int d = 0; d < baseline.dim; ++d) dot += static_cast<double>(a[d]) * b[d];
        sum += dot / (na * nb);
        ++valid;
    }
    return {valid ? sum / valid : NAN, valid};
}

double vector_norm(const float* values, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) sum += static_cast<double>(values[i]) * values[i];
    return std::sqrt(sum);
}

std::vector<float> normalized_mean(const FeatureMatrix& matrix, const std::vector<size_t>& rows = {},
                                   const std::vector<double>* weights = nullptr) {
    std::vector<float> result(static_cast<size_t>(matrix.dim), 0.0f);
    const size_t count = rows.empty() ? matrix.frames() : rows.size();
    double weight_sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const size_t row = rows.empty() ? i : rows[i];
        const float* values = matrix.values.data() + row * matrix.dim;
        const double norm = vector_norm(values, matrix.dim);
        const double weight = weights ? (*weights)[row] : 1.0;
        if (norm <= 1e-12 || weight <= 0.0) continue;
        weight_sum += weight;
        for (int d = 0; d < matrix.dim; ++d)
            result[static_cast<size_t>(d)] += static_cast<float>(weight * values[d] / norm);
    }
    if (weight_sum > 0.0)
        for (float& x : result) x = static_cast<float>(x / weight_sum);
    const double norm = vector_norm(result.data(), matrix.dim);
    if (norm > 1e-12) for (float& x : result) x = static_cast<float>(x / norm);
    return result;
}

std::vector<double> frame_weights(const FeatureMatrix& matrix, const Extraction& extraction,
                                  int native_subsampling_factor) {
    std::vector<double> weights(matrix.frames(), 0.0);
    std::vector<float> best_probs(matrix.frames(), 0.0f);
    std::vector<float> margins(matrix.frames(), 0.0f);
    for (size_t row = 0; row < matrix.frames(); ++row) {
        const int64_t native_begin = matrix.frame_ids[row] * native_subsampling_factor;
        double sums[2] = {0.0, 0.0};
        for (int u = 0; u < native_subsampling_factor; ++u) {
            const int64_t frame = native_begin + u;
            if (frame < extraction.probability_base || frame >= extraction.frame_count) continue;
            const size_t offset = static_cast<size_t>(frame - extraction.probability_base) * extraction.num_speakers;
            const float* prob = extraction.probabilities.data() + offset;
            int best = 0;
            int second = extraction.num_speakers > 1 ? 1 : 0;
            if (prob[second] > prob[best]) std::swap(best, second);
            for (int s = 2; s < extraction.num_speakers; ++s) {
                if (prob[s] > prob[best]) second = best, best = s;
                else if (prob[s] > prob[second]) second = s;
            }
            sums[0] += prob[best];
            sums[1] += prob[second];
        }
        const double divisor = native_subsampling_factor;
        const double activity = sums[0] / divisor;
        const double second = sums[1] / divisor;
        best_probs[row] = static_cast<float>(activity);
        margins[row] = static_cast<float>(activity > 1e-12 ? std::clamp((activity - second) / activity, 0.0, 1.0) : 0.0);
    }
    for (size_t row = 0; row < matrix.frames(); ++row) {
        double stability = 1.0;
        int neighbors = 0;
        auto neighbor_cosine = [&](size_t other) {
            const float* a = matrix.values.data() + row * matrix.dim;
            const float* b = matrix.values.data() + other * matrix.dim;
            const double na = vector_norm(a, matrix.dim), nb = vector_norm(b, matrix.dim);
            if (na <= 1e-12 || nb <= 1e-12) return 0.0;
            double dot = 0.0;
            for (int d = 0; d < matrix.dim; ++d) dot += static_cast<double>(a[d]) * b[d];
            return std::clamp(dot / (na * nb), -1.0, 1.0);
        };
        if (row > 0) stability += neighbor_cosine(row - 1), ++neighbors;
        if (row + 1 < matrix.frames()) stability += neighbor_cosine(row + 1), ++neighbors;
        if (neighbors > 0) stability /= (neighbors + 1);
        // Activity chooses speech-like frames; margin and local consistency
        // softly downweight overlap and unstable acoustics.
        weights[row] = best_probs[row] * margins[row] * std::max(0.0, stability);
    }
    return weights;
}

std::vector<size_t> selected_aosc_rows(const FeatureMatrix& matrix, const Extraction& extraction) {
    std::vector<size_t> rows;
    for (size_t i = 0; i < matrix.frames(); ++i)
        if (extraction.aosc_selected_frames.count(matrix.frame_ids[i])) rows.push_back(i);
    return rows;
}

struct EnrollmentAccumulator {
    int dim = 0;
    std::vector<double> mean_sum;
    std::vector<double> norm_sum;
    std::vector<double> weighted_sum;
    std::vector<double> prototype_sums;
    std::array<size_t, 3> prototype_counts{{0, 0, 0}};
    std::vector<double> aosc_sum;
    size_t frame_count = 0;
    size_t aosc_count = 0;
    double weight_sum = 0.0;
};

void accumulate_enrollment(EnrollmentAccumulator& acc, const FeatureMatrix& matrix,
                           const Extraction& extraction, int native_subsampling_factor) {
    if (acc.dim == 0) {
        acc.dim = matrix.dim;
        acc.mean_sum.assign(matrix.dim, 0.0);
        acc.norm_sum.assign(matrix.dim, 0.0);
        acc.weighted_sum.assign(matrix.dim, 0.0);
        acc.prototype_sums.assign(static_cast<size_t>(3) * matrix.dim, 0.0);
        acc.aosc_sum.assign(matrix.dim, 0.0);
    }
    if (acc.dim != matrix.dim) throw std::runtime_error("enrollment feature dimensions do not match");
    const auto weights = frame_weights(matrix, extraction, native_subsampling_factor);
    const auto aosc_rows = selected_aosc_rows(matrix, extraction);
    for (size_t row = 0; row < matrix.frames(); ++row) {
        const float* values = matrix.values.data() + row * matrix.dim;
        const double norm = vector_norm(values, matrix.dim);
        const size_t bin = std::min<size_t>(2, row * 3 / std::max<size_t>(1, matrix.frames()));
        ++acc.frame_count;
        ++acc.prototype_counts[bin];
        acc.weight_sum += weights[row];
        for (int d = 0; d < matrix.dim; ++d) {
            const size_t i = static_cast<size_t>(d);
            acc.mean_sum[i] += values[d];
            acc.prototype_sums[bin * matrix.dim + i] += values[d];
            if (norm > 1e-12) acc.norm_sum[i] += values[d] / norm;
            acc.weighted_sum[i] += weights[row] * values[d];
        }
    }
    for (size_t row : aosc_rows) {
        const float* values = matrix.values.data() + row * matrix.dim;
        for (int d = 0; d < matrix.dim; ++d) acc.aosc_sum[static_cast<size_t>(d)] += values[d];
        ++acc.aosc_count;
    }
}

Profiles make_profiles(const std::vector<std::pair<Clip, Extraction>>& enrollment,
                      const std::string& representation, int native_subsampling_factor) {
    std::map<std::string, EnrollmentAccumulator> accs;
    for (const auto& [clip, extraction] : enrollment) {
        const auto it = extraction.features.find(representation);
        if (it != extraction.features.end() && it->second.frames() > 0)
            accumulate_enrollment(accs[clip.speaker], it->second, extraction, native_subsampling_factor);
    }
    Profiles profiles;
    for (auto& [speaker, acc] : accs) {
        if (acc.dim == 0 || acc.frame_count == 0) continue;
        PrototypeSet mean(1, std::vector<float>(static_cast<size_t>(acc.dim)));
        PrototypeSet norm_mean(1, std::vector<float>(static_cast<size_t>(acc.dim)));
        PrototypeSet weighted(1, std::vector<float>(static_cast<size_t>(acc.dim)));
        PrototypeSet prototype;
        PrototypeSet aosc;
        for (int d = 0; d < acc.dim; ++d) {
            const size_t i = static_cast<size_t>(d);
            mean[0][i] = static_cast<float>(acc.mean_sum[i] / acc.frame_count);
            norm_mean[0][i] = static_cast<float>(acc.norm_sum[i] / acc.frame_count);
            weighted[0][i] = static_cast<float>(acc.weighted_sum[i] / std::max(acc.weight_sum, 1e-12));
        }
        auto normalize = [](std::vector<float>& v) {
            const double norm = vector_norm(v.data(), static_cast<int>(v.size()));
            if (norm > 1e-12) for (float& value : v) value = static_cast<float>(value / norm);
        };
        normalize(norm_mean[0]);
        normalize(weighted[0]);
        for (size_t bin = 0; bin < 3; ++bin) {
            if (!acc.prototype_counts[bin]) continue;
            std::vector<float> v(static_cast<size_t>(acc.dim));
            for (int d = 0; d < acc.dim; ++d)
                v[static_cast<size_t>(d)] = static_cast<float>(acc.prototype_sums[bin * acc.dim + d] /
                                                                 acc.prototype_counts[bin]);
            normalize(v);
            prototype.push_back(std::move(v));
        }
        if (acc.aosc_count > 0) {
            std::vector<float> v(static_cast<size_t>(acc.dim));
            for (int d = 0; d < acc.dim; ++d)
                v[static_cast<size_t>(d)] = static_cast<float>(acc.aosc_sum[static_cast<size_t>(d)] / acc.aosc_count);
            normalize(v);
            aosc.push_back(std::move(v));
        }
        const std::string separator(1, '\x1f');
        profiles.emplace(speaker + separator + "mean", std::move(mean));
        profiles.emplace(speaker + separator + "l2_normalized_mean", std::move(norm_mean));
        profiles.emplace(speaker + separator + "activity_quality_weighted", std::move(weighted));
        profiles.emplace(speaker + separator + "multiple_prototypes", std::move(prototype));
        profiles.emplace(speaker + separator + "aosc_selected", std::move(aosc));
    }
    return profiles;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return std::numeric_limits<double>::quiet_NaN();
    const double na = vector_norm(a.data(), static_cast<int>(a.size()));
    const double nb = vector_norm(b.data(), static_cast<int>(b.size()));
    if (na <= 1e-12 || nb <= 1e-12) return std::numeric_limits<double>::quiet_NaN();
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) dot += static_cast<double>(a[i]) * b[i];
    return dot / (na * nb);
}

double aggregate_similarity(const PrototypeSet& a, const PrototypeSet& b, const std::string& aggregation) {
    std::vector<double> similarities;
    for (const auto& pa : a)
        for (const auto& pb : b) {
            const double sim = cosine(pa, pb);
            if (std::isfinite(sim)) similarities.push_back(sim);
        }
    if (similarities.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(similarities.begin(), similarities.end(), std::greater<double>());
    if (aggregation == "max_similarity") return similarities.front();
    const size_t k = aggregation == "top_k_similarity" ? std::min<size_t>(2, similarities.size()) : similarities.size();
    return std::accumulate(similarities.begin(), similarities.begin() + static_cast<std::ptrdiff_t>(k), 0.0) / k;
}

PrototypeSet make_evaluation_prototypes(const FeatureMatrix& matrix, const Extraction& extraction,
                                        const std::string& pooling, int native_subsampling_factor) {
    if (matrix.frames() == 0) return {};
    if (pooling == "mean") {
        std::vector<float> v(static_cast<size_t>(matrix.dim), 0.0f);
        for (size_t row = 0; row < matrix.frames(); ++row)
            for (int d = 0; d < matrix.dim; ++d) v[static_cast<size_t>(d)] += matrix.values[row * matrix.dim + d];
        for (float& value : v) value = static_cast<float>(value / matrix.frames());
        return {std::move(v)};
    }
    if (pooling == "l2_normalized_mean") return {normalized_mean(matrix)};
    if (pooling == "activity_quality_weighted") {
        const auto weights = frame_weights(matrix, extraction, native_subsampling_factor);
        return {normalized_mean(matrix, {}, &weights)};
    }
    if (pooling == "aosc_selected") {
        // Enrollment prototypes come from AOSC's compressed cache rows.
        // A short independent query often cannot fill a cache, so score it
        // with the same clean, high-confidence speech selector as method C.
        const auto weights = frame_weights(matrix, extraction, native_subsampling_factor);
        return {normalized_mean(matrix, {}, &weights)};
    }
    if (pooling == "multiple_prototypes") {
        PrototypeSet out;
        for (size_t bin = 0; bin < 3; ++bin) {
            const size_t begin = matrix.frames() * bin / 3;
            const size_t end = matrix.frames() * (bin + 1) / 3;
            if (begin >= end) continue;
            std::vector<size_t> rows(end - begin);
            std::iota(rows.begin(), rows.end(), begin);
            out.push_back(normalized_mean(matrix, rows));
        }
        return out;
    }
    throw std::runtime_error("unknown pooling method: " + pooling);
}

struct Attempt {
    std::string id;
    std::string speaker;
    std::string partition;
    bool known = false;
    bool has_overlap_label = false;
    bool overlap = false;
    double duration_sec = 0.0;
    std::map<std::string, double> identity_scores;
    std::string top1;
    double top1_score = std::numeric_limits<double>::quiet_NaN();
    double top2_score = std::numeric_limits<double>::quiet_NaN();
    double target_score = std::numeric_limits<double>::quiet_NaN();
};

std::vector<Attempt> score_partition(const std::vector<std::pair<Clip, Extraction>>& clips,
                                     const std::string& representation, const std::string& pooling,
                                     const std::string& aggregation, const Profiles& profile_map,
                                     int native_subsampling_factor, double sec_per_coarse_frame) {
    std::map<std::string, PrototypeSet> profiles;
    for (const auto& [key, value] : profile_map) {
        const size_t sep = key.find('\x1f');
        if (sep != std::string::npos && key.substr(sep + 1) == pooling)
            profiles.emplace(key.substr(0, sep), value);
    }
    std::vector<Attempt> out;
    for (const auto& [clip, extraction] : clips) {
        const auto feature = extraction.features.find(representation);
        if (feature == extraction.features.end()) continue;
        const PrototypeSet query = make_evaluation_prototypes(feature->second, extraction, pooling,
                                                              native_subsampling_factor);
        if (query.empty()) continue;
        Attempt attempt;
        attempt.id = clip.id;
        attempt.speaker = clip.speaker;
        attempt.partition = clip.partition;
        attempt.known = clip.known;
        attempt.has_overlap_label = clip.has_overlap_label;
        attempt.overlap = clip.overlap;
        attempt.duration_sec = extraction.frame_count * sec_per_coarse_frame;
        std::vector<std::pair<std::string, double>> ranked;
        for (const auto& [speaker, profile] : profiles) {
            if (profile.empty()) continue;
            const double score = aggregate_similarity(query, profile, aggregation);
            if (!std::isfinite(score)) continue;
            attempt.identity_scores[speaker] = score;
            ranked.emplace_back(speaker, score);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            return a.second == b.second ? a.first < b.first : a.second > b.second;
        });
        if (!ranked.empty()) {
            attempt.top1 = ranked[0].first;
            attempt.top1_score = ranked[0].second;
            if (ranked.size() > 1) attempt.top2_score = ranked[1].second;
        }
        auto target = attempt.identity_scores.find(clip.speaker);
        if (target != attempt.identity_scores.end()) attempt.target_score = target->second;
        out.push_back(std::move(attempt));
    }
    return out;
}

double calibrated_threshold(const std::vector<Attempt>& attempts, size_t& known_n, size_t& unknown_n) {
    known_n = unknown_n = 0;
    std::vector<double> known_scores, unknown_scores, candidates;
    for (const auto& attempt : attempts) {
        if (!std::isfinite(attempt.top1_score)) continue;
        (attempt.known ? known_scores : unknown_scores).push_back(attempt.top1_score);
    }
    known_n = known_scores.size();
    unknown_n = unknown_scores.size();
    if (known_scores.empty() || unknown_scores.empty()) return std::numeric_limits<double>::quiet_NaN();
    candidates = known_scores;
    candidates.insert(candidates.end(), unknown_scores.begin(), unknown_scores.end());
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    std::vector<double> thresholds;
    thresholds.push_back(std::nextafter(candidates.front(), -std::numeric_limits<double>::infinity()));
    for (size_t i = 1; i < candidates.size(); ++i) thresholds.push_back((candidates[i - 1] + candidates[i]) * 0.5);
    thresholds.push_back(std::nextafter(candidates.back(), std::numeric_limits<double>::infinity()));
    double best = std::numeric_limits<double>::infinity();
    double threshold = std::numeric_limits<double>::quiet_NaN();
    for (double candidate : thresholds) {
        size_t false_reject = 0, false_accept = 0;
        for (double score : known_scores) if (score < candidate) ++false_reject;
        for (double score : unknown_scores) if (score >= candidate) ++false_accept;
        const double balanced_error = 0.5 * (false_reject / static_cast<double>(known_n) +
                                              false_accept / static_cast<double>(unknown_n));
        if (balanced_error < best || (balanced_error == best && candidate > threshold)) {
            best = balanced_error;
            threshold = candidate;
        }
    }
    return threshold;
}

struct MetricCounts {
    size_t known_attempts = 0, known_top1_correct = 0, known_confusions = 0;
    size_t known_false_rejections = 0, unknown_attempts = 0, unknown_false_accepts = 0;
    double margin_sum = 0.0;
    size_t margin_count = 0;
    double known_best_sum = 0.0, known_target_sum = 0.0, unknown_best_sum = 0.0;
    size_t known_target_count = 0;
    std::map<std::string, size_t> confusions;
};

MetricCounts evaluate_attempts(const std::vector<Attempt>& attempts, double threshold) {
    MetricCounts counts;
    for (const auto& attempt : attempts) {
        if (!std::isfinite(attempt.top1_score)) continue;
        if (std::isfinite(attempt.top2_score)) counts.margin_sum += attempt.top1_score - attempt.top2_score, ++counts.margin_count;
        if (attempt.known) {
            ++counts.known_attempts;
            counts.known_best_sum += attempt.top1_score;
            if (std::isfinite(attempt.target_score)) counts.known_target_sum += attempt.target_score, ++counts.known_target_count;
            if (attempt.top1 == attempt.speaker) ++counts.known_top1_correct;
            else {
                ++counts.known_confusions;
                counts.confusions[attempt.speaker + "->" + attempt.top1]++;
            }
            if (std::isfinite(threshold) && attempt.top1_score < threshold) ++counts.known_false_rejections;
        } else {
            ++counts.unknown_attempts;
            counts.unknown_best_sum += attempt.top1_score;
            if (std::isfinite(threshold) && attempt.top1_score >= threshold) ++counts.unknown_false_accepts;
        }
    }
    return counts;
}

std::string json_quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
        else out << static_cast<char>(c);
    }
    out << '"';
    return out.str();
}

void json_number(std::ostream& out, double number) {
    if (std::isfinite(number)) out << std::setprecision(9) << number;
    else out << "null";
}

bool same_segments(const std::vector<DiarSegment>& a, const std::vector<DiarSegment>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].speaker != b[i].speaker || a[i].t0 != b[i].t0 || a[i].t1 != b[i].t1) return false;
    return true;
}

std::pair<size_t, size_t> compare_regression_groups(
    const std::vector<std::pair<Clip, Extraction>>& instrumented,
    const std::vector<std::pair<Clip, Extraction>>& baseline,
    size_t& probability_values, size_t& channel_assignment_values,
    size_t& segment_cases, size_t& aosc_profile_cases, size_t& snapshot_cases) {
    size_t cases = 0, mismatches = 0;
    if (instrumented.size() != baseline.size()) return {0, 1};
    for (size_t i = 0; i < instrumented.size(); ++i) {
        const auto& a = instrumented[i].second;
        const auto& b = baseline[i].second;
        ++cases;
        const bool same_probs = a.probabilities == b.probabilities;
        const bool same_frames = a.frame_count == b.frame_count && a.probability_base == b.probability_base;
        const bool same_seg = same_segments(a.segments, b.segments);
        const bool same_snapshot = a.snapshot == b.snapshot;
        const bool same_profile = a.aosc_profile_ready == b.aosc_profile_ready &&
                                  (!a.aosc_profile_ready || a.aosc_profile == b.aosc_profile);
        probability_values += a.probabilities.size();
        if (same_probs && same_frames) channel_assignment_values += a.probabilities.size() / std::max(1, a.num_speakers);
        if (same_seg) ++segment_cases;
        if (same_profile) ++aosc_profile_cases;
        if (same_snapshot) ++snapshot_cases;
        if (!same_probs || !same_frames || !same_seg || !same_snapshot || !same_profile) ++mismatches;
    }
    return {cases, mismatches};
}

struct RocCounts {
    size_t positive = 0;
    size_t negative = 0;
    double eer = std::numeric_limits<double>::quiet_NaN();
};

RocCounts roc_eer(const std::vector<Attempt>& attempts) {
    std::vector<std::pair<double, bool>> pairs;
    for (const auto& attempt : attempts) {
        if (attempt.known) {
            for (const auto& [identity, score] : attempt.identity_scores)
                pairs.emplace_back(score, identity == attempt.speaker);
        } else {
            for (const auto& [identity, score] : attempt.identity_scores) {
                (void)identity;
                pairs.emplace_back(score, false);
            }
        }
    }
    RocCounts out;
    for (const auto& item : pairs) item.second ? ++out.positive : ++out.negative;
    if (out.positive == 0 || out.negative == 0) return out;
    std::vector<double> thresholds;
    thresholds.reserve(pairs.size() + 1);
    for (const auto& item : pairs) thresholds.push_back(item.first);
    std::sort(thresholds.begin(), thresholds.end());
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
    double gap = std::numeric_limits<double>::infinity();
    for (double threshold : thresholds) {
        size_t false_negative = 0, false_positive = 0;
        for (const auto& [score, positive] : pairs) {
            if (positive && score < threshold) ++false_negative;
            if (!positive && score >= threshold) ++false_positive;
        }
        const double fnr = false_negative / static_cast<double>(out.positive);
        const double fpr = false_positive / static_cast<double>(out.negative);
        if (std::fabs(fnr - fpr) < gap) {
            gap = std::fabs(fnr - fpr);
            out.eer = (fnr + fpr) * 0.5;
        }
    }
    return out;
}

struct UnknownPairSummary {
    size_t same_pairs = 0;
    size_t different_pairs = 0;
    double same_sum = 0.0;
    double different_sum = 0.0;
};

UnknownPairSummary unknown_pairs(const std::vector<std::pair<Clip, Extraction>>& clips,
                                const std::string& representation, const std::string& pooling,
                                const std::string& aggregation, int native_subsampling_factor) {
    struct Item { std::string speaker; PrototypeSet prototype; };
    std::vector<Item> unknown;
    for (const auto& [clip, extraction] : clips) {
        if (clip.known) continue;
        const auto feature = extraction.features.find(representation);
        if (feature == extraction.features.end()) continue;
        auto proto = make_evaluation_prototypes(feature->second, extraction, pooling, native_subsampling_factor);
        if (!proto.empty()) unknown.push_back({clip.speaker, std::move(proto)});
    }
    UnknownPairSummary out;
    for (size_t i = 0; i < unknown.size(); ++i)
        for (size_t j = i + 1; j < unknown.size(); ++j) {
            const double score = aggregate_similarity(unknown[i].prototype, unknown[j].prototype, aggregation);
            if (!std::isfinite(score)) continue;
            if (unknown[i].speaker == unknown[j].speaker) out.same_sum += score, ++out.same_pairs;
            else out.different_sum += score, ++out.different_pairs;
        }
    return out;
}

std::vector<std::string> available_representations(const Manifest& manifest) {
    return manifest.representations.empty()
        ? std::vector<std::string>{"pre_encoder", "encoder_output", "encoder_proj", "encoder.layer.0",
                                   "encoder.layer.7", "encoder.layer.15", "encoder.layer.23", "encoder.layer.30"}
        : manifest.representations;
}

std::vector<std::string> pooling_methods() {
    return {"mean", "l2_normalized_mean", "activity_quality_weighted", "multiple_prototypes", "aosc_selected"};
}

std::vector<std::string> similarity_methods() {
    return {"mean_similarity", "max_similarity", "top_k_similarity"};
}

SortformerResearchConfig make_research_config(const std::vector<std::string>& requested) {
    SortformerResearchConfig out;
    const std::vector<std::string> names = requested.empty()
        ? std::vector<std::string>{"pre_encoder", "encoder_output", "encoder_proj", "encoder.layer.0",
                                   "encoder.layer.7", "encoder.layer.15", "encoder.layer.23", "encoder.layer.30"}
        : requested;
    for (const auto& name : names) {
        if (name == "pre_encoder") out.pre_encoder = true;
        else if (name == "encoder_output") out.encoder_output = true;
        else if (name == "encoder_proj") out.encoder_proj = true;
        else if (name.rfind("encoder.layer.", 0) == 0) {
            const std::string suffix = name.substr(std::string("encoder.layer.").size());
            size_t used = 0;
            const int layer = std::stoi(suffix, &used);
            if (used != suffix.size() || layer < 0) throw std::runtime_error("invalid representation name: " + name);
            out.transformer_layers.push_back(layer);
        } else {
            throw std::runtime_error("unsupported representation name: " + name);
        }
    }
    std::sort(out.transformer_layers.begin(), out.transformer_layers.end());
    if (std::adjacent_find(out.transformer_layers.begin(), out.transformer_layers.end()) != out.transformer_layers.end())
        throw std::runtime_error("duplicate transformer representation requested");
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " MANIFEST.json [--device cpu|vulkan] [--gpu-device N] [--repeat N] [--report FILE]\n";
        return 2;
    }
    try {
        int repeats = 1;
        std::string report_path = "-";
        std::string device = "cpu";
        int gpu_device = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--repeat" && i + 1 < argc) repeats = std::stoi(argv[++i]);
            else if (arg == "--report" && i + 1 < argc) report_path = argv[++i];
            else if (arg == "--device" && i + 1 < argc) device = argv[++i];
            else if (arg == "--gpu-device" && i + 1 < argc) gpu_device = std::stoi(argv[++i]);
            else throw std::runtime_error("unknown or incomplete option: " + arg);
        }
        if (repeats < 1 || repeats > 20) throw std::runtime_error("--repeat must be in [1,20]");
        if (device != "cpu" && device != "vulkan") throw std::runtime_error("--device must be cpu or vulkan");
        if (gpu_device < 0 || gpu_device > 1024) throw std::runtime_error("--gpu-device must be in [0,1024]");
        if (device == "cpu" && gpu_device != 0) throw std::runtime_error("--gpu-device requires --device vulkan");
        const Manifest manifest = load_manifest(argv[1]);
        validate_manifest(manifest);
        const auto research = make_research_config(manifest.representations);
        if (!std::filesystem::exists(manifest.model)) throw std::runtime_error("model not found: " + manifest.model);

        ggml_runtime::Params params;
        params.use_gpu = device == "vulkan";
        params.gpu_device_idx = gpu_device;
        params.gpu_backend_name = params.use_gpu ? "Vulkan" : "";
        params.n_threads = 4;
        params.pe_bin_path = const_cast<char*>("");
        ggml_runtime::BackendManager backend(params);
        auto model = std::make_shared<DiarModel>(backend, manifest.model, BatchingConfig{}, research);
        const DiarGeometry geometry = DiarGeometry::v3_streaming();

        std::ostringstream json;
        json << std::setprecision(9);
        json << "{\n  \"schema\":\"nemo-speech.diar-representation-probe.v1\",\n";
        json << "  \"method_version\":\"pool-score-v2\",\n";
        json << "  \"preliminary_fixture_only\":" << (manifest.preliminary_only ? "true" : "false") << ",\n";
        json << "  \"model\":{\"path\":" << json_quote(manifest.model) << ",\"sha256\":\"";
        const auto fingerprint = DiarStreamSnapshotCodec::fingerprint_file(manifest.model);
        for (uint8_t byte : fingerprint) json << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
        json << std::dec << std::setfill(' ') << "\",\"sample_rate\":" << model->cfg().sample_rate
             << ",\"n_layers\":" << model->cfg().encoder.n_layers << ",\"encoder_dim\":" << model->cfg().encoder.d_model
             << ",\"encoder_proj_dim\":" << model->cfg().transformer.hidden_size << ",\"speaker_channels\":"
             << model->cfg().num_speakers << "},\n";
        json << "  \"backend\":{\"requested\":" << json_quote(device)
             << ",\"name\":" << json_quote(params.use_gpu ? backend.selected_gpu_backend_name() : "CPU")
             << ",\"device_name\":";
        if (params.use_gpu) json << json_quote(backend.selected_gpu_device_name());
        else json << "null";
        json << ",\"device_index\":";
        if (params.use_gpu) json << gpu_device;
        else json << "null";
        json << ",\"threads\":" << params.n_threads << ",\"stream_push_ms\":160},\n";
        json << "  \"representations\":[";
        bool first_rep = true;
        for (const auto& rep : available_representations(manifest)) {
            if (!first_rep) json << ',';
            first_rep = false;
            json << '"' << rep << '"';
        }
        json << "],\n  \"dataset\":{\"enrollment_clips\":" << manifest.enrollment.size()
             << ",\"development_clips\":" << manifest.development.size()
             << ",\"calibration_clips\":" << manifest.calibration.size()
             << ",\"holdout_clips\":" << manifest.holdout.size()
             << ",\"preliminary_clips\":" << manifest.preliminary.size()
             << ",\"unique_known_enrollment_speakers\":";
        std::set<std::string> enrolled;
        for (const auto& clip : manifest.enrollment) enrolled.insert(clip.speaker);
        json << enrolled.size();
        auto emit_metadata_counts = [&](const std::string& name, const std::vector<Clip>& clips) {
            std::set<std::string> recordings, sessions, devices;
            for (const auto& clip : clips) {
                recordings.insert(clip.recording_id);
                if (!clip.session.empty()) sessions.insert(clip.session);
                if (!clip.device.empty()) devices.insert(clip.device);
            }
            json << ",\"" << name << "_recordings\":" << recordings.size()
                 << ",\"" << name << "_sessions\":" << sessions.size()
                 << ",\"" << name << "_devices\":" << devices.size();
        };
        emit_metadata_counts("enrollment", manifest.enrollment);
        emit_metadata_counts("development", manifest.development);
        emit_metadata_counts("calibration", manifest.calibration);
        emit_metadata_counts("holdout", manifest.holdout);
        json << "},\n";

        std::vector<std::pair<Clip, Extraction>> enrollment_results;
        std::vector<std::pair<Clip, Extraction>> development_results;
        std::vector<std::pair<Clip, Extraction>> calibration_results;
        std::vector<std::pair<Clip, Extraction>> holdout_results;
        std::vector<std::pair<Clip, Extraction>> preliminary_results;
        auto run_group = [&](const std::vector<Clip>& clips, std::vector<std::pair<Clip, Extraction>>& results,
                             bool diagnostic) {
            for (const auto& clip : clips) results.emplace_back(clip, extract_clip(model, clip, geometry, diagnostic, repeats));
        };
        run_group(manifest.enrollment, enrollment_results, true);
        run_group(manifest.development, development_results, true);
        run_group(manifest.calibration, calibration_results, true);
        run_group(manifest.holdout, holdout_results, true);
        run_group(manifest.preliminary, preliminary_results, true);

        // Enrollment selection follows AOSC's own cache decisions across the
        // complete ordered enrollment sequence, not a local activity cutoff.
        WarmArtifacts warm_artifacts = build_warm_artifacts(model, geometry, manifest.enrollment);

        // Re-run the same 160 ms streams with the original two-output graph.
        // Model instances are sequential to keep memory use close to production.
        model.reset();
        auto baseline_model = std::make_shared<DiarModel>(backend, manifest.model, BatchingConfig{},
                                                          SortformerResearchConfig{});
        std::vector<std::pair<Clip, Extraction>> enrollment_baseline, development_baseline,
            calibration_baseline, holdout_baseline, preliminary_baseline;
        auto run_baseline = [&](const std::vector<Clip>& clips,
                                std::vector<std::pair<Clip, Extraction>>& results) {
            for (const auto& clip : clips)
                results.emplace_back(clip, extract_clip(baseline_model, clip, geometry, false, 1, false));
        };
        run_baseline(manifest.enrollment, enrollment_baseline);
        run_baseline(manifest.development, development_baseline);
        run_baseline(manifest.calibration, calibration_baseline);
        run_baseline(manifest.holdout, holdout_baseline);
        run_baseline(manifest.preliminary, preliminary_baseline);

        size_t probability_values = 0, channel_assignments = 0, segment_cases = 0,
               profile_cases = 0, snapshot_cases = 0, regression_cases = 0, regression_mismatches = 0;
        auto compare_group = [&](const auto& diagnostic, const auto& baseline) {
            const auto counts = compare_regression_groups(diagnostic, baseline, probability_values,
                channel_assignments, segment_cases, profile_cases, snapshot_cases);
            regression_cases += counts.first;
            regression_mismatches += counts.second;
        };
        compare_group(enrollment_results, enrollment_baseline);
        compare_group(development_results, development_baseline);
        compare_group(calibration_results, calibration_baseline);
        compare_group(holdout_results, holdout_baseline);
        compare_group(preliminary_results, preliminary_baseline);

        using ResultGroup = std::vector<std::pair<Clip, Extraction>>;
        auto mean_time = [](const std::vector<const ResultGroup*>& groups) {
            double sum = 0.0;
            size_t count = 0;
            for (const auto* group : groups)
                for (const auto& item : *group) sum += item.second.elapsed_ms, ++count;
            return count ? sum / count : 0.0;
        };
        const double baseline_ms = mean_time({&enrollment_baseline, &development_baseline,
                                              &calibration_baseline, &holdout_baseline,
                                              &preliminary_baseline});
        const double research_ms = mean_time({&enrollment_results, &development_results,
                                              &calibration_results, &holdout_results,
                                              &preliminary_results});
        const std::string dataset_status = manifest.preliminary_only || manifest.holdout.empty() ||
                                                   manifest.calibration.empty()
                                               ? "awaiting_independent_recordings"
                                               : "holdout_evaluated";

        json << "  \"extraction\":{\"items\":[";
        bool first_item = true;
        auto emit_extraction = [&](const std::vector<std::pair<Clip, Extraction>>& group) {
            for (const auto& item : group) {
                if (!first_item) json << ',';
                first_item = false;
                size_t feature_frames = 0;
                for (const auto& [name, matrix] : item.second.features) feature_frames = std::max(feature_frames, matrix.frames());
                json << "{\"index\":" << std::distance(group.data(), &item) << ",\"partition\":"
                     << json_quote(item.first.partition)
                     << ",\"duration_sec\":" << item.second.frame_count * baseline_model->cfg().seconds_per_output_frame()
                     << ",\"diarized_frames\":" << item.second.frame_count << ",\"feature_frames\":" << feature_frames
                     << ",\"aosc_selected_source_frames\":" << item.second.aosc_selected_frames.size()
                     << ",\"feature_inference_ms\":" << item.second.elapsed_ms
                     << ",\"runtime_feature_host_copy_bytes\":"
                     << item.second.runtime_feature_host_copy_bytes
                     << ",\"benchmark_retention_copy_bytes\":"
                     << item.second.benchmark_retention_copy_bytes
                     << ",\"aosc_profile_ready\":" << (item.second.aosc_profile_ready ? "true" : "false")
                     << ",\"snapshot_bytes\":" << item.second.snapshot.size() << '}';
            }
        };
        emit_extraction(enrollment_results);
        emit_extraction(development_results);
        emit_extraction(calibration_results);
        emit_extraction(holdout_results);
        emit_extraction(preliminary_results);
        json << "],\"aosc_enrollment_sequence\":{\"profile_ready\":"
             << (warm_artifacts.profile_ready ? "true" : "false")
             << ",\"profile_bytes\":" << warm_artifacts.aosc_profile.size()
             << ",\"snapshot_bytes\":" << warm_artifacts.full_snapshot.size()
             << ",\"selected_source_frames\":" << warm_artifacts.selected_source_frames
             << "},\"max_chunk_feature_view_bytes\":";
        size_t largest_chunk = 0;
        auto find_chunk_max = [&](const auto& group) {
            for (const auto& item : group) largest_chunk = std::max(largest_chunk, item.second.max_chunk_feature_bytes);
        };
        find_chunk_max(enrollment_results);
        find_chunk_max(development_results);
        find_chunk_max(calibration_results);
        find_chunk_max(holdout_results);
        find_chunk_max(preliminary_results);
        json << largest_chunk << ",\"max_chunk_runtime_feature_host_copy_bytes\":";
        size_t largest_runtime_chunk = 0;
        auto find_runtime_chunk_max = [&](const auto& group) {
            for (const auto& item : group)
                largest_runtime_chunk =
                    std::max(largest_runtime_chunk, item.second.max_chunk_runtime_output_bytes);
        };
        find_runtime_chunk_max(enrollment_results);
        find_runtime_chunk_max(development_results);
        find_runtime_chunk_max(calibration_results);
        find_runtime_chunk_max(holdout_results);
        find_runtime_chunk_max(preliminary_results);
        json << largest_runtime_chunk << "},\n";

        json << "  \"regression\":{\"status\":" << json_quote(regression_mismatches == 0 ? "pass" : "fail")
             << ",\"clips_compared\":" << regression_cases << ",\"mismatching_clips\":" << regression_mismatches
             << ",\"raw_probability_values_compared\":" << probability_values
             << ",\"identical_channel_assignment_frames\":" << channel_assignments
             << ",\"identical_segment_cases\":" << segment_cases
             << ",\"identical_aosc_profile_cases\":" << profile_cases
             << ",\"identical_snapshot_cases\":" << snapshot_cases << "},\n";
        json << "  \"performance\":{\"mean_baseline_stream_ms\":";
        json_number(json, baseline_ms);
        json << ",\"mean_research_stream_ms\":";
        json_number(json, research_ms);
        json << ",\"mean_delta_ms\":";
        json_number(json, research_ms - baseline_ms);
        json << ",\"overhead_percent\":";
        json_number(json, baseline_ms > 0.0 ? 100.0 * (research_ms - baseline_ms) / baseline_ms : NAN);
        json << ",\"runs_per_clip\":" << repeats
             << ",\"repeated_probabilities_and_features_bit_identical\":"
             << ((std::all_of(enrollment_results.begin(), enrollment_results.end(), [](const auto& x) { return x.second.repeat_data_bit_identical; }) &&
                  std::all_of(development_results.begin(), development_results.end(), [](const auto& x) { return x.second.repeat_data_bit_identical; }) &&
                  std::all_of(calibration_results.begin(), calibration_results.end(), [](const auto& x) { return x.second.repeat_data_bit_identical; }) &&
                  std::all_of(holdout_results.begin(), holdout_results.end(), [](const auto& x) { return x.second.repeat_data_bit_identical; }) &&
                  std::all_of(preliminary_results.begin(), preliminary_results.end(), [](const auto& x) { return x.second.repeat_data_bit_identical; }))
                     ? "true" : "false") << "}\n";

        json << "  ,\"evaluation\":{\"status\":" << json_quote(dataset_status)
             << ",\"development_used_for_method_selection\":false"
             << ",\"method_selection\":\"predeclared methods reported; no automatic selection\""
             << ",\"threshold_source\":\"calibration partition only\""
             << ",\"final_holdout_used_for_selection\":false"
             << ",\"diarization_metrics\":";
        if (manifest.reference_turns.empty()) {
            json << "{\"status\":\"reference_rttm_not_configured\"}";
        } else {
            json << "{\"status\":\"computed\",\"frame_period_sec\":";
            json_number(json, baseline_model->cfg().seconds_per_output_frame());
            json << ",\"partitions\":{";
            auto emit_diarization_partition = [&](const std::string& name, const ResultGroup& group) {
                json << json_quote(name) << ":{\"all\":";
                emit_diarization_counts(
                    json, evaluate_diarization(group, manifest.reference_turns,
                                               baseline_model->cfg().seconds_per_output_frame()));
                json << ",\"clean_only\":";
                emit_diarization_counts(
                    json, evaluate_diarization(group, manifest.reference_turns,
                                               baseline_model->cfg().seconds_per_output_frame(), 0));
                json << ",\"overlap_only\":";
                emit_diarization_counts(
                    json, evaluate_diarization(group, manifest.reference_turns,
                                               baseline_model->cfg().seconds_per_output_frame(), 1));
                json << "},";
            };
            emit_diarization_partition("development", development_results);
            emit_diarization_partition("calibration", calibration_results);
            emit_diarization_partition("holdout", holdout_results);
            emit_diarization_partition("preliminary", preliminary_results);
            const auto position = json.tellp();
            if (position > 0) {
                // Drop the final comma added by the partition writer.
                const std::string current = json.str();
                json.str(current.substr(0, current.size() - 1));
                json.clear();
                json.seekp(0, std::ios::end);
            }
            json << "}}";
        }
        json << ",\"speaker_matching\":[";
        bool first_metric = true;
        for (const auto& rep : available_representations(manifest)) {
            const int factor = baseline_model->cfg().upsample_factor;
            auto profile_map = make_profiles(enrollment_results, rep, factor);
            const auto aosc_profiles = make_aosc_selected_profiles(warm_artifacts, rep);
            const std::string aosc_suffix = std::string(1, '\x1f') + "aosc_selected";
            for (const auto& speaker : enrolled) {
                const std::string key = speaker + aosc_suffix;
                const auto selected = aosc_profiles.find(key);
                profile_map[key] = selected == aosc_profiles.end() ? PrototypeSet{} : selected->second;
            }
            int dim = 0;
            if (!enrollment_results.empty()) {
                const auto it = enrollment_results.front().second.features.find(rep);
                if (it != enrollment_results.front().second.features.end()) dim = it->second.dim;
            }
            for (const auto& pooling : pooling_methods()) {
                for (const auto& aggregation : similarity_methods()) {
                    const auto dev = score_partition(development_results, rep, pooling, aggregation, profile_map,
                        factor, baseline_model->cfg().seconds_per_output_frame());
                    const auto cal = score_partition(calibration_results, rep, pooling, aggregation, profile_map,
                        factor, baseline_model->cfg().seconds_per_output_frame());
                    const auto hold = score_partition(holdout_results, rep, pooling, aggregation, profile_map,
                        factor, baseline_model->cfg().seconds_per_output_frame());
                    const auto prelim = score_partition(preliminary_results, rep, pooling, aggregation, profile_map,
                        factor, baseline_model->cfg().seconds_per_output_frame());
                    size_t calibration_known = 0, calibration_unknown = 0;
                    const double threshold = manifest.preliminary_only
                        ? NAN : calibrated_threshold(cal, calibration_known, calibration_unknown);
                    if (!first_metric) json << ',';
                    first_metric = false;
                    json << "{\"representation\":" << json_quote(rep) << ",\"dimension\":" << dim
                         << ",\"pooling\":" << json_quote(pooling)
                         << ",\"similarity_aggregation\":" << json_quote(aggregation)
                         << ",\"profile_speakers\":";
                    size_t profile_speakers = 0, profile_bytes = 0;
                    const std::string suffix = std::string(1, '\x1f') + pooling;
                    for (const auto& [key, protos] : profile_map) {
                        if (key.size() >= suffix.size() && key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
                            if (!protos.empty()) ++profile_speakers;
                            for (const auto& proto : protos) profile_bytes += proto.size() * sizeof(float);
                        }
                    }
                    json << profile_speakers << ",\"profile_bytes\":" << profile_bytes
                         << ",\"calibration\":{\"known_attempts\":" << calibration_known
                         << ",\"unknown_attempts\":" << calibration_unknown << ",\"acceptance_threshold\":";
                    json_number(json, threshold);
                    json << "},\"comparison_costs\":[";
                    const std::string profile_suffix = std::string(1, '\x1f') + pooling;
                    const PrototypeSet* reference_profile = nullptr;
                    for (const auto& [key, protos] : profile_map)
                        if (key.size() >= profile_suffix.size() &&
                            key.compare(key.size() - profile_suffix.size(), profile_suffix.size(), profile_suffix) == 0 &&
                            !protos.empty()) {
                            reference_profile = &protos;
                            break;
                        }
                    PrototypeSet query;
                    const auto* query_group = !development_results.empty() ? &development_results : &enrollment_results;
                    if (!query_group->empty()) {
                        const auto feature = query_group->front().second.features.find(rep);
                        if (feature != query_group->front().second.features.end())
                            query = make_evaluation_prototypes(feature->second, query_group->front().second,
                                                               pooling, factor);
                    }
                    bool first_cost = true;
                    for (size_t profile_count : {size_t(2), size_t(10), size_t(100)}) {
                        double us_per_query = NAN;
                        if (reference_profile && !query.empty()) {
                            volatile double sink = 0.0;
                            constexpr int iterations = 20;
                            const auto cost_begin = std::chrono::steady_clock::now();
                            for (int iteration = 0; iteration < iterations; ++iteration)
                                for (size_t identity = 0; identity < profile_count; ++identity)
                                    sink += aggregate_similarity(query, *reference_profile, aggregation);
                            const auto cost_end = std::chrono::steady_clock::now();
                            us_per_query = std::chrono::duration<double, std::micro>(cost_end - cost_begin).count() /
                                           iterations;
                            (void)sink;
                        }
                        if (!first_cost) json << ',';
                        first_cost = false;
                        json << "{\"profile_count\":" << profile_count
                             << ",\"prototypes_per_identity\":" << (reference_profile ? reference_profile->size() : 0)
                             << ",\"estimated_multiply_adds\":"
                             << (static_cast<uint64_t>(profile_count) * (reference_profile ? reference_profile->size() : 0) *
                                 query.size() * static_cast<uint64_t>(dim))
                             << ",\"synthetic_profile_reuse\":true,\"mean_scoring_us\":";
                        json_number(json, us_per_query);
                        json << '}';
                    }
                    json << "],\"partitions\":{";
                    auto emit_partition_metrics = [&](const std::string& name, const std::vector<Attempt>& attempts,
                                                      bool use_threshold) {
                        const MetricCounts counts = evaluate_attempts(attempts, use_threshold ? threshold : NAN);
                        const RocCounts roc = roc_eer(attempts);
                        if (name != "development") json << ',';
                        json << json_quote(name) << ":{\"attempts\":" << attempts.size()
                             << ",\"known_attempts\":" << counts.known_attempts
                             << ",\"known_top1_correct\":" << counts.known_top1_correct
                             << ",\"correct_identification_rate\":";
                        json_number(json, counts.known_attempts ? counts.known_top1_correct / static_cast<double>(counts.known_attempts) : NAN);
                        json << ",\"known_speaker_confusions\":" << counts.known_confusions
                             << ",\"known_false_rejections\":";
                        json_number(json, use_threshold && std::isfinite(threshold)
                                              ? static_cast<double>(counts.known_false_rejections) : NAN);
                        json << ",\"known_false_rejection_rate\":";
                        json_number(json, use_threshold && std::isfinite(threshold) && counts.known_attempts
                                              ? counts.known_false_rejections / static_cast<double>(counts.known_attempts)
                                              : NAN);
                        json << ",\"unknown_attempts\":" << counts.unknown_attempts
                             << ",\"unknown_false_accepts\":";
                        json_number(json, use_threshold && std::isfinite(threshold)
                                              ? static_cast<double>(counts.unknown_false_accepts) : NAN);
                        json << ",\"unknown_false_accept_rate\":";
                        json_number(json, use_threshold && std::isfinite(threshold) && counts.unknown_attempts
                                              ? counts.unknown_false_accepts / static_cast<double>(counts.unknown_attempts)
                                              : NAN);
                        json << ",\"unknown_rejection_calibrated\":"
                             << (use_threshold && std::isfinite(threshold) ? "true" : "false")
                             << ",\"forced_top1_unknown_false_accept_count\":" << counts.unknown_attempts
                             << ",\"forced_top1_unknown_false_accept_rate\":";
                        json_number(json, counts.unknown_attempts ? 1.0 : NAN);
                        json << ",\"mean_best_cosine_known\":";
                        json_number(json, counts.known_attempts ? counts.known_best_sum / counts.known_attempts : NAN);
                        json << ",\"mean_target_cosine_known\":";
                        json_number(json, counts.known_target_count ? counts.known_target_sum / counts.known_target_count : NAN);
                        json << ",\"mean_best_cosine_unknown\":";
                        json_number(json, counts.unknown_attempts ? counts.unknown_best_sum / counts.unknown_attempts : NAN);
                        json << ",\"top1_top2_margin_mean\":";
                        json_number(json, counts.margin_count ? counts.margin_sum / counts.margin_count : NAN);
                        json << ",\"verification_pair_counts\":{\"positive\":" << roc.positive
                             << ",\"negative\":" << roc.negative << "},\"eer\":";
                        json_number(json, manifest.preliminary_only && name == "preliminary" ? NAN : roc.eer);
                        json << ",\"eer_status\":\""
                             << (manifest.preliminary_only && name == "preliminary"
                                     ? "preliminary_fixture_not_statistical"
                                     : (roc.positive < 10 || roc.negative < 10 ? "underpowered" : "computed"))
                             << "\"";
                        const auto unknown_summary = unknown_pairs(
                            name == "development" ? development_results :
                            name == "calibration" ? calibration_results :
                            name == "holdout" ? holdout_results : preliminary_results,
                            rep, pooling, aggregation, factor);
                        json << ",\"unknown_pairwise\":{\"same_identity_pairs\":" << unknown_summary.same_pairs
                             << ",\"same_identity_mean_similarity\":";
                        json_number(json, unknown_summary.same_pairs ? unknown_summary.same_sum / unknown_summary.same_pairs : NAN);
                        json << ",\"different_unknown_pairs\":" << unknown_summary.different_pairs
                             << ",\"different_unknown_mean_similarity\":";
                        json_number(json, unknown_summary.different_pairs ? unknown_summary.different_sum / unknown_summary.different_pairs : NAN);
                        json << "},\"duration_buckets\":[";
                        const std::array<std::string, 4> bucket_names{{"under_1s", "1_to_2s", "2_to_5s", "over_5s"}};
                        std::array<std::vector<Attempt>, 4> buckets;
                        for (const auto& attempt : attempts) {
                            const size_t bucket = attempt.duration_sec < 1.0 ? 0 : attempt.duration_sec < 2.0 ? 1 :
                                                  attempt.duration_sec < 5.0 ? 2 : 3;
                            buckets[bucket].push_back(attempt);
                        }
                        for (size_t i = 0; i < buckets.size(); ++i) {
                            if (i) json << ',';
                            const MetricCounts bucket_counts = evaluate_attempts(
                                buckets[i], use_threshold ? threshold : NAN);
                            json << "{\"duration\":" << json_quote(bucket_names[i])
                                 << ",\"attempts\":" << buckets[i].size()
                                 << ",\"known_attempts\":" << bucket_counts.known_attempts
                                 << ",\"known_top1_correct\":" << bucket_counts.known_top1_correct
                                 << ",\"known_false_rejection_count\":";
                            json_number(json, use_threshold && std::isfinite(threshold)
                                                  ? static_cast<double>(bucket_counts.known_false_rejections) : NAN);
                            json << ",\"unknown_attempts\":" << bucket_counts.unknown_attempts
                                 << ",\"unknown_false_accept_count\":";
                            json_number(json, use_threshold && std::isfinite(threshold)
                                                  ? static_cast<double>(bucket_counts.unknown_false_accepts) : NAN);
                            json << '}';
                        }
                        json << "],\"overlap_breakdown\":{";
                        const std::array<std::string, 3> overlap_names{{"clean", "overlap", "unlabeled"}};
                        std::array<std::vector<Attempt>, 3> overlap_groups;
                        for (const auto& attempt : attempts) {
                            const size_t group = !attempt.has_overlap_label ? 2 : attempt.overlap ? 1 : 0;
                            overlap_groups[group].push_back(attempt);
                        }
                        for (size_t i = 0; i < overlap_groups.size(); ++i) {
                            if (i) json << ',';
                            const MetricCounts overlap_counts = evaluate_attempts(
                                overlap_groups[i], use_threshold ? threshold : NAN);
                            json << json_quote(overlap_names[i]) << ":{\"attempts\":" << overlap_groups[i].size()
                                 << ",\"known_attempts\":" << overlap_counts.known_attempts
                                 << ",\"known_top1_correct\":" << overlap_counts.known_top1_correct
                                 << ",\"known_false_rejection_count\":";
                            json_number(json, use_threshold && std::isfinite(threshold)
                                                  ? static_cast<double>(overlap_counts.known_false_rejections)
                                                  : NAN);
                            json << ",\"known_false_rejection_rate\":";
                            json_number(json, use_threshold && std::isfinite(threshold) &&
                                                  overlap_counts.known_attempts
                                                  ? overlap_counts.known_false_rejections /
                                                        static_cast<double>(overlap_counts.known_attempts)
                                                  : NAN);
                            json << ",\"unknown_attempts\":" << overlap_counts.unknown_attempts
                                 << ",\"forced_top1_unknown_false_accept_count\":" << overlap_counts.unknown_attempts
                                 << ",\"calibrated_unknown_false_accept_count\":";
                            json_number(json, use_threshold && std::isfinite(threshold)
                                                  ? static_cast<double>(overlap_counts.unknown_false_accepts)
                                                  : NAN);
                            json << '}';
                        }
                        json << "}}";
                    };
                    emit_partition_metrics("development", dev, false);
                    emit_partition_metrics("calibration", cal, true);
                    emit_partition_metrics("holdout", hold, true);
                    emit_partition_metrics("preliminary", prelim, false);
                    json << "}}";
                }
            }
        }
        warm_artifacts.selected_by_speaker.clear();
        json << "]},\n  \"context_probes\":{";
        if (manifest.context_probes.empty()) {
            json << "\"status\":\"not_configured\",\"cases\":[]}\n}\n";
        } else {
            json << "\"status\":\"executed\",\"cases\":[";
            baseline_model.reset();
            model = std::make_shared<DiarModel>(backend, manifest.model, BatchingConfig{}, research);
            std::map<std::string, Clip> clips_by_id;
            auto index_clips = [&](const std::vector<Clip>& clips) {
                for (const auto& clip : clips) clips_by_id.emplace(clip.id, clip);
            };
            index_clips(manifest.enrollment);
            index_clips(manifest.development);
            index_clips(manifest.calibration);
            index_clips(manifest.holdout);
            index_clips(manifest.preliminary);
            auto find_result = [&](const std::string& id) -> const Extraction* {
                for (const auto* group : {&development_results, &calibration_results,
                                          &holdout_results, &preliminary_results})
                    for (const auto& item : *group)
                        if (item.first.id == id) return &item.second;
                return nullptr;
            };
            for (size_t probe_index = 0; probe_index < manifest.context_probes.size(); ++probe_index) {
                const auto& probe = manifest.context_probes[probe_index];
                const auto target_it = clips_by_id.find(probe.target);
                const Extraction* fresh = find_result(probe.target);
                if (target_it == clips_by_id.end() || fresh == nullptr)
                    throw std::runtime_error("context probe target must reference an evaluation clip");
                if (probe_index) json << ',';
                json << "{\"index\":" << probe_index << ",\"context_modes\":[";
                bool first_mode = true;
                auto emit_mode = [&](const std::string& name, const std::vector<Clip>& contexts,
                                     const std::vector<uint8_t>* profile,
                                     const std::vector<uint8_t>* snapshot) {
                    if (!first_mode) json << ',';
                    first_mode = false;
                    try {
                        const Extraction context = extract_target_after_context(
                            model, target_it->second, geometry, contexts, profile, snapshot,
                            snapshot ? warm_artifacts.total_input_samples : 0);
                        size_t target_feature_frames = 0;
                        for (const auto& [feature_name, matrix] : context.features)
                            target_feature_frames = std::max(target_feature_frames, matrix.frames());
                        json << "{\"mode\":" << json_quote(name) << ",\"status\":\"ok\",\"context_clips\":"
                             << contexts.size() << ",\"target_feature_frames\":" << target_feature_frames
                             << ",\"stream_total_frames\":" << context.frame_count
                             << ",\"aosc_profile_changed_after_context_sequence\":";
                        if (warm_artifacts.profile_ready && context.pre_target_aosc_profile_ready)
                            json << (context.pre_target_aosc_profile == warm_artifacts.aosc_profile ? "false" : "true");
                        else json << "null";
                        json << ",\"aosc_profile_changed_after_target\":";
                        if (warm_artifacts.profile_ready && context.aosc_profile_ready)
                            json << (context.aosc_profile == warm_artifacts.aosc_profile ? "false" : "true");
                        else json << "null";
                        json << ",\"enrollment_profile_artifact_unchanged\":true,\"representations\":[";
                        bool first_feature = true;
                        for (const auto& rep : available_representations(manifest)) {
                            const auto base_feature = fresh->features.find(rep);
                            const auto context_feature = context.features.find(rep);
                            if (base_feature == fresh->features.end() || context_feature == context.features.end()) continue;
                            const auto similarity = context_cosine(base_feature->second, context_feature->second);
                            if (!first_feature) json << ',';
                            first_feature = false;
                            json << "{\"name\":" << json_quote(rep) << ",\"dimension\":" << base_feature->second.dim
                                 << ",\"fresh_frames\":" << base_feature->second.frames()
                                 << ",\"context_frames\":" << context_feature->second.frames()
                                 << ",\"aligned_frames_compared\":" << similarity.second
                                 << ",\"mean_frame_cosine\":";
                            json_number(json, similarity.first);
                            json << '}';
                        }
                        json << "]}";
                    } catch (const std::exception& error) {
                        (void)error;
                        json << "{\"mode\":" << json_quote(name) << ",\"status\":\"error\",\"message\":\"context extraction failed\"}";
                    }
                };
                auto resolve_contexts = [&](const std::vector<std::string>& ids, bool known) {
                    std::vector<Clip> contexts;
                    for (const auto& id : ids) {
                        const auto it = clips_by_id.find(id);
                        if (it == clips_by_id.end()) throw std::runtime_error("unknown context clip id: " + id);
                        if (it->second.known != known)
                            throw std::runtime_error("context clip has the wrong known/unknown label: " + id);
                        contexts.push_back(it->second);
                    }
                    return contexts;
                };
                if (!probe.after_known.empty())
                {
                    const auto contexts = resolve_contexts(probe.after_known, true);
                    emit_mode("after_registered_speaker", contexts, nullptr, nullptr);
                    if (warm_artifacts.profile_ready)
                        emit_mode("after_registered_speaker_with_aosc_profile", contexts,
                                  &warm_artifacts.aosc_profile, nullptr);
                }
                if (!probe.after_unknown.empty()) {
                    const auto contexts = resolve_contexts(probe.after_unknown, false);
                    emit_mode("after_unknown_speaker", contexts, nullptr, nullptr);
                    if (warm_artifacts.profile_ready)
                        emit_mode("after_unknown_speaker_with_aosc_profile", contexts,
                                  &warm_artifacts.aosc_profile, nullptr);
                }
                if (!probe.after_unknown_long.empty()) {
                    const auto contexts = resolve_contexts(probe.after_unknown_long, false);
                    emit_mode("long_unknown_sequence", contexts, nullptr, nullptr);
                    if (warm_artifacts.profile_ready)
                        emit_mode("long_unknown_sequence_after_aosc_profile", contexts,
                                  &warm_artifacts.aosc_profile, nullptr);
                }
                if (probe.reverse_enrollment_order) {
                    std::vector<Clip> reversed = manifest.enrollment;
                    std::reverse(reversed.begin(), reversed.end());
                    emit_mode("reverse_enrollment_order", reversed, nullptr, nullptr);
                }
                if (probe.aosc_only_warm_start) {
                    if (warm_artifacts.profile_ready)
                        emit_mode("aosc_only_warm_start", {}, &warm_artifacts.aosc_profile, nullptr);
                    else {
                        if (!first_mode) json << ',';
                        first_mode = false;
                        json << "{\"mode\":\"aosc_only_warm_start\",\"status\":\"profile_not_ready\"}";
                    }
                }
                if (probe.full_snapshot_warm_start)
                    emit_mode("full_snapshot_warm_start", {}, nullptr, &warm_artifacts.full_snapshot);
                json << "]}";
            }
            json << "]}\n}\n";
        }

        if (report_path == "-") std::cout << json.str();
        else {
            std::ofstream out(report_path, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot write report: " + report_path);
            out << json.str();
            if (!out) throw std::runtime_error("failed writing report: " + report_path);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "diar_representation_probe: " << error.what() << '\n';
        return 1;
    }
}
