// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Research-only, fresh-stream Nemotron representation extractor.

#include "audio_file.h"
#include "diar_enrollment_profile.h"
#include "diar_snapshot.h"
#include "diarizer.h"
#include "runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nemo_speech::asr;

namespace {

struct Job {
    std::string id;
    std::string speaker;
    std::string utterance;
    std::string mic;
    std::string source_path;
    std::string source_sha256;
    std::string converted_sha256;
    std::string vad_sha256;
    std::filesystem::path wav_path;
    std::filesystem::path vad_path;
    std::filesystem::path context_wav_path;
    std::string context_mode = "fresh_empty_stream";
    std::string context_speaker;
    std::string context_sha256;
    std::filesystem::path aosc_profile_path;
};

struct Matrix {
    int dim = 0;
    std::vector<int64_t> frame_ids;
    std::vector<uint8_t> speech;
    std::vector<float> values;
};

struct RunResult {
    std::map<std::string, Matrix> matrices;
    std::vector<float> probabilities;
    int64_t frame_count = 0;
    int speakers = 0;
    std::vector<DiarSegment> segments;
};

std::string json_quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
                else out << static_cast<char>(c);
        }
    }
    out << '"';
    return out.str();
}

std::string hex_fingerprint(const std::array<uint8_t, 32>& value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint8_t byte : value) out << std::setw(2) << static_cast<unsigned>(byte);
    return out.str();
}

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> values;
    size_t begin = 0;
    for (;;) {
        size_t end = line.find('\t', begin);
        values.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return values;
}

std::vector<Job> read_jobs(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open job list: " + path.string());
    std::vector<Job> jobs;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        const auto fields = split_tabs(line);
        if (fields.size() != 10 && fields.size() != 15)
            throw std::runtime_error("job list line " + std::to_string(line_number) + " must have ten or fifteen TSV fields");
        Job job;
        job.id = fields[0];
        job.speaker = fields[1];
        job.utterance = fields[2];
        job.mic = fields[3];
        job.source_path = fields[4];
        job.source_sha256 = fields[5];
        job.converted_sha256 = fields[6];
        job.vad_sha256 = fields[7];
        job.wav_path = fields[8];
        job.vad_path = fields[9];
        if (fields.size() == 15) {
            job.context_wav_path = fields[10];
            job.context_mode = fields[11];
            job.context_speaker = fields[12];
            job.context_sha256 = fields[13];
            job.aosc_profile_path = fields[14];
            if (job.context_mode.empty()) job.context_mode = "fresh_empty_stream";
        }
        if (job.id.empty() || job.speaker.empty() || job.utterance.empty() || job.source_sha256.size() != 64)
            throw std::runtime_error("job list line " + std::to_string(line_number) + " has invalid identifiers/hash");
        jobs.push_back(std::move(job));
    }
    if (jobs.empty()) throw std::runtime_error("job list is empty");
    return jobs;
}

void write_u16(std::ostream& out, uint16_t value) {
    out.put(static_cast<char>(value & 0xff));
    out.put(static_cast<char>((value >> 8) & 0xff));
}
void write_u32(std::ostream& out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.put(static_cast<char>((value >> (8 * i)) & 0xff));
}
void write_u64(std::ostream& out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((value >> (8 * i)) & 0xff));
}

void write_record(const std::filesystem::path& path, const Matrix& matrix) {
    if (matrix.dim <= 0 || matrix.frame_ids.empty() || matrix.speech.size() != matrix.frame_ids.size() ||
        matrix.values.size() != matrix.frame_ids.size() * static_cast<size_t>(matrix.dim))
        throw std::runtime_error("refusing to serialize malformed feature matrix: " + path.string());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create feature record: " + path.string());
    out.write("NVFTR001", 8);
    write_u16(out, 1);       // schema version
    write_u16(out, 48);      // fixed header length
    write_u32(out, static_cast<uint32_t>(matrix.dim));
    write_u64(out, static_cast<uint64_t>(matrix.frame_ids.size()));
    write_u32(out, 80000);   // encoder frame period in microseconds
    write_u32(out, 16000);   // canonical audio sample rate
    write_u32(out, 1);       // dtype: IEEE float32
    write_u32(out, 0x01020304); // byte-order marker; payload is always little-endian
    write_u32(out, static_cast<uint32_t>(matrix.speech.size()));
    write_u32(out, 0);       // reserved, must be zero
    for (size_t frame = 0; frame < matrix.frame_ids.size(); ++frame) {
        write_u64(out, static_cast<uint64_t>(matrix.frame_ids[frame]));
        out.put(static_cast<char>(matrix.speech[frame]));
        out.put(0); out.put(0); out.put(0);
        const size_t offset = frame * static_cast<size_t>(matrix.dim);
        for (int d = 0; d < matrix.dim; ++d) {
            uint32_t bits = 0;
            const float value = matrix.values[offset + static_cast<size_t>(d)];
            if (!std::isfinite(value)) throw std::runtime_error("non-finite feature value in " + path.string());
            std::memcpy(&bits, &value, sizeof(bits));
            write_u32(out, bits);
        }
    }
    out.flush();
    if (!out) throw std::runtime_error("short write for feature record: " + path.string());
}

std::string representation_config(const std::vector<std::string>& representations) {
    std::ostringstream out;
    bool first = true;
    for (const auto& item : representations) {
        if (!first) out << ',';
        first = false;
        out << item;
    }
    return out.str();
}

SortformerResearchConfig make_research_config(const std::vector<std::string>& names) {
    SortformerResearchConfig config;
    for (const auto& name : names) {
        if (name == "pre_encoder") config.pre_encoder = true;
        else if (name == "encoder_output") config.encoder_output = true;
        else if (name == "encoder_proj") config.encoder_proj = true;
        else if (name.rfind("encoder.layer.", 0) == 0) {
            size_t consumed = 0;
            const int layer = std::stoi(name.substr(14), &consumed);
            if (consumed != name.size() - 14 || layer < 0 || layer >= 31)
                throw std::runtime_error("invalid layer representation: " + name);
            config.transformer_layers.push_back(layer);
        } else throw std::runtime_error("unsupported representation: " + name);
    }
    std::sort(config.transformer_layers.begin(), config.transformer_layers.end());
    if (std::adjacent_find(config.transformer_layers.begin(), config.transformer_layers.end()) != config.transformer_layers.end())
        throw std::runtime_error("duplicate layer representation requested");
    return config;
}

void feed_stream(DiarizationStream& stream, const std::vector<float>& samples, bool finish = true) {
    constexpr size_t block = 16000 * 160 / 1000;
    for (size_t offset = 0; offset < samples.size(); offset += block) {
        const size_t count = std::min(block, samples.size() - offset);
        stream.push(samples.data() + offset, count, 16000);
    }
    if (finish) stream.finish();
}

RunResult run_one(const std::shared_ptr<DiarModel>& model, const Job& job, bool collect_features,
                  const std::vector<uint8_t>* vad) {
    const auto audio = nemo_speech::audio::load_wav_file(job.wav_path.string());
    if (audio.sample_rate != 16000) throw std::runtime_error("expected converted 16 kHz WAV: " + job.wav_path.string());
    std::vector<float> prefix;
    if (!job.context_wav_path.empty()) {
        if (job.context_sha256.size() != 64 ||
            hex_fingerprint(DiarStreamSnapshotCodec::fingerprint_file(job.context_wav_path.string())) != job.context_sha256)
            throw std::runtime_error("context WAV SHA-256 does not match the predeclared context source");
        const auto context = nemo_speech::audio::load_wav_file(job.context_wav_path.string());
        if (context.sample_rate != 16000) throw std::runtime_error("context WAV must be canonical 16 kHz mono");
        prefix = context.samples;
        constexpr size_t block = 16000 * 160 / 1000;
        const size_t padded = ((prefix.size() + block - 1) / block) * block;
        prefix.resize(padded, 0.0f);
    }
    const int64_t target_start_frame = static_cast<int64_t>(prefix.size() / 1280);
    RunResult result;
    auto stream = std::make_unique<DiarizationStream>(model, DiarGeometry::v3_streaming());
    if (!job.aosc_profile_path.empty()) {
        std::ifstream profile_input(job.aosc_profile_path, std::ios::binary);
        if (!profile_input) throw std::runtime_error("cannot open AOSC profile: " + job.aosc_profile_path.string());
        std::vector<uint8_t> profile((std::istreambuf_iterator<char>(profile_input)), std::istreambuf_iterator<char>());
        DiarAoscProfileCodec::decode(*stream, profile.data(), profile.size());
    }
    if (collect_features) {
        stream->set_research_feature_observer([&](int64_t start, const std::vector<SortformerResearchFeatureView>& views) {
            for (const auto& view : views) {
                auto& matrix = result.matrices[view.name];
                if (matrix.dim == 0) matrix.dim = view.dim;
                if (matrix.dim != view.dim || view.frames < 0 || (view.frames && view.values == nullptr))
                    throw std::runtime_error("research feature tensor shape mismatch");
                for (int frame = 0; frame < view.frames; ++frame) {
                    const int64_t global_frame_id = start + frame;
                    if (global_frame_id < target_start_frame) continue;
                    const int64_t frame_id = global_frame_id - target_start_frame;
                    matrix.frame_ids.push_back(frame_id);
                    uint8_t speech = 0;  // eight 10ms VAD decisions packed low-bit first
                    if (vad) {
                        const size_t first = static_cast<size_t>(frame_id) * 8;
                        for (size_t subframe = 0; subframe < 8 && first + subframe < vad->size(); ++subframe)
                            if ((*vad)[first + subframe]) speech |= static_cast<uint8_t>(1u << subframe);
                    }
                    matrix.speech.push_back(speech);
                    const size_t offset = static_cast<size_t>(frame) * view.dim;
                    matrix.values.insert(matrix.values.end(), view.values + offset, view.values + offset + view.dim);
                }
            }
        });
    }
    if (!prefix.empty()) feed_stream(*stream, prefix, false);
    feed_stream(*stream, audio.samples);
    result.probabilities = stream->frame_probabilities();
    result.frame_count = stream->frame_count();
    result.speakers = stream->num_speakers();
    result.segments = stream->segments();
    return result;
}

std::vector<uint8_t> read_vad(const Job& job, size_t expected_frames) {
    std::ifstream in(job.vad_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open VAD mask: " + job.vad_path.string());
    std::vector<uint8_t> mask((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (mask.size() != expected_frames)
        throw std::runtime_error("VAD mask does not align to canonical 10 ms grid: " + job.id);
    for (uint8_t value : mask) if (value > 1) throw std::runtime_error("VAD mask contains non-binary values");
    return mask;
}

void emit_metadata(std::ostream& out, const Job& job, const std::string& representation, int dim,
                   size_t frames, const std::string& model_hash, const std::string& backend,
                   const std::string& device_name, const std::string& record_file) {
    out << "{\"schema\":\"nemo-speech.voiceid-feature.v1\",\"job_id\":" << json_quote(job.id)
        << ",\"speaker_id\":" << json_quote(job.speaker) << ",\"source_utterance_id\":" << json_quote(job.utterance)
        << ",\"source_microphone\":" << json_quote(job.mic) << ",\"source_path\":" << json_quote(job.source_path)
        << ",\"source_sha256\":" << json_quote(job.source_sha256)
        << ",\"converted_audio_sha256\":" << json_quote(job.converted_sha256)
        << ",\"vad_mask_sha256\":" << json_quote(job.vad_sha256)
        << ",\"feature_representation\":" << json_quote(representation) << ",\"feature_dimension\":" << dim
        << ",\"frame_count\":" << frames << ",\"frame_period_sec\":0.08"
        << ",\"timestamp_alignment\":\"zero-based 80ms encoder frame start; stored frame id * 0.08 seconds\""
        << ",\"model_gguf_fingerprint\":" << json_quote(model_hash)
        << ",\"inference_backend\":" << json_quote(backend)
        << ",\"device_name\":" << json_quote(device_name)
        << ",\"streaming_geometry\":\"v3_streaming; 160ms push blocks; initialization described by context_initialization\""
        << ",\"context_initialization\":" << json_quote(job.context_mode)
        << ",\"context_speaker_id\":" << (job.context_speaker.empty() ? "null" : json_quote(job.context_speaker))
        << ",\"context_audio_sha256\":" << (job.context_sha256.empty() ? "null" : json_quote(job.context_sha256))
        << ",\"aosc_profile_sha256\":";
    if (job.aosc_profile_path.empty()) out << "null";
    else out << json_quote(hex_fingerprint(DiarStreamSnapshotCodec::fingerprint_file(job.aosc_profile_path.string())));
    out
        << ",\"vad_mask\":\"Task08 10ms RMS mask; 20ms window, 10ms hop, max(-45dBFS, peak-28dB), bridge<=60ms, remove runs<30ms; eight 10ms decisions are packed low-bit-first per 80ms feature frame; pooling uses majority-active frames\""
        << ",\"dtype\":\"float32\",\"byte_order\":\"little\",\"record_file\":" << json_quote(record_file) << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path jobs_path, output_dir, profile_jobs_path, profile_out_path;
        std::string model_path, backend_name = "cpu", requested_reps = "pre_encoder,encoder.layer.7,encoder_proj";
        int gpu_device = 0, parity_count = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&](const char* name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + name);
                return argv[++i];
            };
            if (arg == "--jobs") jobs_path = value("--jobs");
            else if (arg == "--build-aosc-profile") profile_jobs_path = value("--build-aosc-profile");
            else if (arg == "--profile-out") profile_out_path = value("--profile-out");
            else if (arg == "--model") model_path = value("--model");
            else if (arg == "--output-dir") output_dir = value("--output-dir");
            else if (arg == "--device") backend_name = value("--device");
            else if (arg == "--gpu-device") gpu_device = std::stoi(value("--gpu-device"));
            else if (arg == "--representations") requested_reps = value("--representations");
            else if (arg == "--parity-first") parity_count = std::stoi(value("--parity-first"));
            else throw std::runtime_error("unknown argument: " + arg);
        }
        if (model_path.empty()) throw std::runtime_error("--model is required");
        if (profile_jobs_path.empty() && (jobs_path.empty() || output_dir.empty()))
            throw std::runtime_error("usage: diar_voiceid_feature_probe --jobs TSV --model GGUF --output-dir DIR [--device cpu|vulkan --gpu-device N --representations LIST --parity-first N]");
        if (!profile_jobs_path.empty() && profile_out_path.empty())
            throw std::runtime_error("--profile-out is required with --build-aosc-profile");
        if (backend_name != "cpu" && backend_name != "vulkan") throw std::runtime_error("--device must be cpu or vulkan");
        if (gpu_device < 0 || (backend_name == "cpu" && gpu_device != 0)) throw std::runtime_error("invalid GPU device selection");
        const auto jobs = jobs_path.empty() ? std::vector<Job>{} : read_jobs(jobs_path);
        std::vector<std::string> representations;
        std::istringstream rep_stream(requested_reps);
        for (std::string rep; std::getline(rep_stream, rep, ',');) if (!rep.empty()) representations.push_back(rep);
        if (representations.empty()) throw std::runtime_error("at least one representation is required");
        const SortformerResearchConfig research = profile_jobs_path.empty()
            ? make_research_config(representations) : SortformerResearchConfig{};
        if (!std::filesystem::exists(model_path)) throw std::runtime_error("model does not exist: " + model_path);
        if (profile_jobs_path.empty()) std::filesystem::create_directories(output_dir);

        ggml_runtime::Params params;
        params.use_gpu = backend_name == "vulkan";
        params.gpu_device_idx = gpu_device;
        params.gpu_backend_name = params.use_gpu ? "Vulkan" : "";
        params.n_threads = 4;
        params.pe_bin_path = const_cast<char*>("");
        ggml_runtime::BackendManager backend(params);
        auto model = std::make_shared<DiarModel>(backend, model_path, BatchingConfig{}, research);
        const std::string model_hash = hex_fingerprint(DiarStreamSnapshotCodec::fingerprint_file(model_path));
        const std::string actual_backend = params.use_gpu ? backend.selected_gpu_backend_name() : "CPU";
        const std::string device_name = params.use_gpu ? backend.selected_gpu_device_name() : "CPU";
        if (params.use_gpu && actual_backend != "Vulkan") throw std::runtime_error("requested Vulkan but runtime selected " + actual_backend);

        if (!profile_jobs_path.empty()) {
            const auto profile_jobs = read_jobs(profile_jobs_path);
            auto stream = std::make_unique<DiarizationStream>(model, DiarGeometry::v3_streaming());
            std::set<std::string> profile_speakers;
            for (const auto& job : profile_jobs) {
                const auto audio = nemo_speech::audio::load_wav_file(job.wav_path.string());
                if (audio.sample_rate != 16000) throw std::runtime_error("profile job WAV must be 16 kHz mono");
                feed_stream(*stream, audio.samples, false);
                std::vector<float> pause(16000 * 250 / 1000, 0.0f);
                feed_stream(*stream, pause, false);
                profile_speakers.insert(job.speaker);
            }
            stream->finish();
            const auto profile = DiarAoscProfileCodec::encode(*stream);
            const auto profile_summary = DiarAoscProfileCodec::summarize(profile);
            std::filesystem::create_directories(profile_out_path.parent_path());
            {
                std::ofstream out(profile_out_path, std::ios::binary | std::ios::trunc);
                out.write(reinterpret_cast<const char*>(profile.data()), static_cast<std::streamsize>(profile.size()));
                if (!out) throw std::runtime_error("failed writing AOSC context profile");
            }
            std::ofstream report(profile_out_path.string() + ".json", std::ios::trunc);
            report << "{\"schema\":\"nemo-speech.voiceid-task09b-context-profile.v1\",\"model_fingerprint\":"
                   << json_quote(model_hash) << ",\"backend\":" << json_quote(actual_backend)
                   << ",\"device_name\":" << json_quote(device_name)
                   << ",\"source_job_count\":" << profile_jobs.size()
                   << ",\"training_context_speakers\":" << profile_speakers.size()
                   << ",\"native_speaker_channels\":" << profile_summary.speaker_channels
                   << ",\"cache_frames\":" << profile_summary.cache_frames
                   << ",\"profile_bytes\":" << profile.size()
                   << ",\"profile_sha256\":" << json_quote(hex_fingerprint(DiarStreamSnapshotCodec::fingerprint_file(profile_out_path.string())))
                   << ",\"context_pause_ms_per_utterance\":250,\"evaluation_identities_used\":false}\n";
            if (!report) throw std::runtime_error("failed writing AOSC context profile summary");
            std::cout << "AOSC context profile: " << profile.size() << " bytes, cache_frames="
                      << profile_summary.cache_frames << ", speakers=" << profile_speakers.size() << '\n';
            return 0;
        }

        std::ofstream metadata(output_dir / "features.jsonl", std::ios::app);
        if (!metadata) throw std::runtime_error("cannot open metadata output");
        size_t feature_records = 0, parity_checked = 0, parity_mismatches = 0;
        std::shared_ptr<DiarModel> baseline_model;
        for (size_t job_index = 0; job_index < jobs.size(); ++job_index) {
            const Job& job = jobs[job_index];
            const auto wav = nemo_speech::audio::load_wav_file(job.wav_path.string());
            if (wav.sample_rate != 16000) throw std::runtime_error("WAV must be canonical 16 kHz mono");
            const auto vad = read_vad(job, (wav.samples.size() + 159) / 160);
            RunResult result = run_one(model, job, true, &vad);
            for (const auto& representation : representations) {
                const auto found = result.matrices.find(representation);
                if (found == result.matrices.end()) throw std::runtime_error("requested feature tap missing: " + representation);
                const std::string stem = job.id + "." + representation;
                const std::string record_name = stem + ".vfeat";
                write_record(output_dir / record_name, found->second);
                emit_metadata(metadata, job, representation, found->second.dim, found->second.frame_ids.size(),
                              model_hash, actual_backend, device_name, record_name);
                ++feature_records;
            }
            if (static_cast<int>(job_index) < parity_count) {
                if (!baseline_model) {
                    baseline_model = std::make_shared<DiarModel>(backend, model_path, BatchingConfig{}, SortformerResearchConfig{});
                }
                const RunResult baseline = run_one(baseline_model, job, false, nullptr);
                ++parity_checked;
                if (result.probabilities != baseline.probabilities || result.frame_count != baseline.frame_count ||
                    result.speakers != baseline.speakers || result.segments.size() != baseline.segments.size())
                    ++parity_mismatches;
                if (static_cast<int>(job_index + 1) == parity_count) baseline_model.reset();
            }
            if ((job_index + 1) % 25 == 0 || job_index + 1 == jobs.size())
                std::cerr << "features: " << (job_index + 1) << '/' << jobs.size() << " jobs, "
                          << feature_records << " representation records\n";
        }
        std::ofstream summary(output_dir / "extractor_summary.json", std::ios::trunc);
        summary << "{\"schema\":\"nemo-speech.voiceid-extractor.v1\",\"model_fingerprint\":"
                << json_quote(model_hash) << ",\"requested_backend\":" << json_quote(backend_name)
                << ",\"actual_backend\":" << json_quote(actual_backend) << ",\"device_name\":" << json_quote(device_name)
                << ",\"job_count\":" << jobs.size() << ",\"feature_record_count\":" << feature_records
                << ",\"representations\":" << json_quote(representation_config(representations))
                << ",\"parity_checked_jobs\":" << parity_checked << ",\"parity_mismatches\":" << parity_mismatches << "}\n";
        if (!summary) throw std::runtime_error("failed writing extractor summary");
        if (parity_mismatches) throw std::runtime_error("research taps changed native diarization outputs");
        std::cout << "device=" << actual_backend << " / " << device_name << ", jobs=" << jobs.size()
                  << ", records=" << feature_records << ", tap_parity=" << parity_checked << '/' << parity_checked
                  << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "diar_voiceid_feature_probe: " << error.what() << '\n';
        return 1;
    }
}
