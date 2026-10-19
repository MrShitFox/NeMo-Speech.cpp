// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>

#include "diarizer.h"
#include "fe.h"

using namespace nemo_speech::asr;

namespace {

void
require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool
rejects_import(DiarizationStream& stream, const std::vector<uint8_t>& snapshot) {
    try {
        stream.import_state(snapshot.data(), snapshot.size());
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

uint32_t
test_crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xffffffffU;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return ~crc;
}

void
refresh_payload_crc(std::vector<uint8_t>& snapshot) {
    const uint32_t crc = test_crc32(snapshot.data() + 108, snapshot.size() - 108);
    for (unsigned i = 0; i < 4; ++i)
        snapshot[104 + i] = static_cast<uint8_t>(crc >> (8 * i));
}

void
push_range(
    DiarizationStream& stream, const std::vector<float>& audio, size_t begin, size_t end,
    int sample_rate, size_t chunk_samples) {
    for (size_t offset = begin; offset < end;) {
        const size_t count = std::min(chunk_samples, end - offset);
        stream.push(audio.data() + offset, count, sample_rate);
        offset += count;
    }
}

struct Results {
    int64_t frames = 0;
    int64_t probs_start = 0;
    std::vector<float> probabilities;
    std::vector<DiarSegment> segments;
};

Results
collect(DiarizationStream& stream) {
    Results r;
    r.frames = stream.frame_count();
    r.probs_start = stream.frame_probs_start();
    r.probabilities = stream.frame_probabilities();
    r.segments = stream.segments();
    return r;
}

void
compare(const Results& a, const Results& b, const char* context) {
    require(a.frames == b.frames, "restored frame count differs");
    require(a.probs_start == b.probs_start, "restored probability base differs");
    require(a.probabilities.size() == b.probabilities.size(), "restored probability shape differs");
    if (!a.probabilities.empty())
        require(
            std::memcmp(a.probabilities.data(), b.probabilities.data(),
                        a.probabilities.size() * sizeof(float)) == 0,
            "restored float32 probabilities are not bit-identical");
    require(a.segments.size() == b.segments.size(), "restored segment count differs");
    for (size_t i = 0; i < a.segments.size(); ++i)
        require(a.segments[i].t0 == b.segments[i].t0 && a.segments[i].t1 == b.segments[i].t1 &&
                    a.segments[i].speaker == b.segments[i].speaker,
                "restored segments differ");
    std::fprintf(stderr, "[snapshot] %s: %lld frames, %zu exact probabilities, %zu segments\n",
                 context, static_cast<long long>(a.frames), a.probabilities.size(), a.segments.size());
}

std::vector<uint8_t>
timed_export(DiarizationStream& stream, const char* stage) {
    const auto start = std::chrono::steady_clock::now();
    auto snapshot = stream.export_state();
    const auto elapsed = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    std::fprintf(stderr, "[snapshot] %s: %zu bytes, save %.2f ms\n", stage, snapshot.size(), elapsed);
    return snapshot;
}

void
timed_import(DiarizationStream& stream, const std::vector<uint8_t>& snapshot, const char* stage) {
    const auto start = std::chrono::steady_clock::now();
    stream.import_state(snapshot.data(), snapshot.size());
    const auto elapsed = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    std::fprintf(stderr, "[snapshot] %s: load %.2f ms\n", stage, elapsed);
}

void
test_empty_and_rejections(const std::shared_ptr<Diarizer>& diarizer) {
    auto source = diarizer->streaming_diarize();
    const auto empty = source->export_state();
    require(empty.size() >= 108, "empty snapshot is missing its header");
    std::fprintf(stderr, "[snapshot] empty state: %zu bytes\n", empty.size());

    auto target = diarizer->streaming_diarize();
    auto bad = empty;
    bad[8] ^= 1;  // format version
    require(rejects_import(*target, bad), "unsupported format version was accepted");
    bad = empty;
    bad.pop_back();
    require(rejects_import(*target, bad), "truncated snapshot was accepted");
    bad = empty;
    bad.back() ^= 1;
    require(rejects_import(*target, bad), "corrupt payload was accepted");
    bad = empty;
    bad[16] ^= 1;  // model fingerprint
    require(rejects_import(*target, bad), "incompatible model fingerprint was accepted");
    bad = empty;
    bad[72] ^= 1;  // first streaming geometry field
    require(rejects_import(*target, bad), "incompatible streaming geometry was accepted");
    bad = empty;
    bad[117] = 0xff;  // first float-array count; keep the CRC valid
    for (size_t i = 118; i < 125; ++i) bad[i] = 0xff;
    refresh_payload_crc(bad);
    require(rejects_import(*target, bad), "oversized snapshot array was accepted");
    bad = empty;
    bad[108] = 0xff;  // unsupported input rate
    bad[109] = 0xff;
    bad[110] = 0xff;
    bad[111] = 0x7f;
    refresh_payload_crc(bad);
    require(rejects_import(*target, bad), "invalid snapshot state value was accepted");
    require(target->frame_count() == 0, "failed imports changed the target stream");

    timed_import(*target, empty, "empty state");
    require(target->export_state() == empty, "empty save/restore/save changed bytes");
    require(rejects_import(*target, empty), "active stream accepted a second import");
}

void
test_first_processed_chunk(
    const std::shared_ptr<Diarizer>& diarizer, const std::vector<float>& audio) {
    constexpr size_t kChunk = 160 * 16;
    const size_t split = std::min<size_t>(audio.size() - 1, 1800 * 16);
    auto reference = diarizer->streaming_diarize();
    push_range(*reference, audio, 0, split, 16000, kChunk);
    push_range(*reference, audio, split, audio.size(), 16000, kChunk);
    reference->finish();
    const Results expected = collect(*reference);

    auto restored = diarizer->streaming_diarize();
    push_range(*restored, audio, 0, split, 16000, kChunk);
    require(restored->frame_count() > 0, "first-chunk checkpoint has not run inference");
    const auto state = timed_export(*restored, "after first processed chunk");
    restored.reset();
    restored = diarizer->streaming_diarize();
    timed_import(*restored, state, "after first processed chunk");
    push_range(*restored, audio, split, audio.size(), 16000, kChunk);
    restored->finish();
    compare(expected, collect(*restored), "first processed chunk");
}

void
test_silence_and_compaction(
    const std::string& model_path, const std::vector<float>& audio) {
    std::vector<float> extended = audio;
    extended.insert(extended.end(), 5 * 16000, 0.f);
    const size_t split = std::min(extended.size() - 1, audio.size() + 4500 * 16);
    constexpr size_t kChunk = 160 * 16;
    auto diarizer = Diarizer::load(-1, model_path);

    auto reference = diarizer->streaming_diarize();
    reference->set_compaction(500, 200);
    push_range(*reference, extended, 0, split, 16000, kChunk);
    push_range(*reference, extended, split, extended.size(), 16000, kChunk);
    reference->finish();
    const Results expected = collect(*reference);

    auto restored = diarizer->streaming_diarize();
    restored->set_compaction(500, 200);
    push_range(*restored, extended, 0, split, 16000, kChunk);
    require(restored->frame_probs_start() > 0, "compaction checkpoint did not freeze a prefix");
    const auto state = timed_export(*restored, "compacted prefix and long-silence tail");
    restored.reset();
    restored = diarizer->streaming_diarize();
    timed_import(*restored, state, "compacted prefix and long-silence tail");
    push_range(*restored, extended, split, extended.size(), 16000, kChunk);
    restored->finish();
    compare(expected, collect(*restored), "compaction with frozen segments");
}

void
test_speaker_changes(
    const std::string& model_path, const std::string& ami_path) {
    std::vector<float> ami;
    int sample_rate = 0;
    if (!read_wav_mono_16k(ami_path, ami, sample_rate) || sample_rate != 16000)
        throw std::runtime_error("snapshot speaker-change test expects 16 kHz AMI audio");
    const size_t begin = 48 * 16000 + 300 * 16;
    const size_t end = std::min(ami.size(), size_t{57 * 16000 + 500 * 16});
    require(end > begin, "AMI speaker-change crop is unavailable");
    std::vector<float> crop(ami.begin() + begin, ami.begin() + end);
    constexpr size_t kChunk = 160 * 16;
    const size_t split = crop.size() / 2 + 731;
    auto diarizer = Diarizer::load(-1, model_path);

    auto reference = diarizer->streaming_diarize();
    push_range(*reference, crop, 0, split, 16000, kChunk);
    push_range(*reference, crop, split, crop.size(), 16000, kChunk);
    reference->finish();
    const Results expected = collect(*reference);

    auto restored = diarizer->streaming_diarize();
    push_range(*restored, crop, 0, split, 16000, kChunk);
    const auto state = timed_export(*restored, "AMI speaker-change crop");
    restored.reset();
    restored = diarizer->streaming_diarize();
    timed_import(*restored, state, "AMI speaker-change crop");
    push_range(*restored, crop, split, crop.size(), 16000, kChunk);
    restored->finish();
    const Results actual = collect(*restored);
    compare(expected, actual, "AMI speaker changes");
    std::set<int> channels;
    for (const auto& segment : expected.segments) channels.insert(segment.speaker);
    require(channels.size() >= 2, "AMI crop did not exercise multiple speaker channels");
}

std::vector<float>
make_48k_test_audio(const std::vector<float>& source) {
    const size_t use = std::min<size_t>(source.size(), 2 * 16000);
    std::vector<float> result;
    result.reserve(use * 3);
    for (size_t i = 0; i < use; ++i)
        result.insert(result.end(), 3, source[i]);
    return result;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <sortformer.gguf> <jfk.wav> <ami.wav>\n", argv[0]);
        return 2;
    }
    try {
        std::vector<float> audio;
        int sample_rate = 0;
        if (!read_wav_mono_16k(argv[2], audio, sample_rate) || sample_rate != 16000)
            throw std::runtime_error("snapshot test expects a readable 16 kHz mono wav");

        // Small valid geometry forces repeated AOSC FIFO updates and cache
        // compression on the short JFK fixture, keeping this acceptance test
        // bounded while exercising the real Sortformer V3 pipeline.
        DiarGeometry geometry{32, 8, 13, 4, 0, 1};
        auto diarizer = Diarizer::load(-1, argv[1], geometry);
        test_empty_and_rejections(diarizer);
        test_first_processed_chunk(diarizer, audio);

        const size_t split1 = std::min<size_t>(audio.size() - 1, 5 * 16000 + 1973);
        const size_t split2 = std::min<size_t>(audio.size() - 1, 8 * 16000 + 613);
        require(split1 > 0 && split2 > split1, "audio fixture is too short");
        constexpr size_t kChunk16k = 160 * 16;

        // Reference run with the same API call boundaries used by the
        // restored path.
        auto reference_model = Diarizer::load(-1, argv[1], geometry);
        auto reference = reference_model->streaming_diarize();
        push_range(*reference, audio, 0, split1, 16000, kChunk16k);
        push_range(*reference, audio, split1, split2, 16000, kChunk16k);
        push_range(*reference, audio, split2, audio.size(), 16000, kChunk16k);
        reference->finish();
        const Results expected = collect(*reference);
        reference.reset();
        reference_model.reset();

        // First checkpoint is deliberately after unaligned 160 ms pushes and
        // after AOSC compression. The model and stream are both destroyed.
        auto part1_model = Diarizer::load(-1, argv[1], geometry);
        auto part1 = part1_model->streaming_diarize();
        push_range(*part1, audio, 0, split1, 16000, kChunk16k);
        const auto state1 = timed_export(*part1, "after AOSC updates / partial audio");
        part1.reset();
        part1_model.reset();

        // Restore, continue part two, save again, destroy everything, restore
        // once more and finish the tail.
        auto part2_model = Diarizer::load(-1, argv[1], geometry);
        auto part2 = part2_model->streaming_diarize();
        timed_import(*part2, state1, "cross-model reload");
        push_range(*part2, audio, split1, split2, 16000, kChunk16k);
        const auto state2 = timed_export(*part2, "repeated checkpoint");
        part2.reset();
        part2_model.reset();

        auto part3_model = Diarizer::load(-1, argv[1], geometry);
        auto part3 = part3_model->streaming_diarize();
        timed_import(*part3, state2, "second cross-model reload");
        push_range(*part3, audio, split2, audio.size(), 16000, kChunk16k);
        part3->finish();
        compare(expected, collect(*part3), "160 ms push / AOSC / repeated restore");

        // Exercise the stateful 48 kHz resampler. Here the model remains
        // alive; the prior scenario already proved model destruction/reload.
        const auto audio48 = make_48k_test_audio(audio);
        const size_t split48 = std::min<size_t>(audio48.size() - 1, 48000 + 941);
        constexpr size_t kChunk48k = 160 * 48;
        auto resample_reference = diarizer->streaming_diarize();
        push_range(*resample_reference, audio48, 0, split48, 48000, kChunk48k);
        push_range(*resample_reference, audio48, split48, audio48.size(), 48000, kChunk48k);
        resample_reference->finish();
        const Results expected48 = collect(*resample_reference);

        auto resample_restored = diarizer->streaming_diarize();
        push_range(*resample_restored, audio48, 0, split48, 48000, kChunk48k);
        const auto resample_state = timed_export(*resample_restored, "active 48 kHz resampler");
        resample_restored.reset();
        resample_restored = diarizer->streaming_diarize();
        timed_import(*resample_restored, resample_state, "active 48 kHz resampler");
        push_range(*resample_restored, audio48, split48, audio48.size(), 48000, kChunk48k);
        resample_restored->finish();
        compare(expected48, collect(*resample_restored), "48 kHz resampling");

        test_silence_and_compaction(argv[1], audio);
        test_speaker_changes(argv[1], argv[3]);

        std::fprintf(stderr, "[snapshot] all CPU continuity checks passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[snapshot] failure: %s\n", e.what());
        return 1;
    }
}
