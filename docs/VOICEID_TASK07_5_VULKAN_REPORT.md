# VoiceID Task 07.5 — Vulkan Build, Correctness & Performance Report

**Decision: A — Vulkan Ready.** Physical Vulkan inference succeeded on the RTX 4060, the scheduler placed the captured Sortformer graph on Vulkan with no CPU fallback, the tested speaker assignments and segments stayed stable, and GPU throughput was materially higher than the CPU baseline. For Task 08, start with `--device vulkan --gpu-device 0 --concurrency 4` and batching enabled.

This validation uses the Task 07 model and fixtures only. It does not run a VCTK mass benchmark.

## Host and toolchain

| Item | Observed value |
|---|---|
| OS | NixOS 26.05.11576.7c8764b7c7b0 (Yarara), Linux 6.18.55, x86_64 |
| CPU | AMD Ryzen 5 1600, 6 cores / 12 threads |
| GPU | NVIDIA GeForce RTX 4060, 8,188 MiB reported VRAM |
| NVIDIA driver | 595.71.05; Vulkan device API 1.4.329 |
| Vulkan loader / instance | 1.4.357 |
| Shader tools | `glslc` from shaderc 2026.1; SPIR-V Tools 1.4.357; SPIR-V headers 1.4.357 |
| CMake / Ninja | 4.1.6 / 1.13.2 |
| Compilers | GCC 15.3.0 / Clang 21.1.8 |
| FFmpeg | 6.1.6 |

Model: `models/Nemotron-3-Diarization.q8_0.gguf`, SHA256 `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1` (the Task 07 model).

`vulkaninfo --summary` exposed the physical NVIDIA GPU and a Mesa llvmpipe software Vulkan device. GPU test runs set `VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json`; reported measurements therefore use the physical RTX 4060, not llvmpipe. GGML CPU fallback is reported separately below.

The standard system environment did not expose `vulkaninfo`, `glslc`, or Vulkan development headers. They were supplied in temporary Nix shells; no system packages or GPU drivers were replaced. The Vulkan preset configured with the existing `GGML_VULKAN=ON` and `NEMO_SPEECH_GGML_PATCHED=OFF` settings. On this Nix host, ggml found `SPIRV-Headers` but did not pass its include directory to compilation, so configuration also supplied `-DCMAKE_CXX_FLAGS=-isystem/nix/store/kxqs59gy03xq59b636wszvfz9y8imdb4-spirv-headers-1.4.357.0/include`. No GGML backend rewrite was needed. The runtime continues to default `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` for the persistent-cache tensor path.

Build commands used:

```bash
git submodule update --init llama.cpp
scripts/configure.sh cpu-diar -DBUILD_TESTING=ON
cmake --build --preset cpu-diar --parallel 4

# Run inside a temporary Nix shell containing vulkan-tools, shaderc,
# vulkan-headers, vulkan-loader, and spirv-headers.
scripts/configure.sh vulkan-diar -DBUILD_TESTING=ON \
  -DCMAKE_CXX_FLAGS=-isystem/nix/store/kxqs59gy03xq59b636wszvfz9y8imdb4-spirv-headers-1.4.357.0/include
cmake --build --preset vulkan-diar --parallel 4
```

Both builds compiled and linked. The Vulkan executable is dynamically linked: `nemo-speech` needs `libggml-vulkan.so.0`, which needs `libvulkan.so.1`. `ldd` resolved both through the build output and Nix loader path; no shared libraries were missing.

## GPU execution and backend selection

The explicit CLI invocation completed JFK diarization with `--device vulkan`. The existing schedule diagnostic reported:

```text
cached graphs: 11
Vulkan0: 9559 nodes
RESHAPE, MUL_MAT, NORM, ROPE, FLASH_ATTN_EXT, ... -> Vulkan0
CPU-fallback ops (none)
```

The runtime selected registry backend `Vulkan`, device `Vulkan0`, corresponding to the pinned NVIDIA GPU. Major matrix, normalization, RoPE, and attention operations ran there. This confirms physical Vulkan execution and compatible model tensor allocation; compiling with `GGML_VULKAN=ON` alone was not treated as proof.

The CLI and both research probes accept explicit `--device cpu|vulkan`; Vulkan also accepts `--gpu-device N`. Backend selection filters by the GGML registry name `Vulkan` before applying the index, so an explicit Vulkan request cannot select an unrelated GPU backend. Probe reports record the requested backend, actual backend, device name, and index. CPU remains the default. Negative checks against CPU-only builds failed clearly with “requested GPU backend 'Vulkan' has no matching device” instead of falling back.

The native matching probe ran on Vulkan with `Vulkan0`. The Task 06 representation probe ran the same small `encoder_proj` fixture on CPU and Vulkan; its built-in regression passed on both (2 clips, 4,408 raw probability values, 551 identical channel-assignment frames, 2 segment cases, 2 AOSC-profile cases, and 2 snapshot cases per report). The reports identify CPU versus Vulkan0 explicitly. A Vulkan-built test suite still runs its normal CTest paths on CPU unless a test explicitly selects Vulkan.

## CPU/Vulkan parity

All direct comparisons used the same GGUF, exact same WAV bytes, 16 kHz input, streaming geometry, 160 ms pushes, and chunk boundaries. The saved summary incorrectly described the raw float comparison files as four channels per frame; that reshape is corrected below. These checks compare numerical outputs without requiring bit-exact CPU/GPU equality.

Acceptance checks used for interpretation:

- Per-element MAE at most 0.001 and RMSE at most 0.004.
- Top-1 channel differences at most 0.5%; each differing frame must have a CPU top-1 margin no greater than twice that frame's maximum CPU/Vulkan channel error.
- Same segment count and speaker IDs, with boundary differences no greater than one 10 ms output frame.
- Record and inspect maximum absolute and relative errors, plus crossings of the default onset/offset thresholds (0.641 / 0.561); these are not waived by the aggregate criteria.

| Input | Frames | Max abs | MAE | RMSE | Max relative¹ | Top-1 channel differences | Segments / speaker IDs | Max boundary shift |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| JFK, one speaker | 1,101 | 0.014664 | 0.000111 | 0.000712 | 0.1044 | 0 / 0% | 3 / same | 0 ms |
| AMI, speaker changes and overlap | 6,001 | 0.062910 | 0.000295 | 0.001961 | 0.5841 | 30 / 0.4999% | 25 / same | 10 ms |
| Silence, 8 s | 801 | 0.000178 | 0.0000068 | 0.000020 | 0.0123 | 0 / 0% | 0 / same | 0 ms |
| VCTK p225 | 206 | 0.003741 | 0.0000369 | 0.000223 | 0.0349 | 0 / 0% | 1 / same | 0 ms |
| VCTK p226 | 280 | 0.019296 | 0.0000625 | 0.000797 | 0.0799 | 0 / 0% | 1 / same | 0 ms |
| VCTK p227 | 343 | 0.018788 | 0.000238 | 0.001279 | 0.1074 | 0 / 0% | 1 / same | 0 ms |

¹ Maximum relative error is computed only where the CPU probability is greater than 0.01.

All 30 AMI top-1 differences were explainable by local numerical error at near-tie frames; there were no top-1 changes on the three VCTK clips after using the correct channel grouping. The largest AMI absolute error occurred at frame 614 (6.14 s), channel 2: CPU 0.73696, Vulkan 0.67405; both kept channel 2 as top-1 and above the onset threshold. Across AMI, 11 onset and 9 offset channel values crossed a threshold; the resulting segment count and speaker IDs remained the same, with at most a 10 ms boundary change. JFK had one offset crossing; p227 had one onset crossing. No meaningful speaker assignment change was found.

### Output geometry discrepancy correction

The loaded GGUF metadata contains `sortformer.num_speakers = 8`,
`sortformer.output_subsampling_factor = 1`, and
`sortformer.preprocessor.window_stride = 0.0099999998` seconds. The loaded
runtime reports `cfg.num_speakers = 8` and
`seconds_per_output_frame() = 0.01`. Its prediction tensor is ordered
`[channels, native_frames, batch]`; for the single-item 60 s AMI run this is
`[8, 6001, 1]`. The C++ probability vector and the CLI's headerless float32
dump are frame-major, `[frame, channel]`, so the dump has 48,008 values
(`6001 × 8`, 192,032 bytes). Top-1 scans index each frame as
`probabilities[frame * 8 + channel]` over channels 0–7.

Task 07.5's comparison script reshaped those same values as 12,002 frames × 4
channels. The runtime, trained output dimensions, C++ vector, and segmenter
were correct; this was a reporting/export interpretation error. We reran the
CPU/Vulkan fixture comparisons with eight channels per 10 ms frame. The
element-wise max error, MAE, RMSE, relative error, and per-channel threshold
crossings did not change because they compare the same float values. The
shape-dependent top-1 result changed to 30/6,001 AMI frames (0.4999%) and
0/206 p225 frames. The 30 AMI changes are all locally explained by numerical
error, and the existing 0.5% gate still passes, narrowly. The binary file
format used in Task 07.5 did not carry its dimensions, which allowed the
mistake; Task 08 writes dimensions and layout with every probability trace.

The runtime now also validates that the prediction tensor's channel dimension
matches `sortformer.num_speakers` before copying outputs. This is a fail-fast
shape check and does not alter model outputs.

The RTTM output was also compared: segment count and speaker IDs matched, with at most the one-frame boundary change shown above. The AOSC-only warm-start comparison used the Task 07 AMI fixture and the same imported CPU profile on CPU and Vulkan. The three query assignments were identical: P1 known → channel 0 for 261 speech frames, P2 overlap → channel 2 for 116 frames, and unknown P3 → channel 1 for 612 frames. Top-1 stability was 1.0 in each case. For the overlap query, raw probability max-absolute error was 0.03201, MAE 0.000888, RMSE 0.003842; all selected frames still assigned to channel 2.

## Streaming snapshots and AOSC profiles

The explicit Vulkan snapshot test destroyed and recreated the Vulkan engine between export and restore. Same-backend continuation matched uninterrupted Vulkan probabilities exactly, including repeated restore and the following cases:

| Case | Frames compared | Probabilities compared | Result |
|---|---:|---:|---|
| 160 ms pushes after AOSC updates / partial audio | 1,101 | 8,808 | Exact; 3 segments |
| 48 kHz input resampling | 201 | 1,608 | Exact; 1 segment |
| Silence and streaming compaction checkpoint | 1,601 | 3,208 | Exact; 3 segments |
| AMI speaker-change crop | 921 | 7,368 | Exact; 4 segments |

CPU enrollment exported the 551,305-byte AOSC profile with CRC32 `334994840`, matching the existing Task 07 result. Vulkan imported that CPU-generated profile and reported the same CRC, mapping, and query channel assignments. Vulkan also exported and re-imported its own 551,305-byte profile (CRC32 `1139101250`). CPU and Vulkan fresh exports have different CRCs because the stored floating-point state differs numerically; both profiles retain the same serialized size and structure. The immutable-profile and unknown-traffic checks passed on both backends.

## Single-stream performance

Each case used one loaded model and three measured warm runs after the first inference. Model initialization and first inference are shown separately. Warm throughput is audio seconds divided by median warm inference time; standard RTF is its reciprocal (`wall seconds / audio seconds`). Model initialization is excluded from warm RTF. RSS is process maximum resident memory.

| Input | Backend | Model init | First inference | Warm median (range) | RTF | Audio / wall second | Max RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| JFK, 11 s | CPU, 4 threads | 579 ms | 2,657 ms | 2,452 ms (2,449–2,650) | 0.223 | 4.49× | 128 MiB |
| JFK, 11 s | Vulkan0 | 741 ms | 147 ms | 69 ms (66–101) | 0.00623 | 160.5× | 118 MiB |
| AMI, 20 s | CPU, 4 threads | 562 ms | 7,851 ms | 7,776 ms (7,756–8,163) | 0.389 | 2.57× | 143 MiB |
| AMI, 20 s | Vulkan0 | 701 ms | 267 ms | 139 ms (136–190) | 0.00693 | 144.3× | 126 MiB |
| AMI, 60 s | CPU, 4 threads | 591 ms | 46,034 ms | 45,475 ms (45,221–45,593) | 0.758 | 1.32× | 198 MiB |
| AMI, 60 s | Vulkan0 | 716 ms | 770 ms | 684 ms (672–687) | 0.0114 | 87.7× | 147 MiB |

Warm repeat variability is shown as the observed min–max, rather than a precise speedup claim from one run. A separate cold schedule-capture launch had higher setup cost and is not mixed into the warm table.

## Multi-stream throughput and batching

The workload contained eight short WAV inputs from three converted VCTK speakers (21.408 s total audio). The source clips were repeated to fill the queue; this is a concurrency smoke workload, not a corpus benchmark. Each table row has three fresh-process runs. Total wall time includes model initialization and output handling. Per-file latency is the average of the three CLI summaries. Batch counts are included where batching was enabled.

| Backend | Concurrency | Batching | Wall median (range) | Audio / wall second | Per-file mean / p95 latency | Batch count / mean size / max |
|---|---:|---|---:|---:|---:|---|
| CPU | 1 | Off | 2.40 s (2.34–2.50) | 8.92× | 228 / 388 ms | — |
| CPU | 2 | Off | 2.56 s (2.54–2.62) | 8.36× | 466 / 775 ms | — |
| CPU | 4 | Off | 2.72 s (2.71–2.80) | 7.87× | 923 / 1,545 ms | — |
| CPU | 8 | Off | 2.76 s (2.72–2.79) | 7.76× | 1,337 / 2,153 ms | — |
| Vulkan | 1 | Off | 0.98 s (0.94–1.52) | 21.84× | 41 / 203 ms | — |
| Vulkan | 2 | Off | 0.96 s (0.95–0.98) | 22.30× | 36 / 59 ms | — |
| Vulkan | 4 | Off | 0.97 s (0.96–0.99) | 22.07× | 66 / 115 ms | — |
| Vulkan | 8 | Off | 0.98 s (0.95–1.12) | 21.84× | 97 / 132 ms | — |
| Vulkan | 2 | On | 1.03 s (1.00–1.07) | 20.78× | 40 / 57 ms | 22 / 1.05 / 2 |
| Vulkan | 4 | On | 0.94 s (0.93–0.94) | 22.77× | 55 / 78 ms | 16 / 1.44 / 4 |
| Vulkan | 8 | On | 0.92 s (0.90–1.72) | 23.27× | 137 / 340 ms | 8, 6, 6 / 3.45 overall / 8 |

The concurrency-8 batched row has one 1.72 s outlier; its median is not a stable recommendation. CPU throughput decreased as concurrency increased on this host. Vulkan handled the short-stream queue about 2.5× faster than the best CPU row. Batching at concurrency 4 was a small, repeatable gain over Vulkan without batching; for Task 08, concurrency 4 with batching enabled is the best-supported starting setting from this sample.

`nvidia-smi` sampled whole-device utilization and memory, not process-specific counters. During the single-stream sample, device utilization peaked at 71% and total used VRAM peaked near 1.99 GiB. During the multi-stream sweep, total used VRAM ranged from 1.80 to 1.96 GiB and sampled utilization from 4% to 47%. Desktop/background use is included; process-attributed VRAM is unavailable. Single-stream process RSS ranged from 128 to 198 MiB on CPU and 118 to 147 MiB on Vulkan; a cold schedule-capture probe peaked near 300 MiB RSS.

## Repeated native AOSC inference

The native matching benchmark imported the same CPU-generated 551,305-byte profile and reused one loaded model per backend. It performed one warm-up, then three measured repeats. Each repeat created three fresh streams using the same 11.29 s of AMI query audio and 160 ms pushes. Model initialization is separate; trial wall time includes audio loading and query processing, while inference time covers stream audio submission through completion.

| Backend | Model init | Warm-up inference | Inference median (range) | Trial wall median (range) | Audio / inference second |
|---|---:|---:|---:|---:|---:|
| CPU, 4 threads | 613 ms | 10.592 s | 10.540 s (10.514–10.611) | 10.569 s (10.550–10.640) | 1.07× |
| Vulkan0 | 732 ms | 191 ms | 114 ms (111–137) | 143 ms (139–168) | 93.65× |

These are three repeats on one fixture and one host, not a general speedup guarantee. The imported profile and all three query channel assignments remained stable across the repeated runs.

## VCTK smoke test and dataset protection

The repository contains `VCTK-Corpus-0.92/wav48_silence_trimmed`, with 88,328 FLAC files and no WAV files; the corpus has 111 speaker directories. Three short mono 48 kHz FLAC clips were selected:

| Source | Duration | Output frames | CPU/Vulkan result |
|---|---:|---:|---|
| `p225/p225_001_mic1.flac` | 2.052 s | 206 | 1 segment on both; same ID and boundaries |
| `p226/p226_001_mic1.flac` | 2.799 s | 280 | 1 segment on both; same ID and boundaries |
| `p227/p227_001_mic1.flac` | 3.429 s | 343 | 1 segment on both; same ID and boundaries |

Each FLAC was converted once to a temporary 16 kHz, mono, PCM16 WAV with FFmpeg (`-map_metadata -1 -ac 1 -ar 16000 -c:a pcm_s16le`). The exact same converted file was used for CPU and Vulkan. All three completed on the physical Vulkan device. Converted files and logs stayed under `/tmp`.

The root `.gitignore` now contains `/VCTK-Corpus-0.92/`; `git check-ignore` confirms the dataset and model are ignored, and `git ls-files VCTK-Corpus-0.92` returns no tracked files. No audio, GGUF, profile, snapshot, or large benchmark logs are included in the change.

## Regression checks and limitations

Final CTest results were **6/6 passed** for both `cpu-diar` and `vulkan-diar` builds: `audio_resampler`, `aosc_profile_intervention`, `diar_state`, `mel_features`, `diar_c_api_model`, and `diar_snapshot_continuity`. The Vulkan build's ordinary CTest cases use CPU by default; explicit physical GPU inference, research probes, and snapshot checks were run separately. Normal CPU CLI inference also completed. The Vulkan CLI's CPU-only fallback check rejected an explicit Vulkan request in the CPU-only build.

`test_sortformer_parity` was not reported as passed: its reference tensor data is absent from the repository/test inputs, so that comparison could not be run. This does not block the direct CPU/Vulkan probability comparisons above.

Remaining limits: performance was measured on one Ryzen 5 1600 / RTX 4060 host; the multi-stream sample used short repeated clips; GPU utilization and VRAM were whole-device estimates; the full VCTK workload was intentionally not run. No CUDA, Metal, or other GPU backend is claimed as tested.

## Task 08 recommendation

Use the physical Vulkan device explicitly with `--device vulkan --gpu-device 0`, keep `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`, and start with concurrency 4 and batching enabled. This setting had the most consistent best aggregate result in the tested multi-stream sample. Keep the CPU backend available for reference and fallback runs, and compare outputs when Task 08 introduces longer or more diverse streams.

The compact measurements and metadata are also stored in [`tests/results/voiceid_task07_5_vulkan.json`](../tests/results/voiceid_task07_5_vulkan.json).
