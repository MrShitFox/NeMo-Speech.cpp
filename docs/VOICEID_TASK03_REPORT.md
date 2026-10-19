# Task 03: Nemotron 3 only cleanup

Branch: `refactor/nemotron3-only`
Base: Task 02 commit `5a916e0`
Model used for checks: local `Nemotron-3-Diarization.q8_0.gguf`, SHA-256
`08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`.

## Result

The repository now builds a standalone NVIDIA Nemotron 3 Diarization
(Sortformer V3) engine, its public C API, and a minimal `nemo-speech diarize`
CLI. The V3 CPU inference and streaming outputs match the saved Task 02
baseline. No accuracy claim against reference labels is made.

The C API retains all `nemo_speech_diar_*` exports and the historical
`nemo_speech_asr_last_error` and `nemo_speech_asr_version` symbols used for
status/error compatibility. No ASR implementation or ASR API is built.

## Dependency audit

The final build graph is:

```text
nemo-speech CLI ───────────────┐
                               v
public C API -> Diarizer -> Sortformer V3 graph -> ggml/GGUF
                    |              |
                    |              +-> RoPE transformer / attention
                    +-> streaming AOSC state and speaker cache
                    +-> mel frontend -> ggml FFT / filterbank operations
                    +-> audio I/O and resampling
```

The CMake and linker closures contain ggml, the diarization runtime, audio
frontend, audio/resampling support, and platform C/C++ runtime libraries. The
CPU build also uses the ggml OpenMP runtime. CUDA alone conditionally finds
`CUDAToolkit`; Vulkan and Metal options are passed to ggml when selected.
There is no SentencePiece, protobuf/gRPC, HTTP/WebSocket, ASR, TTS, NMT, Python,
or model-download dependency.

| Classification | Kept files/components and reason |
|---|---|
| Required | `src/diar/`: Sortformer V3, RoPE encoder, diarizer, streaming pipeline, AOSC, C API. These are the model graph, output lifecycle, and speaker state. |
| Required | `src/audio_frontend/fe.*`: NeMo mel preprocessing used by V3. Moved from `src/asr/features/` without changing the calculations. |
| Required | `src/runtime/ggml/`: GGUF loader, tensors, sessions, backend selection, logging, and the V3-used Linear, LayerNorm, and Conv1D wrappers. |
| Shared | Audio file loading, resampling, batching, JSON, parameter parsing, C ABI error/version helpers, NVTX, and ggml log filtering. Kept or moved to `src/common/` because the CLI, frontend, diarizer, or C API uses them. |
| Removed | ASR recognition, decoders, VAD, punctuation, normalization, subtitles, ASR HTTP routes, and model registry/download code. No final target includes them. |
| Removed | TTS, NMT, S2S/VoiceChat, Riva/gRPC, and their source, dependencies, configs, tests, examples, clients, and build targets. |

Sortformer V2 model loading now fails explicitly: the GGUF parser accepts only
`sortformer.version=v3`. The old FastConformer, relative-position attention,
and V2 transformer encoder sources and target are gone. The V3 graph is built
only from feature stacking, the RoPE encoder, projection, subpixel upsampling,
and the speaker head. The V3 inference and output code remains equivalent to
the baseline, as confirmed by model-backed comparisons below.

`aosc_state.cpp` and `aosc_state.h` were moved from the ASR directory with
their contents unchanged. The streaming pipeline implementation was also
preserved; only its header comments now describe the standalone V3 role. Some
legacy V2 geometry and channel-birth helper code remains in that preserved
state implementation, but V3 model loading always reports V3 and does not
enter those legacy branches. It is retained to avoid changing the streaming
state machine that is the subject of future snapshot/restore work.

The runtime audit also removed unused generic `Conv2D`, depthwise `Conv2D`,
`BatchNorm1d`, `ReLU`, and `SequenceModule` wrappers and the unused CTC
pointwise/depthwise branches from the generic `Conv1D` wrapper. V3's regular
F16 Conv1D, Linear, and LayerNorm paths remain; the post-cleanup model-backed
parity checks passed.

## Removed and relocated files

Removed the old ASR, TTS, NMT, S2S, server, service, conversion, and model
catalog trees, including their CMake files, endpoints, clients, examples,
tests, install/release scripts, configs, and docs. Removed the unneeded
SentencePiece, gRPC/protobuf, cpp-httplib, KenLM, Flashlight, Open JTalk, and
CppJieba submodules and dependencies. `.gitmodules` now contains only the
`llama.cpp` source needed to build ggml.

Moved the shared mel frontend to `src/audio_frontend/`, generic batching to
`src/common/`, and V3 model, RoPE, pipeline, state, and public C API sources to
`src/diar/`. The JFK fixture moved to `test_files/diar/`; AMI fixtures and
their license/attribution notes remain there.

Removed the unused fused-attention/NanoCodec ggml patches and llama.cpp-only
patches. The remaining 14 ggml patches still apply to the pinned submodule and
round-trip through `scripts/llama-patches.sh check`. CPU/CUDA patches used by
ggml operations are retained. `.gitattributes`, pre-commit rules, and CI no
longer refer to removed LFS data, proto trees, installers, or model smoke
scripts.

The old HTTP and WebSocket service was coupled to ASR engine registries and
VoiceChat session handling, so its transport is not reused. For a future
VoiceID REST API, add a separate optional server executable around the public
`nemo_speech_diar_*` C API and keep HTTP request/session ownership outside the
inference library. No network library is added now.

## Targets and configuration

The only configure/build presets are `cpu-diar`, `vulkan-diar`, `cuda-diar`,
and `metal-diar`. They build these project targets:

- `nemo_speech_runtime_ggml`, `nemo_speech_common`, and
  `nemo_speech_c_api_common` (internal static libraries);
- `nemo_speech_audio`, `nemo_speech_audio_frontend`, `nemo_speech_diar`, and
  `nemo_speech_diar_c` (runtime libraries);
- `nemo_speech_cli`, whose executable is `nemo-speech`;
- ggml base/CPU and the selected accelerator backend.

`BUILD_TESTING` defaults to `OFF`. The public headers, CMake package, runtime
libraries, CLI, docs, example YAML, and license notices install successfully.
An external CMake consumer found `NeMoSpeech::Diarization`, linked the
installed C API, loaded the local Q8_0 model, and completed the C API smoke
check.

## Build and test results

The required clean CPU build used:

```bash
scripts/configure.sh cpu-diar
cmake --build --preset cpu-diar
```

It completed 63 Ninja actions in 27.4 seconds with compiler caching disabled.
With `BUILD_TESTING=ON`, the full available CTest set passed 4/4:

| Test | Result |
|---|---|
| `audio_resampler` | Passed |
| `diar_state` | Passed |
| `mel_features` | Passed |
| `diar_c_api_model` | Passed; loads Q8_0 through the public C API, checks 8 speakers and 10 ms frame cadence, and opens/closes a stream |

Model-backed results were compared after normalizing to the `segments` array,
which excludes the changed JFK fixture path:

| Scenario | Result vs Task 02 |
|---|---|
| JFK streaming | Exact match: 3 segments, speaker 1 only. Normalized SHA-256 `0afa4bf606dab6e91371ba3c3b8015605baa526efe20135877a82911c2f03812`. |
| JFK offline | Exact segment and speaker match. |
| AMI streaming | Exact match: 25 segments and speaker slots 1, 2, and 3. Normalized SHA-256 `6fac6b24ca2af9913e02de7325af752f8ae206def50f7abf4a4005cb56c3497f`. |
| Silence | Exact match: zero segments. |
| AMI with 160 ms pushes | RTTM output byte-for-byte matches the Task 02 RTTM. |
| Repeat JFK inference | Its normalized segments match both Task 02 and the first Task 03 run exactly. |

The parity executable `test_sortformer_parity` built, but its NeMo reference
tensor dumps are not present in the checkout, so it was not run. These checks
establish functional regression parity; they do not measure DER or accuracy
against reference labels.

The exported C API symbols were inspected with `nm -D`: all diarization
symbols, plus the two legacy ASR-prefixed error/version symbols, are present.
The clean library's exported symbol set matches the prior standalone C API
set. `ldd` shows no ASR, SentencePiece, TTS, NMT, Python, or network library
dependency. The ignored local model is not tracked by Git.

## Backend checks

The host is Linux without accelerator SDKs. All three non-CPU presets were
attempted at CMake configure time, and none is reported as built:

- Vulkan stopped at ggml's dependency check: Vulkan library/headers and
  `glslc` are missing.
- CUDA stopped because `nvcc`/`CUDAToolkit` is unavailable.
- Metal stopped because the Linux host does not provide Apple's Foundation
  framework.

The presets enable the backend options and preserve the Task 02 patched-ggml
selection: CPU and CUDA use the patch series; Vulkan and Metal use upstream
ggml. CUDA architecture remains configurable with
`-DCMAKE_CUDA_ARCHITECTURES=...`. CI now contains native Linux CPU/Vulkan,
macOS Metal, and CUDA runner builds, but those remote jobs have not been run
from this checkout.

## Size and build comparison

Like-for-like CPU Release numbers from Task 02 and this clean build:

| Metric | Task 02 | Task 03 |
|---|---:|---:|
| Core C/C++ compile commands, tests excluded | 58 | 48 |
| Clean Ninja actions | 75 | 63 |
| Clean build time | 32.3 s | 27.5 s |
| Diarization + C API + shared audio/frontend libraries | 1,197,080 B | 794,368 B |
| `nemo-speech` executable | 667,968 B | 176,184 B |

The library row excludes ggml and the CLI. Task 02 included the shared encoder
library; Task 03 uses the V3 RoPE source directly in `nemo_speech_diar` and no
longer builds that encoder target. The binary reduction is expected because
the old runtime and CLI implementations were physically removed; inference
output remained identical.

The checked-out project source footprint, excluding `.git`, build output,
`.deps`, model files, and the `llama.cpp` submodule, fell from 7,155,870 bytes
at `5a916e0` to 3,129,248 bytes after cleanup, a 56.3% reduction. Git reports
365 deleted paths and 98,569 deleted lines before this report was added; 23
other paths were relocations rather than content removals. These counts include
deleted docs, fixtures, and build infrastructure as well as source code.

## Remaining constraints and AOSC follow-up

Further deletion is limited by shared ggml execution, the NeMo mel frontend,
audio I/O/resampling, batching, the C ABI compatibility names, and the AOSC
streaming state. The remaining AOSC/pipeline V2-compatible helpers are
unreachable for a loaded V3 model; their state implementation was intentionally
left intact. The baseline state tests pass and `aosc_state.cpp/.h` contents
match the Task 02 versions.

This leaves the repository ready to study AOSC snapshot/restore boundaries:
the model handle, `DiarizationStream`, AOSC speaker cache/FIFO, frontend
buffers, and retained probability/segment timelines remain separate concepts.
There is no snapshot/restore API yet; Task 03 did not modify that state or
start its implementation.
