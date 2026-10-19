# Task 02: standalone Nemotron 3 diarization runtime

Branch: `refactor/standalone-diarization`

Baseline: Task 01, source commit `caa3f47c7c75fa2e9e6340272ebad57c1647dd44`; model SHA-256 `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`.

## Result

`NEMO_SPEECH_BUILD_DIAR=ON` with `NEMO_SPEECH_BUILD_ASR=OFF` now produces a standalone diarization build. The ASR implementation target is not created, and its implementation sources are absent from the standalone compile database. No Nemotron computation, Sortformer graph, preprocessing, streaming schedule, or AOSC code was changed.

## Target layout

- `nemo_speech_diar`: Sortformer, AOSC, diarizer/pipeline, RoPE and transformer encoder implementation.
- `nemo_speech_diar_c`: standalone public diarization C ABI library for diar-only builds.
- `nemo_speech_asr`: ASR model, recognizer, runner, decoders, VAD, PnC, postprocessing and ASR-specific encoder support; created only when ASR is enabled. It depends on `nemo_speech_diar` for the existing ASR-integrated speaker sidecar. The reverse dependency does not exist.
- `nemo_speech_audio_frontend`: shared NeMo mel frontend.
- `nemo_speech_encoder`: shared FastConformer and relative-position attention sources needed by ASR and the retained Sortformer V2 path.
- `nemo_speech_audio`: shared audio file I/O and resampling.
- `nemo_speech_common`: JSON and parameter/config helpers.
- `nemo_speech_c_api_common`: shared last-error and version implementation, linked once into the public C ABI library in each build mode.
- GGML/GGUF runtime targets remain shared backend infrastructure.

The C ABI functions moved from `src/asr/c_api.cpp` into `src/asr/diar/c_api.cpp`. Common ABI declarations now live in `include/nemo_speech/common.h`; `asr.h` still includes it. In ASR builds, `nemo_speech_asr_c` continues to contain the ASR and diar API implementations and retains the existing symbol exports. In diar-only ELF builds, `nemo_speech_diar_c` exports the established `nemo_speech_diar_*` functions and the common version/error functions. The old `libnemo_speech_asr_c.so` and SONAME path are compatibility symlinks to the diar C library in that profile; this preserves the old library name for clients that only used diar symbols. It does not provide ASR symbols when ASR is disabled.

The existing `src/asr/c_api.map` export map is reused. Dynamic symbol inspection confirmed the expected diar functions and common error/version functions in the standalone library. The ASR+DIAR library continues to export both existing APIs.

## Compile and dependency isolation

`src/asr/CMakeLists.txt` now uses explicit source lists instead of recursive source globs. In the standalone profile these implementation files are not compiled:

- `src/asr/model.cpp`, `recognizer.cpp`, `runner.cpp`;
- all `src/asr/decoders/`, `src/asr/vad/`, `src/asr/pnc/` and `src/asr/postproc/` implementation files;
- `src/common/audio_decoder.cpp` and `src/common/subtitles.cpp`.

The audio decoder is an ASR-focused test-only target excluded from the default build. Subtitles are configured only with ASR. The regular `cpu-diar` compile database with tests disabled has 58 C/C++ compile commands, 10 in `src/asr`, and none for the excluded paths above. A minimal CLI-off, empty-dependency-prefix profile has 49 commands, also 10 in `src/asr`; it contains the runtime and C ABI only. The Task 01 baseline had 78 compile commands, 29 under `src/asr`.

SentencePiece is used by ASR context-biasing decoders, not by Nemotron 3 Sortformer. Its lookup, include path, link and license install are now inside the ASR CMake branch. A fresh standalone configure and `nemo_speech_diar_c` build with `NEMO_SPEECH_DEPENDENCY_PREFIX=/tmp/nemo-speech-empty-deps` succeeded without finding SentencePiece. SentencePiece remains an ASR dependency; the baseline had statically linked it, so there was no separate dynamic `ldd` entry to remove.

The standalone C ABI library's dynamic dependencies include diarization, audio I/O, frontend, shared encoder and GGML CPU runtime libraries plus system runtime libraries. It has no dependency on `nemo_speech_asr`, SentencePiece, TTS, NMT or Python. `audio_decoder` and subtitles do not enter the diarization library link closure.

## Validation

Build and API commands:

```bash
scripts/configure.sh cpu-diar
cmake --build --preset cpu-diar
scripts/configure.sh cpu-diar -DBUILD_TESTING=ON
cmake --build --preset cpu-diar --target nemo-speech diar_c_smoke \
  test_diar_streaming test_diar_state test_mel_features
ctest --test-dir build/cpu-diar --output-on-failure \
  -R 'diar_c_api_model|diar_state|mel_features'
scripts/configure.sh cpu-asr
cmake --build --preset cpu-asr
```

The required standalone build succeeded. The final focused CTest run passed 3/3: `diar_state`, `mel_features`, and `diar_c_api_model`; the latter loads the local Q8_0 model through the public C API, checks 8 speaker channels and a 10 ms frame period, and opens/closes a stream. The build-tree and installed legacy C library symlinks were checked. `nm -D` confirmed the standalone public exports; `ldd` showed no ASR implementation library.

Regression checks against Task 01 used the same model and CPU backend:

| Check | Comparison after refactor |
|---|---|
| JFK streaming JSON | Exact match with baseline; 3 segments, all speaker 1. |
| JFK offline JSON | Exact match with baseline; same three segments and timestamps. |
| Five-second silence | Exact match; no segments. |
| Repeat JFK streaming inference | Repeated output is byte-identical; SHA-256 `41c76d7d6cceb8cf841b12f138412761ad6d27d4104809e756c739441fcd09e6`, matching Task 01. |
| AMI streaming JSON | Normalized output matches baseline exactly: 25 segments and the same assigned slots. |
| AMI streaming with 160 ms pushes | Normalized RTTM and speaker IDs match the baseline chunked run. |
| C ABI / model geometry | Loads the same Nemotron 3 Q8_0 model; 8 speaker channels, 0.01 s per frame, stream lifecycle succeeds. |

The baseline and post-refactor inference wall times were comparable on the same host: JFK streaming 2.44 s vs 2.455 s; AMI streaming 44.11 s vs 43.289 s; AMI with 160 ms pushes 43.24 s vs 44.459 s. These small differences vary by run and no inference optimization was attempted. Model-load smoke remains about 0.10 s. A peak-RSS measurement was not available for Task 01 or Task 02, so memory use is not compared.

The test audio exists in the repository: JFK and AMI. The AMI output comparison checks regression against the baseline; it is not a DER accuracy pass. The repository lacks the NeMo reference dumps required by `test_sortformer_parity`, so that test was not run. JFK has no RTTM reference. No test was marked successful without its needed model/audio.

## Build and library size comparison

Clean CPU builds were measured with tests disabled and the same GCC/CMake/Ninja environment and patched ggml source. Baseline completed 92 Ninja actions in about 40.0 s; the standalone refactor completed 75 in about 32.3 s. The compile database decreased from 78 to 58 commands (20 fewer, 25.6%); `src/asr` translation units decreased from 29 to 10 (19 fewer, 65.5%). The Ninja action count fell by 17 (18.5%). These are single-host wall measurements, not a build-time guarantee.

| Output libraries | Baseline bytes | Standalone bytes |
|---|---:|---:|
| ASR implementation + ASR C ABI pair vs diarization + C ABI + shared audio/frontend/encoder libraries | 2,968,728 | 1,197,080 |
| Unified CLI `nemo-speech` | 677,680 | 667,968 |

The first row compares the baseline monolithic ASR/C API pair with the new runtime libraries required by the diar CLI and C API; it is a useful build-product size comparison, not a byte-for-byte like-for-like ABI object comparison. The CLI executable is 9,712 bytes smaller. GGML backend libraries are common to both profiles and are not included in the first row.

## Files changed

- Build graph and consumers: root `CMakeLists.txt`, `src/asr/CMakeLists.txt`, `src/common/CMakeLists.txt`, `src/core/CMakeLists.txt`, `app/CMakeLists.txt`, `examples/CMakeLists.txt`, and `src/tts/magpietts/CMakeLists.txt`.
- Public and internal C API: `include/nemo_speech/common.h` (new), `include/nemo_speech/asr.h`, `include/nemo_speech/diar.h`, `src/asr/c_api.cpp`, `src/asr/diar/c_api.cpp` (new), `src/common/c_api_common.cpp` (new), and `src/common/c_api_internal.h` (new).
- Tests: `tests/c/diar_c_smoke.c` (new), `tests/cpp/CMakeLists.txt`, `tests/cpp/asr/CMakeLists.txt`, and `tests/cpp/diar/CMakeLists.txt` (new).
- Documentation: this report and the updated `docs/VOICEID_CLEANUP_PLAN.md`.

`src/asr/c_api.map` was inspected and reused without changing its public symbol policy. The `.gitignore` update for local GGUF files was already present in the Task 01 worktree and is not part of this refactor commit.

## Physical deletion readiness

For a future VoiceID-only fork, ASR model/recognizer/runner, decoder, VAD, PnC and postprocessing implementations can be considered for physical removal after dropping ASR products and APIs and checking their non-ASR consumers. The same applies to ASR subtitles and the test-only audio decoder after their remaining tests/users are removed. Those files remain in this repository and still build when ASR is enabled.

Do not remove the shared audio I/O/resampler, mel frontend, GGML/GGUF runtime, JSON/config helpers, AOSC, diar pipeline, RoPE encoder or current Sortformer implementation. FastConformer, relative-position attention, transformer encoder and the V2 branch remain because the shared project continues to support V2 Sortformer and ASR. Removing V2 is a separate behavior change and requires its own parity review. Task 02 performed no physical deletion.
