# Task 04 — Streaming state snapshot and restore

Branch: `feature/aosc-snapshot-restore` (based on Task 03 commit `4fc69c4`).

## Result

Implemented version 1 binary checkpoints for a live Sortformer V3 streaming
session, including AOSC, unfinished audio and mel context, the streaming
resampler, probability history, and compaction state. A restored session uses
the same absolute frame clock and speaker-channel order. The CPU acceptance
test destroys the original stream and model, reloads the GGUF, and matches the
uninterrupted run bit for bit for frame probabilities and exactly for segment
times and speaker IDs.

This is continuation of one diarization session. It does not identify or
enroll a person across independent sessions.

## Streaming state map

| Owner | State | Dependency and checkpoint treatment |
|---|---|---|
| `DiarizationStream` | Input sample rate, `AudioResampler`, resampler output scratch | Sample rate is fixed after the first push. The resampler state is serialized. `resampled_` is an output scratch buffer cleared before each conversion and is not needed by a later operation. |
| `AudioResampler` | Rate pair, phase geometry, coefficient table, retained input history, absolute input/output counters, source position and phase remainder, finished flag | All fields that affect the next output sample are serialized. The float32 coefficient table is included so restore does not regenerate it with a host-specific math library. |
| `DiarStream` scheduler | `audio_buf_`, `audio_base_`, `mel_buf_`, `mel_base_`, `mel_consumed_`, `provisional_frames_`, `finished_` | Keeps the unconsumed waveform, pre-emphasis/STFT left context, computed mel frames, right-context availability, global sample/frame offsets, and a provisional tail that the next complete chunk may replace. All are serialized. `mel_produced()` is derived from the mel base and retained mel frames. |
| Mel frontend | Window, mel basis, FFT tables, static config | The frontend is stateless per call. The window, filterbank, FFT tables, and frontend config are rebuilt from the loaded model. The model fingerprint binds the snapshot to those model bytes. The waveform/mel buffers and their indices carry the streaming context that the frontend itself does not retain. |
| `AoscState` | Speaker-cache embeddings and frame count; cached speaker predictions and validity sentinel; FIFO embeddings and frame count; running silence embedding and silence-frame count; learned-silence mode | These values determine the model state input, FIFO update, silence estimate, and later cache compression, so they are serialized in their original order. Cache prediction columns retain the original speaker-channel order. Geometry and scoring constants are model/session configuration rather than runtime state. |
| AOSC FIFO predictions | Fresh model predictions for the FIFO and current chunk | They are temporary inside `AoscState::update`; FIFO predictions are recomputed for the next model call. No prediction buffer survives between streaming operations. Cached speaker predictions do survive and are checkpointed. |
| Speaker channel gate | Frame counter, established flags, clean/fading counters, last-win frames, raw revision ring | The complete gate state is serialized. V3 currently passes native Sortformer speaker channels without applying the legacy birth gate, but retaining the gate fields keeps the stream state complete and prevents hidden differences if its state is used. |
| Probability timeline | `probs_`, `probs_base_`, `provisional_frames_` | Stores retained frame-major probabilities and the global index of their first frame. Total frame count is derived as `probs_base_ + retained_frame_count`. Values are written as IEEE float32 bits without numeric conversion. |
| Segmentation and compaction | Frozen segments, trigger/retain thresholds, `hold_from_` | Tail segments are recomputed from retained probabilities; frozen prefix segments are kept as doubles and restored with their absolute times and channel IDs. Compaction is retried from the same retained probabilities with the same thresholds after restore. |
| Speaker mapping | No separate V3 permutation table exists | Native speaker channels remain indexed by their model output column. AOSC entries may be selected/reordered by compression, but their prediction columns are retained unchanged. The cached arrays and output timeline are serialized without channel reindexing. |
| Model and GGML runtime | GGUF weights, graph, session, backend handles, mutexes, scratch tensors | Not serialized. They are rebuilt by loading the model. A SHA-256 digest of the full GGUF file must match before import. |

There is no separate streaming pre-emphasis state: the retained waveform
contains the history required by the stateless FE calls. There is no separate
segmentation hysteresis accumulator; segmentation is derived from the retained
timeline, while already compacted results live in `frozen_segs_`.

## Snapshot contents and binary format

Version 1 uses a fixed 108-byte little-endian header followed by a payload.
Integer fields have explicit 32- or 64-bit widths. Float values use IEEE-754
float32/float64 bit patterns. Array lengths are unsigned 64-bit values; no C++
struct, pointer, `size_t`, mutex, graph, backend object, or weight tensor is
written directly.

| Byte offset | Size | Field |
|---:|---:|---|
| 0 | 8 | Magic `NMSDST01` |
| 8 | 4 | Snapshot format version (`1`) |
| 12 | 4 | Architecture ID (`3`, Sortformer V3) |
| 16 | 32 | SHA-256 of the complete GGUF file |
| 48 | 24 | Model sample rate, speaker channels, mel bins, FFT size, hop size, output subsampling factor |
| 72 | 24 | Six resolved AOSC geometry values: cache, FIFO, chunk, update period, left context, right context |
| 96 | 8 | Payload byte length |
| 104 | 4 | CRC-32/ISO-HDLC of the payload |
| 108 | variable | Payload |

The payload contains, in order: input-rate/resampler state; AOSC caches and
counters; speaker-gate history; audio and mel buffers with global offsets;
provisional/finished flags; retained probabilities and their base frame;
frozen segments; compaction thresholds and hold frontier. The resampler
coefficient table and retained filter history are included. Model weights are
never copied into the snapshot.

The decoder checks magic, version, architecture, the model digest, model and
frame geometry, exact streaming geometry, payload size/checksum, array lengths,
integer overflow, flags, counters, frame shapes, probabilities, and segment
values. The maximum encoded size is 1 GiB. On failure, import returns an
`INVALID_ARGUMENT` status with a diagnostic string and does not mutate the
destination stream. Import is allowed only once into a fresh stream handle;
even importing an empty snapshot marks that handle as restored. Save, load,
push, finish, and compaction changes are serialized by the stream operation
mutex, so a checkpoint observes a completed operation boundary.

The binary format is endian-independent and can be moved between compatible
runtime hosts. This does not promise bit-identical inference across different
backends or compiler/runtime builds. CPU parity was measured on the same host,
build, and backend.

## C and C++ APIs

The internal C++ streaming wrapper exposes:

```cpp
std::vector<uint8_t> DiarizationStream::export_state() const;
void DiarizationStream::import_state(const uint8_t* data, size_t size);
```

The public C API adds three symbols without changing existing declarations or
struct layouts:

- `nemo_speech_diar_stream_state_size(stream, &size_bytes)`
- `nemo_speech_diar_stream_save_state(stream, buffer, capacity, &written)`
- `nemo_speech_diar_stream_load_state(fresh_stream, buffer, size)`

Export uses the size-query then save pattern. Import requires a newly opened
stream associated with a model having the same GGUF contents and streaming
geometry. Continue with the same input sample rate. The API never writes
snapshot data to a file.

## Continuity and regression results

Local CPU Q8_0 model:
`models/Nemotron-3-Diarization.q8_0.gguf`.

| Scenario | Result |
|---|---|
| Empty stream save/restore/save | Pass; second serialization is byte-identical, 2,491 bytes |
| Invalid version, truncation, payload damage, model fingerprint, geometry, oversized array and invalid input-rate field | All rejected; failed imports leave the target fresh |
| First completed inference chunk | Pass; 1,101 final frames, 8,808 probabilities, 3 exact segments |
| 160 ms chunks, partial input tail, AOSC update/compression, model+stream destruction, two repeated restores | Pass; 1,101 frames, 8,808 retained probabilities bit-identical, 3 segments identical |
| 48 kHz input and active resampler restore | Pass; 201 frames, 1,608 probabilities bit-identical, 1 segment identical |
| Long silence, compacted prefix and frozen segments | Pass; 1,601 frames, 3,208 retained probabilities bit-identical, same frame base and 3 segments |
| AMI two-speaker crop with channel changes | Pass; 921 frames, 7,368 probabilities bit-identical, 4 segments and the same speaker IDs |
| Public C API | Pass; size query, short-buffer rejection, save, load into fresh handle, and rejection of a second import |

Snapshot sizes and measured codec time on this CPU build:

| State | Size | Save | Load |
|---|---:|---:|---:|
| Empty | 2,491 B | — | 0.03 ms |
| After first processed chunk | 120,507 B | 1.54 ms | 1.23 ms |
| After AOSC updates, with pending audio | 210,063 B | 2.78 ms | 2.09 ms |
| Repeated checkpoint | 196,687 B | 2.43 ms | 1.96 ms |
| Active 48 kHz resampler | 119,765 B | 1.42 ms | 1.24 ms |
| Compacted probabilities and frozen segments | 492,279 B | 5.82 ms | 4.96 ms |
| AMI speaker-change crop | 178,471 B | 2.17 ms | 1.83 ms |

The timings cover in-memory encoding/decoding and exclude model loading. Model
reload is verified separately by the end-to-end test.

The fingerprint recorded in the local test snapshot matched the independent
`sha256sum` result for the GGUF:
`08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`.
The CRC detects accidental payload damage; it is not an authentication tag.
The snapshot is not encrypted. Callers must protect it in storage and transit.
The debug file utility is POSIX-only; the in-memory C/C++ snapshot APIs and
binary codec are platform-independent.

Task 03 regression checks on the same CPU checkpoint:

- `diar_state`, `mel_features`, `audio_resampler`, `diar_c_api_model`, and the
  new `diar_snapshot_continuity` CTests pass (5/5).
- JFK streaming JSON segment SHA-256 remains
  `0afa4bf606dab6e91371ba3c3b8015605baa526efe20135877a82911c2f03812`;
  offline segments match the Task 03 output.
- AMI streaming remains 25 segments on speaker channels 1, 2, and 3, with
  Task 03 segment SHA-256
  `6fac6b24ca2af9913e02de7325af752f8ae206def50f7abf4a4005cb56c3497f`.
- JFK and AMI 160 ms streaming RTTM/JSON comparisons match after normalizing
  the existing 0-based C++ test labels to the public 1-based labels.
- Five seconds of silence produces 501 frames and zero segments.
- `test_sortformer_parity` was not run: NeMo reference dumps are not present in
  this checkout. CUDA, Vulkan, and Metal backends were not tested.

## Test utility and reproduction

With tests enabled on POSIX, the test-only utility supports two processes:

```bash
build/cpu-diar/bin/diar_snapshot_cli \
  models/Nemotron-3-Diarization.q8_0.gguf part1.wav --save-state /tmp/session.bin
build/cpu-diar/bin/diar_snapshot_cli \
  models/Nemotron-3-Diarization.q8_0.gguf part2.wav --load-state /tmp/session.bin
```

The first command saves before final flush. POSIX output uses mode `0600` and
an atomic rename. The second command prints segment times on the continuing
session clock; the second WAV starts at the current frontier, not at timestamp
zero. The test utility does not add snapshot files to the repository.

Reproduce the model-backed acceptance and available tests:

```bash
cmake --build build/cpu-diar -j4
ctest --test-dir build/cpu-diar --output-on-failure
build/cpu-diar/bin/test_diar_snapshot \
  models/Nemotron-3-Diarization.q8_0.gguf \
  test_files/diar/jfk.wav test_files/diar/ami_en2002d_2132.wav
build/cpu-diar/bin/test_diar_streaming \
  models/Nemotron-3-Diarization.q8_0.gguf test_files/diar/jfk.wav --push-ms 160
```

Additional Task 03 regression commands:

```bash
build/cpu-diar/bin/nemo-speech diarize test_files/diar/jfk.wav --model models/Nemotron-3-Diarization.q8_0.gguf --backend cpu --format json
build/cpu-diar/bin/nemo-speech diarize test_files/diar/jfk.wav --model models/Nemotron-3-Diarization.q8_0.gguf --backend cpu --offline --format json
build/cpu-diar/bin/nemo-speech diarize test_files/diar/ami_en2002d_2132.wav --model models/Nemotron-3-Diarization.q8_0.gguf --backend cpu --format json
build/cpu-diar/bin/test_diar_streaming models/Nemotron-3-Diarization.q8_0.gguf test_files/diar/ami_en2002d_2132.wav --push-ms 160 --rttm ami
```

The ordinary Task 03 pipeline and AOSC math were not modified. No enrollment,
verification, or cross-session identity behavior is implemented.
