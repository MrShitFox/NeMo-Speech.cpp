# Task 06 — Nemotron 3 internal representations and UNKNOWN rejection

## Verdict

**B — Promising but not validated.**

The graph exposes several useful acoustic representations, and the opt-in
research taps leave ordinary diarization outputs unchanged in the checked
CPU runs. The current fixture does not establish speaker verification:
enrollment and evaluation come from one AMI meeting, there is no development,
calibration, or holdout partition, and the unknown query remains highly similar
to enrolled profiles. Do not select a permanent representation or acceptance
threshold from these results.

## Model and graph map

The inspected checkpoint was `Nemotron-3-Diarization.q8_0.gguf`, SHA-256
`08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`. Its GGUF
configuration resolves to 16 kHz audio, 128 mel bins, a feature stacking
factor of 8, a 31-layer RoPE encoder, 512 encoder dimensions, 8 attention
heads (64 dimensions per head), 2048 feed-forward dimensions, a 192-dimensional
projection, and 8 native speaker channels. The native prediction grid is
10 ms; feature-stack rows are on the 80 ms encoder grid.

| Graph point | Shape for a current chunk | Properties |
|---|---:|---|
| Feature-stack projection (`pre_encoder`) | 512 × T3 | Existing raw pre-encoder values used by AOSC. The diagnostic view aliases the existing chunk output. |
| RoPE block outputs 0, 7, 15, 23, 30 | 512 × T3 | Captured after each residual block and before the encoder's final normalization. |
| Full encoder output (`encoder_output`) | 512 × T3 | Output after the 31st block and `encoder.final_norm`. |
| `encoder_proj` | 192 × T3 | Projected encoder output consumed by the subpixel prediction head. |
| Subpixel prediction head | 8 × 10 ms frames | 192-channel Conv1D upsampled by 8, then hidden and speaker linear layers and sigmoid. |

In the graph, AOSC cache and FIFO rows are concatenated before self-attention.
Research outputs crop away the state-prefix rows and return current audio rows,
but those rows have still attended to the cache, FIFO, neighboring audio, and
all active speaker channels. The layer taps are therefore context-conditioned
activations, not independent speaker encoders.

Normalization in the implementation is explicit: an optional
`encoder.embed_norm` is applied according to the GGUF pre-block-normalization
flag; within each block, `norm1` precedes QKV attention and `norm2`
precedes the feed-forward network, with residual additions after both
sub-blocks. `encoder.final_norm` runs after block 30. Intermediate taps are
after the block residual and before this final normalization.

## Instrumentation and output contract

`SortformerResearchConfig` is an opt-in C++ configuration. The default empty
configuration builds the original two-output graph and leaves its output order
unchanged: native probabilities first, existing pre-encoder embeddings second.
Requested taps are appended after those outputs. Intermediate layers are
captured at the existing block outputs; no weights or model operations were
changed. The internal observer only receives committed central streaming
frames. It is not exposed through the C ABI.

With research mode disabled, no additional feature tensors are copied to the
host and no additional graph outputs are requested. The existing probabilities
and pre-encoder output contract remains in place. The selection trace for AOSC
is a sidecar enabled only when a research observer is installed; it is not
part of the profile or snapshot format. Missing frame lineage after state
restore is represented by a `-1` sentinel.

The instrumentation can use the existing GGML backend selected by the runtime.
This task measured CPU only; Vulkan, CUDA, and Metal were not benchmarked.

## Dataset and protocol

`tests/manifests/task06_ami_preliminary.json` is explicitly marked
`preliminary_fixture_only`. It contains two enrollment clips (16.40 s and
13.86 s) and three disjoint-in-time evaluation clips (2.62 s known, 2.57 s
known with overlap, and 6.13 s unknown). All five clips come from the same
60-second AMI EN2002d Mix-Headset recording, session, and device. The RTTM is
used only for the separate diarization comparison.

There are **zero** development clips, **zero** calibration clips, and
**zero** final holdout clips. There is one unknown evaluation utterance and no
unknown-versus-unknown pair. These counts are too small for FAR/FRR, ROC, or
EER claims. The Task 05 mixed-meeting result also showed two of three unknown
turns assigned to enrolled channels; that remains a preliminary same-meeting
observation. The Task 06 evaluation clips are 2.57 s, 2.62 s, and 6.13 s;
there is no under-2-second query, so the short-turn failure case is not
measured by this benchmark.

The manifest accepts separate `development`, `calibration`,
`holdout`, and `preliminary` arrays. Evaluation clips require an explicit
`known` boolean and can include `recording_id`, `session`, `device`,
`overlap`, and an optional `rttm_recording_id`. When
`preliminary_fixture_only` is false, the harness rejects enrollment/evaluation
audio file or recording ID reuse and rejects recording IDs or audio files
reused across development, calibration, and holdout. The flag is an explicit
exception for preliminary fixtures, not a shortcut for final evaluation.

## Regression results

The research harness ran the original graph and the instrumented graph over
the same five clips, with the CPU backend, four threads, 160 ms pushes, and the
same model and geometry.

| Check | Result |
|---|---:|
| CTest suite | 5/5 passed |
| Research/baseline clips | 5 compared, 0 mismatches |
| Raw probability float values | 33,264 identical |
| Speaker-channel assignment frames | 4,158 identical |
| Segment lists | 5/5 identical |
| AOSC profile readiness and bytes | 5/5 identical |
| Snapshot bytes | 5/5 identical |
| Repeated probability and feature hashes | bit-identical |

The CTest suite included audio resampling, AOSC state, mel features, the C ABI
model smoke test, and snapshot continuity. The snapshot test covers JFK and
AMI audio, 160 ms pushes, snapshot/restore, silence plus compaction, and speaker
changes. The separate enrollment probe destroyed the enrollment model and
stream, reloaded both the AOSC-only profile and snapshot from disk, and
completed its Task 05 checks.

That disk round-trip formed a ready 264-row, 512-D AOSC cache profile
(551,305 bytes); the selected cache contained 136, 5, and 60 speech rows in
three local channels. Full-snapshot and AOSC-only starts each mapped both
known queries correctly and did not map the single unknown query to an
enrolled identity in this crop. A fresh stream mapped one of two known queries
correctly and falsely assigned the unknown to an enrolled identity. These are
per-attempt counts from the same meeting, not operating rates.

Across the five clips, the selected research outputs account for 14,596,608
runtime host-copy bytes: one backend output transfer and one per-item split
copy. The benchmark then retains a further 6,815,232 bytes in feature
matrices. The largest per-chunk runtime copy total was 365,568 bytes; the
largest individual central feature view was 26,624 bytes. These are byte
volumes, not process peak RSS. The end-to-end feature-pass times by clip
(16.40 s, 13.86 s enrollment; then 2.62 s, 2.57 s, and 6.13 s queries) were
4,676 ms, 3,850 ms, 300 ms, 231 ms, and 754 ms. They include normal
diarization inference with the requested taps; they are not incremental
copy-only timings. Normal inference requests and copies none of the additional
research outputs. In the latest timing run, mean baseline stream time was
2,024.2 ms and instrumented time was 1,962.1 ms (-3.07%). Across repeated CPU
measurements, the observed delta varied from -3.1% to +2.2%; this small and
noisy timing sample does not establish a stable latency overhead.

## Representation comparison

The table uses mean pooling and mean similarity for the preliminary fixture.
“Known rank 1” is the number of the two known queries whose best enrolled
profile had the right label. Cosines are similarity scores, not probabilities.

| Representation | Dim. | Known rank 1 | Mean best known cosine | Best unknown cosine |
|---|---:|---:|---:|---:|
| `pre_encoder` | 512 | 2/2 | 0.9788 | 0.9923 |
| `encoder_output` | 512 | 1/2 | 0.8578 | 0.9285 |
| `encoder_proj` | 192 | 2/2 | 0.9084 | 0.9344 |
| `encoder.layer.0` | 512 | 2/2 | 0.9886 | 0.9942 |
| `encoder.layer.7` | 512 | 2/2 | 0.9898 | 0.9946 |
| `encoder.layer.15` | 512 | 2/2 | 0.9792 | 0.9853 |
| `encoder.layer.23` | 512 | 2/2 | 0.9682 | 0.9816 |
| `encoder.layer.30` | 512 | 2/2 | 0.9813 | 0.9867 |

No representation separated the unknown query below the known-query scores in
this sample. The encoder output also confused one of the two known queries.
The lower unknown cosine for `encoder_proj` and later encoder taps is
interesting enough to test on independent recordings, but the unknown remains
more similar to an enrolled profile than the mean known score.

## Pooling and similarity

The harness implements mean pooling, per-frame L2-normalized mean, activity and
quality weighted pooling, three temporal prototypes, and AOSC-selected
enrollment rows. The weighted method uses activity to find speech-like rows,
the gap between the best and second-best channel activity as a separation
weight, and adjacent-frame cosine stability. Activity is not treated as an
identity probability. AOSC-selected enrollment prototypes use rows actually
retained by the streaming cache; short evaluation clips use the weighted
speech-row selector because they do not fill a new AOSC cache.

For `pre_encoder` with mean-similarity aggregation, the preliminary results
were:

| Pooling | Known rank 1 | Best unknown cosine | Mean top-1/top-2 margin |
|---|---:|---:|---:|
| Mean | 2/2 | 0.9923 | 0.0245 |
| L2-normalized mean | 2/2 | 0.9930 | 0.0233 |
| Activity/quality weighted | 2/2 | 0.9876 | 0.0508 |
| Three prototypes | 2/2 | 0.9816 | 0.0237 |
| AOSC-selected enrollment | 2/2 | 0.9872 | 0.0537 |

The unknown still ranked against a registered identity for every method. The
larger margins for weighted and AOSC-selected pooling do not establish a safe
rejection rule. For the three-prototype profile, scoring aggregation changed
the known ranking and raised the unknown score under max and top-2:

| Aggregation | Known rank 1 | Best unknown cosine | Mean top-1/top-2 margin |
|---|---:|---:|---:|
| Mean similarity | 2/2 | 0.9816 | 0.0237 |
| Maximum similarity | 0/2 | 0.9946 | 0.0084 |
| Top-2 similarity | 1/2 | 0.9940 | 0.0093 |

That is a warning about max/top-K scoring on this fixture, not a reliable
false-match rate. For single-prototype profiles, these aggregations reduce to
the same score.

The JSON also records genuine/impostor pair counts, top-1/top-2 margins,
duration buckets, unknown-versus-unknown comparisons, profile size, and
scoring cost. Calibration thresholds are selected only from the calibration
partition. Here that partition is empty, so the threshold, calibrated FAR/FRR,
and EER are null. If the system is forced to assign every query to the top
profile, the one unknown query is a false acceptance (1/1). No numeric
acceptance threshold is recommended.

Identity attempts are also split by the manifest's clean/overlap labels. This
fixture has one known and one unknown clean query, and one known overlap query;
there is no unknown overlap query, so overlap-specific UNKNOWN rejection
cannot be evaluated.

## Diarization errors, scored separately

RTTM diarization metrics use the existing segmenter and a 10 ms frame grid.
The predicted speaker channels are mapped to reference speakers separately
inside each clip before counting misses, false alarms, and confusions. This is
a **clip-wise permutation** score; it does not measure persistent identity
consistency across separate clips.

| Scope | Clips | Reference speaker frames | Miss | False alarm | Confusion | Clip-wise DER |
|---|---:|---:|---:|---:|---:|---:|
| All preliminary clips | 3 | 1,079 | 120 | 32 | 22 | 0.1613 |
| Clean clips | 2 | 875 | 93 | 0 | 0 | 0.1063 |
| Overlap clip | 1 | 204 | 27 | 32 | 22 | 0.3971 |

All five annotated turns had predicted speech after the per-clip channel
mapping. These few turns are not a reliable diarization accuracy estimate.
They are reported separately from the identity rank and unknown false
acceptance counts.

## Streaming-context dependence

The context probe aligns the target audio to the 80 ms encoder hop and compares
only target frames against the same target in a fresh stream. This avoids
comparing delayed context frames with the query. `pre_encoder` matched exactly
(mean frame cosine 1.0) in all aligned modes. Encoder representations changed
when AOSC/FIFO history changed.

| Context before the same target | AOSC differs before target? | Pre-encoder cosine | Full encoder cosine | `encoder_proj` cosine |
|---|---:|---:|---:|---:|
| Registered clip on empty stream | n/a | 1.000 | 0.345 | 0.374 |
| Registered clip after AOSC profile | yes | 1.000 | 0.874 | 0.926 |
| One unknown clip on empty stream | n/a | 1.000 | 0.438 | 0.377 |
| One unknown clip after AOSC profile | no | 1.000 | 0.849 | 0.926 |
| Four unknown clips on empty stream | yes | 1.000 | 0.421 | 0.450 |
| Four repeated unknown clips after AOSC profile | yes | 1.000 | 0.875 | 0.935 |
| Reverse enrollment order | yes | 1.000 | 0.415 | 0.294 |
| AOSC-only warm start | no | 1.000 | 0.866 | 0.939 |
| Full snapshot warm start | no | 1.000 | 0.869 | 0.937 |

The one-clip unknown sequence did not change the compressed profile before the
target; four repetitions did. That stress sequence repeats the same unknown
AMI excerpt and is not evidence about diverse unknown speakers. The
enrollment profile artifact remained unchanged; only each temporary
evaluation stream's state changed. Layers 0 and 7 were more stable than the
full encoder output in these probes, while layers 15 and 30 also shifted with
context. No one point can be called an independent fingerprint from this
single recording.

## Profile size, runtime, and privacy

Two one-prototype 512-D identities occupy 4,096 bytes as float32 vectors; the
192-D `encoder_proj` version occupies 1,536 bytes. Three 512-D prototypes
per identity occupy 12,288 bytes. The combined AOSC-only profile in the fixture
occupies 551,305 bytes (264 cache rows, 512 dimensions, prediction and silence
state included); the full snapshot occupies 915,131 bytes in the research
benchmark.

The synthetic cosine benchmark measured about 2.7 µs for 2 one-prototype
profiles, 13.2 µs for 10, and 133 µs for 100. Three-prototype comparisons took
about 23 µs, 119 µs, and 1.18 ms for 2, 10, and 100 profiles. The benchmark
reuses one measured profile to estimate linear comparison cost; it does not
include database lookup or independent profile I/O.

`diar_enrollment_probe` now writes POSIX research profiles through a same-
directory `mkstemp` file (mode 0600 at creation), writes and syncs it, then
atomically renames it into place. AOSC profile and snapshot files both showed
mode `0600` in the manual re-import run. The Windows temporary-file fallback
was not exercised. The production snapshot serialization contract was not
changed. Private profile/audio locations and profile extensions are ignored
by Git. The Task 06 result JSON contains aggregate metrics only; it does not
contain feature vectors or private audio.

## Reproduction

From the repository root, with the local checkpoint and AMI fixtures present:

```sh
cmake --build build/cpu-diar -j4
ctest --test-dir build/cpu-diar --output-on-failure
build/cpu-diar/bin/diar_representation_probe \
  tests/manifests/task06_ami_preliminary.json \
  --repeat 2 \
  --report tests/results/diar_representation_probe_ami_preliminary.json
```

The Task 05 disk round-trip check used:

```sh
build/cpu-diar/bin/diar_enrollment_probe \
  models/Nemotron-3-Diarization.q8_0.gguf \
  --enroll P1=test_files/diar/ami_en2002d_2132.wav@3.12:19.51 \
  --enroll P2=test_files/diar/ami_en2002d_2132.wav@29.20:43.05 \
  --test P1=test_files/diar/ami_en2002d_2132.wav@48.55:51.16 \
  --test P2=test_files/diar/ami_en2002d_2132.wav@45.39:47.95 \
  --test P3=test_files/diar/ami_en2002d_2132.wav@51.22:57.34 \
  --silence-ms 1000 \
  --profile /tmp/task06-profile.aosc \
  --snapshot /tmp/task06-state.snapshot \
  --report /tmp/task06-enrollment-regression.json
```

The enrollment probe report contains frame-level predictions and should remain
in a private temporary location rather than be committed.

## Task 07 recommendation

Keep AOSC for streaming diarization and keep channel IDs local to a stream.
Represent persistent identity as an explicit external identity to profile
mapping; never infer identity from a channel number. Before any production
profile or API work, collect multiple independent recordings per known person,
new unknown people, and sessions/devices/acoustic conditions outside the
current meeting. Use separate development, calibration, and final holdout
sets with recording and audio-file disjointness enforced by the manifest.

Retest `encoder_proj`, selected intermediate layers, activity/quality
weighted pooling, and AOSC-selected enrollment prototypes on those splits.
Calibrate the absolute score and top-1/top-2 margin using calibration data
only. Return UNKNOWN when the query is too short, no valid calibration
threshold exists, or calibrated acceptance conditions are not met. Do not
update permanent profiles from evaluation speech.

No production VoiceID API or permanent profile implementation was added in
Task 06.
