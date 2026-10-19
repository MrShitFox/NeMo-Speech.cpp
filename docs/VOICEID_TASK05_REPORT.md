# Task 05 — AOSC enrollment and cross-session identification research

## Result

**Decision B — Partially Feasible, with preliminary evidence only.** AOSC-only
state can carry the channel assignments made during enrollment into a newly
created streaming session. On disjoint utterance intervals from one AMI
meeting, both full-snapshot warm start and AOSC-only warm start preserved two
different enrolled channels and outperformed an empty session. The same data
also contains false acceptance of an unenrolled speaker on short turns.

This is not evidence of identity recognition across independent recordings.
All current evaluation audio comes from one Mix-Headset recording, with the
same microphone, room, participants, and conversation. No held-out meeting or
independent capture for the same people is present in the checkout. The small
sample cannot support statistical FAR/FRR, ROC, or EER claims.

## What the model stores

The V3 streaming path stores per-frame 512-dimensional pre-encoder acoustic
features in `AoscState::spkcache_`. These are temporal model inputs selected by
the AOSC compression routine; they are not a documented, normalized,
utterance-level biometric speaker embedding. The parallel `spkcache_preds_`
array stores the model's coarse-grid speaker-channel probabilities for those
same cache rows. AOSC uses the predictions to score and retain cache rows,
with a `> 0.5` speech mask, per-channel boosts, and silence sentinel rows.

The selected features retain channel association through their aligned
prediction columns. V3 has no separate speaker permutation or identity table:
the model output columns are the channel indices. Those indices are
session-local roles. The experiment produced P1/P2 channels 0/2 in one order,
then 2/0 when enrollment order was reversed. The unknown person used channel 1
in warm-start tests. This directly rules out interpreting channels 0 and 1 as
fixed people or channels 2–7 as a reserved unknown range.

The cache also depends on model weights, cache geometry, and related
configuration. `spkcache_preds_` is essential: it is the seeded channel-column
association used by later compression. The V3 learned silence embedding is
model state; the cache updater also keeps the silence profile and count. FIFO
embeddings, audio/mel buffers, absolute timestamps, probabilities, and the
chunk scheduler are required to continue the same stream, but are not required
to initialize the experimental AOSC-only warm start. AOSC-only import starts a
new timeline and an empty FIFO. The full snapshot retains all of those
continuation fields.

The research AOSC profile format stores the model fingerprint, resolved
streaming geometry, cache embeddings, cache predictions, and silence state.
It rejects profiles without a completed first compression and rejects model
or geometry mismatches. It deliberately stores no person names or
identity-to-channel map; the probe derives that map from labeled enrollment
audio. A real persistent identity profile would also need a versioned,
validated mapping layer outside the model's AOSC state.

Upstream NeMo describes Streaming Sortformer as using an Arrival-Order Speaker
Cache to keep speaker tracking consistent across chunks, and documents a
speaker-cache input composed of pre-encoder embeddings and their lengths. That
is streaming state, not a promise of cross-recording biometric verification:
[NeMo diarization model notes](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/asr/speaker_diarization/models.rst),
[Sortformer model implementation](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/models/sortformer_diar_models.py).

## Test data and protocol

The model was `models/Nemotron-3-Diarization.q8_0.gguf`, SHA-256
`08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`, on the
CPU backend. It emits 8 native speaker channels at a 10 ms probability period;
AOSC features and cache predictions use the 80 ms coarse grid with 512 feature
dimensions. The fixture is the repository's 60 s, 16 kHz mono excerpt from AMI
meeting EN2002d, Mix-Headset. The reference RTTM labels the actual speakers and
includes overlap.

For the primary experiment, labels P1/P2/P3 below are aliases for MEE071,
FEO070, and MEE073 respectively. Enrollment and evaluation intervals are
disjoint in source time:

| Label | Role | Source interval | Duration |
|---|---|---:|---:|
| P1 / MEE071 | enroll | 3.12–19.51 s | 16.39 s |
| P2 / FEO070 | enroll | 29.20–43.05 s | 13.85 s |
| P1 / MEE071 | test | 48.55–51.16 s | 2.61 s |
| P2 / FEO070 | test | 45.39–47.95 s | 2.56 s |
| P3 / MEE073 | unknown test | 51.22–57.34 s | 6.12 s |

The P2 test interval contains a short MEE073 overlap. The RTTM-backed mixed
test is 45.39–59.92 s and does not overlap enrollment. It tests three
participants and contains two short overlaps. It still comes from the same
meeting recording, so it is a held-out segment rather than an independent
recording.

The probe concatenates enrollment clips in the specified order, inserting one
second of digital silence after each. It exports both a full Task 04 snapshot
and an AOSC-only profile. It then destroys the enrollment stream and model,
reloads the serialized files from disk, constructs new models, and evaluates
each test through a fresh stream. A separate sequence test loads the original
immutable state, processes the unknown clip first, adds two seconds of silence,
then processes the known clips in the same temporary stream. That modified
state is discarded after the sequence.

Each JSON report contains the raw enrollment and evaluation speaker
probabilities, actual channel assignments, segments, RTTM overlaps where
provided, cosine scores, cache summaries, and per-test outcomes. Raw
biometric feature vectors and audio are not included.

## Enrollment and cache formation

The V3 streaming preset used by this GGUF has 264 AOSC cache frames, an 80-frame
FIFO, and a 13-frame center chunk. At 80 ms per coarse frame, the first
compression is reached after roughly 344 frames (27.52 s) have passed through
FIFO and cache. Compression seeds `spkcache_preds_` at that point. This is a
model-state readiness condition, not a recommended enrollment duration; the
amount of clean speech and speaker balance still affect which rows survive.

The combined P1/P2 enrollment stream was 32.24 s including two seconds of
inserted silence. It reached a 264-row cache and a valid cached-prediction
array. The cache summary assigned speech rows to channels 0, 1, and 2 (136,
5, and 60 rows in the normal order). Channel 1 was not a registered person.
P1's enrollment interval was dominated by channel 0 and P2's by channel 2;
the test never forced either person into a chosen slot.

Single-person runs confirmed the readiness limit. P1 used a 17.39 s stream
including trailing silence; P2 used 14.85 s. Neither produced a compressed
cache or valid cached-prediction array, so method B correctly reported
`not_ready`. In each run A and C assigned the test voice to channel 0 and had
the same one-of-one channel-map result. AOSC-only export therefore requires
combined enrollment material long enough to populate the cache and seed its
predictions. A full snapshot can still carry partial cache/FIFO state, but it
remains a streaming continuation checkpoint.

## Methods A, B, and C

| Method | State loaded into each fresh evaluation stream | Result on disjoint single-speaker utterances |
|---|---|---|
| A — Full Snapshot Warm Start | Task 04 snapshot: AOSC, FIFO, scheduler/audio/mel context, channel state, and probability history | Normal order: P1→0, P2→2, P3→1. P1/P2 mapped correctly; P3 did not map to either enrolled identity. |
| B — AOSC-Only Profile | 264 cache embeddings, aligned `spkcache_preds`, and silence metadata; no FIFO or timeline | Same channel assignments as A on all three utterances in this sample. The serialized profile was reloaded from disk into a newly created model/session. |
| C — Fresh Session | Empty AOSC and empty streaming context | P1→0, P2→0, P3→0. P2 was mapped to P1 and P3 was falsely mapped to P1. |

The per-utterance results were A/B: 2/2 known channel mappings correct, 0/2
known false rejections, 0/1 unknown channel-map false acceptances, and 0 speaker
ID swaps. C: 1/2 known correct, 1/2 false rejections, 1/1 unknown false
acceptance, and one known-speaker swap. These are counts from one meeting
fragment, not estimates of operating rates.

The multi-speaker RTTM test gave A/B 5/5 annotated turns for the two enrolled
people mapped to the correct channel by greatest segment overlap. Of three
MEE073 turns, two were assigned to the enrolled FEO070 channel and one to an
unmapped channel. C had 1/5 enrolled turns mapped correctly and 4 false
rejections; it also falsely mapped 2/3 MEE073 turns to an enrolled channel.
There were no reference turns without any overlapping predicted segment in
this crop. These turn counts separate diarization segment presence from
identity mapping: a segment existed for every reference turn, while several
speaker identities were wrong.

In the P2 test clip, A/B emitted channel 2 for FEO070 and channel 1 around the
MEE073 overlap. That shows the system can diarize a short simultaneous or
back-channel speaker into another channel while preserving P2's enrollment
channel. In the longer mixed clip, short MEE073 turns were sometimes folded
into FEO070. Segment allocation is therefore not a reliable unknown-person
decision.

## Enrollment order, unknown speakers, and cache updates

Reversing registration order changed the measured mapping to P2→0 and P1→2.
A/B preserved that reversed mapping on the test utterances; C did not. The
channel swap is expected session behavior, not stable global identity naming.

With an unknown MEE073 clip first, then two seconds of silence, the original
AOSC profile changed in the temporary A/B session. The original immutable
profile file did not change. A's speech-cache row counts changed from
`[136, 5, 60, 0, 0, 0, 0, 0]` to
`[123, 33, 100, 0, 0, 0, 0, 0]`; B changed them to
`[136, 31, 57, 0, 0, 0, 0, 0]`. The subsequent P1 and P2 utterances still
mapped to their enrolled channels in both methods in this one sequence.
This confirms that unknown speech can update and occupy cache capacity. It
does not prove that cache compression will preserve known identities under
longer or repeated unknown traffic.

The silence-gap experiment compared one second between enrollment clips with
zero seconds. Without gaps, the 30.24 s combined stream still formed a 264-row
profile, with channel speech-row counts `[136, 6, 72, 0, 0, 0, 0, 0]`. A/B
again mapped P1/P2 to channels 0/2, and P3 to unmapped channel 1; C again
swapped P2 and falsely mapped P3 to P1. Thus one second of silence did not
change the channel mapping on this fixture. The silence duration also changes
chunk alignment and total input duration, so this small fixture does not
establish a robust silence policy. The reversed-order experiment also used
one-second gaps.

The no-silence run imported the same saved profile twice and replayed the same
P1 WAV interval in separate fresh streams. A, B, and C each produced
bit-identical raw float32 probability arrays for the repeated import. This is
a determinism check on one recording, not an independent-session accuracy
sample.

## Cosine similarity probe

The probe builds temporary channel centroids by averaging cached rows whose
stored channel prediction is the row maximum and exceeds the same 0.5 speech
mask AOSC uses. It compares those centroids to central 512-D pre-encoder
frames from the test utterance. Frames are sampled only when the matching
80 ms prediction window has mean maximum activity above 0.5. This activity
mask selects speech-like frames; it is not an identity threshold.

For the normal-order run, mean cosine scores to channels 0/1/2 were:

| Test | Speech feature frames | Channel 0 | Channel 1 | Channel 2 | Highest score channel |
|---|---:|---:|---:|---:|---:|
| P1 known | 29 | 0.789 | 0.750 | 0.736 | 0 |
| P2 known | 19 | 0.852 | 0.830 | 0.881 | 2 |
| P3 unknown | 45 | 0.850 | 0.840 | 0.789 | 0 |

The unknown P3 score for P1's channel-0 centroid is higher than P1's own score.
The score has no calibrated acceptance threshold and the known/unknown sample
is far too small for ROC/EER. This feature similarity probe is not a speaker
verification mechanism. Reliable UNKNOWN rejection is not implemented by
current AOSC state or these cosine scores.

## Persistent profile and limitations

The AOSC-only binary survived a file write, enrollment model destruction, disk
read, fresh model load, and import. The warm start retained channel behavior
without FIFO/timestamps/probability history. It is still a model-internal
cache snapshot with channel columns; it contains no identity labels by itself.
Mapping external IDs to those columns is a separate artifact. The full
snapshot includes far more stream-specific state and should not be treated as
a reusable person profile. A second, standalone invocation loaded the saved
profile after the enrollment process had exited. B again mapped P1→0 and P2→2
and left P3 on unmapped channel 1; C mapped P2 and P3 to P1's channel 0. The
process-restart report is included as raw JSON. This confirms file persistence
and repeatable import for this fixture, not cross-recording identity accuracy.

Main limitations:

- Only one same-meeting, same-microphone recording is available. No independent
  sessions or changed acoustic conditions were tested.
- Enrolled utterances total 16.39 s and 13.85 s, but the AOSC profile becomes
  exportable only after enough combined audio to complete cache compression.
- There are only two known test utterances, one unknown utterance, and a small
  set of reference turns. Counts are descriptive; FAR/FRR are not statistically
  estimated.
- Channel activity is not identity confidence. A channel mapping has no
  likelihood or margin calibrated for unknown speakers.
- AOSC compression updates after new audio. A temporary test stream can
  overwrite selected cache rows; persistent enrollment bytes must stay
  immutable unless a separately calibrated update policy is designed.
- The per-frame pre-encoder features may carry speaker cues, but their direct
  cosine scores overlap for known and unknown voices in this sample.

## Recommendation for Task 06

Keep the single Nemotron/GGML runtime. Before any production profile/API work,
collect multiple independent recordings per enrolled person and a sizeable
unknown-speaker calibration set across microphones, rooms, words, durations,
and session gaps. Hold calibration and final test people/recordings apart.
Evaluate per-turn identity and diarization separately, calibrate an explicit
unknown-rejection score only if held-out known/unknown distributions support
it, and test compression after sustained unknown traffic. If AOSC channels
continue to help but no score separates unknown people, document AOSC as a
within-stream diarization memory and consider an explicitly researched
single-model scoring head only if the checkpoint exposes one.

## Reproduction

Configure and build the CPU test profile, then run:

```bash
cmake --build build/cpu-diar -j4
build/cpu-diar/bin/diar_enrollment_probe \
  models/Nemotron-3-Diarization.q8_0.gguf \
  --enroll P1=test_files/diar/ami_en2002d_2132.wav@3.12:19.51 \
  --enroll P2=test_files/diar/ami_en2002d_2132.wav@29.20:43.05 \
  --test P1=test_files/diar/ami_en2002d_2132.wav@48.55:51.16 \
  --test P2=test_files/diar/ami_en2002d_2132.wav@45.39:47.95 \
  --test P3=test_files/diar/ami_en2002d_2132.wav@51.22:57.34 \
  --silence-ms 1000 \
  --profile /tmp/task05-dual.aosc \
  --snapshot /tmp/task05-dual.snapshot \
  --report tests/results/diar_enrollment_probe_ami_preliminary.json
```

The reverse-order and no-silence result artifacts were generated with these
commands:

```bash
build/cpu-diar/bin/diar_enrollment_probe models/Nemotron-3-Diarization.q8_0.gguf \
  --enroll P2=test_files/diar/ami_en2002d_2132.wav@29.20:43.05 \
  --enroll P1=test_files/diar/ami_en2002d_2132.wav@3.12:19.51 \
  --test P1=test_files/diar/ami_en2002d_2132.wav@48.55:51.16 \
  --test P2=test_files/diar/ami_en2002d_2132.wav@45.39:47.95 \
  --test P3=test_files/diar/ami_en2002d_2132.wav@51.22:57.34 \
  --silence-ms 1000 --profile /tmp/task05-reverse.aosc \
  --snapshot /tmp/task05-reverse.snapshot \
  --report tests/results/diar_enrollment_probe_ami_reverse_order.json

build/cpu-diar/bin/diar_enrollment_probe models/Nemotron-3-Diarization.q8_0.gguf \
  --enroll P1=test_files/diar/ami_en2002d_2132.wav@3.12:19.51 \
  --enroll P2=test_files/diar/ami_en2002d_2132.wav@29.20:43.05 \
  --test P1=test_files/diar/ami_en2002d_2132.wav@48.55:51.16 \
  --test P2=test_files/diar/ami_en2002d_2132.wav@45.39:47.95 \
  --test P3=test_files/diar/ami_en2002d_2132.wav@51.22:57.34 \
  --silence-ms 0 --profile /tmp/task05-no-silence.aosc \
  --snapshot /tmp/task05-no-silence.snapshot \
  --report tests/results/diar_enrollment_probe_ami_no_silence.json
```

The single-person cache-readiness probes were:

```bash
build/cpu-diar/bin/diar_enrollment_probe models/Nemotron-3-Diarization.q8_0.gguf \
  --enroll P1=test_files/diar/ami_en2002d_2132.wav@3.12:19.51 \
  --test P1=test_files/diar/ami_en2002d_2132.wav@48.55:51.16 \
  --silence-ms 1000 --report tests/results/diar_enrollment_probe_ami_p1_only.json

build/cpu-diar/bin/diar_enrollment_probe models/Nemotron-3-Diarization.q8_0.gguf \
  --enroll P2=test_files/diar/ami_en2002d_2132.wav@29.20:43.05 \
  --test P2=test_files/diar/ami_en2002d_2132.wav@45.39:47.95 \
  --silence-ms 1000 --report tests/results/diar_enrollment_probe_ami_p2_only.json
```

The RTTM-backed mixed run:

```bash
build/cpu-diar/bin/diar_enrollment_probe \
  models/Nemotron-3-Diarization.q8_0.gguf \
  --enroll MEE071=test_files/diar/ami_en2002d_2132.wav@3.12:19.51 \
  --enroll FEO070=test_files/diar/ami_en2002d_2132.wav@29.20:43.05 \
  --test MULTI=test_files/diar/ami_en2002d_2132.wav@45.39:59.92 \
  --rttm test_files/diar/ami_en2002d_2132.rttm \
  --silence-ms 1000 \
  --profile /tmp/task05-multi.aosc \
  --snapshot /tmp/task05-multi.snapshot \
  --report tests/results/diar_enrollment_probe_ami_multispeaker.json
```

To reproduce a separate-process profile import, first run the primary command
above to create `/tmp/task05-dual.aosc`, then start a new probe process:

```bash
build/cpu-diar/bin/diar_enrollment_probe \
  models/Nemotron-3-Diarization.q8_0.gguf \
  --load-profile /tmp/task05-dual.aosc \
  --identity-channel P1=0 --identity-channel P2=2 \
  --test P1=test_files/diar/ami_en2002d_2132.wav@48.55:51.16 \
  --test P2=test_files/diar/ami_en2002d_2132.wav@45.39:47.95 \
  --test P3=test_files/diar/ami_en2002d_2132.wav@51.22:57.34 \
  --report tests/results/diar_enrollment_probe_ami_profile_restart.json
```

Raw JSON artifacts:

- [Normal enrollment order](../tests/results/diar_enrollment_probe_ami_preliminary.json)
- [Reverse enrollment order](../tests/results/diar_enrollment_probe_ami_reverse_order.json)
- [No silence between enrollment clips](../tests/results/diar_enrollment_probe_ami_no_silence.json)
- [RTTM-backed mixed-speaker interval](../tests/results/diar_enrollment_probe_ami_multispeaker.json)
- [P1-only enrollment](../tests/results/diar_enrollment_probe_ami_p1_only.json)
- [P2-only enrollment](../tests/results/diar_enrollment_probe_ami_p2_only.json)
- [Separate-process profile reload](../tests/results/diar_enrollment_probe_ami_profile_restart.json)

The model, test WAVs, RTTM, and generated profiles are not added by this
research report; the JSON files contain probabilities, scores, and assignments
only.
