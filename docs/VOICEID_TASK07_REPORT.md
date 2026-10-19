# Task 07 — Native AOSC speaker matching and counterfactual research

## Decision

**B — Native Matching Promising.** The existing inference path preserved
enrollment channel assignments for two known AMI query clips, and shape-
preserving cache interventions changed those outputs. The reactions were
asymmetric, however, and matched non-target cache changes sometimes damaged
the target speaker more than removing that speaker's own rows. The only
available data is one meeting, one session, and one microphone. There are no
development, calibration, or independent final-holdout recordings. The
observations are preliminary evidence of a context signal, not evidence of
reliable speaker verification or calibrated UNKNOWN rejection.

The results are in
[`tests/results/diar_native_matching_ami_preliminary.json`](../tests/results/diar_native_matching_ami_preliminary.json)
and the separate-process import check is in
[`tests/results/diar_native_matching_restart_ami.json`](../tests/results/diar_native_matching_restart_ami.json).
The JSON includes raw frame-level speaker probabilities from the fixture as
well as aggregate scores. No audio or AOSC profile is committed.

## How the implementation uses AOSC

### Documented behavior

The V3 stream carries 512-dimensional pre-encoder acoustic rows in its speaker
cache and maintains aligned per-channel `spkcache_preds` for those rows. The
cache represents selected temporal context, not an utterance-level biometric
embedding. Prediction columns keep cache rows associated with the model's
session-local speaker channels. FIFO embeddings are the recent context waiting
to enter the cache. The AOSC-only profile stores cache features, prediction
metadata, silence state, and model/streaming-geometry fingerprints; it does
not contain external identity labels.

### Directly observed in this implementation

`SortformerGraph::build_graph()` concatenates `input.state` and current
feature-stack output along the time axis before the RoPE Transformer. The
batcher supplies speaker-cache rows followed by FIFO rows as `input.state`,
then supplies current chunk features. It assigns positions across the whole
joined sequence. Each Transformer block projects Q/K/V from that sequence and
calls fused flash attention with a null mask. Thus saved acoustic rows and
current rows participate in the same self-attention computation. The graph
does not pass `spkcache_preds` to the Transformer.

The speaker head converts logits with an independent sigmoid per channel.
Those values are diarization activity outputs; they are neither normalized
across channels nor exposed as an explicit enrollment-identity verification
score. The streaming updater recomputes FIFO probabilities, appends aligned
features and prediction rows, moves the oldest FIFO rows into the cache, and
compresses once the configured cache capacity is exceeded. Compression uses
per-channel prediction columns, a `> 0.5` speech mask, score boosts, silence
sentinels, and speaker-major row selection. It retains the acoustic row and
its prediction metadata together. Channel indices are local to an enrollment
context and depend on enrollment order.

An AOSC-only import starts a new session with an empty FIFO; a full snapshot
also restores the prior FIFO and stream timeline. Cache interventions
therefore isolate persisted AOSC rows at fixed cache geometry, but they do not
isolate any FIFO-only effect.

### Interpretation and hypothesis

It is an implementation-based inference that cached acoustic rows can affect
current channel assignments through Transformer self-attention. The model may
use those rows as temporal context for consistent diarization. The graph
provides no attention-weight output in this path, so this experiment makes no
claim that any attention weight is a biometric similarity. There is no
separate speaker-verification score in the existing inference graph.

## Harness and data protocol

The research-only probe is
[`tests/cpp/diar/diar_native_matching_probe.cpp`](../tests/cpp/diar/diar_native_matching_probe.cpp).
It creates new streaming sessions per condition and query, imports the
immutable AOSC-only profile, maps external IDs from labeled enrollment
activity, measures raw probabilities only over reference speech frames, and
discards temporary states after evaluation. The intervention helpers are
isolated in
[`tests/cpp/diar/aosc_profile_intervention.h`](../tests/cpp/diar/aosc_profile_intervention.h).
They are not used by normal inference. On POSIX, the exported profile is
created with mode `0600`.

The AMI manifest supports enrollment, development, calibration, final holdout,
known and unknown query categories, recording/session/microphone IDs, clean or
overlap labels, and source intervals. Strict manifest validation rejects
overlapping source intervals across partitions for the same recording. The
checked-in fixture is deliberately marked preliminary and has:

| Partition / group | Clips | Data |
|---|---:|---|
| Enrollment | 2 | P1: 3.12–19.51 s; P2: 29.20–43.05 s |
| Development | 0 | Not available |
| Calibration | 0 | Not available |
| Final holdout | 0 | Not available |
| Known queries | 2 | P1: 48.55–51.16 s; P2: 45.39–47.95 s, overlap-labeled |
| Unknown queries | 1 | P3: 51.22–57.34 s |

All intervals come from `ami_en2002d_2132`, one AMI Mix-Headset recording,
session, and microphone. Query intervals do not overlap enrollment intervals.
This supports a regression experiment, not an estimate of real-world
identification performance.

The model is
`Nemotron-3-Diarization.q8_0.gguf`, SHA-256
`08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`.
Streaming used the V3 preset, 16 kHz CPU inference, four threads, 160 ms
pushes, 264 cache rows, 80 FIFO rows, 13 chunk rows, 40-frame update period,
left context 0, and right context 1. The streaming fingerprint is
`2fa5184871d8e974`.

## Native matching baseline

Combined enrollment produced a 264 × 512 AOSC cache (551,305-byte profile).
The observed mapping was P1 → channel 0 and P2 → channel 2. Per-channel
`spkcache_preds > 0.5` row counts were `[136, 10, 87, 0, 0, 0, 0, 0]`; the
argmax-only summary was `[136, 7, 87, 0, 0, 0, 0, 0]`. The argmax summary is
descriptive only: cache rows may carry multiple positive channel predictions.
The profile is immutable during every query.

| Query | Registered channel activity | Top-1 assignment on speech frames | Top-1 / top-2 margin | Top-1 stability |
|---|---|---|---:|---:|
| P1 known, 2.61 s | P1 ch0: 0.8931; P2 ch2: 0.0015 | ch0 on 261/261 frames | 0.8912 | 1.000, 0 switches |
| P2 known, 2.56 s | P1 ch0: 0.0017; P2 ch2: 0.9387 | ch2 on 116/116 frames | 0.9011 | 1.000, 0 switches |
| P3 unknown, 6.12 s | P1 ch0: 0.00016; P2 ch2: 0.0031 | unregistered ch1 on 612/612 frames | 0.5693 | 1.000, 0 switches |

On the 23 RTTM-overlap frames inside the P2 query, channel 2 remained top-1
for 23/23 frames (mean activity 0.9128); channel 1 had mean activity 0.1554.
The mean top-1/top-2 margin in this subset was 0.7574. This is a small
diarization observation, not an overlap UNKNOWN-rejection test.

A 3-second silence query against the immutable AOSC profile had maximum raw
channel activity `0.000549`. Existing silence sentinel rows were preserved in
the interventions. No standalone sentinel-only replacement was used because
changing those learned silence features would create another non-semantic
cache perturbation.

The forced top-1 result for P3 in the full context was an unregistered
channel, but forced assignment is not an acceptance decision. In an empty
context, P2 and P3 both selected P1's channel 0. That gives 1/2 known queries
on the expected enrolled channel and a false enrolled-channel assignment for
the unknown. The same P3 was forced onto P1 after P1 ablation, after both
registered channels were ablated, and in the 1.5-second fresh-context clip.
The full-context sample therefore does not establish UNKNOWN rejection.

## Counterfactual interventions

### Intervention validity

Deleting cache rows would change tensor geometry and all later RoPE positions,
so it was not used as a valid target ablation. Instead, the probe keeps the
fixed cache shape and row positions. For selected rows it replaces the
512-dimensional feature with the profile's mean silence vector and zeros the
aligned prediction row. Target rows are selected by that channel's saved
prediction being `> 0.5`, not by argmax; this preserves multi-channel overlap
rows. The matched control changes the same number of rows from the other
registered channel, restricted to rows where the target prediction is
`<= 0.5`. Feature/prediction pairs remain aligned.

These are controlled edits to a serialized model context, not exact
counterfactual removal of a person's voice. A temporal feature can carry
information about multiple speakers or recording conditions, while the
prediction metadata affects how a later cache update treats the rows. In
addition, self-attention can redistribute effects across every output frame.
The edits preserve tensor dimensions and cache metadata alignment, but they do
not guarantee a semantically neutral or causally isolated change.

The profile contained 136 P1-positive rows, 87 P2-positive rows, and 87
P2-only rows available as a row-count-matched neutral control for full P1
ablation. There were 136 P1-only rows, enough to match full P2 ablation. A
full-P1 versus equal-count neutral comparison was unavailable, so that
comparison uses 87-row partial interventions for both sides.

### Measured response

| Query | Intervention | Activity on that query's registered channel | Top-1 outcome |
|---|---|---:|---|
| P1 | Full context | 0.8931 on ch0 | P1 ch0, 261/261 |
| P1 | Remove all P1-positive rows (136) | 0.0423 on ch0; Δ full-minus-ablated = 0.8509 | switches to P2 ch2 |
| P1 | Remove 87 P1 rows | 0.6088 on ch0; Δ = 0.2843 | remains P1 ch0 |
| P1 | Matched control: remove 87 P2-only rows | 0.8906 on ch0; Δ = 0.0025 | remains P1 ch0 |
| P2 | Full context | 0.9387 on ch2 | P2 ch2, 116/116 |
| P2 | Remove all P2-positive rows (87) | 0.8244 on ch2; Δ = 0.1143 | remains P2 ch2 |
| P2 | Matched control: remove 87 P1-only rows | 0.0807 on ch2; Δ = 0.8579 | no longer top-1 on a registered channel |

The P1 result is a useful, target-sensitive signal in this fixture: the
partial P1 intervention caused a 0.2843 activity reduction, while its
matched P2-only control reduced activity by 0.0025. P2 is the counterexample
to a general identity score: removing P2 affected it less than removing the
same number of P1-only rows. Removing P1 also lowered P2's channel activity
from 0.9387 to 0.0412 and P3 then chose registered P1. These cross-speaker
changes show broad context disruption as well as target sensitivity.

Removing both registered channels caused P1's query to switch among channels
(top-1 stability 0.517, four switches; registered P1 top-1 on 126/261
frames). P3 selected registered P1. Partial interventions also changed P3's
top-1 channel and reduced its top-1 margin; its 87-row matched P1 intervention
caused 27 top-1 switches and a 0.0544 margin. The matched P2-only control
caused no such changes in P3. These are output differences, not an identity
likelihood.

The neutral-control effects are asymmetric and large. Consequently, the
observed P1 partial effect cannot be generalized to P2, and the larger P2
control effect is evidence that context deletion can degrade diarization
globally. `counterfactual_scores` in the JSON are raw activity deltas and must
not be interpreted as probabilities of identity.

No attention maps were captured: the current path uses fused attention and
does not return weights. No FIFO-only ablation or independent silence-row
count/position sweep was performed; the AOSC-only profile does not serialize
FIFO rows, and a valid full-snapshot intervention would need to preserve its
timeline, positions, and scheduler state. The measured row-count and order
controls therefore characterize AOSC profile sensitivity only.

### Position and enrollment-order controls

Reversing enrollment order changed the external ID mapping to P1 → 2 and P2 →
0. Full-context queries still followed the newly measured mapping. This
confirms that channels are local to an enrollment context. Reversing cache row
order while preserving each feature/prediction pair changed assignments:
P1 selected channel 2, P2 selected channel 0, and P3 mostly selected channel
1. This is a sensitivity control for the positional encoding and
speaker-major cache layout, not a valid alternate enrollment profile.

Four repeated P3 query passes changed temporary cache argmax row counts from
`[136, 7, 87]` to `[117, 91, 48]`; the immutable source profile remained
unchanged. The following P1 and P2 queries still selected their mapped
channels in this short sequence. This shows that unknown traffic can occupy
and change temporary cache state; the repeated clip is not diverse enough to
test long-run retention.

The 1.5-second crops produced 150, 60, and 150 evaluated speech frames for
P1, P2, and P3. Full AOSC context preserved the expected P1/P2 channels and
put P3 on channel 1. Fresh context instead assigned P2 and P3 to P1. P1
ablation shifted the short P1 query to P2 and assigned short P3 to P1. These
very short fixture crops are descriptive; they do not set a duration policy.

## Pairwise native context matching

The harness attempted independent P1-only and P2-only enrollment contexts
using the available labeled clips. They were not AOSC-ready: P1 had 16.39 s
and P2 13.85 s, both shorter than the first V3 compression point (about
27.52 s at the recorded geometry). The pairwise comparison was therefore not
run. It did not duplicate embeddings or fabricate cache rows. Empty context
and combined P1+P2 context were compared as fresh and full AOSC conditions.

The Task 05 full-snapshot experiment on these same disjoint AMI intervals
also mapped P1/P2 to channels 0/2 and P3 to channel 1, matching AOSC-only
warm start on this fixture. Task 07's separate-process AOSC-only profile
import produced bit-identical raw probabilities for all three full-context
query clips compared with the exporting process. This confirms profile
round-trip determinism for the fixture, not identity generalization.

## Cosine baselines and UNKNOWN rejection

Task 06 mean-pooling scores on the same AMI queries were:

| Baseline | Known rank-1 | Mean best-known cosine | Best unknown cosine |
|---|---:|---:|---:|
| `pre_encoder` | 2/2 | 0.9788 | 0.9923 |
| `encoder_proj` | 2/2 | 0.9084 | 0.9344 |
| Activity/quality weighted `pre_encoder` | 2/2 | — | 0.9876 |
| AOSC-selected `pre_encoder` | 2/2 | — | 0.9872 |
| Native full-context channel activity | 2/2 expected mapped channels | P1 0.8931; P2 0.9387 | P3 assigned new channel 1 |

The cosine values rank known speakers correctly in this tiny sample, but the
unknown cosine exceeds the mean best-known score for both principal
representations. The native full-context output assigned this P3 to an
unregistered channel, but the empty-context and targeted-ablation runs
falsely assigned P3 to P1. These measures are not on a shared scale, and
native channel activity is not an identity probability. Neither method
provides a calibrated target/non-target score here.

There were zero calibration and final-holdout queries, so the threshold is
null. No FAR, FRR, ROC, or EER claim is possible. The forced full-context
top-1 fixture count was 2/2 known queries on their enrollment channels and
0/1 unknown queries on an enrolled channel, but this is one meeting crop and
not calibrated rejection. Other contexts produced false enrolled-channel
assignments for the unknown. UNKNOWN rejection remains unvalidated.

## Performance and real-time implications

Measured on the CPU backend with four threads:

| Measurement | Time / size |
|---|---:|
| Mean fresh-session inference | 398 ms |
| Mean full AOSC warm-start inference | 3,407 ms |
| Mean targeted-ablation inference | 3,328 ms |
| Estimated 2-context pairwise pass | 6,815 ms |
| Estimated 10-context pairwise pass | 34,074 ms |
| Model-loaded process RSS | 116.1 MB |
| One AOSC profile | 551,305 bytes |
| Estimated storage for 2 / 10 profiles | 1.10 / 5.51 MB |

The pairwise timings scale the measured full-context inference by the number
of contexts; independent profiles could not be prepared from the current
single-person clips. They are rough serial CPU estimates, not measured
multi-person runs. Full-context inference took longer than the 2.6-second
queries in this setup. Re-running the model for each enrolled person scales
linearly with enrollment count and is not a practical real-time matching
strategy on this CPU configuration. Other hardware and continuous-streaming
amortization were not benchmarked.

## Regression checks and scope

The intervention unit test verifies fixed cache/prediction tensor sizes,
aligned feature/metadata replacement, matched changed-row counts, paired row
reordering, and checksum rejection. The probe, intervention helper, manifest,
and result artifacts live under test/research paths. There are no production
source, GGUF, AOSC compression, snapshot/restore, or public C ABI changes.
All six CTest cases passed, including ordinary C API model smoke, snapshot
continuity with JFK and AMI fixtures, silence/compaction, AOSC updates, and
streaming restore. No research hook was added to normal inference. Task 06
representation-probe parity is rerun as part of
the reproduction sequence below. That parity rerun compared 33,264 raw
probability values across 5 clips with zero mismatches; channel assignments,
segments, AOSC profiles, and snapshots matched in 5/5 cases. With two repeats,
the research probe's measured mean delta was 5.10 ms (0.26%) on this fixture,
and repeated probabilities/features were bit-identical.

## Reproduction

From the repository root, with the local model and AMI fixture available:

```sh
cmake --build build/cpu-diar --target diar_native_matching_probe test_aosc_profile_intervention --parallel 4
build/cpu-diar/bin/test_aosc_profile_intervention
build/cpu-diar/bin/diar_native_matching_probe \
  tests/manifests/task07_native_ami_preliminary.json \
  --report tests/results/diar_native_matching_ami_preliminary.json \
  --profile-out /tmp/task07-dual.aosc --threads 4
build/cpu-diar/bin/diar_native_matching_probe \
  tests/manifests/task07_native_ami_preliminary.json \
  --profile-in /tmp/task07-dual.aosc \
  --identity-channel P1=0 --identity-channel P2=2 \
  --report tests/results/diar_native_matching_restart_ami.json --threads 4
diff -u \
  <(jq -c '[.results[]|select(.method=="full_aosc_warm_start")|{clip_id,raw_speech_probabilities}]' tests/results/diar_native_matching_ami_preliminary.json) \
  <(jq -c '[.results[]|select(.method=="full_aosc_warm_start")|{clip_id,raw_speech_probabilities}]' tests/results/diar_native_matching_restart_ami.json)
build/cpu-diar/bin/diar_representation_probe \
  tests/manifests/task06_ami_preliminary.json --repeat 2 \
  --report /tmp/task07-task06-parity.json
ctest --test-dir build/cpu-diar --output-on-failure
jq empty tests/results/diar_native_matching_ami_preliminary.json \
  tests/results/diar_native_matching_restart_ami.json
git diff --check
```

The separate-process `diff` should be empty. Profile artifacts stay in
`/tmp`; the committed result JSON records the public AMI fixture's raw model
probabilities and aggregate metrics.

## Recommendation for VoiceID Engine

Continue using AOSC as streaming diarization context and treat its channel
mapping as local to the enrollment stream. The experiment supports exploring
native context signals as a research feature, but it does not support treating
speaker-channel activity or counterfactual deltas as identity confidence.
Before a verification design can be selected, gather multiple independent
sessions and microphones for known speakers plus new unknown speakers, keep
enrollment/calibration/final holdout audio disjoint, calibrate thresholds on
the calibration partition, and evaluate once on the frozen holdout. Keep
explicit identity-to-channel metadata outside AOSC and return UNKNOWN only
through a separately validated rejection rule.
