# Task 08 — VCTK Open-Set VoiceID Validation

## Decision

**B — Promising but Limited.** Native AOSC channel matching identified evaluated enrolled queries reliably in these controlled VCTK partitions, but raw open-set rejection was inadequate and profile-to-identity mapping failed often enough to remove a material fraction of trials. At eight enrolled identities, the eight trained channels are all registered, leaving no unregistered channel for the baseline UNKNOWN rule. A calibrated activity threshold improves rejection, at the cost of rejecting many known queries.

This is a controlled read-speech experiment. Utterance-disjoint VCTK partitions do not establish cross-session, cross-device, or real-world biometric verification.

## Scope and reproducibility

The experiment starts from Task 07.5 commit `05010f6` on branch `research/vctk-open-set-voiceid`, uses `models/Nemotron-3-Diarization.q8_0.gguf` (SHA-256 `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`), and runs the existing Nemotron inference path with persisted AOSC. It adds no encoder, changes no model weights, and does not add a production VoiceID API.

The reusable pipeline is [the VCTK preparation and validation utility](../tests/tools/prepare_vctk_voiceid.py), [the native C++ probe](../tests/cpp/diar/diar_vctk_voiceid_probe.cpp), [the main aggregation utility](../tests/tools/summarize_vctk_voiceid.py), [the staged-traffic aggregator](../tests/tools/summarize_vctk_traffic_stages.py), and [the compact-plan writer](../tests/tools/compact_vctk_voiceid_manifest.py). Compact manifests and aggregate results are committed; audio, converted WAVs, profiles, expanded manifests, and frame traces stay outside Git.

The pilot used seed `20261020`, 18 profiles, 210 isolated-query attempts, and all 36 planned pilot conversations. It completed on Vulkan0. It exercised two AOSC-not-ready profiles and one ambiguous four-speaker mapping; those were reported as construction/mapping failures and not counted as identification attempts. All 18 trial result records and all 36 planned conversation traces were present. The pilot’s 12 unknown-traffic sequences returned the known speaker correctly. See the [pilot compact plan](../tests/manifests/task08_vctk_pilot_plan.json) and [pilot aggregate](../tests/results/voiceid_task08_pilot.json).

The bounded full plan contains 288 profiles, 10,440 isolated-query attempts, and 216 conversation specifications. It uses 2, 4, and 8 enrolled identities; requested enrollment budgets of 5, 10, 15, 30, 45, and 60 seconds per person; four independently seeded cohorts per group size; and reversed enrollment order for the 30- and 60-second conditions. The expanded manifest SHA-256 is `be9928d3af71dfde838d950165896105a06a4dbc7b7cff41ae751066e5b25fae`; its preparation command and compact trial plan are recorded in [the full plan](../tests/manifests/task08_vctk_full_plan.json). The manifest validator reported zero leakage or integrity errors, and the full result set contains all 288 trial IDs.

Recreate the full manifest with:

```sh
python3 tests/tools/prepare_vctk_voiceid.py prepare \
  --dataset-root VCTK-Corpus-0.92 \
  --cache-dir ~/.cache/nemo-speech-task08/vctk \
  --output /tmp/task08_vctk_full/manifest.json \
  --preset full --seed 20261020 --group-sizes 2,4,8 \
  --budgets 5,10,15,30,45,60 --cohorts-per-size 4 \
  --unknown-per-cohort 4 --mic2-robustness --conversations
```

The probe ran with one model load per process, Vulkan device 0, the pinned NVIDIA Vulkan ICD, four inference threads, concurrency 4, and batching enabled. Every full-run event identified the selected backend as `Vulkan0`; the harness rejects a requested Vulkan run if the selected GPU backend is absent. CPU was used for the representative parity subset only.

The native harness checkpoints one JSONL result per completed trial, validates the manifest fingerprint on resume, and accepts repeated `--trial-id` selections. The staged contamination replay uses this selector and `--unknown-traffic-only` to run only the requested experiment family.

## Output geometry preflight

The loaded GGUF reports `sortformer.num_speakers = 8`, output subsampling factor 1, and a prediction stride of `0.009999999776` seconds. The loaded runtime reports `cfg.num_speakers = 8` and `seconds_per_output_frame() = 0.009999999776` seconds.

For the known 60.0-second, 16 kHz, mono AMI WAV used for the geometry preflight:

- The final graph-call prediction tensor was `[channels, native_frames, batch] = [8, 2680, 1]`.
- The accumulated stream contained 6,001 output frames. The 1-frame boundary effect is implementation-specific; the observed rate was 100.0167 frames per input second.
- The C++ probability vector was frame-major with logical shape `[6001, 8]` and 48,008 float values.
- Top-1 comparisons use `probabilities[frame * num_speakers + channel]`, channels 0 through 7.
- Task 08 traces use `NMSP8F1` v1: an explicit 32-byte header with channel count, frame count, and frame period, followed by little-endian float32 frame-major `[frame, channel]` values. The legacy `--dump-probs` file is headerless but also frame-major. JSON research exports store eight channel values per frame. There is no native C++ CSV probability exporter; an external CSV conversion must reshape using `cfg.num_speakers`.

The Task 07.5 report treated the same float sequence as 12,002 frames by 4 channels. This was an exporter/analysis reshape error, not a runtime tensor bug. The loaded model dimensions, runtime vector, and segmenter were already eight-channel. A fail-fast runtime shape check now verifies the prediction tensor channel dimension against `sortformer.num_speakers`; it does not change trained output dimensions.

The affected Task 07.5 top-1 CPU/Vulkan comparisons were recalculated with eight channels:

| Fixture | Changed top-1 frames |
|---|---:|
| JFK | 0 / 1,101 |
| AMI | 30 / 6,001 (0.4999%) |
| Silence | 0 / 801 |
| p225 | 0 / 206 |
| p226 | 0 / 280 |
| p227 | 0 / 343 |

The 30 AMI changes are locally explained by numerical differences at near-tie frames. Element-wise error metrics were unchanged because they compare the same values. Updated geometry and comparison details are in the [Task 07.5 report](VOICEID_TASK07_5_VULKAN_REPORT.md) and [Task 07.5 aggregate JSON](../tests/results/voiceid_task07_5_vulkan.json). The preflight facts are also saved in [the Task 08 preflight record](../tests/results/voiceid_task08_preflight.json).

## VCTK inventory

The inventory utility inspected the actual `wav48_silence_trimmed` tree and FLAC STREAMINFO blocks rather than assuming a speaker count.

| Inventory item | Observed value |
|---|---:|
| FLAC files | 88,328 |
| Readable STREAMINFO headers | 88,328 / 88,328 |
| Audio speaker directories | 110 (111 directories including the audio root) |
| Speakers with mic1 audio | 110 |
| Speaker/utterance identities | 44,455 |
| Identities with both mic1 and mic2 | 43,873 |
| Single-microphone identities | 582; all mic1-only |
| Total FLAC duration, both microphones | 297,552.225 s (82.65 h) |
| Per-speaker mic1 duration | 399.076–1,828.577 s; median 1,341.762 s |
| Clip duration | 1.219667–16.556875 s; median 3.124292 s; p90 4.652083 s |
| Sample rate / channels / bit depth | 48 kHz / mono / 16-bit for every file |
| Transcript speaker directories | 109 |
| Transcript identities without mic1 audio | 300 |
| Audio identities without matching transcript | 172 |

Each speaker has 123–502 mic1 utterances (median 416.5). Mic2 has 0–502 utterances per speaker; paired mic variants are grouped by speaker and utterance ID. Mic1 is the canonical evaluation microphone. Mic2 is a separate robustness condition, not an independent utterance or session. The per-speaker inventory is in [the compact inventory JSON](../tests/manifests/task08_vctk_inventory.json).

## Preparation and partitions

Selected FLACs are converted lazily and cached outside Git at `~/.cache/nemo-speech-task08/vctk`. FFmpeg converts to mono, 16 kHz, PCM16 with `-map_metadata -1 -ac 1 -ar 16000 -c:a pcm_s16le`. The manifest records original relative path, speaker ID, utterance ID, microphone, source and converted SHA-256, duration, conversion command/version, and VAD-mask hash. The same cached WAV is used for every backend and condition.

The deterministic 10 ms VAD uses 20 ms RMS windows at 10 ms hop. Its threshold is the greater of −45 dBFS and 28 dB below the peak-window RMS; it bridges gaps up to 60 ms and removes runs shorter than 30 ms. Query buckets and enrollment budgets use actual VAD speech frames, not only file/container duration. Enrollment clip selection can exceed a requested budget because clips are indivisible; across the full randomized profiles, per-person median usable durations were approximately 6.0, 11.0, 16.0, 31.0, 46.2, and 61.0 seconds for the six requested budgets.

Speaker populations are disjoint: 16 development, 33 calibration, and 61 final-holdout speakers. Within each speaker, utterance identities are partitioned 60% enrollment, 10% development, 15% calibration, and 15% final holdout. Mic1/mic2 variants of an utterance stay together. All speakers in each population are represented as known or unknown across the seeded cohorts; no speaker was left unrepresented. Final-holdout unknown identities were not used to fit rejection thresholds. VCTK does not provide independent session labels for this experiment, so utterance partitioning must not be described as cross-session validation.

## Enrollment readiness, mappings, and cache rows

The requested duration is a per-person speech budget, not a guarantee that an AOSC-only profile is ready. A profile is ready only when the native cache can be encoded after its required update/compression. Of 288 full profiles, 278 became AOSC-ready; the other 10 all failed because the cache had not completed its first compression (the two-speaker, 5-second condition across the randomized cohorts and populations). A failed profile is not counted as an identification error.

Among ready or attempted profiles, 224/288 had an unambiguous identity-to-channel map, and 216/288 had both a ready profile and a valid map. The 64 mapping failures were 40 channel collisions and 24 cases where an enrolled speaker did not establish a stable channel from VAD speech frames. Mapping requires a unique candidate channel per external ID; IDs are inferred from enrollment evidence, never assigned by fixed channel number.

Randomized-profile readiness and mapping counts below pool the three disjoint speaker populations and four cohorts per group size (12 profiles per cell):

| Enrolled speakers | AOSC-ready profiles, budgets 5/10/15/30/45/60 s | Valid maps, budgets 5/10/15/30/45/60 s |
|---:|---|---|
| 2 | 2/12, 12/12, 12/12, 12/12, 12/12, 12/12 | 10/12, 10/12, 10/12, 11/12, 11/12, 12/12 |
| 4 | 12/12 for all six budgets | 10/12, 8/12, 9/12, 11/12, 11/12, 10/12 |
| 8 | 12/12 for all six budgets | 5/12, 6/12, 5/12, 6/12, 8/12, 8/12 |

For the final-holdout, randomized 60-second profiles, readiness was 4/4 at each group size. Valid maps were 4/4 for two speakers, 2/4 for four, and 4/4 for eight. In those profiles, channel establishment succeeded for all enrolled people (8/8, 16/16, and 32/32 speaker-profile attempts respectively), but four-speaker profiles still had two identity-map failures. This separates channel establishment from collision-free mapping.

Ready profiles held 264 AOSC cache frames. For the successful final-holdout 60-second profiles, retained positive rows on each enrolled candidate channel ranged from 48–208 (median 128) for two speakers, 48–160 (median 48) for four, and 24–48 (median 24) for eight. Across all mapped identities in the full run, retained candidate-channel rows ranged 14–208 (median 48). No saved cache row had multiple channel metadata values above 0.5 in this experiment. The aggregate records candidate channels and row counts even when two identities collide, so ambiguous enrollment is visible rather than hidden.

## Isolated open-set identification

The native baseline uses the dominant top-1 output channel over VAD speech frames where the winning channel is at least 0.5. A mapped channel yields its enrolled external ID; an unmapped channel yields `UNKNOWN`; no active frame or a tied dominant channel yields no decision. This is a diarization activity rule, not a calibrated identity posterior. Query outcomes below are the primary mic1 randomized final-holdout results. The readiness/map column pools the 12 randomized profiles across development, calibration, and final holdout for each group-size/budget condition. `Not evaluated` attempts were skipped because the profile was not ready or its mapping was ambiguous; they are not false rejections.

| Enrolled | Requested s/person | Ready / map valid (12) | Known correct / evaluated (not evaluated); wrong | Unknown rejected / evaluated; false accept; no decision (not evaluated) |
|---:|---:|---:|---|---|
| 2 | 5 | 2/12; 10/12 | 4/4 (15); 0 | 8/11; 3; 0 (31) |
| 2 | 10 | 12/12; 10/12 | 15/15 (4); 0 | 22/32; 10; 0 (10) |
| 2 | 15 | 12/12; 10/12 | 15/15 (4); 0 | 22/32; 10; 0 (10) |
| 2 | 30 | 12/12; 11/12 | 19/19 (0); 0 | 27/42; 15; 0 (0) |
| 2 | 45 | 12/12; 11/12 | 19/19 (0); 0 | 29/42; 13; 0 (0) |
| 2 | 60 | 12/12; 12/12 | 19/19 (0); 0 | 28/42; 13; 1 (0) |
| 4 | 5 | 12/12; 10/12 | 41/41 (0); 0 | 32/41; 9; 0 (0) |
| 4 | 10 | 12/12; 8/12 | 20/20 (21); 0 | 17/22; 5; 0 (19) |
| 4 | 15 | 12/12; 9/12 | 20/20 (21); 0 | 19/22; 3; 0 (19) |
| 4 | 30 | 12/12; 11/12 | 41/41 (0); 0 | 29/41; 12; 0 (0) |
| 4 | 45 | 12/12; 11/12 | 30/30 (11); 0 | 22/32; 10; 0 (9) |
| 4 | 60 | 12/12; 10/12 | 20/20 (21); 0 | 19/22; 3; 0 (19) |
| 8 | 5 | 12/12; 5/12 | 79/79 (0); 0 | 0/41; 39; 2 (0) |
| 8 | 10 | 12/12; 6/12 | 78/79 (0); 1 | 0/41; 41; 0 (0) |
| 8 | 15 | 12/12; 5/12 | 79/79 (0); 0 | 0/41; 39; 2 (0) |
| 8 | 30 | 12/12; 6/12 | 78/79 (0); 1 | 0/41; 38; 3 (0) |
| 8 | 45 | 12/12; 8/12 | 77/79 (0); 2 | 0/41; 38; 3 (0) |
| 8 | 60 | 12/12; 8/12 | 79/79 (0); 0 | 0/41; 40; 1 (0) |

Across these evaluated conditions, known-speaker rejection as UNKNOWN and known no-decision counts were both zero; known wrong-enrolled assignments are listed in the table. Unknown no-decisions and false accepts are shown separately.

For the 60-second final-holdout condition, known-query correctness was 19/19 at two speakers, 20/20 among the 20 evaluated attempts at four speakers (21 skipped), and 79/79 at eight. Unknown rejection was 28/42, 19/22, and 0/41 respectively. Two-way bootstrap intervals resampled cohort and source-speaker clusters (5,000 requested iterations; frames were never treated as independent attempts):

| Enrolled | Known correct rate, 95% cluster interval | Unknown rejection rate, 95% cluster interval |
|---:|---|---|
| 2 | 19/19; [1.00, 1.00] (4 cohorts, 8 speakers) | 28/42; [0.345, 0.929] (4 cohorts, 16 speakers) |
| 4 | 20/20; [1.00, 1.00] (2 cohorts, 8 speakers) | 19/22; [0.600, 1.000] (2 cohorts, 8 speakers) |
| 8 | 79/79; [1.00, 1.00] (4 cohorts, 32 speakers) | 0/41; [0.00, 0.00] (4 cohorts, 12 speakers) |

The four-speaker interval is based on only two evaluable final-holdout cohorts because the other two maps were ambiguous. Perfect known accuracy therefore describes the evaluated profiles, not all planned profiles. Per-speaker and per-cohort counts, duration conditions, all enrollment budgets, and bootstrap details are in [the full aggregate JSON](../tests/results/voiceid_task08_full.json).

For 60-second mic1 queries, the native baseline by VAD duration bucket was:

| Enrolled | Query duration | Known correct / evaluated | Unknown rejected / evaluated (false accepts, no decisions) | Known / unknown not evaluated |
|---:|---|---:|---|---:|
| 2 | <2 s | 6/6 | 8/14 (5, 1) | 0 / 0 |
| 2 | 2–5 s | 8/8 | 11/16 (5, 0) | 0 / 0 |
| 2 | 5–10 s | 5/5 | 9/12 (3, 0) | 0 / 0 |
| 4 | <2 s | 6/6 | 5/7 (2, 0) | 7 / 7 |
| 4 | 2–5 s | 8/8 | 7/8 (1, 0) | 8 / 8 |
| 4 | 5–10 s | 5/5 | 7/7 (0, 0) | 6 / 4 |
| 4 | >10 s | 1/1 | no unknown query | 0 / 0 |
| 8 | <2 s | 24/24 | 0/13 (12, 1) | 0 / 0 |
| 8 | 2–5 s | 32/32 | 0/16 (16, 0) | 0 / 0 |
| 8 | 5–10 s | 22/22 | 0/12 (12, 0) | 0 / 0 |
| 8 | >10 s | 1/1 | no unknown query | 0 / 0 |

There were 80 >10-second attempts in the full manifest, so that bucket is sparse. Mic2 is reported separately because those files are microphone variants of grouped utterance identities. At 60 seconds, mic2 known queries were 17/17 correct at two speakers, 14/14 among 14 evaluated at four (14 skipped), and 62/62 at eight. Unknown rejection was 18/27 (8 false accepts, 1 no decision), 10/13 (3 false accepts, 17 skipped), and 0/28 (24 false accepts, 4 no decisions), respectively.

With eight enrolled speakers, a valid one-to-one map necessarily occupies all eight trained output channels. There is no unregistered channel for the baseline to identify as UNKNOWN. In that condition, the model rejected 0/41 unknown mic1 queries; 40 were false accepts and one had no decision. Increasing cache memory cannot create a ninth trained speaker channel.

## Calibration-only rejection rule

The optional diagnostic score is the mean raw probability over VAD speech frames on the baseline assigned registered channel. Thresholds were fitted on calibration speakers only, separately per enrollment condition, targeting 5% false accepts; final-holdout unknown speakers were not used to select thresholds. The rule maximizes balanced known-correct and unknown-rejection rates under the calibration FAR target. It is not an identity posterior, and no EER is reported.

| Enrolled, budget | Threshold | Calibration known / unknown; false accepts | Final known correct / evaluated (rejected) | Final unknown rejected / evaluated; false accepts; no decision |
|---|---:|---|---|---|
| 2, 60 s | 0.59519 | 20 / 39; 0 | 17/19 (2) | 36/42; 5; 1 |
| 4, 60 s | 0.85222 | 39 / 41; 2 (4.88%) | 14/20 (6) | 20/22; 2; 0 |
| 8, 60 s | 0.87360 | 43 / 20; 1 (5.00%) | 48/79 (31) | 39/41; 1; 1 |

The calibrated eight-speaker rule lowers final-holdout false accepts from 40/41 to 1/41, but accepts only 48/79 known queries. This is a useful operating-point tradeoff for diagnosis, not sufficient evidence of reliable VoiceID. The two-speaker five-second calibration condition had insufficient scores to fit a threshold.

Three other calibration-only score candidates were compared at the same 60-second enrollment setting: mean top-1/top-2 margin, dominant-channel persistence, and usable VAD speech duration. Each threshold was fit on calibration speakers under the same 5% calibration FAR target, then frozen for final holdout. Counts below show the tradeoff; unknown no-decisions are listed separately from false accepts.

| Candidate score | 2 enrolled: known correct / evaluated; unknown rejected / evaluated (false accepts; no decision) | 4 enrolled: known correct / evaluated; unknown rejected / evaluated (false accepts; no decision) | 8 enrolled: known correct / evaluated; unknown rejected / evaluated (false accepts; no decision) |
|---|---|---|---|
| Registered-channel mean (baseline) | 17/19; 36/42 (5; 1) | 14/20; 20/22 (2; 0) | 48/79; 39/41 (1; 1) |
| Mean top-1/top-2 margin | 19/19; 37/42 (4; 1) | 19/20; 22/22 (0; 0) | 10/79; 40/41 (0; 1) |
| Dominant-channel persistence | 0/19; 41/42 (0; 1) | 0/20; 22/22 (0; 0) | 0/79; 40/41 (0; 1) |
| VAD speech duration | 19/19; 29/42 (12; 1) | 7/20; 22/22 (0; 0) | 4/79; 35/41 (5; 1) |

The margin candidate retained all evaluated known queries at two speakers and 19/20 at four, while it failed to preserve known acceptance at eight speakers. Persistence selected a threshold above the maximum possible score in all three conditions, rejecting every known query. Duration thresholds varied sharply by cohort size and did not separate known from unknown consistently. These candidates are diagnostics, not identity probabilities or a validated general rejection rule; their thresholds and calibration counts are included in the full aggregate JSON.

## Synthetic conversations and diarization

The generator builds deterministic turn-taking, 0.5–2-second short-turn, three overlap-level, and unknown-heavy-then-known-return mixtures from held-out utterances. It records source utterance IDs, external IDs, source VAD masks, gains, sample-aligned start/end positions, overlap, and seed. Overlap reference masks shift the original converted-source VAD masks with the schedule; tapered crop-edge frames are masked. This is more precise than treating scheduled placement as speech activity.

Of 216 planned conversations, 180 were evaluated: 30/36 turn-taking, 30/36 short-turn, 90/108 overlapping, and 30/36 unknown-heavy. The 36 skips came from invalid enrollment profiles or identity maps. DER below is an aggregate no-collar score over source VAD speech-speaker frames. Enrolled IDs remain fixed; unknown speakers get a session-local one-to-one channel mapping chosen to maximize overlap for diarization scoring. That unknown mapping is an oracle for the session and is not persistent VoiceID accuracy.

| Scenario | Complete / planned | Ref speaker frames | Miss | False alarm | Confusion | Aggregate DER |
|---|---:|---:|---:|---:|---:|---:|
| Turn-taking | 30/36 | 121,300 | 15,299 | 10,604 | 20,382 | 0.382 |
| Short turns | 30/36 | 30,109 | 4,523 | 1,366 | 5,859 | 0.390 |
| Overlap | 90/108 | 57,571 | 6,015 | 3,556 | 12,193 | 0.378 |
| Unknown-heavy then known return | 30/36 | 74,428 | 11,338 | 6,105 | 21,352 | 0.521 |

Overlap-target speech-mask intersection fractions had medians 22.8%, 48.4%, and 72.7% for scheduled targets 25%, 50%, and 75%. Across all completed conversations, speech detection produced 236,984 true-positive, 20,107 false-alarm, and 34,117 missed frames (271,101 reference speech frames). Known identity was scored as multi-label: TP/FP/FN were 56,604/24,968/3,835 for turn-taking, 15,151/6,138/228 for short turns, 26,149/14,575/2,107 for overlap, and 12,817/21,019/750 for unknown-heavy sessions. Unknown-presence frame counts across these conversations were TP 82,862, FP 7,581, and FN 82,905. These frame totals describe the synthetic mixtures and should not be interpreted as independent statistical trials.

## Enrollment order and temporary AOSC contamination

The same held-out queries were compared for randomized and reversed enrollment orders at 30 and 60 seconds. Across 72 paired profiles, 1,926 comparable queries changed category in 104 cases (5.4%) and changed exact predicted identity in 206 cases (10.7%). Of 305 shared identity-to-channel assignments, 303 changed (99.3%), confirming that channel numbers are order-local. Identical source utterances were compared on both profiles; the permanent enrollment profile was not edited.

The contamination study completed 121 fresh-profile sequences, each with four distinct unknown speakers followed by a known return, across 2/4/8-speaker groups, 30/60-second budgets, and both tested enrollment orders. The known return was `known_correct` in 121/121 sequences and matched its independent fresh-session decision in 121/121. The immutable source profile was preserved in 121/121.

Across 484 intermediate cache snapshots, the temporary cache differed from its enrollment state at 199 snapshots. Registered-channel positive-row totals ranged from 190 to 256 (median 256); unregistered channels held 0–66 positive rows (median 0). There were 183 negative and 7 positive per-identity registered-row deltas across the snapshots, so some old registered-channel rows were displaced while unknown rows accumulated. In these four-unknown sequences, this did not change the subsequent known identity decision. Longer, more heavily switched traffic remains a limit of this experiment; the unknown-heavy conversation condition is separately scored above. The permanent enrollment profile was never updated from test audio.

To measure the time course directly, a separate staged replay used `final_holdout_n{2,4,8}_r{00,01,02,03}_e60_randomized` with `--unknown-traffic-only` (12 final-holdout profiles, four cohorts each at 2/4/8 speakers). Ten profiles had valid maps; two four-speaker profiles were skipped for ambiguous mapping. All 10 evaluable profiles reproduced the full run’s identity map and profile CRC. The same known mic1 query was tested from a clean AOSC import and from fresh imports of the continuous stream’s saved AOSC state after each of 1, 2, 3, and 4 unknown speakers. Every stage was `known_correct` in 10/10 profiles; exact identity decisions changed in 0/10 at each stage. The uninterrupted stream’s known return was also correct in 10/10. Temporary cache state differed from enrollment in 0/10 profiles after one unknown, 1/10 after two, 5/10 after three, and 6/10 after four. This staged diagnostic forks saved AOSC snapshots for the fresh-start comparisons while the primary traffic stream itself remains continuous. The anonymized per-profile counts are in [the staged traffic aggregate](../tests/results/voiceid_task08_traffic_stages.json).

## CPU/Vulkan correctness sampling

CPU and Vulkan were compared on three final-holdout, 60-second profiles (2/4/8 enrolled) and 33 identical query trials spanning mic and duration conditions. Both backends saved and reloaded all three AOSC profiles; identity-to-channel maps matched for all 14 shared assignments. CPU/Vulkan profile CRCs matched in 0/3 pairs, reflecting numerical differences in independently constructed floating-point profile state.

Identity decisions changed in 3/33 queries (9.1%):

- **PV-01:** four enrolled, mic2, 2–5 s speech. CPU assigned an unregistered channel and rejected UNKNOWN; Vulkan assigned a registered identity (false accept).
- **PV-02:** eight enrolled, mic1, under 2 s speech. CPU had no active decision; Vulkan falsely accepted a registered identity.
- **PV-03:** eight enrolled, mic2, 2–5 s speech. CPU had no active decision; Vulkan falsely accepted a registered identity.

The three disagreements were not all near ties: two CPU no-decision cases became active Vulkan assignments. Across 15,713 compared frames, top-1 channels differed on 1,050 frames (6.68%) and the `>=0.5` activity mask differed on 786 frames (5.00%). Median per-query MAE was 0.00399, mean per-query RMSE was 0.02863, and maximum absolute probability difference was 0.5711. Bitwise equality was not required, but decision-changing disagreements are recorded as a material backend risk for those cases.

## Performance and tests

The full Vulkan run loaded the model in 534 ms and processed all 288 profiles in 3,221.55 seconds (53.7 minutes). It fed 115,259.44 seconds of enrollment, query, conversation, and traffic audio through the model: 35.78 audio seconds per process wall second (RTF 0.02795). Enrollment construction had a 6.645 s median and 26.894 s p95. The 7,554 evaluated isolated queries had 5.522 ms median and 6.078 ms p95 profile-import time; inference time was 152.74 ms median and 317.83 ms p95. The queries contained 31,834.53 seconds of audio and 1,271.44 seconds of query-inference time (25.04× audio/inference). Trial end-to-end time was 8.586 s median and 30.308 s p95. End-to-end throughput includes preprocessing, enrollment, stream/profile operations, and the four-way batcher.

The batcher observed batches up to 4 items (mean 4.0), with 100,989 batches and 403,956 requested items. Peak host RSS was 638,689,280 bytes (about 609 MiB); the maximum sampled RTX 4060 memory use was 1,977 MiB. These figures are for this NixOS host, model, and Vulkan0 configuration.

Both requested regression suites passed after rebuilding all targets:

- CPU: 6/6 CTests passed.
- Vulkan build with the NVIDIA ICD pinned: 6/6 CTests passed.

## Main findings and next architecture step

1. AOSC-only profile readiness depends on actual cache updates and compression; the shortest two-speaker enrollment frequently did not produce an exportable profile.
2. Enrollment often established speaker channels, but collision-free identity mapping was much less reliable at larger groups. More cache memory does not expand the trained capacity beyond eight channels.
3. Native matching gave high known-query accuracy on evaluated profiles, but UNKNOWN rejection degraded with group size and was structurally unavailable for active queries when all eight channels were registered.
4. A calibration threshold reduced unknown false accepts, especially at eight speakers, but rejected many known queries. The score remains a diarization-activity diagnostic rather than an identity posterior.
5. Enrollment order changes channel numbering almost every time and changes a measurable share of query identities/categories. CPU/Vulkan probability differences changed three identity decisions in the representative parity subset.

Keep AOSC as diarization context, not as a validated persistent identity score. A next single-model research step should add or train an identity-discriminative objective/output within the Nemotron model and evaluate it on independently collected multi-session and multi-device speaker data with speaker-disjoint calibration and final holdout. This task did not train or modify such a model. Do not present the current VCTK result as real-world biometric validation.
