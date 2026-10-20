# Task 09C — Frozen pre-encoder VoiceID robustness

Date: 2026-10-20
Branch: `research/voiceid-preencoder-robustness`
Base: `d74d85b3ce0b99c32811fb300b7902a64f364bb0`

## Verdict

**B — Useful but Not Sufficiently Robust.**

Frozen `pre_encoder` features carry speaker information without AOSC state or a second neural network. Training-speaker z-scoring improved the development score over the tested pooling alternatives, and the representation was stable under silence, unrelated-speaker, same-speaker prior-turn, and imported-AOSC contexts. It did not provide reliable microphone transfer, short-query decisions, or a consistent 90% known / 5% UNKNOWN operating point across folds.

The selected Task 09B learned head accepted more known mic1-enrollment/mic2-query clips than the best frozen method under the matched Task 09C protocol (40/96 versus 19/96). Neither method met the operating target in every fold. VCTK mic1/mic2 files are paired microphone renderings of the same utterance, and all 110 identities had appeared in previous project experiments. These results are follow-up evidence, not independent recording-session or identity validation.

The next justified experiment is a held-out, multi-session, multi-device evaluation with independent enrollment and query sessions and a large unknown cohort. Do not start production integration or another learned head from these VCTK results alone.

## Protocol and integrity

- Reused the Task 09B speaker-disjoint folds: 50 training/statistics speakers, 15 model-selection speakers, 15 threshold-calibration speakers, and 30 evaluation speakers per fold. No neural head was trained in Task 09C.
- Kept the Task 09B enrollment/query utterance assignments. Paired mic1/mic2 versions share an utterance identity and remain clustered. Matched-microphone evaluation uses up to four paired query utterances per evaluation identity; there are 32 known and 88 unknown query clips per fold in the 8-person registry matrix.
- Selected one frozen method using only the 15 model-selection speakers per fold, then used calibration speakers for thresholds. The locked evaluation uses the selected method without evaluation-driven changes.
- The 5% calibration target uses leave-one-identity-out maximum impostor scores. Each calibration fold has only 15 identities, hence at most 14 impostor competitors. The 16-person evaluation uses the fixed threshold and does not establish a 16-person-specific calibration.
- Confidence intervals in the result files use speaker-cluster bootstrap resampling. Repeated clips and paired microphone versions are not treated as independent speakers. Speakers can recur across folds, so pooled counts are descriptive rather than independent-trial confidence bounds.
- 88 requested mic2 source files were absent from the local VCTK tree. The primary evaluation query pairs remained complete; one fold had 14/15 mic2 calibration enrollment profiles, while the other two had 15/15. No evaluation speaker was silently dropped.
- No data was downloaded. No model weights, AOSC compression, eight-channel diarization head, public C ABI, or production inference path was changed.

## Task 09B baseline reproduction

The original Task 09B manifests, cached Vulkan features, calibration thresholds, and saved selected heads reproduced the expected primary counts exactly at 10-second enrollment, 3-second mic1 queries, and an 8-person registry.

| Method | Fold 1 known / unknown FA | Fold 2 known / unknown FA | Fold 3 known / unknown FA | Pooled |
|---|---:|---:|---:|---:|
| Pre-encoder mean cosine | 35/41, 5/122 | 40/44, 8/122 | 37/41, 4/121 | 112/126, 17/365 |
| Pre-encoder mean/std cosine | 34/41, 6/122 | 41/44, 4/122 | 39/41, 8/121 | 114/126, 18/365 |
| Task 09B selected learned head | 39/41, 2/122 | 39/44, 4/122 | 25/41, 7/121 | 103/126, 13/365 |

These are exact regression matches. The mean/std pooled result is 90.5% known acceptance and 4.93% UNKNOWN false acceptance, but fold 1 accepts only 34/41 known queries and fold 3 has 8/121 UNKNOWN false accepts. A pooled pass does not mean the per-fold target passed.

## Microphone transfer

The frozen methods and saved selected head were evaluated on the same four-condition mic matrix. Thresholds below were fitted separately by fold and enrollment/query microphone using only that fold's calibration speakers.

| Enrollment → query | Frozen z-scored pre-encoder mean | Selected Task 09B head |
|---|---:|---:|
| mic1 → mic1 | 69/96 known; 12/264 UNKNOWN FA | 55/96; 8/264 |
| mic1 → mic2 | 19/96; 12/264 | 40/96; 15/264 |
| mic2 → mic1 | 32/96; 22/264 | 39/96; 8/264 |
| mic2 → mic2 | 60/96; 14/264 | 56/96; 12/264 |

For the primary mic1-enrollment/mic2-query transfer, the frozen method's per-fold decisions were:

| Fold | Correct known | Wrong registered ID | Rejected known | Rank-1 correct | UNKNOWN false accepts |
|---|---:|---:|---:|---:|---:|
| 1 | 5/32 | 1/32 | 26/32 | 26/32 | 0/88 |
| 2 | 8/32 | 2/32 | 22/32 | 25/32 | 2/88 |
| 3 | 6/32 | 1/32 | 25/32 | 25/32 | 10/88 |

The matched selected-head counts were 18/32, 15/32, and 7/32 known correct, with 2/88, 3/88, and 10/88 UNKNOWN false accepts. The head was better on known transfer in every fold, but fold 3 also exceeded the 5% UNKNOWN limit.

Under the original Task 09B mic1 threshold on the mic2 query clips, mean/std cosine accepted 4/96 known clips with 0/264 UNKNOWN false accepts; the selected head accepted 27/96 with 5/264 UNKNOWN false accepts. The frozen training-only z-score candidate accepted 14/96 with 6/264. These fixed-threshold transfer rows use single paired clips and are reported separately from the original grouped 3-second Task 09B reproduction.

### Ranking and threshold errors

For z-scored pre-encoder means, mic1/mic1 rank-1 identification was 92/96, compared with 76/96 for mic1/mic2. Yet only 19/96 mic1/mic2 known clips were accepted correctly: 57 clips had the correct identity ranked first but fell below the threshold; four more were accepted as the wrong registered identity; 73 were rejected as UNKNOWN. The transfer loss therefore has both ranking errors and a larger score/threshold rejection component.

The mean genuine-score shift from mic1 queries to the paired mic2 rendering was -0.109, -0.214, and -0.188 cosine units in folds 1–3. The corresponding maximum-impostor-score shifts were +0.023, -0.017, and -0.033. The calibrated mic1/mic2 thresholds were close to their mic1/mic1 values (fold deltas +0.026, -0.006, and -0.003). This points to genuine-score loss, not a large threshold movement, as the main driver of rejection. Pairwise verification EERs for the frozen mic1/mic2 condition were 0.161, 0.156, and 0.181 by fold; these are descriptive and do not replace the calibrated open-set counts.

Speaker-cluster 95% intervals for frozen mic1/mic2 known acceptance were wide: 0.031–0.344, 0–0.531, and 0–0.406 by fold. The UNKNOWN false-accept intervals were 0–0, 0–0.057, and 0.034–0.216. The full per-speaker counts and genuine/impostor score quantiles are in `tests/results/voiceid_task09c_microphone.json`.

### What changes between microphones

Across 336 unique matched speaker/utterance pairs, the mean aligned speech-frame cosine was 0.949 and the mean pooled-feature cosine was 0.950. Temporal pooling did not materially amplify the average angular difference. Per-dimension mean shifts and variance ratios are retained in the microphone result.

The speech masks agreed on 87.7% of frames on average, and mic2 had 0.226 seconds less VAD-active speech per paired clip on average. Mic2 input RMS was 1.88 dB lower on average (median -1.35 dB). The average changes in spectral energy fraction for 0–300, 300–1,000, 1,000–3,000, 3,000–6,000, and 6,000–8,000 Hz bands were -0.065, +0.111, -0.033, -0.011, and -0.002. These are input and frontend differences before identity decisions; they do not by themselves imply an identity error.

The evidence points to changed frontend features and VAD-active regions, followed by lower genuine scores. The score-pooling comparison does not show a large average cosine-angle amplification. The full paired per-dimension feature shifts, variance changes, RMS/peak/quiet-sample statistics, and spectral summaries are in the microphone JSON.

## Pooling, normalization, and enrollment

The bounded development comparison selected a 512D pre-encoder mean with per-dimension center/scale fitted on training speakers, clipping at ±8 before L2 normalization and cosine matching. It was the highest-scoring method in each development fold and globally: balanced speaker-macro score 0.766, compared with 0.708 for a mean of per-utterance-normalized prototypes, 0.703 for mean/std, 0.703 for regularized whitening, and 0.706 for plain mean. The development fold scores for the selected z-score method were 0.738, 0.844, and 0.717. It still failed badly on the locked mic1/mic2 evaluation, especially fold 3.

The mean of per-utterance-normalized enrollment embeddings did not improve on one pooled prototype. No evidence justified a multi-prototype maximum-score matcher, so that candidate was not added.

| Requested enrollment speech | Actual mean usable speech | Fold 1 correct / UNKNOWN FA | Fold 2 correct / UNKNOWN FA | Fold 3 correct / UNKNOWN FA | Profile failures |
|---|---:|---:|---:|---:|---:|
| 5 s | 6.14 s | 22/41, 0/122 | 31/44, 3/122 | 37/41, 10/121 | 0/24 |
| 10 s | 11.39 s | 30/41, 4/122 | 36/44, 5/122 | 39/41, 11/121 | 0/24 |
| 15 s | 16.26 s | 33/41, 3/122 | 41/44, 6/122 | 39/41, 15/121 | 0/24 |
| 30 s | 31.81 s | 36/41, 5/122 | 41/44, 7/122 | 41/41, 22/121 | 0/24 |
| 60 s | 61.31 s among attempts | 21/41, 0/122 | 34/44, 11/122 | 27/41, 15/121 | 9/24 |

Longer enrollment generally raised known acceptance through 30 seconds but also raised UNKNOWN false accepts and did not improve every fold at every step. At 60 seconds, the available VCTK enrollment speech was insufficient for 9 of 24 attempted registered profiles. Those failed profile attempts remain visible in the denominator and failure list in `tests/results/voiceid_task09c_enrollment.json`.

At 5 seconds, the median same-speaker prototype cosine to that speaker's 10-second profile was 0.879, 0.900, and 0.920 across the folds. Ten-second profiles succeeded for all 24 evaluation registry attempts. Profile generation reads cached pre-encoder features and does not require AOSC profile readiness or a native Sortformer channel.

The 09B 8-person/3-second fixed-threshold results above are the recognition comparison for enrollment budgets. The new matched-microphone results use 10-second enrollment. Requested duration can overshoot by the final enrollment utterance; actual usable speech and clip counts are recorded per identity in the enrollment result.

## Short queries and evidence accumulation

For one independent query utterance at a time, the frozen z-score method abstained when the requested actual VAD-speech budget was unavailable. Across the three folds:

| Per-utterance speech budget | Known correct / all known attempts | Known with enough speech | UNKNOWN FA / all unknown attempts | Unknown with enough speech |
|---|---:|---:|---:|---:|
| 1 s | 96/288 | 265/288 | 0/792 | 741/792 |
| 2 s | 72/288 | 114/288 | 2/792 | 339/792 |
| 3 s | 22/288 | 29/288 | 4/792 | 122/792 |
| 5 s | 2/288 | 3/288 | 2/792 | 26/792 |
| 7 s | 0/288 | 1/288 | 0/792 | 4/792 |
| 10–15 s | 0/288 | 0/288 | 0/792 | 0/792 |

The 10–15 second entries are all insufficient-evidence abstentions for an individual VCTK query clip; the clips are not that long. At 1 second this method correctly accepted 96/288 attempts and falsely accepted no UNKNOWN clips, while 23 known and 51 unknown attempts had insufficient evidence. The Task 09B selected head's separate 1-second protocol accepted 68/174 known queries and had 3/453 UNKNOWN false accepts; its query grouping and denominators differ.

An oracle upper bound concatenated successive query utterances using their ground-truth speaker identity. It reached 18/24 known correct at 3 seconds and 24/24 at 15 seconds, while UNKNOWN false accepts rose to 14/66 at 15 seconds. This is not a deployable grouping rule. VCTK has no natural multi-speaker conversation sessions to evaluate a turn grouper without identity labels; the independent-utterance rows are the deployable evidence condition, and do not establish cross-turn aggregation reliability.

## Registry size and UNKNOWN rejection

The selected Task 09C frozen z-score method used one fixed fold threshold for query durations and registry sizes. The 3-second results were:

The full outcome breakdown at the 8-person operating point was:

| Fold | Calibration UNKNOWN FA | Known correct / wrong ID / rejected as UNKNOWN | Evaluation UNKNOWN accepted / rejected | No-decision | Pairwise FAR / FRR / EER |
|---|---:|---:|---:|---:|---:|
| 1 | 4/86 (4.65%) | 30/41 / 0 / 11 | 4/122 / 118/122 | 0 | 0.32% / 26.8% / 4.89% |
| 2 | 4/80 (5.00%) | 36/44 / 0 / 8 | 5/122 / 117/122 | 0 | 0.39% / 18.2% / 2.27% |
| 3 | 4/80 (5.00%) | 39/41 / 0 / 2 | 11/121 / 110/121 | 0 | 1.20% / 4.88% / 2.93% |

Pairwise verification rates use the genuine/impostor trial population and are descriptive; they are not the open-set decision rates in the adjacent columns. The calibration cohort reaches the nominal 5% UNKNOWN target, but evaluation fold 3 exceeds it and the calibration cohort is small.

| Registry | Fold 1 known; UNKNOWN FA | Fold 2 known; UNKNOWN FA | Fold 3 known; UNKNOWN FA |
|---:|---:|---:|---:|
| 1 | 4/4; 2/159 | 6/6; 0/160 | 5/5; 3/157 |
| 2 | 9/11; 2/152 | 10/10; 0/156 | 10/10; 2/152 |
| 4 | 16/21; 3/142 | 21/22; 0/144 | 20/22; 4/140 |
| 8 | 30/41; 4/122 | 36/44; 5/122 | 39/41; 11/121 |
| 16 | 59/82; 4/81 | 77/91; 11/75 | 82/85; 8/77 |

Maximum impostor scores and UNKNOWN false accepts rise as the registry grows, while known rejection remains substantial. At registry size 8, fold 3 accepts 39/41 known queries but falsely accepts 11/121 unknowns; no fold satisfies both declared targets. At size 16, the fixed threshold is outside the calibration cohort's direct coverage and UNKNOWN false acceptance is 11/75 and 8/77 in folds 2 and 3. The 16-person rows are a stress test, not evidence of calibrated 16-person reliability. Full results also include 1-, 7-, and 15-second query budgets and score distributions.

## Audio normalization

The single tested audio policy applied the same per-utterance RMS target (-26 dBFS), ±6 dB gain cap, and -1 dBFS peak cap to all identities and both microphones, then recomputed the existing deterministic VAD. It used no speaker labels or evaluation decisions. Across 9,148 jobs, median input RMS was -26.27 dBFS, median gain +0.27 dB, and median output RMS -26.00 dBFS. The gain cap left some quiet clips below target.

With the same z-score method and condition-specific calibration, mic1→mic2 went from 19/96 known and 12/264 UNKNOWN false accepts on raw PCM to 15/96 and 7/264 after RMS normalization. The decrease in false accepts came with fewer correct known accepts. Mic2→mic1 changed from 32/96 and 22/264 to 26/96 and 31/264; mic2→mic2 changed from 60/96 and 14/264 to 59/96 and 16/264. This normalization did not materially improve cross-microphone robustness and did not address spectral or VAD differences.

## Context and AOSC interaction

The strongest frozen method was checked on 60 target utterances per context mode: fresh stream, 2.56 seconds of silence, two unrelated training-speaker prefixes, imported AOSC context, and an oracle same-speaker prior turn. Each fold used a fixed Task 09B threshold and the same eight external identity prototypes.

- Pre-encoder target features after imported AOSC context were bit-identical to fresh features for 60/60 targets.
- For same-speaker prior-turn context, target-relative frame axes aligned in every comparison; 7/60 feature records were bit-identical, and mean pooled cosine to fresh was 0.9999996. Other audio-prefix contexts were also near 1.0 cosine.
- No context changed the top identity or thresholded acceptance for any target. The small feature changes under audio prefixes are consistent with streaming frontend state or window-boundary effects, not AOSC conditioning of the pre-encoder tap.
- The context subset's frozen matcher accepted 18/30 known targets and had 0/30 UNKNOWN false accepts under every mode. Decision stability is not evidence of high recognition accuracy.

The same-speaker prior turn uses the sibling query clip from the same labeled evaluation speaker, followed by 250 ms silence. It is an oracle context stress test, not a deployable speaker-linking rule. The frozen enrollment profile itself requires no AOSC cache or channel warm-up.

## Native feasibility

- The existing Task 06 research observer exposes the 512D `pre_encoder` representation and VAD mask. The extraction probe ran on explicit Vulkan0 with model fingerprint `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`.
- The probe compared runs with research taps enabled and disabled on the first three jobs of each extraction batch. It compared exact diarization probability vectors, frame counts, speaker counts, and segment counts; all recorded probe batches had 3/3 checks and zero mismatches.
- A 12-job cached-WAV feature probe processed 55.32 seconds of audio in 2.52 seconds on Vulkan0 (21.98 audio seconds per wall second; 210 ms/job). This is a small extractor benchmark, not production end-to-end latency.
- A standalone optimized C++ matcher benchmark includes training-stat standardization, clipping, L2 normalization, and dot products against synthetic 512D prototypes. On the Ryzen 5 1600 it measured 4.01 µs/query for 8 identities and 7.45 µs/query for 16 identities. This excludes neural feature extraction and uses synthetic vectors.
- A 512D float32 prototype is 2,048 raw bytes per identity; a 16-person registry is 32 KiB of raw prototypes. The serialization replay stored 16 arbitrary IDs in 16 separate registration processes and reloaded the registry in query processes. The registered known query returned `tenant_001`; the test unknown was rejected. The registry uses no native channel index. Its threshold was calibrated against 15 speakers and is not a calibrated 16-person threshold.
- The replay stores the fold's training-only z-score center and scale in the registry and reloads them in the query process. Base64 prototype payloads decode to 2,048 bytes each; JSON metadata and the shared transform account for additional registry bytes.

The registry replay consumes cached feature arrays in Python. It verifies persistence and matching semantics; it is not a native production VoiceID implementation. No C ABI or production pipeline was changed.

## Answers to the Task 09C questions

1. **Does frozen mean/std outperform the selected trained head on microphone transfer?** No. Under the original fixed threshold it accepted 4/96 known mic2 clips versus 27/96 for the trained head. Under matched condition-specific calibration, the development-selected frozen z-score method accepted 19/96 mic1→mic2 clips versus 40/96 for the head.
2. **Ranking errors or threshold rejection?** Both occur, but correct-rank threshold rejection dominates: 76/96 known queries rank correctly under mic1→mic2, while only 19/96 are accepted correctly; 57 correctly ranked clips are rejected, four wrong identities are accepted, and 16 fewer queries rank correctly than under mic1→mic1.
3. **Does consistent audio normalization improve transfer?** No. It reduced mic1→mic2 UNKNOWN false accepts from 12/264 to 7/264 but also reduced correct known accepts from 19/96 to 15/96; reverse transfer became worse.
4. **Which pooling method is most stable across folds?** Training-speaker diagonal z-score of the pooled pre-encoder mean was selected in all three development folds and had the top global development score. It did not remain reliable on the locked evaluation folds.
5. **Does more enrollment consistently improve?** No. Known acceptance generally rose through 30 seconds, UNKNOWN false accepts also rose, and 60 seconds exhausted available speech for 9/24 profiles.
6. **Do multiple independent utterances form a better prototype?** The tested mean of per-utterance-normalized means scored below one pooled prototype in development. No multi-prototype max matcher was justified.
7. **How much speech is useful for a short query?** One second can produce a provisional result but only 96/288 independent known attempts were accepted correctly. At 2 seconds only 114/288 known clips had sufficient speech; longer single-clip budgets were rarely available. Oracle multi-turn accumulation is not deployable evidence.
8. **How does UNKNOWN FA change with registry size?** It rises under a fixed threshold; size-16 evaluation is additionally beyond this cohort's direct calibration support.
9. **Does the method generalize across folds?** No. Fold 3 has 10/88 mic1→mic2 UNKNOWN false accepts; fold 1 and fold 2 have only 5/32 and 8/32 correct known accepts in that condition.
10. **Can profiles be created without AOSC readiness?** Yes. The frozen feature profile is constructed and serialized from pre-encoder features with no AOSC cache or native channel.
11. **What is the cost per identity?** 2,048 raw bytes for 512D float32; one 512-element dot product per identity per query. The measured synthetic C++ matcher took 4.01 µs for 8 identities and 7.45 µs for 16.
12. **Is additional learned identity modeling justified?** The selected head did better on known mic transfer but did not satisfy open-set targets across folds. These previously studied VCTK identities do not justify another head before independent session/device data.
13. **What evidence is still missing?** Untouched identities, repeated recording sessions, multiple physical devices per speaker, natural conversations with turn labels, and an independent calibration/holdout cohort sized for the intended registry.

## Reproduction commands

Commands below assume the repository branch, model, VCTK corpus, and Task 09B caches used for this run. Generated WAVs, feature records, temporary registries, and large extraction traces remain under `/home/glebus/.cache`.

### Reproduce the Task 09B baseline

```bash
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/evaluate_voiceid_head.py --manifest tests/manifests/task09b_vctk_splits.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --run-dir /home/glebus/.cache/nemo-speech-task09b/run --training-results tests/results/voiceid_task09b_training.json --output /home/glebus/.cache/nemo-speech-task09b/task09c_reproduction_verification.json --open-set-output /home/glebus/.cache/nemo-speech-task09b/task09c_reproduction_open_set.json'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/evaluate_task09c_baselines.py --verification /home/glebus/.cache/nemo-speech-task09b/task09c_reproduction_verification.json --output tests/results/voiceid_task09c_baselines.json'
```

### Prepare and extract mic2 pairs, then evaluate transfer and select pooling

```bash
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/prepare_task09c_microphones.py --manifest tests/manifests/task09b_vctk_splits.json --dataset-root VCTK-Corpus-0.92 --cache-dir /home/glebus/.cache/nemo-speech-task09c/vctk --jobs-tsv /home/glebus/.cache/nemo-speech-task09c/mic2_all_additions.tsv --output tests/manifests/task09c_vctk_microphone_additions.json'
nix-shell -p python3 python3Packages.numpy --run 'python3 tests/tools/extract_voiceid_features.py --jobs-tsv /home/glebus/.cache/nemo-speech-task09c/mic2_all_additions.tsv --probe build/vulkan-diar/bin/diar_voiceid_feature_probe --model models/Nemotron-3-Diarization.q8_0.gguf --feature-dir /home/glebus/.cache/nemo-speech-task09c/features/mic2_additions --device vulkan --gpu-device 0 --representations pre_encoder,encoder.layer.7,encoder_output,encoder_proj --parity-first 3'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/analyze_task09c_microphones.py --manifest tests/manifests/task09b_vctk_splits.json --microphone-additions tests/manifests/task09c_vctk_microphone_additions.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --additional-pooled-features /home/glebus/.cache/nemo-speech-task09c/features/mic2_additions/pooled_features.npz --feature-dir /home/glebus/.cache/nemo-speech-task09b/features/fresh --additional-feature-dir /home/glebus/.cache/nemo-speech-task09c/features/mic2_additions --base-jobs-tsv /home/glebus/.cache/nemo-speech-task09b/task09b_extraction_jobs.tsv --run-dir /home/glebus/.cache/nemo-speech-task09b/run --reproduction /home/glebus/.cache/nemo-speech-task09b/task09c_reproduction_verification.json --output tests/results/voiceid_task09c_microphone.json'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/evaluate_task09c_pooling.py --manifest tests/manifests/task09b_vctk_splits.json --microphone-additions tests/manifests/task09c_vctk_microphone_additions.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --additional-pooled-features /home/glebus/.cache/nemo-speech-task09c/features/mic2_additions/pooled_features.npz --feature-dir /home/glebus/.cache/nemo-speech-task09b/features/fresh --additional-feature-dir /home/glebus/.cache/nemo-speech-task09c/features/mic2_additions --output tests/results/voiceid_task09c_pooling.json'
```

### Normalize audio and evaluate short queries, registry sizes, and enrollment

```bash
nix-shell -p python3 python3Packages.numpy --run 'python3 tests/tools/normalize_task09c_audio.py --base-jobs-tsv /home/glebus/.cache/nemo-speech-task09b/task09b_extraction_jobs.tsv --additional-jobs-tsv /home/glebus/.cache/nemo-speech-task09c/mic2_all_additions.tsv --audio-dir /home/glebus/.cache/nemo-speech-task09c/audio/rmsnorm --jobs-tsv /home/glebus/.cache/nemo-speech-task09c/rmsnorm_jobs.tsv --summary /home/glebus/.cache/nemo-speech-task09c/rmsnorm_summary.json --workers 6'
nix-shell -p python3 python3Packages.numpy --run 'python3 tests/tools/extract_voiceid_features.py --jobs-tsv /home/glebus/.cache/nemo-speech-task09c/rmsnorm_jobs.tsv --probe build/vulkan-diar/bin/diar_voiceid_feature_probe --model models/Nemotron-3-Diarization.q8_0.gguf --feature-dir /home/glebus/.cache/nemo-speech-task09c/features/rmsnorm --device vulkan --gpu-device 0 --representations pre_encoder,encoder.layer.7 --parity-first 3'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/evaluate_task09c_audio_normalization.py --normalized-pooled-features /home/glebus/.cache/nemo-speech-task09c/features/rmsnorm/pooled_features.npz --output tests/results/voiceid_task09c_normalization.json'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/evaluate_task09c_accumulation.py --manifest tests/manifests/task09b_vctk_splits.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --feature-dir /home/glebus/.cache/nemo-speech-task09b/features/fresh --output tests/results/voiceid_task09c_duration.json'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/summarize_task09c_open_set.py --verification /home/glebus/.cache/nemo-speech-task09b/task09c_reproduction_verification.json --output tests/results/voiceid_task09c_open_set.json'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/summarize_task09c_enrollment.py --manifest tests/manifests/task09b_vctk_splits.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --verification /home/glebus/.cache/nemo-speech-task09b/task09c_reproduction_verification.json --output tests/results/voiceid_task09c_enrollment.json'
```

### Context probes, persistence, matcher benchmark, and regression suites

```bash
nix-shell -p python3 python3Packages.numpy --run 'python3 tests/tools/prepare_task09c_context.py --base-jobs-tsv /home/glebus/.cache/nemo-speech-task09b/task09b_extraction_jobs.tsv --jobs-tsv /home/glebus/.cache/nemo-speech-task09c/context_jobs.tsv --output tests/manifests/task09c_vctk_context.json'
nix-shell -p python3 python3Packages.numpy --run 'python3 tests/tools/extract_voiceid_features.py --jobs-tsv /home/glebus/.cache/nemo-speech-task09c/context_jobs.tsv --probe build/vulkan-diar/bin/diar_voiceid_feature_probe --model models/Nemotron-3-Diarization.q8_0.gguf --feature-dir /home/glebus/.cache/nemo-speech-task09c/features/context_same_speaker --device vulkan --gpu-device 0 --representations pre_encoder --parity-first 3'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/analyze_task09c_context.py --manifest tests/manifests/task09b_vctk_splits.json --task09c-context-manifest tests/manifests/task09c_vctk_context.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --fresh-feature-dir /home/glebus/.cache/nemo-speech-task09b/features/fresh --task09b-context-feature-dir /home/glebus/.cache/nemo-speech-task09b/features/context --task09c-context-feature-dir /home/glebus/.cache/nemo-speech-task09c/features/context_same_speaker --output tests/results/voiceid_task09c_context.json'
c++ -O3 -DNDEBUG -march=native -std=c++17 tests/tools/benchmark_task09c_matching.cpp -o /tmp/benchmark_task09c_matching
/tmp/benchmark_task09c_matching
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/replay_task09c_frozen_registry.py --manifest tests/manifests/task09b_vctk_splits.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --registry /home/glebus/.cache/nemo-speech-task09c/frozen_registry_fold1.json --matching-benchmark /home/glebus/.cache/nemo-speech-task09c/matching_benchmark.json --feature-extraction-benchmark /home/glebus/.cache/nemo-speech-task09c/feature_extraction_benchmark.json --output tests/results/voiceid_task09c_native_feasibility.json --fold-id fold_1 --identity-count 16'
ctest --test-dir build/cpu-diar --output-on-failure -j6
ctest --test-dir build/vulkan-diar --output-on-failure -j4
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 -m unittest discover -s tests/tools -p test_task09c_voiceid.py -v'
```

## Regression results and scope

- CPU CTest: 7/7 passed.
- Vulkan CTest: 7/7 passed.
- Task 09C deterministic Python tests: 4/4 passed.
- Native extractor parity checks: 3/3 checked jobs, zero mismatches on the mic2, RMS-normalized, same-speaker-context, and 12-job benchmark batches. The probe comparison checks exact diarization probability vectors and frame/speaker/segment counts with research taps enabled versus disabled.
- Existing Task 09B cross-process registry replay: 16 arbitrary identities registered; known query returned `external_user_01`; unknown query returned `UNKNOWN`; remove and add-after-removal succeeded.
- Frozen pre-encoder cross-process registry replay: 16 arbitrary IDs registered; known query returned `tenant_001`; unknown query returned `UNKNOWN`; training-only transform was reloaded from registry serialization.

No production VoiceID implementation, new neural head, model fine-tune, derived GGUF, or additional dataset was produced. Stop after Task 09C.
