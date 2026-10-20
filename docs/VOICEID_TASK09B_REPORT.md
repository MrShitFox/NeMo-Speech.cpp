# Task 09B — Trainable Single-Model VoiceID Representation

**Verdict: B — Promising but Not Yet Reliable.** Frozen Nemotron features support learned speaker separation on some held-out folds, but the compact head did not improve reliably over controlled frozen-feature baselines. At the declared 8-person operating condition, the selected head accepted 103/126 known queries (81.7%) and falsely accepted 13/365 unknown queries (3.56%) across folds. Fold 3 accepted only 25/41 known queries and exceeded the 5% unknown false-accept target. A small, training-only head therefore does not yet meet the requirement for robust arbitrary-user VoiceID.

The experiment did establish that the permanent registry can use external IDs and grow to 16 users without using native channel indices. A Python research replay registered, queried, removed, and added identities across separate processes. The result is not a native VoiceID engine: the learned head was not ported to GGML or packaged into a derived GGUF. The quality gate for that work was not met.

## 1. Architecture and scope

The evaluated path was:

```text
VCTK audio -> frozen Nemotron GGUF -> selected frozen feature frames
           -> masked mean/std pooling -> small learned head -> unit 128D vector
           -> one cosine score per external-user prototype -> identity or UNKNOWN
```

The original GGUF weights, acoustic frontend, 31-layer Transformer, eight-channel diarization head, AOSC code, snapshots, and public C ABI were not changed. The identity output is independent of native channel indices. The 50-way classifier used by the cross-entropy objective was training-only and was not part of the exported embedding head or registry.

Two head shapes were tested on concatenated feature mean and standard deviation:

| Objective | Embedding path | Trainable parameters | Inference parameters | Export size |
|---|---|---:|---:|---:|
| Supervised contrastive | Linear 1024→128, then L2 normalization | 131,200 | 131,200 | about 0.54 MB |
| Speaker classification | LayerNorm 1024→Linear 256→GELU→Linear 128, then L2 normalization; training-only 128→50 classifier | 303,794 | 297,344 | about 1.23 MB |

All three folds selected `encoder.layer.7`. Fold 1 selected the linear contrastive head; folds 2 and 3 selected the classification MLP. Output dimension was fixed at 128. The other tested source representations were `pre_encoder` (512D), `encoder_output` (512D), and `encoder_proj` (192D). `encoder.layer.0`, `.15`, `.23`, and `.30` were not part of this bounded experiment.

## 2. Feature extraction and parity

The native research probe reuses the existing feature taps. It loads the model once per job list, runs 160ms stream pushes, records committed 80ms frames, and writes versioned `NVFTR001` float32 little-endian records. Every record includes source/model hashes, feature shape and timing, backend/device, context initialization, and an 8-channel 10ms VAD bitset per feature frame. A feature frame is included in pooling when at least four of its eight 10ms mask bits are active.

Bulk extraction used explicitly selected Vulkan device `Vulkan0` and model fingerprint `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`:

| Set | Jobs | Records | Backend | Tap-on vs tap-off probability check |
|---|---:|---:|---|---:|
| Fresh isolated streams | 6,128 | 24,512 | Vulkan0 | 3/3 exact; 0 mismatches |
| Context stress | 240 | 960 | Vulkan0 | 3/3 exact; 0 mismatches |

All record dimensions, frame IDs, byte order, timestamps, finite values, source hashes, masks, and context/profile hashes validated. No job had an empty speech mask. Bulk features and head checkpoints stayed outside Git under `/home/glebus/.cache/nemo-speech-task09b/`.

The compact extraction record is [voiceid_task09b_feature_extraction.json](../tests/results/voiceid_task09b_feature_extraction.json). CPU and Vulkan probe targets both build. The research extraction run itself used Vulkan; a trained-head CPU/Vulkan comparison is not applicable because no native head was implemented.

## 3. Streaming-context controls

Each isolated utterance started from a fresh empty stream. Stress targets were then replayed after 2.56 seconds of silence, after either of two unrelated training-role speakers, and after importing a prebuilt AOSC profile. Profiles used 20 mic1 training utterances from each of eight training speakers, with 250ms pauses; no evaluation identity contributed context or profile audio. Each profile had eight native cache channels. AOSC was not required to produce an identity prototype.

The pre-encoder tap was effectively context invariant: AOSC-import features were bit-identical to fresh features for all 20 targets in each fold. Mean pooled cosine was approximately 1.0 for every tested context; silence/unrelated-stream runs could differ by at most about 0.6 in a frame value because the streaming frontend had advanced.

The selected layer-7 tap changed more. Its fresh-to-context mean pooled cosine over 20 targets per fold was:

| Context | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|
| 2.56s silence | .9987 | .9986 | .9985 |
| Unrelated speaker A | .9969 | .9951 | .9945 |
| Unrelated speaker B | .9974 | .9935 | .9953 |
| Imported AOSC profile | .9912 | .9904 | .9882 |

High feature cosine did not guarantee an unchanged identity decision. Across ten known and ten unknown targets per fold, the selected head’s correct known acceptances were 12/30 after silence, 11/30 after speaker A, 13/30 after speaker B, and 10/30 after AOSC import; unknown false accepts were 0/30 in each small context condition. AOSC context changed the top identity for 11/60 target comparisons, and the mean identity-embedding cosine to fresh was .831, .793, and .808 by fold. These are repeated-target stress observations, not independent accuracy estimates. Full distributions and query-level decision counts are in [voiceid_task09b_context.json](../tests/results/voiceid_task09b_context.json).

`encoder_output` and `encoder_proj` were substantially more context-sensitive than the pre-encoder tap. For example, fold 1 mean pooled cosine to fresh for `encoder_output` was .348 after AOSC import and .462/.470 after the two unrelated speakers. That result supports selecting the pre-encoder or an explicitly context-tested layer for future work; it does not support silently mixing fresh and AOSC-conditioned features.

## 4. Speaker splits and leakage checks

The deterministic manifest uses 110 usable mic1 speakers and three seeds: 20261021, 20261022, and 20261023. Within each fold, roles are disjoint:

| Role | Speakers per fold |
|---|---:|
| Head training | 50 |
| Model selection | 15 |
| Threshold calibration | 15 |
| Locked evaluation | 30 |

Each speaker has 20 head-training, 32 registration, and 12 independent-query mic1 utterance groups. Mic1/mic2 renderings of an utterance stay in the same role; mic2 appears only in the paired query condition. The manifest verifies disjoint role speakers, disjoint training/enrollment/query utterance groups, unique job IDs and source hashes, and mic2-to-mic1 query pairing. Validation passed. The manifest SHA-256 is `236c9c76059ddec1a0dc617ce5418b9d14380fdb474659026716a7058a9e0b90`.

Evaluation speakers were excluded from the head, checkpoint selection, and threshold fitting in their fold. **This is not an untouched VCTK population:** identities recur across folds and all 110 were examined in earlier project tasks. Metrics are reported per fold; any pooled attempt counts across folds reuse some speaker identities and are not independent-trial confidence estimates. The manifest is [task09b_vctk_splits.json](../tests/manifests/task09b_vctk_splits.json).

## 5. Training setup and learning behavior

The cached frozen features were the only audio-model inputs. Per-dimension centering/scaling was fitted from training speakers only; the scale floor was `1e-4`, and normalized values were clipped at ±8. Contrastive batches sampled 12 speakers × four distinct mic1 utterances. The classification model’s speaker classifier was removed from inference. Training used deterministic seeds, AdamW, checkpoint/resume support, early stopping, and saved per-epoch losses and validation metrics.

The Nix environment provided Python 3.13.15, NumPy 2.4.4, and PyTorch 2.11.0, but this PyTorch build had no CUDA. Training therefore ran on CPU rather than the RTX 4060. It evaluated 24 supervised models (four representations × two objectives × three folds) and three per-utterance randomized-label controls. Total recorded model-training time was 41.9 seconds; process high-water memory was about 529 MB. GPU memory use was unavailable. The 27 training records and learning histories are in [voiceid_task09b_training.json](../tests/results/voiceid_task09b_training.json).

The selected models stopped at epochs 7, 3, and 1 and each reached 0.0 pairwise EER on its 15-speaker model-selection set. Their locked-evaluation EERs at eight identities were 2.13%, 2.27%, and 7.48%. The zero validation EERs came from small selection cohorts and did not predict fold 3. This gap is evidence of high selection variance, not proof of backbone-level overfitting by itself.

The saved curves show these training-loss and selected-epoch values; SupCon and classification losses have different scales:

| Fold | Selected epoch / epochs run | Train loss, epoch 1→selected | Selection EER / AUC | Locked n=8 test EER |
|---|---:|---:|---:|---:|
| 1 | 7 / 13 | 1.784→1.166 | 0.0% / 1.000 | 2.13% |
| 2 | 3 / 9 | 3.737→3.351 | 0.0% / 1.000 | 2.27% |
| 3 | 1 / 7 | 3.726→3.726 | 0.0% / 1.000 | 7.48% |

## 6. Frozen-feature baselines

All methods use the same 8-person registry, 10-second enrollment, 3-second query, and calibration-only threshold protocol. The table gives pooled attempt counts and the macro mean of fold EERs; see the result archive for every fold, score quantiles, and intervals.

| Method | Representation | Known correct | Unknown false accepts | Macro fold EER |
|---|---|---:|---:|---:|
| Mean cosine | `pre_encoder` | 112/126 (88.9%) | 17/365 (4.66%) | 3.76% |
| Mean + std cosine | `pre_encoder` | 114/126 (90.5%) | 18/365 (4.93%) | 3.14% |
| Frame-L2 mean cosine | `pre_encoder` | 114/126 (90.5%) | 21/365 (5.75%) | 3.30% |
| Train-only LDA | `pre_encoder` | 107/126 (84.9%) | 38/365 (10.4%) | 3.23% |
| Mean cosine | `encoder.layer.7` | 104/126 (82.5%) | 9/365 (2.47%) | 4.07% |
| Frame-L2 mean cosine | `encoder.layer.7` | 101/126 (80.2%) | 8/365 (2.19%) | 4.78% |
| Mean + std cosine | `encoder.layer.7` | 103/126 (81.7%) | 9/365 (2.47%) | 4.61% |
| Train-only z-score cosine | `encoder.layer.7` | 110/126 (87.3%) | 25/365 (6.85%) | 2.65% |
| Train-only LDA | `encoder.layer.7` | 99/126 (78.6%) | 18/365 (4.93%) | 3.32% |
| Selected learned head, mixed by fold | `encoder.layer.7` | 103/126 (81.7%) | 13/365 (3.56%) | 3.96% |

The simple pre-encoder mean+std baseline narrowly meets the pooled 90%/5% target, but not in every fold: it accepted 34/41 known queries in fold 1, 41/44 in fold 2, and 39/41 in fold 3; its unknown false accepts were 6/122, 4/122, and 8/121. The selected head does not improve that trade consistently. At eight identities it is close to raw layer-7 cosine, while one fold is substantially worse. The untrained `encoder_output` and `encoder_proj` means were weak for identity acceptance; learning improved their pairwise separation, but the resulting known acceptance remained too low to select them.

The verification archive records fold-level and cross-fold speaker-cluster bootstrap intervals for these baselines, alongside their per-speaker macro rates. Cross-fold intervals merge repeated speaker IDs before resampling; the pooled attempt counts are not independent-trial confidence bounds.

## 7. Objective comparison and randomized-label control

At the same eight-person, 10-second enrollment, 3-second query condition, layer-7 results were:

| Fold | SupCon linear: known / unknown FA; EER | CE MLP: known / unknown FA; EER | Random-label SupCon: known / unknown FA; EER |
|---|---|---|---|
| 1 | 39/41; 2/122; 2.13% | 30/41; 4/122; 2.45% | 23/41; 3/122; 7.30% |
| 2 | 37/44; 5/122; 1.84% | 39/44; 4/122; 2.27% | 29/44; 2/122; 4.96% |
| 3 | 39/41; 8/121; 2.45% | 25/41; 7/121; 7.48% | 30/41; 5/121; 7.36% |

The random control shuffled class labels independently across training utterances, breaking same-speaker positive pairs. True labels beat the control on EER in folds 1 and 2, but not fold 3; at fold 3 the selected CE model also lost known acceptance to SupCon and the control. Random-label results retained some discrimination, consistent with the strong frozen-feature geometry and warning against attributing every gain to supervision. Model selection was performed on the selection speakers only and chose SupCon in fold 1 and CE in folds 2 and 3; evaluation outcomes did not alter that choice.

## 8. Open-set identification at eight identities

Each method had one threshold fit on its calibration speakers by leave-one-identity-out maximum impostor similarity, targeting 5% calibration false acceptance. The threshold was then frozen across all registry sizes and query durations. Selected-head calibration had 4 false accepts among 86, 80, and 80 unknown calibration queries in folds 1–3. Calibration known acceptance was 81/86, 78/80, and 43/80; the weak third-fold calibration acceptance foreshadowed poor held-out known acceptance.

| Fold and selected head | Threshold | Known correct / total (speaker-cluster 95% CI) | Unknown FA / total (speaker-cluster 95% CI) | Verification FAR; FRR | Pairwise EER; AUC |
|---|---:|---|---|---|---|
| 1, SupCon linear | .6848 | 39/41, 95.1% [88.5, 100] | 2/122, 1.64% [0, 3.90] | 3/1263; 2/41 | 2.13%; .9990 |
| 2, CE MLP | .6256 | 39/44, 88.6% [71.2, 100] | 4/122, 3.28% [.70, 7.32] | 15/1284; 4/44 | 2.27%; .9960 |
| 3, CE MLP | .8202 | 25/41, 61.0% [42.7, 82.6] | 7/121, 5.79% [0, 16.82] | 15/1255; 14/41 | 7.48%; .9798 |

Intervals use 500 speaker-cluster bootstrap resamples: eight known and 22 unknown speaker clusters per fold. The average within-fold speaker-macro known acceptance was 82.1%; macro unknown false acceptance was 4.0%. Pooled attempts were 103/126 known correct and 13/365 unknown false accepts. The 491 total open-set attempts comprised 103 correct known IDs, 3 wrong enrolled IDs, 20 known queries rejected as UNKNOWN, 352 correctly rejected unknowns, and 13 unknowns accepted as enrolled; there were no no-decisions at this condition. These pooled totals cross folds and must not be treated as independent observations.

The declared target is not met robustly: only fold 1 achieved both at least 90% known acceptance and at most 5% unknown false acceptance. Fold 2 narrowly missed known acceptance; fold 3 missed both. A verification FAR is a per-query/per-claimed-prototype trial rate, while the unknown false-accept rate above is a top-1 open-set query rate; they are different denominators. For illustration only, the evaluation-derived 5% verification-FAR DET points had 0/41, 1/44, and 7/41 genuine rejects by fold. Those thresholds use evaluation scores and were not used for the reported identity decisions. The full genuine, impostor, and unknown-top-1 quantiles and DET points are in [voiceid_task09b_verification.json.gz](../tests/results/voiceid_task09b_verification.json.gz).

Score overlap explains the fold spread. At eight identities, genuine-score medians were .764, .694, and .800; unknown top-1 medians were .455, .376, and .639. Fold 3’s unknown top-1 90th percentile was .798 and its maximum was .877, near or above its .8202 threshold; its genuine median was below threshold. Raising the threshold to protect unknown rejection therefore rejects many legitimate fold-3 queries.

## 9. Registry-size results

Each entry is `known correct / known queries; unknown false accepts / unknown queries`. The persistent registry does not consume a native diarization channel.

| Registry size | Fold 1 | Fold 2 | Fold 3 | Pooled attempts |
|---:|---|---|---|---|
| 1 | 4/4; 3/159 | 6/6; 6/160 | 4/5; 5/157 | 14/15; 14/476 (2.94% unknown FA) |
| 2 | 11/11; 3/152 | 10/10; 7/156 | 7/10; 4/152 | 28/31; 14/460 (3.04%) |
| 4 | 20/21; 3/142 | 19/22; 1/144 | 9/22; 4/140 | 48/65; 8/426 (1.88%) |
| 8 | 39/41; 2/122 | 39/44; 4/122 | 25/41; 7/121 | 103/126; 13/365 (3.56%) |
| 16 | 71/82; 8/81 | 79/91; 9/75 | 45/85; 5/77 | 195/258; 22/233 (9.44%) |

The archive contains each fold’s per-speaker macro rates and 500-resample bootstrap intervals for every size; the pooled columns reuse identities across folds.

At 16 users the measured unknown false-accept rate rose above 5% and known acceptance fell to 75.6%. This is a genuine speaker-independent evaluation up to 16 identities within the current 30-speaker evaluation roles, not evidence for accuracy at arbitrary registry sizes. Additional registry storage and comparisons scale with registry size; simultaneous diarization remains limited to the eight native channels.

## 10. Enrollment duration and prototype policy

The measured speech budget uses VAD-active duration and records whole-utterance overshoot. For the selected head at registry size 8 and 3-second queries:

| Enrollment target | Known correct | Unknown false accepts | Registry profile failures |
|---:|---:|---:|---:|
| 5s | 81/126 (64.3%) | 15/365 (4.11%) | 0 |
| 10s | 103/126 (81.7%) | 13/365 (3.56%) | 0 |
| 15s | 107/126 (84.9%) | 16/365 (4.38%) | 0 |
| 30s | 112/126 (88.9%) | 19/365 (5.21%) | 0 |
| 60s | 74/126 (58.7%) | 15/365 (4.11%) | 9 |

The 60-second failures occurred because the bounded registration utterance set did not contain enough actual speech. Longer enrollment did not improve acceptance monotonically. A learned clean-speech prototype can be made without waiting for AOSC cache readiness, so this path removes the approximately 27.5-second cache-readiness prerequisite structurally. It does not make short registration reliable: the five-second condition accepted only 81/126 known queries.

Enrollment-budget confidence intervals and per-fold attempts are included in the full result archives; the table is an attempt-level pooled summary.

The main policy uses a single pooled prototype. A validation-selected alternative averages L2-normalized utterance embeddings. At eight identities, 10-second enrollment, and 3-second queries, that alternative yielded 100/126 known accepts and 8/365 unknown false accepts: lower unknown acceptance but also lower known acceptance than single pooling. No max-over-many-prototypes policy was used. The complete budget and failure counts are in the compressed result archives.

## 11. Query duration

The primary query-budget results at registry size 8 and 10-second enrollment were:

| Query budget | Known correct | Unknown false accepts |
|---:|---:|---:|
| 1s | 68/174 (39.1%) | 3/453 (0.66%) |
| 3s | 103/126 (81.7%) | 13/365 (3.56%) |
| 7s | 56/60 (93.3%) | 10/176 (5.68%) |
| 15s | 25/26 (96.2%) | 3/76 (3.95%) |

The duration report additionally groups results by actual VAD speech duration. For `<2s`, the head accepted 68/174 known queries and falsely accepted 3/453 unknown queries. In the `2–5s` actual-duration bin (3-second query budget), it accepted 83/100 known queries and falsely accepted 7/282 unknowns. In the `5–10s` bin (7-second query budget), it accepted 55/59 known queries and falsely accepted 10/164 unknowns. In the `>10s` bin (15-second query budget), it accepted 25/26 known queries and falsely accepted 3/76 unknowns. The 1-, 3-, 7-, and 15-second groupings can reuse utterances across budgets; speaker-cluster intervals are reported per fold and those repeated query groups are not independent. See [voiceid_task09b_duration_buckets.json](../tests/results/voiceid_task09b_duration_buckets.json) for bucket-specific verification, open-set, score, and interval records.

## 12. Microphone and difficult-condition coverage

Mic1 was the primary condition. Mic2 robustness used up to four paired mic2 renderings of independent query utterances while registering only from mic1. At eight identities and 10-second enrollment, the selected heads accepted 27/96 mic2 known queries (28.1%) and falsely accepted 5/264 mic2 unknown queries (1.89%): 13/32 and 1/88 in fold 1, 12/32 and 2/88 in fold 2, and 2/32 and 2/88 in fold 3. This is poor known-speaker microphone transfer, not a pass on mic2 robustness. Paired mic1/mic2 versions were never split across enrollment and query.

The experiment covers clean isolated VCTK utterances, VAD duration, and the contexts described above. It does not measure overlap, background noise, gain perturbations, native diarization masks, or end-to-end simultaneous-speaker identity. No oracle masks or labels were used at identity scoring time. Same-text shortcuts and loudness effects were not isolated as separate ablations; repeated source utterance and microphone-pair leakage were checked by the manifest.

## 13. Historical baseline comparison

Task 08’s channel rule rejected 0/41 unknown queries with eight channels registered (40 false accepts and one no-decision); its calibrated rule reduced false accepts to 1/41 but accepted only 48/79 known queries. Task 09A’s selected eight-speaker profiles falsely accepted 59/60 unknown queries. Task 09B’s selected head produced 13/365 unknown false accepts at eight registered users, with 103/126 known accepts.

These are historical comparisons, not matched head-to-head percentages: the earlier tasks used AOSC construction, conversation ordering, and different cohorts. The channel matcher was not rerun on this fresh-stream single-utterance protocol because its mapping and state construction are not equivalent. Direct frozen-feature cosine, normalization, mean/std, and train-only LDA baselines were rerun on the Task 09B folds and are the controlled comparisons above. References: [Task 08 report](VOICEID_TASK08_REPORT.md) and [Task 09A report](VOICEID_TASK09A_REPORT.md).

## 14. Native integration and regression status

There is no C++/GGML VoiceID head implementation, no cross-language numerical parity result, and no derived single-GGUF. The learned head did not pass the quality gate, so head integration and GGUF packaging were not attempted. Original GGUF tensors remain unchanged. Normal diarization continues to work without any VoiceID head because the work adds only a research probe and research scripts; it adds no production inference path or public API.

Both configured regression trees passed after building the feature probe:

| Backend | CTest |
|---|---:|
| CPU | 7/7 passed |
| Vulkan | 7/7 passed |

This includes the existing C API model, AOSC profile intervention, state, mapping research, and snapshot continuity tests. The feature-on/off checks found zero native probability mismatches in the six Vulkan jobs summarized above. No normal diarization probability or C ABI source was changed.

## 15. Compute, storage, and limitations

Feature extraction took about 2.5 minutes on Vulkan0 for the fresh set; context extraction was a small additional run. The selected head’s measured Python CPU single-vector forward latency was about 204µs for the linear head and 474–548µs for the MLP. These timings exclude Nemotron feature extraction, include PyTorch dispatch, and do not predict native GGML latency. The CPU-only PyTorch environment prevented GPU training and GPU-memory measurement.

The three-fold speaker count is small, identities recur across folds, and VCTK does not provide independent recording-session labels for this split. Fold-to-fold performance and mic2 degradation are material. Calibration had only 15 speaker identities per fold; 4 false accepts among 80–86 calibration queries does not establish a precise population FAR. The bootstrap intervals reflect speaker clusters, but not a new external dataset. Results should not be represented as real-world verification accuracy or an EER guarantee.

Recommendation: keep the head as a research prototype. Before GGML/GGUF work, validate a frozen-feature baseline and selected head on a speaker population not previously used in this project, with independent sessions and microphones. This run does not justify backbone adaptation yet; the stable pre-encoder baseline and large fold variance make broader data the next useful evidence.

## 16. Reproduction

All commands below were run from the repository root. The cached VCTK conversions, feature records, profiles, checkpoints, and full uncompressed evaluator output live outside Git.

Create the fixed manifest and bounded audio catalogs:

```bash
nix-shell -p python3 python3Packages.numpy --run 'python3 tests/tools/prepare_task09b_splits.py --dataset-root VCTK-Corpus-0.92 --cache-dir /home/glebus/.cache/nemo-speech-task09b/vctk --jobs-tsv /home/glebus/.cache/nemo-speech-task09b/task09b_extraction_jobs.tsv --output tests/manifests/task09b_vctk_splits.json --folds 3 --seeds 20261021 20261022 20261023'
```

Build fresh features and AOSC context profiles using the explicitly selected Vulkan device:

```bash
nix-shell -p python3 python3Packages.numpy --run 'python3 tests/tools/extract_voiceid_features.py --jobs-tsv /home/glebus/.cache/nemo-speech-task09b/task09b_extraction_jobs.tsv --probe build/vulkan-diar/bin/diar_voiceid_feature_probe --model models/Nemotron-3-Diarization.q8_0.gguf --feature-dir /home/glebus/.cache/nemo-speech-task09b/features/fresh --device vulkan --gpu-device 0 --parity-first 3'
build/vulkan-diar/bin/diar_voiceid_feature_probe --build-aosc-profile /home/glebus/.cache/nemo-speech-task09b/vctk/task09b_aosc_profile_jobs_fold_1.tsv --model models/Nemotron-3-Diarization.q8_0.gguf --device vulkan --gpu-device 0 --profile-out /home/glebus/.cache/nemo-speech-task09b/vctk/aosc_context_profile_fold_1.bin
```

Repeat the profile command for `fold_2` and `fold_3`, substituting the fold in the TSV and output filename. Extract context probes, then train, evaluate, and analyze:

```bash
nix-shell -p python3 python3Packages.numpy --run 'python3 tests/tools/extract_voiceid_features.py --jobs-tsv /home/glebus/.cache/nemo-speech-task09b/vctk/task09b_context_jobs.tsv --probe build/vulkan-diar/bin/diar_voiceid_feature_probe --model models/Nemotron-3-Diarization.q8_0.gguf --feature-dir /home/glebus/.cache/nemo-speech-task09b/features/context --device vulkan --gpu-device 0 --parity-first 3'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/train_voiceid_head.py --manifest tests/manifests/task09b_vctk_splits.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --run-dir /home/glebus/.cache/nemo-speech-task09b/run --device cpu --threads 6 --representations pre_encoder,encoder.layer.7,encoder_output,encoder_proj --objectives supcon_linear,speaker_ce_mlp --negative-control'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/evaluate_voiceid_head.py --manifest tests/manifests/task09b_vctk_splits.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --run-dir /home/glebus/.cache/nemo-speech-task09b/run --training-results tests/results/voiceid_task09b_training.json --output /home/glebus/.cache/nemo-speech-task09b/verification_full.json --open-set-output /home/glebus/.cache/nemo-speech-task09b/open_set_full.json'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/evaluate_voiceid_head.py --duration-buckets-only --manifest tests/manifests/task09b_vctk_splits.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --run-dir /home/glebus/.cache/nemo-speech-task09b/run --training-results tests/results/voiceid_task09b_training.json --output /home/glebus/.cache/nemo-speech-task09b/duration_bucket_verification.json --open-set-output /home/glebus/.cache/nemo-speech-task09b/duration_bucket_open_set.json'
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/analyze_voiceid_context.py --manifest tests/manifests/task09b_vctk_splits.json --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --context-feature-dir /home/glebus/.cache/nemo-speech-task09b/features/context --cache-dir /home/glebus/.cache/nemo-speech-task09b/vctk --run-dir /home/glebus/.cache/nemo-speech-task09b/run --verification-results /home/glebus/.cache/nemo-speech-task09b/verification_full.json --output tests/results/voiceid_task09b_context.json'
nix-shell -p python3 python3Packages.numpy --run 'python3 tests/tools/package_task09b_results.py --verification /home/glebus/.cache/nemo-speech-task09b/verification_full.json --open-set /home/glebus/.cache/nemo-speech-task09b/open_set_full.json --duration-buckets /home/glebus/.cache/nemo-speech-task09b/duration_bucket_open_set.json --training tests/results/voiceid_task09b_training.json --output-dir tests/results'
```

The 16-identity cross-process registry replay and regression commands were:

```bash
nix-shell -p python3 python3Packages.numpy python3Packages.torch --run 'python3 tests/tools/replay_task09b_registry.py --manifest tests/manifests/task09b_vctk_splits.json --training-results tests/results/voiceid_task09b_training.json --verification-results tests/results/voiceid_task09b_verification.json.gz --pooled-features /home/glebus/.cache/nemo-speech-task09b/features/fresh/pooled_features.npz --run-dir /home/glebus/.cache/nemo-speech-task09b/run --registry /home/glebus/.cache/nemo-speech-task09b/voiceid_registry_fold1_replay.json --fold-id fold_1 --identity-count 16 --output tests/results/voiceid_task09b_registry.json'
cmake --build build/cpu-diar --target diar_voiceid_feature_probe -j6
ctest --test-dir build/cpu-diar --output-on-failure -j6
cmake --build build/vulkan-diar --target diar_voiceid_feature_probe -j6
ctest --test-dir build/vulkan-diar --output-on-failure -j4
```

Full metric archives are gzip-compressed JSON with every row preserved: [verification](../tests/results/voiceid_task09b_verification.json.gz) and [open set](../tests/results/voiceid_task09b_open_set.json.gz). The training, context, duration-bucket, registry-replay, split manifest, and feature-extraction summaries are separate small artifacts in `tests/results/` and `tests/manifests/`.
