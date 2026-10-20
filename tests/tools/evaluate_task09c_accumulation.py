#!/usr/bin/env python3
"""Measure short-query and progressive frozen pre-encoder evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_voiceid_head import Method, threshold_for_method
from task09b_voiceid import FeatureData, cosine, load_data
from task09c_voiceid import RawFeatureStore, progressive_prefix


BUDGETS = (1, 2, 3, 5, 7, 10, 15)


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min_p10_p25_p50_p75_p90_max": None}
    return {"count": len(values), "min_p10_p25_p50_p75_p90_max":
            [float(value) for value in np.quantile(np.asarray(values), [0, .1, .25, .5, .75, .9, 1])]}


def cluster_ci(counts: dict[str, tuple[int, int]], seed: int) -> dict[str, Any]:
    rates = np.asarray([num / den for num, den in counts.values() if den], dtype=np.float64)
    if not len(rates):
        return {"lower_95": None, "upper_95": None, "speaker_clusters": 0,
                "method": "speaker-cluster bootstrap, 1000 resamples"}
    rng = np.random.default_rng(seed)
    samples = rng.choice(rates, size=(1000, len(rates)), replace=True).mean(axis=1)
    return {"lower_95": float(np.quantile(samples, .025)), "upper_95": float(np.quantile(samples, .975)),
            "speaker_clusters": int(len(rates)), "method": "speaker-cluster bootstrap over per-speaker rates, 1000 resamples"}


def utterance_frames(store: RawFeatureStore, job_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frames = []
    weights = []
    for job_id in job_ids:
        values, active_bits, _ = store.frames(job_id)
        if len(values):
            frames.append(values)
            weights.append(active_bits.astype(np.float64) * .01)
    if not frames:
        return np.empty((0, 512), dtype=np.float32), np.empty(0, dtype=np.float64)
    return np.concatenate(frames, axis=0), np.concatenate(weights)


def decide(query: np.ndarray, prototypes: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    scores = {speaker: cosine(query, prototype) for speaker, prototype in prototypes.items()}
    if not scores or not np.isfinite(list(scores.values())).all():
        return {"top_id": None, "top_score": None, "accepted_id": None, "scores": scores}
    top_id = max(scores, key=scores.get)
    top_score = float(scores[top_id])
    return {"top_id": top_id, "top_score": top_score,
            "accepted_id": top_id if top_score >= threshold else None, "scores": scores}


def summarize_independent_turns(fold: dict[str, Any], method: Method, store: RawFeatureStore,
                                prototypes: dict[str, np.ndarray], threshold: float,
                                budgets: tuple[int, ...], seed: int) -> list[dict[str, Any]]:
    people = fold["evaluation_people"]
    registry = set(fold["evaluation_registry_order"][:8])
    by_budget: dict[int, dict[str, Any]] = {}
    for budget in budgets:
        counts: dict[str, dict[str, int]] = {}
        genuine_scores: list[float] = []
        unknown_scores: list[float] = []
        for speaker, person in people.items():
            known = speaker in registry
            values = counts.setdefault(speaker, {"attempts": 0, "sufficient": 0, "correct": 0,
                                                 "wrong": 0, "rejected": 0, "unknown_fa": 0,
                                                 "unknown_rejected": 0, "no_decision": 0})
            for job_id in person["query_job_ids"]:
                frames, weights = utterance_frames(store, [job_id])
                values["attempts"] += 1
                vector, actual, _ = progressive_prefix(frames, weights, float(budget))
                if actual + 1e-9 < budget:
                    values["no_decision"] += 1
                    continue
                values["sufficient"] += 1
                embedding = method.embed_many([{"mean": vector}])[0]
                decision = decide(embedding, prototypes, threshold)
                if decision["top_score"] is None:
                    values["no_decision"] += 1
                    continue
                if known:
                    if speaker in decision["scores"]:
                        genuine_scores.append(decision["scores"][speaker])
                    if decision["accepted_id"] == speaker:
                        values["correct"] += 1
                    elif decision["accepted_id"] is None:
                        values["rejected"] += 1
                    else:
                        values["wrong"] += 1
                else:
                    unknown_scores.append(float(decision["top_score"]))
                    if decision["accepted_id"] is None:
                        values["unknown_rejected"] += 1
                    else:
                        values["unknown_fa"] += 1
        known_speakers = {s: v for s, v in counts.items() if s in registry}
        unknown_speakers = {s: v for s, v in counts.items() if s not in registry}
        known_total = sum(v["attempts"] for v in known_speakers.values())
        unknown_total = sum(v["attempts"] for v in unknown_speakers.values())
        known_sufficient = sum(v["sufficient"] for v in known_speakers.values())
        unknown_sufficient = sum(v["sufficient"] for v in unknown_speakers.values())
        known_correct = sum(v["correct"] for v in known_speakers.values())
        unknown_fa = sum(v["unknown_fa"] for v in unknown_speakers.values())
        known_ci_counts = {s: (v["correct"], v["attempts"]) for s, v in known_speakers.items()}
        unknown_ci_counts = {s: (v["unknown_fa"], v["attempts"]) for s, v in unknown_speakers.items()}
        by_budget[budget] = {
            "fold_id": fold["fold_id"], "query_evidence_budget_sec": budget,
            "method": "train_zscore_cosine on pre_encoder mean; Task 09B training-only statistics",
            "decision_threshold": threshold, "query_attempts": known_total + unknown_total,
            "known_queries": known_total, "known_sufficient_evidence": known_sufficient,
            "known_correct_accepts": known_correct, "known_correct_accept_rate_all_attempts": known_correct / known_total if known_total else None,
            "known_correct_accept_rate_sufficient_evidence": known_correct / known_sufficient if known_sufficient else None,
            "known_wrong_identity_accepts": sum(v["wrong"] for v in known_speakers.values()),
            "known_rejected_as_unknown": sum(v["rejected"] for v in known_speakers.values()),
            "known_no_decisions": sum(v["no_decision"] for v in known_speakers.values()),
            "known_correct_accept_ci95_all_attempts": cluster_ci(known_ci_counts, seed + budget * 17),
            "unknown_queries": unknown_total, "unknown_sufficient_evidence": unknown_sufficient,
            "unknown_false_accepts": unknown_fa,
            "unknown_false_accept_rate_all_attempts": unknown_fa / unknown_total if unknown_total else None,
            "unknown_false_accept_rate_sufficient_evidence": unknown_fa / unknown_sufficient if unknown_sufficient else None,
            "unknown_correct_rejections": sum(v["unknown_rejected"] for v in unknown_speakers.values()),
            "unknown_no_decisions": sum(v["no_decision"] for v in unknown_speakers.values()),
            "unknown_false_accept_ci95_all_attempts": cluster_ci(unknown_ci_counts, seed + budget * 29),
            "genuine_score_distribution": quantiles(genuine_scores),
            "unknown_top1_score_distribution": quantiles(unknown_scores),
            "per_speaker": counts,
            "insufficient_evidence_policy": "abstain when the individual utterance has not reached the requested speech budget; no query fragments are joined using identity labels",
        }
    return list(by_budget.values())


def summarize_oracle_turn_accumulation(fold: dict[str, Any], method: Method, store: RawFeatureStore,
                                       prototypes: dict[str, np.ndarray], threshold: float,
                                       budgets: tuple[int, ...], seed: int) -> list[dict[str, Any]]:
    people = fold["evaluation_people"]
    registry = set(fold["evaluation_registry_order"][:8])
    decisions_by_speaker: dict[str, dict[int, dict[str, Any]]] = {}
    rows = []
    for speaker, person in people.items():
        frames, weights = utterance_frames(store, person["query_job_ids"])
        decisions_by_speaker[speaker] = {}
        for budget in budgets:
            vector, actual, frame_count = progressive_prefix(frames, weights, float(budget))
            if actual + 1e-9 < budget:
                decisions_by_speaker[speaker][budget] = {"actual_sec": actual, "no_decision": True,
                                                         "accepted_id": None, "top_id": None, "top_score": None}
                continue
            embedding = method.embed_many([{"mean": vector}])[0]
            decision = decide(embedding, prototypes, threshold)
            decisions_by_speaker[speaker][budget] = {
                "actual_sec": actual, "feature_frames": frame_count, "no_decision": decision["top_score"] is None,
                "accepted_id": decision["accepted_id"], "top_id": decision["top_id"],
                "top_score": decision["top_score"], "genuine_score": decision["scores"].get(speaker),
            }
    for budget in budgets:
        counts = {"known_queries": 0, "known_correct_accepts": 0, "known_wrong_identity_accepts": 0,
                  "known_rejected_as_unknown": 0, "known_no_decisions": 0, "unknown_queries": 0,
                  "unknown_false_accepts": 0, "unknown_correct_rejections": 0, "unknown_no_decisions": 0}
        actual_durations = []
        genuine_scores = []
        unknown_scores = []
        per_speaker = {}
        for speaker, decisions in decisions_by_speaker.items():
            decision = decisions[budget]
            actual_durations.append(decision["actual_sec"])
            known = speaker in registry
            per_speaker[speaker] = {**decision, "known": known}
            if known:
                counts["known_queries"] += 1
                if decision["no_decision"]:
                    counts["known_no_decisions"] += 1
                elif decision["accepted_id"] == speaker:
                    counts["known_correct_accepts"] += 1
                elif decision["accepted_id"] is None:
                    counts["known_rejected_as_unknown"] += 1
                else:
                    counts["known_wrong_identity_accepts"] += 1
                if decision.get("genuine_score") is not None:
                    genuine_scores.append(float(decision["genuine_score"]))
            else:
                counts["unknown_queries"] += 1
                if decision["no_decision"]:
                    counts["unknown_no_decisions"] += 1
                elif decision["accepted_id"] is None:
                    counts["unknown_correct_rejections"] += 1
                else:
                    counts["unknown_false_accepts"] += 1
                if decision.get("top_score") is not None:
                    unknown_scores.append(float(decision["top_score"]))
        first_correct = {}
        for speaker in registry:
            first = next((budget_value for budget_value in budgets
                          if decisions_by_speaker[speaker][budget_value].get("accepted_id") == speaker), None)
            first_correct[speaker] = first
        identity_stability = {}
        transitions = 0
        stable_transitions = 0
        for earlier, later in zip(budgets, budgets[1:]):
            compare = [(decisions_by_speaker[s][earlier].get("accepted_id"),
                        decisions_by_speaker[s][later].get("accepted_id")) for s in decisions_by_speaker]
            identity_stability[f"{earlier}_to_{later}_sec"] = {
                "same_accepted_identity": sum(a == b for a, b in compare), "speaker_queries": len(compare),
                "changed_or_abstained": sum(a != b for a, b in compare),
            }
            transitions += len(compare)
            stable_transitions += sum(a == b for a, b in compare)
        rows.append({
            "fold_id": fold["fold_id"], "query_evidence_budget_sec": budget,
            "condition": "oracle_same_speaker_turn_accumulation_upper_bound",
            "query_grouping": "concatenates independent query utterances using their ground-truth speaker identity; not a deployable grouping rule",
            "decision_threshold": threshold, **counts,
            "known_correct_accept_rate": counts["known_correct_accepts"] / counts["known_queries"] if counts["known_queries"] else None,
            "unknown_false_accept_rate_all_attempts": counts["unknown_false_accepts"] / counts["unknown_queries"] if counts["unknown_queries"] else None,
            "actual_accumulated_speech_sec": quantiles(actual_durations),
            "genuine_score_distribution": quantiles(genuine_scores),
            "unknown_top1_score_distribution": quantiles(unknown_scores),
            "first_correct_accept_budget_sec_by_known_speaker": first_correct,
            "known_speakers_correct_at_some_budget": sum(value is not None for value in first_correct.values()),
            "identity_stability_by_increment": identity_stability,
            "identity_stability_rate_across_transitions": stable_transitions / transitions if transitions else None,
            "per_speaker": per_speaker,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--pooled-features", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09c_duration.json"))
    args = parser.parse_args()
    manifest, assets, pooled = load_data(args.manifest, args.pooled_features)
    store = RawFeatureStore(args.feature_dir, "pre_encoder", max_cached_records=64)
    independent_rows = []
    oracle_rows = []
    thresholds = []
    for fold in manifest["folds"]:
        data = FeatureData(assets, pooled, "pre_encoder")
        train_ids = [job for jobs in fold["train_utterance_job_ids"].values() for job in jobs]
        method = Method("train_zscore_cosine", "pre_encoder", data, fold["train_utterance_job_ids"])
        threshold, calibration = threshold_for_method(method, fold)
        prototypes = {}
        profile_durations = {}
        profile_failures = []
        for speaker in fold["evaluation_registry_order"][:8]:
            vector, duration = method.prototype(fold["evaluation_people"][speaker]["enrollment_job_ids"], 10.0)
            if vector is None:
                profile_failures.append(speaker)
            else:
                prototypes[speaker] = vector
                profile_durations[speaker] = duration
        thresholds.append({"fold_id": fold["fold_id"], "threshold": threshold,
                           "calibration": calibration, "registry_profile_failures": profile_failures,
                           "registry_profile_successes": len(prototypes),
                           "actual_enrollment_speech_sec": profile_durations})
        independent_rows.extend(summarize_independent_turns(fold, method, store, prototypes, threshold,
                                                             BUDGETS, 61000 + int(fold["speaker_seed"])))
        oracle_rows.extend(summarize_oracle_turn_accumulation(fold, method, store, prototypes, threshold,
                                                               BUDGETS, 71000 + int(fold["speaker_seed"])))
    output = {
        "schema": "nemo-speech.voiceid-task09c-duration-accumulation.v1",
        "evaluation_status": "VCTK follow-up; no new speaker population or independent recording session",
        "representation": "frozen pre_encoder mean with per-dimension center/scale fitted on 50 training speakers per fold",
        "enrollment": {"requested_speech_sec": 10, "registry_size": 8,
                       "profile_construction": "single pooled prototype from mic1 enrollment utterances; no AOSC data or channel assignment",
                       "fold_results": thresholds},
        "decision_policy": "one 09B calibration threshold per fold, fixed across all query evidence budgets; abstain until the requested actual VAD speech duration is available",
        "independent_utterance_prefixes": independent_rows,
        "oracle_multi_turn_accumulation_upper_bound": oracle_rows,
        "matched_learned_head_reference": {
            "task09b_selected_head_1s": {"known_correct_accepts": 68, "known_queries": 174,
                                         "unknown_false_accepts": 3, "unknown_queries": 453},
            "source": "Task 09B evaluation report, same 8-person/10s enrollment condition; the grouped-query protocol differs from the prefix measurements above",
        },
        "limitations": "same-speaker multi-utterance concatenation is labeled an oracle upper bound because it uses ground-truth speaker grouping; the independent-prefix rows are the deployable evidence condition",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "independent_rows": len(independent_rows),
                      "oracle_rows": len(oracle_rows), "selected_method": "train_zscore_cosine"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"evaluate_task09c_accumulation: {exc}", file=sys.stderr)
        raise
