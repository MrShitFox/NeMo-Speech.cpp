#!/usr/bin/env python3
"""Measure frozen pre-encoder identity decisions under Task 09B and repeat-turn contexts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_voiceid_head import Method, threshold_for_method
from extract_voiceid_features import read_feature
from task09b_voiceid import FeatureData, cosine, load_data
from task09c_voiceid import RawFeatureStore


CONTEXT_MODES = ("silent_2_56s", "unrelated_train_speaker_a", "unrelated_train_speaker_b",
                 "aosc_profile_import", "same_speaker_prior_turn")


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min_p10_p25_p50_p75_p90_max": None}
    return {"count": len(values), "min_p10_p25_p50_p75_p90_max":
            [float(value) for value in np.quantile(np.asarray(values), [0, .1, .25, .5, .75, .9, 1])]}


def cluster_ci(counts: dict[str, tuple[int, int]], seed: int) -> dict[str, Any]:
    rates = np.asarray([num / den for num, den in counts.values() if den], dtype=np.float64)
    if not len(rates):
        return {"lower_95": None, "upper_95": None, "speaker_clusters": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(rates, size=(1000, len(rates)), replace=True).mean(axis=1)
    return {"lower_95": float(np.quantile(samples, .025)), "upper_95": float(np.quantile(samples, .975)),
            "speaker_clusters": int(len(rates)), "method": "speaker-cluster bootstrap, 1000 resamples"}


def context_records(feature_dir: Path) -> dict[str, dict[str, Any]]:
    metadata = [json.loads(line) for line in (feature_dir / "features.jsonl").read_text().splitlines() if line.strip()]
    result = {}
    for row in metadata:
        if row["feature_representation"] != "pre_encoder":
            continue
        result[row["job_id"]] = row
    return result


def read_raw(feature_dir: Path, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    path = feature_dir / record["record_file"]
    _, frame_ids, speech_bits, values, digest = read_feature(path)
    return frame_ids, speech_bits, values, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--task09c-context-manifest", type=Path, default=Path("tests/manifests/task09c_vctk_context.json"))
    parser.add_argument("--pooled-features", type=Path, required=True)
    parser.add_argument("--fresh-feature-dir", type=Path, required=True)
    parser.add_argument("--task09b-context-feature-dir", type=Path, required=True)
    parser.add_argument("--task09c-context-feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09c_context.json"))
    args = parser.parse_args()

    manifest, assets, pooled = load_data(args.manifest, args.pooled_features)
    task09c_context = json.loads(args.task09c_context_manifest.read_text(encoding="utf-8"))
    task09b_context_rows = {row["job_id"]: row for row in manifest["context_stress_protocol"]["context_probe_jobs"]}
    task09b_context_meta = context_records(args.task09b_context_feature_dir)
    task09c_context_meta = context_records(args.task09c_context_feature_dir)
    if set(task09b_context_meta) != set(task09b_context_rows):
        raise RuntimeError("Task 09B context feature metadata does not cover its fixed context manifest")
    task09c_jobs = {row["job_id"]: row for fold in task09c_context["folds"] for row in fold["jobs"]}
    if set(task09c_context_meta) != set(task09c_jobs):
        raise RuntimeError("Task 09C same-speaker context feature metadata coverage mismatch")
    stores = RawFeatureStore([args.fresh_feature_dir, args.task09b_context_feature_dir,
                              args.task09c_context_feature_dir], "pre_encoder", max_cached_records=64)
    all_fold_results = []

    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        data = FeatureData(assets, pooled, "pre_encoder")
        method = Method("train_zscore_cosine", "pre_encoder", data, fold["train_utterance_job_ids"])
        threshold, calibration = threshold_for_method(method, fold)
        registry = list(fold["evaluation_registry_order"][:8])
        prototypes = {}
        profile_attempts = 0
        profile_failures = []
        profile_durations = {}
        for speaker in registry:
            profile_attempts += 1
            person = fold["evaluation_people"][speaker]
            vector, actual = method.prototype(person["enrollment_job_ids"], 10.0)
            if vector is None:
                profile_failures.append(speaker)
            else:
                prototypes[speaker] = vector
                profile_durations[speaker] = actual

        fold_contexts = {row["context_job_id"]: row for row in fold["context_probe_jobs"]}
        job_to_mode: dict[str, dict[str, Any]] = {}
        for context_job_id, target in task09b_context_rows.items():
            if context_job_id not in fold_contexts:
                continue
            job_to_mode.setdefault(target["target_job_id"], {})[target["context_mode"]] = {
                "job_id": context_job_id, "mode": target["context_mode"],
                "context_speaker_id": target.get("context_speaker_id"),
            }
        for row in task09c_context["folds"]:
            if row["fold_id"] == fold_id:
                for item in row["jobs"]:
                    job_to_mode.setdefault(item["target_job_id"], {})[item["context_mode"]] = {
                        "job_id": item["job_id"], "mode": item["context_mode"],
                        "context_speaker_id": item["speaker_id"],
                        "prior_turn_job_id": item["prior_turn_job_id"],
                    }
        registry_set = set(registry)
        decision_rows: list[dict[str, Any]] = []
        feature_rows: list[dict[str, Any]] = []
        for target_id, contexts in sorted(job_to_mode.items()):
            target_asset = assets[target_id]
            speaker = target_asset["speaker_id"]
            if not contexts:
                continue
            base_pool = data.pool([target_id])
            base_vector = method.embed_many([base_pool])[0]
            base_frames, base_bits, base_values, _ = read_raw(
                args.fresh_feature_dir, {"record_file": f"{target_id}.pre_encoder.vfeat"})
            vectors = {"fresh_empty_stream": (base_vector, base_pool)}
            for mode, info in contexts.items():
                if mode in CONTEXT_MODES[:4]:
                    feature_dir, meta = args.task09b_context_feature_dir, task09b_context_meta[info["job_id"]]
                else:
                    feature_dir, meta = args.task09c_context_feature_dir, task09c_context_meta[info["job_id"]]
                frame_ids, bits, values, record_sha = read_raw(feature_dir, meta)
                context_pool = stores.pool([info["job_id"]])
                context_vector = method.embed_many([context_pool])[0]
                vectors[mode] = (context_vector, context_pool)
                common_ids = np.intersect1d(base_frames, frame_ids, assume_unique=True)
                base_index = np.searchsorted(base_frames, common_ids)
                context_index = np.searchsorted(frame_ids, common_ids)
                aligned_equal = bool(len(common_ids) and np.array_equal(base_values[base_index], values[context_index]))
                common_mask_agreement = float(np.mean(base_bits[base_index] == bits[context_index])) if len(common_ids) else None
                feature_rows.append({
                    "fold_id": fold_id, "target_job_id": target_id, "speaker_id": speaker,
                    "known_in_registry8": speaker in registry_set, "context_mode": mode,
                    "fresh_frame_count": int(len(base_frames)), "context_frame_count": int(len(frame_ids)),
                    "common_target_frames": int(len(common_ids)),
                    "frame_id_axes_equal": bool(np.array_equal(base_frames, frame_ids)),
                    "speech_mask_agreement_on_common_frame_ids": common_mask_agreement,
                    "mean_pooled_cosine_to_fresh": cosine(base_pool["mean"], context_pool["mean"]),
                    "max_abs_frame_difference_common_ids": float(np.max(np.abs(base_values[base_index] - values[context_index]))) if len(common_ids) else None,
                    "bit_identical_common_target_features": aligned_equal,
                    "context_record_sha256": record_sha,
                    "prior_turn_job_id": info.get("prior_turn_job_id"),
                })
            for mode, (vector, pool) in vectors.items():
                scores = {identity: cosine(vector, prototype) for identity, prototype in prototypes.items()}
                top_id = max(scores, key=scores.get) if scores else None
                top_score = float(scores[top_id]) if top_id is not None else None
                accepted = top_id if top_score is not None and top_score >= threshold else None
                decision_rows.append({
                    "fold_id": fold_id, "target_job_id": target_id, "speaker_id": speaker,
                    "known_in_registry8": speaker in registry_set, "context_mode": mode,
                    "top_identity": top_id, "top_score": top_score,
                    "accepted_identity": accepted, "genuine_score": scores.get(speaker),
                    "identity_cosine_to_fresh": cosine(base_vector, vector),
                    "actual_query_speech_sec": pool["speech_duration_sec"],
                })

        fresh_decisions = {row["target_job_id"]: row for row in decision_rows if row["context_mode"] == "fresh_empty_stream"}
        per_mode = []
        for mode in ("fresh_empty_stream", *CONTEXT_MODES):
            rows = [row for row in decision_rows if row["context_mode"] == mode]
            known_rows = [row for row in rows if row["known_in_registry8"]]
            unknown_rows = [row for row in rows if not row["known_in_registry8"]]
            known_speaker_rates = {}
            unknown_speaker_rates = {}
            for row in known_rows:
                sp = row["speaker_id"]
                den, num = known_speaker_rates.get(sp, (0, 0))
                known_speaker_rates[sp] = (den + 1, num + int(row["accepted_identity"] == sp))
            for row in unknown_rows:
                sp = row["speaker_id"]
                den, num = unknown_speaker_rates.get(sp, (0, 0))
                unknown_speaker_rates[sp] = (den + 1, num + int(row["accepted_identity"] is not None))
            known_ok = sum(row["accepted_identity"] == row["speaker_id"] for row in known_rows)
            unknown_fa = sum(row["accepted_identity"] is not None for row in unknown_rows)
            if mode == "fresh_empty_stream":
                changed_top = changed_accept = 0
            else:
                compared = [(fresh_decisions[row["target_job_id"]], row) for row in rows
                            if row["target_job_id"] in fresh_decisions]
                changed_top = sum(base["top_identity"] != row["top_identity"] for base, row in compared)
                changed_accept = sum(base["accepted_identity"] != row["accepted_identity"] for base, row in compared)
            per_mode.append({
                "fold_id": fold_id, "context_mode": mode, "registry_size": len(prototypes),
                "threshold": threshold, "threshold_calibration": calibration,
                "known_correct_accepts": known_ok, "known_queries": len(known_rows),
                "known_correct_accept_rate": known_ok / len(known_rows) if known_rows else None,
                "known_correct_accept_speaker_macro": float(np.mean([n / d for d, n in known_speaker_rates.values()])) if known_speaker_rates else None,
                "known_correct_accept_ci95": cluster_ci({sp: (n, d) for sp, (d, n) in known_speaker_rates.items()}, 60900 + len(per_mode)),
                "known_wrong_identity_accepts": sum(row["accepted_identity"] not in (None, row["speaker_id"]) for row in known_rows),
                "known_rejected_as_unknown": sum(row["accepted_identity"] is None for row in known_rows),
                "unknown_false_accepts": unknown_fa, "unknown_queries": len(unknown_rows),
                "unknown_false_accept_rate": unknown_fa / len(unknown_rows) if unknown_rows else None,
                "unknown_false_accept_speaker_macro": float(np.mean([n / d for d, n in unknown_speaker_rates.values()])) if unknown_speaker_rates else None,
                "unknown_false_accept_ci95": cluster_ci(unknown_speaker_rates, 70900 + len(per_mode)),
                "top_identity_changes_vs_fresh": changed_top,
                "thresholded_acceptance_changes_vs_fresh": changed_accept,
                "identity_cosine_to_fresh": quantiles([row["identity_cosine_to_fresh"] for row in rows]),
                "known_genuine_score_distribution": quantiles([float(row["genuine_score"]) for row in known_rows if row["genuine_score"] is not None]),
                "unknown_top_score_distribution": quantiles([float(row["top_score"]) for row in unknown_rows if row["top_score"] is not None]),
                "target_speakers": len({row["speaker_id"] for row in rows}),
            })
        all_fold_results.append({
            "fold_id": fold_id, "representation": "pre_encoder 512D mean; training-only diagonal z-score",
            "registry_size": len(prototypes), "profile_attempts": profile_attempts,
            "profile_successes": len(prototypes), "profile_failures": profile_failures,
            "actual_profile_speech_sec": profile_durations,
            "feature_context_comparison": feature_rows,
            "identity_decision_comparison": per_mode,
            "per_target_decisions": decision_rows,
        })

    output = {
        "schema": "nemo-speech.voiceid-task09c-context.v1",
        "evaluation_status": "VCTK follow-up using previously studied evaluation speakers; context targets are paired utterance probes, not independent sessions",
        "representation": "frozen pre_encoder mean with Task 09B training-speaker-only per-dimension z-score; one profile per arbitrary external speaker ID",
        "protocol": {
            "registry_size": 8, "enrollment_budget_sec": 10,
            "threshold": "one fixed Task 09B calibration threshold per fold; applied unchanged to fresh and contextual queries",
            "contexts": ["fresh_empty_stream", *CONTEXT_MODES],
            "same_speaker_prior_turn": "for each target, the other disjoint query utterance from that same evaluation speaker is streamed first, followed by 250ms silence; same-speaker grouping is an oracle context stress test, not a deployable turn-association rule",
            "aosc_import": "uses the existing Task 09B research profile fixture built from training speakers; no AOSC context is required to construct the frozen profile",
            "feature_comparison": "matching target-relative frame IDs are compared separately from pooled representation and identity decisions to expose frontend or boundary changes",
        },
        "folds": all_fold_results,
        "limitations": "The frozen feature is observed before Transformer/AOSC context; changes can still arise from streaming frontend state and target frame alignment. The reused Task 09B threshold was calibrated on grouped 3-second queries, while context checks use individual full query clips.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "folds": len(all_fold_results),
                      "target_decision_rows": sum(len(row["per_target_decisions"]) for row in all_fold_results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"analyze_task09c_context: {exc}", file=sys.stderr)
        raise
