#!/usr/bin/env python3
"""Select bounded frozen pre-encoder pooling/normalization on development roles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyze_task09c_microphones import (
    calibration_threshold,
    evaluate_condition,
    load_feature_union,
)
from task09b_voiceid import FeatureData, unit
from task09c_voiceid import (
    RawFeatureStore,
    apply_population_transform,
    embed_pool,
    fit_diagonal_standardizer,
    fit_regularized_whitener,
)


CONDITIONS = (("mic1", "mic1"), ("mic1", "mic2"), ("mic2", "mic1"), ("mic2", "mic2"))


class FrozenCandidate:
    def __init__(self, name: str, feature_data: FeatureData, raw_store: RawFeatureStore,
                 transform: dict[str, Any] | None = None):
        self.kind = name
        self.name = name
        self.representation = "pre_encoder"
        self.data = feature_data
        self.raw_store = raw_store
        self.transform = transform
        self.model_key = None
        self.validation_selected = False
        self.pool_mode = "single_pooled"
        self._job_ids = self.data.pooled["job_ids"].tolist()

    def embed_many(self, pools: list[dict[str, Any]]) -> np.ndarray:
        if not pools:
            return np.empty((0, self.data.dim), dtype=np.float32)
        vectors = []
        for pool in pools:
            if self.name in ("median", "trimmed_mean"):
                ids = [self._job_ids[index] for index in pool.get("selected_job_indices", [])]
                raw = self.raw_store.pool(ids)
                vectors.append(embed_pool(raw, self.name))
            elif self.name in ("train_zscore_mean", "train_whiten_mean"):
                vectors.append(np.asarray(pool["mean"], dtype=np.float32))
            else:
                pooling = ("frame_l2_mean" if self.name == "frame_l2_mean" else
                           "mean" if self.name == "mean_norm_utterance_proto" else self.name)
                vectors.append(embed_pool(pool, pooling))
        matrix = np.stack(vectors).astype(np.float32)
        if self.name in ("train_zscore_mean", "train_whiten_mean"):
            return apply_population_transform(matrix, self.transform)
        return unit(matrix)

    def prototype(self, job_ids: list[str], budget_sec: float) -> tuple[np.ndarray | None, float]:
        if self.name == "mean_norm_utterance_proto":
            rows = self.data.utterance_stats(job_ids, budget_sec)
            duration = sum(row["speech_duration_sec"] for row in rows)
            if duration < budget_sec or not rows:
                return None, duration
            vectors = unit(np.stack([row["mean"] for row in rows]))
            return unit(vectors.mean(axis=0)), duration
        pool = self.data.pool(job_ids, budget_sec)
        if pool["speech_duration_sec"] < budget_sec:
            return None, pool["speech_duration_sec"]
        return self.embed_many([pool])[0], float(pool["speech_duration_sec"])


def candidate_names() -> tuple[str, ...]:
    return ("mean", "mean_std", "frame_l2_mean", "mean_norm_utterance_proto",
            "median", "trimmed_mean", "train_zscore_mean", "train_whiten_mean")


def training_transform(name: str, fold: dict[str, Any], data: FeatureData) -> dict[str, Any] | None:
    if name not in ("train_zscore_mean", "train_whiten_mean"):
        return None
    job_ids = [job for jobs in fold["train_utterance_job_ids"].values() for job in jobs]
    indices = [data.index[job] for job in job_ids if data.speech_frames[data.index[job]] > 0]
    training_vectors = data.mean[indices]
    if name == "train_zscore_mean":
        return fit_diagonal_standardizer(training_vectors, floor=1e-4)
    return fit_regularized_whitener(training_vectors, shrinkage=.10, ridge_fraction=1e-3)


def compact_result(row: dict[str, Any]) -> dict[str, Any]:
    fields = ("fold_id", "enrollment_microphone", "query_microphone", "registry_size_requested",
              "registry_size_usable", "threshold", "enrollment_profile_attempts", "enrollment_profile_successes",
              "enrollment_profile_failures", "query_attempts", "known_queries", "known_correct_accepts",
              "known_wrong_identity_accepts", "known_rejected_as_unknown", "known_correct_accept_rate",
              "known_correct_accept_speaker_macro", "known_correct_accept_ci95", "known_rank1_correct",
              "unknown_queries", "unknown_false_accepts", "unknown_false_accept_rate",
              "unknown_false_accept_speaker_macro", "unknown_false_accept_ci95", "unknown_correct_rejections",
              "no_decisions", "no_decision_rate", "open_set_correct", "open_set_queries", "open_set_accuracy",
              "known_genuine_scores", "known_impostor_scores", "unknown_top1_scores", "verification_far",
              "verification_frr", "verification_eer_descriptive_only", "verification_auc_descriptive_only",
              "threshold_calibration")
    return {key: row[key] for key in fields if key in row}


def balanced_score(row: dict[str, Any]) -> float:
    known = row["known_correct_accept_speaker_macro"]
    unknown_fa = row["unknown_false_accept_speaker_macro"]
    if known is None or unknown_fa is None:
        return float("nan")
    return (float(known) + 1.0 - float(unknown_fa)) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--microphone-additions", type=Path, default=Path("tests/manifests/task09c_vctk_microphone_additions.json"))
    parser.add_argument("--pooled-features", type=Path, required=True)
    parser.add_argument("--additional-pooled-features", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--additional-feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09c_pooling.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    additions = json.loads(args.microphone_additions.read_text(encoding="utf-8"))
    base_assets = {row["job_id"]: row for row in manifest["assets"]}
    added_assets = {row["job_id"]: row for row in additions["new_assets"]}
    assets = {**base_assets, **added_assets}
    pooled = load_feature_union(args.pooled_features, args.additional_pooled_features)
    raw = RawFeatureStore([args.feature_dir, args.additional_feature_dir], "pre_encoder", max_cached_records=48)
    add_folds = {row["fold_id"]: row for row in additions["folds"]}
    development_rows: list[dict[str, Any]] = []
    per_fold_winners = []
    eval_rows: list[dict[str, Any]] = []

    for fold in manifest["folds"]:
        add_fold = add_folds[fold["fold_id"]]
        data = FeatureData(assets, pooled, "pre_encoder")
        candidates = []
        transforms = {}
        for name in candidate_names():
            transform = training_transform(name, fold, data)
            transforms[name] = transform
            candidates.append(FrozenCandidate(name, data, raw, transform))
        selection_fold = dict(fold)
        selection_fold["evaluation_people"] = fold["selection_people"]
        selection_fold["evaluation_registry_order"] = fold["selection_registry_order"]
        selection_add = {**add_fold, "evaluation_people": add_fold["selection_people"]}
        for candidate in candidates:
            for enrollment_mic, query_mic in CONDITIONS:
                threshold, calibration = calibration_threshold(candidate, fold, add_fold,
                                                               enrollment_mic, query_mic, 10.0)
                row, _ = evaluate_condition(candidate, selection_fold, selection_add,
                                            enrollment_mic, query_mic, threshold,
                                            56000 + len(development_rows) * 11, 8, 10.0)
                row.update({"method": candidate.name, "selection_score_balanced_macro": balanced_score(row),
                            "threshold_calibration": calibration})
                development_rows.append(row)
        scores_by_method = {}
        for candidate in candidates:
            selected_rows = [row for row in development_rows if row["fold_id"] == fold["fold_id"]
                             and row["method"] == candidate.name]
            scores_by_method[candidate.name] = float(np.nanmean([row["selection_score_balanced_macro"]
                                                                  for row in selected_rows]))
        fold_winner = max(candidate_names(), key=lambda name: scores_by_method[name])
        per_fold_winners.append({"fold_id": fold["fold_id"], "selected_by_development_only": fold_winner,
                                 "balanced_macro_score_by_method": scores_by_method})

    # Freeze one global method by averaging only model-selection-role scores.
    global_scores = {}
    for name in candidate_names():
        values = [row["selection_score_balanced_macro"] for row in development_rows if row["method"] == name]
        global_scores[name] = float(np.nanmean(values))
    selected_name = max(candidate_names(), key=lambda name: global_scores[name])
    candidate_map: dict[str, FrozenCandidate] = {}
    for fold in manifest["folds"]:
        add_fold = add_folds[fold["fold_id"]]
        data = FeatureData(assets, pooled, "pre_encoder")
        transform = training_transform(selected_name, fold, data)
        candidate = FrozenCandidate(selected_name, data, raw, transform)
        candidate_map[fold["fold_id"]] = candidate
        for enrollment_mic, query_mic in CONDITIONS:
            threshold, calibration = calibration_threshold(candidate, fold, add_fold,
                                                           enrollment_mic, query_mic, 10.0)
            row, _ = evaluate_condition(candidate, fold, add_fold, enrollment_mic, query_mic,
                                        threshold, 86000 + len(eval_rows) * 13, 8, 10.0)
            row.update({"method": selected_name, "threshold_calibration": calibration,
                        "selection_global_balanced_macro_score": global_scores[selected_name]})
            eval_rows.append(row)

    output = {
        "schema": "nemo-speech.voiceid-task09c-pooling-selection.v1",
        "dataset_status": "model selection and evaluation reuse previously studied VCTK identities; follow-up evidence, not external validation",
        "candidate_methods": list(candidate_names()),
        "candidate_definitions": {
            "mean": "512D mean of Task09B speech-masked pre-encoder frames",
            "mean_std": "1024D concatenated per-dimension mean and standard deviation; Task09B control",
            "frame_l2_mean": "L2-normalize each speech frame, then arithmetic mean",
            "mean_norm_utterance_proto": "L2-normalize each utterance mean before averaging enrollment utterances; one prototype",
            "median": "dimension-wise temporal median over speech-masked frames",
            "trimmed_mean": "dimension-wise 10% trimmed temporal mean over speech-masked frames",
            "train_zscore_mean": "training-speaker-only per-dimension standardization; 1e-4 scale floor and +/-8 clip",
            "train_whiten_mean": "training-speaker-only 512D covariance whitening; 10% diagonal shrinkage and 0.1% ridge",
        },
        "protocol": {
            "method_selection": "global method chosen only by mean balanced speaker-macro open-set accuracy on the 15 model-selection speakers per fold; known acceptance and unknown rejection receive equal weight",
            "thresholds": "per-method, per-enrollment-mic, per-query-mic thresholds fitted only on the 15 calibration speakers, using leave-one-identity-out maximum impostor scores",
            "evaluation": "one global selected pooling method evaluated once on the 30 evaluation speakers per fold",
            "registry_size": 8, "enrollment_budget_sec": 10,
            "paired_queries": "up to four same-source-utterance mic1/mic2 pairs; query clips are not treated as independent trials",
            "normalization_fit_role": "50 training speakers per fold; no selection, calibration, or evaluation identity contributes to fitted transforms within its fold",
        },
        "global_selection_balanced_macro_score_by_method": global_scores,
        "selected_global_method": selected_name,
        "fold_specific_development_winners_diagnostic_only": per_fold_winners,
        "development_rows": [compact_result(row) | {"method": row["method"],
                                                     "selection_score_balanced_macro": row["selection_score_balanced_macro"]}
                             for row in development_rows],
        "locked_evaluation_selected_method": [compact_result(row) | {
            "method": row["method"], "selection_global_balanced_macro_score": row["selection_global_balanced_macro_score"]}
            for row in eval_rows],
        "fit_transform_metadata_by_fold": {
            fold_id: ({"kind": transform["kind"], "floor": transform.get("floor"),
                       "shrinkage": transform.get("shrinkage"), "ridge_fraction": transform.get("ridge_fraction"),
                       "ridge": transform.get("ridge"), "eigenvalue_floor": transform.get("eigenvalue_floor"),
                       "input_dimension": int(len(transform["center"]))}
                      if transform else None)
            for fold_id, transform in ((fold["fold_id"], training_transform(selected_name, fold,
                                            FeatureData(assets, pooled, "pre_encoder"))) for fold in manifest["folds"])
        },
        "uncertainty_note": "speaker-cluster bootstrap intervals are within fold; identities recur across folds and pooled counts are not independent trials",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_global_method": selected_name,
                      "global_scores": global_scores, "evaluation_rows": len(eval_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"evaluate_task09c_pooling: {exc}", file=sys.stderr)
        raise
