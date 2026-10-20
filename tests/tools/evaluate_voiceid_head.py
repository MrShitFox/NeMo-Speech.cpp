#!/usr/bin/env python3
"""Calibrate and evaluate Task 09B heads/baselines on locked unseen speakers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from task09b_voiceid import FeatureData, concat_pool, cosine, eer_threshold, far_target_threshold, load_data, rank_auc, unit


ENROLLMENT_BUDGETS = [5, 10, 15, 30, 60]
QUERY_BUDGETS = [1, 3, 7, 15]
REGISTRY_SIZES = [1, 2, 4, 8, 16]
REPRESENTATIONS = ["pre_encoder", "encoder.layer.7", "encoder_output", "encoder_proj"]


class Method:
    def __init__(self, kind: str, representation: str, feature_data: FeatureData,
                 training_speakers: dict[str, list[str]], bundle_path: Path | None = None,
                 pool_mode: str = "single_pooled"):
        self.kind = kind
        self.representation = representation
        self.data = feature_data
        self.pool_mode = pool_mode
        self.bundle_path = bundle_path
        self.bundle = None
        self.model_key: str | None = None
        self.validation_selected = False
        train_ids = [job_id for values in training_speakers.values() for job_id in values]
        indices = [feature_data.index[job_id] for job_id in train_ids
                   if feature_data.speech_frames[feature_data.index[job_id]] > 0]
        train_means = feature_data.mean[indices].astype(np.float64)
        self.mean_center = train_means.mean(axis=0).astype(np.float32)
        self.mean_scale = np.maximum(train_means.std(axis=0), 1e-4).astype(np.float32)
        self.lda_center = self.lda_scale = self.lda_projection = None
        if kind == "lda_train_only":
            if bundle_path is None:
                raise ValueError("LDA path missing")
            lda = np.load(bundle_path, allow_pickle=False)
            self.lda_center, self.lda_scale, self.lda_projection = lda["center"], lda["scale"], lda["projection"]
        if kind in ("supcon_linear", "speaker_ce_mlp"):
            if bundle_path is None:
                raise ValueError("head export missing")
            with np.load(bundle_path, allow_pickle=False) as archive:
                self.bundle = {key: archive[key] for key in archive.files}

    @property
    def name(self) -> str:
        return self.kind if self.kind not in ("supcon_linear", "speaker_ce_mlp") else self.kind

    def embed_many(self, pools: list[dict[str, Any]]) -> np.ndarray:
        if not pools:
            return np.empty((0, 128 if self.bundle is not None else self.data.dim), dtype=np.float32)
        if self.kind == "mean_cosine":
            return unit(np.stack([p["mean"] for p in pools]))
        if self.kind == "frame_l2_cosine":
            return unit(np.stack([p["frame_l2_mean"] for p in pools]))
        if self.kind == "mean_std_cosine":
            return unit(np.stack([concat_pool(p) for p in pools]))
        if self.kind == "train_zscore_cosine":
            x = (np.stack([p["mean"] for p in pools]) - self.mean_center) / self.mean_scale
            return unit(np.clip(x, -8.0, 8.0))
        if self.kind == "lda_train_only":
            x = np.clip((np.stack([p["mean"] for p in pools]) - self.lda_center) / self.lda_scale, -8.0, 8.0)
            return unit(x @ self.lda_projection)
        if self.kind in ("supcon_linear", "speaker_ce_mlp"):
            bundle = self.bundle
            x = np.stack([concat_pool(p) for p in pools]).astype(np.float32)
            x = np.clip((x - bundle["normalization_center"]) / bundle["normalization_scale"], -8.0, 8.0)
            tx = torch.from_numpy(x)
            with torch.no_grad():
                if self.kind == "supcon_linear":
                    projection = F.linear(tx, torch.from_numpy(bundle["weight_projection_weight"]),
                                          torch.from_numpy(bundle["weight_projection_bias"]))
                else:
                    norm_weight = torch.from_numpy(bundle["weight_norm_weight"])
                    norm_bias = torch.from_numpy(bundle["weight_norm_bias"])
                    value = F.layer_norm(tx, (tx.shape[-1],), norm_weight, norm_bias, 1e-5)
                    value = F.linear(value, torch.from_numpy(bundle["weight_linear1_weight"]),
                                     torch.from_numpy(bundle["weight_linear1_bias"]))
                    value = F.gelu(value)
                    projection = F.linear(value, torch.from_numpy(bundle["weight_linear2_weight"]),
                                          torch.from_numpy(bundle["weight_linear2_bias"]))
                return F.normalize(projection, dim=-1, eps=1e-12).numpy()
        raise ValueError(self.kind)

    def prototype(self, job_ids: list[str], budget_sec: float) -> tuple[np.ndarray | None, float]:
        if self.pool_mode == "single_pooled":
            pooled = self.data.pool(job_ids, budget_sec)
            if pooled["speech_duration_sec"] < budget_sec:
                return None, pooled["speech_duration_sec"]
            return self.embed_many([pooled])[0], pooled["speech_duration_sec"]
        utterances = self.data.utterance_stats(job_ids, budget_sec)
        measured = sum(item["speech_duration_sec"] for item in utterances)
        if measured < budget_sec or not utterances:
            return None, measured
        embeddings = self.embed_many(utterances)
        return unit(embeddings.mean(axis=0, keepdims=True))[0], measured


def query_groups(data: FeatureData, job_ids: list[str], requested_sec: float) -> list[list[str]]:
    if requested_sec == 1:
        return [[job_id] for job_id in job_ids
                if 0 < float(data.duration[data.index[job_id]]) < 2.0]
    groups: list[list[str]] = []
    current: list[str] = []
    duration = 0.0
    for job_id in job_ids:
        part_duration = float(data.duration[data.index[job_id]])
        if part_duration <= 0:
            continue
        current.append(job_id)
        duration += part_duration
        if duration >= requested_sec:
            groups.append(current)
            current, duration = [], 0.0
    return groups


def query_bucket(seconds: float) -> str:
    if seconds < 2.0:
        return "under_2s"
    if seconds < 5.0:
        return "2_to_5s"
    if seconds < 10.0:
        return "5_to_10s"
    return "over_10s"


def threshold_for_method(method: Method, fold: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    data = method.data
    people = fold["calibration_people"]
    prototypes: dict[str, np.ndarray] = {}
    prototype_duration: dict[str, float] = {}
    for speaker, person in people.items():
        vector, duration = method.prototype(person["enrollment_job_ids"], 10.0)
        if vector is not None:
            prototypes[speaker] = vector
            prototype_duration[speaker] = duration
    unknown_scores: list[float] = []
    genuine_scores: list[float] = []
    unknown_speakers = 0
    known_attempts = 0
    for speaker in fold["calibration_registry_order"]:
        if speaker not in prototypes:
            continue
        groups = query_groups(data, people[speaker]["query_job_ids"], 3.0)
        if groups:
            unknown_speakers += 1
        for group in groups:
            query_stats = data.pool(group)
            if query_stats["speech_duration_sec"] <= 0:
                continue
            query_vector = method.embed_many([query_stats])[0]
            genuine_scores.append(cosine(query_vector, prototypes[speaker]))
            others = [cosine(query_vector, prototype) for identity, prototype in prototypes.items() if identity != speaker]
            if others:
                unknown_scores.append(max(others))
                known_attempts += 1
    threshold = far_target_threshold(np.asarray(unknown_scores), 0.05)
    calibration_false_accepts = int(sum(score >= threshold for score in unknown_scores))
    known_accepts = int(sum(score >= threshold for score in genuine_scores))
    return threshold, {
        "policy": "single global threshold fitted on leave-one-identity-out maximum impostor scores; reused for every evaluation registry and duration",
        "target_calibration_false_accept_rate": 0.05,
        "calibration_unknown_speaker_clusters": unknown_speakers,
        "calibration_unknown_queries": len(unknown_scores),
        "calibration_unknown_false_accepts": calibration_false_accepts,
        "calibration_known_queries": len(genuine_scores),
        "calibration_known_accepts": known_accepts,
        "calibration_known_accept_rate": known_accepts / len(genuine_scores) if genuine_scores else None,
        "maximum_calibrated_registry_size": max(0, len(prototypes) - 1),
        "threshold": float(threshold),
        "uncertainty_note": "calibration speaker population is only 15 identities per fold; pair/query counts are correlated within speaker",
    }


def ci_rates(by_speaker: dict[str, list[int]], seed: int) -> tuple[float | None, dict[str, Any]]:
    rates = [sum(values) / len(values) for values in by_speaker.values() if values]
    if not rates:
        return None, {"lower_95": None, "upper_95": None, "speaker_clusters": 0, "method": "speaker-cluster bootstrap"}
    array = np.asarray(rates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.choice(array, size=(500, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), {"lower_95": float(np.quantile(sample, 0.025)),
                                 "upper_95": float(np.quantile(sample, 0.975)),
                                 "speaker_clusters": int(len(array)), "method": "speaker-cluster bootstrap over per-speaker attempt rates, 500 resamples"}


def evaluate_method(method: Method, fold: dict[str, Any], threshold: float,
                    threshold_details: dict[str, Any], seed: int,
                    enrollment_budgets: list[int] = ENROLLMENT_BUDGETS,
                    registry_sizes: list[int] = REGISTRY_SIZES,
                    query_budgets: list[int] = QUERY_BUDGETS,
                    include_duration_bucket_metrics: bool = False) -> list[dict[str, Any]]:
    data = method.data
    people = fold["evaluation_people"]
    eval_order = fold["evaluation_registry_order"]
    # Build every independent query group once per requested duration.
    queries: dict[tuple[str, int, int], dict[str, Any]] = {}
    for speaker, person in people.items():
        for query_budget in query_budgets:
            for ordinal, group in enumerate(query_groups(data, person["query_job_ids"], query_budget)):
                pooled = data.pool(group)
                if pooled["speech_duration_sec"] <= 0:
                    continue
                queries[(speaker, query_budget, ordinal)] = {
                    "pool": pooled, "vector": None, "seconds": pooled["speech_duration_sec"],
                    "utterance_count": len(group), "query_bucket": query_bucket(pooled["speech_duration_sec"]),
                }
    query_refs = list(queries)
    query_embeddings = method.embed_many([queries[key]["pool"] for key in query_refs])
    for key, vector in zip(query_refs, query_embeddings):
        queries[key]["vector"] = vector

    summaries = []
    for enrollment_budget in enrollment_budgets:
        prototypes: dict[str, np.ndarray] = {}
        actual_enrollment_seconds: dict[str, float] = {}
        for speaker, person in people.items():
            vector, actual = method.prototype(person["enrollment_job_ids"], float(enrollment_budget))
            if vector is not None:
                prototypes[speaker] = vector
                actual_enrollment_seconds[speaker] = actual
        for registry_size in registry_sizes:
            if registry_size > len(eval_order):
                continue
            registry = eval_order[:registry_size]
            registry_set = set(registry)
            active_prototypes = {speaker: prototypes[speaker] for speaker in registry if speaker in prototypes}
            profile_failures = [speaker for speaker in registry if speaker not in active_prototypes]
            for query_budget in query_budgets:
                rows: list[dict[str, Any]] = []
                genuine_scores: list[float] = []
                impostor_scores: list[float] = []
                genuine_accept_by_speaker: dict[str, list[int]] = {}
                impostor_accept_by_speaker: dict[str, list[int]] = {}
                true_scores: list[float] = []
                unknown_top_scores: list[float] = []
                query_speakers = set()
                for (speaker, requested, ordinal), query in queries.items():
                    if requested != query_budget:
                        continue
                    query_speakers.add(speaker)
                    known = speaker in registry_set
                    vector = query["vector"]
                    scores = {identity: cosine(vector, prototype) for identity, prototype in active_prototypes.items()}
                    if scores:
                        top_identity = max(scores, key=scores.get)
                        top_score = scores[top_identity]
                        accepted_identity = top_identity if top_score >= threshold else None
                        sorted_scores = sorted(scores.values(), reverse=True)
                        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else None
                    else:
                        top_identity, top_score, accepted_identity, margin = None, None, None, None
                    correct = bool(known and accepted_identity == speaker)
                    wrong = bool(known and accepted_identity is not None and accepted_identity != speaker)
                    known_rejected = bool(known and accepted_identity is None)
                    unknown_false_accept = bool(not known and accepted_identity is not None)
                    unknown_rejected = bool(not known and accepted_identity is None)
                    row = {"speaker_id": speaker, "known": known, "correct": correct, "wrong": wrong,
                           "known_rejected": known_rejected, "unknown_false_accept": unknown_false_accept,
                           "unknown_rejected": unknown_rejected, "no_decision": not bool(scores),
                           "top1_identification_correct": bool(known and top_identity == speaker),
                           "query_sec": query["seconds"], "top_score": top_score, "margin": margin,
                           "query_bucket": query_bucket(query["seconds"]),
                           "verification_genuine_score": scores.get(speaker) if known else None,
                           "verification_impostor_scores": [score for identity, score in scores.items()
                                                            if identity != speaker]}
                    rows.append(row)
                    if known and speaker in scores:
                        score = scores[speaker]
                        genuine_scores.append(score)
                        true_scores.append(score)
                        genuine_accept_by_speaker.setdefault(speaker, []).append(int(score >= threshold))
                    for identity, score in scores.items():
                        if identity != speaker:
                            impostor_scores.append(score)
                            impostor_accept_by_speaker.setdefault(speaker, []).append(int(score >= threshold))
                    if not known and top_score is not None:
                        unknown_top_scores.append(top_score)

                known_rows = [row for row in rows if row["known"]]
                unknown_rows = [row for row in rows if not row["known"]]
                known_accepted = sum(row["correct"] for row in known_rows)
                known_rank1_correct = sum(row["top1_identification_correct"] for row in known_rows)
                known_any_accept = sum(row["correct"] or row["wrong"] for row in known_rows)
                known_wrong = sum(row["wrong"] for row in known_rows)
                known_rejected = sum(row["known_rejected"] for row in known_rows)
                unknown_fa = sum(row["unknown_false_accept"] for row in unknown_rows)
                unknown_correct_reject = sum(row["unknown_rejected"] for row in unknown_rows)
                no_decisions = sum(row["no_decision"] for row in rows)
                known_by_speaker: dict[str, list[int]] = {}
                unknown_by_speaker: dict[str, list[int]] = {}
                open_set_by_speaker: dict[str, list[int]] = {}
                for row in known_rows:
                    known_by_speaker.setdefault(row["speaker_id"], []).append(int(row["correct"]))
                    open_set_by_speaker.setdefault(row["speaker_id"], []).append(int(row["correct"]))
                for row in unknown_rows:
                    unknown_by_speaker.setdefault(row["speaker_id"], []).append(int(row["unknown_false_accept"]))
                    open_set_by_speaker.setdefault(row["speaker_id"], []).append(int(row["unknown_rejected"]))
                known_macro, known_ci = ci_rates(known_by_speaker, seed + enrollment_budget * 19 + registry_size * 3 + query_budget)
                unknown_macro_fa, unknown_ci = ci_rates(unknown_by_speaker, seed + enrollment_budget * 23 + registry_size * 5 + query_budget)
                open_set_macro, open_set_ci = ci_rates(open_set_by_speaker, seed + enrollment_budget * 29 + registry_size * 7 + query_budget)
                verify_far_macro, verify_far_ci = ci_rates(impostor_accept_by_speaker, seed + enrollment_budget * 31 + registry_size * 11 + query_budget)
                verify_frr_macro, verify_frr_ci = ci_rates(
                    {speaker: [1 - value for value in values] for speaker, values in genuine_accept_by_speaker.items()},
                    seed + enrollment_budget * 37 + registry_size * 13 + query_budget)
                speaker_counts = {
                    "known": {speaker: {"attempts": len(values), "correct": sum(values)}
                              for speaker, values in known_by_speaker.items()},
                    "unknown": {speaker: {"attempts": len(values), "false_accepts": sum(values)}
                                for speaker, values in unknown_by_speaker.items()},
                    "open_set": {speaker: {"attempts": len(values), "correct": sum(values)}
                                 for speaker, values in open_set_by_speaker.items()},
                    "genuine": {speaker: {"trials": len(values), "accepts": sum(values)}
                                for speaker, values in genuine_accept_by_speaker.items()},
                    "impostor": {speaker: {"trials": len(values), "accepts": sum(values)}
                                 for speaker, values in impostor_accept_by_speaker.items()},
                }
                eer_cutoff, eer = eer_threshold(np.asarray(genuine_scores), np.asarray(impostor_scores))
                threshold_far = float(np.mean(np.asarray(impostor_scores) >= threshold)) if impostor_scores else None
                threshold_frr = float(np.mean(np.asarray(genuine_scores) < threshold)) if genuine_scores else None
                det_points = []
                if impostor_scores and genuine_scores:
                    imp_array = np.asarray(impostor_scores)
                    gen_array = np.asarray(genuine_scores)
                    for target_far in (0.001, 0.01, 0.05, 0.10):
                        det_threshold = far_target_threshold(imp_array, target_far)
                        det_points.append({"descriptive_target_far": target_far,
                            "threshold_from_evaluation_scores_not_used_for_decision": float(det_threshold),
                            "impostor_accepts": int(np.sum(imp_array >= det_threshold)),
                            "impostor_trials": int(len(imp_array)),
                            "far": float(np.mean(imp_array >= det_threshold)),
                            "genuine_rejects": int(np.sum(gen_array < det_threshold)),
                            "genuine_trials": int(len(gen_array)),
                            "frr": float(np.mean(gen_array < det_threshold))})
                # At a verification operating point, include all genuine and impostor comparisons.
                score_quantiles = {}
                for name, values in (("genuine", true_scores), ("impostor", impostor_scores),
                                     ("unknown_top1", unknown_top_scores)):
                    if values:
                        q = np.quantile(np.asarray(values), [0, .1, .25, .5, .75, .9, 1]).tolist()
                        score_quantiles[name] = {"min_p10_p25_p50_p75_p90_max": q, "count": len(values)}
                duration_bucket_metrics = {}
                if include_duration_bucket_metrics:
                    for bucket_index, bucket in enumerate(("under_2s", "2_to_5s", "5_to_10s", "over_10s")):
                        bucket_rows = [row for row in rows if row["query_bucket"] == bucket]
                        bucket_known = [row for row in bucket_rows if row["known"]]
                        bucket_unknown = [row for row in bucket_rows if not row["known"]]
                        known_counts: dict[str, list[int]] = {}
                        unknown_counts: dict[str, list[int]] = {}
                        open_counts: dict[str, list[int]] = {}
                        genuine_counts: dict[str, list[int]] = {}
                        impostor_counts: dict[str, list[int]] = {}
                        bucket_genuine_scores: list[float] = []
                        bucket_impostor_scores: list[float] = []
                        bucket_unknown_top: list[float] = []
                        for row in bucket_known:
                            known_counts.setdefault(row["speaker_id"], []).append(int(row["correct"]))
                            open_counts.setdefault(row["speaker_id"], []).append(int(row["correct"]))
                            genuine = row["verification_genuine_score"]
                            if genuine is not None:
                                bucket_genuine_scores.append(float(genuine))
                                genuine_counts.setdefault(row["speaker_id"], []).append(int(genuine >= threshold))
                            for score in row["verification_impostor_scores"]:
                                bucket_impostor_scores.append(float(score))
                                impostor_counts.setdefault(row["speaker_id"], []).append(int(score >= threshold))
                        for row in bucket_unknown:
                            unknown_counts.setdefault(row["speaker_id"], []).append(int(row["unknown_false_accept"]))
                            open_counts.setdefault(row["speaker_id"], []).append(int(row["unknown_rejected"]))
                            if row["top_score"] is not None:
                                bucket_unknown_top.append(float(row["top_score"]))
                            for score in row["verification_impostor_scores"]:
                                bucket_impostor_scores.append(float(score))
                                impostor_counts.setdefault(row["speaker_id"], []).append(int(score >= threshold))
                        known_macro_bucket, known_ci_bucket = ci_rates(known_counts, seed + 97000 + bucket_index)
                        unknown_macro_bucket, unknown_ci_bucket = ci_rates(unknown_counts, seed + 98000 + bucket_index)
                        open_macro_bucket, open_ci_bucket = ci_rates(open_counts, seed + 99000 + bucket_index)
                        far_macro_bucket, far_ci_bucket = ci_rates(impostor_counts, seed + 100000 + bucket_index)
                        frr_macro_bucket, frr_ci_bucket = ci_rates(
                            {speaker: [1 - accepted for accepted in values]
                             for speaker, values in genuine_counts.items()}, seed + 101000 + bucket_index)
                        bucket_eer_threshold, bucket_eer = eer_threshold(np.asarray(bucket_genuine_scores),
                                                                          np.asarray(bucket_impostor_scores))
                        distributions = {}
                        for name, values in (("genuine", bucket_genuine_scores),
                                             ("impostor", bucket_impostor_scores),
                                             ("unknown_top1", bucket_unknown_top)):
                            if values:
                                quantiles = np.quantile(np.asarray(values), [0, .1, .25, .5, .75, .9, 1]).tolist()
                                distributions[name] = {"count": len(values),
                                                       "min_p10_p25_p50_p75_p90_max": quantiles}
                        known_correct_bucket = sum(row["correct"] for row in bucket_known)
                        unknown_fa_bucket = sum(row["unknown_false_accept"] for row in bucket_unknown)
                        genuine_accepts_bucket = sum(score >= threshold for score in bucket_genuine_scores)
                        impostor_accepts_bucket = sum(score >= threshold for score in bucket_impostor_scores)
                        duration_bucket_metrics[bucket] = {
                            "query_attempts": len(bucket_rows),
                            "known_queries": len(bucket_known),
                            "known_speaker_clusters": len(known_counts),
                            "known_correct_accepts": int(known_correct_bucket),
                            "known_correct_accept_rate": known_correct_bucket / len(bucket_known) if bucket_known else None,
                            "known_correct_accept_speaker_macro": known_macro_bucket,
                            "known_correct_accept_ci95": known_ci_bucket,
                            "unknown_queries": len(bucket_unknown),
                            "unknown_speaker_clusters": len(unknown_counts),
                            "unknown_false_accepts": int(unknown_fa_bucket),
                            "unknown_false_accept_rate": unknown_fa_bucket / len(bucket_unknown) if bucket_unknown else None,
                            "unknown_false_accept_speaker_macro": unknown_macro_bucket,
                            "unknown_false_accept_ci95": unknown_ci_bucket,
                            "known_wrong_identity_accepts": sum(row["wrong"] for row in bucket_known),
                            "known_rejected_as_unknown": sum(row["known_rejected"] for row in bucket_known),
                            "unknown_correct_rejections": sum(row["unknown_rejected"] for row in bucket_unknown),
                            "no_decisions": sum(row["no_decision"] for row in bucket_rows),
                            "open_set_correct": known_correct_bucket + sum(row["unknown_rejected"] for row in bucket_unknown),
                            "open_set_accuracy": ((known_correct_bucket + sum(row["unknown_rejected"] for row in bucket_unknown)) / len(bucket_rows)
                                                  if bucket_rows else None),
                            "open_set_speaker_macro_accuracy": open_macro_bucket,
                            "open_set_accuracy_ci95": open_ci_bucket,
                            "verification_genuine_accepts": int(genuine_accepts_bucket),
                            "verification_genuine_trials": len(bucket_genuine_scores),
                            "verification_frr": (1.0 - genuine_accepts_bucket / len(bucket_genuine_scores)
                                                  if bucket_genuine_scores else None),
                            "verification_frr_speaker_macro": frr_macro_bucket,
                            "verification_frr_ci95": frr_ci_bucket,
                            "verification_impostor_accepts": int(impostor_accepts_bucket),
                            "verification_impostor_trials": len(bucket_impostor_scores),
                            "verification_far": (impostor_accepts_bucket / len(bucket_impostor_scores)
                                                  if bucket_impostor_scores else None),
                            "verification_far_speaker_macro": far_macro_bucket,
                            "verification_far_ci95": far_ci_bucket,
                            "verification_eer": bucket_eer,
                            "verification_eer_threshold": bucket_eer_threshold,
                            "verification_auc": rank_auc(np.asarray(bucket_genuine_scores),
                                                         np.asarray(bucket_impostor_scores)),
                            "verification_policy_threshold": threshold,
                            "score_distributions": distributions,
                        }
                summaries.append({
                    "fold_id": fold["fold_id"], "method": method.name, "representation": method.representation,
                    "model_key": method.model_key, "validation_selected_model": method.validation_selected,
                    "pooling": method.pool_mode, "enrollment_budget_sec": enrollment_budget,
                    "mean_actual_enrollment_sec": float(np.mean([actual_enrollment_seconds[speaker] for speaker in active_prototypes])) if active_prototypes else None,
                    "registry_size_requested": registry_size, "registry_size_usable": len(active_prototypes),
                    "registry_profile_failures": len(profile_failures), "registry_failure_ids": profile_failures,
                    "query_budget_sec": query_budget, "query_attempts": len(rows),
                    "query_speaker_clusters": len(query_speakers), "known_queries": len(known_rows),
                    "unknown_queries": len(unknown_rows), "known_correct_accepts": known_accepted,
                    "known_correct_accept_rate": known_accepted / len(known_rows) if known_rows else None,
                    "known_correct_accept_speaker_macro": known_macro, "known_correct_accept_ci95": known_ci,
                    "known_rank1_correct": known_rank1_correct,
                    "known_rank1_identification_rate": known_rank1_correct / len(known_rows) if known_rows else None,
                    "known_accepted_any_identity": known_any_accept, "known_wrong_identity_accepts": known_wrong,
                    "known_rejected_as_unknown": known_rejected,
                    "known_rejection_rate": known_rejected / len(known_rows) if known_rows else None,
                    "unknown_false_accepts": unknown_fa,
                    "unknown_false_accept_rate": unknown_fa / len(unknown_rows) if unknown_rows else None,
                    "unknown_false_accept_speaker_macro": unknown_macro_fa, "unknown_false_accept_ci95": unknown_ci,
                    "unknown_correct_rejections": unknown_correct_reject,
                    "unknown_correct_rejection_rate": unknown_correct_reject / len(unknown_rows) if unknown_rows else None,
                    "no_decisions": no_decisions, "no_decision_rate": no_decisions / len(rows) if rows else None,
                    "open_set_correct": known_accepted + unknown_correct_reject,
                    "open_set_accuracy": (known_accepted + unknown_correct_reject) / len(rows) if rows else None,
                    "open_set_speaker_macro_accuracy": open_set_macro, "open_set_accuracy_ci95": open_set_ci,
                    "verification_genuine_accepts": int(sum(score >= threshold for score in genuine_scores)),
                    "verification_genuine_trials": len(genuine_scores), "verification_frr": threshold_frr,
                    "verification_frr_speaker_macro": verify_frr_macro, "verification_frr_ci95": verify_frr_ci,
                    "verification_impostor_accepts": int(sum(score >= threshold for score in impostor_scores)),
                    "verification_impostor_trials": len(impostor_scores), "verification_far": threshold_far,
                    "verification_far_speaker_macro": verify_far_macro, "verification_far_ci95": verify_far_ci,
                    "verification_eer": eer, "verification_eer_threshold": eer_cutoff,
                    "descriptive_det_operating_points": det_points,
                    "verification_auc": rank_auc(np.asarray(genuine_scores), np.asarray(impostor_scores)),
                    "verification_policy_threshold": threshold,
                    "threshold_calibration": threshold_details,
                    "score_distributions": score_quantiles,
                    "actual_query_duration_bucket_metrics": duration_bucket_metrics,
                    "decision_rule": "accept top cosine identity only when top score >= frozen calibration threshold; otherwise UNKNOWN",
                    "actual_query_duration_buckets": sorted({row["query_sec"] for row in rows})[:0],
                    "_speaker_cluster_counts": speaker_counts,
                })
                summaries[-1]["actual_query_duration_buckets"] = {
                    bucket: sum(query_bucket(row["query_sec"]) == bucket for row in rows)
                    for bucket in ("under_2s", "2_to_5s", "5_to_10s", "over_10s")
                }
    return summaries


def evaluate_mic2(method: Method, fold: dict[str, Any], threshold: float,
                  enrollment_budget: float = 10.0, registry_size: int = 8) -> dict[str, Any]:
    data = method.data
    people = fold["evaluation_people"]
    registry = fold["evaluation_registry_order"][:registry_size]
    prototypes = {}
    for speaker in registry:
        vector, _ = method.prototype(people[speaker]["enrollment_job_ids"], enrollment_budget)
        if vector is not None:
            prototypes[speaker] = vector
    known_scores, unknown_scores = [], []
    known_correct = known_total = unknown_fa = unknown_total = 0
    for speaker, person in people.items():
        ids = person["query_mic2_job_ids"]
        for job_id in ids:
            pooled = data.pool([job_id])
            if pooled["speech_duration_sec"] <= 0:
                continue
            query = method.embed_many([pooled])[0]
            scores = {identity: cosine(query, proto) for identity, proto in prototypes.items()}
            if not scores:
                continue
            top = max(scores, key=scores.get)
            known = speaker in registry
            accepted = scores[top] >= threshold
            if known:
                known_total += 1
                known_correct += int(accepted and top == speaker)
                if speaker in scores:
                    known_scores.append(scores[speaker])
            else:
                unknown_total += 1
                unknown_fa += int(accepted)
                unknown_scores.append(max(scores.values()))
    return {"microphone": "mic2", "fold_id": fold["fold_id"], "registry_size": registry_size,
            "enrollment_budget_sec": enrollment_budget, "query_utterances_per_identity_max": 4,
            "known_correct_accepts": known_correct, "known_queries": known_total,
            "known_correct_accept_rate": known_correct / known_total if known_total else None,
            "unknown_false_accepts": unknown_fa, "unknown_queries": unknown_total,
            "unknown_false_accept_rate": unknown_fa / unknown_total if unknown_total else None,
            "known_speaker_clusters": len(set(registry)),
            "unknown_speaker_clusters": len(set(people) - set(registry)),
            "paired_mic1_mic2_utterances": "query utterance IDs only; no mic2 registration audio",
            "decision_threshold": threshold}


def cluster_ci_from_counts(counts: dict[str, dict[str, int]], numerator: str, denominator: str,
                           seed: int) -> tuple[float | None, dict[str, Any]]:
    rates = [value[numerator] / value[denominator] for value in counts.values() if value[denominator] > 0]
    if not rates:
        return None, {"lower_95": None, "upper_95": None, "speaker_clusters": 0,
                      "method": "speaker-cluster bootstrap over per-speaker rates, 500 resamples"}
    values = np.asarray(rates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(500, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), {"lower_95": float(np.quantile(sample, 0.025)),
                                  "upper_95": float(np.quantile(sample, 0.975)),
                                  "speaker_clusters": int(len(values)),
                                  "method": "speaker-cluster bootstrap over per-speaker rates, 500 resamples"}


def aggregate_folds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("method", "representation", "pooling", "enrollment_budget_sec", "registry_size_requested", "query_budget_sec")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group_index, (key, parts) in enumerate(sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0]))):
        merged: dict[str, dict[str, dict[str, int]]] = {name: {} for name in ("known", "unknown", "open_set", "genuine", "impostor")}
        for part in parts:
            for category, speakers in part["_speaker_cluster_counts"].items():
                for speaker, values in speakers.items():
                    target = merged[category].setdefault(speaker, {metric: 0 for metric in values})
                    for metric, value in values.items():
                        target[metric] += int(value)
        known = merged["known"]
        unknown = merged["unknown"]
        opened = merged["open_set"]
        genuine = merged["genuine"]
        impostor = merged["impostor"]
        known_correct = sum(value["correct"] for value in known.values())
        known_attempts = sum(value["attempts"] for value in known.values())
        known_rank1 = sum(part["known_rank1_correct"] for part in parts)
        unknown_fa = sum(value["false_accepts"] for value in unknown.values())
        unknown_attempts = sum(value["attempts"] for value in unknown.values())
        open_correct = sum(value["correct"] for value in opened.values())
        open_attempts = sum(value["attempts"] for value in opened.values())
        genuine_accepts = sum(value["accepts"] for value in genuine.values())
        genuine_trials = sum(value["trials"] for value in genuine.values())
        impostor_accepts = sum(value["accepts"] for value in impostor.values())
        impostor_trials = sum(value["trials"] for value in impostor.values())
        known_macro, known_ci = cluster_ci_from_counts(known, "correct", "attempts", 91000 + group_index)
        unknown_macro, unknown_ci = cluster_ci_from_counts(unknown, "false_accepts", "attempts", 92000 + group_index)
        open_macro, open_ci = cluster_ci_from_counts(opened, "correct", "attempts", 93000 + group_index)
        genuine_macro, genuine_ci = cluster_ci_from_counts(genuine, "accepts", "trials", 94000 + group_index)
        impostor_macro, impostor_ci = cluster_ci_from_counts(impostor, "accepts", "trials", 95000 + group_index)
        finite_eers = [part["verification_eer"] for part in parts if part["verification_eer"] is not None and np.isfinite(part["verification_eer"])]
        finite_aucs = [part["verification_auc"] for part in parts if part["verification_auc"] is not None and np.isfinite(part["verification_auc"])]
        output.append({
            **dict(zip(keys, key)), "fold_ids": [part["fold_id"] for part in parts],
            "fold_count": len(parts), "validation_selected_fold_count": sum(part["validation_selected_model"] for part in parts),
            "model_keys": sorted({part["model_key"] for part in parts if part["model_key"]}),
            "known_correct_accepts": known_correct, "known_queries": known_attempts,
            "known_correct_accept_rate": known_correct / known_attempts if known_attempts else None,
            "known_correct_accept_speaker_macro": known_macro, "known_correct_accept_ci95": known_ci,
            "known_rank1_correct": known_rank1,
            "known_rank1_identification_rate": known_rank1 / known_attempts if known_attempts else None,
            "unknown_false_accepts": unknown_fa, "unknown_queries": unknown_attempts,
            "unknown_false_accept_rate": unknown_fa / unknown_attempts if unknown_attempts else None,
            "unknown_false_accept_speaker_macro": unknown_macro, "unknown_false_accept_ci95": unknown_ci,
            "open_set_correct": open_correct, "open_set_queries": open_attempts,
            "open_set_accuracy": open_correct / open_attempts if open_attempts else None,
            "open_set_speaker_macro_accuracy": open_macro, "open_set_accuracy_ci95": open_ci,
            "verification_genuine_accepts": genuine_accepts, "verification_genuine_trials": genuine_trials,
            "verification_frr": 1 - genuine_accepts / genuine_trials if genuine_trials else None,
            "verification_frr_speaker_macro": 1 - genuine_macro if genuine_macro is not None else None,
            "verification_frr_ci95": {"lower_95": 1 - genuine_ci["upper_95"] if genuine_ci["upper_95"] is not None else None,
                                      "upper_95": 1 - genuine_ci["lower_95"] if genuine_ci["lower_95"] is not None else None,
                                      "speaker_clusters": genuine_ci["speaker_clusters"], "method": genuine_ci["method"]},
            "verification_impostor_accepts": impostor_accepts, "verification_impostor_trials": impostor_trials,
            "verification_far": impostor_accepts / impostor_trials if impostor_trials else None,
            "verification_far_speaker_macro": impostor_macro, "verification_far_ci95": impostor_ci,
            "verification_eer_macro_fold": float(np.mean(finite_eers)) if finite_eers else None,
            "verification_eer_fold_min_max": [float(min(finite_eers)), float(max(finite_eers))] if finite_eers else None,
            "verification_auc_macro_fold": float(np.mean(finite_aucs)) if finite_aucs else None,
            "registry_profile_failures": sum(part["registry_profile_failures"] for part in parts),
            "no_decisions": sum(part["no_decisions"] for part in parts),
            "actual_enrollment_sec_mean_across_folds": float(np.mean([part["mean_actual_enrollment_sec"] for part in parts if part["mean_actual_enrollment_sec"] is not None])) if any(part["mean_actual_enrollment_sec"] is not None for part in parts) else None,
            "uncertainty_note": "rates are speaker-cluster bootstrap intervals after combining repeated speaker IDs across folds; EER is a macro mean of fold-specific EERs",
        })
    return output


def open_set_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "fold_id", "fold_ids", "fold_count", "method", "representation", "pooling", "model_key",
        "validation_selected_model", "enrollment_budget_sec", "mean_actual_enrollment_sec",
        "registry_size_requested", "registry_size_usable", "registry_profile_failures",
        "query_budget_sec", "query_attempts", "query_speaker_clusters", "known_queries",
        "known_correct_accepts", "known_correct_accept_rate", "known_correct_accept_speaker_macro",
        "known_correct_accept_ci95", "known_rank1_correct", "known_rank1_identification_rate",
        "known_accepted_any_identity", "known_wrong_identity_accepts", "known_rejected_as_unknown",
        "known_rejection_rate", "unknown_queries", "unknown_false_accepts", "unknown_false_accept_rate",
        "unknown_false_accept_speaker_macro", "unknown_false_accept_ci95", "unknown_correct_rejections",
        "unknown_correct_rejection_rate", "no_decisions", "no_decision_rate", "open_set_correct",
        "open_set_queries", "open_set_accuracy", "open_set_speaker_macro_accuracy", "open_set_accuracy_ci95",
        "verification_policy_threshold", "threshold_calibration", "decision_rule",
        "actual_enrollment_sec_mean_across_folds", "uncertainty_note",
        "actual_query_duration_bucket_metrics",
    )
    return [{key: row[key] for key in fields if key in row} for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--pooled-features", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-results", type=Path, default=Path("tests/results/voiceid_task09b_training.json"))
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09b_verification.json"))
    parser.add_argument("--open-set-output", type=Path, default=Path("tests/results/voiceid_task09b_open_set.json"))
    parser.add_argument("--folds", default="all")
    parser.add_argument("--duration-buckets-only", action="store_true",
                        help="run selected heads plus two frozen baselines at 10s enrollment / 8 users and report actual-duration bins")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest, assets, pooled = load_data(args.manifest, args.pooled_features)
    training = json.loads(args.training_results.read_text(encoding="utf-8"))
    model_rows = {row["model_key"]: row for row in training["models"]}
    selected_folds = manifest["folds"] if args.folds == "all" else [fold for fold in manifest["folds"] if fold["fold_id"] in args.folds.split(",")]
    all_results = []
    calibration_results = []
    mic2_results = []
    methods_tested = []
    for fold in selected_folds:
        head_methods = []
        for row in training["models"]:
            if row["fold_id"] != fold["fold_id"]:
                continue
            rep = row["representation"]
            feature_data = FeatureData(assets, pooled, rep)
            method = Method(row["objective"], rep, feature_data, fold["train_utterance_job_ids"], args.run_dir / row["portable_export"])
            method.model_key = row["model_key"]
            method.validation_selected = row["model_key"] == training["selected_models_by_fold"][fold["fold_id"]]
            head_methods.append((method, row))
        selected_key = training["selected_models_by_fold"][fold["fold_id"]]
        selected_row = model_rows[selected_key]
        selected_rep = selected_row["representation"]
        selected_feature_data = FeatureData(assets, pooled, selected_rep)
        selected_mean_utt = Method(selected_row["objective"], selected_rep, selected_feature_data,
                                   fold["train_utterance_job_ids"], args.run_dir / selected_row["portable_export"],
                                   pool_mode="mean_normalized_utterance_embeddings")
        selected_mean_utt.model_key = selected_key
        selected_mean_utt.validation_selected = True
        methods: list[Method] = []
        for representation in REPRESENTATIONS:
            data = FeatureData(assets, pooled, representation)
            train_ids = fold["train_utterance_job_ids"]
            methods.extend([
                Method("mean_cosine", representation, data, train_ids),
                Method("frame_l2_cosine", representation, data, train_ids),
                Method("mean_std_cosine", representation, data, train_ids),
                Method("train_zscore_cosine", representation, data, train_ids),
                Method("lda_train_only", representation, data, train_ids,
                       args.run_dir / f"{fold['fold_id']}_{representation.replace('.', '_')}_lda.npz"),
            ])
        methods.extend(method for method, _ in head_methods)
        methods.append(selected_mean_utt)
        if args.duration_buckets_only:
            methods = [method for method in methods if method.validation_selected or
                       (method.kind == "mean_std_cosine" and method.representation == "pre_encoder") or
                       (method.kind == "mean_cosine" and method.representation == "encoder.layer.7")]
        for index, method in enumerate(methods):
            threshold, details = threshold_for_method(method, fold)
            calibration_results.append({"fold_id": fold["fold_id"], "method": method.name,
                                        "representation": method.representation, "pooling": method.pool_mode,
                                        "model_key": method.model_key,
                                        "validation_selected_model": method.validation_selected,
                                        **details})
            rows = evaluate_method(method, fold, threshold, details,
                                   20262000 + index * 101 + int(fold["speaker_seed"]),
                                   enrollment_budgets=[10] if args.duration_buckets_only else ENROLLMENT_BUDGETS,
                                   registry_sizes=[8] if args.duration_buckets_only else REGISTRY_SIZES,
                                   include_duration_bucket_metrics=args.duration_buckets_only)
            all_results.extend(rows)
            methods_tested.append({"fold_id": fold["fold_id"], "method": method.name,
                                   "representation": method.representation, "pooling": method.pool_mode,
                                   "model_key": method.model_key,
                                   "validation_selected_model": method.validation_selected,
                                   "threshold": threshold})
            if not args.duration_buckets_only and method is not selected_mean_utt and method.kind == selected_row["objective"]:
                for size in (2, 8, 16):
                    mic2_row = evaluate_mic2(method, fold, threshold, 10.0, size)
                    mic2_row.update({"method": method.name, "representation": method.representation,
                                     "model_key": method.model_key,
                                     "validation_selected_model": method.validation_selected})
                    mic2_results.append(mic2_row)
            print(json.dumps({"fold": fold["fold_id"], "method": method.name,
                              "representation": method.representation, "pooling": method.pool_mode,
                              "conditions": len(rows), "threshold": threshold}), flush=True)
    aggregate_results = aggregate_folds(all_results)
    for row in all_results:
        row.pop("_speaker_cluster_counts", None)
    output = {
        "schema": "nemo-speech.voiceid-task09b-verification.v1",
        "protocol": {
            "evaluation_speakers": "speaker-disjoint 30-person fold role; no evaluation identity used for head training, model selection, or threshold fitting",
            "threshold": "one frozen model-specific threshold per fold fitted only on calibration speakers by leave-one-identity-out max impostor cosine; same threshold at every test duration and registry size",
            "registry_sizes": REGISTRY_SIZES,
            "enrollment_speech_budgets_sec": ENROLLMENT_BUDGETS,
            "query_speech_budgets_sec": QUERY_BUDGETS,
            "query_duration_buckets": {"under_2s": "individual query utterances below 2.0s VAD speech", "2_to_5s": "actual grouped query VAD duration", "5_to_10s": "actual grouped query VAD duration", "over_10s": "actual grouped query VAD duration"},
            "actual_duration_note": "enrollment/query pools use whole independent mic1 utterances and stop after reaching the requested VAD speech budget; actual VAD seconds including final-utterance overshoot are recorded",
            "attempt_dependence": "query attempts within an identity and cross-fold re-use are clustered by speaker; confidence intervals resample speaker-level rates",
            "open_set_policy": "top-1 prototype cosine accepted only at or above threshold; otherwise UNKNOWN; no free native diarization channel is required",
            "enrollment_pooling": ["single pooled prototype", "mean of L2-normalized utterance embeddings for the validation-selected head"],
            "prototype_aggregation": "one cosine to each identity prototype; no max over per-user prototypes",
            "native_channel_comparison": "not rerun: existing channel baseline constructs AOSC state from an ordered conversation and is not equivalent to this fresh-stream single-utterance representation protocol; historical Task08/09A counts are not directly comparable",
            "duration_bucket_report_only": args.duration_buckets_only,
        },
        "method_count": len(methods_tested), "methods": methods_tested,
        "calibration": calibration_results,
        "verification_and_open_set_by_condition": all_results,
        "aggregate_across_speaker_folds": aggregate_results,
        "mic2_robustness": mic2_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    open_set_output = {
        "schema": "nemo-speech.voiceid-task09b-open-set.v1",
        "protocol": output["protocol"],
        "by_condition": open_set_projection(all_results),
        "aggregate_across_speaker_folds": open_set_projection(aggregate_results),
        "mic2_robustness": output["mic2_robustness"],
    }
    args.open_set_output.parent.mkdir(parents=True, exist_ok=True)
    args.open_set_output.write_text(json.dumps(open_set_output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "condition_rows": len(all_results),
                      "open_set_output": str(args.open_set_output),
                      "calibration_methods": len(calibration_results), "mic2_rows": len(mic2_results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"evaluate_voiceid_head: {exc}", file=sys.stderr)
        raise
