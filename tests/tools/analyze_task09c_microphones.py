#!/usr/bin/env python3
"""Evaluate frozen Task 09B/09C representations on matched VCTK microphones."""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_voiceid_head import Method, evaluate_mic2, threshold_for_method
from extract_voiceid_features import load_jobs, read_feature
from task09b_voiceid import FeatureData, cosine, eer_threshold, far_target_threshold, load_data, rank_auc
from task09c_voiceid import RawFeatureStore


def load_feature_union(base_path: Path, additional_path: Path) -> dict[str, np.ndarray]:
    with np.load(base_path, allow_pickle=False) as base_archive, np.load(additional_path, allow_pickle=False) as add_archive:
        base = {key: base_archive[key] for key in base_archive.files}
        additional = {key: add_archive[key] for key in add_archive.files}
    if set(base) != set(additional):
        raise ValueError("base and added pooled feature archives have different schemas")
    base_ids = base["job_ids"].tolist()
    add_ids = additional["job_ids"].tolist()
    if set(base_ids) & set(add_ids):
        raise ValueError("added feature archive repeats an existing job ID")
    return {key: np.concatenate((base[key], additional[key]), axis=0) for key in base}


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min_p10_p25_p50_p75_p90_max": None}
    q = np.quantile(np.asarray(values, dtype=np.float64), [0, .1, .25, .5, .75, .9, 1])
    return {"count": len(values), "min_p10_p25_p50_p75_p90_max": [float(value) for value in q]}


def cluster_interval(counts: dict[str, tuple[int, int]], seed: int) -> dict[str, Any]:
    rates = np.asarray([num / den for num, den in counts.values() if den], dtype=np.float64)
    if not len(rates):
        return {"lower_95": None, "upper_95": None, "speaker_clusters": 0,
                "method": "speaker-cluster bootstrap, 1000 resamples"}
    rng = np.random.default_rng(seed)
    sample = rng.choice(rates, size=(1000, len(rates)), replace=True).mean(axis=1)
    return {"lower_95": float(np.quantile(sample, .025)), "upper_95": float(np.quantile(sample, .975)),
            "speaker_clusters": int(len(rates)), "method": "speaker-cluster bootstrap over per-speaker rates, 1000 resamples"}


def selected_method(fold: dict[str, Any], manifest: dict[str, Any], assets: dict[str, dict[str, Any]],
                    pooled: dict[str, np.ndarray], training: dict[str, Any], run_dir: Path,
                    kind: str, representation: str = "pre_encoder") -> Method:
    feature_data = FeatureData(assets, pooled, representation)
    if kind in ("mean_cosine", "mean_std_cosine"):
        return Method(kind, representation, feature_data, fold["train_utterance_job_ids"])
    model_key = training["selected_models_by_fold"][fold["fold_id"]]
    row = next(model for model in training["models"] if model["model_key"] == model_key)
    return Method(row["objective"], row["representation"], feature_data,
                  fold["train_utterance_job_ids"], run_dir / row["portable_export"])


def enrollment_jobs(person: dict[str, Any], additions_person: dict[str, Any], microphone: str) -> list[str]:
    if microphone == "mic1":
        return list(person["enrollment_job_ids"])
    return list(additions_person["mic2_enrollment_job_ids"])


def build_prototypes(method: Method, people: dict[str, Any], add_people: dict[str, Any], microphone: str,
                     budget: float = 10.0) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, int]]:
    prototypes: dict[str, np.ndarray] = {}
    durations: dict[str, float] = {}
    attempts: dict[str, int] = {}
    for speaker, person in people.items():
        jobs = enrollment_jobs(person, add_people[speaker], microphone)
        vector, duration = method.prototype(jobs, budget)
        attempts[speaker] = len(jobs)
        if vector is not None:
            prototypes[speaker] = vector
            durations[speaker] = duration
    return prototypes, durations, attempts


def calibration_threshold(method: Method, fold: dict[str, Any], add_fold: dict[str, Any],
                          enrollment_mic: str, query_mic: str, budget: float = 10.0) -> tuple[float, dict[str, Any]]:
    people = fold["calibration_people"]
    additions = add_fold["calibration_people"]
    prototypes, durations, source_counts = build_prototypes(method, people, additions, enrollment_mic, budget)
    unknown_scores: list[float] = []
    genuine_scores: list[float] = []
    usable_queries = 0
    for speaker in fold["calibration_registry_order"]:
        if speaker not in people:
            continue
        query_pairs = additions[speaker]["paired_query_job_ids"]
        for pair in query_pairs:
            query_id = pair[f"{query_mic}_job_id"]
            pool = method.data.pool([query_id])
            if pool["speech_duration_sec"] <= 0:
                continue
            query = method.embed_many([pool])[0]
            scores = {identity: cosine(query, vector) for identity, vector in prototypes.items()}
            if not scores:
                continue
            usable_queries += 1
            others = [score for identity, score in scores.items() if identity != speaker]
            if speaker in scores:
                genuine_scores.append(scores[speaker])
            if others:
                unknown_scores.append(max(others))
            elif speaker not in scores:
                unknown_scores.append(max(scores.values()))
    threshold = far_target_threshold(np.asarray(unknown_scores), .05)
    false_accepts = int(sum(score >= threshold for score in unknown_scores))
    return threshold, {
        "threshold": float(threshold), "calibration_target_far": .05,
        "calibration_unknown_false_accepts": false_accepts,
        "calibration_unknown_max_score_trials": len(unknown_scores),
        "calibration_unknown_far": false_accepts / len(unknown_scores) if unknown_scores else None,
        "calibration_genuine_scores": len(genuine_scores),
        "calibration_known_accepts_at_threshold": int(sum(score >= threshold for score in genuine_scores)),
        "calibration_known_queries": len(genuine_scores), "calibration_usable_query_clips": usable_queries,
        "calibration_profile_successes": len(prototypes), "calibration_profile_attempts": len(people),
        "calibration_profile_failures": len(people) - len(prototypes),
        "calibration_actual_enrollment_sec": {speaker: durations.get(speaker) for speaker in people},
        "calibration_enrollment_source_count": source_counts,
        "policy": "leave-one-identity-out maximum impostor cosine over paired query clips; 5% target fitted on calibration speakers only",
        "registry_coverage_limit": max(0, len(prototypes) - 1),
    }


def evaluate_condition(method: Method, fold: dict[str, Any], add_fold: dict[str, Any],
                       enrollment_mic: str, query_mic: str, threshold: float, seed: int,
                       registry_size: int = 8, budget: float = 10.0) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    people = fold["evaluation_people"]
    additions = add_fold["evaluation_people"]
    registry = list(fold["evaluation_registry_order"][:registry_size])
    registry_set = set(registry)
    prototypes, durations, source_counts = build_prototypes(method, people, additions, enrollment_mic, budget)
    active = {speaker: prototypes[speaker] for speaker in registry if speaker in prototypes}
    profile_failures = [speaker for speaker in registry if speaker not in active]
    counts: dict[str, dict[str, int]] = {}
    genuine: list[float] = []
    impostor: list[float] = []
    unknown_top: list[float] = []
    detail: dict[tuple[str, str], dict[str, Any]] = {}
    totals = defaultdict(int)
    query_duration_values: list[float] = []

    for speaker, person in people.items():
        known = speaker in registry_set
        per = counts.setdefault(speaker, {"queries": 0, "known_correct": 0, "known_wrong": 0,
                                          "known_rejected": 0, "rank1_correct": 0,
                                          "unknown_false_accept": 0, "unknown_rejected": 0,
                                          "no_decision": 0})
        for pair in additions[speaker]["paired_query_job_ids"]:
            query_id = pair[f"{query_mic}_job_id"]
            pooled = method.data.pool([query_id])
            per["queries"] += 1
            totals["query_attempts"] += 1
            query_duration_values.append(float(pooled["speech_duration_sec"]))
            if pooled["speech_duration_sec"] <= 0 or not active:
                per["no_decision"] += 1
                totals["no_decisions"] += 1
                detail[(speaker, pair["utterance_id"])] = {"query_id": query_id, "known": known,
                                                            "rank1": False, "top_score": None,
                                                            "genuine_score": None, "max_impostor_score": None,
                                                            "accepted": None,
                                                            "no_decision": True}
                continue
            query = method.embed_many([pooled])[0]
            scores = {identity: cosine(query, vector) for identity, vector in active.items()}
            if not scores or not np.isfinite(list(scores.values())).all():
                per["no_decision"] += 1
                totals["no_decisions"] += 1
                detail[(speaker, pair["utterance_id"])] = {"query_id": query_id, "known": known,
                                                            "rank1": False, "top_score": None,
                                                            "genuine_score": None, "max_impostor_score": None,
                                                            "accepted": None,
                                                            "no_decision": True}
                continue
            top_id = max(scores, key=scores.get)
            top_score = float(scores[top_id])
            accepted = top_id if top_score >= threshold else None
            rank1 = known and top_id == speaker
            genuine_score = scores.get(speaker) if known else None
            max_impostor_score = max((score for identity, score in scores.items() if identity != speaker),
                                     default=None) if known else None
            if known:
                totals["known_queries"] += 1
                if rank1:
                    per["rank1_correct"] += 1
                    totals["known_rank1_correct"] += 1
                if genuine_score is not None:
                    genuine.append(float(genuine_score))
                impostor.extend(float(score) for identity, score in scores.items() if identity != speaker)
                if accepted == speaker:
                    per["known_correct"] += 1
                    totals["known_correct_accepts"] += 1
                elif accepted is None:
                    per["known_rejected"] += 1
                    totals["known_rejected_as_unknown"] += 1
                else:
                    per["known_wrong"] += 1
                    totals["known_wrong_identity_accepts"] += 1
            else:
                totals["unknown_queries"] += 1
                unknown_top.append(top_score)
                if accepted is None:
                    per["unknown_rejected"] += 1
                    totals["unknown_correct_rejections"] += 1
                else:
                    per["unknown_false_accept"] += 1
                    totals["unknown_false_accepts"] += 1
            detail[(speaker, pair["utterance_id"])] = {
                "query_id": query_id, "known": known, "rank1": bool(rank1), "top_id": top_id,
                "top_score": top_score, "genuine_score": float(genuine_score) if genuine_score is not None else None,
                "max_impostor_score": float(max_impostor_score) if max_impostor_score is not None else None,
                "accepted": accepted, "no_decision": False,
            }

    known_speakers = {speaker: values for speaker, values in counts.items() if speaker in registry_set}
    unknown_speakers = {speaker: values for speaker, values in counts.items() if speaker not in registry_set}
    known_attempts = sum(value["queries"] for value in known_speakers.values())
    unknown_attempts = sum(value["queries"] for value in unknown_speakers.values())
    known_correct = sum(value["known_correct"] for value in known_speakers.values())
    unknown_fa = sum(value["unknown_false_accept"] for value in unknown_speakers.values())
    verification_genuine_accepts = int(sum(score >= threshold for score in genuine))
    verification_impostor_accepts = int(sum(score >= threshold for score in impostor))
    verification_eer_threshold, verification_eer = eer_threshold(np.asarray(genuine), np.asarray(impostor))
    verification_auc = rank_auc(np.asarray(genuine), np.asarray(impostor))
    known_rate_by_speaker = {speaker: (value["known_correct"], value["queries"])
                             for speaker, value in known_speakers.items()}
    unknown_rate_by_speaker = {speaker: (value["unknown_false_accept"], value["queries"])
                               for speaker, value in unknown_speakers.items()}
    known_macro = float(np.mean([n / d for n, d in known_rate_by_speaker.values() if d])) if known_rate_by_speaker else None
    unknown_macro = float(np.mean([n / d for n, d in unknown_rate_by_speaker.values() if d])) if unknown_rate_by_speaker else None
    result = {
        "fold_id": fold["fold_id"], "enrollment_microphone": enrollment_mic, "query_microphone": query_mic,
        "registry_size_requested": registry_size, "registry_size_usable": len(active),
        "enrollment_budget_sec": budget, "threshold": float(threshold),
        "enrollment_profile_attempts": registry_size, "enrollment_profile_successes": len(active),
        "enrollment_profile_failures": len(profile_failures), "enrollment_failed_identity_ids": profile_failures,
        "enrollment_source_clip_counts": {speaker: source_counts.get(speaker, 0) for speaker in registry},
        "enrollment_actual_speech_sec": {speaker: durations.get(speaker) for speaker in registry},
        "query_attempts": int(totals["query_attempts"]), "known_queries": known_attempts,
        "known_correct_accepts": known_correct, "known_wrong_identity_accepts": int(totals["known_wrong_identity_accepts"]),
        "known_rejected_as_unknown": int(totals["known_rejected_as_unknown"]),
        "known_correct_accept_rate": known_correct / known_attempts if known_attempts else None,
        "known_correct_accept_speaker_macro": known_macro,
        "known_correct_accept_ci95": cluster_interval(known_rate_by_speaker, 10700 + registry_size + len(genuine)),
        "known_rank1_correct": int(totals["known_rank1_correct"]),
        "known_rank1_identification_rate": totals["known_rank1_correct"] / known_attempts if known_attempts else None,
        "unknown_queries": unknown_attempts, "unknown_false_accepts": unknown_fa,
        "unknown_false_accept_rate": unknown_fa / unknown_attempts if unknown_attempts else None,
        "unknown_false_accept_speaker_macro": unknown_macro,
        "unknown_false_accept_ci95": cluster_interval(unknown_rate_by_speaker, 20700 + registry_size + len(unknown_top)),
        "unknown_correct_rejections": int(totals["unknown_correct_rejections"]),
        "no_decisions": int(totals["no_decisions"]),
        "no_decision_rate": int(totals["no_decisions"]) / int(totals["query_attempts"])
        if totals["query_attempts"] else None,
        "open_set_correct": known_correct + int(totals["unknown_correct_rejections"]),
        "open_set_queries": known_attempts + unknown_attempts,
        "open_set_accuracy": (known_correct + int(totals["unknown_correct_rejections"])) /
        (known_attempts + unknown_attempts) if known_attempts + unknown_attempts else None,
        "known_genuine_scores": quantiles(genuine), "known_impostor_scores": quantiles(impostor),
        "unknown_top1_scores": quantiles(unknown_top),
        "verification_genuine_accepts": verification_genuine_accepts,
        "verification_genuine_trials": len(genuine),
        "verification_frr": 1.0 - verification_genuine_accepts / len(genuine) if genuine else None,
        "verification_impostor_accepts": verification_impostor_accepts,
        "verification_impostor_trials": len(impostor),
        "verification_far": verification_impostor_accepts / len(impostor) if impostor else None,
        "verification_eer_threshold_descriptive_only": verification_eer_threshold,
        "verification_eer_descriptive_only": verification_eer,
        "verification_auc_descriptive_only": verification_auc,
        "query_actual_speech_sec": quantiles(query_duration_values),
        "per_speaker": {
            speaker: {**values, "known": speaker in registry_set,
                      "correct_accept_rate": values["known_correct"] / values["queries"]
                      if speaker in registry_set and values["queries"] else None,
                      "unknown_false_accept_rate": values["unknown_false_accept"] / values["queries"]
                      if speaker not in registry_set and values["queries"] else None}
            for speaker, values in counts.items()
        },
    }
    return result, detail


def summarize_reference_reproduction(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    wanted = {("mean_cosine", "pre_encoder"), ("mean_std_cosine", "pre_encoder")}
    by_condition = [row for row in source["verification_and_open_set_by_condition"]
                    if row["enrollment_budget_sec"] == 10 and row["registry_size_requested"] == 8
                    and row["query_budget_sec"] == 3 and (row["method"], row["representation"]) in wanted]
    by_condition.extend(row for row in source["verification_and_open_set_by_condition"]
                        if row["enrollment_budget_sec"] == 10 and row["registry_size_requested"] == 8
                        and row["query_budget_sec"] == 3 and row.get("validation_selected_model"))
    aggregate = [row for row in source["aggregate_across_speaker_folds"]
                 if row["enrollment_budget_sec"] == 10 and row["registry_size_requested"] == 8
                 and row["query_budget_sec"] == 3 and (row["method"], row["representation"]) in wanted]
    selected_rows = [row for row in by_condition if row.get("validation_selected_model")
                     and row.get("pooling") == "single_pooled"]
    selected_known = sum(row["known_correct_accepts"] for row in selected_rows)
    selected_known_total = sum(row["known_queries"] for row in selected_rows)
    selected_unknown = sum(row["unknown_false_accepts"] for row in selected_rows)
    selected_unknown_total = sum(row["unknown_queries"] for row in selected_rows)
    expected = {
        "pre_encoder_mean": (112, 126, 17, 365),
        "pre_encoder_mean_std": (114, 126, 18, 365),
        "selected_learned_head": (103, 126, 13, 365),
    }
    actual = {
        "pre_encoder_mean": (next(row for row in aggregate if row["method"] == "mean_cosine")["known_correct_accepts"],
                             next(row for row in aggregate if row["method"] == "mean_cosine")["known_queries"],
                             next(row for row in aggregate if row["method"] == "mean_cosine")["unknown_false_accepts"],
                             next(row for row in aggregate if row["method"] == "mean_cosine")["unknown_queries"]),
        "pre_encoder_mean_std": (next(row for row in aggregate if row["method"] == "mean_std_cosine")["known_correct_accepts"],
                                 next(row for row in aggregate if row["method"] == "mean_std_cosine")["known_queries"],
                                 next(row for row in aggregate if row["method"] == "mean_std_cosine")["unknown_false_accepts"],
                                 next(row for row in aggregate if row["method"] == "mean_std_cosine")["unknown_queries"]),
        "selected_learned_head": (selected_known, selected_known_total, selected_unknown, selected_unknown_total),
    }
    return {"expected_reference": {key: list(value) for key, value in expected.items()},
            "reproduced": {key: list(value) for key, value in actual.items()},
            "exact_count_match": actual == expected,
            "fold_rows": [{key: row.get(key) for key in ("fold_id", "method", "representation", "model_key",
                                                            "validation_selected_model", "verification_policy_threshold",
                                                            "known_correct_accepts", "known_queries", "known_wrong_identity_accepts",
                                                            "known_rejected_as_unknown", "unknown_false_accepts", "unknown_queries",
                                                            "unknown_correct_rejections")}
                          for row in by_condition],
            "method_thresholds_and_decisions": "reproduced by the original Task 09B evaluator on its cached NVFTR features and fixed manifest",
            "reproduction_source": str(path)}


def read_jobs(paths: list[Path]) -> dict[str, dict[str, str]]:
    jobs: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in load_jobs(path):
            jobs[row["job_id"]] = row
    return jobs


def audio_file_for(job_id: str, jobs: dict[str, dict[str, str]], extra_assets: dict[str, dict[str, Any]],
                   extra_cache: Path) -> Path:
    if job_id in jobs:
        return Path(jobs[job_id]["wav_path"])
    return extra_cache / extra_assets[job_id]["audio_file"]


def vad_file_for(job_id: str, jobs: dict[str, dict[str, str]], extra_assets: dict[str, dict[str, Any]],
                 extra_cache: Path) -> Path:
    if job_id in jobs:
        return Path(jobs[job_id]["vad_path"])
    return extra_cache / extra_assets[job_id]["vad_mask_file"]


def pcm_stats(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if channels != 1 or width != 2 or rate != 16000:
        raise ValueError(f"unexpected PCM WAV format in {path}")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    frequencies = np.fft.rfftfreq(max(1, len(x)), d=1 / rate)
    power = np.abs(np.fft.rfft(x)) ** 2 if len(x) else np.zeros(1)
    bands = ((0, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 8000))
    band_energy = [float(power[(frequencies >= lo) & (frequencies < hi)].sum()) for lo, hi in bands]
    total = sum(band_energy)
    return {"duration_sec": frames / rate, "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
            "peak_dbfs": 20 * math.log10(max(peak, 1e-12)),
            "quiet_sample_fraction_below_minus_60_dbfs": float(np.mean(np.abs(x) < 0.001)) if len(x) else None,
            "band_energy_fraction": [value / total for value in band_energy] if total else [None] * len(bands)}


def summarize_values(values: list[float]) -> dict[str, Any]:
    values = [float(value) for value in values if np.isfinite(value)]
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "quantiles": None}
    arr = np.asarray(values)
    return {"count": len(values), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "std": float(arr.std()), "quantiles": quantiles(values)["min_p10_p25_p50_p75_p90_max"]}


def microphone_diagnostics(folds: list[dict[str, Any]], additions: dict[str, Any],
                           assets: dict[str, dict[str, Any]], jobs: dict[str, dict[str, str]],
                           feature_dirs: list[Path]) -> dict[str, Any]:
    store = RawFeatureStore(feature_dirs, "pre_encoder", max_cached_records=64)
    extra_assets = {item["job_id"]: item for item in additions["new_assets"]}
    cache = Path(additions["cache_dir"])
    paired: dict[tuple[str, str], tuple[str, str]] = {}
    for fold in folds:
        add_fold = next(row for row in additions["folds"] if row["fold_id"] == fold["fold_id"])
        for role in ("calibration_people", "evaluation_people"):
            for speaker, person in add_fold[role].items():
                for pair in person["paired_query_job_ids"]:
                    paired[(speaker, pair["utterance_id"])] = (pair["mic1_job_id"], pair["mic2_job_id"])
    feature_cosines: list[float] = []
    frame_cosines: list[float] = []
    per_utterance_frame_cosines: list[float] = []
    mean_differences: list[np.ndarray] = []
    variance_differences: list[np.ndarray] = []
    mic1_frames: list[np.ndarray] = []
    mic2_frames: list[np.ndarray] = []
    mask_agreement: list[float] = []
    mask_duration_diff: list[float] = []
    pcm: dict[str, list[dict[str, Any]]] = {"mic1": [], "mic2": []}
    pcm_gain_delta: list[float] = []
    spectral_delta: list[np.ndarray] = []
    for mic1_id, mic2_id in paired.values():
        frames1, _, _ = store.frames(mic1_id)
        frames2, _, _ = store.frames(mic2_id)
        if len(frames1) and len(frames2):
            mean1 = frames1.mean(axis=0, dtype=np.float64)
            mean2 = frames2.mean(axis=0, dtype=np.float64)
            mean_differences.append(mean2 - mean1)
            variance_differences.append(frames2.var(axis=0) - frames1.var(axis=0))
            feature_cosines.append(cosine(mean1, mean2))
            mic1_frames.append(frames1)
            mic2_frames.append(frames2)
        def feature_path(job_id: str) -> Path:
            name = f"{job_id}.pre_encoder.vfeat"
            path = next((root / name for root in feature_dirs if (root / name).is_file()), None)
            if path is None:
                raise FileNotFoundError(f"no raw feature record {name}")
            return path
        _, ids1, bits1, raw1, _ = read_feature(feature_path(mic1_id))
        _, ids2, bits2, raw2, _ = read_feature(feature_path(mic2_id))
        common, index1, index2 = np.intersect1d(ids1, ids2, assume_unique=True, return_indices=True)
        speech1 = np.asarray([int(value).bit_count() >= 4 for value in bits1], dtype=bool)
        speech2 = np.asarray([int(value).bit_count() >= 4 for value in bits2], dtype=bool)
        active = speech1[index1] & speech2[index2]
        if active.any():
            left = raw1[index1[active]].astype(np.float64)
            right = raw2[index2[active]].astype(np.float64)
            denominators = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
            valid = denominators > 0
            pair_cosines = np.sum(left[valid] * right[valid], axis=1) / denominators[valid]
            frame_cosines.extend(float(value) for value in pair_cosines)
            per_utterance_frame_cosines.append(float(pair_cosines.mean()))
        mask1 = np.frombuffer(vad_file_for(mic1_id, jobs, extra_assets, cache).read_bytes(), dtype=np.uint8)
        mask2 = np.frombuffer(vad_file_for(mic2_id, jobs, extra_assets, cache).read_bytes(), dtype=np.uint8)
        common = min(len(mask1), len(mask2))
        width = max(len(mask1), len(mask2))
        mismatch = int(np.count_nonzero(mask1[:common] != mask2[:common])) + abs(len(mask1) - len(mask2))
        mask_agreement.append(1.0 - mismatch / width if width else 1.0)
        mask_duration_diff.append((int(mask2.sum()) - int(mask1.sum())) * 0.01)
        stat1 = pcm_stats(audio_file_for(mic1_id, jobs, extra_assets, cache))
        stat2 = pcm_stats(audio_file_for(mic2_id, jobs, extra_assets, cache))
        pcm["mic1"].append(stat1)
        pcm["mic2"].append(stat2)
        pcm_gain_delta.append(stat2["rms_dbfs"] - stat1["rms_dbfs"])
        spectral_delta.append(np.asarray(stat2["band_energy_fraction"]) - np.asarray(stat1["band_energy_fraction"]))
    if mean_differences:
        mean_shift = np.mean(np.stack(mean_differences), axis=0)
        variance_shift = np.mean(np.stack(variance_differences), axis=0)
        pooled1 = np.concatenate(mic1_frames, axis=0)
        pooled2 = np.concatenate(mic2_frames, axis=0)
        variance1 = np.maximum(pooled1.var(axis=0), 1e-12)
        variance2 = np.maximum(pooled2.var(axis=0), 1e-12)
    else:
        mean_shift = variance_shift = np.zeros(512, dtype=np.float64)
        variance1 = variance2 = np.ones(512, dtype=np.float64)
    def signal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result = {}
        for key in ("duration_sec", "rms_dbfs", "peak_dbfs", "quiet_sample_fraction_below_minus_60_dbfs"):
            result[key] = summarize_values([row[key] for row in rows if row[key] is not None])
        result["mean_band_energy_fraction"] = np.mean(np.asarray([row["band_energy_fraction"] for row in rows]), axis=0).tolist() if rows else None
        result["spectral_bands_hz"] = [[0, 300], [300, 1000], [1000, 3000], [3000, 6000], [6000, 8000]]
        return result
    return {
        "matched_speaker_utterance_pairs": len(paired), "feature_valid_pairs": len(feature_cosines),
        "paired_feature_mean_cosine": summarize_values(feature_cosines),
        "paired_feature_frame_cosine_on_aligned_vad_frames": summarize_values(frame_cosines),
        "paired_feature_frame_cosine_mean_per_utterance": summarize_values(per_utterance_frame_cosines),
        "per_dimension_feature_mean_shift_mic2_minus_mic1": mean_shift.tolist(),
        "per_dimension_feature_variance_delta_mic2_minus_mic1": variance_shift.tolist(),
        "feature_mean_shift_summary": {"mean_absolute_shift": float(np.mean(np.abs(mean_shift))),
                                       "rms_shift": float(np.sqrt(np.mean(mean_shift ** 2))),
                                       "largest_absolute_dimensions": sorted(
                                           [{"dimension": int(i), "shift": float(mean_shift[i])}
                                            for i in np.argsort(np.abs(mean_shift))[-10:]],
                                           key=lambda row: abs(row["shift"]), reverse=True)},
        "pooled_feature_variance_ratio_mic2_over_mic1": (variance2 / variance1).tolist(),
        "speech_mask_agreement": summarize_values(mask_agreement),
        "speech_mask_active_duration_delta_sec_mic2_minus_mic1": summarize_values(mask_duration_diff),
        "audio_signal_mic1": signal_summary(pcm["mic1"]), "audio_signal_mic2": signal_summary(pcm["mic2"]),
        "paired_audio_rms_db_delta_mic2_minus_mic1": summarize_values(pcm_gain_delta),
        "paired_spectral_band_energy_fraction_delta_mic2_minus_mic1": np.mean(np.stack(spectral_delta), axis=0).tolist()
        if spectral_delta else None,
        "spectral_band_hz": [[0, 300], [300, 1000], [1000, 3000], [3000, 6000], [6000, 8000]],
        "dependence_note": "mic1 and mic2 files share one source utterance; each paired clip and speaker is clustered, not an independent session",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--microphone-additions", type=Path, default=Path("tests/manifests/task09c_vctk_microphone_additions.json"))
    parser.add_argument("--pooled-features", type=Path, required=True)
    parser.add_argument("--additional-pooled-features", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--additional-feature-dir", type=Path, required=True)
    parser.add_argument("--base-jobs-tsv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-results", type=Path, default=Path("tests/results/voiceid_task09b_training.json"))
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09c_microphone.json"))
    args = parser.parse_args()

    torch.set_num_threads(4)
    manifest, base_assets, _ = load_data(args.manifest, args.pooled_features)
    additions = json.loads(args.microphone_additions.read_text(encoding="utf-8"))
    pooled = load_feature_union(args.pooled_features, args.additional_pooled_features)
    new_assets = {item["job_id"]: item for item in additions["new_assets"]}
    assets = {**base_assets, **new_assets}
    training = json.loads(args.training_results.read_text(encoding="utf-8"))
    add_folds = {row["fold_id"]: row for row in additions["folds"]}
    direct_reference: list[dict[str, Any]] = []
    matrix_results: list[dict[str, Any]] = []
    score_shift_results: list[dict[str, Any]] = []
    for fold in manifest["folds"]:
        add_fold = add_folds[fold["fold_id"]]
        method_specs = [
            ("mean_cosine", "pre_encoder", None),
            ("mean_std_cosine", "pre_encoder", None),
            ("train_zscore_cosine", "pre_encoder", None),
        ]
        selected_key = training["selected_models_by_fold"][fold["fold_id"]]
        selected_row = next(row for row in training["models"] if row["model_key"] == selected_key)
        method_specs.append((selected_row["objective"], selected_row["representation"], selected_row))
        methods: list[tuple[str, Method]] = []
        for kind, representation, row in method_specs:
            data = FeatureData(assets, pooled, representation)
            model = Method(kind, representation, data, fold["train_utterance_job_ids"],
                           args.run_dir / row["portable_export"] if row else None)
            if row:
                model.model_key = row["model_key"]
                model.validation_selected = True
            methods.append((kind, model))
        standard_thresholds: dict[str, tuple[float, dict[str, Any]]] = {}
        for label, method in methods:
            threshold, detail = threshold_for_method(method, fold)
            standard_thresholds[label] = (threshold, detail)
            direct = evaluate_mic2(method, fold, threshold, 10.0, 8)
            direct.update({"method": label, "representation": method.representation,
                           "model_key": method.model_key,
                           "validation_selected_model": method.validation_selected,
                           "threshold_policy": "Task 09B frozen mic1 calibration threshold",
                           "threshold_calibration": detail})
            direct_reference.append(direct)

        for label, method in methods:
            all_details: dict[tuple[str, str, str, str], dict[str, Any]] = {}
            condition_rows = {}
            for enrollment_mic, query_mic in (("mic1", "mic1"), ("mic1", "mic2"),
                                              ("mic2", "mic1"), ("mic2", "mic2")):
                calibrated_threshold, calibration = calibration_threshold(
                    method, fold, add_fold, enrollment_mic, query_mic, 10.0)
                result, details = evaluate_condition(method, fold, add_fold, enrollment_mic, query_mic,
                                                     calibrated_threshold, 28100 + len(matrix_results) * 7, 8, 10.0)
                result["method"] = label
                result["representation"] = method.representation
                result["model_key"] = method.model_key
                result["threshold_calibration"] = calibration
                fixed_threshold = standard_thresholds[label][0]
                result["task09b_mic1_reference_threshold"] = fixed_threshold
                if enrollment_mic == "mic1":
                    fixed_result, _ = evaluate_condition(method, fold, add_fold, enrollment_mic, query_mic,
                                                         fixed_threshold, 39100 + len(matrix_results) * 7, 8, 10.0)
                    result["task09b_reference_threshold_decisions"] = {
                        key: fixed_result[key] for key in ("threshold", "known_queries", "known_correct_accepts",
                                                           "known_wrong_identity_accepts", "known_rejected_as_unknown",
                                                           "unknown_queries", "unknown_false_accepts",
                                                           "unknown_correct_rejections", "no_decisions")
                    }
                condition_rows[(enrollment_mic, query_mic)] = result
                matrix_results.append(result)
                for (speaker, utterance), detail_row in details.items():
                    all_details[(enrollment_mic, query_mic, speaker, utterance)] = detail_row
            for enrollment_mic in ("mic1", "mic2"):
                left = condition_rows[(enrollment_mic, "mic1")]
                right = condition_rows[(enrollment_mic, "mic2")]
                genuine_diffs = []
                impostor_diffs = []
                top_diffs = []
                rank_changes = 0
                decision_changes = 0
                pair_keys = {(key[2], key[3]) for key in all_details
                             if key[0] == enrollment_mic and key[1] == "mic1"}
                for speaker, utterance in sorted(pair_keys):
                    d1 = all_details.get((enrollment_mic, "mic1", speaker, utterance))
                    d2 = all_details.get((enrollment_mic, "mic2", speaker, utterance))
                    if d1 is None or d2 is None:
                        continue
                    if d1["genuine_score"] is not None and d2["genuine_score"] is not None:
                        genuine_diffs.append(d2["genuine_score"] - d1["genuine_score"])
                    if d1.get("max_impostor_score") is not None and d2.get("max_impostor_score") is not None:
                        impostor_diffs.append(d2["max_impostor_score"] - d1["max_impostor_score"])
                    if d1["top_score"] is not None and d2["top_score"] is not None:
                        top_diffs.append(d2["top_score"] - d1["top_score"])
                    rank_changes += int(d1["rank1"] != d2["rank1"])
                    decision_changes += int(d1["accepted"] != d2["accepted"])
                score_shift_results.append({"fold_id": fold["fold_id"], "method": label,
                                            "enrollment_microphone": enrollment_mic,
                                            "mic1_query_threshold": left["threshold"],
                                            "mic2_query_threshold": right["threshold"],
                                            "paired_query_utterances": len(pair_keys),
                                            "genuine_score_shift_mic2_minus_mic1": summarize_values(genuine_diffs),
                                            "maximum_impostor_score_shift_mic2_minus_mic1": summarize_values(impostor_diffs),
                                            "top1_score_shift_mic2_minus_mic1": summarize_values(top_diffs),
                                            "rank1_identity_changed_queries": rank_changes,
                                            "thresholded_decision_changed_queries": decision_changes,
                                            "score_shift_note": "same enrollment prototype and source utterance ID; mic1/mic2 query files are paired"})
    jobs = read_jobs([args.base_jobs_tsv, Path(additions["jobs_tsv"])])
    diagnostics = microphone_diagnostics(manifest["folds"], additions, assets, jobs,
                                         [args.feature_dir, args.additional_feature_dir])
    reproduction = summarize_reference_reproduction(args.reproduction)
    if not reproduction["exact_count_match"]:
        raise RuntimeError(f"Task 09B count reproduction mismatch: {reproduction['reproduced']}")
    output = {
        "schema": "nemo-speech.voiceid-task09c-microphone.v1",
        "evaluation_status": "VCTK follow-up evidence; evaluation speakers were used in previous project tasks and this is not independent external validation",
        "reference_reproduction": reproduction,
        "protocol": {
            "speaker_roles": "original Task 09B folds; 50 train/statistics, 15 selection, 15 calibration, 30 evaluation speakers per fold",
            "mic_matrix_query_policy": "up to four paired query utterances per identity; each mic1/mic2 rendering uses the same source utterance IDs; enrollment utterance IDs remain disjoint",
            "primary_registry": 8, "enrollment_speech_budget_sec": 10,
            "threshold_policy": "condition-specific calibration threshold and separately reported fixed Task 09B mic1 threshold for mic1 enrollment transfer",
            "primary_reference": "original Task 09B 10s enrollment, 3s grouped mic1 queries, 8-person registry, one frozen calibration threshold",
            "paired_trials_are_clustered": True,
            "method_selection": "only predeclared Task 09B frozen methods and the Task 09B selected head; no evaluation-driven selection",
        },
        "task09b_threshold_mic2_transfer": direct_reference,
        "matched_microphone_matrix": matrix_results,
        "paired_score_shifts": score_shift_results,
        "microphone_input_and_feature_diagnostics": diagnostics,
        "extractor": {
            "base_feature_backend": "Vulkan0 per Task 09B extraction summary",
            "added_feature_backend": json.loads((args.additional_feature_dir / "extractor_summary.json").read_text())
            .get("actual_backend"),
            "added_device": json.loads((args.additional_feature_dir / "extractor_summary.json").read_text())
            .get("device_name"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reference_rows": len(direct_reference),
                      "matrix_rows": len(matrix_results), "feature_pairs": diagnostics["feature_valid_pairs"],
                      "reproduction_exact": reproduction["exact_count_match"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"analyze_task09c_microphones: {exc}", file=sys.stderr)
        raise
