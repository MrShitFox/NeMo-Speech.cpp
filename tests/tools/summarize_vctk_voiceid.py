#!/usr/bin/env python3
"""Summarize Task 08 trial JSONL without treating frames as independent trials."""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import statistics
import struct
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def fnv64(value: str) -> str:
    result = 1469598103934665603
    for byte in value.encode():
        result = ((result ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def category_counts(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    selected = [row for row in rows if row["known"] is (kind == "known") and row["evaluated"]]
    counts = collections.Counter(row["category"] for row in selected)
    if kind == "known":
        names = ("known_correct", "known_wrong_enrolled_identity", "known_rejected_as_unknown", "no_decision")
    else:
        names = ("unknown_correctly_rejected", "unknown_false_accept", "no_decision")
    return {
        "evaluated_attempts": len(selected),
        "not_evaluated_attempts": sum(1 for row in rows if row["known"] is (kind == "known") and not row["evaluated"]),
        "counts": {name: counts.get(name, 0) for name in names},
        "rates_over_evaluated_attempts": {
            name: counts.get(name, 0) / len(selected) if selected else None for name in names
        },
    }


def query_rows(
    manifest: dict[str, Any], results: list[dict[str, Any]], expected_fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trials = {trial["trial_id"]: trial for trial in manifest["trials"]}
    result_by_trial = {}
    for result in results:
        trial_id = result.get("trial_id")
        if trial_id not in trials:
            raise ValueError(f"result references trial absent from manifest: {trial_id}")
        if result.get("manifest_fingerprint_fnv1a64") != expected_fingerprint:
            raise ValueError(f"result belongs to a different manifest: {trial_id}")
        if trial_id in result_by_trial:
            raise ValueError(f"duplicate result for trial: {trial_id}")
        result_by_trial[trial_id] = result
    fingerprints = {row.get("manifest_fingerprint_fnv1a64") for row in results}
    if len(fingerprints) > 1:
        raise ValueError("results contain multiple manifest fingerprints")
    output = []
    for trial in manifest["trials"]:
        result = result_by_trial.get(trial["trial_id"])
        if result is None:
            continue
        returned = {item.get("query_id"): item for item in result.get("queries", [])}
        for planned in trial.get("queries", []):
            observed = returned.get(planned["query_id"], {})
            category = observed.get("baseline_category", "not_evaluated_missing_result")
            output.append({
                "trial_id": trial["trial_id"],
                "cohort_id": trial["cohort_id"],
                "population": trial["population"],
                "group_size": trial["group_size"],
                "budget_sec": trial["enrollment_budget_speech_sec_per_person"],
                "order_variant": trial["enrollment_order_variant"],
                "query_id": planned["query_id"],
                "speaker_id": planned["speaker_id"],
                "known": bool(planned["known"]),
                "mic": planned["mic"],
                "duration_bucket": planned["duration_bucket"],
                "speech_duration_sec": observed.get("actual_vad_speech_duration_sec", planned["speech_duration_sec"]),
                "recording_duration_sec": observed.get("recording_duration_sec", planned["duration_sec"]),
                "category": category,
                "evaluated": not category.startswith("not_evaluated"),
                "score": candidate_score(observed),
                "diagnostic_scores": {
                    "registered_channel_mean": candidate_score(observed),
                    "top1_dominance": observed.get("top1_dominance"),
                    "mean_top1_top2_margin": observed.get("mean_top1_top2_margin"),
                    "vad_speech_duration_sec": observed.get(
                        "actual_vad_speech_duration_sec", planned["speech_duration_sec"]
                    ),
                },
                "result": observed,
            })
    completeness = {
        "manifest_trials": len(trials),
        "completed_trials": len(result_by_trial),
        "missing_trial_ids": sorted(set(trials) - set(result_by_trial)),
        "complete": len(trials) == len(result_by_trial),
    }
    return output, completeness


def candidate_score(result: dict[str, Any]) -> float | None:
    predicted = result.get("predicted_identity")
    means = result.get("registered_channel_means", {})
    if predicted in means:
        try:
            return float(means[predicted])
        except (TypeError, ValueError):
            return None
    return None


def retained_rows_by_candidate_identity(result: dict[str, Any]) -> dict[str, Any]:
    """Join each enrolled speaker's established candidate channel to cache rows.

    Keep collisions visible: multiple identities may point to the same native
    channel even when the external-ID mapping is therefore invalid.
    """
    stats = result.get("speaker_channel_establishment", {})
    owners: dict[int, list[str]] = collections.defaultdict(list)
    for speaker, entry in stats.items():
        channel = entry.get("candidate_channel")
        if channel is not None:
            owners[int(channel)].append(speaker)
    cache = result.get("cache_row_summary") or {}
    positive = cache.get("positive_rows_by_channel_gt_0_5", [])
    argmax = cache.get("argmax_rows_by_channel_gt_0_5", [])
    output = {}
    for speaker in result.get("enrolled_speaker_ids", []):
        entry = stats.get(speaker, {})
        candidate = entry.get("candidate_channel")
        channel = int(candidate) if candidate is not None else None
        shared = sorted(owners[channel]) if channel is not None else []
        output[speaker] = {
            "candidate_channel": channel,
            "channel_established": bool(entry.get("channel_established")),
            "candidate_channel_owner_count": len(shared),
            "other_enrolled_identities_sharing_candidate_channel": [
                owner for owner in shared if owner != speaker
            ],
            "positive_rows_gt_0_5_including_overlaps": (
                positive[channel] if channel is not None and channel < len(positive) else None
            ),
            "argmax_rows_gt_0_5": (
                argmax[channel] if channel is not None and channel < len(argmax) else None
            ),
        }
    return output


def summarize_profile_conditions(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for result in results:
        groups[(result["population"], result["group_size"], result["enrollment_budget_speech_sec_per_person"], result["enrollment_order_variant"])].append(result)
    output = []
    for key, rows in sorted(groups.items()):
        population, group_size, budget, order = key
        output.append({
            "population": population,
            "group_size": group_size,
            "enrollment_budget_speech_sec_per_person": budget,
            "enrollment_order_variant": order,
            "profile_attempts": len(rows),
            "aosc_ready_count": sum(bool(row.get("aosc_ready")) for row in rows),
            "aosc_ready_denominator": len(rows),
            "identity_mapping_valid_count": sum(bool(row.get("identity_mapping_valid")) for row in rows),
            "identity_mapping_denominator": len(rows),
            "evaluated_profile_count": sum(bool(row.get("aosc_ready")) and bool(row.get("identity_mapping_valid")) for row in rows),
            "mapping_failures": [
                {"trial_id": row["trial_id"], "reason": row.get("identity_mapping_error")}
                for row in rows if not row.get("identity_mapping_valid")
            ],
            "not_ready_profiles": [
                {"trial_id": row["trial_id"], "reason": row.get("aosc_not_ready_reason")}
                for row in rows if not row.get("aosc_ready")
            ],
            "profile_details": [
                {
                    "trial_id": row["trial_id"],
                    "aosc_ready": row.get("aosc_ready"),
                    "identity_mapping_valid": row.get("identity_mapping_valid"),
                    "identity_mapping_error": row.get("identity_mapping_error"),
                    "identity_to_channel": row.get("identity_to_channel", {}),
                    "actual_vad_speech_duration_sec_by_identity": {
                        speaker: stats.get("vad_speech_frames", 0) * 0.01
                        for speaker, stats in row.get("speaker_channel_establishment", {}).items()
                    },
                    "channel_establishment": row.get("speaker_channel_establishment", {}),
                    "retained_rows_by_enrolled_identity": retained_rows_by_candidate_identity(row),
                    "cache_row_summary": row.get("cache_row_summary"),
                } for row in rows
            ],
            "median_cache_frames": statistics.median(
                (row.get("cache_row_summary") or {})["cache_frames"] for row in rows
                if (row.get("cache_row_summary") or {}).get("cache_frames") is not None
            ) if any((row.get("cache_row_summary") or {}).get("cache_frames") is not None for row in rows) else None,
            "median_enrollment_construction_ms": statistics.median(row["timing"]["enrollment_construction_ms"] for row in rows),
            "median_trial_end_to_end_ms": statistics.median(row["timing"]["trial_end_to_end_ms"] for row in rows),
        })
    return output


def metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "known": category_counts(rows, "known"),
        "unknown": category_counts(rows, "unknown"),
    }


def bootstrap_rate(
    rows: list[dict[str, Any]], success_categories: set[str], iterations: int, seed: int,
) -> dict[str, Any]:
    eligible = [row for row in rows if row["evaluated"]]
    cohorts = sorted({row["cohort_id"] for row in eligible})
    speakers = sorted({row["speaker_id"] for row in eligible})
    if len(cohorts) < 2 or len(speakers) < 2:
        return {"ci95": None, "reason": "requires at least two speaker and cohort clusters", "cohort_clusters": len(cohorts), "speaker_clusters": len(speakers)}
    rng = random.Random(seed)
    rates = []
    for _ in range(iterations):
        cohort_weights = collections.Counter(rng.choices(cohorts, k=len(cohorts)))
        speaker_weights = collections.Counter(rng.choices(speakers, k=len(speakers)))
        denominator = numerator = 0.0
        for row in eligible:
            weight = cohort_weights[row["cohort_id"]] * speaker_weights[row["speaker_id"]]
            denominator += weight
            if row["category"] in success_categories:
                numerator += weight
        if denominator:
            rates.append(numerator / denominator)
    if not rates:
        return {"ci95": None, "reason": "all bootstrap denominators were empty", "cohort_clusters": len(cohorts), "speaker_clusters": len(speakers)}
    rates.sort()
    return {
        "ci95": [rates[int(0.025 * (len(rates) - 1))], rates[int(0.975 * (len(rates) - 1))]],
        "method": "two-way cluster bootstrap over cohort_id and source speaker_id",
        "iterations": len(rates),
        "cohort_clusters": len(cohorts),
        "speaker_clusters": len(speakers),
    }


def primary_metrics(rows: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, Any]:
    output = {}
    conditions = sorted({(row["population"], row["group_size"], row["budget_sec"], row["order_variant"], row["mic"], row["duration_bucket"]) for row in rows})
    for condition in conditions:
        population, group_size, budget, order, mic, bucket = condition
        subset = [row for row in rows if (
            row["population"], row["group_size"], row["budget_sec"], row["order_variant"], row["mic"], row["duration_bucket"]
        ) == condition]
        key = f"{population}/n{group_size}/e{budget}/{order}/{mic}/{bucket}"
        metrics = metric_rows(subset)
        for kind, success in (("known", {"known_correct"}), ("unknown", {"unknown_correctly_rejected"})):
            metrics[kind]["speaker_cohort_bootstrap"] = bootstrap_rate(
                [row for row in subset if row["known"] is (kind == "known")], success,
                iterations, seed + len(output) * 13 + (0 if kind == "known" else 7),
            )
        output[key] = metrics
    aggregate_conditions = sorted({
        (row["population"], row["group_size"], row["budget_sec"], row["order_variant"], row["mic"])
        for row in rows
    })
    for condition in aggregate_conditions:
        population, group_size, budget, order, mic = condition
        subset = [row for row in rows if (
            row["population"], row["group_size"], row["budget_sec"], row["order_variant"], row["mic"]
        ) == condition]
        key = f"{population}/n{group_size}/e{budget}/{order}/{mic}/all_duration_buckets"
        metrics = metric_rows(subset)
        for kind, success in (("known", {"known_correct"}), ("unknown", {"unknown_correctly_rejected"})):
            metrics[kind]["speaker_cohort_bootstrap"] = bootstrap_rate(
                [row for row in subset if row["known"] is (kind == "known")], success,
                iterations, seed + len(output) * 13 + (0 if kind == "known" else 7),
            )
        output[key] = metrics
    return output


def speaker_cohort_variation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["population"] == "final_holdout" and row["order_variant"] == "randomized"]
    conditions = sorted({(row["group_size"], row["budget_sec"], row["mic"], row["duration_bucket"]) for row in eligible})
    output = []
    for group_size, budget, mic, bucket in conditions:
        subset = [row for row in eligible if (row["group_size"], row["budget_sec"], row["mic"], row["duration_bucket"]) == (group_size, budget, mic, bucket)]
        cohort_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        speaker_groups: dict[tuple[str, bool], list[dict[str, Any]]] = collections.defaultdict(list)
        for row in subset:
            cohort_groups[row["cohort_id"]].append(row)
            speaker_groups[(row["speaker_id"], row["known"])].append(row)

        def concise_counts(items: list[dict[str, Any]]) -> dict[str, Any]:
            valid = [item for item in items if item["evaluated"]]
            counts = collections.Counter(item["category"] for item in valid)
            success_name = "known_correct" if valid and valid[0]["known"] else "unknown_correctly_rejected"
            return {
                "evaluated_attempts": len(valid),
                "successful_attempts": counts.get(success_name, 0),
                "wrong_enrolled_assignments": counts.get("known_wrong_enrolled_identity", 0),
                "unknown_false_accepts": counts.get("unknown_false_accept", 0),
                "rejected_or_no_decision": counts.get("known_rejected_as_unknown", 0) + counts.get("no_decision", 0),
            }

        output.append({
            "group_size": group_size,
            "enrollment_budget_speech_sec_per_person": budget,
            "microphone": mic,
            "duration_bucket": bucket,
            "per_cohort": {cohort: concise_counts(items) for cohort, items in sorted(cohort_groups.items())},
            "per_speaker": {
                f"{speaker}/{'known' if known else 'unknown'}": concise_counts(items)
                for (speaker, known), items in sorted(speaker_groups.items())
            },
        })
    return output


def fit_threshold(
    calibration_rows: list[dict[str, Any]], target_far: float,
    score_name: str = "registered_channel_mean",
) -> dict[str, Any]:
    eligible = [row for row in calibration_rows if row["evaluated"] and row["mic"] == "mic1"]
    known = [row for row in eligible if row["known"]]
    unknown = [row for row in eligible if not row["known"]]
    score_for = lambda row: row.get("diagnostic_scores", {}).get(score_name, row.get("score"))
    scores = sorted({score_for(row) for row in eligible if score_for(row) is not None})
    if not known or not unknown or not scores:
        return {
            "status": "unsupported_insufficient_calibration_scores",
            "score_name": score_name,
            "known_attempts": len(known),
            "unknown_attempts": len(unknown),
        }
    thresholds = [0.0] + scores + [scores[-1] + 1e-9]
    candidates = []
    for threshold in thresholds:
        known_correct = sum(
            1 for row in known
            if row["category"] == "known_correct" and score_for(row) is not None
            and score_for(row) >= threshold
        )
        unknown_false_accept = sum(
            1 for row in unknown
            if row["category"] == "unknown_false_accept" and score_for(row) is not None
            and score_for(row) >= threshold
        )
        unknown_reject = sum(1 for row in unknown if row["category"] == "unknown_correctly_rejected" or (
            row["category"] == "unknown_false_accept" and score_for(row) is not None
            and score_for(row) < threshold
        ))
        far = unknown_false_accept / len(unknown)
        balanced = 0.5 * (known_correct / len(known) + unknown_reject / len(unknown))
        candidates.append((far <= target_far, balanced, -threshold, threshold, known_correct, unknown_false_accept, unknown_reject, far))
    supported = [candidate for candidate in candidates if candidate[0]]
    chosen = max(supported or candidates, key=lambda item: (item[1], item[2]))
    return {
        "status": "calibrated",
        "score_name": score_name,
        "score": {
            "registered_channel_mean": "mean probability over VAD speech frames on the baseline assigned registered channel",
            "top1_dominance": "fraction of active VAD frames assigned to the dominant top-1 channel; persistence proxy",
            "mean_top1_top2_margin": "mean top-1 minus top-2 probability over active VAD frames",
            "vad_speech_duration_sec": "usable VAD speech duration in seconds",
        }.get(score_name, score_name),
        "threshold": chosen[3],
        "target_false_accept_rate": target_far,
        "target_met": chosen[0],
        "calibration_known_attempts": len(known),
        "calibration_unknown_attempts": len(unknown),
        "calibration_known_correct_identification_count": chosen[4],
        "calibration_unknown_false_accept_count": chosen[5],
        "calibration_unknown_reject_count": chosen[6],
        "calibration_false_accept_rate": chosen[7],
        "calibration_balanced_open_set_accuracy": chosen[1],
        "candidate_threshold_count": len(thresholds),
        "selection_rule": "maximize mean of known correct identification rate and unknown rejection rate subject to the calibration FAR target; no-decision remains separate",
    }


def apply_threshold(
    rows: list[dict[str, Any]], threshold: float,
    score_name: str = "registered_channel_mean",
) -> dict[str, Any]:
    score_for = lambda row: row.get("diagnostic_scores", {}).get(score_name, row.get("score"))
    known = [row for row in rows if row["known"] and row["evaluated"] and row["mic"] == "mic1"]
    unknown = [row for row in rows if not row["known"] and row["evaluated"] and row["mic"] == "mic1"]
    known_correct = 0
    known_wrong = 0
    known_reject = 0
    known_no_decision = 0
    unknown_reject = 0
    unknown_false_accept = 0
    unknown_no_decision = 0
    for row in known:
        category = row["category"]
        score = score_for(row)
        accepted = score is not None and score >= threshold
        if category == "known_correct" and accepted:
            known_correct += 1
        elif category == "known_wrong_enrolled_identity" and accepted:
            known_wrong += 1
        elif category == "no_decision":
            known_no_decision += 1
        else:
            known_reject += 1
    for row in unknown:
        category = row["category"]
        if category == "no_decision":
            unknown_no_decision += 1
        elif category == "unknown_correctly_rejected" or (
            category == "unknown_false_accept" and score_for(row) is not None
            and score_for(row) < threshold
        ):
            unknown_reject += 1
        else:
            unknown_false_accept += 1
    return {
        "known": {
            "evaluated_attempts": len(known),
            "correct_identification": known_correct,
            "wrong_enrolled_identity": known_wrong,
            "rejected_as_unknown": known_reject,
            "no_decision": known_no_decision,
        },
        "unknown": {
            "evaluated_attempts": len(unknown),
            "correct_unknown_rejection": unknown_reject,
            "false_accept": unknown_false_accept,
            "no_decision": unknown_no_decision,
            "false_accept_rate": unknown_false_accept / len(unknown) if unknown else None,
        },
    }


def calibration_summary(rows: list[dict[str, Any]], target_far: float) -> dict[str, Any]:
    settings = sorted({(row["group_size"], row["budget_sec"]) for row in rows})
    thresholds = {}
    for group_size, budget in settings:
        cal = [row for row in rows if row["population"] == "calibration" and row["group_size"] == group_size and row["budget_sec"] == budget and row["order_variant"] == "randomized"]
        holdout = [row for row in rows if row["population"] == "final_holdout" and row["group_size"] == group_size and row["budget_sec"] == budget and row["order_variant"] == "randomized"]
        fitted = fit_threshold(cal, target_far)
        key = f"n{group_size}/e{budget}"
        if fitted.get("status") == "calibrated":
            fitted["final_holdout"] = apply_threshold(holdout, fitted["threshold"])
        thresholds[key] = fitted
    diagnostic_scores = {}
    for score_name in (
        "registered_channel_mean", "top1_dominance", "mean_top1_top2_margin",
        "vad_speech_duration_sec",
    ):
        score_points = {}
        for group_size, budget in settings:
            cal = [row for row in rows if row["population"] == "calibration" and row["group_size"] == group_size and row["budget_sec"] == budget and row["order_variant"] == "randomized"]
            holdout = [row for row in rows if row["population"] == "final_holdout" and row["group_size"] == group_size and row["budget_sec"] == budget and row["order_variant"] == "randomized"]
            fitted = fit_threshold(cal, target_far, score_name)
            if fitted.get("status") == "calibrated":
                fitted["final_holdout"] = apply_threshold(holdout, fitted["threshold"], score_name)
            score_points[f"n{group_size}/e{budget}"] = fitted
        diagnostic_scores[score_name] = score_points
    return {
        "criterion": "registered candidate channel mean probability threshold",
        "calibration_partition_only": True,
        "final_holdout_speakers_used_for_fitting": False,
        "operating_points": thresholds,
        "diagnostic_score_operating_points": diagnostic_scores,
        "limitations": [
            "The activity score is a diagnostic native output, not a calibrated posterior probability of identity.",
            "Thresholds are condition-specific and should be interpreted with per-cohort results and cluster bootstrap intervals.",
        ],
    }


def order_sensitivity(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {}
    for trial in results:
        key = (trial["cohort_id"], trial["enrollment_budget_speech_sec_per_person"])
        by_key.setdefault(key, {})[trial["enrollment_order_variant"]] = trial
    output = []
    for (cohort_id, budget), variants in sorted(by_key.items()):
        if "randomized" not in variants or "reverse" not in variants:
            continue
        left, right = variants["randomized"], variants["reverse"]
        left_q = {q["query_id"]: q for q in left.get("queries", [])}
        right_q = {q["query_id"]: q for q in right.get("queries", [])}
        comparable = [(left_q[qid], right_q[qid]) for qid in sorted(left_q.keys() & right_q.keys())
                      if not left_q[qid].get("baseline_category", "").startswith("not_evaluated")
                      and not right_q[qid].get("baseline_category", "").startswith("not_evaluated")]
        changed = sum(a.get("baseline_category") != b.get("baseline_category") for a, b in comparable)
        identity_changed = sum(a.get("predicted_identity") != b.get("predicted_identity") for a, b in comparable)
        ids = set(left.get("identity_to_channel", {})) & set(right.get("identity_to_channel", {}))
        mapping_changed = sum(left["identity_to_channel"].get(speaker) != right["identity_to_channel"].get(speaker) for speaker in ids)
        output.append({
            "cohort_id": cohort_id,
            "group_size": left["group_size"],
            "population": left["population"],
            "enrollment_budget_speech_sec_per_person": budget,
            "query_decisions_compared": len(comparable),
            "query_decision_changes": changed,
            "query_decision_change_rate": changed / len(comparable) if comparable else None,
            "query_identity_decision_changes": identity_changed,
            "query_identity_decision_change_rate": identity_changed / len(comparable) if comparable else None,
            "identity_channels_compared": len(ids),
            "identity_channel_assignments_changed": mapping_changed,
            "randomized_mapping": left.get("identity_to_channel"),
            "reverse_mapping": right.get("identity_to_channel"),
        })
    return output


def contamination_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for trial in results:
        traffic = trial.get("unknown_traffic")
        if not traffic or traffic.get("status") not in ("complete", "partial"):
            continue
        known_query_id = traffic.get("known_return_query_id")
        independent = next((q for q in trial.get("queries", []) if q.get("query_id") == known_query_id), None)
        mapping = {speaker: int(channel) for speaker, channel in trial.get("identity_to_channel", {}).items()}
        enrollment_rows = (trial.get("cache_row_summary") or {}).get("positive_rows_by_channel_gt_0_5", [])
        stages = []
        for stage in traffic.get("stages", []):
            rows_by_channel = stage.get("cache_rows", {}).get("positive_rows_by_channel_gt_0_5", [])
            registered_by_identity = {
                speaker: rows_by_channel[channel] if channel < len(rows_by_channel) else 0
                for speaker, channel in mapping.items()
            }
            enrollment_by_identity = {
                speaker: enrollment_rows[channel] if channel < len(enrollment_rows) else 0
                for speaker, channel in mapping.items()
            }
            stages.append({
                "after_unknown_speakers": stage.get("after_unknown_speakers"),
                "registered_channel_positive_rows_by_identity": registered_by_identity,
                "registered_channel_positive_row_delta_from_enrollment_by_identity": {
                    speaker: registered_by_identity[speaker] - enrollment_by_identity[speaker]
                    for speaker in mapping
                },
                "registered_channel_positive_rows_total": sum(registered_by_identity.values()),
                "unregistered_channel_positive_rows": sum(
                    value for channel, value in enumerate(rows_by_channel)
                    if channel not in set(mapping.values())
                ),
                "cache_frames": stage.get("cache_rows", {}).get("cache_frames"),
                "temporary_profile_changed": stage.get("temporary_profile_changed"),
            })
        output.append({
            "trial_id": trial["trial_id"],
            "unknown_query_order": traffic.get("unknown_query_order", []),
            "enrollment_cache_rows": trial.get("cache_row_summary"),
            "stages": stages,
            "known_return_category_after_contamination": traffic.get("known_return_category"),
            "independent_fresh_session_category": independent.get("baseline_category") if independent else None,
            "same_as_independent_fresh_session": bool(independent and independent.get("baseline_category") == traffic.get("known_return_category")),
            "enrollment_profile_immutable": traffic.get("immutable_enrollment_profile_preserved"),
        })
    return output


def conversation_metrics(
    manifest: dict[str, Any], results: list[dict[str, Any]], trace_dir: Path | None,
) -> list[dict[str, Any]]:
    trial_by_id = {trial["trial_id"]: trial for trial in manifest["trials"]}
    output = []
    for result in results:
        trial = trial_by_id[result["trial_id"]]
        result_conversations = {row.get("conversation_id"): row for row in result.get("conversations", [])}
        mapping = {speaker: int(channel) for speaker, channel in result.get("identity_to_channel", {}).items()}
        enrolled = set(mapping)
        for planned in trial.get("conversations", []):
            observed = result_conversations.get(planned["conversation_id"], {})
            base = {
                "trial_id": trial["trial_id"],
                "conversation_id": planned["conversation_id"],
                "scenario": planned["scenario"],
                "group_size": trial["group_size"],
                "budget_sec": trial["enrollment_budget_speech_sec_per_person"],
                "population": trial["population"],
                "scheduled_duration_sec": planned["duration_sec"],
                "actual_vad_speech_duration_sec": planned["speech_duration_sec"],
                "source_speaker_count": len(planned["reference_tracks"]),
                "scheduled_overlap_target": planned.get("overlap_level"),
                "actual_source_vad_overlap_fraction": planned.get("mix", {}).get("actual_source_vad_overlap_fraction"),
                "status": observed.get("status", "not_returned"),
                "source_annotation_protocol": "10 ms source VAD masks from converted originals, shifted with exact sample-aligned schedule; cosine-tapered crop edge frames are masked out",
            }
            if not trace_dir or observed.get("status") != "complete":
                output.append(base)
                continue
            trace_path = trace_dir / observed["trace_file"]
            frames, channels, period, probabilities = read_trace(trace_path)
            reference: dict[str, bytearray] = {}
            for track in planned["reference_tracks"]:
                mask = bytearray(planned["vad_frames"])
                for start, end in track["speech_intervals_frames"]:
                    mask[start:end] = bytes([1]) * (end - start)
                reference[track["speaker_id"]] = mask
            unknown_ids = [speaker for speaker in reference if speaker not in enrolled]
            unregistered_channels = [channel for channel in range(channels) if channel not in set(mapping.values())]
            intersection_scores = {
                speaker: {
                    channel: sum(
                        1 for frame in range(min(frames, len(reference[speaker])))
                        if reference[speaker][frame] and probabilities[frame * channels + channel] >= 0.5
                    ) for channel in unregistered_channels
                } for speaker in unknown_ids
            }
            unknown_mapping = best_channel_mapping(unknown_ids, unregistered_channels, intersection_scores)
            predicted_id = {channel: speaker for speaker, channel in unknown_mapping.items()}
            predicted_id.update({channel: speaker for speaker, channel in mapping.items()})
            known_tp = known_fp = known_fn = 0
            unknown_tp = unknown_fp = unknown_fn = 0
            speech_tp = speech_fa = speech_miss = 0
            ref_speaker_frames = 0
            der_miss = der_fa = der_confusion = 0
            per_speaker = collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            for frame in range(frames):
                active_ref = {speaker for speaker, mask in reference.items() if frame < len(mask) and mask[frame]}
                active_pred_channels = {channel for channel in range(channels) if probabilities[frame * channels + channel] >= 0.5}
                active_pred_ids = {predicted_id[channel] for channel in active_pred_channels if channel in predicted_id}
                ref_speech = bool(active_ref)
                pred_speech = bool(active_pred_channels)
                speech_tp += int(ref_speech and pred_speech)
                speech_fa += int(not ref_speech and pred_speech)
                speech_miss += int(ref_speech and not pred_speech)
                known_ref = {speaker for speaker in active_ref if speaker in enrolled}
                known_pred = {speaker for speaker in active_pred_ids if speaker in enrolled}
                known_tp += len(known_ref & known_pred)
                known_fn += len(known_ref - known_pred)
                known_fp += len(known_pred - known_ref)
                unknown_ref = bool(active_ref & set(unknown_ids))
                unknown_pred = any(channel in unregistered_channels for channel in active_pred_channels)
                unknown_tp += int(unknown_ref and unknown_pred)
                unknown_fn += int(unknown_ref and not unknown_pred)
                unknown_fp += int(unknown_pred and not unknown_ref)
                for speaker in enrolled:
                    ref_active = speaker in active_ref
                    pred_active = speaker in active_pred_ids
                    per_speaker[speaker]["tp"] += int(ref_active and pred_active)
                    per_speaker[speaker]["fp"] += int(pred_active and not ref_active)
                    per_speaker[speaker]["fn"] += int(ref_active and not pred_active)
                ref_speaker_frames += len(active_ref)
                der_pred_ids = set(active_pred_ids)
                der_pred_ids.update(
                    f"__unmapped_channel_{channel}"
                    for channel in active_pred_channels if channel not in predicted_id
                )
                correct = len(active_ref & der_pred_ids)
                matched_slots = min(len(active_ref), len(der_pred_ids))
                der_miss += max(len(active_ref) - len(der_pred_ids), 0)
                der_fa += max(len(der_pred_ids) - len(active_ref), 0)
                der_confusion += max(0, matched_slots - correct)
            base.update({
                "trace_frames": frames,
                "trace_channels": channels,
                "seconds_per_output_frame": period,
                "unknown_anonymous_channel_mapping_for_diarization_only": unknown_mapping,
                "speech_detection": {
                    "tp_frames": speech_tp,
                    "miss_frames": speech_miss,
                    "false_alarm_frames": speech_fa,
                    "reference_speech_frames": speech_tp + speech_miss,
                    "predicted_speech_frames": speech_tp + speech_fa,
                },
                "known_identity_multilabel": {
                    "tp": known_tp, "fp": known_fp, "fn": known_fn,
                    "precision": known_tp / (known_tp + known_fp) if known_tp + known_fp else None,
                    "recall": known_tp / (known_tp + known_fn) if known_tp + known_fn else None,
                },
                "unknown_presence_multilabel": {
                    "tp_frames": unknown_tp, "fp_frames": unknown_fp, "fn_frames": unknown_fn,
                    "unknown_false_accept_frames": unknown_fp,
                },
                "diarization_error_components_no_collar": {
                    "speaker_time_reference_frames": ref_speaker_frames,
                    "miss_speaker_frames": der_miss,
                    "false_alarm_speaker_frames": der_fa,
                    "speaker_confusion_frames": der_confusion,
                    "der": (der_miss + der_fa + der_confusion) / ref_speaker_frames if ref_speaker_frames else None,
                    "identity_protocol": "enrolled external IDs stay fixed; unknown speaker channels receive a session-local one-to-one overlap-maximizing mapping, so this is diarization scoring and not persistent VoiceID accuracy",
                },
                "known_identity_per_speaker": per_speaker,
            })
            output.append(base)
    return output


def best_channel_mapping(
    speakers: list[str], channels: list[int], scores: dict[str, dict[int, int]],
) -> dict[str, int]:
    if not speakers or not channels:
        return {}
    best_score = -1
    best: dict[str, int] = {}
    # The full preset reserves at most four unknown identities per conversation.
    # Limit the assignment search defensively if a user supplies a larger cohort.
    search_speakers = speakers[: min(6, len(speakers))]
    for choice in itertools.permutations(channels, min(len(search_speakers), len(channels))):
        assignment = dict(zip(search_speakers, choice))
        total = sum(scores.get(speaker, {}).get(channel, 0) for speaker, channel in assignment.items())
        if total > best_score:
            best_score = total
            best = assignment
    return best


def read_trace(path: Path) -> tuple[int, int, float, list[float]]:
    data = path.read_bytes()
    if len(data) < 32 or data[:8] != b"NMSP8F1\0":
        raise ValueError(f"invalid probability trace: {path}")
    version, channels = struct.unpack_from("<II", data, 8)
    frames = struct.unpack_from("<Q", data, 16)[0]
    period = struct.unpack_from("<d", data, 24)[0]
    if version != 1 or len(data) != 32 + frames * channels * 4:
        raise ValueError(f"probability trace dimensions disagree with byte count: {path}")
    values = list(struct.unpack_from(f"<{frames * channels}f", data, 32))
    return frames, channels, period, values


def cpu_vulkan_parity(
    cpu_results: list[dict[str, Any]], vulkan_results: list[dict[str, Any]],
    cpu_trace_dir: Path | None, vulkan_trace_dir: Path | None,
) -> dict[str, Any]:
    cpu = {row["trial_id"]: row for row in cpu_results}
    vulkan = {row["trial_id"]: row for row in vulkan_results}
    details = []
    probability_differences = []
    identity_changed = 0
    category_changed = 0
    compared = 0
    profile_details = []
    channel_mapping_assignment_count = 0
    channel_mapping_assignment_changes = 0
    exact_channel_mapping_equal_pairs = 0
    for trial_id in sorted(cpu.keys() & vulkan.keys()):
        left, right = cpu[trial_id], vulkan[trial_id]
        cpu_mapping = {
            str(speaker): int(channel)
            for speaker, channel in left.get("identity_to_channel", {}).items()
        }
        vulkan_mapping = {
            str(speaker): int(channel)
            for speaker, channel in right.get("identity_to_channel", {}).items()
        }
        shared_speakers = cpu_mapping.keys() & vulkan_mapping.keys()
        changed_assignments = sum(
            cpu_mapping[speaker] != vulkan_mapping[speaker]
            for speaker in shared_speakers
        )
        channel_mapping_assignment_count += len(shared_speakers)
        channel_mapping_assignment_changes += changed_assignments
        exact_mapping_equal = cpu_mapping == vulkan_mapping
        exact_channel_mapping_equal_pairs += int(exact_mapping_equal)
        cpu_imports = [
            float(query["profile_import_ms"])
            for query in left.get("queries", []) if query.get("profile_import_ms") is not None
        ]
        vulkan_imports = [
            float(query["profile_import_ms"])
            for query in right.get("queries", []) if query.get("profile_import_ms") is not None
        ]
        profile_details.append({
            "trial_id": trial_id,
            "cpu_profile_saved_and_reloaded": left.get("profile_saved_and_reloaded"),
            "vulkan_profile_saved_and_reloaded": right.get("profile_saved_and_reloaded"),
            "cpu_profile_crc32": left.get("profile_crc32"),
            "vulkan_profile_crc32": right.get("profile_crc32"),
            "profile_crc32_equal": left.get("profile_crc32") == right.get("profile_crc32"),
            "cpu_identity_to_channel": cpu_mapping,
            "vulkan_identity_to_channel": vulkan_mapping,
            "channel_mapping_equal": exact_mapping_equal,
            "shared_channel_assignments_compared": len(shared_speakers),
            "shared_channel_assignments_changed": changed_assignments,
            "cpu_query_profile_import_count": len(cpu_imports),
            "vulkan_query_profile_import_count": len(vulkan_imports),
            "cpu_median_query_profile_import_ms": statistics.median(cpu_imports) if cpu_imports else None,
            "vulkan_median_query_profile_import_ms": statistics.median(vulkan_imports) if vulkan_imports else None,
        })
        lq = {q.get("query_id"): q for q in left.get("queries", [])}
        rq = {q.get("query_id"): q for q in right.get("queries", [])}
        for query_id in sorted(lq.keys() & rq.keys()):
            a, b = lq[query_id], rq[query_id]
            if a.get("baseline_category", "").startswith("not_evaluated") or b.get("baseline_category", "").startswith("not_evaluated"):
                continue
            compared += 1
            identity_decision_changed = a.get("predicted_identity") != b.get("predicted_identity")
            category_decision_changed = a.get("baseline_category") != b.get("baseline_category")
            identity_changed += int(identity_decision_changed)
            category_changed += int(category_decision_changed)
            item = {
                "trial_id": trial_id,
                "query_id": query_id,
                "cpu_decision": a.get("baseline_category"),
                "vulkan_decision": b.get("baseline_category"),
                "cpu_predicted_identity": a.get("predicted_identity"),
                "vulkan_predicted_identity": b.get("predicted_identity"),
                "identity_decision_changed": identity_decision_changed,
                "category_decision_changed": category_decision_changed,
                "cpu_profile_crc32": left.get("profile_crc32"),
                "vulkan_profile_crc32": right.get("profile_crc32"),
            }
            if cpu_trace_dir and vulkan_trace_dir:
                fcpu = cpu_trace_dir / a["trace_file"]
                fgpu = vulkan_trace_dir / b["trace_file"]
                if fcpu.exists() and fgpu.exists():
                    n1, c1, p1, x = read_trace(fcpu)
                    n2, c2, p2, y = read_trace(fgpu)
                    if n1 != n2 or c1 != c2 or abs(p1 - p2) > 1e-9:
                        raise ValueError(f"CPU/Vulkan trace geometry mismatch for {trial_id}/{query_id}")
                    n = n1
                    diffs = [abs(x[i] - y[i]) for i in range(n * c1)]
                    top1_flips = sum(
                        max(range(c1), key=lambda c: x[f * c1 + c]) != max(range(c1), key=lambda c: y[f * c1 + c])
                        for f in range(n)
                    )
                    activity_flips = sum(
                        (max(x[f * c1:(f + 1) * c1]) >= 0.5) != (max(y[f * c1:(f + 1) * c1]) >= 0.5)
                        for f in range(n)
                    )
                    diff = {
                        "trace_frames_compared": n,
                        "probability_channels": c1,
                        "max_absolute_difference": max(diffs, default=0.0),
                        "mean_absolute_difference": statistics.mean(diffs) if diffs else 0.0,
                        "rmse": math.sqrt(statistics.mean(value * value for value in diffs)) if diffs else 0.0,
                        "top1_channel_flip_frames": top1_flips,
                        "activity_ge_0_5_changed_frames": activity_flips,
                    }
                    item["native_probability_parity"] = diff
                    probability_differences.append(diff)
            details.append(item)
    return {
        "cpu_trials": len(cpu_results),
        "vulkan_trials": len(vulkan_results),
        "identity_decisions_compared": compared,
        "identity_decisions_changed": identity_changed,
        "identity_decision_change_rate": identity_changed / compared if compared else None,
        "baseline_category_decisions_changed": category_changed,
        "baseline_category_decision_change_rate": category_changed / compared if compared else None,
        "profile_pairs": len(profile_details),
        "channel_mapping_equal_pairs": exact_channel_mapping_equal_pairs,
        "channel_mapping_shared_assignments_compared": channel_mapping_assignment_count,
        "channel_mapping_shared_assignments_changed": channel_mapping_assignment_changes,
        "profile_import_roundtrip_pairs": sum(
            item["cpu_profile_saved_and_reloaded"] is True
            and item["vulkan_profile_saved_and_reloaded"] is True
            for item in profile_details
        ),
        "profile_crc32_equal_pairs": sum(item["profile_crc32_equal"] for item in profile_details),
        "profile_details": profile_details,
        "probability_trace_query_count": len(probability_differences),
        "median_query_mae": statistics.median(row["mean_absolute_difference"] for row in probability_differences) if probability_differences else None,
        "maximum_query_max_absolute_difference": max((row["max_absolute_difference"] for row in probability_differences), default=None),
        "details": details,
    }


def performance_summary(
    manifest: dict[str, Any], results: list[dict[str, Any]], run_events: list[dict[str, Any]],
) -> dict[str, Any]:
    trial_by_id = {trial["trial_id"]: trial for trial in manifest["trials"]}
    queries = [query for result in results for query in result.get("queries", []) if query.get("query_evaluated", True) and query.get("inference_ms") is not None]
    durations = [float(query.get("recording_duration_sec", 0.0)) for query in queries]
    inference_ms = [float(query["inference_ms"]) for query in queries]
    profile_import_ms = [
        float(query["profile_import_ms"])
        for query in queries if query.get("profile_import_ms") is not None
    ]
    enrollment_construction_ms = [
        float(result["timing"]["enrollment_construction_ms"])
        for result in results
        if result.get("timing", {}).get("enrollment_construction_ms") is not None
    ]
    trial_end_to_end_ms = [
        float(result["timing"]["trial_end_to_end_ms"])
        for result in results
        if result.get("timing", {}).get("trial_end_to_end_ms") is not None
    ]
    realtime = [query.get("audio_per_inference_second") for query in queries if query.get("audio_per_inference_second") is not None]

    def p95(values: list[float]) -> float | None:
        return sorted(values)[int(0.95 * (len(values) - 1))] if values else None

    audio_fed_to_model_sec = 0.0
    for result in results:
        trial = trial_by_id[result["trial_id"]]
        audio_fed_to_model_sec += float(result.get("enrollment_audio_duration_sec_with_gaps", 0.0))
        audio_fed_to_model_sec += sum(float(query.get("recording_duration_sec", 0.0)) for query in result.get("queries", [])
                                      if query.get("query_evaluated", True) and query.get("inference_ms") is not None)
        completed_conversations = {item.get("conversation_id") for item in result.get("conversations", []) if item.get("status") == "complete"}
        audio_fed_to_model_sec += sum(float(item.get("duration_sec", 0.0)) for item in trial.get("conversations", [])
                                      if item.get("conversation_id") in completed_conversations)
        traffic = result.get("unknown_traffic") or {}
        if traffic.get("status") == "complete":
            queries_by_id = {item.get("query_id"): item for item in trial.get("queries", [])}
            audio_fed_to_model_sec += sum(float(queries_by_id.get(query_id, {}).get("duration_sec", 0.0))
                                          for query_id in traffic.get("unknown_query_order", []))
            known_return = queries_by_id.get(traffic.get("known_return_query_id"), {})
            audio_fed_to_model_sec += float(known_return.get("duration_sec", 0.0))
            audio_fed_to_model_sec += 0.25 * len(traffic.get("unknown_query_order", []))
    process_wall_sec = sum(event.get("process_elapsed_ms", 0.0) for event in run_events) / 1000.0
    return {
        "completed_trials": len(results),
        "per_query_inference_count": len(queries),
        "median_query_inference_ms": statistics.median(inference_ms) if inference_ms else None,
        "p95_query_inference_ms": p95(inference_ms),
        "profile_import_count": len(profile_import_ms),
        "median_profile_import_ms": statistics.median(profile_import_ms) if profile_import_ms else None,
        "p95_profile_import_ms": p95(profile_import_ms),
        "sum_profile_import_ms": sum(profile_import_ms),
        "enrollment_construction_count": len(enrollment_construction_ms),
        "median_enrollment_construction_ms": statistics.median(enrollment_construction_ms) if enrollment_construction_ms else None,
        "p95_enrollment_construction_ms": p95(enrollment_construction_ms),
        "trial_end_to_end_count": len(trial_end_to_end_ms),
        "median_trial_end_to_end_ms": statistics.median(trial_end_to_end_ms) if trial_end_to_end_ms else None,
        "p95_trial_end_to_end_ms": p95(trial_end_to_end_ms),
        "sum_query_audio_duration_sec": sum(durations),
        "sum_query_inference_sec": sum(inference_ms) / 1000.0,
        "sum_query_audio_over_sum_query_inference_wall_time": sum(durations) / (sum(inference_ms) / 1000.0) if inference_ms and sum(inference_ms) else None,
        "median_per_query_audio_per_inference_second": statistics.median(realtime) if realtime else None,
        "audio_fed_to_model_duration_sec": audio_fed_to_model_sec,
        "process_wall_time_sec_from_run_events": process_wall_sec if run_events else None,
        "audio_throughput_per_process_wall_second": audio_fed_to_model_sec / process_wall_sec if process_wall_sec else None,
        "process_wall_rtf": process_wall_sec / audio_fed_to_model_sec if process_wall_sec and audio_fed_to_model_sec else None,
        "sum_trial_end_to_end_ms": sum(result.get("timing", {}).get("trial_end_to_end_ms", 0.0) for result in results),
        "run_events": run_events,
        "max_host_peak_rss_bytes": max((event.get("host_peak_rss_bytes", 0) for event in run_events), default=None),
        "max_gpu_memory_used_mib_sample": max((event["gpu_memory_used_mib_sample"] for event in run_events if event.get("gpu_memory_used_mib_sample") is not None), default=None),
        "max_observed_batch": max((event.get("max_observed_batch", 0) for event in run_events), default=None),
        "total_batcher_batches": sum(event.get("batches", 0) for event in run_events),
        "total_batcher_requested_items": sum(event.get("requested_items", 0) for event in run_events),
        "mean_batch_size_items": (
            sum(event.get("requested_items", 0) for event in run_events)
            / sum(event.get("batches", 0) for event in run_events)
            if sum(event.get("batches", 0) for event in run_events) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--cpu-results", type=Path)
    parser.add_argument("--cpu-trace-dir", type=Path)
    parser.add_argument("--vulkan-trace-dir", type=Path)
    parser.add_argument("--trace-dir", type=Path, help="trace directory for the primary results, including conversation scoring")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    results = read_jsonl(args.results)
    expected_fingerprint = fnv64(args.manifest.read_text(encoding="utf-8"))
    rows, completeness = query_rows(manifest, results, expected_fingerprint)
    run_events = read_jsonl(Path(str(args.results) + ".runs.jsonl"))
    output = {
        "schema": "nemo-speech.voiceid-task08-summary.v1",
        "manifest": args.manifest.name,
        "results": args.results.name,
        "seed": args.seed,
        "completeness": completeness,
        "profile_readiness_and_mapping": summarize_profile_conditions(results),
        "isolated_query_metrics": primary_metrics(rows, args.bootstrap_iterations, args.seed),
        "speaker_and_cohort_variation": speaker_cohort_variation(rows),
        "open_set_calibration": calibration_summary(rows, args.target_far),
        "enrollment_order_sensitivity": order_sensitivity(results),
        "aosc_unknown_traffic_contamination": contamination_summary(results),
        "synthetic_conversations": conversation_metrics(manifest, results, args.trace_dir),
        "performance": performance_summary(manifest, results, run_events),
    }
    if args.cpu_results:
        output["cpu_vulkan_correctness_sampling"] = cpu_vulkan_parity(
            read_jsonl(args.cpu_results), results, args.cpu_trace_dir,
            args.vulkan_trace_dir or args.trace_dir,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "completed_trials": completeness["completed_trials"],
        "manifest_trials": completeness["manifest_trials"],
        "query_attempts": len(rows),
        "conversation_runs": sum(item.get("status") == "complete" for item in output["synthetic_conversations"]),
        "threshold_settings": len(output["open_set_calibration"]["operating_points"]),
    }, indent=2))


if __name__ == "__main__":
    main()
