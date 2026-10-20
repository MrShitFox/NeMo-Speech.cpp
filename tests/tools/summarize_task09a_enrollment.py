#!/usr/bin/env python3
"""Aggregate Task 09A failure taxonomy and validated enrollment trials."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compact_compression_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not state:
        return None
    return {
        "cache_rows": state.get("cache_rows"),
        "positive_rows_by_prediction_channel_gt_0_5": state.get("positive_rows_by_channel_gt_0_5"),
        "argmax_rows_by_prediction_channel_gt_0_5": state.get("argmax_rows_by_channel_gt_0_5"),
    }


def compact_compression_events(
    events: list[dict[str, Any]], include_source_identity_lineage: bool = False,
) -> list[dict[str, Any]]:
    output = []
    for event in events:
        row = {
            "compression_index": event.get("compression_index"),
            "before": compact_compression_state(event.get("before")),
            "after": compact_compression_state(event.get("after")),
        }
        if include_source_identity_lineage:
            row["source_identity_rows_and_channel_activity_before"] = (
                event.get("before", {}).get("source_identity_rows_and_channel_activity"))
            row["source_identity_rows_and_channel_activity_after"] = (
                event.get("after", {}).get("source_identity_rows_and_channel_activity"))
            row["retained_source_row_delta_by_identity"] = event.get("retained_source_row_delta_by_identity")
        output.append(row)
    return output


def matrix_by_identity(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = record.get("task09a_enrollment_diagnostics") or {}
    all_top1: dict[str, list[int]] = collections.defaultdict(list)
    for clip in diagnostics.get("clips", []):
        identity = clip.get("speaker_id")
        counts = clip.get("top1_channel_counts_all_vad_frames", [])
        if identity is None:
            continue
        if not all_top1[identity]:
            all_top1[identity] = [0] * len(counts)
        for channel, count in enumerate(counts):
            all_top1[identity][channel] += int(count)
    out = {}
    for identity, stats in record.get("speaker_channel_establishment", {}).items():
        out[identity] = {
            "ground_truth_vad_speech_frames": stats.get("vad_speech_frames", 0),
            "native_active_frames": stats.get("model_activity_frames", 0),
            "native_active_fraction_of_vad": safe_rate(
                stats.get("model_activity_frames", 0), stats.get("vad_speech_frames", 0)),
            "independent_dominant_channel_candidate": stats.get("candidate_channel"),
            "channel_established": stats.get("channel_established", False),
            "dominance": stats.get("dominant_channel_share", 0.0),
            "mean_probabilities_by_native_channel": stats.get("mean_probability_by_channel", []),
            "top1_frame_counts_by_native_channel_over_all_vad_frames": all_top1.get(identity, []),
            "active_top1_frame_counts_by_native_channel": stats.get("top1_frames_by_channel", []),
        }
    return out


def failure_taxonomy(record: dict[str, Any]) -> dict[str, Any]:
    mapping_error = record.get("identity_mapping_error")
    if mapping_error == "multiple enrolled identities selected the same native channel":
        mapping_cause = "speaker_channel_collision"
    elif mapping_error == "one or more speakers did not establish a stable channel from VAD speech frames":
        mapping_cause = "channel_establishment_failure"
    elif not record.get("identity_mapping_valid", False):
        mapping_cause = "other_identity_mapping_failure"
    else:
        mapping_cause = None
    if not record.get("aosc_ready"):
        primary = "aosc_not_ready"
    else:
        speakers = record.get("speaker_channel_establishment", {})
        if mapping_cause == "channel_establishment_failure":
            primary = "channel_establishment_failure"
        elif mapping_cause == "speaker_channel_collision":
            primary = "speaker_channel_collision"
        elif not record.get("identity_mapping_valid"):
            primary = "other_identity_mapping_failure"
        else:
            primary = "usable_under_task08_strict_rule"

    owners: dict[int, list[str]] = collections.defaultdict(list)
    all_candidate_owners: dict[int, list[str]] = collections.defaultdict(list)
    for identity, stats in record.get("speaker_channel_establishment", {}).items():
        channel = stats.get("candidate_channel")
        if channel is not None:
            all_candidate_owners[int(channel)].append(identity)
            if stats.get("channel_established"):
                owners[int(channel)].append(identity)
    collisions = {str(channel): ids for channel, ids in owners.items() if len(ids) > 1}
    unestablished = [
        identity for identity, stats in record.get("speaker_channel_establishment", {}).items()
        if not stats.get("channel_established")
    ]
    details = record.get("task09a_enrollment_diagnostics") or {}
    temporal = details.get("per_identity_temporal_activity", {})
    compression = details.get("aosc_compression_events", [])
    final_cache = compact_compression_state(compression[-1]["after"]) if compression else None
    final_input = compact_compression_state(compression[-1]["before"]) if compression else None
    segment_overlap = (details.get("native_segmenter") or {}).get(
        "vad_speech_frame_intersections_by_identity_and_channel", {})
    shared_segment_channels = {}
    if segment_overlap:
        channel_count = max((len(counts) for counts in segment_overlap.values()), default=0)
        shared_segment_channels = {
            str(channel): [identity for identity, counts in segment_overlap.items() if counts[channel] > 0]
            for channel in range(channel_count)
        }
        shared_segment_channels = {
            channel: identities for channel, identities in shared_segment_channels.items()
            if len(identities) > 1
        }
    activity_support = {}
    low_probability_identities = []
    weak_dominance_identities = []
    no_native_activity_identities = []
    multi_channel_active_top1_identities = []
    for identity, stats in record.get("speaker_channel_establishment", {}).items():
        vad_frames = int(stats.get("vad_speech_frames", 0))
        active_frames = int(stats.get("model_activity_frames", 0))
        channel = stats.get("candidate_channel")
        means = stats.get("mean_probability_by_channel", [])
        candidate_mean = means[channel] if channel is not None and channel < len(means) else None
        active_top1 = stats.get("top1_frames_by_channel", [])
        active_channel_count = sum(int(value) > 0 for value in active_top1)
        support = {
            "ground_truth_vad_speech_frames": vad_frames,
            "native_active_frames_ge_0_5": active_frames,
            "native_active_fraction_of_vad": safe_rate(active_frames, vad_frames),
            "candidate_channel": channel,
            "candidate_mean_probability": candidate_mean,
            "candidate_dominance_share": stats.get("dominant_channel_share"),
            "candidate_dominant_tied": stats.get("dominant_channel_tie"),
            "active_top1_channel_count": active_channel_count,
        }
        activity_support[identity] = support
        if vad_frames > 0 and active_frames == 0:
            no_native_activity_identities.append(identity)
        if candidate_mean is not None and candidate_mean < 0.5:
            low_probability_identities.append(identity)
        if stats.get("dominant_channel_tie") or float(stats.get("dominant_channel_share", 0.0)) < 0.5:
            weak_dominance_identities.append(identity)
        if active_channel_count > 1:
            multi_channel_active_top1_identities.append(identity)
    per_clip = details.get("clips", [])
    within_utterance_transitions = sum(
        int(item.get("consecutive_vad_frame_top1_channel_transitions", 0)) for item in per_clip)
    clip_to_clip_transitions = sum(
        int(item.get("clip_to_clip_dominant_channel_transitions", 0)) for item in temporal.values())
    compression_row_losses: dict[str, int] = collections.defaultdict(int)
    compression_events_with_row_loss = 0
    for event in compression:
        deltas = event.get("retained_source_row_delta_by_identity", {})
        losses = {identity: int(delta) for identity, delta in deltas.items() if int(delta) < 0}
        if losses:
            compression_events_with_row_loss += 1
            for identity, delta in losses.items():
                compression_row_losses[identity] += -delta
    evidence_tags = []
    if not record.get("aosc_ready"):
        evidence_tags.append("aosc_not_ready")
    if collisions:
        evidence_tags.append("multiple_identities_share_strict_dominant_channel")
    if shared_segment_channels:
        evidence_tags.append("native_segments_overlap_multiple_labeled_people")
    if clip_to_clip_transitions:
        evidence_tags.append("dominant_channel_changes_across_utterances")
    if within_utterance_transitions:
        evidence_tags.append("top1_channel_changes_within_vad_speech")
    if low_probability_identities:
        evidence_tags.append("candidate_mean_probability_below_existing_0_5_requirement")
    if weak_dominance_identities:
        evidence_tags.append("candidate_dominance_below_existing_0_5_requirement_or_tied")
    if multi_channel_active_top1_identities:
        evidence_tags.append("identity_has_active_top1_evidence_on_multiple_channels")
    if no_native_activity_identities:
        evidence_tags.append("no_native_active_frames_during_ground_truth_speech")
    if compression_row_losses:
        evidence_tags.append("positive_source_rows_removed_by_aosc_compression")
    return {
        "primary_cause": primary,
        "mapping_failure_cause": mapping_cause,
        "diagnostic_evidence_tags": evidence_tags,
        "aosc_ready": record.get("aosc_ready", False),
        "aosc_not_ready_reason": record.get("aosc_not_ready_reason"),
        "strict_mapping_valid": record.get("identity_mapping_valid", False),
        "strict_mapping_error": record.get("identity_mapping_error"),
        "identities_without_established_channel": unestablished,
        "candidate_channel_owners": {str(ch): ids for ch, ids in owners.items()},
        "all_dominant_candidate_owners_including_unestablished": {
            str(ch): ids for ch, ids in all_candidate_owners.items()
        },
        "shared_candidate_channels": collisions,
        "channels_with_native_segments_overlapping_multiple_labeled_people": shared_segment_channels,
        "enrollment_label_confusion_matrix_identity_by_native_channel": matrix_by_identity(record),
        "activity_support_by_identity": activity_support,
        "within_utterance_top1_transition_count": within_utterance_transitions,
        "clip_to_clip_dominant_channel_transition_count": clip_to_clip_transitions,
        "identities_with_active_top1_evidence_on_multiple_channels": multi_channel_active_top1_identities,
        "identities_below_existing_mean_probability_requirement": low_probability_identities,
        "identities_below_existing_dominance_requirement_or_tied": weak_dominance_identities,
        "identities_with_vad_speech_but_no_native_activity": no_native_activity_identities,
        "global_assignment": record.get("research_global_assignment"),
        "temporal_channel_stability_by_identity": temporal,
        "aosc_compression": {
            "compression_count": len(compression),
            "events_with_positive_source_row_loss": compression_events_with_row_loss,
            "positive_source_rows_removed_by_identity": dict(compression_row_losses),
            "retention_scope": "per-channel prediction-row counts; source-identity attribution is included only for replay records with corrected source-frame lineage",
            "final_precompression_state": final_input,
            "final_retained_state": final_cache,
            "events": compact_compression_events(
                compression,
                (record.get("task09a_enrollment_diagnostics") or {}).get(
                    "compression_source_identity_lineage_version", 0) >= 2),
        },
        "native_segmenter": details.get("native_segmenter"),
    }


def task08_baseline(raw_records: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    records = {item["trial_id"]: item for item in raw_records}
    ready = sum(bool(item.get("aosc_ready")) for item in raw_records)
    map_valid = sum(bool(item.get("identity_mapping_valid")) for item in raw_records)
    ready_and_mapped = sum(bool(item.get("aosc_ready")) and bool(item.get("identity_mapping_valid")) for item in raw_records)
    causes = collections.Counter(failure_taxonomy(item)["primary_cause"] for item in raw_records)
    group = collections.Counter()
    for item in raw_records:
        if not item.get("identity_mapping_valid"):
            taxonomy = failure_taxonomy(item)["mapping_failure_cause"]
            if taxonomy == "speaker_channel_collision":
                group["mapping_failures_channel_collision"] += 1
            elif taxonomy == "channel_establishment_failure":
                group["mapping_failures_channel_not_established"] += 1
    return {
        "source": "Task 08 original raw result checkpoint and aggregate",
        "attempted_profiles": len(raw_records),
        "aosc_ready": {"numerator": ready, "denominator": len(raw_records), "rate": safe_rate(ready, len(raw_records))},
        "identity_map_valid": {"numerator": map_valid, "denominator": len(raw_records), "rate": safe_rate(map_valid, len(raw_records))},
        "aosc_ready_and_identity_map_valid": {
            "numerator": ready_and_mapped, "denominator": len(raw_records),
            "rate": safe_rate(ready_and_mapped, len(raw_records)),
        },
        "profile_taxonomy_counts": dict(causes),
        "mapping_failure_counts": dict(group),
        "profiles_not_ready_but_mapping_invalid": sum(
            not item.get("aosc_ready") and not item.get("identity_mapping_valid") for item in raw_records),
        "profiles_not_ready_but_mapping_valid": sum(
            not item.get("aosc_ready") and item.get("identity_mapping_valid") for item in raw_records),
        "readiness_mapping_cross_tab": {
            "aosc_not_ready_map_invalid": sum(not item.get("aosc_ready") and not item.get("identity_mapping_valid") for item in raw_records),
            "aosc_not_ready_map_valid": sum(not item.get("aosc_ready") and item.get("identity_mapping_valid") for item in raw_records),
            "aosc_ready_map_invalid": sum(item.get("aosc_ready") and not item.get("identity_mapping_valid") for item in raw_records),
            "aosc_ready_map_valid": sum(item.get("aosc_ready") and item.get("identity_mapping_valid") for item in raw_records),
        },
        "task08_aggregate_profile_readiness_rows": len(aggregate.get("profile_readiness_and_mapping", [])),
        "task08_query_metrics_available": bool(aggregate.get("isolated_query_metrics")),
        "task08_query_metrics_source": "tests/results/voiceid_task08_full.json",
        "raw_trial_ids": sorted(records),
    }


def reproduced_baseline(
    raw_records: list[dict[str, Any]], diagnostics_records: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline = {item["trial_id"]: item for item in raw_records}
    diagnostics = {item["trial_id"]: item for item in diagnostics_records}
    selected = {
        key for key, item in baseline.items()
        if not item.get("aosc_ready") or not item.get("identity_mapping_valid")
    }
    checked = sorted(selected & diagnostics.keys())
    ready_match = sum(baseline[key].get("aosc_ready") == diagnostics[key].get("aosc_ready") for key in checked)
    map_match = sum(baseline[key].get("identity_mapping_valid") == diagnostics[key].get("identity_mapping_valid") for key in checked)
    crc_match = sum(baseline[key].get("profile_crc32") == diagnostics[key].get("profile_crc32") for key in checked)
    matrix_match = sum(
        baseline[key].get("speaker_channel_establishment") == diagnostics[key].get("speaker_channel_establishment")
        for key in checked
    )
    return {
        "failed_profile_ids_expected": len(selected),
        "failed_profile_ids_replayed": len(checked),
        "aosc_readiness_match": {"numerator": ready_match, "denominator": len(checked)},
        "strict_mapping_match": {"numerator": map_match, "denominator": len(checked)},
        "profile_crc_match_including_null_not_ready": {"numerator": crc_match, "denominator": len(checked)},
        "per_identity_activity_matrix_exact_match": {"numerator": matrix_match, "denominator": len(checked)},
        "missing_replay_ids": sorted(selected - diagnostics.keys()),
        "unexpected_replay_ids": sorted(diagnostics.keys() - selected),
    }


def validation_score(record: dict[str, Any], method: str) -> float | None:
    validation = record.get("enrollment_validation", {})
    key = "strict_mapping" if method == "strict" else "research_global_assignment"
    result = validation.get(key, {})
    if not result.get("passed"):
        return None
    clips = result.get("clips", [])
    margins = [float(item["assigned_channel_top1_margin_mean"]) for item in clips if item.get("passed")]
    return statistics.mean(margins) if margins else None


def eligible_mapping(record: dict[str, Any]) -> tuple[str | None, float | None]:
    validation = record.get("enrollment_validation", {})
    if not record.get("aosc_ready"):
        return None, None
    if record.get("identity_mapping_valid") and validation.get("strict_verification_passed"):
        return "strict", validation_score(record, "strict")
    if (
        record.get("research_global_assignment", {}).get("mapping_accepted_for_research_comparison")
        and validation.get("research_global_verification_passed")
    ):
        return "global_assignment", validation_score(record, "global_assignment")
    return None, None


def category_counts(
    result: dict[str, Any] | None, method: str | None, planned_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = len(planned_queries)
    known_expected = sum(bool(q.get("known")) for q in planned_queries)
    unknown_expected = expected - known_expected
    categories = collections.Counter()
    evaluated_queries: list[dict[str, Any]] = []
    not_evaluated_count = 0
    if result is not None and method is not None:
        field = "baseline_category" if method == "strict" else "research_global_assignment_category"
        for query in result.get("queries", []):
            category = query.get(field, "not_evaluated")
            if (not query.get("query_evaluated", not str(category).startswith("not_evaluated"))
                    or str(category).startswith("not_evaluated")):
                not_evaluated_count += 1
                continue
            categories[category] += 1
            evaluated_queries.append(query)
    evaluated = len(evaluated_queries)
    known_correct = categories["known_correct"]
    known_wrong = categories["known_wrong_enrolled_identity"]
    known_rejected = categories["known_rejected_as_unknown"]
    unknown_rejected = categories["unknown_correctly_rejected"]
    unknown_false_accept = categories["unknown_false_accept"]
    no_decision = categories["no_decision"]
    known_evaluated = sum(bool(q.get("known")) for q in evaluated_queries)
    unknown_evaluated = evaluated - known_evaluated
    return {
        "planned_query_attempts": expected,
        "known_planned": known_expected,
        "unknown_planned": unknown_expected,
        "evaluated_query_count": evaluated,
        "known_evaluated_query_count": known_evaluated,
        "unknown_evaluated_query_count": unknown_evaluated,
        "known_correct_identification": {"numerator": known_correct, "denominator": known_expected,
                                         "conditional_rate": safe_rate(known_correct, known_evaluated),
                                         "unconditional_rate": safe_rate(known_correct, known_expected)},
        "known_wrong_enrolled_identity": known_wrong,
        "known_rejected_as_unknown": known_rejected,
        "known_no_decision": sum(1 for q in evaluated_queries if q.get("known") and q.get("no_decision")),
        "unknown_correctly_rejected": {"numerator": unknown_rejected, "denominator": unknown_expected,
                                       "conditional_rate": safe_rate(unknown_rejected, unknown_evaluated),
                                       "unconditional_rate": safe_rate(unknown_rejected, unknown_expected)},
        "unknown_false_accept": unknown_false_accept,
        "unknown_no_decision": sum(1 for q in evaluated_queries if not q.get("known") and q.get("no_decision")),
        "no_decision_total": no_decision,
        "not_evaluated_invalid_profile_or_mapping": max(0, expected - evaluated),
        "not_evaluated_query_rows_reported": not_evaluated_count,
        "category_counts": dict(categories),
    }


def aggregate_query_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    planned = evaluated = known_planned = unknown_planned = 0
    known_evaluated = unknown_evaluated = 0
    totals = collections.Counter()
    for row in rows:
        planned += row["planned_query_attempts"]
        evaluated += row["evaluated_query_count"]
        known_planned += row["known_planned"]
        unknown_planned += row["unknown_planned"]
        known_evaluated += row["known_evaluated_query_count"]
        unknown_evaluated += row["unknown_evaluated_query_count"]
        for key in (
            "known_correct_identification", "known_wrong_enrolled_identity",
            "known_rejected_as_unknown", "known_no_decision",
            "unknown_correctly_rejected", "unknown_false_accept",
            "unknown_no_decision", "no_decision_total",
        ):
            value = row[key]
            totals[key] += value["numerator"] if isinstance(value, dict) else value
    not_evaluated = planned - evaluated

    def metric(name: str, planned_denominator: int, evaluated_denominator: int) -> dict[str, Any]:
        numerator = totals[name]
        return {
            "numerator": numerator,
            "unconditional_denominator": planned_denominator,
            "unconditional_rate": safe_rate(numerator, planned_denominator),
            "conditional_denominator": evaluated_denominator,
            "conditional_rate": safe_rate(numerator, evaluated_denominator),
        }

    return {
        "planned_query_attempts": planned,
        "evaluated_query_attempts": evaluated,
        "not_evaluated_query_attempts": {"numerator": not_evaluated, "denominator": planned,
                                          "rate": safe_rate(not_evaluated, planned)},
        "known": {
            "planned_attempts": known_planned,
            "evaluated_attempts": known_evaluated,
            "correct_identification": metric("known_correct_identification", known_planned, known_evaluated),
            "wrong_enrolled_identity": metric("known_wrong_enrolled_identity", known_planned, known_evaluated),
            "rejected_as_unknown": metric("known_rejected_as_unknown", known_planned, known_evaluated),
            "no_decision": metric("known_no_decision", known_planned, known_evaluated),
        },
        "unknown": {
            "planned_attempts": unknown_planned,
            "evaluated_attempts": unknown_evaluated,
            "correctly_rejected": metric("unknown_correctly_rejected", unknown_planned, unknown_evaluated),
            "false_accept": metric("unknown_false_accept", unknown_planned, unknown_evaluated),
            "no_decision": metric("unknown_no_decision", unknown_planned, unknown_evaluated),
        },
        "all_queries": {
            "no_decision": metric("no_decision_total", planned, evaluated),
        },
    }


def candidate_selection(
    plan_trials: list[dict[str, Any]], result_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    order_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for trial in plan_trials:
        if trial.get("task09a_experiment_family") == "order_search":
            order_groups[trial["task09a_candidate_group_id"]].append(trial)
    selections = []
    selected_results = {}
    for group_id, trials in sorted(order_groups.items()):
        eligible = []
        attempts = []
        returned_attempts = 0
        for index, trial in enumerate(trials, 1):
            record = result_by_id.get(trial["trial_id"])
            returned_attempts += record is not None
            method, score = eligible_mapping(record) if record else (None, None)
            if record is None:
                eligibility_reason = "candidate result missing"
            elif not record.get("aosc_ready"):
                eligibility_reason = "AOSC profile not ready"
            elif method is not None:
                eligibility_reason = f"{method} map passed independent enrollment validation"
            elif not record.get("identity_mapping_valid") and not record.get("research_global_assignment", {}).get("mapping_accepted_for_research_comparison"):
                eligibility_reason = "strict map invalid and global assignment lacks unique independently supported evidence"
            else:
                eligibility_reason = "no mapping passed independent enrollment validation"
            attempts.append({
                "trial_id": trial["trial_id"],
                "candidate_variant": trial.get("task09a_candidate_variant"),
                "attempt_index": index,
                "aosc_ready": record.get("aosc_ready") if record else None,
                "strict_mapping_valid": record.get("identity_mapping_valid") if record else None,
                "strict_verification_passed": record.get("enrollment_validation", {}).get("strict_verification_passed") if record else None,
                "global_assignment_supported_and_verified": method == "global_assignment",
                "eligible_mapping_method": method,
                "eligibility_reason": eligibility_reason,
                "validation_margin_score": score,
                "selection_score_source": "equal-identity mean of assigned top1/top2 margins on held-out enrollment utterances",
            })
            if method is not None and score is not None:
                eligible.append((score, trial["trial_id"], method, record, index))
        complete_search = returned_attempts == len(trials)
        if complete_search and eligible:
            score, trial_id, method, record, attempts_used = sorted(
                eligible, key=lambda item: (-item[0], item[1]))[0]
            selected_results[group_id] = {"record": record, "method": method, "trial": next(t for t in trials if t["trial_id"] == trial_id)}
            reason = "eligible after independent enrollment validation; highest validation margin; lexicographic trial ID breaks exact ties"
        else:
            trial_id, method, score, attempts_used = None, None, None, len(trials)
            reason = (
                "abstain because the bounded candidate search is incomplete; every planned candidate must return before selection"
                if not complete_search else
                "abstain because no candidate was ready, had an evidence-supported map, and passed independent enrollment validation"
            )
        selections.append({
            "candidate_group_id": group_id,
            "candidate_attempt_budget": len(trials),
            "candidate_attempts_executed": returned_attempts,
            "selected_trial_id": trial_id,
            "selected_mapping_method": method,
            "selected_validation_margin_score": score,
            "selection_reason": reason,
            "attempts": attempts,
        })
    return selections, selected_results


def registration_metrics(
    selected: dict[str, dict[str, Any]],
    plan_trials: list[dict[str, Any]],
    profile_group_key: str = "task09a_candidate_group_id",
) -> dict[str, Any]:
    trial_by_group = {}
    for trial in plan_trials:
        if trial.get("task09a_experiment_family") == "order_search":
            trial_by_group.setdefault(trial[profile_group_key], trial)
    outcomes = collections.Counter()
    system = collections.Counter()
    attempts_used = []
    per_group = []
    for group_id, planned in sorted(trial_by_group.items()):
        chosen = selected.get(group_id)
        if chosen:
            record, method = chosen["record"], chosen["method"]
            outcome = "aosc_ready_map_valid_verification_passed"
            # The experiment evaluates the complete four-candidate fixed
            # budget before selecting by held-out validation margin.
            attempts_used.append(sum(
                1 for t in plan_trials
                if t.get(profile_group_key) == group_id
                and t.get("task09a_experiment_family") == "order_search"))
        else:
            record, method = None, None
            outcome = "registration_failed"
        outcomes[outcome] += 1
        planned_queries = planned.get("queries", [])
        counts = category_counts(record, method, planned_queries)
        all_known_correct = counts["known_correct_identification"]["numerator"] == counts["known_planned"]
        all_unknown_rejected = counts["unknown_correctly_rejected"]["numerator"] == counts["unknown_planned"]
        if not chosen:
            system["registration_failed"] += 1
        elif all_known_correct and all_unknown_rejected:
            system["successfully_registered_and_all_sampled_queries_correct"] += 1
        else:
            system["registered_but_subsequent_recognition_failed"] += 1
        if chosen and counts["unknown_false_accept"]:
            system["registered_but_unknown_false_accept_occurred"] += 1
        per_group.append({
            "candidate_group_id": group_id,
            "population": planned.get("population"),
            "group_size": planned.get("group_size"),
            "budget_target_sec_per_person": planned.get("enrollment_budget_speech_sec_per_person"),
            "selected_trial_id": chosen["trial"]["trial_id"] if chosen else None,
            "mapping_method": method,
            "registration_status": outcome,
            "query_metrics": counts,
        })
    total = len(trial_by_group)
    query_metrics = aggregate_query_metrics([item["query_metrics"] for item in per_group])
    return {
        "attempted_registration_groups": total,
        "selected_usable_profiles": {"numerator": len(selected), "denominator": total, "rate": safe_rate(len(selected), total)},
        "mean_candidate_attempts_per_usable_profile": statistics.mean(attempts_used) if attempts_used else None,
        "registration_outcome_counts": dict(outcomes),
        "system_level_outcomes": dict(system),
        "query_metrics": query_metrics,
        "per_registration_group": per_group,
    }


def ablation_summary(
    plan_trials: list[dict[str, Any]], result_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    grouped_controls = {
        trial["task09a_candidate_group_id"]: trial
        for trial in plan_trials
        if trial.get("task09a_experiment_family") == "order_search"
        and trial.get("task09a_candidate_variant") == "grouped"
    }
    for trial in plan_trials:
        if trial.get("task09a_experiment_family") in (
            "pause_ablation", "inter_speaker_pause_ablation", "utterance_selection"):
            groups[trial["task09a_matched_comparison_id"]].append(trial)
    output = []
    for matched_id, trials in sorted(groups.items()):
        candidates = []
        for trial in trials:
            record = result_by_id.get(trial["trial_id"])
            method, score = eligible_mapping(record) if record else (None, None)
            candidates.append({
                "trial_id": trial["trial_id"],
                "candidate_variant": trial.get("task09a_candidate_variant"),
                "gap_ms": trial.get("task09a_enrollment_gap_ms"),
                "inter_speaker_pause_only": trial.get("task09a_inter_speaker_pause_only", False),
                "aosc_ready": record.get("aosc_ready") if record else None,
                "strict_mapping_valid": record.get("identity_mapping_valid") if record else None,
                "strict_validation_passed": record.get("enrollment_validation", {}).get("strict_verification_passed") if record else None,
                "global_validation_passed": record.get("enrollment_validation", {}).get("research_global_verification_passed") if record else None,
                "eligible_mapping_method": method,
                "validation_margin_score": score,
                "enrollment_speech_sec": record.get("enrollment_speech_duration_sec") if record else None,
                "cache_rows_by_channel": record.get("cache_row_summary") if record else None,
                "strict_end_to_end_success": record.get("enrollment_validation", {}).get("strict_end_to_end_success") if record else None,
            })
        control = None
        control_group_id = None
        if matched_id.startswith("pause_"):
            control_group_id = matched_id[len("pause_"):]
        elif matched_id.startswith("inter_speaker_pause_"):
            control_group_id = matched_id[len("inter_speaker_pause_"):]
        if control_group_id in grouped_controls:
            control_trial = grouped_controls[control_group_id]
            control_record = result_by_id.get(control_trial["trial_id"])
            control = {
                "trial_id": control_trial["trial_id"],
                "candidate_variant": "Task 08 default 250 ms gap after every enrollment clip",
                "gap_ms": control_trial.get("task09a_enrollment_gap_ms", 250),
                "inter_speaker_pause_only": False,
                "aosc_ready": control_record.get("aosc_ready") if control_record else None,
                "strict_mapping_valid": control_record.get("identity_mapping_valid") if control_record else None,
                "strict_validation_passed": control_record.get("enrollment_validation", {}).get("strict_verification_passed") if control_record else None,
                "enrollment_speech_sec": control_record.get("enrollment_speech_duration_sec") if control_record else None,
                "cache_rows_by_channel": control_record.get("cache_row_summary") if control_record else None,
            }
        output.append({
            "matched_comparison_id": matched_id,
            "task08_default_per_clip_pause_control": control,
            "candidates": candidates,
        })
    return {"matched_ablation_groups": output}


def compact_query(query: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "query_id", "speaker_id", "known", "microphone", "duration_bucket",
        "actual_vad_speech_duration_sec", "activity_frames", "no_decision",
        "dominant_channel", "predicted_identity", "baseline_category",
        "research_global_assignment_category", "research_global_assignment_predicted_identity",
        "top1_dominance", "mean_top1_top2_margin", "channel_mean_probability_over_vad_frames",
        "registered_channel_means", "inference_ms",
    )
    return {key: query.get(key) for key in fields if key in query}


def grouped_baseline_outcomes(
    plan_trials: list[dict[str, Any]], result_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    counts = collections.Counter()
    for trial in plan_trials:
        if trial.get("task09a_experiment_family") != "order_search" or trial.get("task09a_candidate_variant") != "grouped":
            continue
        result = result_by_id.get(trial["trial_id"])
        if result is None:
            outcome = "not_returned"
        elif not result.get("aosc_ready"):
            outcome = "aosc_not_ready"
        elif not result.get("identity_mapping_valid"):
            outcome = "aosc_ready_identity_map_invalid"
        elif result.get("enrollment_validation", {}).get("strict_verification_passed"):
            outcome = "aosc_ready_identity_map_valid_verification_passed"
        else:
            outcome = "aosc_ready_identity_map_valid_verification_failed"
        counts[outcome] += 1
        groups[trial["task09a_candidate_group_id"]] = {
            "trial_id": trial["trial_id"], "outcome": outcome,
        }
    return {"counts": dict(counts), "by_registration_group": groups}


def alternate_mapping_failure_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        result = record.get("research_global_assignment") or {}
        candidate = result.get("candidate_identity_to_channel", {})
        strict = record.get("identity_to_channel", {})
        evidence = result.get("evidence_by_identity", {})
        supported = sum(bool(item.get("independent_enrollment_evidence_supported")) for item in evidence.values())
        collision_owners: dict[int, list[str]] = collections.defaultdict(list)
        for identity, stats in record.get("speaker_channel_establishment", {}).items():
            if stats.get("channel_established") and stats.get("candidate_channel") is not None:
                collision_owners[int(stats["candidate_channel"])].append(identity)
        collision_channels = {channel: ids for channel, ids in collision_owners.items() if len(ids) > 1}
        alternative_active_evidence = {}
        for channel, identities in collision_channels.items():
            for identity in identities:
                counts = record["speaker_channel_establishment"][identity].get("top1_frames_by_channel", [])
                alternatives = {str(index): int(count) for index, count in enumerate(counts)
                                if index != channel and int(count) > 0}
                if alternatives:
                    alternative_active_evidence[identity] = alternatives
        rows.append({
            "trial_id": record.get("trial_id"),
            "mapping_failure_cause": failure_taxonomy(record)["mapping_failure_cause"],
            "unique_global_optimum": result.get("unique_optimum", False),
            "global_assignment_supported": result.get("mapping_accepted_for_research_comparison", False),
            "global_assignment_differs_from_strict_candidates": any(
                identity in strict and strict[identity] != channel
                for identity, channel in candidate.items()),
            "identities_with_supported_assigned_channel": supported,
            "enrolled_identity_count": len(evidence),
            "shared_dominant_candidate_channels": {str(channel): ids for channel, ids in collision_channels.items()},
            "colliding_identities_with_non_candidate_active_top1_evidence": alternative_active_evidence,
            "collision_has_any_non_candidate_active_top1_evidence": bool(alternative_active_evidence),
        })
    by_cause: dict[str, dict[str, int]] = {}
    for cause in ("speaker_channel_collision", "channel_establishment_failure"):
        group = [row for row in rows if row["mapping_failure_cause"] == cause]
        by_cause[cause] = {
            "failed_profiles": len(group),
            "unique_global_optimum": sum(row["unique_global_optimum"] for row in group),
            "evidence_supported_global_assignment": sum(row["global_assignment_supported"] for row in group),
            "supported_assignment_differs_from_strict_candidate": sum(
                row["global_assignment_supported"] and row["global_assignment_differs_from_strict_candidates"]
                for row in group),
            "profiles_with_non_candidate_active_top1_evidence_among_colliding_identities": sum(
                row["collision_has_any_non_candidate_active_top1_evidence"] for row in group),
        }
    return {"by_mapping_failure_cause": by_cause, "profiles": rows}


def aosc_retention_by_enrollment_position(
    failed_profiles: list[dict[str, Any]], replay_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ratios: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    paired_first_minus_later: dict[int, list[float]] = collections.defaultdict(list)
    profile_events = collections.Counter()
    for baseline in failed_profiles:
        replay = replay_by_id.get(baseline["trial_id"])
        if replay is None:
            continue
        diagnostics = replay.get("task09a_enrollment_diagnostics") or {}
        if diagnostics.get("compression_source_identity_lineage_version", 0) < 2:
            continue
        order = baseline.get("enrollment_order", [])
        if not order:
            continue
        n = int(baseline["group_size"])
        events = diagnostics.get("aosc_compression_events", [])
        profile_events[len(events)] += 1
        for event in events:
            before = event.get("before", {}).get("source_identity_rows_and_channel_activity", {})
            after = event.get("after", {}).get("source_identity_rows_and_channel_activity", {})
            event_ratios = []
            for position, identity in enumerate(order):
                before_positive = sum(before.get(identity, {}).get("positive_rows_by_prediction_channel", []))
                after_positive = sum(after.get(identity, {}).get("positive_rows_by_prediction_channel", []))
                if before_positive <= 0:
                    continue
                ratio = after_positive / before_positive
                ratios[(n, position)].append(ratio)
                event_ratios.append((position, ratio))
            first = [ratio for position, ratio in event_ratios if position == 0]
            later = [ratio for position, ratio in event_ratios if position > 0]
            if first and later:
                paired_first_minus_later[n].append(first[0] - statistics.mean(later))
    return {
        "source": "corrected Stage 1 source-frame lineage; ratios compare positive cache prediction rows attributed to each enrolled speaker before and after each compression",
        "profile_count_by_compression_event_count": {str(key): value for key, value in sorted(profile_events.items())},
        "by_group_size_and_enrollment_position": {
            str(n): {
                str(position): {
                    "compression_identity_events_with_positive_input_rows": len(values),
                    "retention_ratio_mean": statistics.mean(values),
                    "retention_ratio_median": statistics.median(values),
                    "events_with_decrease": sum(value < 1.0 for value in values),
                    "events_with_no_decrease": sum(value >= 1.0 for value in values),
                }
                for (size, position), values in sorted(ratios.items()) if size == n
            }
            for n in sorted({size for size, _ in ratios})
        },
        "paired_earliest_minus_mean_later_retention_by_group_size": {
            str(n): {
                "compression_events_compared": len(values),
                "mean_difference": statistics.mean(values),
                "median_difference": statistics.median(values),
                "note": "positive indicates higher retention ratio for the first enrolled person; descriptive association, not an isolated causal estimate",
            }
            for n, values in sorted(paired_first_minus_later.items())
        },
    }


def candidate_attempt_reliability(records: list[dict[str, Any] | None]) -> dict[str, Any]:
    outcomes = collections.Counter()
    mapping_causes = collections.Counter()
    ready = mapping_valid = strict_verified = research_verified = usable = 0
    for record in records:
        if record is None:
            outcomes["candidate_not_returned"] += 1
            continue
        ready += bool(record.get("aosc_ready"))
        mapping_valid += bool(record.get("identity_mapping_valid"))
        validation = record.get("enrollment_validation", {})
        strict_pass = bool(validation.get("strict_verification_passed"))
        research_pass = bool(
            record.get("research_global_assignment", {}).get("mapping_accepted_for_research_comparison")
            and validation.get("research_global_verification_passed"))
        strict_verified += strict_pass
        research_verified += research_pass
        usable += strict_pass or research_pass
        outcomes[record.get("registration_outcome", "missing_registration_outcome")] += 1
        cause = failure_taxonomy(record)["mapping_failure_cause"]
        if cause:
            mapping_causes[cause] += 1
    total = len(records)
    return {
        "attempted_candidate_profiles": total,
        "returned_candidate_profiles": sum(record is not None for record in records),
        "aosc_ready": {"numerator": ready, "denominator": total, "rate": safe_rate(ready, total)},
        "strict_identity_map_valid": {"numerator": mapping_valid, "denominator": total, "rate": safe_rate(mapping_valid, total)},
        "strict_independent_enrollment_validation_pass": {"numerator": strict_verified, "denominator": total, "rate": safe_rate(strict_verified, total)},
        "evidence_guarded_global_map_and_validation_pass": {"numerator": research_verified, "denominator": total, "rate": safe_rate(research_verified, total)},
        "usable_profile_after_research_rule": {
            "numerator": usable,
            "denominator": total,
            "note": "count includes strict validation passes and guarded global-assignment validation passes; strict map failure remains visible in registration_outcome_counts",
        },
        "registration_outcome_counts": dict(outcomes),
        "mapping_failure_cause_counts": dict(mapping_causes),
    }


def candidate_search_cost_summary(
    plan_trials: list[dict[str, Any]], result_by_id: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for trial in plan_trials:
        if trial.get("task09a_experiment_family") == "order_search":
            groups[trial["task09a_candidate_group_id"]].append(trial)
    construction_by_group = []
    validation_inference_by_group = []
    search_by_group = []
    standard_by_group = []
    query_ms_all_candidates = 0.0
    profile_sizes = []
    attempts_per_group = []
    attempts_for_usable_profiles = []
    validation_wall_samples = []
    for group_id, trials in sorted(groups.items()):
        attempts_per_group.append(sum(trial["trial_id"] in result_by_id for trial in trials))
        build_ms = validation_ms = query_ms = 0.0
        for trial in trials:
            record = result_by_id.get(trial["trial_id"])
            if record is None:
                continue
            timing = record.get("timing", {})
            build_ms += float(timing.get("enrollment_construction_ms", 0.0))
            validation_ms += float(timing.get("strict_additional_enrollment_validation_ms", 0.0))
            validation_ms += float(timing.get("research_global_additional_enrollment_validation_ms", 0.0))
            query_ms += sum(float(query.get("inference_ms", 0.0)) for query in record.get("queries", []))
            if record.get("profile_size_bytes") is not None:
                profile_sizes.append(int(record["profile_size_bytes"]))
        construction_by_group.append(build_ms)
        validation_inference_by_group.append(validation_ms)
        search_by_group.append(build_ms + validation_ms)
        query_ms_all_candidates += query_ms
        choice = selected.get(group_id)
        if choice:
            attempts_for_usable_profiles.append(sum(
                1 for trial in trials if trial["trial_id"] in result_by_id))
            selected_timing = choice["record"].get("timing", {})
            selected_validation_ms = (
                selected_timing.get("strict_additional_enrollment_validation_ms", 0.0)
                if choice["method"] == "strict"
                else selected_timing.get("research_global_additional_enrollment_validation_ms", 0.0))
            standard_by_group.append(
                float(selected_timing.get("enrollment_construction_ms", 0.0)) +
                float(selected_validation_ms))
    for record in result_by_id.values():
        timing = record.get("timing", {})
        strict_wall = float(timing.get("strict_additional_enrollment_validation_wall_ms", 0.0))
        global_wall = float(timing.get("research_global_additional_enrollment_validation_wall_ms", 0.0))
        if strict_wall > 0.0 or global_wall > 0.0:
            validation_wall_samples.append({
                "trial_id": record.get("trial_id"),
                "group_size": record.get("group_size"),
                "strict_validation_wall_ms": strict_wall,
                "global_validation_wall_ms": global_wall,
            })
    median = lambda values: statistics.median(values) if values else None
    return {
        "candidate_attempt_budget_per_registration": 4,
        "timing_scope": "construction_ms includes enrollment asset reading, inference, compression, and profile encoding; validation values are model inference time only; process wall also includes query evaluation and setup",
        "mean_candidate_attempts_executed_per_registration": statistics.mean(attempts_per_group) if attempts_per_group else None,
        "mean_candidate_attempts_per_usable_profile": statistics.mean(attempts_for_usable_profiles) if attempts_for_usable_profiles else None,
        "candidate_construction_ms_total": sum(construction_by_group),
        "candidate_validation_inference_ms_total_including_global_diagnostics": sum(validation_inference_by_group),
        "median_four_candidate_construction_plus_validation_inference_ms": median(search_by_group),
        "median_selected_profile_construction_plus_validation_inference_ms": median(standard_by_group),
        "median_candidate_construction_ms_per_attempt": median([
            float(result_by_id[t["trial_id"]].get("timing", {}).get("enrollment_construction_ms", 0.0))
            for t in plan_trials if t.get("task09a_experiment_family") == "order_search" and t["trial_id"] in result_by_id]),
        "median_additional_validation_ms_per_attempt_including_global_diagnostics": median([
            float(result_by_id[t["trial_id"]].get("timing", {}).get("strict_additional_enrollment_validation_ms", 0.0)) +
            float(result_by_id[t["trial_id"]].get("timing", {}).get("research_global_additional_enrollment_validation_ms", 0.0))
            for t in plan_trials if t.get("task09a_experiment_family") == "order_search" and t["trial_id"] in result_by_id]),
        "median_global_assignment_calculation_ms_per_attempt": median([
            float(record.get("timing", {}).get("research_global_assignment_calculation_ms", 0.0))
            for record in result_by_id.values()
            if record.get("timing", {}).get("research_global_assignment_calculation_ms") is not None]),
        "instrumented_validation_wall_sample": {
            "profile_count": len(validation_wall_samples),
            "median_strict_validation_wall_ms": median([item["strict_validation_wall_ms"] for item in validation_wall_samples]),
            "median_global_validation_wall_ms": median([item["global_validation_wall_ms"] for item in validation_wall_samples]),
            "records": validation_wall_samples,
        },
        "median_profile_size_bytes": median(profile_sizes),
        "query_inference_ms_spent_on_all_candidate_evaluation_runs_not_used_for_candidate_selection": query_ms_all_candidates,
        "selection_reads_only_enrollment_validation_fields": True,
        "model_instance_reused_within_probe_process": True,
    }


def retention_validation_relationship(
    records: list[dict[str, Any]], plan_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {
        "profiles": set(), "argmax_rows": [], "positive_rows": [], "known_correct_rates": [],
        "unknown_false_accept_rates": [], "mapped_identities": 0, "zero_argmax_rows": 0,
    })
    ready_profiles = 0
    cache_frames = []
    occupied_channels = []
    by_size: dict[int, dict[str, Any]] = collections.defaultdict(lambda: {
        "planned": 0, "returned": 0, "ready": 0, "strict_map_valid": 0, "strict_validation_pass": 0,
        "cache_frames": [], "occupied_channels": [], "mapped_channel_rows": [],
        "zero_mapped_channel_rows": 0, "known_evaluated": 0, "known_correct": 0,
        "unknown_evaluated": 0, "unknown_false_accept": 0,
        "planned_known_queries": 0, "planned_unknown_queries": 0,
    })
    by_budget: dict[tuple[int, int], dict[str, Any]] = collections.defaultdict(lambda: {
        "planned": 0, "returned": 0, "ready": 0, "strict_map_valid": 0,
        "strict_validation_pass": 0, "cache_frames": [], "mapped_channel_rows": [],
        "actual_speech_sec_per_person": [],
    })
    for trial in plan_trials:
        if trial.get("task09a_experiment_family") != "order_search":
            continue
        bucket = by_size[int(trial["group_size"])]
        budget_bucket = by_budget[(int(trial["group_size"]),
                                   int(trial["enrollment_budget_speech_sec_per_person"]))]
        bucket["planned"] += 1
        budget_bucket["planned"] += 1
        bucket["planned_known_queries"] += sum(bool(item.get("known")) for item in trial.get("queries", []))
        bucket["planned_unknown_queries"] += sum(not bool(item.get("known")) for item in trial.get("queries", []))
    for record in records:
        group_size = int(record.get("group_size", 0))
        budget_target = int(record.get("enrollment_budget_speech_sec_per_person", 0))
        bucket_size = by_size[group_size]
        budget_bucket = by_budget[(group_size, budget_target)]
        bucket_size["returned"] += 1
        budget_bucket["returned"] += 1
        if group_size > 0 and record.get("enrollment_speech_duration_sec") is not None:
            budget_bucket["actual_speech_sec_per_person"].append(
                float(record["enrollment_speech_duration_sec"]) / group_size)
        if not record.get("aosc_ready"):
            continue
        ready_profiles += 1
        bucket_size["ready"] += 1
        budget_bucket["ready"] += 1
        row_summary = record.get("cache_row_summary") or {}
        if row_summary.get("cache_frames") is not None:
            cache_frames.append(int(row_summary["cache_frames"]))
            bucket_size["cache_frames"].append(int(row_summary["cache_frames"]))
            budget_bucket["cache_frames"].append(int(row_summary["cache_frames"]))
        argmax_rows = row_summary.get("argmax_rows_by_channel_gt_0_5", [])
        positive_rows = row_summary.get("positive_rows_by_channel_gt_0_5", [])
        occupied_channels.append(sum(int(value) > 0 for value in argmax_rows))
        bucket_size["occupied_channels"].append(sum(int(value) > 0 for value in argmax_rows))
        if not record.get("identity_mapping_valid"):
            continue
        bucket_size["strict_map_valid"] += 1
        budget_bucket["strict_map_valid"] += 1
        verification = record.get("enrollment_validation", {}).get("strict_mapping", {})
        per_identity = verification.get("per_identity", {})
        strict_pass = bool(record.get("enrollment_validation", {}).get("strict_verification_passed"))
        bucket_size["strict_validation_pass"] += strict_pass
        budget_bucket["strict_validation_pass"] += strict_pass
        label = "strict_validation_pass" if strict_pass else "strict_validation_fail"
        bucket = groups[label]
        bucket["profiles"].add(record.get("trial_id"))
        query_rows = record.get("queries", [])
        known = [q for q in query_rows if q.get("known") and q.get("query_evaluated")]
        unknown = [q for q in query_rows if not q.get("known") and q.get("query_evaluated")]
        if known:
            bucket["known_correct_rates"].append(
                sum(q.get("baseline_category") == "known_correct" for q in known) / len(known))
        if unknown:
            bucket["unknown_false_accept_rates"].append(
                sum(q.get("baseline_category") == "unknown_false_accept" for q in unknown) / len(unknown))
        for identity, channel in record.get("identity_to_channel", {}).items():
            if channel < 0 or channel >= len(argmax_rows):
                continue
            bucket["mapped_identities"] += 1
            row_count = int(argmax_rows[channel])
            bucket_size["mapped_channel_rows"].append(row_count)
            budget_bucket["mapped_channel_rows"].append(row_count)
            bucket_size["zero_mapped_channel_rows"] += row_count == 0
            bucket["argmax_rows"].append(row_count)
            bucket["positive_rows"].append(int(positive_rows[channel]) if channel < len(positive_rows) else 0)
            bucket["zero_argmax_rows"] += row_count == 0
            if strict_pass and identity in per_identity and not per_identity[identity].get("passed"):
                raise ValueError("strict per-identity enrollment validation disagrees with profile status")
        for query in query_rows:
            if not query.get("query_evaluated"):
                continue
            if query.get("known"):
                bucket_size["known_evaluated"] += 1
                bucket_size["known_correct"] += query.get("baseline_category") == "known_correct"
            else:
                bucket_size["unknown_evaluated"] += 1
                bucket_size["unknown_false_accept"] += query.get("baseline_category") == "unknown_false_accept"
    output = {}
    for label, group in groups.items():
        output[label] = {
            "profiles": len(group["profiles"]),
            "mapped_identities": group["mapped_identities"],
            "identity_channel_argmax_rows_median": statistics.median(group["argmax_rows"]) if group["argmax_rows"] else None,
            "identity_channel_argmax_rows_min": min(group["argmax_rows"]) if group["argmax_rows"] else None,
            "mapped_identity_channels_with_zero_argmax_rows": group["zero_argmax_rows"],
            "identity_channel_positive_rows_median": statistics.median(group["positive_rows"]) if group["positive_rows"] else None,
            "mean_profile_known_correct_query_rate": statistics.mean(group["known_correct_rates"]) if group["known_correct_rates"] else None,
            "mean_profile_unknown_false_accept_query_rate": statistics.mean(group["unknown_false_accept_rates"]) if group["unknown_false_accept_rates"] else None,
        }
    return {
        "ready_profiles": ready_profiles,
        "profile_cache_frames_median": statistics.median(cache_frames) if cache_frames else None,
        "profile_cache_frames_min": min(cache_frames) if cache_frames else None,
        "profile_cache_frames_max": max(cache_frames) if cache_frames else None,
        "occupied_predicted_channels_median": statistics.median(occupied_channels) if occupied_channels else None,
        "note": "Rows count positive AOSC prediction rows in the final bounded cache; associations use strict identity-to-channel maps. This is descriptive, not causal.",
        "strict_validation_groups": output,
        "by_enrolled_group_size": {
            str(size): {
                "planned_candidate_profiles": group["planned"],
                "returned_profile_records": group["returned"],
                "aosc_ready": {"numerator": group["ready"], "denominator": group["planned"],
                               "rate": safe_rate(group["ready"], group["planned"])},
                "strict_map_valid": {"numerator": group["strict_map_valid"], "denominator": group["planned"],
                                     "rate": safe_rate(group["strict_map_valid"], group["planned"])},
                "strict_validation_pass": {"numerator": group["strict_validation_pass"],
                                           "denominator": group["planned"],
                                           "rate": safe_rate(group["strict_validation_pass"], group["planned"]),
                                           "rate_among_strict_maps": safe_rate(group["strict_validation_pass"], group["strict_map_valid"])},
                "profile_cache_rows_median": statistics.median(group["cache_frames"]) if group["cache_frames"] else None,
                "occupied_prediction_channels_median": statistics.median(group["occupied_channels"]) if group["occupied_channels"] else None,
                "mapped_identity_channel_argmax_rows_median": statistics.median(group["mapped_channel_rows"]) if group["mapped_channel_rows"] else None,
                "mapped_identity_channel_argmax_rows_min": min(group["mapped_channel_rows"]) if group["mapped_channel_rows"] else None,
                "mapped_identity_channels_with_zero_argmax_rows": group["zero_mapped_channel_rows"],
                "known_correct_identification": {
                    "numerator": group["known_correct"],
                    "unconditional_denominator": group["planned_known_queries"],
                    "unconditional_rate": safe_rate(group["known_correct"], group["planned_known_queries"]),
                    "conditional_denominator": group["known_evaluated"],
                    "conditional_rate": safe_rate(group["known_correct"], group["known_evaluated"])},
                "unknown_false_accept": {
                    "numerator": group["unknown_false_accept"],
                    "unconditional_denominator": group["planned_unknown_queries"],
                    "unconditional_rate": safe_rate(group["unknown_false_accept"], group["planned_unknown_queries"]),
                    "conditional_denominator": group["unknown_evaluated"],
                    "conditional_rate": safe_rate(group["unknown_false_accept"], group["unknown_evaluated"])},
            }
            for size, group in sorted(by_size.items())
        },
        "by_group_size_and_construction_budget_seconds_per_person": {
            f"n{size}_e{budget}s": {
                "planned_candidate_profiles": group["planned"],
                "returned_profile_records": group["returned"],
                "aosc_ready_profiles": group["ready"],
                "strict_map_valid_profiles": group["strict_map_valid"],
                "strict_validation_pass_profiles": group["strict_validation_pass"],
                "profile_cache_rows_median": statistics.median(group["cache_frames"]) if group["cache_frames"] else None,
                "actual_speech_seconds_per_person_median": statistics.median(group["actual_speech_sec_per_person"]) if group["actual_speech_sec_per_person"] else None,
                "mapped_identity_channel_argmax_rows_median": statistics.median(group["mapped_channel_rows"]) if group["mapped_channel_rows"] else None,
                "mapped_identity_channel_argmax_rows_min": min(group["mapped_channel_rows"]) if group["mapped_channel_rows"] else None,
            }
            for (size, budget), group in sorted(by_budget.items())
        },
    }


def task08_matched_cohort_control(
    plan_trials: list[dict[str, Any]], task08_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in task08_records:
        if record.get("enrollment_order_variant") != "randomized":
            continue
        key = (
            record.get("population"), record.get("cohort_id"), record.get("group_size"),
            record.get("enrollment_budget_speech_sec_per_person"),
        )
        baseline_by_key[key] = record
    controls = []
    outcomes = collections.Counter()
    mapping_causes = collections.Counter()
    system = collections.Counter()
    query_rows = []
    by_population: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {
        "attempted": 0, "ready": 0, "map_valid": 0, "ready_and_map_valid": 0,
        "outcomes": collections.Counter(), "system": collections.Counter(), "query_rows": [],
    })
    missing_profiles = []
    missing_query_ids = []
    ready = map_valid = both = 0
    for trial in plan_trials:
        if trial.get("task09a_experiment_family") != "order_search" or trial.get("task09a_candidate_variant") != "grouped":
            continue
        key = (trial["population"], trial["cohort_id"], trial["group_size"],
               trial["enrollment_budget_speech_sec_per_person"])
        record = baseline_by_key.get(key)
        if record is None:
            missing_profiles.append({"candidate_group_id": trial["task09a_candidate_group_id"], "task08_key": list(key)})
            outcome = "not_matched_task08_profile"
            query_result = None
        else:
            is_ready = bool(record.get("aosc_ready"))
            is_mapped = bool(record.get("identity_mapping_valid"))
            ready += is_ready
            map_valid += is_mapped
            both += is_ready and is_mapped
            if not is_ready:
                outcome = "aosc_not_ready"
            elif not is_mapped:
                outcome = "aosc_ready_identity_map_invalid"
                cause = failure_taxonomy(record)["mapping_failure_cause"]
                if cause:
                    mapping_causes[cause] += 1
            else:
                outcome = "aosc_ready_identity_map_valid_verification_not_provided"
            expected_ids = {item["query_id"] for item in trial.get("queries", [])}
            observed_ids = {item.get("query_id") for item in record.get("queries", [])}
            missing = sorted(expected_ids - observed_ids)
            if missing:
                missing_query_ids.extend({"candidate_group_id": trial["task09a_candidate_group_id"], "query_id": query_id} for query_id in missing)
            query_result = dict(record)
            query_result["queries"] = [q for q in record.get("queries", []) if q.get("query_id") in expected_ids]
        outcomes[outcome] += 1
        pop_bucket = by_population[trial["population"]]
        pop_bucket["attempted"] += 1
        pop_bucket["ready"] += bool(record and record.get("aosc_ready"))
        pop_bucket["map_valid"] += bool(record and record.get("identity_mapping_valid"))
        pop_bucket["ready_and_map_valid"] += bool(
            record and record.get("aosc_ready") and record.get("identity_mapping_valid"))
        pop_bucket["outcomes"][outcome] += 1
        metrics = category_counts(query_result, "strict" if query_result is not None else None,
                                  trial.get("queries", []))
        registration_valid = bool(record and record.get("aosc_ready") and record.get("identity_mapping_valid"))
        if not registration_valid:
            system["registration_failed"] += 1
            pop_bucket["system"]["registration_failed"] += 1
        else:
            all_known_correct = (
                metrics["known_correct_identification"]["numerator"] == metrics["known_planned"]
                and metrics["known_wrong_enrolled_identity"] == 0
                and metrics["known_rejected_as_unknown"] == 0
                and metrics["known_no_decision"] == 0
            )
            all_unknown_rejected = (
                metrics["unknown_correctly_rejected"]["numerator"] == metrics["unknown_planned"]
                and metrics["unknown_false_accept"] == 0
                and metrics["unknown_no_decision"] == 0
            )
            if all_known_correct and all_unknown_rejected:
                system["successfully_registered_and_all_sampled_queries_correct"] += 1
                pop_bucket["system"]["successfully_registered_and_all_sampled_queries_correct"] += 1
            else:
                system["registered_but_subsequent_recognition_failed"] += 1
                pop_bucket["system"]["registered_but_subsequent_recognition_failed"] += 1
            if metrics["unknown_false_accept"]:
                system["registered_but_unknown_false_accept_occurred"] += 1
                pop_bucket["system"]["registered_but_unknown_false_accept_occurred"] += 1
        query_rows.append(metrics)
        pop_bucket["query_rows"].append(metrics)
        controls.append({
            "candidate_group_id": trial["task09a_candidate_group_id"],
            "task08_trial_id": record.get("trial_id") if record else None,
            "aosc_ready": record.get("aosc_ready") if record else None,
            "identity_mapping_valid": record.get("identity_mapping_valid") if record else None,
            "identity_mapping_error": record.get("identity_mapping_error") if record else None,
            "profile_crc32": record.get("profile_crc32") if record else None,
            "query_metrics_on_task09a_matched_query_ids": metrics,
        })
    return {
        "comparison_scope": "same population, cohort, group size, requested construction budget, randomized Task 08 profile, and matched Task 09A query IDs; Task 09A excludes enrollment-validation utterances from profile construction",
        "attempted_matched_registration_groups": len(controls),
        "task08_aosc_ready": {"numerator": ready, "denominator": len(controls), "rate": safe_rate(ready, len(controls))},
        "task08_identity_map_valid": {"numerator": map_valid, "denominator": len(controls), "rate": safe_rate(map_valid, len(controls))},
        "task08_aosc_ready_and_map_valid": {"numerator": both, "denominator": len(controls), "rate": safe_rate(both, len(controls))},
        "registration_outcome_counts": dict(outcomes),
        "system_level_outcomes": dict(system),
        "mapping_failure_cause_counts": dict(mapping_causes),
        "query_metrics": aggregate_query_metrics(query_rows),
        "by_population": {
            population: {
                "attempted_matched_registration_groups": values["attempted"],
                "task08_aosc_ready": {"numerator": values["ready"], "denominator": values["attempted"],
                                       "rate": safe_rate(values["ready"], values["attempted"])},
                "task08_identity_map_valid": {"numerator": values["map_valid"], "denominator": values["attempted"],
                                              "rate": safe_rate(values["map_valid"], values["attempted"])},
                "task08_aosc_ready_and_map_valid": {
                    "numerator": values["ready_and_map_valid"], "denominator": values["attempted"],
                    "rate": safe_rate(values["ready_and_map_valid"], values["attempted"])},
                "registration_outcome_counts": dict(values["outcomes"]),
                "system_level_outcomes": dict(values["system"]),
                "query_metrics": aggregate_query_metrics(values["query_rows"]),
            }
            for population, values in sorted(by_population.items())
        },
        "missing_task08_profiles": missing_profiles,
        "missing_matched_query_ids": missing_query_ids,
        "per_registration_group": controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task08-results", required=True, type=Path)
    parser.add_argument("--task08-aggregate", required=True, type=Path)
    parser.add_argument("--stage1-results", required=True, type=Path)
    parser.add_argument("--candidate-plan", type=Path)
    parser.add_argument("--candidate-results", type=Path)
    parser.add_argument("--candidate-run-events", type=Path)
    parser.add_argument("--stage1-run-events", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw_task08 = read_jsonl(args.task08_results)
    task08_aggregate = read_json(args.task08_aggregate)
    stage1 = read_jsonl(args.stage1_results)
    stage1_by_id = {item["trial_id"]: item for item in stage1}
    failures = [item for item in raw_task08 if not item.get("aosc_ready") or not item.get("identity_mapping_valid")]
    failure_records = []
    for original in failures:
        replay = stage1_by_id.get(original["trial_id"], original)
        failure_records.append({
            "trial_id": original["trial_id"],
            "population": original.get("population"),
            "cohort_id": original.get("cohort_id"),
            "group_size": original.get("group_size"),
            "budget_target_sec_per_person": original.get("enrollment_budget_speech_sec_per_person"),
            "enrollment_order_variant": original.get("enrollment_order_variant"),
            **failure_taxonomy(replay),
        })
    output: dict[str, Any] = {
        "schema": "nemo-speech.voiceid-task09a-enrollment-report.v1",
        "task08_baseline": task08_baseline(raw_task08, task08_aggregate),
        "stage1_exact_failure_replay": reproduced_baseline(raw_task08, stage1),
        "stage1_failed_profile_taxonomy": failure_records,
        "stage1_failure_cause_counts": dict(collections.Counter(x["primary_cause"] for x in failure_records)),
        "stage1_mapping_failure_cause_counts": dict(collections.Counter(
            x["mapping_failure_cause"] for x in failure_records if x["mapping_failure_cause"])),
        "stage1_alternative_mapping_analysis": alternate_mapping_failure_analysis(
            [stage1_by_id[item["trial_id"]] for item in failures if item["trial_id"] in stage1_by_id]),
        "stage1_aosc_retention_by_enrollment_order_position": aosc_retention_by_enrollment_position(
            failures, stage1_by_id),
        "stage1_failed_profile_examples": {
            "first_channel_collision": next((x for x in failure_records if x["primary_cause"] == "speaker_channel_collision"), None),
            "first_channel_establishment_failure": next((x for x in failure_records if x["primary_cause"] == "channel_establishment_failure"), None),
            "first_aosc_not_ready": next((x for x in failure_records if x["primary_cause"] == "aosc_not_ready"), None),
        },
        "dataset_and_leakage_note": "Task 08 expanded manifest SHA-256 be9928d3af71dfde838d950165896105a06a4dbc7b7cff41ae751066e5b25fae; Task 09A candidates use only its enrollment-partition audio for construction and validation. Development and calibration populations select the rule; final-holdout reuse is follow-up analysis because Task 08 results were already inspected.",
        "stage1_execution_run_events": read_jsonl(args.stage1_run_events) if args.stage1_run_events else [],
    }

    if args.candidate_plan and args.candidate_results:
        plan = read_json(args.candidate_plan)
        result_rows = read_jsonl(args.candidate_results)
        result_by_id = {item["trial_id"]: item for item in result_rows}
        selections, selected = candidate_selection(plan["trials"], result_by_id)
        partition_rows = {}
        for population in ("development", "calibration", "final_holdout"):
            groups = {
                group_id: value for group_id, value in selected.items()
                if value["trial"].get("population") == population
            }
            partition_plan = [trial for trial in plan["trials"] if trial.get("population") == population]
            partition_rows[population] = registration_metrics(groups, partition_plan)
        result_manifest_fingerprints = collections.Counter(
            item.get("manifest_fingerprint_fnv1a64")
            for item in result_rows if item.get("manifest_fingerprint_fnv1a64") is not None)
        # The baseline member of each repair search is the grouped candidate;
        # its profile and queries are compared on exactly the same cohort.
        baseline_selected = {}
        for trial in plan["trials"]:
            if trial.get("task09a_experiment_family") != "order_search" or trial.get("task09a_candidate_variant") != "grouped":
                continue
            record = result_by_id.get(trial["trial_id"])
            if record and record.get("aosc_ready") and record.get("identity_mapping_valid") and record.get("enrollment_validation", {}).get("strict_verification_passed"):
                baseline_selected[trial["task09a_candidate_group_id"]] = {"record": record, "method": "strict", "trial": trial}
        output["stage2_candidate_suite"] = {
            "candidate_plan_sha256": __import__("hashlib").sha256(args.candidate_plan.read_bytes()).hexdigest(),
            "candidate_result_manifest_fingerprint_counts": dict(result_manifest_fingerprints),
            "execution_run_events": read_jsonl(args.candidate_run_events) if args.candidate_run_events else [],
            "attempted_candidate_profiles": len(plan["trials"]),
            "returned_candidate_profiles": len(result_rows),
            "fixed_search_rule": "Four order candidates per registration. Every planned candidate must return before selection; otherwise abstain. A candidate is eligible only when AOSC is ready and either the strict mapping or evidence-guarded maximum-weight mapping passes independent held-out enrollment validation. Prefer strict mapping if both pass. Select the eligible candidate with the largest equal-identity mean assigned-channel top1/top2 margin on validation utterances; break exact ties by trial ID. Exhaust the bounded set and abstain when none passes.",
            "candidate_selection": selections,
            "candidate_attempt_reliability_by_population": {
                population: candidate_attempt_reliability([
                    result_by_id.get(t["trial_id"]) for t in plan["trials"]
                    if t.get("population") == population
                ])
                for population in ("development", "calibration", "final_holdout")
            },
            "automatic_repair_cost": candidate_search_cost_summary(
                plan["trials"], result_by_id, selected),
            "aosc_retained_evidence_vs_validation_and_query_quality": retention_validation_relationship(
                [result_by_id[t["trial_id"]] for t in plan["trials"]
                 if t.get("task09a_experiment_family") == "order_search" and t["trial_id"] in result_by_id],
                plan["trials"]),
            "grouped_baseline_registration_outcomes": {
                population: grouped_baseline_outcomes(
                    [trial for trial in plan["trials"] if trial.get("population") == population], result_by_id)
                for population in ("development", "calibration", "final_holdout")
            },
            "auto_repair_outcomes_by_partition": partition_rows,
            "grouped_order_strict_mapping_control_by_partition": {
                population: registration_metrics(
                    {key: val for key, val in baseline_selected.items()
                     if val["trial"].get("population") == population},
                    [t for t in plan["trials"] if t.get("population") == population and t.get("task09a_experiment_family") == "order_search"])
                for population in ("development", "calibration", "final_holdout")
            },
            "matched_pause_and_utterance_selection_ablations": ablation_summary(plan["trials"], result_by_id),
            "task08_matched_cohort_control": task08_matched_cohort_control(
                plan["trials"], raw_task08),
            "per_candidate_status": [
                {
                    "trial_id": trial["trial_id"],
                    "manifest_fingerprint_fnv1a64": result_by_id.get(trial["trial_id"], {}).get("manifest_fingerprint_fnv1a64"),
                    "candidate_group_id": trial.get("task09a_candidate_group_id"),
                    "candidate_variant": trial.get("task09a_candidate_variant"),
                    "experiment_family": trial.get("task09a_experiment_family"),
                    "population": trial.get("population"),
                    "group_size": trial.get("group_size"),
                    "budget_target_sec_per_person": trial.get("enrollment_budget_speech_sec_per_person"),
                    "gap_ms": trial.get("task09a_enrollment_gap_ms"),
                    "inter_speaker_pause_only": trial.get("task09a_inter_speaker_pause_only", False),
                    "aosc_ready": result_by_id.get(trial["trial_id"], {}).get("aosc_ready"),
                    "strict_mapping_valid": result_by_id.get(trial["trial_id"], {}).get("identity_mapping_valid"),
                    "registration_outcome": result_by_id.get(trial["trial_id"], {}).get("registration_outcome"),
                    "strict_verification_passed": result_by_id.get(trial["trial_id"], {}).get("enrollment_validation", {}).get("strict_verification_passed"),
                    "global_assignment_supported_and_verified": result_by_id.get(trial["trial_id"], {}).get("enrollment_validation", {}).get("research_global_verification_passed"),
                    "actual_construction_speech_sec": result_by_id.get(trial["trial_id"], {}).get("enrollment_speech_duration_sec"),
                    "profile_bytes": result_by_id.get(trial["trial_id"], {}).get("profile_size_bytes"),
                    "timing": result_by_id.get(trial["trial_id"], {}).get("timing"),
                    "speaker_channel_establishment": result_by_id.get(trial["trial_id"], {}).get("speaker_channel_establishment"),
                    "research_global_assignment": result_by_id.get(trial["trial_id"], {}).get("research_global_assignment"),
                    "enrollment_validation": result_by_id.get(trial["trial_id"], {}).get("enrollment_validation"),
                    "cache_row_summary": result_by_id.get(trial["trial_id"], {}).get("cache_row_summary"),
                    "queries": [compact_query(q) for q in result_by_id.get(trial["trial_id"], {}).get("queries", [])],
                }
                for trial in plan["trials"]
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    suite = output.get("stage2_candidate_suite", {})
    print(json.dumps({
        "output": str(args.output),
        "task08_profiles": len(raw_task08),
        "stage1_replays": len(stage1),
        "stage2_candidate_profiles_attempted": suite.get("attempted_candidate_profiles", 0),
        "stage2_candidate_profiles_returned": suite.get("returned_candidate_profiles", 0),
    }, indent=2))


if __name__ == "__main__":
    main()
