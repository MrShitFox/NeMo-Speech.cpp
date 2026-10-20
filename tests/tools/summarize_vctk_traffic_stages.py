#!/usr/bin/env python3
"""Aggregate staged known-query probes after successive unknown AOSC updates."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def fnv64(value: str) -> str:
    result = 1469598103934665603
    for byte in value.encode():
        result = ((result ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest_text = args.manifest.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    expected = fnv64(manifest_text)
    trial_by_id = {trial["trial_id"]: trial for trial in manifest["trials"]}
    raw = read_jsonl(args.results)
    reference_by_id = {
        row["trial_id"]: row for row in read_jsonl(args.reference_results)
    } if args.reference_results else {}
    if any(row.get("manifest_fingerprint_fnv1a64") != expected for row in reference_by_id.values()):
        raise ValueError("reference results belong to a different manifest")
    profiles = []
    for row in raw:
        trial_id = row.get("trial_id")
        if trial_id not in trial_by_id:
            raise ValueError(f"result references trial absent from manifest: {trial_id}")
        if row.get("manifest_fingerprint_fnv1a64") != expected:
            raise ValueError(f"result belongs to a different manifest: {trial_id}")
        if row.get("evaluation_mode") != "unknown_traffic_only":
            raise ValueError(f"trial was not run with --unknown-traffic-only: {trial_id}")
        trial = trial_by_id[trial_id]
        reference = reference_by_id.get(trial_id)
        if args.reference_results and reference is None:
            raise ValueError(f"trial is missing from reference results: {trial_id}")
        traffic = row.get("unknown_traffic") or {}
        if not row.get("aosc_ready") or not row.get("identity_mapping_valid"):
            profiles.append({
                "case_id": None,
                "trial_id": trial_id,
                "population": trial["population"],
                "group_size": trial["group_size"],
                "budget_sec": trial["enrollment_budget_speech_sec_per_person"],
                "order_variant": trial["enrollment_order_variant"],
                "status": "not_evaluated_profile_or_mapping_invalid",
                "aosc_ready": row.get("aosc_ready"),
                "identity_mapping_valid": row.get("identity_mapping_valid"),
                "identity_mapping_error": row.get("identity_mapping_error"),
            })
            continue
        if traffic.get("status") != "complete":
            raise ValueError(f"valid profile did not complete staged traffic: {trial_id}")
        if traffic.get("immutable_enrollment_profile_preserved") is not True:
            raise ValueError(f"permanent enrollment profile changed during traffic: {trial_id}")
        if row.get("profile_immutable_after_evaluation") is not True:
            raise ValueError(f"saved enrollment profile changed during evaluation: {trial_id}")
        baseline = traffic.get("independent_fresh_session_probe") or {}
        if not baseline.get("predicted_identity"):
            raise ValueError(f"missing clean-profile identity decision: {trial_id}")
        stages = traffic.get("stages", [])
        if len(stages) != len(trial.get("unknown_traffic_query_ids", [])):
            raise ValueError(f"staged probe count disagrees with traffic sequence: {trial_id}")
        stage_rows = []
        for stage in stages:
            probe = stage.get("known_return_from_fresh_import_of_accumulated_aosc") or {}
            if not probe:
                raise ValueError(f"missing known-return stage probe: {trial_id}")
            stage_rows.append({
                "after_unknown_speakers": stage["after_unknown_speakers"],
                "temporary_profile_changed": stage["temporary_profile_changed"],
                "cache_rows": stage["cache_rows"],
                "known_return_category": probe.get("baseline_category"),
                "identity_decision_changed_from_clean_profile": (
                    probe.get("predicted_identity") != baseline.get("predicted_identity")
                ),
                "activity_frames": probe.get("activity_frames"),
                "top1_dominance": probe.get("top1_dominance"),
                "mean_top1_top2_margin": probe.get("mean_top1_top2_margin"),
                "diagnostic_max_registered_channel_mean": probe.get("diagnostic_max_registered_channel_mean"),
                "profile_import_ms": probe.get("profile_import_ms"),
                "inference_ms": probe.get("inference_ms"),
            })
        profiles.append({
            "case_id": None,
            "trial_id": trial_id,
            "population": trial["population"],
            "group_size": trial["group_size"],
            "budget_sec": trial["enrollment_budget_speech_sec_per_person"],
            "order_variant": trial["enrollment_order_variant"],
            "status": "complete",
            "identity_mapping_matches_full_run": (
                row.get("identity_to_channel") == reference.get("identity_to_channel")
                if reference else None
            ),
            "profile_crc32_matches_full_run": (
                row.get("profile_crc32") == reference.get("profile_crc32")
                if reference else None
            ),
            "unknown_speaker_count": traffic.get("unknown_speaker_count"),
            "clean_profile_category": baseline.get("baseline_category"),
            "continuous_return_category_after_all_unknowns": traffic.get("known_return_category"),
            "continuous_return_matches_clean_category": (
                traffic.get("known_return_category") == baseline.get("baseline_category")
            ),
            "permanent_profile_immutable": traffic.get("immutable_enrollment_profile_preserved"),
            "stages": stage_rows,
        })

    profiles.sort(key=lambda row: (
        row["population"], row["group_size"], row["budget_sec"], row["order_variant"], row["trial_id"]
    ))
    for index, profile in enumerate(profiles, 1):
        profile["case_id"] = f"TR{index:02d}"
        profile.pop("trial_id", None)

    by_stage: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    complete_profiles = [row for row in profiles if row["status"] == "complete"]
    for profile in complete_profiles:
        for stage in profile["stages"]:
            by_stage[stage["after_unknown_speakers"]].append(stage)

    stage_summary = []
    for stage_index, items in sorted(by_stage.items()):
        categories = collections.Counter(row.get("known_return_category") for row in items)
        stage_summary.append({
            "after_unknown_speakers": stage_index,
            "evaluated_known_returns": len(items),
            "category_counts": dict(sorted(categories.items())),
            "identity_decisions_changed_from_clean_profile": sum(
                bool(row["identity_decision_changed_from_clean_profile"]) for row in items
            ),
            "temporary_profiles_changed": sum(bool(row["temporary_profile_changed"]) for row in items),
            "median_max_registered_channel_mean": statistics.median(
                [
                row["diagnostic_max_registered_channel_mean"] for row in items
                if row.get("diagnostic_max_registered_channel_mean") is not None
                ]
            )
            if any(row.get("diagnostic_max_registered_channel_mean") is not None for row in items) else None,
        })

    output = {
        "schema": "nemo-speech.voiceid-task08-traffic-stages.v1",
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "manifest_fingerprint_fnv1a64": expected,
        "raw_result_records": len(raw),
        "profiles_with_stage_evaluation": len(complete_profiles),
        "profiles_not_evaluated": len(profiles) - len(complete_profiles),
        "profiles_matching_full_run_mapping": sum(
            row.get("identity_mapping_matches_full_run") is True for row in complete_profiles
        ) if args.reference_results else None,
        "profiles_matching_full_run_profile_crc32": sum(
            row.get("profile_crc32_matches_full_run") is True for row in complete_profiles
        ) if args.reference_results else None,
        "unknown_speakers_per_sequence": sorted({
            row["unknown_speaker_count"] for row in complete_profiles
        }),
        "per_stage": stage_summary,
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "profiles_with_stage_evaluation": output["profiles_with_stage_evaluation"],
        "profiles_not_evaluated": output["profiles_not_evaluated"],
        "per_stage": output["per_stage"],
    }, indent=2))


if __name__ == "__main__":
    main()
