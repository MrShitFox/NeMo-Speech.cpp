#!/usr/bin/env python3
"""Build bounded Task 09A enrollment protocol and repair-search candidates.

The planner reuses only the Task 08 enrollment utterances and evaluation
queries. It reserves separate enrollment utterances for profile validation and
emits deterministic, matched order/gap/selection variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ORDER_VARIANTS = ("grouped", "reverse_groups", "round_robin", "seeded_shuffle")
GROUP_BUDGETS = (15, 30, 45, 60)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def choose_validation_clip(clips: list[dict[str, Any]]) -> dict[str, Any]:
    if not clips:
        raise ValueError("each enrolled identity must have at least one enrollment clip")
    # Selection depends only on manifest speech duration and utterance ID, not
    # model outputs or evaluation queries.
    return sorted(clips, key=lambda c: (-float(c["speech_duration_sec"]), c["utterance_id"]))[0]


def grouped_clips(
    speaker_ids: list[str], clips: list[dict[str, Any]], order: list[str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    by_speaker: dict[str, list[dict[str, Any]]] = {speaker: [] for speaker in speaker_ids}
    for clip in clips:
        speaker = clip["speaker_id"]
        if speaker not in by_speaker:
            raise ValueError(f"enrollment clip speaker {speaker} is outside enrolled_speaker_ids")
        if clip.get("source_partition") != "enrollment":
            raise ValueError(f"candidate audio is not from enrollment partition: {clip['utterance_id']}")
        by_speaker[speaker].append(clip)
    for speaker, items in by_speaker.items():
        items.sort(key=lambda c: (c.get("clip_index", 0), c["utterance_id"]))
        if not items:
            raise ValueError(f"no enrollment clips for {speaker}")
    return by_speaker


def make_training_and_validation(
    speaker_ids: list[str], clips: list[dict[str, Any]], budget: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_speaker = grouped_clips(speaker_ids, clips)
    construction: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for speaker in speaker_ids:
        source_clips = by_speaker[speaker]
        held = choose_validation_clip(source_clips)
        validation.append(held)
        candidates = [clip for clip in source_clips if clip["utterance_id"] != held["utterance_id"]]
        total = 0.0
        for clip in candidates:
            construction.append(clip)
            total += float(clip["speech_duration_sec"])
            if total >= budget:
                break
    return construction, validation


def order_clips(
    clips: list[dict[str, Any]], speaker_ids: list[str], policy: str, seed: int
) -> list[dict[str, Any]]:
    by_speaker = grouped_clips(speaker_ids, clips)
    if policy == "grouped":
        return [clip for speaker in speaker_ids for clip in by_speaker[speaker]]
    if policy == "reverse_groups":
        return [clip for speaker in reversed(speaker_ids) for clip in by_speaker[speaker]]
    if policy == "round_robin":
        result: list[dict[str, Any]] = []
        depth = max(map(len, by_speaker.values()))
        for index in range(depth):
            for speaker in speaker_ids:
                if index < len(by_speaker[speaker]):
                    result.append(by_speaker[speaker][index])
        return result
    if policy == "seeded_shuffle":
        result = list(clips)
        random.Random(seed).shuffle(result)
        return result
    raise ValueError(f"unknown enrollment order policy: {policy}")


def selected_queries(trial: dict[str, Any], speaker_ids: list[str]) -> list[dict[str, Any]]:
    queries = [q for q in trial["queries"] if q.get("mic") == "mic1"]
    result: list[dict[str, Any]] = []
    for speaker in speaker_ids:
        matches = [q for q in queries if q.get("known") and q.get("speaker_id") == speaker]
        if matches:
            result.append(sorted(matches, key=lambda q: (q.get("duration_bucket", ""), q["query_id"]))[0])
    unknowns: dict[str, dict[str, Any]] = {}
    for query in sorted(queries, key=lambda q: (q.get("speaker_id", ""), q["query_id"])):
        if not query.get("known"):
            unknowns.setdefault(query["speaker_id"], query)
    result.extend(list(unknowns.values())[:4])
    return result


def candidate_record(
    source: dict[str, Any],
    trial_id: str,
    group_id: str,
    variant: str,
    construction: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    gap_ms: int = 250,
    family: str = "order_search",
    matched_id: str | None = None,
    budget: int | None = None,
) -> dict[str, Any]:
    speaker_ids = list(source["enrollment_order"])
    ordered = order_clips(construction, speaker_ids, variant if variant in ORDER_VARIANTS else "grouped", source["seed"] + 91009)
    out = dict(source)
    out.update(
        {
            "trial_id": trial_id,
            "cohort_id": source["cohort_id"],
            "evaluation_partition": source["evaluation_partition"],
            "enrolled_speaker_ids": speaker_ids,
            "enrollment_order": list(dict.fromkeys(c["speaker_id"] for c in ordered)),
            "enrollment_order_variant": variant,
            "enrollment": ordered,
            "validation_enrollment": validation,
            "queries": queries,
            "conversations": [],
            "run_unknown_traffic": False,
            "unknown_traffic_query_ids": [],
            "task09a_candidate_group_id": group_id,
            "task09a_candidate_variant": variant,
            "task09a_experiment_family": family,
            "task09a_enrollment_gap_ms": gap_ms,
            "task09a_matched_comparison_id": matched_id or group_id,
            "task09a_candidate_attempt_limit": len(ORDER_VARIANTS) if family == "order_search" else 1,
        }
    )
    if budget is not None:
        out["enrollment_budget_speech_sec_per_person"] = budget
        out["task09a_construction_budget_target_sec_per_person"] = budget
    out["enrollment_speech_sec_total"] = sum(float(c["speech_duration_sec"]) for c in ordered)
    return out


ASSET_FIELDS = (
    "audio_file",
    "vad_mask_file",
    "speaker_id",
    "utterance_id",
    "query_id",
    "known",
    "source_partition",
    "mic",
    "duration_bucket",
    "duration_sec",
    "speech_duration_sec",
    "source_path",
    "source_sha256",
    "converted_sha256",
    "vad_sha256",
)


TRIAL_FIELDS = (
    "trial_id",
    "cohort_id",
    "population",
    "evaluation_partition",
    "seed",
    "group_size",
    "enrollment_budget_speech_sec_per_person",
    "enrollment_order_variant",
    "enrolled_speaker_ids",
    "enrollment_order",
    "unknown_speaker_ids",
    "task09a_candidate_group_id",
    "task09a_candidate_variant",
    "task09a_experiment_family",
    "task09a_enrollment_gap_ms",
    "task09a_matched_comparison_id",
    "task09a_candidate_attempt_limit",
    "task09a_construction_budget_target_sec_per_person",
    "task09a_inter_speaker_pause_only",
    "enrollment_speech_sec_total",
    "run_unknown_traffic",
    "unknown_traffic_query_ids",
)


def asset_id(item: dict[str, Any]) -> str:
    return Path(item["audio_file"]).stem


def compact_asset(item: dict[str, Any]) -> dict[str, Any]:
    # Known/ID labels are trial-specific for query audio and are kept in the
    # small reference object rather than duplicated in the shared catalog.
    return {key: item[key] for key in ASSET_FIELDS if key in item and key not in ("known",)}


def compact_trial(trial: dict[str, Any]) -> dict[str, Any]:
    out = {key: trial[key] for key in TRIAL_FIELDS if key in trial}
    out["enrollment"] = [asset_id(item) for item in trial["enrollment"]]
    out["validation_enrollment"] = [asset_id(item) for item in trial["validation_enrollment"]]
    out["queries"] = [
        {"asset_id": asset_id(item), "query_id": item["query_id"], "known": item["known"]}
        for item in trial["queries"]
    ]
    out["conversations"] = []
    return out


def validate_compact_plan(manifest: dict[str, Any]) -> None:
    catalog = manifest["task09a_asset_catalog"]
    for trial in manifest["trials"]:
        trial_id = trial["trial_id"]
        enrolled = set(trial["enrolled_speaker_ids"])
        reserved_unknown = set(trial["unknown_speaker_ids"])
        construction: set[tuple[str, str]] = set()
        validation: set[tuple[str, str]] = set()
        for asset_id_value in trial["enrollment"]:
            asset = catalog[asset_id_value]
            key = (asset["speaker_id"], asset["utterance_id"])
            if asset.get("source_partition") != "enrollment" or asset.get("mic") != "mic1":
                raise ValueError(f"{trial_id}: construction asset outside allowed mic1 enrollment partition")
            if asset["speaker_id"] not in enrolled or key in construction:
                raise ValueError(f"{trial_id}: construction identity mismatch or duplicate utterance")
            construction.add(key)
        for asset_id_value in trial["validation_enrollment"]:
            asset = catalog[asset_id_value]
            key = (asset["speaker_id"], asset["utterance_id"])
            if asset.get("source_partition") != "enrollment" or asset.get("mic") != "mic1":
                raise ValueError(f"{trial_id}: validation asset outside allowed mic1 enrollment partition")
            if asset["speaker_id"] not in enrolled or key in validation:
                raise ValueError(f"{trial_id}: validation identity mismatch or duplicate utterance")
            validation.add(key)
        if construction & validation:
            raise ValueError(f"{trial_id}: construction and held-out enrollment utterances overlap")
        for asset_id_value in trial["queries"]:
            asset = catalog[asset_id_value["asset_id"]]
            known = bool(asset_id_value["known"])
            speaker = asset["speaker_id"]
            if asset_id_value["query_id"] != asset["query_id"]:
                raise ValueError(f"{trial_id}: query asset ID mismatch")
            if asset.get("source_partition") != trial["evaluation_partition"]:
                raise ValueError(f"{trial_id}: query asset is from the wrong evaluation partition")
            if asset.get("mic") != "mic1":
                raise ValueError(f"{trial_id}: query is outside the primary mic1 evaluation set")
            if known and speaker not in enrolled:
                raise ValueError(f"{trial_id}: known query identity is not enrolled")
            if not known and (speaker in enrolled or speaker not in reserved_unknown):
                raise ValueError(f"{trial_id}: unknown query identity is enrolled or not reserved as unknown")
            if known and (speaker, asset["utterance_id"]) in (construction | validation):
                raise ValueError(f"{trial_id}: enrollment/query utterance leakage")


def create_plan(source_path: Path, output_path: Path) -> None:
    source_text = source_path.read_bytes()
    source = json.loads(source_text)
    if source.get("schema") != "nemo-speech.voiceid-task08.v1":
        raise ValueError("input must be the validated Task 08 expanded manifest")
    cohort_trials: dict[tuple[str, int, str], dict[str, Any]] = {}
    for trial in source["trials"]:
        if trial["enrollment_budget_speech_sec_per_person"] != 60:
            continue
        if trial["enrollment_order_variant"] != "randomized":
            continue
        cohort_trials[(trial["population"], trial["group_size"], trial["cohort_id"])] = trial

    counts = {"development": 2, "calibration": 1, "final_holdout": 1}
    selected_cohorts: list[dict[str, Any]] = []
    for population in ("development", "calibration", "final_holdout"):
        for size in (2, 4, 8):
            matches = sorted(
                (trial for (pop, n, _), trial in cohort_trials.items() if pop == population and n == size),
                key=lambda trial: trial["cohort_id"],
            )
            if len(matches) < counts[population]:
                raise ValueError(f"not enough {population} n={size} randomized Task 08 cohorts")
            selected_cohorts.extend(matches[: counts[population]])

    out_trials: list[dict[str, Any]] = []
    for parent in selected_cohorts:
        speaker_ids = list(parent["enrollment_order"])
        queries = selected_queries(parent, speaker_ids)
        cohort_short = parent["cohort_id"].split("_")[-1]
        for budget in GROUP_BUDGETS:
            construction, validation = make_training_and_validation(speaker_ids, parent["enrollment"], budget)
            group_id = f"{parent['population']}_n{parent['group_size']}_{cohort_short}_e{budget:02d}"
            for policy in ORDER_VARIANTS:
                variant = policy
                trial_id = f"task09a_{group_id}_{variant}"
                record = candidate_record(
                    parent, trial_id, group_id, variant, construction, validation, queries,
                    family="order_search", budget=budget)
                record["task09a_candidate_attempt_limit"] = len(ORDER_VARIANTS)
                out_trials.append(record)

        # Silence spacing is an ablation, not part of profile selection.
        construction, validation = make_training_and_validation(speaker_ids, parent["enrollment"], 30)
        base_group = f"{parent['population']}_n{parent['group_size']}_{cohort_short}_e30"
        if parent["population"] in ("development", "calibration"):
            for gap_ms in (0, 500):
                matched = f"pause_{base_group}"
                out_trials.append(candidate_record(
                    parent, f"task09a_pause_{base_group}_gap{gap_ms}",
                    f"{matched}_gap{gap_ms}", f"grouped_gap{gap_ms}",
                    construction, validation, queries, gap_ms=gap_ms,
                    family="pause_ablation", matched_id=matched, budget=30))
            # This second pause study changes only silence at actual speaker
            # boundaries. The grouped e30 order candidate is the Task 08
            # 250 ms per-clip gap control on the same enrollment audio.
            for gap_ms in (0, 250, 500):
                matched = f"inter_speaker_pause_{base_group}"
                record = candidate_record(
                    parent, f"task09a_inter_speaker_pause_{base_group}_gap{gap_ms}",
                    f"{matched}_gap{gap_ms}", f"inter_speaker_gap{gap_ms}",
                    construction, validation, queries, gap_ms=gap_ms,
                    family="inter_speaker_pause_ablation", matched_id=matched, budget=30)
                record["task09a_inter_speaker_pause_only"] = True
                out_trials.append(record)

        # Compare two deterministic, utterance-disjoint selections from the
        # same enrollment-pool utterances at a matched 30-second target.
        if parent["population"] == "development" and cohort_short == "r00":
            base_heldout = {clip["utterance_id"] for clip in validation}
            pool = [clip for clip in parent["enrollment"] if clip["utterance_id"] not in base_heldout]
            by_speaker = grouped_clips(speaker_ids, pool)
            selection_sets: dict[str, list[dict[str, Any]]] = {"selection_a": [], "selection_b": []}
            complements: dict[str, list[dict[str, Any]]] = {"selection_a": [], "selection_b": []}
            for speaker in speaker_ids:
                ordered_pool = sorted(by_speaker[speaker], key=lambda c: (stable_key(c["utterance_id"]), c["utterance_id"]))
                for index, clip in enumerate(ordered_pool):
                    choice = "selection_a" if index % 2 == 0 else "selection_b"
                    selection_sets[choice].append(clip)
                    complement = "selection_b" if choice == "selection_a" else "selection_a"
                    complements[complement].append(clip)
            for variant in ("selection_a", "selection_b"):
                validation_set = list(validation) + complements[variant]
                group_id = f"selection_{parent['cohort_id']}_e30"
                trial_id = f"task09a_{group_id}_{variant}"
                record = candidate_record(
                    parent, trial_id, f"{group_id}_{variant}", variant,
                    selection_sets[variant], validation_set, queries,
                    family="utterance_selection", matched_id=group_id, budget=30)
                record["task09a_candidate_attempt_limit"] = 2
                out_trials.append(record)

    asset_catalog: dict[str, dict[str, Any]] = {}
    for trial in out_trials:
        for field in ("enrollment", "validation_enrollment", "queries"):
            for item in trial[field]:
                key = asset_id(item)
                compact = compact_asset(item)
                existing = asset_catalog.setdefault(key, compact)
                if existing != compact:
                    raise ValueError(f"asset metadata disagrees across trial references: {key}")

    out_manifest = {
        "schema": "nemo-speech.voiceid-task09a-enrollment.v1",
        "source_manifest_sha256": hashlib.sha256(source_text).hexdigest(),
        "source_manifest_schema": source["schema"],
        "random_seed": source["random_seed"],
        "cohort_selection": {"development": 2, "calibration": 1, "final_holdout_followup": 1},
        "group_sizes": [2, 4, 8],
        "construction_budget_targets_sec_per_person": list(GROUP_BUDGETS),
        "order_search_budget_per_registration": len(ORDER_VARIANTS),
        "order_search_variants": list(ORDER_VARIANTS),
        "validation_protocol": "one longest deterministic enrollment-partition utterance per identity is withheld from candidate construction; the same held-out utterance validates every order candidate",
        "utterance_selection_protocol": "two parity-balanced, stable-hash partitions of each selected 60-second enrollment utterance pool; one side builds the profile and the other side plus the withheld utterance validates it",
        "query_protocol": "one deterministic mic1 known query per enrolled identity and up to four unknown mic1 speakers from the Task 08 partition for matched follow-up scoring",
        "profile_attempt_count": len(out_trials),
        "source_manifest_task08_trial_count": len(source["trials"]),
        "task09a_asset_catalog": asset_catalog,
        "trials": [compact_trial(trial) for trial in out_trials],
    }
    validate_compact_plan(out_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "profile_attempts": len(out_trials), "selected_cohorts": [t["cohort_id"] for t in selected_cohorts]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-plan")
    build.add_argument("--task08-manifest", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build-plan":
        create_plan(args.task08_manifest, args.output)


if __name__ == "__main__":
    main()
