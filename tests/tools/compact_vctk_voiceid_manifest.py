#!/usr/bin/env python3
"""Write a compact, repository-safe summary of an expanded Task 08 manifest."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_trial(trial: dict[str, Any]) -> dict[str, Any]:
    query_counts = collections.Counter(
        f"{'known' if query['known'] else 'unknown'}:{query['mic']}:{query['duration_bucket']}"
        for query in trial.get("queries", [])
    )
    enrollment_by_speaker: dict[str, dict[str, Any]] = {}
    for asset in trial.get("enrollment", []):
        speaker = asset["speaker_id"]
        entry = enrollment_by_speaker.setdefault(
            speaker, {"utterance_ids": [], "usable_speech_sec": 0.0}
        )
        entry["utterance_ids"].append(asset["utterance_id"])
        entry["usable_speech_sec"] += asset["speech_duration_sec"]
    for entry in enrollment_by_speaker.values():
        entry["usable_speech_sec"] = round(entry["usable_speech_sec"], 4)
    return {
        "trial_id": trial["trial_id"],
        "cohort_id": trial["cohort_id"],
        "population": trial["population"],
        "group_size": trial["group_size"],
        "enrollment_budget_speech_sec_per_person": trial[
            "enrollment_budget_speech_sec_per_person"
        ],
        "enrollment_order_variant": trial["enrollment_order_variant"],
        "enrolled_speaker_ids": trial["enrolled_speaker_ids"],
        "enrollment_order": trial["enrollment_order"],
        "unknown_speaker_ids": trial["unknown_speaker_ids"],
        "enrollment_utterances_by_speaker": enrollment_by_speaker,
        "query_count": sum(query_counts.values()),
        "query_counts": dict(sorted(query_counts.items())),
        "unknown_traffic_query_count": len(trial.get("unknown_traffic_query_ids", []))
        if trial.get("run_unknown_traffic") else 0,
        "conversation_scenarios": [
            conversation["scenario"] for conversation in trial.get("conversations", [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="expanded manifest, stored outside Git")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--results", type=Path, help="optional C++ JSONL to capture its manifest fingerprint")
    parser.add_argument("--prepare-command", required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    inventory = load_json(args.inventory)
    trials = manifest.get("trials", [])
    queries = [query for trial in trials for query in trial.get("queries", [])]
    conversations = [conversation for trial in trials for conversation in trial.get("conversations", [])]
    results_fingerprint = None
    if args.results and args.results.is_file():
        with args.results.open("r", encoding="utf-8") as stream:
            first = next((line for line in stream if line.strip()), None)
        if first:
            results_fingerprint = json.loads(first).get("manifest_fingerprint_fnv1a64")

    dataset_fields = (
        "file_count", "header_readable_file_count", "unreadable_header_file_count",
        "speaker_count", "mic1_usable_speaker_count", "utterance_identity_count",
        "total_header_duration_sec", "clip_duration_sec", "sample_rate_file_counts",
        "channel_file_counts", "bits_per_sample_file_counts", "microphone_variant_counts",
        "missing_mic1_audio_for_transcript_count", "audio_without_matching_transcript_count",
    )
    compact = {
        "schema": "nemo-speech.vctk-voiceid-compact-plan.v1",
        "preset": manifest["preset"],
        "random_seed": manifest["random_seed"],
        "model": manifest["model"],
        "prepare_command": args.prepare_command,
        "expanded_manifest": {
            "sha256": sha256(args.manifest),
            "bytes": args.manifest.stat().st_size,
            "cpp_fnv1a64": results_fingerprint,
            "storage": "external to Git; regenerate with prepare_command",
        },
        "preparation_script_sha256": sha256(Path(__file__).with_name("prepare_vctk_voiceid.py")),
        "compactor_script_sha256": sha256(Path(__file__).resolve()),
        "dataset_inventory_sha256": sha256(args.inventory),
        "dataset_inventory": {key: inventory[key] for key in dataset_fields if key in inventory},
        "conversion": manifest["conversion"],
        "vad": manifest["vad"],
        "partition_rules": manifest["partition_rules"],
        "speaker_populations": manifest["speaker_populations"],
        "speaker_role_coverage": manifest.get("speaker_role_coverage", {}),
        "experiment_counts": {
            "trials": len(trials),
            "isolated_query_attempts": len(queries),
            "known_queries": sum(bool(query["known"]) for query in queries),
            "unknown_queries": sum(not query["known"] for query in queries),
            "query_duration_buckets": dict(sorted(collections.Counter(
                query["duration_bucket"] for query in queries
            ).items())),
            "microphones": dict(sorted(collections.Counter(query["mic"] for query in queries).items())),
            "conversations": len(conversations),
            "conversation_scenarios": dict(sorted(collections.Counter(
                conversation["scenario"] for conversation in conversations
            ).items())),
            "prepared_assets_referenced": len(manifest.get("preparation_assets", {})),
        },
        "trials": [compact_trial(trial) for trial in trials],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "trials": len(trials),
        "queries": len(queries),
        "conversations": len(conversations),
        "compact_bytes": args.output.stat().st_size,
        "expanded_sha256": compact["expanded_manifest"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
