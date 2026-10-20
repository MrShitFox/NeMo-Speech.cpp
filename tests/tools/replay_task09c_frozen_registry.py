#!/usr/bin/env python3
"""Register frozen pre-encoder prototypes in separate processes, then reload and query."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"registry command returned no JSON: {command[1]}")
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--pooled-features", type=Path, required=True)
    parser.add_argument("--task09b-verification", type=Path,
                        default=Path("/home/glebus/.cache/nemo-speech-task09b/task09c_reproduction_verification.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("/home/glebus/.cache/nemo-speech-task09b/run"))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--matching-benchmark", type=Path)
    parser.add_argument("--feature-extraction-benchmark", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09c_native_feasibility.json"))
    parser.add_argument("--fold-id", default="fold_1")
    parser.add_argument("--identity-count", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verification = json.loads(args.task09b_verification.read_text(encoding="utf-8"))
    fold = next(row for row in manifest["folds"] if row["fold_id"] == args.fold_id)
    order = fold["evaluation_registry_order"]
    if args.identity_count < 1 or args.identity_count + 1 >= len(order):
        parser.error("identity-count must leave at least one unknown evaluation speaker")
    calibration = next(row for row in verification["calibration"]
                       if row["fold_id"] == args.fold_id and row["method"] == "train_zscore_cosine"
                       and row["representation"] == "pre_encoder" and row["pooling"] == "single_pooled")
    threshold = float(calibration["threshold"])
    script = Path(__file__).with_name("voiceid_registry_research.py").resolve()
    shared = ["--voiceid-method", "pre_encoder_zscore", "--manifest", str(args.manifest),
              "--pooled-features", str(args.pooled_features), "--run-dir", str(args.run_dir),
              "--fold-id", args.fold_id, "--registry", str(args.registry)]
    command_base = [sys.executable, str(script)]
    registrations = []
    for index, speaker in enumerate(order[:args.identity_count], start=1):
        registrations.append(run(command_base + ["register"] + shared + [
            "--source-speaker", speaker, "--identity", f"tenant_{index:03d}", "--enrollment-sec", "10"]))
    known_query = run(command_base + ["query"] + shared + [
        "--source-speaker", order[0], "--query-sec", "3", "--threshold", str(threshold)])
    unknown_query = run(command_base + ["query"] + shared + [
        "--source-speaker", order[args.identity_count], "--query-sec", "3", "--threshold", str(threshold)])

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    identity_ids = sorted(registry["identities"])
    expected_ids = [f"tenant_{index:03d}" for index in range(1, args.identity_count + 1)]
    if identity_ids != expected_ids or registry.get("embedding_dimension") != 512:
        raise RuntimeError("serialized frozen registry failed identity/dimension reload validation")
    if registry.get("embedding_transform", {}).get("kind") != "training_speaker_diagonal_zscore":
        raise RuntimeError("serialized registry is missing its training-only transform")
    prototype_payload_bytes = [len(base64.b64decode(record["prototype_f32le_base64"], validate=True))
                               for record in registry["identities"].values()]
    prototype_base64_chars = [len(record["prototype_f32le_base64"])
                              for record in registry["identities"].values()]
    result = {
        "schema": "nemo-speech.voiceid-task09c-native-feasibility.v1",
        "evaluation_status": "research registry serialization and matching feasibility only; not a recognition reliability validation",
        "method": "pre_encoder 512D mean, fold training-only diagonal z-score, normalized cosine prototypes",
        "fold_id": args.fold_id,
        "threshold": threshold,
        "threshold_limitation": "threshold is calibrated on 15 speakers with at most 14 leave-one-out impostor competitors; the 16-profile registry is a storage/matching exercise, not a calibrated 16-person operating point",
        "cross_process_registry": {
            "external_id_count": len(registrations), "external_ids": identity_ids,
            "registration_processes": len(registrations), "query_processes": 2,
            "registered_profiles_successful": sum(row.get("registered_identities", 0) > 0 for row in registrations),
            "dimensions": sorted({row["embedding_dimension"] for row in registrations}),
            "raw_float32_bytes_per_profile": sorted({row["prototype_raw_bytes"] for row in registrations}),
            "serialized_prototype_payload_bytes_mean": sum(prototype_payload_bytes) / len(prototype_payload_bytes) if prototype_payload_bytes else None,
            "serialized_prototype_base64_chars_mean": sum(prototype_base64_chars) / len(prototype_base64_chars) if prototype_base64_chars else None,
            "registry_json_bytes": args.registry.stat().st_size,
            "training_transform_reloaded_from_registry": True,
            "native_channel_indices_used": False,
            "known_query": known_query,
            "unknown_query": unknown_query,
        },
        "representation_storage": {
            "prototype_dimension": 512,
            "raw_float32_bytes_per_identity": 512 * 4,
            "mean_std_control_dimension": 1024,
            "mean_std_raw_float32_bytes_per_identity": 1024 * 4,
            "match_cost": "one 512-element dot product per registered identity after a 512-element standardization and L2 normalization",
            "complexity": "O(512 * registry_size) multiply-accumulate terms per query",
        },
        "feature_extraction_interface": {
            "existing_observer": "Task 06 SortformerResearchConfig pre_encoder tap; Task 09B NVFTR probe already emits 512D frame features and VAD masks",
            "inference_backend_checked": "Vulkan0, model fingerprint 08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1",
            "context_profile_required": False,
            "public_c_abi_changed": False,
        },
        "limitations": "Registry replay consumes cached Task 09B feature arrays in Python. It verifies arbitrary external IDs, profile serialization, process restart, and matching semantics, but does not implement native production inference or measure end-to-end audio latency.",
    }
    if args.matching_benchmark:
        result["cpu_matching_benchmark"] = json.loads(args.matching_benchmark.read_text(encoding="utf-8"))
    if args.feature_extraction_benchmark:
        result["vulkan_feature_extraction_benchmark"] = json.loads(
            args.feature_extraction_benchmark.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "identity_count": len(identity_ids),
                      "known_decision": known_query.get("decision"),
                      "unknown_decision": unknown_query.get("decision"),
                      "registry_json_bytes": args.registry.stat().st_size}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"replay_task09c_frozen_registry: {exc}", file=sys.stderr)
        raise
