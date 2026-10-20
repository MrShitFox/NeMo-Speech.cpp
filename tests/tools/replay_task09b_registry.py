#!/usr/bin/env python3
"""Exercise unseen identity registration/query across fresh processes."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"registry command returned no JSON: {command[1]}")
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--training-results", type=Path, default=Path("tests/results/voiceid_task09b_training.json"))
    parser.add_argument("--verification-results", type=Path, default=Path("tests/results/voiceid_task09b_verification.json.gz"))
    parser.add_argument("--pooled-features", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09b_registry.json"))
    parser.add_argument("--fold-id", default="fold_1")
    parser.add_argument("--identity-count", type=int, default=16)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    training = read_json(args.training_results)
    verification = read_json(args.verification_results)
    fold = next(row for row in manifest["folds"] if row["fold_id"] == args.fold_id)
    order = fold["evaluation_registry_order"]
    if args.identity_count < 1 or args.identity_count >= len(order):
        parser.error("identity-count must be between 1 and evaluation speaker count minus one")
    model_key = training["selected_models_by_fold"][args.fold_id]
    model = next(row for row in training["models"] if row["model_key"] == model_key)
    calibration = next(row for row in verification["calibration"]
                       if row["fold_id"] == args.fold_id and row["model_key"] == model_key
                       and row["pooling"] == "single_pooled")
    threshold = float(calibration["threshold"])
    script = Path(__file__).with_name("voiceid_registry_research.py").resolve()
    common = [sys.executable, str(script)]
    shared = ["--manifest", str(args.manifest), "--pooled-features", str(args.pooled_features),
              "--training-results", str(args.training_results), "--run-dir", str(args.run_dir),
              "--fold-id", args.fold_id, "--registry", str(args.registry)]

    registrations = []
    for index, speaker in enumerate(order[:args.identity_count], start=1):
        registrations.append(run(common + ["register"] + shared + [
            "--source-speaker", speaker, "--identity", f"external_user_{index:02d}",
            "--enrollment-sec", "10"]))
    known_query = run(common + ["query"] + shared + [
        "--source-speaker", order[0], "--query-sec", "3", "--threshold", str(threshold)])
    unknown_speaker = order[args.identity_count]
    unknown_query = run(common + ["query"] + shared + [
        "--source-speaker", unknown_speaker, "--query-sec", "3", "--threshold", str(threshold)])

    removed_identity = "external_user_01"
    remove_result = run(common + ["remove", "--registry", str(args.registry), "--identity", removed_identity])
    added_identity = "external_user_after_removal"
    add_after_remove = run(common + ["register"] + shared + [
        "--source-speaker", unknown_speaker, "--identity", added_identity, "--enrollment-sec", "10"])
    result = {
        "schema": "nemo-speech.voiceid-task09b-registry-replay.v1",
        "fold_id": args.fold_id,
        "selected_model_key": model_key,
        "representation": model["representation"],
        "head_objective": model["objective"],
        "threshold_from_calibration_only": threshold,
        "identity_count_before_queries": len(registrations),
        "registration_processes": len(registrations),
        "registry_file": str(args.registry),
        "known_query_in_separate_process": known_query,
        "unknown_query_in_separate_process": unknown_query,
        "remove_in_separate_process": remove_result,
        "add_after_removal_in_separate_process": add_after_remove,
        "all_external_ids_independent_of_native_channels": True,
        "native_channel_index_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "known_decision": known_query.get("decision"),
                      "unknown_decision": unknown_query.get("decision"),
                      "registered_identities": len(registrations)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"replay_task09b_registry: {exc}", file=sys.stderr)
        raise
