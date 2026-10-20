#!/usr/bin/env python3
"""Package exhaustive Task 09B evaluation JSON as reproducible gzip JSON."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def write_gzip_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as output:
        json.dump(value, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")


def annotate_mic2(verification: dict[str, Any], training: dict[str, Any]) -> None:
    """Attach model keys to mic2 rows emitted before the evaluator added them."""
    if verification.get("mic2_robustness") and "model_key" in verification["mic2_robustness"][0]:
        return
    model_by_key = {row["model_key"]: row for row in training["models"]}
    selected = training["selected_models_by_fold"]
    methods = verification["methods"]
    for fold_id in selected:
        selected_objective = model_by_key[selected[fold_id]]["objective"]
        candidates = [
            row for row in methods
            if row["fold_id"] == fold_id
            and row["method"] == selected_objective
            and row["pooling"] == "single_pooled"
        ]
        rows = [row for row in verification["mic2_robustness"] if row["fold_id"] == fold_id]
        if len(rows) != 3 * len(candidates):
            raise ValueError(f"mic2 row/model count mismatch for {fold_id}: {len(rows)} vs {len(candidates)}x3")
        for candidate_index, method in enumerate(candidates):
            for row in rows[candidate_index * 3:(candidate_index + 1) * 3]:
                row.update({
                    "method": method["method"],
                    "representation": method["representation"],
                    "model_key": method["model_key"],
                    "validation_selected_model": method["validation_selected_model"],
                })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--open-set", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--duration-buckets", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    open_set = json.loads(args.open_set.read_text(encoding="utf-8"))
    training = json.loads(args.training.read_text(encoding="utf-8"))
    annotate_mic2(verification, training)
    open_set["mic2_robustness"] = verification["mic2_robustness"]

    verification_path = args.output_dir / "voiceid_task09b_verification.json.gz"
    open_set_path = args.output_dir / "voiceid_task09b_open_set.json.gz"
    write_gzip_json(verification, verification_path)
    write_gzip_json(open_set, open_set_path)
    output = {
        "verification_gzip": str(verification_path),
        "verification_rows": len(verification["verification_and_open_set_by_condition"]),
        "open_set_gzip": str(open_set_path),
        "open_set_rows": len(open_set["by_condition"]),
        "mic2_rows": len(verification["mic2_robustness"]),
        "compression": "gzip level 9, compact JSON; all result rows preserved",
    }
    if args.duration_buckets is not None:
        duration = json.loads(args.duration_buckets.read_text(encoding="utf-8"))
        duration["schema"] = "nemo-speech.voiceid-task09b-duration-buckets.v1"
        duration["protocol"]["focus"] = (
            "10 seconds enrollment, 8 registered identities; selected validation head, selected-head mean of normalized utterance embeddings, "
            "pre_encoder mean+std cosine, and encoder.layer.7 mean cosine; threshold retained from 15 calibration speakers only"
        )
        duration["protocol"]["query_buckets"] = (
            "VAD speech seconds grouped by actual utterance duration: under 2, 2 to 5, 5 to 10, over 10 seconds; "
            "per-speaker 500-resample cluster intervals; query budgets remain separately identified because grouping can reuse source utterances across budgets"
        )
        duration_path = args.output_dir / "voiceid_task09b_duration_buckets.json"
        duration_path.parent.mkdir(parents=True, exist_ok=True)
        duration_path.write_text(json.dumps(duration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output["duration_buckets"] = str(duration_path)
        output["duration_bucket_rows"] = len(duration["by_condition"])
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
