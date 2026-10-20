#!/usr/bin/env python3
"""Package exact Task 09B reference reproduction for Task 09C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED = {
    "pre_encoder_mean": (112, 126, 17, 365),
    "pre_encoder_mean_std": (114, 126, 18, 365),
    "selected_learned_head": (103, 126, 13, 365),
}


def compact(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "fold_id", "method", "representation", "pooling", "model_key", "validation_selected_model",
        "enrollment_budget_sec", "mean_actual_enrollment_sec", "registry_size_requested",
        "registry_size_usable", "query_budget_sec", "query_attempts", "known_queries",
        "known_correct_accepts", "known_wrong_identity_accepts", "known_rejected_as_unknown",
        "known_correct_accept_rate", "known_correct_accept_speaker_macro", "known_correct_accept_ci95",
        "known_rank1_correct", "unknown_queries", "unknown_false_accepts", "unknown_false_accept_rate",
        "unknown_false_accept_speaker_macro", "unknown_false_accept_ci95", "unknown_correct_rejections",
        "no_decisions", "no_decision_rate", "open_set_correct", "open_set_queries", "open_set_accuracy",
        "verification_policy_threshold", "threshold_calibration", "score_distributions",
        "verification_genuine_accepts", "verification_genuine_trials", "verification_frr",
        "verification_impostor_accepts", "verification_impostor_trials", "verification_far",
        "verification_eer", "verification_auc", "actual_query_duration_bucket_metrics",
    )
    return {key: row[key] for key in fields if key in row}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True,
                        help="full JSON written by the original Task 09B evaluator")
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09c_baselines.json"))
    args = parser.parse_args()
    source = json.loads(args.verification.read_text(encoding="utf-8"))
    rows = [row for row in source["verification_and_open_set_by_condition"]
            if row["enrollment_budget_sec"] == 10 and row["registry_size_requested"] == 8
            and row["query_budget_sec"] == 3 and row["pooling"] == "single_pooled"
            and ((row["method"], row["representation"]) in {
                ("mean_cosine", "pre_encoder"), ("mean_std_cosine", "pre_encoder")
            } or row.get("validation_selected_model"))]
    expected_methods = {"mean_cosine", "mean_std_cosine"}
    if len([row for row in rows if row["method"] in expected_methods]) != 6:
        raise RuntimeError("reproduction is missing one or more frozen pre-encoder baseline folds")
    selected = [row for row in rows if row.get("validation_selected_model")]
    if len(selected) != 3:
        raise RuntimeError("reproduction must contain one selected learned head per fold")

    def sums(method_rows: list[dict[str, Any]]) -> tuple[int, int, int, int]:
        return (sum(row["known_correct_accepts"] for row in method_rows),
                sum(row["known_queries"] for row in method_rows),
                sum(row["unknown_false_accepts"] for row in method_rows),
                sum(row["unknown_queries"] for row in method_rows))

    method_rows = {
        "pre_encoder_mean": [row for row in rows if row["method"] == "mean_cosine"],
        "pre_encoder_mean_std": [row for row in rows if row["method"] == "mean_std_cosine"],
        "selected_learned_head": selected,
    }
    reproduced = {name: sums(value) for name, value in method_rows.items()}
    matched = reproduced == EXPECTED
    if not matched:
        raise RuntimeError(f"Task 09B baseline changed: expected {EXPECTED}, reproduced {reproduced}")
    output = {
        "schema": "nemo-speech.voiceid-task09c-baselines.v1",
        "reproduction_status": "exact count match",
        "source_evaluator": "tests/tools/evaluate_voiceid_head.py on original Task 09B manifest and cached Vulkan feature pools",
        "source_full_result": str(args.verification.resolve()),
        "source_full_result_sha256": __import__("hashlib").sha256(args.verification.read_bytes()).hexdigest(),
        "primary_condition": {"enrollment_sec": 10, "registry_size": 8, "query_sec": 3,
                              "query_microphone": "mic1", "threshold": "Task 09B calibration threshold, frozen per fold"},
        "expected_pooled_reference": {name: list(values) for name, values in EXPECTED.items()},
        "reproduced_pooled_counts": {name: list(values) for name, values in reproduced.items()},
        "fold_results": {name: [compact(row) for row in value] for name, value in method_rows.items()},
        "interpretation": "pooled attempts span repeated speaker identities across folds and are not independent-trial confidence bounds; use fold and speaker-cluster results",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "exact_count_match": matched,
                      "reproduced": {key: list(value) for key, value in reproduced.items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"evaluate_task09c_baselines: {exc}", file=sys.stderr)
        raise
