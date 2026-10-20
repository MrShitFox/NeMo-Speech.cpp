#!/usr/bin/env python3
"""Extract denominator-preserving registry-size results from the Task 09B cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


FIELDS = (
    "fold_id", "registry_size_requested", "registry_size_usable", "enrollment_budget_sec",
    "query_budget_sec", "verification_policy_threshold", "threshold_calibration",
    "registry_profile_failures", "registry_failure_ids", "query_attempts", "known_queries",
    "known_correct_accepts", "known_correct_accept_rate", "known_rank1_correct",
    "known_rank1_identification_rate", "known_wrong_identity_accepts", "known_rejected_as_unknown",
    "no_decisions", "no_decision_rate", "unknown_queries", "unknown_false_accepts",
    "unknown_false_accept_rate", "unknown_correct_rejections", "verification_genuine_accepts",
    "verification_genuine_trials", "verification_impostor_accepts", "verification_impostor_trials",
    "verification_far", "verification_frr", "verification_eer", "verification_auc",
    "known_correct_accept_ci95", "unknown_false_accept_ci95", "verification_far_ci95",
    "verification_frr_ci95", "score_distributions", "actual_query_duration_buckets",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09c_open_set.json"))
    args = parser.parse_args()
    source = json.loads(args.verification.read_text(encoding="utf-8"))
    selected = [row for row in source["verification_and_open_set_by_condition"]
                if row["method"] == "train_zscore_cosine" and row["representation"] == "pre_encoder"
                and row["pooling"] == "single_pooled" and row["enrollment_budget_sec"] == 10
                and row["registry_size_requested"] in (1, 2, 4, 8, 16)
                and row["query_budget_sec"] in (1, 3, 7, 15)]
    expected = 3 * 5 * 4
    if len(selected) != expected:
        raise RuntimeError(f"expected {expected} fold/registry/query rows; found {len(selected)}")
    selected.sort(key=lambda row: (row["fold_id"], row["registry_size_requested"], row["query_budget_sec"]))
    rows = [{key: row[key] for key in FIELDS if key in row} for row in selected]
    output = {
        "schema": "nemo-speech.voiceid-task09c-open-set.v1",
        "evaluation_status": "Task 09B baseline reproduced as Task 09C regression evidence on previously studied VCTK identities",
        "method": "frozen pre_encoder mean with training-speaker-only per-dimension z-score and cosine; one per-fold calibration threshold",
        "enrollment_budget_sec": 10,
        "registry_sizes": [1, 2, 4, 8, 16],
        "query_budgets_sec": [1, 3, 7, 15],
        "threshold_policy": "single threshold per fold calibrated on 15 calibration speakers from leave-one-identity-out maximum impostor scores; no evaluation outcomes used",
        "registry_size_calibration_limit": {
            "calibration_speakers_per_fold": 15,
            "leave_one_out_maximum_impostor_competitors": 14,
            "largest_registry_directly_supported_by_this_cohort": 15,
            "sixteen_identity_interpretation": "reported evaluation uses the fixed threshold, but calibration cohort does not contain a 16-person registry condition and cannot establish a registry-size-specific 16-identity threshold",
        },
        "fold_condition_rows": rows,
        "interpretation": "Counts and denominators are preserved per fold, registry size, and query duration. VCTK microphones are paired renderings of utterances, not independent recording sessions.",
        "source_full_result": str(args.verification.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "registry_sizes": [1, 2, 4, 8, 16]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"summarize_task09c_open_set: {exc}", file=sys.stderr)
        raise
