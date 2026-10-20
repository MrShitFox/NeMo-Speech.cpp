#!/usr/bin/env python3
"""Summarize enrollment budgets, speech availability, and frozen-prototype stability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_voiceid_head import Method
from task09b_voiceid import FeatureData, cosine, load_data


BUDGETS = (5, 10, 15, 30, 60)


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min_p10_p25_p50_p75_p90_max": None}
    return {"count": len(values), "min_p10_p25_p50_p75_p90_max":
            [float(value) for value in np.quantile(np.asarray(values), [0, .1, .25, .5, .75, .9, 1])]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--pooled-features", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09c_enrollment.json"))
    args = parser.parse_args()
    manifest, assets, pooled = load_data(args.manifest, args.pooled_features)
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    rows = [row for row in verification["verification_and_open_set_by_condition"]
            if row["method"] == "train_zscore_cosine" and row["representation"] == "pre_encoder"
            and row["pooling"] == "single_pooled" and row["registry_size_requested"] == 8
            and row["query_budget_sec"] == 3 and row["enrollment_budget_sec"] in BUDGETS]
    expected = 3 * len(BUDGETS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} per-fold enrollment rows, found {len(rows)}")
    rows.sort(key=lambda row: (row["fold_id"], row["enrollment_budget_sec"]))
    compact_fields = ("fold_id", "enrollment_budget_sec", "registry_size_requested", "registry_size_usable",
        "registry_profile_failures", "registry_failure_ids", "mean_actual_enrollment_sec", "verification_policy_threshold",
        "known_correct_accepts", "known_queries", "known_correct_accept_rate", "known_rank1_correct",
        "known_wrong_identity_accepts", "known_rejected_as_unknown", "unknown_false_accepts", "unknown_queries",
        "unknown_false_accept_rate", "no_decisions", "verification_genuine_accepts", "verification_genuine_trials",
        "verification_impostor_accepts", "verification_impostor_trials", "verification_far", "verification_frr",
        "verification_eer", "verification_auc", "score_distributions")

    stability_rows = []
    profile_details: dict[str, dict[str, Any]] = {}
    for fold in manifest["folds"]:
        data = FeatureData(assets, pooled, "pre_encoder")
        method = Method("train_zscore_cosine", "pre_encoder", data, fold["train_utterance_job_ids"])
        profiles: dict[str, dict[int, dict[str, Any]]] = {}
        for speaker in fold["evaluation_registry_order"][:8]:
            person = fold["evaluation_people"][speaker]
            profiles[speaker] = {}
            for budget in BUDGETS:
                pooled_audio = data.pool(person["enrollment_job_ids"], float(budget))
                actual = float(pooled_audio["speech_duration_sec"])
                embedding = method.embed_many([pooled_audio])[0] if actual >= budget else None
                profiles[speaker][budget] = {
                    "requested_speech_sec": budget, "actual_usable_speech_sec": actual,
                    "selected_enrollment_clip_count": len(pooled_audio["selected_job_indices"]),
                    "profile_success": embedding is not None,
                    "failure_reason": None if embedding is not None else "enrollment_data_exhaustion_before_budget",
                    "embedding": embedding,
                }
        profile_details[fold["fold_id"]] = {
            speaker: {str(budget): {key: value for key, value in record.items() if key != "embedding"}
                      for budget, record in per_budget.items()}
            for speaker, per_budget in profiles.items()
        }
        for budget in BUDGETS:
            records = [profiles[speaker][budget] for speaker in profiles]
            vectors = {speaker: record["embedding"] for speaker, per in profiles.items()
                       if (record := per[budget])["embedding"] is not None}
            stability_to_5 = [cosine(vectors[speaker], profiles[speaker][5]["embedding"])
                              for speaker in vectors if profiles[speaker][5]["embedding"] is not None]
            stability_to_10 = [cosine(vectors[speaker], profiles[speaker][10]["embedding"])
                               for speaker in vectors if profiles[speaker][10]["embedding"] is not None]
            stability_rows.append({
                "fold_id": fold["fold_id"], "requested_enrollment_speech_sec": budget,
                "profile_attempts": len(records), "profile_successes": sum(r["profile_success"] for r in records),
                "profile_failures": sum(not r["profile_success"] for r in records),
                "failure_reasons": {"enrollment_data_exhaustion_before_budget": sum(not r["profile_success"] for r in records)},
                "actual_usable_speech_sec": quantiles([r["actual_usable_speech_sec"] for r in records]),
                "selected_enrollment_clip_count": quantiles([r["selected_enrollment_clip_count"] for r in records]),
                "prototype_cosine_to_5s_profile": quantiles(stability_to_5),
                "prototype_cosine_to_10s_profile": quantiles(stability_to_10),
            })
        for per in profiles.values():
            for record in per.values():
                record.pop("embedding")

    output = {
        "schema": "nemo-speech.voiceid-task09c-enrollment.v1",
        "evaluation_status": "Task 09B 8-person/3-second enrollment budget regression evidence on previously studied VCTK identities",
        "method": "frozen pre_encoder mean, training-speaker-only diagonal z-score, one pooled cosine prototype",
        "enrollment_budget_results": [{key: row[key] for key in compact_fields if key in row} for row in rows],
        "profile_generation_and_stability": {
            "protocol": "one profile attempt for each of the eight registered evaluation identities per fold; no profile uses AOSC or one of eight native channels",
            "fold_results": stability_rows,
            "per_identity_budget_outcomes": profile_details,
        },
        "prototype_construction_selection": {
            "selection_source": "tests/results/voiceid_task09c_pooling.json",
            "selected_global_method": "train_zscore_mean: one pooled mean prototype, then training-only diagonal standardization",
            "alternative": "mean_norm_utterance_proto: L2-normalize each enrollment utterance mean before averaging",
            "development_balanced_macro_scores": "see normalization result; no evaluation identities selected the prototype method",
        },
        "limitations": "Enrollment clip budgets can overshoot by the final utterance; the same-speaker prototype stability measurements are VCTK utterance evidence, not session stability.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "enrollment_rows": len(rows),
                      "stability_rows": len(stability_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"summarize_task09c_enrollment: {exc}", file=sys.stderr)
        raise
