#!/usr/bin/env python3
"""Measure a fixed RMS-normalization policy on the locked Task 09C mic matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyze_task09c_microphones import calibration_threshold, evaluate_condition
from evaluate_task09c_pooling import compact_result
from task09b_voiceid import FeatureData
from evaluate_voiceid_head import Method, threshold_for_method


SCHEMA = "nemo-speech.voiceid-task09c-rms-normalized-audio.v1"
CONDITIONS = (("mic1", "mic1"), ("mic1", "mic2"), ("mic2", "mic1"), ("mic2", "mic2"))


def normalized_id(job_id: str) -> str:
    return hashlib.sha256(f"{SCHEMA}|{job_id}".encode("utf-8")).hexdigest()[:24]


def quantile(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p10_p50_p90": None}
    return {"count": len(values), "p10_p50_p90": [float(value) for value in np.quantile(values, [.1, .5, .9])]}


def load_normalized_pool(path: Path, expected_ids: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        pooled = {key: archive[key] for key in archive.files}
    inverse = {normalized_id(job_id): job_id for job_id in expected_ids}
    raw_ids = pooled["job_ids"].tolist()
    if len(raw_ids) != len(inverse) or set(raw_ids) != set(inverse):
        missing = len(set(inverse) - set(raw_ids))
        extra = len(set(raw_ids) - set(inverse))
        raise RuntimeError(f"normalized feature job coverage mismatch: missing={missing}, extra={extra}")
    pooled["job_ids"] = np.asarray([inverse[job_id] for job_id in raw_ids], dtype="U32")
    return pooled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--microphone-additions", type=Path, default=Path("tests/manifests/task09c_vctk_microphone_additions.json"))
    parser.add_argument("--raw-result", type=Path, default=Path("tests/results/voiceid_task09c_pooling.json"))
    parser.add_argument("--raw-microphone-result", type=Path, default=Path("tests/results/voiceid_task09c_microphone.json"))
    parser.add_argument("--audio-summary", type=Path,
                        default=Path("/home/glebus/.cache/nemo-speech-task09c/rmsnorm_summary.json"))
    parser.add_argument("--normalized-pooled-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09c_normalization.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    additions = json.loads(args.microphone_additions.read_text(encoding="utf-8"))
    base_assets = {row["job_id"]: row for row in manifest["assets"]}
    assets = {**base_assets, **{row["job_id"]: row for row in additions["new_assets"]}}
    pooled = load_normalized_pool(args.normalized_pooled_features, set(assets))
    data_by_fold = {fold["fold_id"]: FeatureData(assets, pooled, "pre_encoder") for fold in manifest["folds"]}
    add_folds = {row["fold_id"]: row for row in additions["folds"]}
    normalized_rows = []
    fixed_transfer_rows = []
    raw_json = json.loads(args.raw_result.read_text(encoding="utf-8"))
    audio_summary = json.loads(args.audio_summary.read_text(encoding="utf-8"))
    raw_rows = {(row["fold_id"], row["enrollment_microphone"], row["query_microphone"]): row
                for row in raw_json["locked_evaluation_selected_method"]
                if row["method"] == "train_zscore_mean"}
    raw_mic_json = json.loads(args.raw_microphone_result.read_text(encoding="utf-8"))
    raw_transfers = {(row["fold_id"], row["method"]): row
                     for row in raw_mic_json.get("task09b_threshold_mic2_transfer", [])}

    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        data = data_by_fold[fold_id]
        method = Method("train_zscore_cosine", "pre_encoder", data, fold["train_utterance_job_ids"])
        addition_fold = add_folds[fold_id]
        legacy_threshold, legacy_calibration = threshold_for_method(method, fold)
        for enrollment_mic, query_mic in CONDITIONS:
            threshold, calibration = calibration_threshold(method, fold, addition_fold,
                                                            enrollment_mic, query_mic, 10.0)
            row, _ = evaluate_condition(method, fold, addition_fold, enrollment_mic, query_mic,
                                        threshold, 94100 + len(normalized_rows) * 17, 8, 10.0)
            raw = raw_rows[(fold_id, enrollment_mic, query_mic)]
            compact = compact_result(row)
            normalized_rows.append({
                **compact,
                "method": "Task09B train_zscore_cosine on pre_encoder mean",
                "audio_preprocessing": "fixed RMS -26 dBFS target; gain capped +/-6 dB; peak capped -1 dBFS; VAD recomputed",
                "condition_calibration": calibration,
                "raw_same_condition_reference": {
                    key: raw.get(key) for key in ("threshold", "known_correct_accepts", "known_queries",
                        "known_wrong_identity_accepts", "known_rejected_as_unknown", "unknown_false_accepts",
                        "unknown_queries", "known_rank1_correct", "known_genuine_scores", "unknown_top1_scores")
                },
                "raw_same_condition_threshold": raw["threshold"],
                "known_correct_accept_delta_vs_raw": row["known_correct_accepts"] - raw["known_correct_accepts"],
                "unknown_false_accept_delta_vs_raw": row["unknown_false_accepts"] - raw["unknown_false_accepts"],
            })
            if enrollment_mic == "mic1":
                normalized_fixed_row, _ = evaluate_condition(method, fold, addition_fold, "mic1", query_mic,
                    legacy_threshold, 109100 + len(fixed_transfer_rows) * 23, 8, 10.0)
                raw_transfer = raw_transfers.get((fold_id, "train_zscore_cosine"))
                fixed_transfer_rows.append({
                    "fold_id": fold_id, "query_microphone": query_mic,
                    "normalized_mic1_calibration_threshold": legacy_threshold,
                    "normalized_threshold_calibration": legacy_calibration,
                    "normalized_threshold_applied_to_normalized_audio": {key: normalized_fixed_row.get(key) for key in ("known_correct_accepts", "known_queries",
                        "known_wrong_identity_accepts", "known_rejected_as_unknown", "unknown_false_accepts",
                        "unknown_queries", "known_rank1_correct")},
                    "raw_mic1_calibration_evaluation": {key: raw_transfer.get(key) if raw_transfer else None
                        for key in ("known_correct_accepts", "known_queries", "known_wrong_identity_accepts",
                            "known_rejected_as_unknown", "unknown_false_accepts", "unknown_queries",
                            "known_rank1_correct")},
                })

    output = {
        "schema": "nemo-speech.voiceid-task09c-audio-normalization.v1",
        "dataset_status": "VCTK follow-up on previously studied identities; not independent validation",
        "feature_representation": "frozen pre_encoder mean with Task 09B training-speaker-only per-dimension z-score and cosine matching",
        "audio_policy": {
            "target_rms_dbfs": -26.0, "gain_limit_db": [-6.0, 6.0], "peak_limit_dbfs": -1.0,
            "same_policy_for_all_speakers_and_microphones": True,
            "speaker_labels_or_evaluation_outcomes_used": False,
            "job_count": audio_summary["input_job_count"],
            "input_rms_dbfs": audio_summary["input_rms_dbfs"],
            "output_rms_dbfs": audio_summary["output_rms_dbfs"],
            "applied_gain_db": audio_summary["gain_db"],
            "source_summary": str(args.audio_summary.resolve()),
        },
        "protocol": {
            "same_utterance_pairs": True, "registry_size": 8, "enrollment_speech_sec": 10,
            "query_clips_per_speaker": "up to four; each original mic1/mic2 source utterance remains paired",
            "threshold": "per fold and mic condition, calibrated only on calibration speakers; separate frozen mic1-calibration transfer rows are also reported",
            "raw_comparison": "same frozen z-score representation and same speaker/microphone protocol on original PCM16 audio; audio variant selected without evaluation outcomes",
        },
        "normalized_condition_results": normalized_rows,
        "mic1_calibration_threshold_transfer": fixed_transfer_rows,
        "score_shift_summary": "raw and normalized condition score quantiles are included per fold; repeated mic renderings are clustered by source utterance and speaker",
        "limitations": "RMS normalization changes waveform amplitude only; it does not equalize microphone frequency response and the gain cap leaves some low-level clips below target",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(normalized_rows), "fixed_transfer_rows": len(fixed_transfer_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"evaluate_task09c_audio_normalization: {exc}", file=sys.stderr)
        raise
