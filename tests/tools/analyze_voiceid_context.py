#!/usr/bin/env python3
"""Compare identical VCTK targets across fresh, unrelated, silent, and AOSC contexts."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_voiceid_head import Method
from extract_voiceid_features import read_feature
from task09b_voiceid import FeatureData, cosine, load_data, unit


REPRESENTATIONS = ["pre_encoder", "encoder.layer.7", "encoder_output", "encoder_proj"]
CONTEXT_MODES = ["silent_2_56s", "unrelated_train_speaker_a", "unrelated_train_speaker_b", "aosc_profile_import"]


def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def speech_mask(bits: np.ndarray) -> np.ndarray:
    table = np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    return table[bits] >= 4


def pool_record(record: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    dim, ids, bits, values, digest = read_feature(record)
    selected = speech_mask(bits)
    frames = values[selected]
    counts = np.array([int(value).bit_count() for value in bits[selected]], dtype=np.int32)
    if len(frames):
        norms = np.linalg.norm(frames, axis=1, keepdims=True)
        pool = {
            "mean": frames.mean(axis=0, dtype=np.float64).astype(np.float32),
            "std": frames.std(axis=0, dtype=np.float64).astype(np.float32),
            "frame_l2_mean": (frames / np.maximum(norms, 1e-12)).mean(axis=0, dtype=np.float64).astype(np.float32),
            "speech_duration_sec": float(counts.sum()) * 0.01,
            "speech_frames": int(len(frames)),
        }
    else:
        pool = {"mean": np.zeros(dim, np.float32), "std": np.zeros(dim, np.float32),
                "frame_l2_mean": np.zeros(dim, np.float32), "speech_duration_sec": 0.0, "speech_frames": 0}
    return pool, ids, bits, values


def quantiles(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {"count": len(values), "min_p10_p25_p50_p75_p90_max": np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--pooled-features", type=Path, required=True)
    parser.add_argument("--context-feature-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-results", type=Path, default=Path("tests/results/voiceid_task09b_training.json"))
    parser.add_argument("--verification-results", type=Path, default=Path("tests/results/voiceid_task09b_verification.json.gz"))
    parser.add_argument("--output", type=Path, default=Path("tests/results/voiceid_task09b_context.json"))
    args = parser.parse_args()
    manifest, assets, pooled = load_data(args.manifest, args.pooled_features)
    training = json.loads(args.training_results.read_text(encoding="utf-8"))
    verification = read_json(args.verification_results)
    context_manifest = manifest["context_stress_protocol"]["context_probe_jobs"]
    job_by_context_id = {item["job_id"]: item for item in context_manifest}
    metadata = [json.loads(line) for line in (args.context_feature_dir / "features.jsonl").read_text().splitlines() if line.strip()]
    metadata_by_key = {(row["job_id"], row["feature_representation"]): row for row in metadata}
    base_metadata = [json.loads(line) for line in (args.pooled_features.parent / "features.jsonl").read_text().splitlines() if line.strip()]
    base_metadata_by_key = {(row["job_id"], row["feature_representation"]): row for row in base_metadata}
    expected = {(job_id, rep) for job_id in job_by_context_id for rep in REPRESENTATIONS}
    if set(metadata_by_key) != expected:
        raise RuntimeError(f"context feature coverage mismatch: expected {len(expected)}, got {len(metadata_by_key)}")

    model_rows = {row["model_key"]: row for row in training["models"]}
    results = []
    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        selected_key = training["selected_models_by_fold"][fold_id]
        selected_model = model_rows[selected_key]
        selected_rep = selected_model["representation"]
        base_data = FeatureData(assets, pooled, selected_rep)
        head = Method(selected_model["objective"], selected_rep, base_data,
                      fold["train_utterance_job_ids"], args.run_dir / selected_model["portable_export"])
        eval_rows = fold["evaluation_people"]
        registry = fold["evaluation_registry_order"][:8]
        prototypes = {}
        for speaker in registry:
            profile = base_data.pool(eval_rows[speaker]["enrollment_job_ids"], 10.0)
            if profile["speech_duration_sec"] >= 10.0:
                prototypes[speaker] = head.embed_many([profile])[0]
        calibration = next(row for row in verification["calibration"]
                           if row["fold_id"] == fold_id and row["method"] == selected_model["objective"]
                           and row["representation"] == selected_rep and row["pooling"] == "single_pooled")
        threshold = float(calibration["threshold"])

        target_values: dict[tuple[str, str], dict[str, Any]] = {}
        context_feature_summary = defaultdict(lambda: {"cosines": [], "max_abs": [], "bit_exact": 0, "targets": 0})
        for context_job_id, target in job_by_context_id.items():
            fold_context = next((row for row in fold["context_probe_jobs"] if row["context_job_id"] == context_job_id), None)
            if fold_context is None:
                continue
            mode = target["context_mode"]
            for representation in REPRESENTATIONS:
                base_row = base_metadata_by_key[(target["target_job_id"], representation)]
                context_row = metadata_by_key[(context_job_id, representation)]
                base_pool, base_ids, base_bits, base_values = pool_record(args.pooled_features.parent / base_row["record_file"])
                context_pool, context_ids, context_bits, context_values = pool_record(args.context_feature_dir / context_row["record_file"])
                if (base_values.shape != context_values.shape or not np.array_equal(base_ids, context_ids) or
                        not np.array_equal(base_bits, context_bits)):
                    raise RuntimeError(f"context target shape/timestamp mismatch for {context_job_id}/{representation}")
                diff = np.abs(base_values - context_values)
                row = context_feature_summary[(fold_id, representation, mode)]
                row["cosines"].append(cosine(base_pool["mean"], context_pool["mean"]))
                row["max_abs"].append(float(diff.max(initial=0)))
                row["bit_exact"] += int(np.array_equal(base_values, context_values))
                row["targets"] += 1

                if representation == selected_rep:
                    base_vector = head.embed_many([base_pool])[0]
                    context_vector = head.embed_many([context_pool])[0]
                    target_key = (fold_id, target["target_job_id"])
                    speaker_id = fold_context["speaker_id"]
                    known = speaker_id in set(registry)
                    for context_kind, vector, pool in (("fresh_empty_stream", base_vector, base_pool),
                                                        (mode, context_vector, context_pool)):
                        scores = {identity: cosine(vector, proto) for identity, proto in prototypes.items()}
                        top_id = max(scores, key=scores.get) if scores else None
                        top_score = scores[top_id] if top_id is not None else None
                        accepted = top_id if top_score is not None and top_score >= threshold else None
                        target_values.setdefault(target_key, {"fold_id": fold_id, "target_job_id": target["target_job_id"],
                            "speaker_id": speaker_id, "known_in_registry8": known, "actual_speech_sec": pool["speech_duration_sec"],
                            "decisions": {}})["decisions"][context_kind] = {
                                "top_identity": top_id, "top_score": top_score, "accepted_identity": accepted,
                                "identity_cosine_to_fresh": cosine(base_vector, vector),
                                "genuine_score": scores.get(speaker_id),
                                "_embedding": vector,
                            }

        context_feature_results = []
        for (fold_id_key, representation, mode), values in sorted(context_feature_summary.items()):
            context_feature_results.append({"fold_id": fold_id_key, "representation": representation,
                "context_mode": mode, "target_count": values["targets"],
                "mean_pooled_cosine_to_fresh": float(np.mean(values["cosines"])) if values["cosines"] else None,
                "pooled_cosine_quantiles": quantiles(values["cosines"]),
                "maximum_absolute_frame_difference": float(np.max(values["max_abs"])) if values["max_abs"] else None,
                "bit_identical_target_features": values["bit_exact"],
                "target_features_compared": values["targets"]})
        for target in target_values.values():
            fresh = target["decisions"].get("fresh_empty_stream", {})
            for mode in CONTEXT_MODES:
                context = target["decisions"].get(mode)
                if context is None:
                    continue
                context["top_identity_changed"] = context.get("top_identity") != fresh.get("top_identity")
                context["acceptance_changed"] = context.get("accepted_identity") != fresh.get("accepted_identity")
        model_decisions = []
        for mode in CONTEXT_MODES:
            items = [row for row in target_values.values() if mode in row["decisions"]]
            known = [row for row in items if row["known_in_registry8"]]
            unknown = [row for row in items if not row["known_in_registry8"]]
            vectors = [(row["speaker_id"], row["decisions"][mode]["_embedding"]) for row in items]
            within, between = [], []
            for left in range(len(vectors)):
                for right in range(left + 1, len(vectors)):
                    score = cosine(vectors[left][1], vectors[right][1])
                    (within if vectors[left][0] == vectors[right][0] else between).append(score)
            known_accept = sum(row["decisions"][mode]["accepted_identity"] == row["speaker_id"] for row in known)
            unknown_fa = sum(row["decisions"][mode]["accepted_identity"] is not None for row in unknown)
            model_decisions.append({"fold_id": fold_id, "model_key": selected_key, "context_mode": mode,
                "registry_size": 8, "threshold": threshold,
                "known_correct_accepts": known_accept, "known_queries": len(known),
                "known_correct_accept_rate": known_accept / len(known) if known else None,
                "unknown_false_accepts": unknown_fa, "unknown_queries": len(unknown),
                "unknown_false_accept_rate": unknown_fa / len(unknown) if unknown else None,
                "top_identity_changes_vs_fresh": sum(row["decisions"][mode]["top_identity_changed"] for row in items),
                "acceptance_changes_vs_fresh": sum(row["decisions"][mode]["acceptance_changed"] for row in items),
                "mean_identity_cosine_to_fresh": float(np.mean([row["decisions"][mode]["identity_cosine_to_fresh"] for row in items])) if items else None,
                "within_speaker_embedding_cosine": quantiles(within),
                "between_speaker_embedding_cosine": quantiles(between),
                "known_genuine_score_distribution": quantiles([row["decisions"][mode]["genuine_score"] for row in known
                                                                 if row["decisions"][mode]["genuine_score"] is not None]),
                "unknown_top_score_distribution": quantiles([row["decisions"][mode]["top_score"] for row in unknown
                                                               if row["decisions"][mode]["top_score"] is not None]),
                "target_count": len(items), "target_speakers": len({row["speaker_id"] for row in items})})
        profile_summaries = {}
        for mode in ("aosc_profile_import",):
            profile_job_tsv_name = manifest["context_stress_protocol"]["aosc_profile_job_tsv_by_fold"][fold_id]
            profile_job_tsv = args.cache_dir / profile_job_tsv_name
            profile_path = profile_job_tsv.parent / f"aosc_context_profile_{fold_id}.bin"
            sidecar = Path(str(profile_path) + ".json")
            if sidecar.exists():
                profile_summaries[mode] = json.loads(sidecar.read_text())
        results.append({"fold_id": fold_id, "selected_model_key": selected_key,
                        "feature_context_comparison": context_feature_results,
                        "identity_decision_comparison": model_decisions,
                        "aosc_context_profile": profile_summaries})

    output = {"schema": "nemo-speech.voiceid-task09b-context.v1",
              "protocol": {"contexts": CONTEXT_MODES,
                  "context_audio": "same query speech follows either 2.56s silence or one unrelated training-speaker utterance; AOSC mode imports a fixed profile made only from eight training speakers",
                  "target_speakers": "ten per fold (five from registry8 and five unknown), two independent query utterances each",
                  "features": "three taps compared at matching target-relative 80ms timestamps; fresh baseline comes from the bulk extraction record",
                  "threshold": "fixed selected-head threshold calibrated on calibration speakers for 10s enrollment and 3s query; reused unchanged under all context modes"},
              "folds": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "folds": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"analyze_voiceid_context: {exc}", file=sys.stderr)
        raise
