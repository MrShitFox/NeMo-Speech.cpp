#!/usr/bin/env python3
"""Research-only persisted prototype registry for unseen VCTK identities.

This exercises registration and matching across separate processes using
cached frozen Nemotron features and a trained research head. It is not a
production inference API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_voiceid_head import Method
from task09b_voiceid import FeatureData, cosine, load_data


SCHEMA = "nemo-speech.voiceid-task09b-research-registry.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_head(args: argparse.Namespace, manifest: dict[str, Any], assets: dict[str, dict[str, Any]],
              pooled: dict[str, np.ndarray]) -> tuple[Method, dict[str, Any], dict[str, Any]]:
    training = json.loads(args.training_results.read_text(encoding="utf-8"))
    models = {row["model_key"]: row for row in training["models"]}
    fold = next(row for row in manifest["folds"] if row["fold_id"] == args.fold_id)
    model_key = training["selected_models_by_fold"][args.fold_id]
    model = models[model_key]
    data = FeatureData(assets, pooled, model["representation"])
    head = Method(model["objective"], model["representation"], data,
                  fold["train_utterance_job_ids"], args.run_dir / model["portable_export"])
    return head, fold, model


def source_jobs(fold: dict[str, Any], source_speaker: str, role: str) -> list[str]:
    people = fold["evaluation_people"]
    if source_speaker not in people:
        raise ValueError(f"source speaker is not an unseen evaluation identity in {fold['fold_id']}: {source_speaker}")
    person = people[source_speaker]
    if role == "enrollment":
        return list(person["enrollment_job_ids"])
    if role == "query":
        return list(person["query_job_ids"])
    raise ValueError(role)


def read_registry(path: Path, expected: dict[str, str] | None = None) -> dict[str, Any]:
    if not path.exists():
        if expected is None:
            raise FileNotFoundError(path)
        return {"schema": SCHEMA, **expected, "embedding_dimension": 128, "identities": {}}
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema") != SCHEMA:
        raise ValueError("unsupported research registry schema")
    if expected is not None:
        for key, value in expected.items():
            if registry.get(key) != value:
                raise ValueError(f"registry {key} does not match selected head/base model")
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(command: argparse.ArgumentParser, need_features: bool = True) -> None:
        command.add_argument("--registry", type=Path, required=True)
        if need_features:
            command.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
            command.add_argument("--pooled-features", type=Path, required=True)
            command.add_argument("--training-results", type=Path, default=Path("tests/results/voiceid_task09b_training.json"))
            command.add_argument("--run-dir", type=Path, required=True)
            command.add_argument("--fold-id", required=True)

    register = sub.add_parser("register", help="create one unseen identity prototype")
    add_shared(register)
    register.add_argument("--source-speaker", required=True, help="VCTK enrollment-audio selector; never used as registry key")
    register.add_argument("--identity", required=True, help="arbitrary external registry ID")
    register.add_argument("--enrollment-sec", type=float, choices=(5, 10, 15, 30, 60), default=10)
    register.add_argument("--replace", action="store_true")

    query = sub.add_parser("query", help="match independent unseen query audio against a saved registry")
    add_shared(query)
    query.add_argument("--source-speaker", required=True, help="VCTK query-audio selector; labels are not used to score")
    query.add_argument("--query-sec", type=float, choices=(1, 3, 7, 15), default=3)
    query.add_argument("--threshold", type=float, required=True, help="frozen threshold from calibration speakers")

    remove = sub.add_parser("remove", help="remove an external identity without retraining")
    add_shared(remove, need_features=False)
    remove.add_argument("--identity", required=True)

    args = parser.parse_args()
    if args.command == "remove":
        registry = read_registry(args.registry)
        if args.identity not in registry["identities"]:
            raise KeyError(f"identity is not registered: {args.identity}")
        del registry["identities"][args.identity]
        atomic_json(args.registry, registry)
        print(json.dumps({"removed": args.identity, "registered_identities": len(registry["identities"])}))
        return 0

    manifest, assets, pooled = load_data(args.manifest, args.pooled_features)
    head, fold, model = load_head(args, manifest, assets, pooled)
    metadata_path = args.pooled_features.parent / "features.jsonl"
    first_metadata = json.loads(metadata_path.open(encoding="utf-8").readline())
    export_path = args.run_dir / model["portable_export"]
    expected = {
        "fold_id": args.fold_id,
        "model_key": model["model_key"],
        "representation": model["representation"],
        "objective": model["objective"],
        "base_model_fingerprint": first_metadata["model_gguf_fingerprint"],
        "head_sha256": sha256_file(export_path),
    }
    registry = read_registry(args.registry, expected)
    data = head.data

    if args.command == "register":
        if not args.identity.strip():
            raise ValueError("external identity ID must not be empty")
        if args.identity in registry["identities"] and not args.replace:
            raise ValueError(f"identity already registered; pass --replace to refresh it: {args.identity}")
        jobs = source_jobs(fold, args.source_speaker, "enrollment")
        utterances = [assets[job_id]["utterance_id"] for job_id in jobs]
        if len(utterances) != len(set(utterances)):
            raise ValueError("registration contains duplicate source utterance groups")
        pooled_audio = data.pool(jobs, args.enrollment_sec)
        if pooled_audio["speech_duration_sec"] < args.enrollment_sec:
            raise ValueError("not enough VAD-active enrollment speech for requested budget")
        embedding = head.embed_many([pooled_audio])[0].astype(np.float32)
        selected_job_ids = [str(data.pooled["job_ids"][index])
                            for index in pooled_audio["selected_job_indices"]]
        registry["identities"][args.identity] = {
            "prototype": embedding.tolist(),
            "embedding_norm": float(np.linalg.norm(embedding)),
            "actual_enrollment_speech_sec": pooled_audio["speech_duration_sec"],
            "source_utterance_count": len(pooled_audio["selected_job_indices"]),
            "source_audio_hashes": sorted(assets[job_id]["source_sha256"] for job_id in selected_job_ids),
            "pooling": "single pooled prototype over VAD-active frames",
        }
        atomic_json(args.registry, registry)
        print(json.dumps({"registered_identity": args.identity,
                          "source_selector_used_only_for_audio": args.source_speaker,
                          "actual_enrollment_speech_sec": pooled_audio["speech_duration_sec"],
                          "embedding_dimension": len(embedding),
                          "native_channel_index": None,
                          "registered_identities": len(registry["identities"])}))
        return 0

    jobs = source_jobs(fold, args.source_speaker, "query")
    pooled_query = data.pool(jobs, args.query_sec)
    if pooled_query["speech_duration_sec"] < args.query_sec:
        raise ValueError("not enough VAD-active query speech for requested budget")
    query_embedding = head.embed_many([pooled_query])[0]
    scores = {identity: cosine(query_embedding, np.asarray(record["prototype"], dtype=np.float32))
              for identity, record in registry["identities"].items()}
    best_identity = max(scores, key=scores.get) if scores else None
    best_score = scores.get(best_identity) if best_identity is not None else None
    accepted = best_identity if best_score is not None and best_score >= args.threshold else None
    print(json.dumps({"source_selector_used_only_for_audio": args.source_speaker,
                      "actual_query_speech_sec": pooled_query["speech_duration_sec"],
                      "registry_size": len(scores), "scores": scores,
                      "best_identity": best_identity, "best_cosine": best_score,
                      "threshold": args.threshold, "decision": accepted or "UNKNOWN",
                      "native_channel_index": None}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"voiceid_registry_research: {exc}", file=sys.stderr)
        raise
