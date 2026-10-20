#!/usr/bin/env python3
"""Add matched mic2 enrollment and calibration-query assets to Task 09B folds.

The original Task 09B manifest remains authoritative for all speaker and
utterance splits. This extension only prepares mic2 renderings of already
assigned enrollment clips and up to four already-assigned query clips.
Converted audio, VAD masks, and the native extractor TSV stay in the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from prepare_vctk_voiceid import Preparer, audio_root_for, flac_streaminfo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("VCTK-Corpus-0.92"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/manifests/task09c_vctk_microphone_additions.json"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets_by_id = {item["job_id"]: item for item in manifest["assets"]}
    base_mic2_by_source = {item["source_path"]: item for item in manifest["assets"] if item["mic"] == "mic2"}
    audio_root = audio_root_for(args.dataset_root.resolve())
    preparer = Preparer(audio_root, args.cache_dir.resolve())
    additional_assets: dict[str, dict[str, Any]] = {}
    mic2_by_source: dict[str, dict[str, Any]] = {}
    prepared_by_source: dict[str, dict[str, Any]] = {}
    missing_sources: set[str] = set()
    enrollment_asset_ids: set[str] = set()
    query_asset_ids: set[str] = set()

    # Predeclare all mic2 source paths, then let a small fixed worker pool do
    # independent FFmpeg conversions. The assignments are still read only
    # from the frozen Task 09B manifest.
    required_mic1_jobs: dict[str, dict[str, Any]] = {}
    for fold in manifest["folds"]:
        for role in ("selection_people", "calibration_people", "evaluation_people"):
            for person in fold[role].values():
                for job_id in person["enrollment_job_ids"]:
                    required_mic1_jobs[job_id] = assets_by_id[job_id]
                paired_count = 0
                for job_id in person["query_job_ids"]:
                    mic1 = assets_by_id[job_id]
                    source_path = mic1["source_path"].replace("_mic1.flac", "_mic2.flac")
                    if (audio_root / source_path).is_file():
                        required_mic1_jobs[job_id] = mic1
                        paired_count += 1
                    if paired_count == 4:
                        break
    tasks: dict[str, tuple[dict[str, Any], float]] = {}
    for mic1 in required_mic1_jobs.values():
        source_path = mic1["source_path"].replace("_mic1.flac", "_mic2.flac")
        if source_path in base_mic2_by_source or source_path in tasks:
            continue
        source = audio_root / source_path
        if not source.is_file():
            missing_sources.add(source_path)
            continue
        rate, channels, bits, duration = flac_streaminfo(source)
        if rate != 48000 or channels != 1 or bits != 16:
            raise RuntimeError(f"unexpected mic2 source format: {source_path} ({rate} Hz, {channels} ch, {bits} bit)")
        tasks[source_path] = (mic1, duration)

    thread_state = threading.local()

    def prepare_source(source_path: str, mic1: dict[str, Any], duration: float) -> tuple[str, dict[str, Any]]:
        worker_preparer = getattr(thread_state, "preparer", None)
        if worker_preparer is None:
            worker_preparer = Preparer(audio_root, args.cache_dir.resolve())
            thread_state.preparer = worker_preparer
        return source_path, worker_preparer.prepare(source_path, {
            "speaker_id": mic1["speaker_id"], "utterance_id": mic1["utterance_id"], "mic": "mic2",
            "source_sample_rate_hz": 48000, "source_channels": 1, "source_duration_sec": duration,
        })

    prepared_total = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(prepare_source, source_path, mic1, duration)
                   for source_path, (mic1, duration) in tasks.items()]
        for future in as_completed(futures):
            source_path, prepared = future.result()
            mic1 = tasks[source_path][0]
            record = {
                "job_id": prepared["source_sha256"][:24], "speaker_id": mic1["speaker_id"],
                "utterance_id": mic1["utterance_id"], "mic": "mic2", "source_path": source_path,
                "source_sha256": prepared["source_sha256"], "converted_sha256": prepared["converted_sha256"],
                "vad_sha256": prepared["vad_sha256"], "speech_duration_sec": prepared["speech_duration_sec"],
                "source_duration_sec": tasks[source_path][1], "audio_file": prepared["audio_file"],
                "vad_mask_file": prepared["vad_mask_file"],
                "paired_mic1_job_id": next(job_id for job_id, item in required_mic1_jobs.items()
                                            if item["source_path"].replace("_mic1.flac", "_mic2.flac") == source_path),
            }
            prepared_by_source[source_path] = record
            additional_assets[record["job_id"]] = record
            prepared_total += 1
            if prepared_total % 250 == 0:
                print(f"prepared paired mic2 sources: {prepared_total}/{len(tasks)}", file=sys.stderr, flush=True)

    def paired_mic2(mic1_job_id: str) -> dict[str, Any] | None:
        mic1 = assets_by_id[mic1_job_id]
        source_path = mic1["source_path"].replace("_mic1.flac", "_mic2.flac")
        if source_path in mic2_by_source:
            return mic2_by_source[source_path]
        existing = base_mic2_by_source.get(source_path)
        if existing is not None:
            mic2_by_source[source_path] = existing
            return existing
        source = audio_root / source_path
        if not source.is_file():
            missing_sources.add(source_path)
            return None
        record = prepared_by_source[source_path]
        record["paired_mic1_job_id"] = mic1_job_id
        mic2_by_source[source_path] = record
        return record

    fold_additions: list[dict[str, Any]] = []
    for fold in manifest["folds"]:
        fold_row: dict[str, Any] = {"fold_id": fold["fold_id"]}
        for role in ("selection_people", "calibration_people", "evaluation_people"):
            people: dict[str, Any] = {}
            for speaker, person in fold[role].items():
                mic2_enrollment = [paired_mic2(job_id) for job_id in person["enrollment_job_ids"]]
                mic2_enrollment = [item for item in mic2_enrollment if item is not None]
                enrollment_asset_ids.update(item["job_id"] for item in mic2_enrollment)
                query_pairs = []
                for mic1_job_id in person["query_job_ids"]:
                    paired = paired_mic2(mic1_job_id)
                    if paired is not None:
                        query_asset_ids.add(paired["job_id"])
                        query_pairs.append({"mic1_job_id": mic1_job_id, "mic2_job_id": paired["job_id"],
                                            "utterance_id": assets_by_id[mic1_job_id]["utterance_id"]})
                    if len(query_pairs) == 4:
                        break
                if role == "evaluation_people":
                    declared = person.get("query_mic2_job_ids", [])
                    if declared and declared != [pair["mic2_job_id"] for pair in query_pairs]:
                        raise RuntimeError(f"Task 09B mic2 pairing changed for {fold['fold_id']} / {speaker}")
                people[speaker] = {
                    "mic2_enrollment_job_ids": [item["job_id"] for item in mic2_enrollment],
                    "mic2_enrollment_utterance_ids": [item["utterance_id"] for item in mic2_enrollment],
                    "paired_query_job_ids": query_pairs,
                    "mic2_enrollment_source_count": len(mic2_enrollment),
                    "mic2_enrollment_requested_source_count": len(person["enrollment_job_ids"]),
                }
            fold_row[role] = people
        fold_additions.append(fold_row)

    args.jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.jobs_tsv.open("w", encoding="utf-8") as output:
        output.write("# job_id\tspeaker_id\tutterance_id\tmic\tsource_path\tsource_sha256\tconverted_sha256\tvad_sha256\twav_path\tvad_path\n")
        for item in sorted(additional_assets.values(), key=lambda row: row["job_id"]):
            wav_path = args.cache_dir.resolve() / item["audio_file"]
            vad_path = args.cache_dir.resolve() / item["vad_mask_file"]
            output.write("\t".join(str(value) for value in (
                item["job_id"], item["speaker_id"], item["utterance_id"], item["mic"], item["source_path"],
                item["source_sha256"], item["converted_sha256"], item["vad_sha256"], wav_path, vad_path,
            )) + "\n")

    result = {
        "schema": "nemo-speech.voiceid-task09c-microphone-additions.v1",
        "base_manifest": str(manifest_path),
        "base_manifest_sha256": sha256_file(manifest_path),
        "pairing_rule": "Task 09B utterance assignments retained; first four query utterances with an available mic2 rendering are paired; enrollment mic2 files retain their mic1 utterance IDs",
        "microphone_enrollment_assets": len(enrollment_asset_ids),
        "microphone_query_assets": len(query_asset_ids),
        "new_enrollment_assets": len(enrollment_asset_ids & set(additional_assets)),
        "new_query_assets": len(query_asset_ids & set(additional_assets)),
        "paired_query_assets_reused_from_task09b": sum(1 for item in mic2_by_source.values() if item.get("job_id") in assets_by_id),
        "new_assets": sorted(additional_assets.values(), key=lambda row: row["job_id"]),
        "missing_mic2_source_paths": sorted(missing_sources),
        "folds": fold_additions,
        "vctk_audio_root": str(audio_root),
        "cache_dir": str(args.cache_dir.resolve()),
        "jobs_tsv": str(args.jobs_tsv.resolve()),
        "ffmpeg_version": preparer.ffmpeg_version,
    }
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "jobs_tsv": str(args.jobs_tsv),
                      "new_assets": len(additional_assets), "missing_sources": len(missing_sources),
                      "folds": len(fold_additions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"prepare_task09c_microphones: {exc}", file=sys.stderr)
        raise
