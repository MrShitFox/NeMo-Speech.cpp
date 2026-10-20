#!/usr/bin/env python3
"""Run the native Task 09B extractor and validate/cache bounded feature pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


HEADER = struct.Struct("<8sHHIQIIIIII")
MAGIC = b"NVFTR001"
RECORD_SCHEMA = "nemo-speech.voiceid-feature.v1"


def read_feature(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, str]:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"truncated feature header: {path}")
    magic, version, header_size, dim, frames, frame_period_us, sample_rate, dtype, endian, mask_bytes, reserved = HEADER.unpack_from(data)
    if magic != MAGIC or version != 1 or header_size != HEADER.size:
        raise ValueError(f"unsupported feature record header: {path}")
    if dim < 1 or frames < 1 or frame_period_us != 80000 or sample_rate != 16000:
        raise ValueError(f"invalid feature shape/timing in {path}: dim={dim}, frames={frames}")
    if dtype != 1 or endian != 0x01020304 or mask_bytes != frames or reserved != 0:
        raise ValueError(f"unsupported feature encoding in {path}")
    frame_bytes = 12 + 4 * dim
    expected = HEADER.size + frames * frame_bytes
    if len(data) != expected:
        raise ValueError(f"feature record length mismatch in {path}: expected {expected}, got {len(data)}")
    ids = np.empty(frames, dtype=np.int64)
    speech_bits = np.empty(frames, dtype=np.uint8)
    values = np.empty((frames, dim), dtype=np.float32)
    offset = HEADER.size
    for index in range(frames):
        ids[index] = struct.unpack_from("<Q", data, offset)[0]
        speech_bits[index] = data[offset + 8]
        if data[offset + 9:offset + 12] != b"\0\0\0":
            raise ValueError(f"nonzero reserved frame bytes in {path}")
        offset += 12
        values[index] = np.frombuffer(data, dtype="<f4", count=dim, offset=offset)
        offset += dim * 4
    if np.any(np.diff(ids) != 1) or ids[0] < 0:
        raise ValueError(f"non-contiguous or invalid frame timestamps in {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite value in {path}")
    return dim, ids, speech_bits, values, hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bitcount8(values: np.ndarray) -> np.ndarray:
    # np.bit_count is unavailable in some NumPy builds; a tiny lookup table is stable.
    table = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    return table[values]


def load_jobs(path: Path) -> list[dict[str, str]]:
    jobs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) not in (10, 15):
            raise ValueError(f"expected ten or fifteen TSV columns, got {len(fields)}")
        row = dict(zip(("job_id", "speaker_id", "utterance_id", "mic", "source_path", "source_sha256",
                        "converted_sha256", "vad_sha256", "wav_path", "vad_path"), fields[:10]))
        if len(fields) == 15:
            row.update(dict(zip(("context_wav_path", "context_mode", "context_speaker_id", "context_sha256",
                                 "aosc_profile_path"), fields[10:])))
        jobs.append(row)
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("job list contains duplicate IDs")
    return jobs


def validate_and_pool(feature_dir: Path, jobs: list[dict[str, str]], representations: list[str],
                      pooled_path: Path, metadata_path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {(job["job_id"], representation) for job in jobs for representation in representations}
    actual = {(row.get("job_id"), row.get("feature_representation")) for row in records}
    if len(actual) != len(records) or actual != expected:
        raise ValueError(f"metadata coverage mismatch: expected {len(expected)}, got {len(records)} unique/{len(actual)}")
    by_key = {(row["job_id"], row["feature_representation"]): row for row in records}
    model_hashes = {row.get("model_gguf_fingerprint") for row in records}
    backends = {row.get("inference_backend") for row in records}
    devices = {row.get("device_name") for row in records}
    if len(model_hashes) != 1 or None in model_hashes or len(backends) != 1 or len(devices) != 1:
        raise ValueError("mixed or missing model/backend metadata in feature records")
    dimensions: dict[str, int] = {}
    pooled: dict[str, dict[str, list[Any]]] = {
        representation: {"mean": [], "std": [], "frame_l2_mean": [], "speech_frames": [], "speech_duration_sec": [], "frame_count": []}
        for representation in representations
    }
    job_order = [job["job_id"] for job in jobs]
    for job in jobs:
        for representation in representations:
            row = by_key[(job["job_id"], representation)]
            if (row.get("schema") != RECORD_SCHEMA or row.get("source_sha256") != job["source_sha256"] or
                    row.get("speaker_id") != job["speaker_id"] or row.get("source_utterance_id") != job["utterance_id"] or
                    row.get("source_microphone") != job["mic"] or row.get("converted_audio_sha256") != job["converted_sha256"] or
                    row.get("vad_mask_sha256") != job["vad_sha256"]):
                raise ValueError(f"metadata source/schema mismatch for {job['job_id']} / {representation}")
            if Path(row["record_file"]).name != row["record_file"]:
                raise ValueError(f"feature record path escapes output directory: {row['record_file']}")
            record = feature_dir / row["record_file"]
            dim, ids, vad_bits, values, record_sha256 = read_feature(record)
            if dim != int(row["feature_dimension"]) or len(ids) != int(row["frame_count"]):
                raise ValueError(f"metadata shape mismatch for {record}")
            if row.get("context_initialization") != job.get("context_mode", "fresh_empty_stream"):
                raise ValueError(f"context metadata mismatch for {job['job_id']}")
            expected_context_sha = job.get("context_sha256") or None
            expected_context_speaker = job.get("context_speaker_id") or None
            if (row.get("context_audio_sha256") != expected_context_sha or
                    row.get("context_speaker_id") != expected_context_speaker):
                raise ValueError(f"context source metadata mismatch for {job['job_id']}")
            profile_path = job.get("aosc_profile_path") or ""
            if profile_path:
                if row.get("aosc_profile_sha256") != sha256_file(Path(profile_path)):
                    raise ValueError(f"AOSC profile fingerprint mismatch for {job['job_id']}")
            elif row.get("aosc_profile_sha256") is not None:
                raise ValueError(f"unexpected AOSC profile in fresh/context audio job {job['job_id']}")
            row["feature_file_sha256"] = record_sha256
            if row.get("frame_period_sec") != 0.08 or row.get("dtype") != "float32" or row.get("byte_order") != "little":
                raise ValueError(f"unsupported feature timing/dtype in metadata for {job['job_id']}")
            if representation in dimensions and dimensions[representation] != dim:
                raise ValueError(f"feature dimension changed for representation {representation}")
            dimensions[representation] = dim
            counts = bitcount8(vad_bits)
            speech = counts >= 4
            chosen = values[speech]
            if chosen.shape[0]:
                norms = np.linalg.norm(chosen, axis=1, keepdims=True)
                normalized = chosen / np.maximum(norms, 1e-12)
                mean = chosen.mean(axis=0, dtype=np.float64).astype(np.float32)
                std = chosen.std(axis=0, dtype=np.float64).astype(np.float32)
                frame_l2_mean = normalized.mean(axis=0, dtype=np.float64).astype(np.float32)
            else:
                mean = np.zeros(dim, dtype=np.float32)
                std = np.zeros(dim, dtype=np.float32)
                frame_l2_mean = np.zeros(dim, dtype=np.float32)
            item = pooled[representation]
            item["mean"].append(mean)
            item["std"].append(std)
            item["frame_l2_mean"].append(frame_l2_mean)
            item["speech_frames"].append(int(speech.sum()))
            item["speech_duration_sec"].append(float(counts[speech].sum()) * 0.01)
            item["frame_count"].append(int(len(ids)))
    arrays: dict[str, Any] = {"job_ids": np.asarray(job_order, dtype="U32")}
    for representation, values in pooled.items():
        for field, rows in values.items():
            arrays[f"{representation}__{field}"] = np.asarray(rows, dtype=np.float32 if field in ("mean", "std", "frame_l2_mean") else np.float32)
    pooled_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pooled_path, **arrays)
    metadata_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    summary = {
        "schema": "nemo-speech.voiceid-task09b-pooled-features.v1",
        "job_count": len(jobs),
        "feature_records": len(records),
        "representations": {name: {"dimension": int(pooled[name]["mean"][0].shape[0]),
                                   "zero_speech_mask_jobs": sum(count == 0 for count in pooled[name]["speech_frames"])}
                            for name in representations},
        "pooled_feature_file": str(pooled_path),
        "record_shape_validation": "passed; headers, byte order, lengths, frame timestamps, masks, finite values, and metadata checked",
        "source_file_hash_validation": "passed against split job catalog",
        "model_fingerprint": next(iter(model_hashes)),
        "inference_backend": next(iter(backends)),
        "device_name": next(iter(devices)),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/Nemotron-3-Diarization.q8_0.gguf"))
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--representations", default="pre_encoder,encoder.layer.7,encoder_output,encoder_proj")
    parser.add_argument("--device", choices=("cpu", "vulkan"), default="vulkan")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--parity-first", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    jobs = load_jobs(args.jobs_tsv)
    representations = [item for item in args.representations.split(",") if item]
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.feature_dir / "features.jsonl"
    if not args.validate_only:
        if args.probe is None:
            parser.error("--probe is required unless --validate-only is set")
        metadata_path.unlink(missing_ok=True)
        (args.feature_dir / "extractor_summary.json").unlink(missing_ok=True)
        command = [str(args.probe), "--jobs", str(args.jobs_tsv), "--model", str(args.model),
                   "--output-dir", str(args.feature_dir), "--device", args.device,
                   "--gpu-device", str(args.gpu_device), "--representations", ",".join(representations),
                   "--parity-first", str(args.parity_first)]
        print("running native extractor:", " ".join(command), file=sys.stderr, flush=True)
        env = dict(os.environ)
        if args.device == "vulkan":
            env.setdefault("VK_DRIVER_FILES", "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json")
        subprocess.run(command, check=True, env=env)
    summary = validate_and_pool(args.feature_dir, jobs, representations,
                                args.feature_dir / "pooled_features.npz", metadata_path)
    extractor_summary = json.loads((args.feature_dir / "extractor_summary.json").read_text())
    if args.device == "vulkan" and extractor_summary.get("actual_backend") != "Vulkan":
        raise RuntimeError("explicit Vulkan request did not execute on Vulkan")
    summary["extractor"] = extractor_summary
    output = args.feature_dir / "feature_validation.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"extract_voiceid_features: {exc}", file=sys.stderr)
        raise
