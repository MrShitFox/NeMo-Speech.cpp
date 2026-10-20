#!/usr/bin/env python3
"""Create a deterministic, identity-agnostic RMS-normalized audio ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import wave
from pathlib import Path
from typing import Any

import numpy as np

from prepare_vctk_voiceid import vad_mask_from_pcm16


SCHEMA = "nemo-speech.voiceid-task09c-rms-normalized-audio.v1"
TARGET_RMS_DBFS = -26.0
GAIN_LIMIT_DB = 6.0
PEAK_LIMIT_DBFS = -1.0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jobs(path: Path) -> list[dict[str, str]]:
    jobs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 10:
            raise ValueError(f"expected ten TSV fields in {path}, got {len(fields)}")
        jobs.append(dict(zip(("job_id", "speaker_id", "utterance_id", "mic", "source_path",
                              "source_sha256", "converted_sha256", "vad_sha256", "wav_path", "vad_path"), fields)))
    return jobs


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_one(job: dict[str, str], output_dir: Path) -> tuple[dict[str, Any], str]:
    source = Path(job["wav_path"])
    with wave.open(str(source), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16000:
            raise ValueError(f"expected mono PCM16/16k WAV: {source}")
        sample_count = handle.getnframes()
        pcm = handle.readframes(sample_count)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    target = 10.0 ** (TARGET_RMS_DBFS / 20.0)
    gain = target / rms if rms > 0 else 1.0
    limit = 10.0 ** (GAIN_LIMIT_DB / 20.0)
    gain = min(max(gain, 1.0 / limit), limit)
    peak_limit = 10.0 ** (PEAK_LIMIT_DBFS / 20.0)
    if peak > 0 and peak * gain > peak_limit:
        gain = peak_limit / peak
    normalized = np.rint(samples * gain * 32768.0).clip(-32768, 32767).astype("<i2")
    normalized_pcm = normalized.tobytes()
    mask = vad_mask_from_pcm16(normalized_pcm)
    normalized_id = hashlib.sha256(f"{SCHEMA}|{job['job_id']}".encode("utf-8")).hexdigest()[:24]
    wav_path = output_dir / f"{normalized_id}.wav"
    vad_path = output_dir / f"{normalized_id}.vad10.bin"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(normalized_pcm)
    vad_path.write_bytes(mask)
    converted_hash = sha256_file(wav_path)
    vad_hash = sha256_bytes(mask)
    row = {
        "source_job_id": job["job_id"], "normalized_job_id": normalized_id,
        "speaker_id": job["speaker_id"], "utterance_id": job["utterance_id"], "mic": job["mic"],
        "source_sha256": job["source_sha256"], "source_converted_sha256": job["converted_sha256"],
        "normalized_converted_sha256": converted_hash, "normalized_vad_sha256": vad_hash,
        "wav_path": str(wav_path), "vad_path": str(vad_path), "gain_db": 20.0 * math.log10(max(gain, 1e-12)),
        "input_rms_dbfs": 20.0 * math.log10(max(rms, 1e-12)),
        "input_peak_dbfs": 20.0 * math.log10(max(peak, 1e-12)),
        "output_rms_dbfs": 20.0 * math.log10(max(float(np.sqrt(np.mean((normalized.astype(np.float64) / 32768.0) ** 2))), 1e-12)),
        "output_peak_dbfs": 20.0 * math.log10(max(float(np.max(np.abs(normalized)) / 32768.0), 1e-12)),
        "speech_duration_sec": float(sum(mask) * 0.01),
    }
    tsv = "\t".join((normalized_id, job["speaker_id"], job["utterance_id"], job["mic"],
                       job["source_path"], job["source_sha256"], converted_hash, vad_hash,
                       str(wav_path), str(vad_path)))
    return row, tsv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jobs-tsv", type=Path, required=True)
    parser.add_argument("--additional-jobs-tsv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    jobs = read_jobs(args.base_jobs_tsv) + read_jobs(args.additional_jobs_tsv)
    ids = [row["job_id"] for row in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("input job lists contain duplicate original job IDs")
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
    results_by_source: dict[str, tuple[dict[str, Any], str]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(normalize_one, job, args.audio_dir.resolve()) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            row, tsv = future.result()
            results_by_source[row["source_job_id"]] = (row, tsv)
            if index % 500 == 0:
                print(f"normalized {index}/{len(jobs)} audio clips", file=sys.stderr, flush=True)
    ordered = [results_by_source[key] for key in sorted(results_by_source)]
    rows = [item[0] for item in ordered]
    tsv_rows = [item[1] for item in ordered]
    args.jobs_tsv.write_text(
        "# job_id\tspeaker_id\tutterance_id\tmic\tsource_path\tsource_sha256\tconverted_sha256\tvad_sha256\twav_path\tvad_path\n"
        + "\n".join(tsv_rows) + "\n", encoding="utf-8")
    gains = np.asarray([row["gain_db"] for row in rows], dtype=np.float64)
    input_rms = np.asarray([row["input_rms_dbfs"] for row in rows], dtype=np.float64)
    output_rms = np.asarray([row["output_rms_dbfs"] for row in rows], dtype=np.float64)
    summary = {
        "schema": SCHEMA,
        "policy": {"target_rms_dbfs": TARGET_RMS_DBFS, "gain_limit_db": [-GAIN_LIMIT_DB, GAIN_LIMIT_DB],
                   "peak_limit_dbfs": PEAK_LIMIT_DBFS,
                   "scope": "same per-utterance waveform policy for every identity and microphone; no speaker labels or evaluation outcomes used",
                   "vad": "recomputed using the existing Task 09B deterministic 10ms VAD on normalized PCM16"},
        "input_job_count": len(jobs), "unique_original_job_ids": len(set(ids)),
        "base_job_list_sha256": sha256_file(args.base_jobs_tsv),
        "additional_job_list_sha256": sha256_file(args.additional_jobs_tsv),
        "normalized_jobs_tsv": str(args.jobs_tsv.resolve()),
        "audio_cache_dir": str(args.audio_dir.resolve()),
        "gain_db": {"min": float(gains.min()), "median": float(np.median(gains)), "mean": float(gains.mean()),
                    "max": float(gains.max()), "p10_p90": np.quantile(gains, [.1, .9]).tolist()},
        "input_rms_dbfs": {"min": float(input_rms.min()), "median": float(np.median(input_rms)),
                           "mean": float(input_rms.mean()), "max": float(input_rms.max())},
        "output_rms_dbfs": {"min": float(output_rms.min()), "median": float(np.median(output_rms)),
                            "mean": float(output_rms.mean()), "max": float(output_rms.max())},
        "mapping_rule": "normalized_job_id = first 24 hex digits of SHA256(schema + vertical bar + original job ID)",
    }
    write_json(args.summary, summary)
    print(json.dumps({"input_job_count": len(jobs), "output": str(args.summary),
                      "gain_db_median": summary["gain_db"]["median"],
                      "output_rms_dbfs_median": summary["output_rms_dbfs"]["median"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"normalize_task09c_audio: {exc}", file=sys.stderr)
        raise
