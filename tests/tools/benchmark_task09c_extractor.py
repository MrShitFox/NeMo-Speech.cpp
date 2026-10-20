#!/usr/bin/env python3
"""Time a deterministic cached-job subset through the native Vulkan feature probe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

from extract_voiceid_features import load_jobs


def cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representations", default="pre_encoder")
    parser.add_argument("--gpu-device", type=int, default=0)
    args = parser.parse_args()

    jobs = load_jobs(args.jobs_tsv)
    total_audio_sec = 0.0
    for job in jobs:
        with wave.open(job["wav_path"], "rb") as stream:
            total_audio_sec += stream.getnframes() / stream.getframerate()
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    command = [str(args.probe.resolve()), "--jobs", str(args.jobs_tsv.resolve()),
               "--model", str(args.model.resolve()), "--output-dir", str(args.feature_dir.resolve()),
               "--device", "vulkan", "--gpu-device", str(args.gpu_device),
               "--representations", args.representations, "--parity-first", "3"]
    environment = dict(os.environ)
    environment.setdefault("VK_DRIVER_FILES", "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json")
    started = time.perf_counter()
    subprocess.run(command, check=True, env=environment)
    wall_seconds = time.perf_counter() - started
    summary = json.loads((args.feature_dir / "extractor_summary.json").read_text(encoding="utf-8"))
    if summary.get("actual_backend") != "Vulkan" or summary.get("device_name") != "Vulkan0":
        raise RuntimeError("extractor benchmark did not execute on explicitly selected Vulkan0")
    if summary.get("parity_checked_jobs") != 3 or summary.get("parity_mismatches") != 0:
        raise RuntimeError("extractor benchmark parity checks did not pass")
    result: dict[str, Any] = {
        "schema": "nemo-speech.voiceid-task09c-feature-extraction-benchmark.v1",
        "backend": summary["actual_backend"], "device": summary["device_name"],
        "model_fingerprint": summary["model_fingerprint"],
        "feature_representation": args.representations,
        "job_count": summary["job_count"], "total_audio_sec": total_audio_sec,
        "wall_seconds": wall_seconds,
        "audio_seconds_per_wall_second": total_audio_sec / wall_seconds if wall_seconds else None,
        "wall_milliseconds_per_job": wall_seconds * 1000 / summary["job_count"] if summary["job_count"] else None,
        "parity_checked_jobs": summary["parity_checked_jobs"], "parity_mismatches": summary["parity_mismatches"],
        "cpu_model": cpu_model(),
        "scope": "native feature extraction microbenchmark over cached PCM16 WAVs from a fixed 12-job Task 09B subset; excludes source FLAC conversion and production integration",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"benchmark_task09c_extractor: {exc}", file=sys.stderr)
        raise
