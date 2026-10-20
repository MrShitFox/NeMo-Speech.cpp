#!/usr/bin/env python3
"""Create same-speaker prior-turn context jobs from the frozen Task 09B query pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

from extract_voiceid_features import load_jobs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def append_gap(prior_path: Path, output_path: Path, gap_ms: int = 250) -> str:
    with wave.open(str(prior_path), "rb") as source:
        channels, width, rate, count = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes())
        pcm = source.readframes(count)
    if (channels, width, rate) != (1, 2, 16000):
        raise ValueError(f"context source is not mono PCM16/16k: {prior_path}")
    samples = np.frombuffer(pcm, dtype="<i2")
    gap = np.zeros(round(rate * gap_ms / 1000), dtype="<i2")
    joined = np.concatenate((samples, gap))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(joined.tobytes())
    return sha256(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--base-jobs-tsv", type=Path,
                        default=Path("/home/glebus/.cache/nemo-speech-task09b/task09b_extraction_jobs.tsv"))
    parser.add_argument("--context-dir", type=Path, default=Path("/home/glebus/.cache/nemo-speech-task09c/audio/context"))
    parser.add_argument("--jobs-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/manifests/task09c_vctk_context.json"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = {row["job_id"]: row for row in manifest["assets"]}
    context_dir = args.context_dir.resolve()
    base_jobs = {row["job_id"]: row for row in load_jobs(args.base_jobs_tsv)}
    jobs: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in manifest["folds"]:
        candidates: dict[str, list[str]] = {}
        for item in fold["context_probe_jobs"]:
            candidates.setdefault(item["speaker_id"], [])
            if item["target_job_id"] not in candidates[item["speaker_id"]]:
                candidates[item["speaker_id"]].append(item["target_job_id"])
        fold_jobs = []
        for speaker, targets in sorted(candidates.items()):
            if len(targets) != 2:
                raise RuntimeError(f"expected two disjoint target utterances for {fold['fold_id']}/{speaker}; got {len(targets)}")
            for target_id, prior_id in ((targets[0], targets[1]), (targets[1], targets[0])):
                target, prior = assets[target_id], assets[prior_id]
                if target["speaker_id"] != prior["speaker_id"] or target["utterance_id"] == prior["utterance_id"]:
                    raise RuntimeError("same-speaker context must use a distinct query utterance")
                mode = "same_speaker_prior_turn"
                key = f"task09c-context:{fold['fold_id']}:{target_id}:{prior_id}:{mode}"
                context_job_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
                context_path = context_dir / f"{context_job_id}.wav"
                prior_wav = Path(base_jobs[prior_id]["wav_path"])
                context_hash = append_gap(prior_wav, context_path)
                target_wav = Path(base_jobs[target_id]["wav_path"])
                target_vad = Path(base_jobs[target_id]["vad_path"])
                job = {
                    "job_id": context_job_id, "speaker_id": speaker,
                    "utterance_id": target["utterance_id"], "mic": target["mic"],
                    "source_path": target["source_path"], "source_sha256": target["source_sha256"],
                    "converted_sha256": target["converted_sha256"], "vad_sha256": target["vad_sha256"],
                    "wav_path": str(target_wav), "vad_path": str(target_vad),
                    "context_wav_path": str(context_path), "context_mode": mode,
                    "context_speaker_id": speaker, "context_sha256": context_hash,
                    "aosc_profile_path": "", "target_job_id": target_id,
                    "prior_turn_job_id": prior_id,
                }
                jobs.append(job)
                fold_jobs.append({key: job[key] for key in ("job_id", "target_job_id", "prior_turn_job_id",
                    "speaker_id", "utterance_id", "context_sha256", "context_mode")})
        fold_rows.append({"fold_id": fold["fold_id"], "target_speakers": len(candidates), "jobs": fold_jobs})

    args.jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.jobs_tsv.open("w", encoding="utf-8") as output:
        output.write("# job_id\tspeaker_id\tutterance_id\tmic\tsource_path\tsource_sha256\tconverted_sha256\tvad_sha256\twav_path\tvad_path\tcontext_wav_path\tcontext_mode\tcontext_speaker_id\tcontext_sha256\taosc_profile_path\n")
        for job in jobs:
            output.write("\t".join(str(job[key]) for key in ("job_id", "speaker_id", "utterance_id", "mic",
                "source_path", "source_sha256", "converted_sha256", "vad_sha256", "wav_path", "vad_path",
                "context_wav_path", "context_mode", "context_speaker_id", "context_sha256", "aosc_profile_path")) + "\n")
    result = {
        "schema": "nemo-speech.voiceid-task09c-context-pairs.v1",
        "base_manifest": str(args.manifest.resolve()),
        "protocol": "each target is one of two utterance-disjoint query clips; the other clip from the same evaluation speaker is fed as a prior stream turn, followed by 250ms silence; these are controlled same-speaker context probes, not independent sessions",
        "jobs_tsv": str(args.jobs_tsv.resolve()), "context_dir": str(context_dir),
        "context_jobs": len(jobs), "folds": fold_rows,
    }
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "jobs_tsv": str(args.jobs_tsv),
                      "context_jobs": len(jobs), "folds": len(fold_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"prepare_task09c_context: {exc}", file=sys.stderr)
        raise
