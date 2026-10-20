#!/usr/bin/env python3
"""Inventory, lazily prepare, partition, and score VCTK VoiceID experiments.

No audio is copied into the repository. Converted files, VAD masks, generated
conversations, and raw probability traces live under --cache-dir / --trace-dir.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "nemo-speech.voiceid-task08.v1"
TASK09A_SCHEMA = "nemo-speech.voiceid-task09a-enrollment.v1"
CONVERSION = "ffmpeg -nostdin -v error -y -i <source> -map_metadata -1 -ac 1 -ar 16000 -c:a pcm_s16le <wav>"
VAD = {
    "method": "20ms RMS windows at 10ms hop; speech threshold is max(-45 dBFS, peak-window RMS - 28 dB); bridge gaps <= 60ms; drop runs < 30ms",
    "window_samples": 320,
    "hop_samples": 160,
    "floor_dbfs": -45.0,
    "relative_db": 28.0,
    "bridge_frames": 6,
    "minimum_run_frames": 3,
    "frame_period_sec": 0.01,
}
BUCKETS = ("under_2s", "2_to_5s", "5_to_10s", "over_10s")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audio_root_for(dataset_root: Path) -> Path:
    for candidate in (dataset_root / "wav48_silence_trimmed", dataset_root):
        if candidate.is_dir() and any(candidate.glob("p*/p*_mic*.flac")):
            return candidate.resolve()
    raise RuntimeError(f"cannot find VCTK FLACs below {dataset_root}")


def flac_streaminfo(path: Path) -> tuple[int, int, int, float]:
    with path.open("rb") as audio:
        prefix = audio.read(42)
    if len(prefix) < 42 or prefix[:4] != b"fLaC" or (prefix[4] & 0x7F) != 0:
        raise ValueError("missing FLAC STREAMINFO block")
    metadata_size = int.from_bytes(prefix[5:8], "big")
    if metadata_size != 34:
        raise ValueError(f"unexpected FLAC STREAMINFO size {metadata_size}")
    packed = int.from_bytes(prefix[18:26], "big")
    sample_rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    bits_per_sample = ((packed >> 36) & 0x1F) + 1
    total_samples = packed & ((1 << 36) - 1)
    if sample_rate <= 0 or total_samples <= 0:
        raise ValueError("invalid sample rate or total sample count")
    return sample_rate, channels, bits_per_sample, total_samples / sample_rate


def speaker_and_utterance(path: Path) -> tuple[str, str, str]:
    speaker = path.parent.name
    match = re.fullmatch(r"(.+)_mic([0-9]+)\.flac", path.name)
    if match:
        return speaker, match.group(1), f"mic{match.group(2)}"
    return speaker, path.stem, "unknown"


def scan_dataset(dataset_root: Path) -> tuple[Path, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    root = audio_root_for(dataset_root)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rates: Counter[int] = Counter()
    channel_counts: Counter[int] = Counter()
    bit_depth_counts: Counter[int] = Counter()
    bad: list[dict[str, str]] = []
    present_files: set[str] = set()
    total_duration = 0.0
    durations = []
    total_files = 0
    identities: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source in sorted(root.glob("*/*.flac")):
        total_files += 1
        present_files.add(source.relative_to(root).as_posix())
        speaker, utterance, mic = speaker_and_utterance(source)
        try:
            rate, channels, bits, duration = flac_streaminfo(source)
        except (OSError, ValueError) as exc:
            bad.append({"path": source.relative_to(root).as_posix(), "error": str(exc)})
            continue
        record = {
            "speaker_id": speaker,
            "utterance_id": utterance,
            "mic": mic,
            "source_path": source.relative_to(root).as_posix(),
            "sample_rate_hz": rate,
            "channels": channels,
            "bits_per_sample": bits,
            "duration_sec": round(duration, 6),
            "size_bytes": source.stat().st_size,
        }
        by_speaker[speaker].append(record)
        identities[(speaker, utterance)].add(mic)
        rates[rate] += 1
        channel_counts[channels] += 1
        bit_depth_counts[bits] += 1
        total_duration += duration
        durations.append(duration)
    if total_files == 0:
        raise RuntimeError(f"no FLAC files found below {root}")

    speaker_summary: dict[str, Any] = {}
    canonical_speakers = []
    for speaker in sorted(by_speaker):
        items = by_speaker[speaker]
        by_mic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_mic[item["mic"]].append(item)
        bases = {item["utterance_id"] for item in items}
        mic1 = {item["utterance_id"] for item in by_mic.get("mic1", [])}
        mic2 = {item["utterance_id"] for item in by_mic.get("mic2", [])}
        if mic1:
            canonical_speakers.append(speaker)
        speaker_summary[speaker] = {
            "file_count": len(items),
            "utterance_identity_count": len(bases),
            "mic_file_counts": {mic: len(values) for mic, values in sorted(by_mic.items())},
            "mic1_utterance_count": len(mic1),
            "mic2_utterance_count": len(mic2),
            "mic1_mic2_paired_utterances": len(mic1 & mic2),
            "mic1_only_utterances": len(mic1 - mic2),
            "mic2_only_utterances": len(mic2 - mic1),
            "duration_sec_by_mic": {
                mic: round(sum(item["duration_sec"] for item in values), 3)
                for mic, values in sorted(by_mic.items())
            },
            "sample_rates_hz": sorted({item["sample_rate_hz"] for item in items}),
        }
    pair_counts = Counter(len(mics) for mics in identities.values())
    audio_speaker_dirs = sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and any(path.glob("*.flac"))
    )
    transcript_root = dataset_root / "txt"
    expected_mic1 = set()
    transcript_ids = set()
    if transcript_root.is_dir():
        for transcript in transcript_root.glob("*/*.txt"):
            transcript_id = f"{transcript.parent.name}/{transcript.stem}"
            transcript_ids.add(transcript_id)
            expected_mic1.add(f"{transcript_id}_mic1.flac")
    missing_mic1 = sorted(expected_mic1 - present_files)
    audio_transcript_ids = {
        f"{Path(relative).parent.name}/{Path(relative).name.rsplit('_mic', 1)[0]}"
        for relative in present_files
    }
    orphan_audio = sorted(audio_transcript_ids - transcript_ids) if transcript_root.is_dir() else []
    transcript_speaker_dirs = sorted(
        path.name for path in transcript_root.iterdir()
        if path.is_dir() and any(path.glob("*.txt"))
    ) if transcript_root.is_dir() else []
    ordered_durations = sorted(durations)

    def percentile(fraction: float) -> float | None:
        if not ordered_durations:
            return None
        index = round((len(ordered_durations) - 1) * fraction)
        return round(ordered_durations[index], 6)

    summary = {
        "schema": "nemo-speech.vctk-inventory.v1",
        "dataset_root": dataset_root.name,
        "audio_root_relative": root.relative_to(dataset_root.resolve()).as_posix()
        if root.is_relative_to(dataset_root.resolve()) else str(root),
        "audio_speaker_directory_count": len(audio_speaker_dirs),
        "audio_speaker_directory_ids": audio_speaker_dirs,
        "audio_directory_tree_count_including_root": len(audio_speaker_dirs) + 1,
        "transcript_speaker_directory_count": len(transcript_speaker_dirs),
        "transcript_speaker_directory_ids": transcript_speaker_dirs,
        "file_count": total_files,
        "header_readable_file_count": sum(len(items) for items in by_speaker.values()),
        "unreadable_header_file_count": len(bad),
        "unreadable_headers": bad,
        "missing_mic1_audio_for_transcript_count": len(missing_mic1),
        "missing_mic1_audio_for_transcript": missing_mic1,
        "audio_without_matching_transcript_count": len(orphan_audio),
        "audio_without_matching_transcript": orphan_audio,
        "speaker_ids": sorted(by_speaker),
        "speaker_count": len(by_speaker),
        "mic1_usable_speaker_ids": canonical_speakers,
        "mic1_usable_speaker_count": len(canonical_speakers),
        "sample_rate_file_counts": {str(rate): count for rate, count in sorted(rates.items())},
        "channel_file_counts": {str(channels): count for channels, count in sorted(channel_counts.items())},
        "bits_per_sample_file_counts": {str(bits): count for bits, count in sorted(bit_depth_counts.items())},
        "total_header_duration_sec": round(total_duration, 3),
        "clip_duration_sec": {
            "min": round(min(durations), 6),
            "p25": percentile(0.25),
            "median": round(statistics.median(durations), 6),
            "p75": percentile(0.75),
            "p90": percentile(0.90),
            "max": round(max(durations), 6),
        },
        "utterance_identity_count": len(identities),
        "microphone_variant_counts": {
            "single_variant": pair_counts.get(1, 0),
            "paired_mic1_mic2": sum(1 for mics in identities.values() if "mic1" in mics and "mic2" in mics),
            "other_variant_count": sum(1 for mics in identities.values() if len(mics) > 1 and not ({"mic1", "mic2"} <= mics)),
        },
        "speakers": speaker_summary,
        "limitations": [
            "Duration and readability here are FLAC STREAMINFO header checks; the preparation path decodes each selected file with FFmpeg before use.",
            "VCTK speaker directory and microphone metadata do not establish independent recording sessions.",
        ],
    }
    return root, by_speaker, summary


def vad_mask_from_pcm16(pcm: bytes) -> bytes:
    samples = [sample[0] for sample in struct.iter_unpack("<h", pcm)]
    if not samples:
        return b""
    window = int(VAD["window_samples"])
    hop = int(VAD["hop_samples"])
    rms = []
    for start in range(0, len(samples), hop):
        frame = samples[start : min(start + window, len(samples))]
        rms.append(math.sqrt(sum(float(value) * value for value in frame) / max(1, len(frame))))
    peak = max(rms, default=0.0)
    floor = 32768.0 * 10.0 ** (float(VAD["floor_dbfs"]) / 20.0)
    threshold = max(floor, peak * 10.0 ** (-float(VAD["relative_db"]) / 20.0))
    mask = [1 if value >= threshold else 0 for value in rms]
    bridge = int(VAD["bridge_frames"])
    pos = 0
    while pos < len(mask):
        if mask[pos]:
            pos += 1
            continue
        end = pos
        while end < len(mask) and not mask[end]:
            end += 1
        if pos > 0 and end < len(mask) and end - pos <= bridge:
            mask[pos:end] = [1] * (end - pos)
        pos = end
    minimum = int(VAD["minimum_run_frames"])
    pos = 0
    while pos < len(mask):
        if not mask[pos]:
            pos += 1
            continue
        end = pos
        while end < len(mask) and mask[end]:
            end += 1
        if end - pos < minimum:
            mask[pos:end] = [0] * (end - pos)
        pos = end
    return bytes(mask)


class Preparer:
    def __init__(self, audio_root: Path, cache_dir: Path, ffmpeg: str = "ffmpeg") -> None:
        self.audio_root = audio_root.resolve()
        self.cache_dir = cache_dir.resolve()
        self.ffmpeg = ffmpeg
        version = subprocess.run(
            [self.ffmpeg, "-version"], check=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        ).stdout.splitlines()
        self.ffmpeg_version = version[0] if version else "unknown"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.assets: dict[str, dict[str, Any]] = {}

    def prepare(self, source_relative: str, source_meta: dict[str, Any]) -> dict[str, Any]:
        source = (self.audio_root / source_relative).resolve()
        if not source.is_relative_to(self.audio_root) or not source.is_file():
            raise RuntimeError(f"source is absent or outside the VCTK audio root: {source_relative}")
        source_hash = sha256_file(source)
        key_data = f"{source_hash}|{CONVERSION}|{self.ffmpeg_version}|vad={json.dumps(VAD,sort_keys=True)}".encode()
        key = hashlib.sha256(key_data).hexdigest()
        wav_path = self.cache_dir / f"{key}.wav"
        mask_path = self.cache_dir / f"{key}.vad10.bin"
        if key not in self.assets:
            if not wav_path.exists() or not mask_path.exists():
                fd, temp_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".wav", dir=self.cache_dir)
                os.close(fd)
                temp_path = Path(temp_name)
                try:
                    command = [
                        self.ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source),
                        "-map_metadata", "-1", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                        str(temp_path),
                    ]
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    with wave.open(str(temp_path), "rb") as converted:
                        if converted.getnchannels() != 1 or converted.getframerate() != 16000 or converted.getsampwidth() != 2:
                            raise RuntimeError(f"FFmpeg produced unexpected PCM format for {source_relative}")
                        n_samples = converted.getnframes()
                        pcm = converted.readframes(n_samples)
                    if len(pcm) != n_samples * 2:
                        raise RuntimeError(f"truncated converted audio for {source_relative}")
                    mask = vad_mask_from_pcm16(pcm)
                    os.replace(temp_path, wav_path)
                    temporary_mask = mask_path.with_suffix(".tmp")
                    temporary_mask.write_bytes(mask)
                    os.replace(temporary_mask, mask_path)
                finally:
                    temp_path.unlink(missing_ok=True)
            with wave.open(str(wav_path), "rb") as converted:
                n_samples = converted.getnframes()
                duration = n_samples / 16000.0
            mask = mask_path.read_bytes()
            self.assets[key] = {
                "cache_key": key,
                "audio_file": wav_path.name,
                "vad_mask_file": mask_path.name,
                "source_path": source_relative,
                "source_sha256": source_hash,
                "ffmpeg_version": self.ffmpeg_version,
                "conversion_parameters": CONVERSION,
                "converted_sha256": sha256_file(wav_path),
                "converted_sample_rate_hz": 16000,
                "converted_channels": 1,
                "converted_sample_format": "pcm_s16le",
                "duration_sec": round(duration, 6),
                "speech_duration_sec": round(sum(mask) * 0.01, 4),
                "vad_frames": len(mask),
                "vad_sha256": hashlib.sha256(mask).hexdigest(),
            }
        result = dict(source_meta)
        result.update(self.assets[key])
        return result


def duration_bucket(speech_sec: float) -> str:
    if speech_sec < 2.0:
        return "under_2s"
    if speech_sec < 5.0:
        return "2_to_5s"
    if speech_sec < 10.0:
        return "5_to_10s"
    return "over_10s"


def partition_records(
    by_speaker: dict[str, list[dict[str, Any]]], seed: int, population_parts: tuple[float, float, float] = (0.15, 0.30, 0.55)
) -> tuple[dict[str, str], dict[str, dict[str, list[str]]], str]:
    speakers = sorted(speaker for speaker, items in by_speaker.items() if any(item["mic"] == "mic1" for item in items))
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_dev = max(1, round(len(speakers) * population_parts[0]))
    n_cal = max(1, round(len(speakers) * population_parts[1]))
    if n_dev + n_cal >= len(speakers):
        n_cal = max(1, len(speakers) - n_dev - 1)
    populations = {
        "development": speakers[:n_dev],
        "calibration": speakers[n_dev : n_dev + n_cal],
        "final_holdout": speakers[n_dev + n_cal :],
    }
    population_for = {speaker: name for name, ids in populations.items() for speaker in ids}
    per_speaker: dict[str, dict[str, list[str]]] = {}
    full_material = []
    for speaker in sorted(population_for):
        utterances = sorted({item["utterance_id"] for item in by_speaker[speaker] if item["mic"] == "mic1"})
        local = random.Random(f"{seed}:{speaker}:utterance-partition")
        local.shuffle(utterances)
        n = len(utterances)
        n_dev_u = max(1, round(n * 0.10))
        n_cal_u = max(1, round(n * 0.15))
        n_final_u = max(1, round(n * 0.15))
        if n_dev_u + n_cal_u + n_final_u >= n:
            n_dev_u, n_cal_u, n_final_u = 1, 1, 1
        n_enroll = n - n_dev_u - n_cal_u - n_final_u
        assigned = {
            "enrollment": utterances[:n_enroll],
            "development": utterances[n_enroll : n_enroll + n_dev_u],
            "calibration": utterances[n_enroll + n_dev_u : n_enroll + n_dev_u + n_cal_u],
            "final_holdout": utterances[n_enroll + n_dev_u + n_cal_u :],
        }
        per_speaker[speaker] = assigned
        for part, ids in assigned.items():
            full_material.append((speaker, part, *sorted(ids)))
    digest = hashlib.sha256(json.dumps(full_material, separators=(",", ":")).encode()).hexdigest()
    return population_for, per_speaker, digest


def files_for_utterance(
    by_speaker: dict[str, list[dict[str, Any]]], speaker: str, utterance: str, mic: str
) -> dict[str, Any] | None:
    for item in by_speaker.get(speaker, []):
        if item["utterance_id"] == utterance and item["mic"] == mic:
            return item
    return None


def choose_assets(
    prep: Preparer,
    by_speaker: dict[str, list[dict[str, Any]]],
    speaker: str,
    utterance_ids: Iterable[str],
    mic: str,
    seed: int,
    wanted_bucket: str | None = None,
    speech_budget: float | None = None,
) -> list[dict[str, Any]]:
    candidates = []
    for utterance in utterance_ids:
        source = files_for_utterance(by_speaker, speaker, utterance, mic)
        if source:
            candidates.append(source)
    if wanted_bucket:
        # Header duration is an inexpensive candidate index. The final bucket
        # is always assigned from decoded 10 ms VAD speech duration below.
        bounds = {
            "under_2s": (0.0, 2.2),
            "2_to_5s": (1.8, 5.3),
            "5_to_10s": (4.7, 10.5),
            "over_10s": (9.5, float("inf")),
        }
        lower, upper = bounds[wanted_bucket]
        candidates = [
            source for source in candidates
            if lower <= source["duration_sec"] <= upper
        ]
    local = random.Random(f"{seed}:{speaker}:{mic}:{wanted_bucket}:{speech_budget}")
    local.shuffle(candidates)
    selected = []
    accumulated = 0.0
    for source in candidates:
        asset = prep.prepare(source["source_path"], {
            "speaker_id": speaker,
            "utterance_id": source["utterance_id"],
            "mic": source["mic"],
            "source_sample_rate_hz": source["sample_rate_hz"],
            "source_channels": source["channels"],
            "source_duration_sec": source["duration_sec"],
        })
        if wanted_bucket and duration_bucket(asset["speech_duration_sec"]) != wanted_bucket:
            continue
        selected.append(asset)
        accumulated += asset["speech_duration_sec"]
        if speech_budget is None or accumulated >= speech_budget:
            break
    return selected


def trial_asset(asset: dict[str, Any], known: bool, partition: str, **extra: Any) -> dict[str, Any]:
    result = {key: asset[key] for key in (
        "speaker_id", "utterance_id", "mic", "source_path", "source_sample_rate_hz",
        "source_duration_sec", "audio_file", "vad_mask_file", "source_sha256",
        "converted_sha256", "vad_sha256", "duration_sec", "speech_duration_sec", "vad_frames",
    )}
    result.update({"known": known, "source_partition": partition, "duration_bucket": duration_bucket(asset["speech_duration_sec"])})
    result.update(extra)
    return result


def mask_intervals(mask: bytes, offset_frames: int = 0) -> list[list[int]]:
    intervals = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            intervals.append([offset_frames + start, offset_frames + index])
            start = None
    if start is not None:
        intervals.append([offset_frames + start, offset_frames + len(mask)])
    return intervals


class ConversationBuilder:
    """Deterministic PCM mixer with source-level VAD reference tracks."""

    def __init__(self, prep: Preparer, seed: int) -> None:
        self.prep = prep
        self.seed = seed
        self.loaded: dict[str, tuple[array.array, bytes]] = {}

    def load(self, item: dict[str, Any]) -> tuple[array.array, bytes]:
        wav_name = item["audio_file"]
        if wav_name not in self.loaded:
            wav_path = self.prep.cache_dir / wav_name
            with wave.open(str(wav_path), "rb") as source:
                if source.getnchannels() != 1 or source.getframerate() != 16000 or source.getsampwidth() != 2:
                    raise RuntimeError(f"conversation source has unexpected PCM format: {wav_name}")
                samples = array.array("h")
                samples.frombytes(source.readframes(source.getnframes()))
            if os.sys.byteorder != "little":
                samples.byteswap()
            mask = (self.prep.cache_dir / item["vad_mask_file"]).read_bytes()
            if len(mask) != (len(samples) + 159) // 160:
                raise RuntimeError(f"conversation source VAD mask does not match audio: {wav_name}")
            self.loaded[wav_name] = (samples, mask)
        return self.loaded[wav_name]

    def window(
        self, item: dict[str, Any], target_frames: int | None, seed: int,
    ) -> tuple[array.array, bytes, int, int]:
        samples, mask = self.load(item)
        frame_count = len(mask)
        if target_frames is None or target_frames >= frame_count:
            start_frame = 0
            end_frame = frame_count
        else:
            target_frames = max(1, target_frames)
            prefix = [0]
            for active in mask:
                prefix.append(prefix[-1] + (1 if active else 0))
            counts = [prefix[start + target_frames] - prefix[start]
                      for start in range(frame_count - target_frames + 1)]
            best = max(counts, default=0)
            near_best = [index for index, count in enumerate(counts) if count >= max(1, best - 2)]
            if not near_best:
                near_best = [0]
            start_frame = random.Random(seed).choice(near_best)
            end_frame = min(frame_count, start_frame + target_frames)
        start_sample = start_frame * 160
        end_sample = min(len(samples), end_frame * 160)
        crop = array.array("h", samples[start_sample:end_sample])
        crop_mask = bytearray(mask[start_frame:end_frame])
        # A 10 ms cosine taper avoids hard cut artifacts. Exclude tapered edge
        # frames from reference speech masks so labels follow surviving source speech.
        fade_samples = min(160, len(crop) // 2)
        for index in range(fade_samples):
            factor = 0.5 - 0.5 * math.cos(math.pi * (index + 1) / (fade_samples + 1))
            crop[index] = int(round(crop[index] * factor))
            crop[-1 - index] = int(round(crop[-1 - index] * factor))
        if crop_mask:
            crop_mask[0] = 0
            crop_mask[-1] = 0
        return crop, bytes(crop_mask), start_frame, end_frame

    def build(
        self, trial: dict[str, Any], conversation_id: str, scenario: str,
        ordered_sources: list[dict[str, Any]], placements: list[dict[str, Any]],
        seed: int, overlap_target: float | None = None,
    ) -> dict[str, Any] | None:
        if not ordered_sources or len(ordered_sources) != len(placements):
            return None
        extent = max(
            placement["start_frame"] + len(placement["mask"])
            for placement in placements
        )
        sample_count = extent * 160
        mix = [0.0] * sample_count
        track_masks: dict[str, bytearray] = {}
        source_metadata = []
        for source, placement in zip(ordered_sources, placements):
            audio: array.array = placement["audio"]
            mask: bytes = placement["mask"]
            start_frame = placement["start_frame"]
            speaker = source["speaker_id"]
            track = track_masks.setdefault(speaker, bytearray(extent))
            for index, active in enumerate(mask):
                if active:
                    track[start_frame + index] = 1
            start_sample = start_frame * 160
            gain = float(placement["gain"])
            for index, sample in enumerate(audio):
                mix[start_sample + index] += (sample / 32768.0) * gain
            source_metadata.append({
                "speaker_id": speaker,
                "known": bool(source["known"]),
                "query_id": source["query_id"],
                "utterance_id": source["utterance_id"],
                "source_path": source["source_path"],
                "source_sha256": source["source_sha256"],
                "converted_sha256": source["converted_sha256"],
                "microphone": source["mic"],
                "source_partition": source["source_partition"],
                "source_window_start_sec": round(placement["source_start_frame"] * 0.01, 4),
                "source_window_end_sec": round(placement["source_end_frame"] * 0.01, 4),
                "start_sec": round(start_frame * 0.01, 4),
                "end_sec": round((start_frame + len(mask)) * 0.01, 4),
                "gain_before_peak_scale": gain,
                "source_vad_speech_sec": round(sum(mask) * 0.01, 4),
                "speech_intervals_frames": mask_intervals(mask, start_frame),
                "role": placement.get("role", "turn"),
            })
        max_abs = max((abs(value) for value in mix), default=0.0)
        peak_scale = min(1.0, 0.98 / max_abs) if max_abs > 0.98 else 1.0
        encoded = array.array("h", (max(-32768, min(32767, round(value * peak_scale * 32768))) for value in mix))
        if os.sys.byteorder != "little":
            encoded.byteswap()
        pcm = encoded.tobytes()
        union_mask = bytearray(extent)
        reference_tracks = []
        for speaker, track in sorted(track_masks.items()):
            union_mask = bytearray(1 if union_mask[i] or track[i] else 0 for i in range(extent))
            source = next(entry for entry in ordered_sources if entry["speaker_id"] == speaker)
            reference_tracks.append({
                "speaker_id": speaker,
                "known": bool(source["known"]),
                "speech_intervals_frames": mask_intervals(bytes(track)),
                "speech_duration_sec": round(sum(track) * 0.01, 4),
            })
        actual_overlap = None
        if len(track_masks) >= 2:
            active_tracks = list(track_masks.values())
            intersections = sum(1 for i in range(extent) if sum(track[i] for track in active_tracks) > 1)
            min_active = min(sum(track) for track in active_tracks)
            actual_overlap = intersections / min_active if min_active else 0.0
        raw_sources_key = [
            (source["speaker_id"], source["utterance_id"], source["converted_sha256"], placement["start_frame"],
             placement["source_start_frame"], placement["source_end_frame"], placement["gain"])
            for source, placement in zip(ordered_sources, placements)
        ]
        key_material = json.dumps(
            [self.seed, trial["trial_id"], conversation_id, scenario, seed, raw_sources_key, peak_scale],
            sort_keys=True, separators=(",", ":"),
        ).encode()
        key = hashlib.sha256(key_material).hexdigest()
        wav_path = self.prep.cache_dir / f"conversation-{key}.wav"
        mask_path = self.prep.cache_dir / f"conversation-{key}.vad10.bin"
        if not wav_path.exists() or not mask_path.exists():
            with wave.open(str(wav_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(pcm)
            temporary_mask = mask_path.with_suffix(".tmp")
            temporary_mask.write_bytes(union_mask)
            os.replace(temporary_mask, mask_path)
        audio_sha = sha256_file(wav_path)
        mask_bytes = mask_path.read_bytes()
        metadata = {
            "cache_key": key,
            "audio_file": wav_path.name,
            "vad_mask_file": mask_path.name,
            "source_path": f"synthetic/{conversation_id}",
            "source_sha256": hashlib.sha256("".join(source["source_sha256"] for source in ordered_sources).encode()).hexdigest(),
            "converted_sha256": audio_sha,
            "converted_sample_rate_hz": 16000,
            "converted_channels": 1,
            "converted_sample_format": "pcm_s16le",
            "conversion_parameters": "deterministic additive waveform mixing from prepared VCTK PCM16 assets",
            "ffmpeg_version": "not used for generated conversations",
            "duration_sec": round(sample_count / 16000.0, 6),
            "speech_duration_sec": round(sum(union_mask) * 0.01, 4),
            "vad_frames": len(union_mask),
            "vad_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        }
        self.prep.assets[key] = metadata
        info: dict[str, Any] = {
            "conversation_id": conversation_id,
            "scenario": scenario,
            "seed": seed,
            "source_partition": trial["evaluation_partition"],
            **metadata,
            "sources": source_metadata,
            "reference_tracks": reference_tracks,
            "mix": {
                "type": "additive_float_waveform_then_pcm16",
                "peak_limit": 0.98,
                "global_peak_scale": peak_scale,
                "scheduled_overlap_target": overlap_target,
                "actual_source_vad_overlap_fraction": actual_overlap,
            },
        }
        return info

    def build_scenarios(self, trial: dict[str, Any]) -> list[dict[str, Any]]:
        by_speaker: dict[str, dict[str, Any]] = {}
        for item in trial["queries"]:
            if item["mic"] != "mic1":
                continue
            previous = by_speaker.get(item["speaker_id"])
            if previous is None or item["speech_duration_sec"] > previous["speech_duration_sec"]:
                by_speaker[item["speaker_id"]] = item
        known = [item for item in by_speaker.values() if item["known"]]
        unknown = [item for item in by_speaker.values() if not item["known"]]
        all_sources = known + unknown
        if len(all_sources) < 2:
            return []
        base_seed = int(trial["seed"])
        output: list[dict[str, Any]] = []

        order = list(all_sources)
        random.Random(f"{base_seed}:conversation:turn-order").shuffle(order)
        cursor_frame = 0
        placed = []
        for index, source in enumerate(order):
            audio, mask, source_start, source_end = self.window(source, None, base_seed + index)
            placed.append({
                "audio": audio, "mask": mask, "start_frame": cursor_frame,
                "source_start_frame": source_start, "source_end_frame": source_end,
                "gain": 0.75, "role": "turn",
            })
            cursor_frame += len(mask) + 25  # 250 ms configurable silence.
        result = self.build(trial, "turn_taking", "turn_taking", order, placed, base_seed)
        if result:
            result["pause_between_turns_sec"] = 0.25
            output.append(result)

        short_order = list(all_sources)
        random.Random(f"{base_seed}:conversation:short-order").shuffle(short_order)
        cursor_frame = 0
        placed = []
        for index, source in enumerate(short_order):
            rng = random.Random(f"{base_seed}:{source['speaker_id']}:short-turn")
            target = rng.randint(50, 200)
            audio, mask, source_start, source_end = self.window(source, target, base_seed + index + 31)
            placed.append({
                "audio": audio, "mask": mask, "start_frame": cursor_frame,
                "source_start_frame": source_start, "source_end_frame": source_end,
                "gain": 0.75, "role": "short_turn",
            })
            cursor_frame += len(mask) + 10  # 100 ms between short turns.
        result = self.build(trial, "short_turns", "short_turns", short_order, placed, base_seed + 1)
        if result:
            result["short_turn_duration_sec_range"] = [0.5, 2.0]
            result["pause_between_turns_sec"] = 0.1
            output.append(result)

        if known and unknown:
            pair_rng = random.Random(f"{base_seed}:conversation:overlap-pair")
            pair = [pair_rng.choice(known), pair_rng.choice(unknown)]
        else:
            pair = all_sources[:2]
        if len(pair) == 2:
            loaded = [self.load(item) for item in pair]
            target_frames = min(400, len(loaded[0][1]), len(loaded[1][1]))
            if target_frames >= 50:
                for overlap_index, overlap in enumerate((0.25, 0.50, 0.75)):
                    placements = []
                    for index, source in enumerate(pair):
                        audio, mask, source_start, source_end = self.window(
                            source, target_frames, base_seed + 100 + overlap_index * 7 + index,
                        )
                        placements.append({
                            "audio": audio, "mask": mask,
                            "start_frame": 0 if index == 0 else round(target_frames * (1.0 - overlap)),
                            "source_start_frame": source_start, "source_end_frame": source_end,
                            "gain": 0.65, "role": "overlap_source",
                        })
                    result = self.build(
                        trial, f"overlap_{int(overlap * 100):02d}", "overlapping_speech",
                        pair, placements, base_seed + 100 + overlap_index, overlap,
                    )
                    if result:
                        result["overlap_level"] = overlap
                        output.append(result)

        if unknown and known:
            traffic_order = list(unknown) + [known[0]]
            cursor_frame = 0
            placed = []
            for index, source in enumerate(traffic_order):
                audio, mask, source_start, source_end = self.window(source, None, base_seed + 300 + index)
                placed.append({
                    "audio": audio, "mask": mask, "start_frame": cursor_frame,
                    "source_start_frame": source_start, "source_end_frame": source_end,
                    "gain": 0.75, "role": "unknown_traffic" if not source["known"] else "registered_return",
                })
                cursor_frame += len(mask) + 25
            result = self.build(
                trial, "unknown_heavy", "unknown_heavy_then_known_return",
                traffic_order, placed, base_seed + 3,
            )
            if result:
                result["unknown_speakers_before_known_return"] = len(unknown)
                result["pause_between_turns_sec"] = 0.25
                output.append(result)
        return output


def make_trials(
    prep: Preparer,
    by_speaker: dict[str, list[dict[str, Any]]],
    population_for: dict[str, str],
    utterance_parts: dict[str, dict[str, list[str]]],
    preset: str,
    seed: int,
    cohorts_per_size: int,
    group_sizes: list[int],
    budgets: list[int],
    unknown_per_cohort: int,
    include_mic2: bool,
    include_conversations: bool,
) -> list[dict[str, Any]]:
    trials = []
    conversation_builder = ConversationBuilder(prep, seed) if include_conversations else None
    # The identity pools are speaker-disjoint. Development and calibration
    # identities cannot appear as final-holdout people or query sources.
    for population in ("development", "calibration", "final_holdout"):
        speaker_pool = sorted(s for s, p in population_for.items() if p == population)
        covered_as_known: set[str] = set()
        covered_as_unknown: set[str] = set()
        cycles_by_size = {}
        all_enrolled: set[str] = set()
        for group_size in group_sizes:
            if group_size >= 8 and len(speaker_pool) < group_size + unknown_per_cohort:
                continue
            cycle = list(speaker_pool)
            random.Random(f"{seed}:{population}:{group_size}:cohort-speaker-cycle").shuffle(cycle)
            cycles_by_size[group_size] = cycle
            for replicate in range(cohorts_per_size):
                offset = (replicate * group_size) % len(cycle)
                all_enrolled.update(cycle[(offset + index) % len(cycle)] for index in range(group_size))
        eval_partition = population
        for group_size in group_sizes:
            cycle = cycles_by_size.get(group_size)
            if cycle is None:
                continue
            # Cycle through speaker IDs in seeded orders so all source identities
            # can appear as known and as never-enrolled unknowns over cohorts.
            for replicate in range(cohorts_per_size):
                offset = (replicate * group_size) % len(cycle)
                enrolled = [cycle[(offset + index) % len(cycle)] for index in range(group_size)]
                remaining = [speaker for speaker in cycle if speaker not in enrolled]
                if not remaining:
                    continue
                unknown_offset = (replicate * unknown_per_cohort + group_size) % len(remaining)
                unknown_limit = min(unknown_per_cohort, len(remaining))
                covered_as_known.update(enrolled)
                unseen = [
                    speaker for speaker in remaining
                    if speaker not in all_enrolled and speaker not in covered_as_unknown
                ]
                unknowns = unseen[:unknown_limit]
                fallback = [remaining[(unknown_offset + index) % len(remaining)]
                            for index in range(len(remaining))]
                unknowns.extend(
                    speaker for speaker in fallback
                    if speaker not in unknowns and len(unknowns) < unknown_limit
                )
                covered_as_unknown.update(unknowns)
                population_seed = {"development": 0, "calibration": 250_000, "final_holdout": 500_000}[population]
                cohort_seed = seed + replicate * 1009 + group_size * 37 + population_seed
                order = list(enrolled)
                random.Random(f"{cohort_seed}:registration-order").shuffle(order)
                enrollment_by_speaker = {}
                for speaker in enrolled:
                    parts = utterance_parts[speaker]
                    clips = choose_assets(
                        prep, by_speaker, speaker, parts["enrollment"], "mic1", cohort_seed,
                        speech_budget=float(max(budgets)),
                    )
                    enrollment_by_speaker[speaker] = clips
                query_by_speaker: dict[tuple[str, str, str], dict[str, Any]] = {}
                for speaker in enrolled + unknowns:
                    mic1_query_ids: set[str] = set()
                    for bucket in BUCKETS:
                        clips = choose_assets(
                            prep, by_speaker, speaker, utterance_parts[speaker][eval_partition], "mic1",
                            cohort_seed + sum(ord(c) for c in bucket), wanted_bucket=bucket,
                        )
                        if clips:
                            asset = clips[0]
                            mic1_query_ids.add(asset["utterance_id"])
                            query_by_speaker[(speaker, bucket, "mic1")] = trial_asset(
                                asset, speaker in enrolled, eval_partition,
                            )
                    if include_mic2:
                        mic2_eval_ids = [
                            utterance for utterance in utterance_parts[speaker][eval_partition]
                            if utterance not in mic1_query_ids
                        ]
                        for bucket in BUCKETS:
                            clips = choose_assets(
                                prep, by_speaker, speaker, mic2_eval_ids, "mic2",
                                cohort_seed + 100 + sum(ord(c) for c in bucket), wanted_bucket=bucket,
                            )
                            if clips:
                                query_by_speaker[(speaker, bucket, "mic2")] = trial_asset(
                                    clips[0], speaker in enrolled, eval_partition, robustness="mic2",
                                )
                for budget in budgets:
                    variants = ["randomized"]
                    if budget in (30, 60):
                        variants.append("reverse")
                    for variant in variants:
                        selected_order = list(order if variant == "randomized" else reversed(order))
                        enrollment = []
                        enrollment_speech = 0.0
                        enrollment_duration = 0.0
                        clip_index = 0
                        for speaker in selected_order:
                            accumulated = 0.0
                            for asset in enrollment_by_speaker[speaker]:
                                entry = trial_asset(asset, True, "enrollment", enrollment_order=selected_order.index(speaker), clip_index=clip_index)
                                enrollment.append(entry)
                                clip_index += 1
                                accumulated += asset["speech_duration_sec"]
                                enrollment_speech += asset["speech_duration_sec"]
                                enrollment_duration += asset["duration_sec"]
                                if accumulated >= budget:
                                    break
                        queries = []
                        for speaker in enrolled + unknowns:
                            for bucket in BUCKETS:
                                for mic in (("mic1", "mic2") if include_mic2 else ("mic1",)):
                                    query = query_by_speaker.get((speaker, bucket, mic))
                                    if query:
                                        query = dict(query)
                                        query["query_id"] = f"{speaker}_{query['utterance_id']}_{bucket}_{mic}"
                                        queries.append(query)
                        # VCTK is single-speaker read speech. Any paired mic
                        # variants have one shared utterance_id and are grouped.
                        profile_id = f"{population}_n{group_size}_r{replicate:02d}_e{budget:02d}_{variant}"
                        traffic_ids = []
                        traffic_speakers = set()
                        for query in queries:
                            speaker = query["speaker_id"]
                            if not query["known"] and query["mic"] == "mic1" and speaker not in traffic_speakers:
                                traffic_ids.append(query["query_id"])
                                traffic_speakers.add(speaker)
                        trial = {
                            "trial_id": profile_id,
                            "cohort_id": f"{population}_n{group_size}_r{replicate:02d}",
                            "population": population,
                            "evaluation_partition": eval_partition,
                            "seed": cohort_seed,
                            "enrollment_budget_speech_sec_per_person": budget,
                            "enrollment_order_variant": variant,
                            "enrollment_order": selected_order,
                            "enrolled_speaker_ids": selected_order,
                            "unknown_speaker_ids": unknowns,
                            "group_size": group_size,
                            "enrollment": enrollment,
                            "enrollment_speech_sec_total": round(enrollment_speech, 4),
                            "enrollment_audio_sec_total": round(enrollment_duration, 4),
                            "queries": queries,
                            "run_unknown_traffic": budget in (30, 60),
                            "unknown_traffic_query_ids": traffic_ids,
                            "conversations": [],
                        }
                        if (conversation_builder is not None and budget == max(budgets)
                                and variant == "randomized"):
                            trial["conversations"] = conversation_builder.build_scenarios(trial)
                        trials.append(trial)
        uncovered = set(speaker_pool) - covered_as_known - covered_as_unknown
        if uncovered and preset == "full":
            raise RuntimeError(
                f"cohort plan left {population} speakers unused as both known and unknown: "
                f"{sorted(uncovered)}"
            )
    return trials


def _expand_task09a_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Adapt the compact Task 09A asset catalog to the shared leakage checks."""
    errors: list[str] = []
    catalog = manifest.get("task09a_asset_catalog", {})
    expanded_trials = []
    population_ids: dict[str, set[str]] = defaultdict(set)
    for trial in manifest.get("trials", []):
        trial_id = trial.get("trial_id", "unnamed")
        enrolled = set(trial.get("enrolled_speaker_ids", []))
        unknown = set(trial.get("unknown_speaker_ids", []))
        population_ids[trial.get("population", "")].update(enrolled | unknown)

        def resolve(reference: Any, role: str) -> dict[str, Any] | None:
            if not isinstance(reference, str):
                errors.append(f"{trial_id}: compact {role} reference is not an asset ID")
                return None
            asset = catalog.get(reference)
            if not isinstance(asset, dict):
                errors.append(f"{trial_id}: {role} references unknown asset {reference}")
                return None
            if Path(asset.get("audio_file", "")).stem != reference:
                errors.append(f"{trial_id}: asset ID does not match its audio path: {reference}")
            return dict(asset)

        construction = [resolve(reference, "enrollment") for reference in trial.get("enrollment", [])]
        validation = [resolve(reference, "validation enrollment")
                      for reference in trial.get("validation_enrollment", [])]
        queries = []
        for reference in trial.get("queries", []):
            if not isinstance(reference, dict):
                errors.append(f"{trial_id}: compact query reference is not an object")
                continue
            asset = catalog.get(reference.get("asset_id"))
            if not isinstance(asset, dict):
                errors.append(f"{trial_id}: query references unknown asset {reference.get('asset_id')}")
                continue
            if Path(asset.get("audio_file", "")).stem != reference.get("asset_id"):
                errors.append(f"{trial_id}: query asset ID does not match its audio path")
            queries.append({**asset, "known": bool(reference.get("known")),
                            "query_id": reference.get("query_id")})
        expanded_trials.append({
            **trial,
            "enrollment": [item for item in construction if item is not None],
            "validation_enrollment": [item for item in validation if item is not None],
            "queries": queries,
            "conversations": [],
        })
    expanded = {
        **manifest,
        "schema": SCHEMA,
        "speaker_populations": {key: sorted(value) for key, value in population_ids.items()},
        "trials": expanded_trials,
    }
    return expanded, errors


def validate_manifest(manifest: dict[str, Any], cache_dir: Path | None = None, verify_hashes: bool = False) -> list[str]:
    initial_errors: list[str] = []
    if manifest.get("schema") == TASK09A_SCHEMA:
        manifest, initial_errors = _expand_task09a_manifest(manifest)
    errors = initial_errors
    if manifest.get("schema") != SCHEMA:
        errors.append("unexpected Task 08 manifest schema")
    population_ids = manifest.get("speaker_populations", {})
    for left in population_ids:
        for right in population_ids:
            if left >= right:
                continue
            overlap = set(population_ids[left]) & set(population_ids[right])
            if overlap:
                errors.append(f"speaker populations overlap: {left}/{right}: {sorted(overlap)[:5]}")
    observed_role_coverage: dict[str, dict[str, set[str]]] = {
        population: {"known": set(), "unknown": set()}
        for population in population_ids
    }
    trial_ids = set()
    for trial in manifest.get("trials", []):
        trial_id = trial.get("trial_id", "")
        if trial_id in trial_ids:
            errors.append(f"duplicate trial_id {trial_id}")
        trial_ids.add(trial_id)
        enrolled = set(trial.get("enrolled_speaker_ids", []))
        observed_role_coverage.setdefault(
            trial.get("population", ""), {"known": set(), "unknown": set()}
        )["known"].update(enrolled)
        if len(enrolled) != trial.get("group_size"):
            errors.append(f"{trial_id}: enrolled speaker count disagrees with group_size")
        unknown_ids = set(trial.get("unknown_speaker_ids", []))
        observed_role_coverage[trial.get("population", "")]["unknown"].update(unknown_ids)
        if enrolled & unknown_ids:
            errors.append(f"{trial_id}: an unknown identity is enrolled")
        if not set(enrolled | unknown_ids) <= set(population_ids.get(trial.get("population"), [])):
            errors.append(f"{trial_id}: enrolled or unknown identity is outside its speaker population")
        enroll_pairs = set()
        enroll_by_speaker: dict[str, set[str]] = defaultdict(set)
        for item in trial.get("enrollment", []):
            if item.get("speaker_id") not in enrolled:
                errors.append(f"{trial_id}: enrollment item speaker is not registered")
            pair = (item.get("speaker_id"), item.get("utterance_id"))
            if pair in enroll_pairs:
                errors.append(f"{trial_id}: duplicate enrollment utterance identity {pair}")
            enroll_pairs.add(pair)
            enroll_by_speaker[pair[0]].add(pair[1])
            if item.get("mic") != "mic1":
                errors.append(f"{trial_id}: primary enrollment contains non-mic1 audio")
            if item.get("source_partition") != "enrollment":
                errors.append(f"{trial_id}: enrollment item came from a non-enrollment utterance partition")
        validation_pairs = set()
        validation_by_speaker: dict[str, set[str]] = defaultdict(set)
        for item in trial.get("validation_enrollment", []):
            if item.get("speaker_id") not in enrolled:
                errors.append(f"{trial_id}: validation item speaker is not registered")
            pair = (item.get("speaker_id"), item.get("utterance_id"))
            if pair in enroll_pairs or pair in validation_pairs:
                errors.append(f"{trial_id}: duplicate or construction-overlapping validation utterance identity {pair}")
            validation_pairs.add(pair)
            validation_by_speaker[pair[0]].add(pair[1])
            if item.get("mic") != "mic1":
                errors.append(f"{trial_id}: held-out enrollment validation contains non-mic1 audio")
            if item.get("source_partition") != "enrollment":
                errors.append(f"{trial_id}: validation item came from a non-enrollment utterance partition")
        for query in trial.get("queries", []):
            speaker = query.get("speaker_id")
            if query.get("known") and speaker not in enrolled:
                errors.append(f"{trial_id}/{query.get('query_id')}: known query identity is not enrolled")
            if not query.get("known") and speaker in enrolled:
                errors.append(f"{trial_id}/{query.get('query_id')}: unknown query identity is enrolled")
            if query.get("known") and (
                query.get("utterance_id") in enroll_by_speaker.get(speaker, set())
                or query.get("utterance_id") in validation_by_speaker.get(speaker, set())
            ):
                errors.append(f"{trial_id}/{query.get('query_id')}: enrollment or validation/query utterance leakage")
            if query.get("mic") not in ("mic1", "mic2"):
                errors.append(f"{trial_id}/{query.get('query_id')}: unknown microphone")
            if query.get("source_partition") != trial.get("evaluation_partition"):
                errors.append(f"{trial_id}/{query.get('query_id')}: query came from the wrong evaluation partition")
            if cache_dir:
                for field in ("audio_file", "vad_mask_file"):
                    rel = Path(query.get(field, ""))
                    if rel.is_absolute() or ".." in rel.parts:
                        errors.append(f"{trial_id}/{query.get('query_id')}: unsafe cache path")
                    elif not (cache_dir / rel).is_file():
                        errors.append(f"{trial_id}/{query.get('query_id')}: missing cache asset {rel}")
                if verify_hashes and (cache_dir / query["audio_file"]).is_file():
                    if sha256_file(cache_dir / query["audio_file"]) != query.get("converted_sha256"):
                        errors.append(f"{trial_id}/{query.get('query_id')}: converted WAV SHA-256 mismatch")
                if verify_hashes and (cache_dir / query["vad_mask_file"]).is_file():
                    if sha256_file(cache_dir / query["vad_mask_file"]) != query.get("vad_sha256"):
                        errors.append(f"{trial_id}/{query.get('query_id')}: VAD mask SHA-256 mismatch")
        if cache_dir:
            for item in [*trial.get("enrollment", []), *trial.get("validation_enrollment", [])]:
                for field in ("audio_file", "vad_mask_file"):
                    rel = Path(item.get(field, ""))
                    if rel.is_absolute() or ".." in rel.parts or not (cache_dir / rel).is_file():
                        errors.append(f"{trial_id}: missing/unsafe enrollment or validation asset {rel}")
                if verify_hashes and (cache_dir / item.get("audio_file", "")).is_file():
                    if sha256_file(cache_dir / item["audio_file"]) != item.get("converted_sha256"):
                        errors.append(f"{trial_id}: converted enrollment WAV SHA-256 mismatch")
                if verify_hashes and (cache_dir / item.get("vad_mask_file", "")).is_file():
                    if sha256_file(cache_dir / item["vad_mask_file"]) != item.get("vad_sha256"):
                        errors.append(f"{trial_id}: enrollment VAD mask SHA-256 mismatch")
        query_by_id = {item.get("query_id"): item for item in trial.get("queries", [])}
        for conversation in trial.get("conversations", []):
            conversation_id = conversation.get("conversation_id", "unnamed")
            conversation_sources = conversation.get("sources", [])
            source_speakers = set()
            for source in conversation_sources:
                speaker = source.get("speaker_id")
                source_speakers.add(speaker)
                if source.get("known") and speaker not in enrolled:
                    errors.append(f"{trial_id}/{conversation_id}: known source speaker is not enrolled")
                if not source.get("known") and speaker not in unknown_ids:
                    errors.append(f"{trial_id}/{conversation_id}: unknown source speaker is not reserved unknown")
                if source.get("source_partition") != trial.get("evaluation_partition"):
                    errors.append(f"{trial_id}/{conversation_id}: source came from wrong utterance partition")
                if source.get("microphone") != "mic1":
                    errors.append(f"{trial_id}/{conversation_id}: synthetic conversation source is not mic1")
                if (speaker, source.get("utterance_id")) in enroll_pairs:
                    errors.append(f"{trial_id}/{conversation_id}: enrollment/conversation utterance leakage")
                query = query_by_id.get(source.get("query_id"))
                if query is None or query.get("utterance_id") != source.get("utterance_id"):
                    errors.append(f"{trial_id}/{conversation_id}: source does not resolve to a manifest query")
            reference_speakers = {track.get("speaker_id") for track in conversation.get("reference_tracks", [])}
            if reference_speakers != source_speakers:
                errors.append(f"{trial_id}/{conversation_id}: reference tracks disagree with source speakers")
            for track in conversation.get("reference_tracks", []):
                for start, end in track.get("speech_intervals_frames", []):
                    if start < 0 or end <= start or end > conversation.get("vad_frames", 0):
                        errors.append(f"{trial_id}/{conversation_id}: invalid source speech interval")
            if cache_dir:
                for field in ("audio_file", "vad_mask_file"):
                    rel = Path(conversation.get(field, ""))
                    if rel.is_absolute() or ".." in rel.parts or not (cache_dir / rel).is_file():
                        errors.append(f"{trial_id}/{conversation_id}: missing/unsafe generated asset {rel}")
                if verify_hashes and (cache_dir / conversation.get("audio_file", "")).is_file():
                    if sha256_file(cache_dir / conversation["audio_file"]) != conversation.get("converted_sha256"):
                        errors.append(f"{trial_id}/{conversation_id}: generated WAV SHA-256 mismatch")
    declared_coverage = manifest.get("speaker_role_coverage")
    if declared_coverage is not None:
        for population, speakers in population_ids.items():
            observed = observed_role_coverage.get(
                population, {"known": set(), "unknown": set()}
            )
            missing = set(speakers) - observed["known"] - observed["unknown"]
            if missing and manifest.get("preset") == "full":
                errors.append(
                    f"{population}: speakers unused as known or unknown: {sorted(missing)}"
                )
            declared = declared_coverage.get(population, {})
            for role in ("known", "unknown"):
                if set(declared.get(f"represented_as_{role}", [])) != observed[role]:
                    errors.append(f"{population}: declared {role} speaker coverage does not match trials")
            if set(declared.get("unrepresented", [])) != missing:
                errors.append(f"{population}: declared unrepresented speaker list does not match trials")
    return errors


def read_trace(path: Path) -> tuple[int, int, float, list[float]]:
    data = path.read_bytes()
    if len(data) < 32 or data[:8] != b"NMSP8F1\0":
        raise ValueError(f"invalid trace magic/header: {path}")
    version, channels = struct.unpack_from("<II", data, 8)
    frames = struct.unpack_from("<Q", data, 16)[0]
    period = struct.unpack_from("<d", data, 24)[0]
    if version != 1 or channels <= 0 or period <= 0:
        raise ValueError(f"invalid trace metadata: {path}")
    offset = 32
    expected = offset + frames * channels * 4
    if len(data) != expected:
        raise ValueError(f"trace shape/byte-size mismatch: expected {expected}, found {len(data)}")
    values = list(struct.unpack_from(f"<{frames * channels}f", data, offset))
    return frames, channels, period, values


def command_inventory(args: argparse.Namespace) -> None:
    root, _, summary = scan_dataset(Path(args.dataset_root).resolve())
    write_json(Path(args.output), summary)
    print(json.dumps({k: summary[k] for k in (
        "file_count", "header_readable_file_count", "unreadable_header_file_count",
        "audio_speaker_directory_count", "speaker_count", "mic1_usable_speaker_count",
        "utterance_identity_count", "missing_mic1_audio_for_transcript_count",
        "audio_without_matching_transcript_count", "sample_rate_file_counts",
        "clip_duration_sec", "total_header_duration_sec", "microphone_variant_counts",
    )}, indent=2))


def command_prepare(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    root, by_speaker, inventory = scan_dataset(dataset_root)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    prep = Preparer(root, cache_dir, args.ffmpeg)
    pop_for, utterance_parts, partition_digest = partition_records(by_speaker, args.seed)
    selected_group_sizes = args.group_sizes or ("2,4" if args.preset == "pilot" else "2,4,8")
    selected_budgets = args.budgets or "5,10,15,30,45,60"
    group_sizes = [int(value) for value in selected_group_sizes.split(",") if value]
    budgets = [int(value) for value in selected_budgets.split(",") if value]
    cohorts_per_size = args.cohorts_per_size or (1 if args.preset == "pilot" else 4)
    unknown_per_cohort = args.unknown_per_cohort or (2 if args.preset == "pilot" else 4)
    include_mic2 = args.mic2_robustness if args.mic2_robustness is not None else args.preset == "full"
    include_conversations = args.conversations if args.conversations is not None else args.preset == "full"
    trials = make_trials(
        prep, by_speaker, pop_for, utterance_parts, args.preset, args.seed,
        cohorts_per_size, group_sizes, budgets, unknown_per_cohort, include_mic2,
        include_conversations,
    )
    populations = {name: sorted(s for s, value in pop_for.items() if value == name) for name in (
        "development", "calibration", "final_holdout",
    )}
    speaker_role_coverage = {}
    for population, speakers in populations.items():
        represented_as_known = sorted({
            speaker for trial in trials if trial["population"] == population
            for speaker in trial["enrolled_speaker_ids"]
        })
        represented_as_unknown = sorted({
            speaker for trial in trials if trial["population"] == population
            for speaker in trial["unknown_speaker_ids"]
        })
        speaker_role_coverage[population] = {
            "speaker_count": len(speakers),
            "represented_as_known": represented_as_known,
            "represented_as_unknown": represented_as_unknown,
            "unrepresented": sorted(
                set(speakers) - set(represented_as_known) - set(represented_as_unknown)
            ),
        }
    manifest = {
        "schema": SCHEMA,
        "preset": args.preset,
        "seed": args.seed,
        "dataset": {
            "name": "VCTK-Corpus-0.92",
            "audio_layout": "wav48_silence_trimmed",
            "audio_root_relative_to_dataset": root.relative_to(dataset_root).as_posix()
            if root.is_relative_to(dataset_root) else str(root),
            "file_count": inventory["file_count"],
            "speaker_count": inventory["speaker_count"],
            "mic1_usable_speaker_count": inventory["mic1_usable_speaker_count"],
        },
        "model": "models/Nemotron-3-Diarization.q8_0.gguf",
        "random_seed": args.seed,
        "conversion": CONVERSION,
        "conversion_cache": "external; pass the same --cache-dir to the C++ harness",
        "vad": VAD,
        "partition_rules": {
            "speaker_seed": args.seed,
            "speaker_cohorts_disjoint": True,
            "speaker_population_fractions": {"development": 0.15, "calibration": 0.30, "final_holdout": 0.55},
            "utterance_identity_fractions_per_speaker": {"enrollment": 0.60, "development": 0.10, "calibration": 0.15, "final_holdout": 0.15},
            "mic1_mic2_group_key": "speaker_id + filename with _micN suffix removed",
            "primary_microphone": "mic1",
            "mic2_experiment": "separate robustness queries; source utterance IDs remain grouped across microphones",
            "duration_candidate_index": "FLAC header duration narrows conversion candidates; final duration buckets use converted PCM VAD speech duration",
            "partition_digest_sha256": partition_digest,
        },
        "speaker_populations": populations,
        "speaker_role_coverage": speaker_role_coverage,
        "trial_count": len(trials),
        "trials": trials,
        "preparation_assets": prep.assets,
    }
    errors = validate_manifest(manifest, cache_dir)
    if errors:
        raise RuntimeError("generated manifest failed validation: " + "; ".join(errors[:20]))
    write_json(Path(args.output), manifest)
    print(json.dumps({
        "manifest": str(Path(args.output).resolve()),
        "preset": args.preset,
        "trials": len(trials),
        "profile_budgets_sec": budgets,
        "group_sizes": group_sizes,
        "speaker_populations": {key: len(value) for key, value in populations.items()},
        "converted_assets": len(prep.assets),
        "conversations": sum(len(trial["conversations"]) for trial in trials),
        "cache_dir": str(cache_dir),
        "validation_errors": 0,
    }, indent=2))


def command_validate(args: argparse.Namespace) -> None:
    manifest = load_json(Path(args.manifest))
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    errors = validate_manifest(manifest, cache_dir, args.verify_hashes)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"status": "pass", "trials": len(manifest.get("trials", [])), "errors": 0}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="inspect FLAC headers and microphone/utterance structure")
    inventory.add_argument("--dataset-root", default="VCTK-Corpus-0.92")
    inventory.add_argument("--output", default="tests/manifests/task08_vctk_inventory.json")
    inventory.set_defaults(func=command_inventory)
    prepare = sub.add_parser("prepare", help="lazily convert selected clips and build disjoint cohort manifests")
    prepare.add_argument("--dataset-root", default="VCTK-Corpus-0.92")
    prepare.add_argument("--cache-dir", default="~/.cache/nemo-speech-task08/vctk")
    prepare.add_argument("--output", default="tests/manifests/task08_vctk_pilot.json")
    prepare.add_argument("--preset", choices=("pilot", "full"), default="pilot")
    prepare.add_argument("--seed", type=int, default=20261020)
    prepare.add_argument("--group-sizes")
    prepare.add_argument("--budgets")
    prepare.add_argument("--cohorts-per-size", type=int)
    prepare.add_argument("--unknown-per-cohort", type=int)
    mic2 = prepare.add_mutually_exclusive_group()
    mic2.add_argument("--mic2-robustness", dest="mic2_robustness", action="store_true")
    mic2.add_argument("--no-mic2-robustness", dest="mic2_robustness", action="store_false")
    prepare.set_defaults(mic2_robustness=None)
    conversations = prepare.add_mutually_exclusive_group()
    conversations.add_argument("--conversations", dest="conversations", action="store_true")
    conversations.add_argument("--no-conversations", dest="conversations", action="store_false")
    prepare.set_defaults(conversations=None)
    prepare.add_argument("--ffmpeg", default="ffmpeg")
    prepare.set_defaults(func=command_prepare)
    validate = sub.add_parser("validate", help="fail on identity, utterance, microphone, or cache leakage")
    validate.add_argument("manifest")
    validate.add_argument("--cache-dir")
    validate.add_argument("--verify-hashes", action="store_true")
    validate.set_defaults(func=command_validate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
