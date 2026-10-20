#!/usr/bin/env python3
"""Build deterministic speaker/utterance-disjoint Task 09B VCTK folds.

Only a bounded mic1 sample is converted. The output manifest contains source
hashes and relative VCTK paths, while converted audio and native extractor TSV
are written outside Git in the supplied cache directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import wave
from pathlib import Path
from typing import Any

from prepare_vctk_voiceid import CONVERSION, VAD, Preparer, audio_root_for, flac_streaminfo, speaker_and_utterance


SEEDS = [20261021, 20261022, 20261023]
ROLE_COUNTS = {"train": 50, "selection": 15, "calibration": 15, "evaluation": 30}
TRAIN_UTTERANCES = 20
ENROLL_UTTERANCES = 32
QUERY_UTTERANCES = 12
REGISTRY_SIZES = [1, 2, 4, 8, 16]


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


def write_silence(path: Path, duration_sec: float = 2.56) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        frames = int(round(duration_sec * 16000))
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(bytes(frames * 2))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_mics(dataset_root: Path) -> tuple[Path, dict[str, dict[str, dict[str, Any]]]]:
    audio_root = audio_root_for(dataset_root)
    by_speaker: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(audio_root.glob("*/*_mic*.flac")):
        speaker, utterance, mic = speaker_and_utterance(path)
        rate, channels, bits, duration = flac_streaminfo(path)
        if rate != 48000 or channels != 1 or bits != 16:
            continue
        by_speaker.setdefault(speaker, {}).setdefault(utterance, {})[mic] = {
            "path": path,
            "source_path": path.relative_to(audio_root).as_posix(),
            "duration_sec": duration,
            "sample_rate_hz": rate,
            "channels": channels,
            "bits_per_sample": bits,
        }
    usable = {
        speaker: rows for speaker, rows in by_speaker.items()
        if sum("mic1" in mics for mics in rows.values()) >= TRAIN_UTTERANCES + ENROLL_UTTERANCES + QUERY_UTTERANCES
    }
    if len(usable) < sum(ROLE_COUNTS.values()):
        raise RuntimeError(f"need 110 usable speakers; found {len(usable)}")
    return audio_root, usable


def stable_utterance_sets(by_speaker: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, list[str]]]:
    assignments: dict[str, dict[str, list[str]]] = {}
    for speaker, utterances in sorted(by_speaker.items()):
        ids = [utt for utt, mics in utterances.items() if "mic1" in mics]
        random.Random(f"task09b-utterances-v1:{speaker}").shuffle(ids)
        required = TRAIN_UTTERANCES + ENROLL_UTTERANCES + QUERY_UTTERANCES
        if len(ids) < required:
            raise RuntimeError(f"{speaker} has only {len(ids)} mic1 utterances, need {required}")
        assignments[speaker] = {
            "head_training": ids[:TRAIN_UTTERANCES],
            "registration": ids[TRAIN_UTTERANCES:TRAIN_UTTERANCES + ENROLL_UTTERANCES],
            "independent_query": ids[TRAIN_UTTERANCES + ENROLL_UTTERANCES:required],
        }
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("VCTK-Corpus-0.92"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--jobs-tsv", type=Path)
    args = parser.parse_args()
    if args.folds < 1 or args.folds > len(args.seeds):
        parser.error("--folds must be in [1, number of --seeds]")

    dataset_root = args.dataset_root.resolve()
    audio_root, source_rows = scan_mics(dataset_root)
    utterance_sets = stable_utterance_sets(source_rows)
    speakers = sorted(utterance_sets)
    if len(speakers) != 110:
        raise RuntimeError(f"Task 09B requires the 110-speaker mic1 population; found {len(speakers)}")

    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    preparer = Preparer(audio_root, cache_dir)
    assets: dict[str, dict[str, Any]] = {}

    def prepare_asset(speaker: str, utterance: str, mic: str, partition: str) -> dict[str, Any]:
        source = source_rows[speaker][utterance].get(mic)
        if source is None:
            raise RuntimeError(f"missing {mic} rendering for {speaker}/{utterance}")
        key = source["source_path"]
        if key not in assets:
            prepared = preparer.prepare(source["source_path"], {
                "speaker_id": speaker,
                "utterance_id": utterance,
                "mic": mic,
                "source_sample_rate_hz": source["sample_rate_hz"],
                "source_channels": source["channels"],
                "source_duration_sec": source["duration_sec"],
            })
            job_id = prepared["source_sha256"][:24]
            assets[key] = {
                "job_id": job_id,
                "speaker_id": speaker,
                "utterance_id": utterance,
                "mic": mic,
                "source_path": source["source_path"],
                "source_sha256": prepared["source_sha256"],
                "converted_sha256": prepared["converted_sha256"],
                "vad_sha256": prepared["vad_sha256"],
                "audio_file": prepared["audio_file"],
                "vad_mask_file": prepared["vad_mask_file"],
                "speech_duration_sec": prepared["speech_duration_sec"],
                "source_duration_sec": source["duration_sec"],
                "partition_role": partition,
            }
            if len(assets) % 100 == 0:
                print(f"prepared source assets: {len(assets)}", file=sys.stderr, flush=True)
        return assets[key]

    folds: list[dict[str, Any]] = []
    profile_jobs_by_fold: dict[str, list[dict[str, Any]]] = {}
    context_jobs: list[dict[str, Any]] = []
    for fold_index, seed in enumerate(args.seeds[:args.folds]):
        rng = random.Random(seed)
        order = list(speakers)
        rng.shuffle(order)
        cursor = 0
        roles: dict[str, list[str]] = {}
        for role, count in ROLE_COUNTS.items():
            roles[role] = order[cursor:cursor + count]
            cursor += count
        if cursor != 110 or any(len(roles[role]) != count for role, count in ROLE_COUNTS.items()):
            raise RuntimeError("speaker role allocation does not sum to the full VCTK population")

        def role_assets(role_speakers: list[str], source_partition: str, include_mic2: bool = False) -> dict[str, Any]:
            people = {}
            for speaker in role_speakers:
                sets = utterance_sets[speaker]
                enrollment = [prepare_asset(speaker, utt, "mic1", source_partition)
                              for utt in sets["registration"]]
                queries = [prepare_asset(speaker, utt, "mic1", source_partition)
                           for utt in sets["independent_query"]]
                mic2_queries = []
                if include_mic2:
                    for utt in sets["independent_query"]:
                        if "mic2" in source_rows[speaker][utt]:
                            mic2_queries.append(prepare_asset(speaker, utt, "mic2", source_partition))
                        if len(mic2_queries) == 4:
                            break
                people[speaker] = {
                    "external_identity_id": f"user_{speaker}",
                    "enrollment_job_ids": [item["job_id"] for item in enrollment],
                    "query_job_ids": [item["job_id"] for item in queries],
                    "query_mic2_job_ids": [item["job_id"] for item in mic2_queries],
                    "enrollment_utterance_ids": [item["utterance_id"] for item in enrollment],
                    "query_utterance_ids": [item["utterance_id"] for item in queries],
                }
            return people

        train_people: dict[str, list[str]] = {}
        for speaker in roles["train"]:
            items = [prepare_asset(speaker, utt, "mic1", "head_training")
                     for utt in utterance_sets[speaker]["head_training"]]
            train_people[speaker] = [item["job_id"] for item in items]
        selection = role_assets(roles["selection"], "model_selection")
        calibration = role_assets(roles["calibration"], "threshold_calibration")
        evaluation = role_assets(roles["evaluation"], "locked_evaluation", include_mic2=True)
        eval_order = list(roles["evaluation"])
        random.Random(f"task09b-registry-order:{seed}").shuffle(eval_order)
        calibration_order = list(roles["calibration"])
        random.Random(f"task09b-calibration-order:{seed}").shuffle(calibration_order)
        selection_order = list(roles["selection"])
        random.Random(f"task09b-selection-order:{seed}").shuffle(selection_order)
        job_by_id = {item["job_id"]: item for item in assets.values()}
        profile_speakers = roles["train"][:8]
        profile_jobs_by_fold[f"fold_{fold_index + 1}"] = [job_by_id[job_id]
            for speaker in profile_speakers for job_id in train_people[speaker]]
        silence_path = cache_dir / "task09b_context_silence_2_56s.wav"
        silence_sha = write_silence(silence_path)
        profile_name = f"aosc_context_profile_fold_{fold_index + 1}.bin"
        context_modes = [
            ("silent_2_56s", str(silence_path), "", silence_sha, ""),
            ("unrelated_train_speaker_a", str(cache_dir / job_by_id[train_people[profile_speakers[0]][0]]["audio_file"]),
             profile_speakers[0], job_by_id[train_people[profile_speakers[0]][0]]["converted_sha256"], ""),
            ("unrelated_train_speaker_b", str(cache_dir / job_by_id[train_people[profile_speakers[1]][0]]["audio_file"]),
             profile_speakers[1], job_by_id[train_people[profile_speakers[1]][0]]["converted_sha256"], ""),
            ("aosc_profile_import", "", "", "", profile_name),
        ]
        target_speakers = eval_order[:5] + eval_order[-5:]
        fold_contexts = []
        for target_speaker in target_speakers:
            for target_job_id in evaluation[target_speaker]["query_job_ids"][:2]:
                target = job_by_id[target_job_id]
                for mode, context_wav, context_speaker, context_sha, profile_path in context_modes:
                    mode_key = f"fold_{fold_index + 1}:{target_job_id}:{mode}:{context_sha}:{Path(profile_path).name if profile_path else ''}"
                    context_job_id = hashlib.sha256(mode_key.encode()).hexdigest()[:24]
                    row = {"job_id": context_job_id, "target_job_id": target_job_id,
                           "speaker_id": target["speaker_id"], "utterance_id": target["utterance_id"],
                           "mic": target["mic"], "source_path": target["source_path"],
                           "source_sha256": target["source_sha256"], "converted_sha256": target["converted_sha256"],
                           "vad_sha256": target["vad_sha256"], "audio_file": target["audio_file"],
                           "vad_mask_file": target["vad_mask_file"], "context_mode": mode,
                           "context_wav_path": context_wav, "context_speaker_id": context_speaker or None,
                           "context_sha256": context_sha or None,
                           "aosc_profile_name": Path(profile_path).name if profile_path else None,
                           "aosc_profile_path": str(cache_dir / profile_path) if profile_path else ""}
                    context_jobs.append(row)
                    fold_contexts.append({"context_job_id": context_job_id, "target_job_id": target_job_id,
                                          "speaker_id": target["speaker_id"], "utterance_id": target["utterance_id"],
                                          "context_mode": mode, "context_speaker_id": context_speaker or None,
                                          "context_sha256": context_sha or None,
                                          "aosc_profile_name": Path(profile_path).name if profile_path else None})
        folds.append({
            "fold_id": f"fold_{fold_index + 1}",
            "speaker_seed": seed,
            "role_speakers": roles,
            "train_utterance_job_ids": train_people,
            "selection_people": selection,
            "calibration_people": calibration,
            "evaluation_people": evaluation,
            "selection_registry_order": selection_order,
            "calibration_registry_order": calibration_order,
            "evaluation_registry_order": eval_order,
            "context_profile_speaker_ids": profile_speakers,
            "context_probe_jobs": fold_contexts,
            "registry_sizes": REGISTRY_SIZES,
            "registry_rule": "nested prefix of the fixed per-fold registry order; external user IDs are independent of Sortformer channels",
            "enrollment_budgets_sec": [5, 10, 15, 30, 60],
            "query_speech_budgets_sec": [1, 3, 7, 15],
        })

    # A source is extracted once under an empty stream and is then referenced
    # by one or more fold roles. Fold-local role and utterance disjointness are
    # validated below; cross-fold reuse is explicit and never mixes head state.
    jobs = sorted(assets.values(), key=lambda item: item["job_id"])
    job_ids = [item["job_id"] for item in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise RuntimeError("source SHA prefix collision in job catalog")
    asset_by_id = {item["job_id"]: item for item in jobs}

    for fold in folds:
        role_sets = {role: set(speakers_for_role)
                     for role, speakers_for_role in fold["role_speakers"].items()}
        for role, expected_count in ROLE_COUNTS.items():
            if len(role_sets[role]) != expected_count:
                raise RuntimeError(f"speaker count mismatch in {fold['fold_id']} role {role}")
        role_names = list(role_sets)
        for left_index, left in enumerate(role_names):
            for right in role_names[left_index + 1:]:
                if role_sets[left] & role_sets[right]:
                    raise RuntimeError(f"speaker leakage between {left} and {right} in {fold['fold_id']}")

        all_by_speaker: dict[str, set[str]] = {}
        train_by_speaker = fold["train_utterance_job_ids"]
        for speaker, job_list in fold["train_utterance_job_ids"].items():
            all_by_speaker.setdefault(speaker, set()).update(job_list)
        for field in ("selection_people", "calibration_people", "evaluation_people"):
            for speaker, person in fold[field].items():
                ids = set(person["enrollment_job_ids"])
                queries = set(person["query_job_ids"])
                if ids & queries:
                    raise RuntimeError(f"mic1 enrollment/query utterance leak for {fold['fold_id']} {speaker}")
                if train_by_speaker.get(speaker, set()) & (ids | queries):
                    raise RuntimeError(f"head-training/enrollment/query utterance leak for {fold['fold_id']} {speaker}")
                if field == "evaluation_people":
                    query_utterances = set(person["query_utterance_ids"])
                    for job_id in person["query_mic2_job_ids"]:
                        item = asset_by_id.get(job_id)
                        if item is None or item["mic"] != "mic2" or item["utterance_id"] not in query_utterances:
                            raise RuntimeError(f"mic2 job is not paired to an evaluation query utterance: {job_id}")
                all_by_speaker.setdefault(speaker, set()).update(ids | queries)
        if len(all_by_speaker) != 110:
            raise RuntimeError(f"speaker coverage failure in {fold['fold_id']}")
        if set(fold["role_speakers"]["train"]) & set(fold["role_speakers"]["evaluation"]):
            raise RuntimeError("speaker identity leakage into evaluation role")

    compact_assets = [{key: item[key] for key in (
        "job_id", "speaker_id", "utterance_id", "mic", "source_path", "source_sha256",
        "converted_sha256", "vad_sha256", "speech_duration_sec", "source_duration_sec",
    )} for item in jobs]
    manifest = {
        "schema": "nemo-speech.voiceid-task09b-vctk-splits.v1",
        "base_commit": "614d1008274498ac5976af69e9ef3e27e597c7a2",
        "dataset": "VCTK-Corpus-0.92",
        "preparation_runtime": {"python_version": sys.version.split()[0], "ffmpeg_version": preparer.ffmpeg_version,
                                 "conversion": CONVERSION, "vad_parameters": VAD},
        "inventory": {"usable_mic1_speakers": len(speakers), "speaker_ids": speakers},
        "speaker_roles_per_fold": ROLE_COUNTS,
        "fold_seeds": args.seeds[:args.folds],
        "primary_microphone": "mic1",
        "mic2_protocol": "four paired query utterances when available; same source utterance excluded from registration set",
        "utterance_allocation": {
            "per_speaker_stable_hash_order": True,
            "head_training_utterances": TRAIN_UTTERANCES,
            "registration_utterances": ENROLL_UTTERANCES,
            "independent_query_utterances": QUERY_UTTERANCES,
            "cross_role_overlap": False,
            "microphone_pairing_rule": "mic1/mic2 files share one utterance ID and never cross enrollment/query roles",
        },
        "feature_extraction_protocol": {
            "source_representation_candidates": ["pre_encoder", "encoder.layer.7", "encoder_output", "encoder_proj"],
            "model": "models/Nemotron-3-Diarization.q8_0.gguf",
            "initial_state": "fresh empty stream per source utterance",
            "geometry": "V3 streaming, 160ms push blocks, committed 80ms feature frames",
            "mask": "Task08 10ms RMS VAD, projected to 80ms frames by majority vote",
        },
        "registration_and_query_protocol": {
            "external_id_rule": "user_<speaker_id>; persistent identity is independent of native speaker-channel number",
            "registration_identity_speakers": "registry prefix only",
            "unknown_speakers": "evaluation identities after registry prefix",
            "registry_sizes": REGISTRY_SIZES,
            "actual_speech_duration": "sum VAD-active 80ms feature frames; include whole utterances until the budget is reached and report measured overshoot",
            "selection_roles": "selection speakers choose model/checkpoint only",
            "calibration_roles": "calibration speakers choose one frozen global open-set threshold per model",
            "evaluation_roles": "evaluation speakers are excluded from head training, model selection, and threshold selection",
        },
        "validation": {
            "status": "passed",
            "checks": [
                "exactly 110 usable speaker IDs per fold",
                "50/15/15/30 speaker-disjoint roles per fold",
                "all four speaker role sets are pairwise disjoint",
                "head-training utterance groups are disjoint from registration and independent query groups",
                "mic1 utterance groups are disjoint between registration and independent query per speaker",
                "every mic2 robustness asset is paired to a mic1 query utterance",
                "all source audio has a SHA-256 and unique source job ID",
            ],
            "cross_fold_identity_reuse": "allowed; each fold head and evaluation are scored independently, speaker-clustered uncertainty is required",
        },
        "assets": compact_assets,
        "folds": folds,
    }
    jobs_path = args.jobs_tsv or cache_dir / "task09b_extraction_jobs.tsv"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    with jobs_path.open("w", encoding="utf-8") as output:
        output.write("# job_id\tspeaker_id\tutterance_id\tmic\tsource_path\tsource_sha256\tconverted_sha256\tvad_sha256\twav_path\tvad_path\n")
        for item in jobs:
            output.write("\t".join([
                item["job_id"], item["speaker_id"], item["utterance_id"], item["mic"], item["source_path"],
                item["source_sha256"], item["converted_sha256"], item["vad_sha256"],
                str(cache_dir / item["audio_file"]), str(cache_dir / item["vad_mask_file"]),
            ]) + "\n")
    context_jobs_path = cache_dir / "task09b_context_jobs.tsv"
    with context_jobs_path.open("w", encoding="utf-8") as output:
        output.write("# job_id\tspeaker_id\tutterance_id\tmic\tsource_path\tsource_sha256\tconverted_sha256\tvad_sha256\twav_path\tvad_path\tcontext_wav_path\tcontext_mode\tcontext_speaker_id\tcontext_sha256\taosc_profile_path\n")
        for item in context_jobs:
            output.write("\t".join([
                item["job_id"], item["speaker_id"], item["utterance_id"], item["mic"], item["source_path"],
                item["source_sha256"], item["converted_sha256"], item["vad_sha256"],
                str(cache_dir / item["audio_file"]), str(cache_dir / item["vad_mask_file"]),
                item["context_wav_path"], item["context_mode"], item["context_speaker_id"] or "",
                item["context_sha256"] or "", item["aosc_profile_path"] or "",
            ]) + "\n")
    profile_job_paths = {}
    for fold_id, profile_jobs in profile_jobs_by_fold.items():
        path = cache_dir / f"task09b_aosc_profile_jobs_{fold_id}.tsv"
        with path.open("w", encoding="utf-8") as output:
            output.write("# job_id\tspeaker_id\tutterance_id\tmic\tsource_path\tsource_sha256\tconverted_sha256\tvad_sha256\twav_path\tvad_path\n")
            for item in profile_jobs:
                output.write("\t".join([
                    item["job_id"], item["speaker_id"], item["utterance_id"], item["mic"], item["source_path"],
                    item["source_sha256"], item["converted_sha256"], item["vad_sha256"],
                    str(cache_dir / item["audio_file"]), str(cache_dir / item["vad_mask_file"]),
                ]) + "\n")
        profile_job_paths[fold_id] = path.name
    manifest["context_stress_protocol"] = {
        "status": "predeclared; identity-head evaluation IDs never appear in context audio or AOSC profile inputs",
        "modes": ["fresh_empty_stream", "silent_2_56s", "unrelated_train_speaker_a", "unrelated_train_speaker_b", "aosc_profile_import"],
        "target_count_per_fold": len(folds[0]["context_probe_jobs"]) // 4 if folds else 0,
        "context_probe_jobs": [{key: item[key] for key in (
            "job_id", "target_job_id", "speaker_id", "utterance_id", "context_mode",
            "context_speaker_id", "context_sha256", "aosc_profile_name")}
            for item in context_jobs],
        "aosc_profile_job_tsv_by_fold": profile_job_paths,
        "aosc_profile_duration_protocol": "20 mic1 training utterances per each of eight head-training speakers, 250ms silence after every utterance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, manifest)
    print(json.dumps({"manifest": str(args.output), "jobs_tsv": str(jobs_path),
                      "context_jobs_tsv": str(context_jobs_path), "aosc_profile_job_tsvs": profile_job_paths,
                      "context_job_count": len(context_jobs), "folds": len(folds),
                      "speaker_count": len(speakers), "unique_source_jobs": len(jobs),
                      "cache_dir": str(cache_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"prepare_task09b_splits: {exc}", file=sys.stderr)
        raise
