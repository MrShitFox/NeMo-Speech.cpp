#!/usr/bin/env python3
"""Run fixed-profile CPU/Vulkan parity experiments for Task 09A PV-01..03."""

from __future__ import annotations

import argparse
import copy
import collections
import hashlib
import json
import math
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any


CASES = (
    ("PV-01", "final_holdout_n4_r00_e60_randomized", "p268_p268_259_2_to_5s_mic2"),
    ("PV-02", "final_holdout_n8_r00_e60_randomized", "p345_p345_001_under_2s_mic1"),
    ("PV-03", "final_holdout_n8_r00_e60_randomized", "p345_p345_126_2_to_5s_mic2"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fnv1a64_hex(data: bytes) -> str:
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def physical_gpu_metadata(device_index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi", "--id", str(device_index),
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {
            "requested_device_index": device_index,
            "status": "unavailable",
            "reason": str(error),
        }
    fields = [field.strip() for field in completed.stdout.strip().split(",", 3)]
    if len(fields) != 4:
        return {
            "requested_device_index": device_index,
            "status": "unavailable",
            "reason": f"unexpected nvidia-smi output: {completed.stdout.strip()}",
        }
    return {
        "requested_device_index": device_index,
        "status": "available",
        "name": fields[0],
        "uuid": fields[1],
        "memory_total": fields[2],
        "driver_version": fields[3],
        "source": "nvidia-smi",
    }


def write_manifest(path: Path, source: dict[str, Any], trials: list[dict[str, Any]]) -> None:
    manifest = {key: value for key, value in source.items() if key not in ("trials", "preparation_assets")}
    manifest["trials"] = trials
    write_json(path, manifest)


def base_trial(source_trial: dict[str, Any], query_id: str) -> dict[str, Any]:
    trial = copy.deepcopy(source_trial)
    query = next((item for item in trial["queries"] if item["query_id"] == query_id), None)
    if query is None:
        raise ValueError(f"Task 08 query is missing: {trial['trial_id']}/{query_id}")
    trial["queries"] = [query]
    trial["conversations"] = []
    trial["run_unknown_traffic"] = False
    trial["unknown_traffic_query_ids"] = []
    trial.pop("task09a_identity_to_channel", None)
    return trial


def source_manifests(
    source: dict[str, Any], work_dir: Path,
) -> tuple[Path, dict[str, Path]]:
    trials_by_id = {item["trial_id"]: item for item in source["trials"]}
    query_ids_by_trial: dict[str, list[str]] = {}
    for _, trial_id, query_id in CASES:
        query_ids_by_trial.setdefault(trial_id, []).append(query_id)
    all_trials = []
    for trial_id, query_ids in query_ids_by_trial.items():
        trial = base_trial(trials_by_id[trial_id], query_ids[0])
        trial["queries"] = [
            query for query_id in query_ids
            for query in base_trial(trials_by_id[trial_id], query_id)["queries"]
        ]
        all_trials.append(trial)
    build_manifest_path = work_dir / "manifest_pv_builds.json"
    write_manifest(build_manifest_path, source, all_trials)
    fixed: dict[str, Path] = {}
    for case_name, trial_id, query_id in CASES:
        trial = base_trial(trials_by_id[trial_id], query_id)
        fixed[case_name] = work_dir / f"manifest_{case_name.lower()}_fixed_profile.json"
        write_manifest(fixed[case_name], source, [trial])
    return build_manifest_path, fixed


def invoke(
    name: str,
    probe: Path,
    manifest: Path,
    cache_dir: Path,
    work_dir: Path,
    device: str,
    gpu_device: int,
    profile_in: Path | None = None,
    diagnostics: bool = False,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    run_dir = work_dir / "runs" / name
    trace_dir = run_dir / "traces"
    profile_dir = run_dir / "profiles"
    output = run_dir / "results.jsonl"
    for folder in (trace_dir, profile_dir):
        folder.mkdir(parents=True, exist_ok=True)
    command = [
        str(probe), "--manifest", str(manifest), "--cache-dir", str(cache_dir),
        "--trace-dir", str(trace_dir), "--profile-dir", str(profile_dir),
        "--output-jsonl", str(output), "--device", device, "--threads", "4",
        "--concurrency", "4", "--no-resume",
    ]
    if device == "vulkan":
        command.extend(("--gpu-device", str(gpu_device)))
    if profile_in is not None:
        command.extend(("--profile-in", str(profile_in)))
    if diagnostics:
        command.append("--task09a-diagnostics")
    log_path = run_dir / "run.log"
    if reuse_existing and output.exists():
        rows = read_jsonl(output)
        run_event_path = Path(str(output) + ".runs.jsonl")
        run_events = read_jsonl(run_event_path) if run_event_path.exists() else []
        if not rows:
            raise RuntimeError(f"{name} has an empty existing result file: {output}")
        return {
            "name": name,
            "device": device,
            "manifest": str(manifest),
            "profile_input": str(profile_in) if profile_in else None,
            "output": str(output),
            "run_event_path": str(run_event_path),
            "run_events": run_events,
            "trace_dir": str(trace_dir),
            "profile_dir": str(profile_dir),
            "log": str(log_path),
            "command": command,
            "rows": rows,
        }
    print(f"RUN {name}: {device}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}; see {log_path}")
    rows = read_jsonl(output)
    run_event_path = Path(str(output) + ".runs.jsonl")
    run_events = read_jsonl(run_event_path) if run_event_path.exists() else []
    return {
        "name": name,
        "device": device,
        "manifest": str(manifest),
        "profile_input": str(profile_in) if profile_in else None,
        "output": str(output),
        "run_event_path": str(run_event_path),
        "run_events": run_events,
        "trace_dir": str(trace_dir),
        "profile_dir": str(profile_dir),
        "log": str(log_path),
        "command": command,
        "rows": rows,
    }


def read_trace(path: Path) -> tuple[int, int, float, list[float]]:
    data = path.read_bytes()
    if len(data) < 32 or data[:8] != b"NMSP8F1\0":
        raise ValueError(f"invalid probability trace: {path}")
    version, channels = struct.unpack_from("<II", data, 8)
    frames = struct.unpack_from("<Q", data, 16)[0]
    period = struct.unpack_from("<d", data, 24)[0]
    if version != 1 or len(data) != 32 + frames * channels * 4:
        raise ValueError(f"trace geometry disagrees with bytes: {path}")
    values = list(struct.unpack_from(f"<{frames * channels}f", data, 32))
    return frames, channels, period, values


def trace_diff(left_path: Path, right_path: Path) -> dict[str, Any]:
    lf, lc, lp, left = read_trace(left_path)
    rf, rc, rp, right = read_trace(right_path)
    if lf != rf or lc != rc or abs(lp - rp) > 1e-9:
        raise ValueError(f"probability trace geometry mismatch: {left_path} vs {right_path}")
    diffs = [abs(a - b) for a, b in zip(left, right)]
    top1 = 0
    activity = 0
    for frame in range(lf):
        a = left[frame * lc:(frame + 1) * lc]
        b = right[frame * rc:(frame + 1) * rc]
        top1 += max(range(lc), key=a.__getitem__) != max(range(rc), key=b.__getitem__)
        activity += (max(a) >= 0.5) != (max(b) >= 0.5)
    return {
        "frames": lf,
        "channels": lc,
        "frame_period_sec": lp,
        "max_absolute_probability_error": max(diffs, default=0.0),
        "mean_absolute_probability_error": sum(diffs) / len(diffs) if diffs else 0.0,
        "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs)) if diffs else 0.0,
        "top1_channel_changed_frames": top1,
        "activity_ge_0_5_changed_frames": activity,
        "activity_threshold_crossings": activity,
    }


def segment_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a = left.get("native_segment_boundaries", [])
    b = right.get("native_segment_boundaries", [])
    paired = min(len(a), len(b))
    start_errors = [abs(float(a[i]["start_sec"]) - float(b[i]["start_sec"])) for i in range(paired)]
    end_errors = [abs(float(a[i]["end_sec"]) - float(b[i]["end_sec"])) for i in range(paired)]
    return {
        "left_segment_count": len(a),
        "right_segment_count": len(b),
        "segment_count_delta": len(b) - len(a),
        "paired_in_order_count": paired,
        "paired_channel_changes": sum(a[i]["channel"] != b[i]["channel"] for i in range(paired)),
        "max_paired_start_boundary_error_sec": max(start_errors, default=0.0),
        "max_paired_end_boundary_error_sec": max(end_errors, default=0.0),
    }


def aosc_query_state_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a = left.get("aosc_query_compression_events", [])
    b = right.get("aosc_query_compression_events", [])
    event_rows = []
    for index in range(max(len(a), len(b))):
        event_a = a[index] if index < len(a) else None
        event_b = b[index] if index < len(b) else None
        input_a = event_a.get("input_source_frame_ids", []) if event_a else []
        input_b = event_b.get("input_source_frame_ids", []) if event_b else []
        retained_a = event_a.get("retained_source_frame_ids", []) if event_a else []
        retained_b = event_b.get("retained_source_frame_ids", []) if event_b else []
        event_rows.append({
            "compression_index": index,
            "left_event_present": event_a is not None,
            "right_event_present": event_b is not None,
            "input_source_rows_equal": input_a == input_b,
            "retained_source_rows_equal": retained_a == retained_b,
            "left_retained_query_audio_rows": len(event_a.get("retained_query_audio_source_frame_ids", [])) if event_a else 0,
            "right_retained_query_audio_rows": len(event_b.get("retained_query_audio_source_frame_ids", [])) if event_b else 0,
            "left_positive_rows_by_channel_after": event_a.get("after", {}).get("positive_rows_by_channel_gt_0_5") if event_a else None,
            "right_positive_rows_by_channel_after": event_b.get("after", {}).get("positive_rows_by_channel_gt_0_5") if event_b else None,
            "left_cache_rows_after": event_a.get("after", {}).get("cache_rows") if event_a else None,
            "right_cache_rows_after": event_b.get("after", {}).get("cache_rows") if event_b else None,
        })
    return {
        "left_compression_event_count": len(a),
        "right_compression_event_count": len(b),
        "compression_event_count_equal": len(a) == len(b),
        "events": event_rows,
    }


def parse_profile(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 113 or data[:8] != b"NMSAOSC1":
        raise ValueError(f"invalid AOSC profile: {path}")
    version, architecture = struct.unpack_from("<II", data, 8)
    sample_rate, channels, dimension, output_factor = struct.unpack_from("<IIII", data, 48)
    speaker_cache_len, fifo_len, chunk_len, update_period, left_context, right_context = struct.unpack_from("<iiiiii", data, 64)
    cache_frames = struct.unpack_from("<i", data, 88)[0]
    payload_size = struct.unpack_from("<Q", data, 101)[0]
    checksum = struct.unpack_from("<I", data, 109)[0]
    payload = data[113:]
    if len(payload) != payload_size or zlib.crc32(payload) & 0xFFFFFFFF != checksum:
        raise ValueError(f"AOSC payload checksum or size mismatch: {path}")
    offset = 0
    cache_count = struct.unpack_from("<Q", payload, offset)[0]
    offset += 8
    cache = list(struct.unpack_from(f"<{cache_count}f", payload, offset))
    offset += cache_count * 4
    prediction_count = struct.unpack_from("<Q", payload, offset)[0]
    offset += 8
    predictions = list(struct.unpack_from(f"<{prediction_count}f", payload, offset))
    offset += prediction_count * 4
    silence_count = struct.unpack_from("<Q", payload, offset)[0]
    offset += 8
    silence = list(struct.unpack_from(f"<{silence_count}f", payload, offset))
    offset += silence_count * 4
    if offset != len(payload) or cache_count != cache_frames * dimension or prediction_count != cache_frames * channels:
        raise ValueError(f"AOSC payload vector dimensions disagree: {path}")
    positive = [0] * channels
    argmax = [0] * channels
    channel_counts = [0] * channels
    channel_sums = [[0.0] * dimension for _ in range(channels)]
    for row in range(cache_frames):
        values = predictions[row * channels:(row + 1) * channels]
        winner = max(range(channels), key=values.__getitem__)
        for channel, value in enumerate(values):
            if value > 0.5:
                positive[channel] += 1
                channel_counts[channel] += 1
                embedding = cache[row * dimension:(row + 1) * dimension]
                for index, value in enumerate(embedding):
                    channel_sums[channel][index] += value
        if values[winner] > 0.5:
            argmax[winner] += 1
    centroids = []
    for channel, count in enumerate(channel_counts):
        centroids.append([value / count for value in channel_sums[channel]] if count else [])
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "crc32": checksum,
        "version": version,
        "architecture": architecture,
        "sample_rate": sample_rate,
        "speaker_channels": channels,
        "embedding_dimension": dimension,
        "output_factor": output_factor,
        "speaker_cache_len": speaker_cache_len,
        "fifo_len": fifo_len,
        "chunk_len": chunk_len,
        "speaker_cache_update_period": update_period,
        "chunk_left_context": left_context,
        "chunk_right_context": right_context,
        "cache_frames": cache_frames,
        "acoustic_cache": cache,
        "prediction_state": predictions,
        "learned_silence_state": silence,
        "positive_prediction_rows_by_channel_gt_0_5": positive,
        "argmax_prediction_rows_by_channel_gt_0_5": argmax,
        "acoustic_positive_row_counts_by_channel_gt_0_5": channel_counts,
        "acoustic_centroids_by_channel_gt_0_5": centroids,
    }


def float_diff(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right):
        return {"left_values": len(left), "right_values": len(right), "shape_equal": False}
    diffs = [abs(a - b) for a, b in zip(left, right)]
    return {
        "shape_equal": True,
        "values_compared": len(diffs),
        "exactly_equal_values": sum(a == b for a, b in zip(left, right)),
        "different_values": sum(a != b for a, b in zip(left, right)),
        "max_absolute_difference": max(diffs, default=0.0),
        "mean_absolute_difference": sum(diffs) / len(diffs) if diffs else 0.0,
        "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs)) if diffs else 0.0,
    }


def row_selection_diff(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_events = left.get("task09a_enrollment_diagnostics", {}).get("aosc_compression_events", [])
    right_events = right.get("task09a_enrollment_diagnostics", {}).get("aosc_compression_events", [])
    output = []
    for index in range(max(len(left_events), len(right_events))):
        a = left_events[index] if index < len(left_events) else None
        b = right_events[index] if index < len(right_events) else None
        def compare_rows(left_ids: list[int], right_ids: list[int], prefix: str) -> dict[str, Any]:
            set_a = {frame for frame in left_ids if frame >= 0}
            set_b = {frame for frame in right_ids if frame >= 0}
            union = set_a | set_b
            multiset_common = sum((collections.Counter(left_ids) & collections.Counter(right_ids)).values())
            return {
                f"{prefix}_left_rows": len(left_ids),
                f"{prefix}_right_rows": len(right_ids),
                f"{prefix}_same_ordered_source_frame_selection": left_ids == right_ids,
                f"{prefix}_same_source_frame_set": set_a == set_b,
                f"{prefix}_source_frame_set_intersection": len(set_a & set_b),
                f"{prefix}_source_frame_set_union": len(union),
                f"{prefix}_source_frame_set_jaccard": len(set_a & set_b) / len(union) if union else 1.0,
                f"{prefix}_multiset_row_intersection_including_silence_sentinel": multiset_common,
                f"{prefix}_left_silence_or_unavailable_rows": sum(frame < 0 for frame in left_ids),
                f"{prefix}_right_silence_or_unavailable_rows": sum(frame < 0 for frame in right_ids),
            }

        input_left = a.get("input_source_frame_ids", []) if a else []
        input_right = b.get("input_source_frame_ids", []) if b else []
        retained_left = a.get("retained_source_frame_ids", []) if a else []
        retained_right = b.get("retained_source_frame_ids", []) if b else []
        output.append({
            "compression_index": index,
            **compare_rows(input_left, input_right, "input_before_compression"),
            **compare_rows(retained_left, retained_right, "retained_after_compression"),
        })
    return output


def result_by_trial(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for item in run["rows"]:
        trial_id = item["trial_id"]
        if trial_id not in results:
            results[trial_id] = item
            continue
        # PV-02 and PV-03 are two queries against the same enrollment profile.
        # Keep both query rows when the construction manifest contains that
        # profile twice with a one-query slice for each disagreement case.
        merged = results[trial_id]
        by_query_id = {query.get("query_id"): query for query in merged.get("queries", [])}
        by_query_id.update({query.get("query_id"): query for query in item.get("queries", [])})
        merged["queries"] = list(by_query_id.values())
        conversations = {entry.get("conversation_id"): entry for entry in merged.get("conversations", [])}
        conversations.update({entry.get("conversation_id"): entry for entry in item.get("conversations", [])})
        merged["conversations"] = list(conversations.values())
    return results


def profile_path(run: dict[str, Any], trial_id: str) -> Path:
    return Path(run["profile_dir"]) / f"{trial_id}.nmsaosc"


def query_record(run: dict[str, Any], trial_id: str, query_id: str) -> dict[str, Any]:
    result = result_by_trial(run)[trial_id]
    return next(query for query in result["queries"] if query["query_id"] == query_id)


def query_trace(run: dict[str, Any], trial_id: str, query_id: str) -> Path:
    query = query_record(run, trial_id, query_id)
    return Path(run["trace_dir"]) / query["trace_file"]


def run_contains_query(run: dict[str, Any], trial_id: str, query_id: str) -> bool:
    try:
        query_record(run, trial_id, query_id)
        return True
    except StopIteration:
        return False


def compare_queries(
    left_run: dict[str, Any], right_run: dict[str, Any], trial_id: str, query_id: str,
) -> dict[str, Any]:
    left = query_record(left_run, trial_id, query_id)
    right = query_record(right_run, trial_id, query_id)
    return {
        "left_run": left_run["name"],
        "right_run": right_run["name"],
        "cpu_or_vulkan_decision_left": left.get("baseline_category"),
        "cpu_or_vulkan_decision_right": right.get("baseline_category"),
        "predicted_identity_left": left.get("predicted_identity"),
        "predicted_identity_right": right.get("predicted_identity"),
        "decision_category_changed": left.get("baseline_category") != right.get("baseline_category"),
        "identity_changed": left.get("predicted_identity") != right.get("predicted_identity"),
        "activity_frames_left": left.get("activity_frames"),
        "activity_frames_right": right.get("activity_frames"),
        "profile_import_ms_left": left.get("profile_import_ms"),
        "profile_import_ms_right": right.get("profile_import_ms"),
        "query_inference_ms_left": left.get("inference_ms"),
        "query_inference_ms_right": right.get("inference_ms"),
        "segment_differences": segment_diff(left, right),
        "aosc_query_state_differences": aosc_query_state_diff(left, right),
        "probability_differences": trace_diff(
            query_trace(left_run, trial_id, query_id), query_trace(right_run, trial_id, query_id)),
    }


def aggregate(
    runs: dict[str, dict[str, Any]],
    task08_aggregate: dict[str, Any],
    task08_raw_results: list[dict[str, Any]],
    task08_profile_dir: Path | None,
) -> dict[str, Any]:
    old_cases = {
        item["query_id"]: item
        for item in task08_aggregate.get("cpu_vulkan_correctness_sampling", {}).get("details", [])
    }
    profile_old = {
        item["trial_id"]: item
        for item in task08_aggregate.get("cpu_vulkan_correctness_sampling", {}).get("profile_details", [])
    }
    raw_old = {row["trial_id"]: row for row in task08_raw_results}
    cases = []
    for case_name, trial_id, query_id in CASES:
        cpu1, cpu2 = runs["cpu_build_1"], runs["cpu_build_2"]
        vk1, vk2 = runs["vulkan_build_1"], runs["vulkan_build_2"]
        cpu_repeat_first = cpu1
        if not run_contains_query(cpu_repeat_first, trial_id, query_id):
            # The original CPU pass completed PV-02's shared profile before
            # the compact two-query manifest was introduced. Its PV-03
            # repeat is evaluated from the exact serialized CPU profile.
            cpu_repeat_first = runs[f"cpu_profile_on_cpu_{case_name.lower()}"]
        cpu_profile = parse_profile(profile_path(cpu1, trial_id))
        cpu_repeat_profile = parse_profile(profile_path(cpu2, trial_id))
        vk_profile = parse_profile(profile_path(vk1, trial_id))
        vk_repeat_profile = parse_profile(profile_path(vk2, trial_id))
        cpu_result = result_by_trial(cpu1)[trial_id]
        vk_result = result_by_trial(vk1)[trial_id]
        case = {
            "case_id": case_name,
            "trial_id": trial_id,
            "query_id": query_id,
            "task08_identity_mapping_equal": profile_old.get(trial_id, {}).get("channel_mapping_equal"),
            "task08_cpu_profile_crc32": profile_old.get(trial_id, {}).get("cpu_profile_crc32"),
            "task08_vulkan_profile_crc32": profile_old.get(trial_id, {}).get("vulkan_profile_crc32"),
            "task08_decision": old_cases.get(query_id),
            "task08_original_profile_and_decision": {
                "aosc_ready": raw_old.get(trial_id, {}).get("aosc_ready"),
                "identity_mapping_valid": raw_old.get(trial_id, {}).get("identity_mapping_valid"),
                "identity_to_channel": raw_old.get(trial_id, {}).get("identity_to_channel"),
                "profile_crc32": raw_old.get(trial_id, {}).get("profile_crc32"),
                "query": next((q for q in raw_old.get(trial_id, {}).get("queries", []) if q.get("query_id") == query_id), None),
            },
            "source_profiles": {
                "cpu": {key: value for key, value in cpu_profile.items() if key not in ("acoustic_cache", "prediction_state", "learned_silence_state", "acoustic_centroids_by_channel_gt_0_5")},
                "vulkan": {key: value for key, value in vk_profile.items() if key not in ("acoustic_cache", "prediction_state", "learned_silence_state", "acoustic_centroids_by_channel_gt_0_5")},
                "cpu_identity_to_channel": cpu_result.get("identity_to_channel"),
                "vulkan_identity_to_channel": vk_result.get("identity_to_channel"),
                "cpu_vulkan_identity_map_equal": cpu_result.get("identity_to_channel") == vk_result.get("identity_to_channel"),
                "cpu_construction_timing": cpu_result.get("timing"),
                "vulkan_construction_timing": vk_result.get("timing"),
                "cpu_vs_vulkan_profile_state": {
                    "acoustic_cache_float_difference": float_diff(cpu_profile["acoustic_cache"], vk_profile["acoustic_cache"]),
                    "prediction_state_float_difference": float_diff(cpu_profile["prediction_state"], vk_profile["prediction_state"]),
                    "learned_silence_float_difference": float_diff(cpu_profile["learned_silence_state"], vk_profile["learned_silence_state"]),
                    "per_channel_acoustic_centroid_difference": [
                        {
                            "channel": channel,
                            "cpu_positive_rows": cpu_profile["acoustic_positive_row_counts_by_channel_gt_0_5"][channel],
                            "vulkan_positive_rows": vk_profile["acoustic_positive_row_counts_by_channel_gt_0_5"][channel],
                            "centroid_difference": float_diff(
                                cpu_profile["acoustic_centroids_by_channel_gt_0_5"][channel],
                                vk_profile["acoustic_centroids_by_channel_gt_0_5"][channel]),
                        }
                        for channel in range(cpu_profile["speaker_channels"])
                    ],
                    "retained_positive_rows_by_channel_differ": cpu_profile["positive_prediction_rows_by_channel_gt_0_5"] != vk_profile["positive_prediction_rows_by_channel_gt_0_5"],
                    "cpu_vs_vulkan_retained_source_rows": row_selection_diff(cpu_result, vk_result),
                },
                "same_backend_repeatability": {
                    "cpu_profile_bytes_equal": cpu_profile["sha256"] == cpu_repeat_profile["sha256"],
                    "cpu_profile_sha256_first": cpu_profile["sha256"],
                    "cpu_profile_sha256_repeat": cpu_repeat_profile["sha256"],
                    "vulkan_profile_bytes_equal": vk_profile["sha256"] == vk_repeat_profile["sha256"],
                    "vulkan_profile_sha256_first": vk_profile["sha256"],
                    "vulkan_profile_sha256_repeat": vk_repeat_profile["sha256"],
                    "cpu_query_repeatability_runs": {
                        "first": cpu_repeat_first["name"],
                        "repeat": cpu2["name"],
                    },
                    "cpu_query_trace_equal": query_trace(cpu_repeat_first, trial_id, query_id).read_bytes() == query_trace(cpu2, trial_id, query_id).read_bytes(),
                    "vulkan_query_trace_equal": query_trace(vk1, trial_id, query_id).read_bytes() == query_trace(vk2, trial_id, query_id).read_bytes(),
                    "cpu_query_probability_repeat_diff": trace_diff(query_trace(cpu_repeat_first, trial_id, query_id), query_trace(cpu2, trial_id, query_id)),
                    "vulkan_query_probability_repeat_diff": trace_diff(query_trace(vk1, trial_id, query_id), query_trace(vk2, trial_id, query_id)),
                    "cpu_retained_source_row_selection_repeat": row_selection_diff(cpu_result, result_by_trial(cpu2)[trial_id]),
                    "vulkan_retained_source_row_selection_repeat": row_selection_diff(vk_result, result_by_trial(vk2)[trial_id]),
                    "cpu_repeat_profile_state": {
                        "acoustic_cache_float_difference": float_diff(cpu_profile["acoustic_cache"], cpu_repeat_profile["acoustic_cache"]),
                        "prediction_state_float_difference": float_diff(cpu_profile["prediction_state"], cpu_repeat_profile["prediction_state"]),
                    },
                    "vulkan_repeat_profile_state": {
                        "acoustic_cache_float_difference": float_diff(vk_profile["acoustic_cache"], vk_repeat_profile["acoustic_cache"]),
                        "prediction_state_float_difference": float_diff(vk_profile["prediction_state"], vk_repeat_profile["prediction_state"]),
                    },
                },
                "saved_task08_vulkan_profile_equal": None,
            },
            "fixed_profile_factorial": {
                "same_cpu_built_profile_imported_on_cpu_vs_vulkan": compare_queries(
                    runs[f"cpu_profile_on_cpu_{case_name.lower()}"],
                    runs[f"cpu_profile_on_vulkan_{case_name.lower()}"], trial_id, query_id),
                "same_vulkan_built_profile_imported_on_cpu_vs_vulkan": compare_queries(
                    runs[f"vulkan_profile_on_cpu_{case_name.lower()}"],
                    runs[f"vulkan_profile_on_vulkan_{case_name.lower()}"], trial_id, query_id),
                "independently_built_profiles_imported_on_cpu": compare_queries(
                    runs[f"cpu_profile_on_cpu_{case_name.lower()}"],
                    runs[f"vulkan_profile_on_cpu_{case_name.lower()}"], trial_id, query_id),
                "independently_built_profiles_imported_on_vulkan": compare_queries(
                    runs[f"cpu_profile_on_vulkan_{case_name.lower()}"],
                    runs[f"vulkan_profile_on_vulkan_{case_name.lower()}"], trial_id, query_id),
            },
            "import_validation": {
                "cpu_source_profile_crc32": cpu_result.get("profile_crc32"),
                "vulkan_source_profile_crc32": vk_result.get("profile_crc32"),
                "imports": {
                    f"{source_backend}_source_on_{eval_backend}": {
                        "source_bytes_fnv1a64": fnv1a64_hex(profile_path(
                            runs[f"{source_backend}_build_1"], trial_id).read_bytes()),
                        "imported_bytes_fnv1a64": result_by_trial(
                            runs[f"{source_backend}_profile_on_{eval_backend}_{case_name.lower()}"]
                        )[trial_id].get("profile_import_bytes_fnv1a64"),
                        "source_crc32": result_by_trial(runs[f"{source_backend}_build_1"])[trial_id].get("profile_crc32"),
                        "imported_crc32": result_by_trial(
                            runs[f"{source_backend}_profile_on_{eval_backend}_{case_name.lower()}"]
                        )[trial_id].get("profile_crc32"),
                    }
                    for source_backend in ("cpu", "vulkan")
                    for eval_backend in ("cpu", "vulkan")
                },
            },
        }
        if task08_profile_dir:
            prior = task08_profile_dir / f"{trial_id}.nmsaosc"
            if prior.exists():
                previous = parse_profile(prior)
                case["source_profiles"]["saved_task08_vulkan_profile_equal"] = previous["sha256"] == vk_profile["sha256"]
                case["source_profiles"]["saved_task08_vulkan_profile_sha256"] = previous["sha256"]
                case["source_profiles"]["saved_task08_vulkan_profile_crc32"] = previous["crc32"]
        cases.append(case)
    return {
        "schema": "nemo-speech.voiceid-task09a-backend-parity.v1",
        "experiment_design": {
            "fixed_source_profiles": ["CPU-built profile bytes", "Vulkan-built profile bytes"],
            "evaluation_backends": ["CPU", "explicit Vulkan GPU 0"],
            "independent_profile_construction_repeats_per_backend": 2,
            "query_set": "exact Task 08 disagreements PV-01, PV-02, PV-03",
        "profile_imports": "same-backend and cross-backend imports run in separate probe processes from source profile construction",
            "notes": "Task 08 final-holdout cases are follow-up analyses; they are not new untouched holdout evidence.",
        },
        "run_metadata": [
            {
                **{key: value for key, value in run.items() if key not in ("rows", "command")},
                "backend_metadata": run["rows"][0].get("backend") if run.get("rows") else None,
                "model_metadata": run["rows"][0].get("model") if run.get("rows") else None,
            }
            for run in runs.values()
        ],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--task08-manifest", required=True, type=Path)
    parser.add_argument("--task08-results", required=True, type=Path)
    parser.add_argument("--task08-aggregate", required=True, type=Path)
    parser.add_argument("--task08-profile-dir", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--reuse-existing", action="store_true",
                        help="reuse completed run artifacts under --work-dir and execute only missing runs")
    args = parser.parse_args()
    source = read_json(args.task08_manifest)
    task08_aggregate = read_json(args.task08_aggregate)
    task08_raw_results = read_jsonl(args.task08_results)
    raw_by_id = {row["trial_id"]: row for row in task08_raw_results}
    missing = [trial_id for _, trial_id, _ in CASES if trial_id not in raw_by_id]
    if missing:
        raise ValueError(f"Task 08 raw results are missing fixed parity profiles: {missing}")
    for _, trial_id, query_id in CASES:
        if not any(q.get("query_id") == query_id for q in raw_by_id[trial_id].get("queries", [])):
            raise ValueError(f"Task 08 raw results are missing fixed parity query: {trial_id}/{query_id}")
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    build_manifest, fixed_manifests = source_manifests(source, work_dir)
    runs: dict[str, dict[str, Any]] = {}
    for backend in ("cpu", "vulkan"):
        build1_manifest = build_manifest
        if backend == "cpu" and args.reuse_existing:
            original_manifest = work_dir / "manifest_pv_builds_original.json"
            if original_manifest.exists():
                build1_manifest = original_manifest
        runs[f"{backend}_build_1"] = invoke(
            f"{backend}_build_1", args.probe.resolve(), build1_manifest, args.cache_dir.resolve(),
            work_dir, backend, args.gpu_device, diagnostics=True,
            reuse_existing=args.reuse_existing)
        runs[f"{backend}_build_2"] = invoke(
            f"{backend}_build_2", args.probe.resolve(), build_manifest, args.cache_dir.resolve(),
            work_dir, backend, args.gpu_device, diagnostics=True,
            reuse_existing=args.reuse_existing)

    build_records = {backend: result_by_trial(runs[f"{backend}_build_1"]) for backend in ("cpu", "vulkan")}
    for case_name, trial_id, query_id in CASES:
        for source_backend in ("cpu", "vulkan"):
            for eval_backend in ("cpu", "vulkan"):
                mapping = build_records[source_backend][trial_id]["identity_to_channel"]
                if not build_records[source_backend][trial_id].get("identity_mapping_valid"):
                    raise RuntimeError(f"source backend did not establish a strict identity map: {source_backend}/{trial_id}")
                trial_source = {item["trial_id"]: item for item in source["trials"]}[trial_id]
                trial = base_trial(trial_source, query_id)
                trial["task09a_identity_to_channel"] = mapping
                manifest = work_dir / f"manifest_{case_name.lower()}_{source_backend}_profile_{eval_backend}.json"
                write_manifest(manifest, source, [trial])
                profile = profile_path(runs[f"{source_backend}_build_1"], trial_id)
                run_name = f"{source_backend}_profile_on_{eval_backend}_{case_name.lower()}"
                runs[run_name] = invoke(
                    run_name, args.probe.resolve(), manifest, args.cache_dir.resolve(), work_dir,
                    eval_backend, args.gpu_device, profile_in=profile, diagnostics=True,
                    reuse_existing=args.reuse_existing)
                imported = result_by_trial(runs[run_name])[trial_id]
                if imported.get("profile_import_bytes_fnv1a64") is None:
                    raise RuntimeError(f"imported profile fingerprint missing in {run_name}")
                expected_fnv = fnv1a64_hex(profile.read_bytes())
                if imported["profile_import_bytes_fnv1a64"] != expected_fnv:
                    raise RuntimeError(f"imported AOSC bytes differ from source in {run_name}")
                if imported.get("profile_crc32") != build_records[source_backend][trial_id].get("profile_crc32"):
                    raise RuntimeError(f"imported AOSC CRC differs from source in {run_name}")
                if imported.get("identity_to_channel") != mapping:
                    raise RuntimeError(f"fixed identity map changed during imported-profile test in {run_name}")

    output = aggregate(runs, task08_aggregate, task08_raw_results,
                       args.task08_profile_dir.resolve() if args.task08_profile_dir else None)
    output["physical_gpu_metadata"] = physical_gpu_metadata(args.gpu_device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(output["cases"]), "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
