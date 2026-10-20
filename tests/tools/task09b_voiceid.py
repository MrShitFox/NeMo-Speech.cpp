"""Shared loading, pooling, normalization, and score metrics for Task 09B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_data(manifest_path: Path, pooled_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = {item["job_id"]: item for item in manifest["assets"]}
    packed = np.load(pooled_path, allow_pickle=False)
    pooled = {name: packed[name] for name in packed.files}
    job_ids = pooled["job_ids"].tolist()
    if len(job_ids) != len(set(job_ids)) or set(job_ids) != set(assets):
        raise ValueError("pooled feature IDs do not match the split asset catalog")
    return manifest, assets, pooled


class FeatureData:
    def __init__(self, assets: dict[str, dict[str, Any]], pooled: dict[str, np.ndarray], representation: str):
        self.assets = assets
        self.pooled = pooled
        self.representation = representation
        self.index = {job_id: index for index, job_id in enumerate(pooled["job_ids"].tolist())}
        self.mean = pooled[f"{representation}__mean"]
        self.std = pooled[f"{representation}__std"]
        self.frame_l2 = pooled[f"{representation}__frame_l2_mean"]
        self.speech_frames = pooled[f"{representation}__speech_frames"]
        self.duration = pooled[f"{representation}__speech_duration_sec"]
        self.frame_count = pooled[f"{representation}__frame_count"]

    @property
    def dim(self) -> int:
        return int(self.mean.shape[1])

    def indices(self, job_ids: Iterable[str]) -> list[int]:
        return [self.index[job_id] for job_id in job_ids]

    def speech_jobs(self, job_ids: Iterable[str]) -> list[str]:
        return [job_id for job_id in job_ids if self.speech_frames[self.index[job_id]] > 0]

    def pool(self, job_ids: Iterable[str], budget_sec: float | None = None) -> dict[str, Any]:
        """Pool complete VAD speech feature frames; the final utterance can overshoot by < one clip."""
        selected: list[int] = []
        measured = 0.0
        frame_total = 0
        for job_id in job_ids:
            index = self.index[job_id]
            n_frames = int(self.speech_frames[index])
            duration = float(self.duration[index])
            if n_frames <= 0 or duration <= 0:
                continue
            selected.append(index)
            frame_total += n_frames
            measured += duration
            if budget_sec is not None and measured >= budget_sec:
                break
        if not selected:
            return {"mean": np.zeros(self.dim, np.float32), "std": np.zeros(self.dim, np.float32),
                    "frame_l2_mean": np.zeros(self.dim, np.float32), "speech_duration_sec": 0.0,
                    "speech_frames": 0, "selected_job_indices": []}
        weights = self.speech_frames[selected].astype(np.float64)
        total = float(weights.sum())
        means = self.mean[selected].astype(np.float64)
        variances = self.std[selected].astype(np.float64) ** 2
        mean = (means * weights[:, None]).sum(axis=0) / total
        second_moment = ((variances + means ** 2) * weights[:, None]).sum(axis=0) / total
        std = np.sqrt(np.maximum(second_moment - mean ** 2, 0.0))
        frame_l2_mean = (self.frame_l2[selected].astype(np.float64) * weights[:, None]).sum(axis=0) / total
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
                "frame_l2_mean": frame_l2_mean.astype(np.float32),
                "speech_duration_sec": measured, "speech_frames": frame_total,
                "selected_job_indices": selected}

    def utterance_stats(self, job_ids: Iterable[str], budget_sec: float | None = None) -> list[dict[str, Any]]:
        selected = []
        measured = 0.0
        for job_id in job_ids:
            index = self.index[job_id]
            if self.speech_frames[index] <= 0:
                continue
            selected.append({"mean": self.mean[index], "std": self.std[index],
                             "frame_l2_mean": self.frame_l2[index],
                             "speech_duration_sec": float(self.duration[index]),
                             "speech_frames": int(self.speech_frames[index]), "job_id": job_id})
            measured += float(self.duration[index])
            if budget_sec is not None and measured >= budget_sec:
                break
        return selected


def concat_pool(pool: dict[str, Any]) -> np.ndarray:
    return np.concatenate((pool["mean"], pool["std"])).astype(np.float32, copy=False)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def unit(x: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(den, 1e-12)


def rank_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if not genuine.size or not impostor.size:
        return float("nan")
    # AUC = P(genuine > impostor) with half credit for exact ties.
    sorted_imp = np.sort(impostor)
    left = np.searchsorted(sorted_imp, genuine, side="left")
    right = np.searchsorted(sorted_imp, genuine, side="right")
    return float(np.sum(left + 0.5 * (right - left)) / (genuine.size * impostor.size))


def eer_threshold(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if not genuine.size or not impostor.size:
        return float("nan"), float("nan")
    scores = np.unique(np.concatenate((genuine, impostor)))
    best = (float("inf"), float("nan"), float("nan"))
    for threshold in scores:
        far = float(np.mean(impostor >= threshold))
        frr = float(np.mean(genuine < threshold))
        diff = abs(far - frr)
        if diff < best[0]:
            best = (diff, float(threshold), (far + frr) / 2.0)
    return best[1], best[2]


def far_target_threshold(impostor: np.ndarray, far_target: float = 0.05) -> float:
    values = np.asarray(impostor, dtype=np.float64)
    if not values.size:
        return float("inf")
    # Choose the most permissive observed threshold whose empirical FAR is <= target.
    candidates = np.unique(values)
    eligible = [float(threshold) for threshold in candidates if np.mean(values >= threshold) <= far_target]
    return min(eligible) if eligible else float(np.nextafter(values.max(), np.inf))


def open_set_calibration_scores(data: FeatureData, people: dict[str, Any], registry_order: list[str],
                                embed_one, budget_sec: float = 10.0) -> tuple[list[float], list[float], dict[str, Any]]:
    """Leave-one-identity-out calibration maxima over the other 14 calibration people."""
    prototypes: dict[str, np.ndarray] = {}
    for speaker, person in people.items():
        pooled = data.pool(person["enrollment_job_ids"], budget_sec)
        if pooled["speech_duration_sec"] >= budget_sec:
            prototypes[speaker] = embed_one(pooled)
    unknown_maxima = []
    genuine_scores = []
    skipped = 0
    speakers = [speaker for speaker in registry_order if speaker in people and speaker in prototypes]
    for speaker in speakers:
        person = people[speaker]
        query = data.pool(person["query_job_ids"], 3.0)
        if query["speech_duration_sec"] < 1.0:
            skipped += 1
            continue
        query_vec = embed_one(query)
        genuine_scores.append(cosine(query_vec, prototypes[speaker]))
        other = [cosine(query_vec, prototype) for identity, prototype in prototypes.items() if identity != speaker]
        unknown_maxima.append(max(other, default=float("nan")))
    details = {"calibration_speakers": len(people), "prototypes": len(prototypes),
               "leave_one_out_unknown_speaker_trials": len(unknown_maxima),
               "known_speaker_trials": len(genuine_scores), "no_speech_skips": skipped,
               "max_registry_size": max(0, len(prototypes) - 1)}
    return genuine_scores, unknown_maxima, details


def speaker_cluster_bootstrap(rows: list[dict[str, Any]], statistic, n_bootstrap: int = 2000,
                              seed: int = 20261029) -> dict[str, float]:
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_speaker.setdefault(row["speaker_id"], []).append(row)
    speakers = sorted(by_speaker)
    if not speakers:
        return {"lower_95": float("nan"), "upper_95": float("nan"), "bootstrap_replicates": 0}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        sample = [row for speaker in sampled for row in by_speaker[str(speaker)]]
        value = statistic(sample)
        if np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return {"lower_95": float("nan"), "upper_95": float("nan"), "bootstrap_replicates": 0}
    return {"lower_95": float(np.quantile(estimates, 0.025)),
            "upper_95": float(np.quantile(estimates, 0.975)),
            "bootstrap_replicates": len(estimates)}
