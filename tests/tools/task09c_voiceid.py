"""Bounded, non-neural pooling and training-only transforms for Task 09C."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from extract_voiceid_features import read_feature
from task09b_voiceid import FeatureData, unit


POOLING_CANDIDATES = ("mean", "mean_std", "frame_l2_mean", "median", "trimmed_mean")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    return float(np.dot(a64, b64) / denominator) if denominator else float("nan")


def frame_statistics(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Return simple frame pooling statistics for already speech-masked frames."""
    x = np.asarray(frames, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("frames must be a [frame, feature] matrix")
    if x.shape[0] == 0:
        zero = np.zeros(x.shape[1], dtype=np.float32)
        return {"mean": zero.copy(), "std": zero.copy(), "frame_l2_mean": zero.copy(),
                "median": zero.copy(), "trimmed_mean": zero.copy()}
    normalized = unit(x)
    ordered = np.sort(x, axis=0)
    trim = min(int(np.floor(0.10 * len(x))), max(0, (len(x) - 1) // 2))
    trimmed = ordered[trim:len(x) - trim] if trim else ordered
    return {
        "mean": x.mean(axis=0, dtype=np.float64).astype(np.float32),
        "std": x.std(axis=0, dtype=np.float64).astype(np.float32),
        "frame_l2_mean": normalized.mean(axis=0, dtype=np.float64).astype(np.float32),
        "median": np.median(x, axis=0).astype(np.float32),
        "trimmed_mean": trimmed.mean(axis=0, dtype=np.float64).astype(np.float32),
    }


class RawFeatureStore:
    """Read NVFTR records lazily with a bounded in-memory cache."""

    def __init__(self, feature_dir: Path | Iterable[Path], representation: str, max_cached_records: int = 96):
        if isinstance(feature_dir, (str, Path)):
            self.feature_dirs = [Path(feature_dir)]
        else:
            self.feature_dirs = [Path(path) for path in feature_dir]
        self.representation = representation
        self.max_cached_records = max(1, int(max_cached_records))
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self.dimension = 0

    def frames(self, job_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if job_id in self._cache:
            self._cache.move_to_end(job_id)
            return self._cache[job_id]
        name = f"{job_id}.{self.representation}.vfeat"
        path = next((root / name for root in self.feature_dirs if (root / name).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"no feature record {name} under {self.feature_dirs}")
        _, _, speech_bits, values, _ = read_feature(path)
        self.dimension = int(values.shape[1])
        counts = np.fromiter((int(value).bit_count() for value in speech_bits), dtype=np.uint8, count=len(speech_bits))
        selected = counts >= 4
        record = (values[selected], counts[selected], selected)
        self._cache[job_id] = record
        while len(self._cache) > self.max_cached_records:
            self._cache.popitem(last=False)
        return record

    def pool(self, job_ids: Iterable[str], budget_sec: float | None = None) -> dict[str, Any]:
        all_frames: list[np.ndarray] = []
        utterance_frames: list[np.ndarray] = []
        selected_job_ids: list[str] = []
        duration_sec = 0.0
        frame_count = 0
        for job_id in job_ids:
            frames, active_bits, _ = self.frames(job_id)
            if not len(frames):
                continue
            all_frames.append(frames)
            utterance_frames.append(frames)
            selected_job_ids.append(job_id)
            duration_sec += float(active_bits.sum()) * 0.01
            frame_count += len(frames)
            if budget_sec is not None and duration_sec >= budget_sec:
                break
        dim = all_frames[0].shape[1] if all_frames else self.dimension
        if not all_frames:
            stats = frame_statistics(np.zeros((0, dim), dtype=np.float32))
        else:
            stats = frame_statistics(np.concatenate(all_frames, axis=0))
        return {**stats, "speech_duration_sec": duration_sec, "speech_frames": frame_count,
                "selected_job_ids": selected_job_ids, "utterance_frames": utterance_frames}


def embed_pool(pool: dict[str, Any], pooling: str) -> np.ndarray:
    if pooling == "mean":
        return unit(np.asarray(pool["mean"]))
    if pooling == "mean_std":
        return unit(np.concatenate((pool["mean"], pool["std"])).astype(np.float32, copy=False))
    if pooling in ("frame_l2_mean", "median", "trimmed_mean"):
        return unit(np.asarray(pool[pooling]))
    raise ValueError(f"unknown pooling method: {pooling}")


def embed_utterance_normalized_mean(store: RawFeatureStore, job_ids: Iterable[str],
                                   budget_sec: float | None = None) -> tuple[np.ndarray | None, float]:
    pool = store.pool(job_ids, budget_sec)
    selected = pool["selected_job_ids"]
    if not selected:
        return None, pool["speech_duration_sec"]
    utterance_vectors = []
    for frames in pool["utterance_frames"]:
        if len(frames):
            utterance_vectors.append(unit(frame_statistics(frames)["mean"]))
    if not utterance_vectors:
        return None, pool["speech_duration_sec"]
    return unit(np.mean(np.stack(utterance_vectors), axis=0)), pool["speech_duration_sec"]


def fit_diagonal_standardizer(training_vectors: np.ndarray, floor: float = 1e-4) -> dict[str, Any]:
    x = np.asarray(training_vectors, dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError("training vectors must be a non-empty matrix")
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), floor)
    return {"kind": "diagonal_standardization", "center": center.astype(np.float32),
            "scale": scale.astype(np.float32), "floor": float(floor)}


def fit_regularized_whitener(training_vectors: np.ndarray, shrinkage: float = 0.10,
                             ridge_fraction: float = 1e-3) -> dict[str, Any]:
    """Fit a centered, shrinkage-regularized full-covariance whitening map."""
    x = np.asarray(training_vectors, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("whitening requires at least two training vectors")
    if not 0.0 <= shrinkage <= 1.0 or ridge_fraction <= 0.0:
        raise ValueError("invalid whitening regularization")
    center = x.mean(axis=0)
    centered = x - center
    covariance = centered.T @ centered / max(1, len(x) - 1)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    ridge = max(float(np.mean(np.diag(covariance))) * ridge_fraction, 1e-8)
    covariance.flat[::len(covariance) + 1] += ridge
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * ridge_fraction, 1e-8)
    projection = (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))) @ eigenvectors.T
    return {"kind": "regularized_whitening", "center": center.astype(np.float32),
            "projection": projection.astype(np.float32), "shrinkage": float(shrinkage),
            "ridge_fraction": float(ridge_fraction), "ridge": ridge, "eigenvalue_floor": floor}


def apply_population_transform(vectors: np.ndarray, transform: dict[str, Any], clip: float = 8.0) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float32)
    centered = x - transform["center"]
    if transform["kind"] == "diagonal_standardization":
        result = centered / transform["scale"]
    elif transform["kind"] == "regularized_whitening":
        result = centered @ transform["projection"]
    else:
        raise ValueError(f"unknown population transform: {transform['kind']}")
    return unit(np.clip(result, -clip, clip))


def progressive_prefix(frames: np.ndarray, speech_seconds_per_frame: np.ndarray,
                       budget_sec: float) -> tuple[np.ndarray, float, int]:
    """Select a chronological speech prefix up to a budget, allowing one-frame overshoot."""
    x = np.asarray(frames, dtype=np.float32)
    weights = np.asarray(speech_seconds_per_frame, dtype=np.float64)
    if x.ndim != 2 or weights.ndim != 1 or len(x) != len(weights):
        raise ValueError("frames and speech weights must have matching frame axes")
    if budget_sec <= 0.0 or not len(x):
        return np.zeros(x.shape[1] if x.ndim == 2 else 0, dtype=np.float32), 0.0, 0
    used: list[np.ndarray] = []
    duration = 0.0
    for frame, weight in zip(x, weights):
        if weight <= 0.0:
            continue
        used.append(frame)
        duration += float(weight)
        if duration >= budget_sec:
            break
    if not used:
        return np.zeros(x.shape[1], dtype=np.float32), 0.0, 0
    return np.mean(np.stack(used), axis=0, dtype=np.float64).astype(np.float32), duration, len(used)
