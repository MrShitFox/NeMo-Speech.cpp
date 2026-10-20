#!/usr/bin/env python3
"""Train compact heads on cached, frozen Nemotron features (research only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from task09b_voiceid import FeatureData, concat_pool, eer_threshold, load_data, rank_auc, unit


REPRESENTATIONS = ["pre_encoder", "encoder.layer.7", "encoder_output", "encoder_proj"]
OBJECTIVES = ["supcon_linear", "speaker_ce_mlp"]
MAX_EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = {"supcon_linear": 0.001, "speaker_ce_mlp": 0.0008}
TEMPERATURE = 0.1


class LinearEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=-1, eps=1e-12)


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = F.gelu(self.linear1(x))
        return F.normalize(self.linear2(x), dim=-1, eps=1e-12)


class SpeakerClassifier(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.embedding = MLPEmbedding(input_dim, 128)
        self.classifier = nn.Linear(128, classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embedding(x)
        return embedding, self.classifier(embedding)


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_normalizer(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = rows.astype(np.float64).mean(axis=0).astype(np.float32)
    scale = rows.astype(np.float64).std(axis=0).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-4))
    return center, scale


def standardized(pool: dict[str, Any], center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((concat_pool(pool) - center) / scale, -8.0, 8.0).astype(np.float32)


def open_set_validation_eer(model: nn.Module, feature_data: FeatureData, people: dict[str, Any],
                            center: np.ndarray, scale: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    prototypes: dict[str, np.ndarray] = {}
    queries: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for speaker, person in people.items():
            enrollment = feature_data.pool(person["enrollment_job_ids"], 10.0)
            query = feature_data.pool(person["query_job_ids"], 3.0)
            if enrollment["speech_duration_sec"] < 10.0 or query["speech_duration_sec"] < 3.0:
                continue
            x_e = torch.from_numpy(standardized(enrollment, center, scale)[None, :]).to(device)
            x_q = torch.from_numpy(standardized(query, center, scale)[None, :]).to(device)
            if isinstance(model, SpeakerClassifier):
                e = model.embedding(x_e)
                q = model.embedding(x_q)
            else:
                e, q = model(x_e), model(x_q)
            prototypes[speaker] = as_numpy(e[0])
            queries[speaker] = as_numpy(q[0])
    genuine: list[float] = []
    impostor: list[float] = []
    for speaker, query in queries.items():
        for identity, prototype in prototypes.items():
            value = float(np.dot(query, prototype))
            (genuine if speaker == identity else impostor).append(value)
    threshold, eer = eer_threshold(np.asarray(genuine), np.asarray(impostor))
    return {"genuine_scores": len(genuine), "impostor_scores": len(impostor),
            "eer": eer, "eer_threshold": threshold, "auc": rank_auc(np.asarray(genuine), np.asarray(impostor)),
            "validation_speakers": len(queries)}


def supervised_contrastive_loss(embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = embedding @ embedding.T / TEMPERATURE
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    mask = labels[:, None].eq(labels[None, :]).to(embedding.dtype)
    eye = torch.eye(mask.shape[0], device=mask.device, dtype=mask.dtype)
    mask = mask - eye
    exp_logits = torch.exp(logits) * (1.0 - eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives = mask.sum(dim=1).clamp_min(1.0)
    return -((mask * log_prob).sum(dim=1) / positives).mean()


def balanced_batch(rows_by_class: dict[int, list[int]], rng: random.Random, device: torch.device,
                   speakers_per_batch: int = 12, utterances_per_speaker: int = 4) -> tuple[list[int], list[int]]:
    classes = list(rows_by_class)
    chosen_classes = rng.sample(classes, min(speakers_per_batch, len(classes)))
    indices, labels = [], []
    for label in chosen_classes:
        rows = rows_by_class[label]
        sampled = rng.sample(rows, utterances_per_speaker) if len(rows) >= utterances_per_speaker else [rng.choice(rows) for _ in range(utterances_per_speaker)]
        indices.extend(sampled)
        labels.extend([label] * len(sampled))
    return indices, labels


def save_export(path: Path, model: nn.Module, center: np.ndarray, scale: np.ndarray,
                metadata: dict[str, Any]) -> tuple[int, int]:
    if isinstance(model, SpeakerClassifier):
        state = model.embedding.state_dict()
    else:
        state = model.state_dict()
    export = {f"weight_{name.replace('.', '_')}": as_numpy(value).astype(np.float32)
              for name, value in state.items()}
    export["normalization_center"] = center.astype(np.float32)
    export["normalization_scale"] = scale.astype(np.float32)
    np.savez(path, **export)
    inference_params = sum(int(value.numel()) for name, value in state.items())
    classifier_params = int(model.classifier.weight.numel() + model.classifier.bias.numel()) if isinstance(model, SpeakerClassifier) else 0
    return inference_params, classifier_params


def measure_inference(model: nn.Module, input_tensor: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    one = input_tensor[:1].to(device)
    batch = input_tensor[:min(64, len(input_tensor))].to(device)
    def forward(value):
        return model.embedding(value) if isinstance(model, SpeakerClassifier) else model(value)
    with torch.no_grad():
        for _ in range(10):
            forward(one)
            forward(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        single_runs, batch_runs = 300, 150
        begin = time.perf_counter()
        for _ in range(single_runs):
            forward(one)
        single_elapsed = time.perf_counter() - begin
        begin = time.perf_counter()
        for _ in range(batch_runs):
            forward(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_elapsed = time.perf_counter() - begin
    return {"device": str(device), "single_input_latency_us": single_elapsed * 1e6 / single_runs,
            "batch_size": int(len(batch)), "batch_throughput_embeddings_per_sec": batch_runs * len(batch) / batch_elapsed,
            "measurement_scope": "head-only forward after pooled feature vector is available; includes framework dispatch and normalization"}


def fit_lda(feature_data: FeatureData, train_ids: dict[str, list[str]],
            speaker_labels: dict[str, int]) -> dict[str, Any]:
    rows, labels = [], []
    for speaker, job_ids in train_ids.items():
        for job_id in job_ids:
            index = feature_data.index[job_id]
            if feature_data.speech_frames[index] <= 0:
                continue
            rows.append(feature_data.mean[index])
            labels.append(speaker_labels[speaker])
    matrix = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    center, scale = fit_normalizer(matrix)
    x = np.clip((matrix - center) / scale, -8, 8).astype(np.float64)
    overall = x.mean(axis=0)
    within = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    between = np.zeros_like(within)
    for label in np.unique(y):
        part = x[y == label]
        mean = part.mean(axis=0)
        centered = part - mean
        within += centered.T @ centered
        delta = (mean - overall)[:, None]
        between += part.shape[0] * (delta @ delta.T)
    within /= max(1, x.shape[0] - len(np.unique(y)))
    between /= max(1, x.shape[0])
    ridge = max(float(np.trace(within) / within.shape[0]) * 0.05, 1e-4)
    regularized = within + ridge * np.eye(within.shape[0])
    chol = np.linalg.cholesky(regularized)
    whitened_between = np.linalg.solve(chol, between)
    whitened_between = np.linalg.solve(chol, whitened_between.T).T
    eigval, eigvec = np.linalg.eigh((whitened_between + whitened_between.T) * 0.5)
    output_dim = min(49, len(np.unique(y)) - 1)
    selected = np.argsort(eigval)[::-1][:output_dim]
    projection = np.linalg.solve(chol.T, eigvec[:, selected])
    return {"center": center, "scale": scale, "projection": projection.astype(np.float32),
            "regularization": ridge, "output_dim": output_dim, "train_utterances": int(x.shape[0]),
            "train_speakers": int(len(np.unique(y))), "eigenvalues": eigval[selected].astype(np.float32)}


def experiment_key(fold_id: str, representation: str, objective: str) -> str:
    safe_rep = representation.replace(".", "_")
    return f"{fold_id}_{safe_rep}_{objective}"


def train_one(fold: dict[str, Any], representation: str, objective: str,
              feature_data: FeatureData, run_dir: Path, device: torch.device,
              resume: bool, randomized_labels: bool = False) -> dict[str, Any]:
    fold_id = fold["fold_id"]
    suffix = "_random_labels" if randomized_labels else ""
    key = experiment_key(fold_id, representation, objective) + suffix
    best_path = run_dir / f"{key}.pt"
    last_path = run_dir / f"{key}.last.pt"
    export_path = run_dir / f"{key}.npz"
    if resume and best_path.exists() and export_path.exists():
        result_path = run_dir / f"{key}.json"
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
    train_ids = fold["train_utterance_job_ids"]
    train_speakers = sorted(train_ids)
    speaker_to_label = {speaker: index for index, speaker in enumerate(train_speakers)}
    x_rows, y_rows, row_speakers = [], [], []
    rows_by_actual: dict[int, list[int]] = {}
    for speaker in train_speakers:
        label = speaker_to_label[speaker]
        for job_id in train_ids[speaker]:
            index = feature_data.index[job_id]
            if feature_data.speech_frames[index] <= 0:
                continue
            x_rows.append(np.concatenate((feature_data.mean[index], feature_data.std[index])))
            y_rows.append(label)
            row_speakers.append(speaker)
    raw_x = np.asarray(x_rows, dtype=np.float32)
    labels = np.asarray(y_rows, dtype=np.int64)
    if len(raw_x) == 0 or len(set(row_speakers)) != 50:
        raise RuntimeError(f"insufficient voiced training rows in {fold_id}/{representation}")
    center, scale = fit_normalizer(raw_x)
    x = np.clip((raw_x - center) / scale, -8.0, 8.0).astype(np.float32)
    original_by_class: dict[int, list[int]] = {i: [] for i in range(len(train_speakers))}
    for index, label in enumerate(labels):
        original_by_class[int(label)].append(index)
    if randomized_labels:
        rng_label = np.random.default_rng(9100 + int(fold["speaker_seed"]) % 100000)
        shuffled = rng_label.permutation(labels)
        labels = shuffled
        rows_by_class = {i: [] for i in range(len(train_speakers))}
        for index, label in enumerate(labels):
            rows_by_class[int(label)].append(index)
    else:
        rows_by_class = original_by_class
    if any(len(rows_by_class[label]) < 4 for label in rows_by_class):
        raise RuntimeError("balanced training needs four utterances per speaker class")

    model_seed = (5100 + int(fold["speaker_seed"]) + REPRESENTATIONS.index(representation) * 101 +
                  OBJECTIVES.index(objective) + (293 if randomized_labels else 0))
    torch.manual_seed(model_seed)
    model: nn.Module
    input_dim = x.shape[1]
    if objective == "supcon_linear":
        model = LinearEmbedding(input_dim).to(device)
    elif objective == "speaker_ce_mlp":
        model = SpeakerClassifier(input_dim, len(train_speakers)).to(device)
    else:
        raise ValueError(objective)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE[objective], weight_decay=1e-4)
    start_epoch, best_eer, best_state, patience_used, history = 0, float("inf"), None, 0, []
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("input_dim") != input_dim or checkpoint.get("objective") != objective or checkpoint.get("representation") != representation:
            raise ValueError(f"resume checkpoint is incompatible: {last_path}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        best_eer = float(checkpoint["best_eer"])
        best_state = checkpoint["best_state"]
        patience_used = int(checkpoint["patience_used"])
        history = checkpoint["history"]
    selection = fold["selection_people"]
    begin = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampler_rng = random.Random(12000 + int(fold["speaker_seed"]) + REPRESENTATIONS.index(representation) * 101 + OBJECTIVES.index(objective) + (293 if randomized_labels else 0))
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(labels.astype(np.int64))
    steps_per_epoch = 24
    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            batch_indices, batch_labels = balanced_batch(rows_by_class, sampler_rng, device)
            xb = x_tensor[batch_indices].to(device)
            yb = torch.tensor(batch_labels, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if objective == "supcon_linear":
                embedding = model(xb)
                loss = supervised_contrastive_loss(embedding, yb)
            else:
                embedding, logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {key} epoch {epoch + 1}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite gradient norm in {key} epoch {epoch + 1}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = open_set_validation_eer(model, feature_data, selection, center, scale, device)
        epoch_record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **validation}
        history.append(epoch_record)
        score = validation["eer"]
        if np.isfinite(score) and score < best_eer - 1e-5:
            best_eer = score
            patience_used = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save({"model_state": best_state, "objective": objective, "representation": representation,
                        "input_dim": input_dim, "epoch": epoch + 1, "best_validation_eer": best_eer,
                        "normalization_center": center, "normalization_scale": scale,
                        "training_speaker_ids": train_speakers, "seed": int(fold["speaker_seed"])}, best_path)
        else:
            patience_used += 1
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch + 1, "best_eer": best_eer, "best_state": best_state,
                    "patience_used": patience_used, "history": history, "input_dim": input_dim,
                    "objective": objective, "representation": representation,
                    "normalization_center": center, "normalization_scale": scale}, last_path)
        if patience_used >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"no valid validation checkpoint for {key}")
    model.load_state_dict(best_state)
    inference_parameters, classifier_parameters = save_export(export_path, model, center, scale,
                                                              {"fold": fold_id, "representation": representation, "objective": objective})
    training_seconds = time.perf_counter() - begin
    inference_cost = measure_inference(model, x_tensor[:min(64, len(x_tensor))], device)
    peak_memory_bytes = (int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda"
                         else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
    result = {
        "model_key": key, "fold_id": fold_id, "representation": representation, "objective": objective,
        "negative_control_randomized_labels": randomized_labels,
        "architecture": "linear 1024->128" if objective == "supcon_linear" else "LayerNorm D -> Linear 256 -> GELU -> Linear 128; training-only 128->50 classifier",
        "input_dimension": int(input_dim), "embedding_dimension": 128,
        "inference_parameter_count": inference_parameters,
        "training_only_classifier_parameter_count": classifier_parameters,
        "head_float32_bytes": inference_parameters * 4,
        "checkpoint_bytes": best_path.stat().st_size,
        "checkpoint": best_path.name, "portable_export": export_path.name,
        "trainable_parameter_count": inference_parameters + classifier_parameters,
        "train_speakers": len(train_speakers), "train_utterances": len(raw_x),
        "validation_speakers": len(selection), "selected_epoch": int(torch.load(best_path, map_location="cpu", weights_only=False)["epoch"]),
        "best_validation_eer": float(best_eer), "training_duration_sec": training_seconds,
        "inference_cost": inference_cost,
        "training_device": str(device), "peak_training_memory_bytes": peak_memory_bytes,
        "peak_memory_source": "torch.cuda.max_memory_allocated" if device.type == "cuda" else "process ru_maxrss high-water mark; cumulative across experiments in this process",
        "peak_gpu_memory_bytes": peak_memory_bytes if device.type == "cuda" else None,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE[objective], "weight_decay": 1e-4,
        "batch_protocol": "12 speakers x 4 distinct mic1 utterances; balanced speaker batches",
        "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
        "history": history,
    }
    (run_dir / f"{key}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def save_lda(fold: dict[str, Any], representation: str, feature_data: FeatureData,
             run_dir: Path) -> dict[str, Any]:
    train_ids = fold["train_utterance_job_ids"]
    labels = {speaker: index for index, speaker in enumerate(sorted(train_ids))}
    lda = fit_lda(feature_data, train_ids, labels)
    path = run_dir / f"{fold['fold_id']}_{representation.replace('.', '_')}_lda.npz"
    np.savez(path, center=lda["center"], scale=lda["scale"], projection=lda["projection"], eigenvalues=lda["eigenvalues"])
    return {"fold_id": fold["fold_id"], "representation": representation,
            "output_dimension": lda["output_dim"], "training_speakers": lda["train_speakers"],
            "training_utterances": lda["train_utterances"], "regularization": lda["regularization"],
            "projection_parameters": int(lda["projection"].size), "float32_bytes": int(lda["projection"].nbytes),
            "path": path.name, "training_labels_used_only": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/manifests/task09b_vctk_splits.json"))
    parser.add_argument("--pooled-features", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--folds", default="all")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build has no CUDA support")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    representations = [item for item in args.representations.split(",") if item]
    objectives = [item for item in args.objectives.split(",") if item]
    manifest, assets, pooled = load_data(args.manifest, args.pooled_features)
    feature_validation_path = args.pooled_features.parent / "feature_validation.json"
    feature_validation = json.loads(feature_validation_path.read_text(encoding="utf-8"))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    selected_folds = manifest["folds"] if args.folds == "all" else [fold for fold in manifest["folds"] if fold["fold_id"] in args.folds.split(",")]
    results = []
    lda_results = []
    for fold in selected_folds:
        candidate_results = []
        for representation in representations:
            feature_data = FeatureData(assets, pooled, representation)
            lda_results.append(save_lda(fold, representation, feature_data, args.run_dir))
            for objective in objectives:
                result = train_one(fold, representation, objective, feature_data, args.run_dir, device, args.resume)
                candidate_results.append(result)
                results.append(result)
                print(json.dumps({"fold": fold["fold_id"], "model": result["model_key"],
                                  "epoch": result["selected_epoch"], "validation_eer": result["best_validation_eer"],
                                  "duration_sec": round(result["training_duration_sec"], 2), "device": str(device)}),
                      flush=True)
        chosen = min(candidate_results, key=lambda row: (row["best_validation_eer"], row["model_key"]))
        fold["selected_model_key"] = chosen["model_key"]
        if args.negative_control:
            feature_data = FeatureData(assets, pooled, chosen["representation"])
            control = train_one(fold, chosen["representation"], "supcon_linear", feature_data,
                                args.run_dir, device, args.resume, randomized_labels=True)
            results.append(control)
            fold["negative_control_model_key"] = control["model_key"]
            print(json.dumps({"fold": fold["fold_id"], "negative_control": control["model_key"],
                              "validation_eer": control["best_validation_eer"]}), flush=True)
    method_metadata = {
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "device": str(device), "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_num_threads": torch.get_num_threads(), "python_version": sys.version.split()[0],
        "seed_protocol": "speaker fold seed plus fixed representation/objective offsets; per-batch sampler seeded; deterministic torch algorithms",
        "original_nemotron_weights_frozen": True,
        "training_inputs": "cached pooled frozen Nemotron frame features; no audio encoder or pretrained speaker model",
        "selected_representations": representations, "selected_objectives": objectives,
        "input_artifacts": {
            "split_manifest": args.manifest.name,
            "split_manifest_sha256": sha256_file(args.manifest),
            "pooled_features": args.pooled_features.name,
            "pooled_features_sha256": sha256_file(args.pooled_features),
            "base_model_fingerprint": feature_validation["model_fingerprint"],
            "feature_backend": feature_validation["inference_backend"],
            "feature_device": feature_validation["device_name"],
        },
        "projection_dimensions_tested": [128], "normalization": "per-input-dimension center/scale fit on head-training utterances only; scale floor 1e-4; z-score clamp +/-8",
        "model_selection": "minimum pairwise verification EER on model-selection speakers at 10s registration and 3s query; threshold is diagnostic only and is refit from calibration speakers",
        "lda_baseline": "shrinkage LDA on training-speaker utterance means only; calibration/evaluation data excluded",
    }
    output = {"schema": "nemo-speech.voiceid-task09b-training.v1", "configuration": method_metadata,
              "models": results, "lda_baselines": lda_results,
              "selected_models_by_fold": {fold["fold_id"]: fold.get("selected_model_key") for fold in selected_folds},
              "negative_controls_by_fold": {fold["fold_id"]: fold.get("negative_control_model_key") for fold in selected_folds}}
    results_path = Path("tests/results/voiceid_task09b_training.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Record model selection in an external run manifest instead of modifying the locked speaker splits.
    (args.run_dir / "training_manifest.json").write_text(json.dumps({"selected_models_by_fold": output["selected_models_by_fold"],
        "negative_controls_by_fold": output["negative_controls_by_fold"], "source_split_manifest": str(args.manifest)},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(results_path), "models": len(results), "selected_models": output["selected_models_by_fold"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"train_voiceid_head: {exc}", file=sys.stderr)
        raise
