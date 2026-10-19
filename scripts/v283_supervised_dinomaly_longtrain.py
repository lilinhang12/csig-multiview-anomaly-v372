from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path.home() / "csig"
DEFAULT_PROXY_INDEX = (
    ROOT
    / "outputs"
    / "v280_realiad_supervised_auto"
    / "prepared"
    / "proxy_index.jsonl"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "v283_supervised_dinomaly_longtrain"
DEFAULT_WEIGHT_DIR = ROOT / "Dinomaly2_dinov3" / "backbones" / "weights"
DEFAULT_TRAIN_ZIP = ROOT / "Train.zip"
DEFAULT_SOURCE_STEP = 100000
PATCH_GRID = 28


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checkpoint_steps(value: str, target_steps: int) -> tuple[int, ...]:
    try:
        steps = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"checkpoint steps must be integers: {value}") from error
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("checkpoint steps must be positive and non-empty")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("checkpoint steps must be unique and strictly increasing")
    if steps[-1] != int(target_steps):
        raise ValueError("checkpoint steps must end at target_steps")
    return steps


def _import_adapter() -> Any:
    try:
        from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    except ImportError:
        import v182_dinomaly2_dinov3_csig_official as adapter
    return adapter


def _source_checkpoint(source_run_root: Path, source_step: int) -> Path:
    return Path(source_run_root) / "checkpoints" / f"state_i{int(source_step):06d}.pt"


def _load_source_model(args: argparse.Namespace, device: Any) -> tuple[Any, dict[str, Any], dict[str, int]]:
    import torch

    try:
        from scripts.v276_i448_state_infer import extract_model_state
    except ImportError:
        from v276_i448_state_infer import extract_model_state

    adapter = _import_adapter()
    deps = adapter._import_official_modules()
    checkpoint = _source_checkpoint(args.source_run_root, args.source_step)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    model_state = extract_model_state(
        payload,
        expected_step=int(args.source_step),
        expected_input_size=int(args.input_size),
        expected_dropout=float(args.source_dropout),
    )
    model, _trainable = adapter._build_model(
        deps=deps,
        backbone=args.backbone,
        dropout=float(args.source_dropout),
        reduce_dim=int(args.reduce_dim),
        context_aware_recenter=True,
        mask_neighbor_size=0,
        weights_dir=args.weights_dir,
    )
    model.load_state_dict(model_state)
    del payload, model_state
    counts = configure_trainable_parameters(model)
    return model.to(device), deps, counts


class RealIADSupervisedDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]], input_size: int = 448) -> None:
        self.rows = [dict(row) for row in rows]
        self.input_size = int(input_size)
        if not self.rows:
            raise ValueError("RealIADSupervisedDataset requires rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any, int, str, str]:
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional

        row = self.rows[int(index)]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
            image = functional.resize(
                image,
                [self.input_size, self.input_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            image_tensor = functional.to_tensor(image)
            image_tensor = functional.normalize(
                image_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        mask_path = row.get("mask_path")
        if mask_path:
            with Image.open(mask_path) as source_mask:
                mask_image = source_mask.convert("L")
                mask_image = functional.resize(
                    mask_image,
                    [self.input_size, self.input_size],
                    interpolation=InterpolationMode.NEAREST,
                )
                mask = (functional.to_tensor(mask_image) > 0.5).to(torch.float32)
        else:
            mask = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)
        return (
            image_tensor,
            mask,
            int(row.get("label", 0)),
            str(row["category"]),
            str(row.get("group", "")),
        )


class AnchorTeacherDataset:
    def __init__(self, base_dataset: Any, teacher_root: Path) -> None:
        self.base_dataset = base_dataset
        self.teacher_root = Path(teacher_root)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        image, category, sample, view = self.base_dataset[int(index)]
        teacher_path = self.teacher_root / str(category) / str(sample) / f"{view}.npy"
        if not teacher_path.is_file():
            raise FileNotFoundError(teacher_path)
        teacher = torch.from_numpy(np.load(teacher_path).astype(np.float32, copy=False)).unsqueeze(0)
        if teacher.shape != (1, PATCH_GRID, PATCH_GRID):
            raise ValueError(f"Invalid teacher map shape {tuple(teacher.shape)} for {teacher_path}")
        return image, teacher


def make_balanced_sampler(rows: Sequence[Mapping[str, Any]], seed: int) -> Any:
    import torch

    labels = [int(row.get("label", 0)) for row in rows]
    counts = {label: labels.count(label) for label in set(labels)}
    if set(counts) != {0, 1}:
        raise ValueError(f"Training rows require normal and anomaly labels, found {counts}")
    weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples=2 * max(counts.values()),
        replacement=True,
        generator=generator,
    )


def differentiable_anomaly_map(
    encoded: Sequence[Any], decoded: Sequence[Any], out_size: int
) -> Any:
    import torch
    import torch.nn.functional as functional

    if len(encoded) != len(decoded) or not encoded:
        raise ValueError("encoded and decoded feature lists must be non-empty and aligned")
    maps = []
    for source, reconstruction in zip(encoded, decoded):
        if source.shape != reconstruction.shape:
            raise ValueError(
                f"feature shape mismatch: encoded={tuple(source.shape)} decoded={tuple(reconstruction.shape)}"
            )
        layer = 1.0 - functional.cosine_similarity(
            source.detach(), reconstruction, dim=1
        )
        maps.append(
            functional.interpolate(
                layer.unsqueeze(1),
                size=(int(out_size), int(out_size)),
                mode="bilinear",
                align_corners=True,
            )
        )
    return torch.cat(maps, dim=1).mean(dim=1, keepdim=True)


def binary_focal_loss_with_logits(
    logits: Any, targets: Any, *, alpha: float = 0.75, gamma: float = 2.0
) -> Any:
    import torch
    import torch.nn.functional as functional

    targets = targets.to(dtype=logits.dtype)
    base = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    probability_true = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_weight = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
    return (alpha_weight * (1.0 - probability_true).pow(float(gamma)) * base).mean()


def soft_dice_loss(probabilities: Any, targets: Any, eps: float = 1e-6) -> Any:
    targets = targets.to(dtype=probabilities.dtype)
    reduce_dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
    score = (2.0 * intersection + float(eps)) / (denominator + float(eps))
    return (1.0 - score).mean()


def normal_false_positive_loss(
    probabilities: Any,
    targets: Any,
    *,
    mean_budget: float = 0.01,
    quantile_budget: float = 0.10,
    quantile: float = 0.995,
) -> Any:
    """Penalize anomaly response on normal images without weakening positive masks."""

    import torch
    import torch.nn.functional as functional

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"probability/target shapes must match, found {tuple(probabilities.shape)} "
            f"and {tuple(targets.shape)}"
        )
    if not 0.0 <= float(mean_budget) <= 1.0:
        raise ValueError(f"mean_budget must be in [0, 1], found {mean_budget}")
    if not 0.0 <= float(quantile_budget) <= 1.0:
        raise ValueError(f"quantile_budget must be in [0, 1], found {quantile_budget}")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError(f"quantile must be in (0, 1), found {quantile}")

    targets = targets.to(device=probabilities.device, dtype=probabilities.dtype)
    flat_prob = probabilities.flatten(start_dim=1)
    flat_targets = targets.flatten(start_dim=1)
    normal_rows = flat_targets.sum(dim=1) <= 0.0
    if not bool(normal_rows.any().item()):
        return probabilities.sum() * 0.0
    normal_values = flat_prob[normal_rows]
    mean_excess = functional.relu(normal_values.mean(dim=1) - float(mean_budget))
    tail = torch.quantile(normal_values, float(quantile), dim=1)
    tail_excess = functional.relu(tail - float(quantile_budget))
    return (mean_excess + 0.25 * tail_excess).mean()


def region_quantile_margin_loss(
    anomaly_map: Any, targets: Any, *, margin: float = 0.03
) -> Any:
    import torch
    import torch.nn.functional as functional

    targets = targets.to(dtype=torch.bool)
    losses = []
    for index in range(int(anomaly_map.shape[0])):
        values = anomaly_map[index, 0]
        positive = values[targets[index, 0]]
        negative = values[~targets[index, 0]]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        positive_floor = torch.quantile(positive, 0.25)
        negative_ceiling = torch.quantile(negative, 0.95)
        losses.append(
            functional.relu(float(margin) + negative_ceiling - positive_floor)
        )
    if not losses:
        return anomaly_map.sum() * 0.0
    return torch.stack(losses).mean()


def global_cosine_loss(encoded: Sequence[Any], decoded: Sequence[Any]) -> Any:
    import torch
    import torch.nn.functional as functional

    losses = []
    for source, reconstruction in zip(encoded, decoded):
        losses.append(
            (
                1.0
                - functional.cosine_similarity(
                    source.detach().reshape(source.shape[0], -1),
                    reconstruction.reshape(reconstruction.shape[0], -1),
                    dim=1,
                )
            ).mean()
        )
    if not losses:
        raise ValueError("feature lists cannot be empty")
    return torch.stack(losses).mean()


def compute_supervised_losses(
    encoded: Sequence[Any],
    decoded: Sequence[Any],
    masks: Any,
    *,
    out_size: int,
    normal_loss_weight: float = 0.25,
    normal_mean_budget: float = 0.01,
    normal_quantile_budget: float = 0.10,
) -> dict[str, Any]:
    import torch

    anomaly_map = differentiable_anomaly_map(encoded, decoded, out_size).float()
    masks = masks.to(device=anomaly_map.device, dtype=anomaly_map.dtype)
    if masks.shape != anomaly_map.shape:
        raise ValueError(
            f"mask shape {tuple(masks.shape)} does not match map {tuple(anomaly_map.shape)}"
        )
    logits = (anomaly_map - 0.10) / 0.025
    probability = torch.sigmoid(logits)
    official = global_cosine_loss(encoded, decoded)
    focal = binary_focal_loss_with_logits(logits, masks, alpha=0.75, gamma=2.0)
    dice = soft_dice_loss(probability, masks)
    rank = region_quantile_margin_loss(anomaly_map, masks, margin=0.03)
    normal_fp = normal_false_positive_loss(
        probability,
        masks,
        mean_budget=float(normal_mean_budget),
        quantile_budget=float(normal_quantile_budget),
    )
    total = official + focal + 0.75 * dice + 0.25 * rank + float(normal_loss_weight) * normal_fp
    return {
        "map": anomaly_map,
        "official": official,
        "focal": focal,
        "dice": dice,
        "rank": rank,
        "normal_fp": normal_fp,
        "total": total,
    }


def anchor_distillation_loss(current_map: Any, teacher_map: Any) -> Any:
    import torch.nn.functional as functional

    return functional.smooth_l1_loss(current_map, teacher_map.detach())


def configure_trainable_parameters(model: Any) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in ("bottleneck", "decoder"):
        module = getattr(model, module_name, None)
        if module is None:
            raise AttributeError(f"Dinomaly model is missing {module_name}")
        for parameter in module.parameters():
            parameter.requires_grad = True

    counts = {
        "encoder_trainable": sum(
            parameter.numel()
            for parameter in model.encoder.parameters()
            if parameter.requires_grad
        ),
        "bottleneck_trainable": sum(
            parameter.numel()
            for parameter in model.bottleneck.parameters()
            if parameter.requires_grad
        ),
        "decoder_trainable": sum(
            parameter.numel()
            for parameter in model.decoder.parameters()
            if parameter.requires_grad
        ),
    }
    if counts["encoder_trainable"] != 0:
        raise RuntimeError("encoder freeze contract failed")
    if min(counts["bottleneck_trainable"], counts["decoder_trainable"]) <= 0:
        raise RuntimeError("bottleneck and decoder must have trainable parameters")
    return counts


def load_proxy_rows(index_path: Path) -> list[dict[str, Any]]:
    with Path(index_path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_category_folds(
    rows: Sequence[Mapping[str, Any]], fold_count: int = 4
) -> dict[str, Any]:
    count = int(fold_count)
    if count < 2:
        raise ValueError("fold_count must be at least 2")

    stats: dict[str, dict[str, int]] = {}
    for row in rows:
        if str(row.get("phase")) != "test":
            continue
        category = str(row["category"])
        item = stats.setdefault(
            category, {"test_images": 0, "anomaly_images": 0}
        )
        item["test_images"] += 1
        item["anomaly_images"] += int(row.get("label", 0))
    if len(stats) < count:
        raise ValueError(
            f"Need at least {count} categories for {count} folds, found {len(stats)}"
        )

    ordered = sorted(
        stats,
        key=lambda category: (
            -stats[category]["test_images"],
            -stats[category]["anomaly_images"],
            category,
        ),
    )
    folds: list[dict[str, Any]] = [
        {
            "fold": fold,
            "categories": [],
            "test_images": 0,
            "anomaly_images": 0,
        }
        for fold in range(count)
    ]
    for category in ordered:
        target = min(
            folds,
            key=lambda fold: (
                int(fold["test_images"]),
                int(fold["anomaly_images"]),
                len(fold["categories"]),
                int(fold["fold"]),
            ),
        )
        target["categories"].append(category)
        target["test_images"] += stats[category]["test_images"]
        target["anomaly_images"] += stats[category]["anomaly_images"]

    return {
        "version": "v283_category_folds_v1",
        "fold_count": count,
        "category_count": len(stats),
        "folds": folds,
    }


def partition_rows_for_fold(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_out: set[str],
    partition_key: str = "category",
) -> dict[str, list[dict[str, Any]]]:
    held_out_categories = {str(category) for category in held_out}
    if not held_out_categories:
        raise ValueError("held_out cannot be empty")
    key = str(partition_key)
    if not key:
        raise ValueError("partition_key cannot be empty")
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if key not in row:
            raise KeyError(f"partition_key {key!r} is missing from a proxy row")
        category = str(row[key])
        phase = str(row.get("phase"))
        if category in held_out_categories:
            if phase == "test":
                validation.append(row)
            continue
        if phase in {"train", "test"}:
            train.append(row)
    if not train or not validation:
        raise ValueError(
            f"Fold partition is empty: train={len(train)} validation={len(validation)}"
        )
    if any(str(row[key]) in held_out_categories for row in train):
        raise RuntimeError(f"held-out {key} leaked into training rows")
    return {"train": train, "validation": validation}


def checkpoint_is_eligible(
    report: Mapping[str, Any], *, max_normal_area_growth: float = 0.01
) -> bool:
    return bool(report.get("finite", False)) and float(
        report.get("composite_gain", float("-inf"))
    ) > 0.0 and float(report.get("normal_area_growth", float("inf"))) <= float(
        max_normal_area_growth
    )


def route_gate_decision(
    *, fold_gains: Sequence[float], category_gains: Sequence[float]
) -> dict[str, Any]:
    folds = np.asarray(list(fold_gains), dtype=np.float64)
    categories = np.asarray(list(category_gains), dtype=np.float64)
    if folds.size != 4:
        raise ValueError(f"route gate requires four fold gains, found {folds.size}")
    if categories.size == 0:
        raise ValueError("route gate requires category gains")
    positive_folds = int(np.sum(folds > 0.0))
    mean_gain = float(np.mean(folds))
    nonregression_rate = float(np.mean(categories >= 0.0))
    worst_quartile = float(np.quantile(categories, 0.25))
    checks = {
        "positive_folds": positive_folds == 4,
        "mean_composite_gain": mean_gain >= 0.02,
        "category_nonregression": nonregression_rate >= 0.80,
        "worst_quartile": worst_quartile >= -0.005,
        "finite": bool(np.isfinite(folds).all() and np.isfinite(categories).all()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "fold_gains": folds.tolist(),
        "positive_folds": positive_folds,
        "mean_composite_gain": mean_gain,
        "category_nonregression_rate": nonregression_rate,
        "worst_quartile_gain": worst_quartile,
    }


def run_aggregate_gate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    selected_reports = []
    missing_folds = []
    for fold in range(4):
        fold_root = output_root / f"fold{fold}"
        train_report_path = fold_root / "train_report.json"
        if not train_report_path.is_file():
            missing_folds.append(fold)
            continue
        train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
        step = train_report.get("best_validation_step")
        if step is None:
            missing_folds.append(fold)
            continue
        validation_path = fold_root / f"validation_i{int(step):06d}.json"
        if not validation_path.is_file():
            missing_folds.append(fold)
            continue
        selected_reports.append(json.loads(validation_path.read_text(encoding="utf-8")))

    if missing_folds:
        result = {
            "version": "v283_route_gate_v1",
            "passed": False,
            "failed_checks": ["missing_eligible_fold"],
            "missing_folds": missing_folds,
            "selected_reports": selected_reports,
        }
    else:
        fold_gains = [float(report["composite_gain"]) for report in selected_reports]
        category_gains = [
            float(gain)
            for report in selected_reports
            for gain in report["category_gains"].values()
        ]
        gate = route_gate_decision(
            fold_gains=fold_gains, category_gains=category_gains
        )
        result = {
            "version": "v283_route_gate_v1",
            **gate,
            "selected_reports": [
                {
                    "fold_id": int(report["fold_id"]),
                    "step": int(report["step"]),
                    "checkpoint": str(report["checkpoint"]),
                    "composite_gain": float(report["composite_gain"]),
                    "held_out_categories": report["held_out_categories"],
                }
                for report in selected_reports
            ],
        }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if args.require_pass and not result["passed"]:
        raise SystemExit(3)
    return result


def metric_triplet(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "ap": float("nan"), "best_f1": float("nan")}
    precision, recall, _thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "best_f1": float(np.nanmax(f1)),
    }


def composite_from_metrics(metrics: Mapping[str, Mapping[str, float]]) -> float:
    values = [
        float(value)
        for level in ("pixel", "image", "object")
        for value in metrics[level].values()
        if math.isfinite(float(value))
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def top_mean(values: np.ndarray, ratio: float = 0.005) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    count = max(1, int(math.ceil(flat.size * float(ratio))))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def sample_validation_pixels(
    anomaly_map: np.ndarray,
    mask: np.ndarray,
    *,
    identity: str,
    count_per_class: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    flat_scores = np.asarray(anomaly_map, dtype=np.float32).reshape(-1)
    flat_labels = np.asarray(mask, dtype=np.uint8).reshape(-1)
    seed = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in (0, 1):
        candidates = np.flatnonzero(flat_labels == label)
        if candidates.size == 0:
            continue
        size = min(int(count_per_class), int(candidates.size))
        selected.append(rng.choice(candidates, size=size, replace=False))
    if not selected:
        raise ValueError(f"No pixels available for {identity}")
    indices = np.concatenate(selected)
    return flat_labels[indices], flat_scores[indices]


def _object_arrays(
    labels: Sequence[int], scores: Sequence[float], groups: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, dict[str, list[float] | int]] = {}
    for label, score, group in zip(labels, scores, groups):
        item = grouped.setdefault(str(group), {"label": 0, "scores": []})
        item["label"] = max(int(item["label"]), int(label))
        cast_scores = item["scores"]
        assert isinstance(cast_scores, list)
        cast_scores.append(float(score))
    object_labels = []
    object_scores = []
    for group in sorted(grouped):
        item = grouped[group]
        values = item["scores"]
        assert isinstance(values, list)
        object_labels.append(int(item["label"]))
        object_scores.append(float(np.mean(sorted(values, reverse=True)[:2])))
    return np.asarray(object_labels, dtype=np.uint8), np.asarray(object_scores, dtype=np.float64)


def evaluate_model(
    *,
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    input_size: int,
    device: Any,
    batch_size: int,
    num_workers: int,
    normal_threshold: float | None = None,
    limit_rows: int = 0,
    score_scale: float = 1.0,
) -> dict[str, Any]:
    import torch

    if not math.isfinite(float(score_scale)) or float(score_scale) <= 0.0:
        raise ValueError(f"score_scale must be finite and positive, found {score_scale}")
    selected_rows = [dict(row) for row in rows]
    if int(limit_rows) > 0:
        selected_rows = selected_rows[: int(limit_rows)]
    dataset = RealIADSupervisedDataset(selected_rows, input_size=input_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
    )
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    pixel_categories: list[np.ndarray] = []
    image_labels: list[int] = []
    image_scores: list[float] = []
    image_categories: list[str] = []
    groups: list[str] = []
    normal_sample_scores: list[np.ndarray] = []
    normal_areas: list[float] = []
    edge_responses: list[float] = []
    saturations: list[float] = []

    model.eval()
    row_offset = 0
    with torch.no_grad():
        for images, masks, labels, categories, batch_groups in loader:
            images = images.to(device, non_blocking=True)
            encoded, decoded = model(images)
            maps = differentiable_anomaly_map(encoded, decoded, int(input_size)).float().cpu().numpy()
            mask_arrays = masks.numpy()
            for index in range(int(maps.shape[0])):
                row = selected_rows[row_offset + index]
                anomaly_map = maps[index, 0] * float(score_scale)
                mask = mask_arrays[index, 0] > 0.5
                identity = str(row.get("record_id", f"{row_offset + index}"))
                sampled_labels, sampled_scores = sample_validation_pixels(
                    anomaly_map, mask, identity=identity
                )
                pixel_labels.append(sampled_labels)
                pixel_scores.append(sampled_scores)
                pixel_categories.append(
                    np.full(sampled_labels.shape, str(categories[index]), dtype=object)
                )
                label = int(labels[index])
                image_labels.append(label)
                image_scores.append(top_mean(anomaly_map, 0.005))
                image_categories.append(str(categories[index]))
                groups.append(str(batch_groups[index]))
                if label == 0:
                    normal_sample_scores.append(sampled_scores[sampled_labels == 0])
                    if normal_threshold is not None:
                        normal_areas.append(float(np.mean(anomaly_map > float(normal_threshold))))
                border = max(1, int(input_size) // 32)
                edge_mask = np.zeros_like(anomaly_map, dtype=bool)
                edge_mask[:border] = True
                edge_mask[-border:] = True
                edge_mask[:, :border] = True
                edge_mask[:, -border:] = True
                edge_responses.append(float(anomaly_map[edge_mask].mean()))
                saturations.append(float(np.mean(anomaly_map >= 0.95)))
            row_offset += int(maps.shape[0])

    pixel_label_array = np.concatenate(pixel_labels)
    pixel_score_array = np.concatenate(pixel_scores)
    pixel_category_array = np.concatenate(pixel_categories)
    image_label_array = np.asarray(image_labels, dtype=np.uint8)
    image_score_array = np.asarray(image_scores, dtype=np.float64)
    image_category_array = np.asarray(image_categories, dtype=object)
    object_labels, object_scores = _object_arrays(image_labels, image_scores, groups)
    metrics = {
        "pixel": metric_triplet(pixel_label_array, pixel_score_array),
        "image": metric_triplet(image_label_array, image_score_array),
        "object": metric_triplet(object_labels, object_scores),
    }
    category_composites: dict[str, float] = {}
    for category in sorted(set(image_categories)):
        image_selected = image_category_array == category
        pixel_selected = pixel_category_array == category
        category_groups = [group for group, cat in zip(groups, image_categories) if cat == category]
        category_object_labels, category_object_scores = _object_arrays(
            image_label_array[image_selected].tolist(),
            image_score_array[image_selected].tolist(),
            category_groups,
        )
        category_metrics = {
            "pixel": metric_triplet(pixel_label_array[pixel_selected], pixel_score_array[pixel_selected]),
            "image": metric_triplet(image_label_array[image_selected], image_score_array[image_selected]),
            "object": metric_triplet(category_object_labels, category_object_scores),
        }
        category_composites[category] = composite_from_metrics(category_metrics)
    return {
        "metrics": metrics,
        "composite": composite_from_metrics(metrics),
        "category_composites": category_composites,
        "normal_sample_scores": np.concatenate(normal_sample_scores),
        "normal_area": float(np.mean(normal_areas)) if normal_areas else float("nan"),
        "edge_response": float(np.mean(edge_responses)),
        "saturation": float(np.mean(saturations)),
        "row_count": len(selected_rows),
    }


def build_validation_report(
    *,
    step: int,
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    baseline_normal_area: float,
) -> dict[str, Any]:
    category_gains = {
        category: float(current["category_composites"][category])
        - float(baseline["category_composites"][category])
        for category in sorted(current["category_composites"])
    }
    finite_values = [
        float(current["composite"]),
        float(baseline["composite"]),
        float(current["normal_area"]),
        float(baseline_normal_area),
        *category_gains.values(),
    ]
    report: dict[str, Any] = {
        "version": "v283_fold_validation_v1",
        "step": int(step),
        "finite": all(math.isfinite(value) for value in finite_values),
        "metrics": current["metrics"],
        "baseline_metrics": baseline["metrics"],
        "composite": float(current["composite"]),
        "baseline_composite": float(baseline["composite"]),
        "composite_gain": float(current["composite"]) - float(baseline["composite"]),
        "normal_area": float(current["normal_area"]),
        "baseline_normal_area": float(baseline_normal_area),
        "normal_area_growth": float(current["normal_area"]) - float(baseline_normal_area),
        "edge_response": float(current["edge_response"]),
        "saturation": float(current["saturation"]),
        "category_gains": category_gains,
    }
    report["eligible"] = checkpoint_is_eligible(report)
    return report


def atomic_torch_save(torch: Any, payload: Mapping[str, Any], path: Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)


def latest_state_checkpoint(checkpoint_dir: Path, target_steps: int) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in Path(checkpoint_dir).glob("state_i*.pt"):
        try:
            step = int(path.stem.split("i")[-1])
        except ValueError:
            continue
        if step <= int(target_steps):
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _serializable_evaluation(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metrics": result["metrics"],
        "composite": float(result["composite"]),
        "category_composites": {
            str(key): float(value) for key, value in result["category_composites"].items()
        },
        "normal_area": float(result["normal_area"]),
        "edge_response": float(result["edge_response"]),
        "saturation": float(result["saturation"]),
        "row_count": int(result["row_count"]),
    }


def run_precompute_anchor(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    adapter = _import_adapter()
    model, _deps, parameter_counts = _load_source_model(args, device)
    model.eval()
    transform = adapter._build_transform(args.input_size, args.input_size)
    dataset = adapter.CsigZipDataset(
        Path(args.train_zip),
        "Train",
        transform,
        include_metadata=True,
    )
    output_root = Path(args.output_root)
    manifest_path = output_root / "teacher_manifest.json"
    checkpoint = _source_checkpoint(args.source_run_root, args.source_step)
    source_sha = sha256_file(checkpoint)
    if manifest_path.is_file() and not args.force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "ok"
            and int(previous.get("saved", -1)) == len(dataset)
            and previous.get("source_checkpoint_sha256") == source_sha
        ):
            print(json.dumps(previous, indent=2), flush=True)
            return previous

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.infer_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )
    saved = 0
    started = time.time()
    with torch.no_grad():
        for images, categories, samples, views in loader:
            images = images.to(device, non_blocking=True)
            encoded, decoded = model(images)
            maps = (
                differentiable_anomaly_map(encoded, decoded, PATCH_GRID)
                .float()
                .cpu()
                .numpy()
            )
            for index in range(int(maps.shape[0])):
                target = (
                    output_root
                    / str(categories[index])
                    / str(samples[index])
                    / f"{views[index]}.npy"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".npy.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, maps[index, 0].astype(np.float16))
                os.replace(temporary, target)
                saved += 1
            if saved % 500 == 0:
                print(f"teacher maps saved {saved}/{len(dataset)}", flush=True)
    report = {
        "version": "v283_anchor_teacher_cache_v1",
        "status": "ok",
        "source_run_root": str(args.source_run_root),
        "source_step": int(args.source_step),
        "source_dropout": float(args.source_dropout),
        "source_checkpoint_sha256": source_sha,
        "input_size": int(args.input_size),
        "patch_grid": PATCH_GRID,
        "saved": saved,
        "parameter_counts": parameter_counts,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    if saved != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} teacher maps, saved {saved}")
    atomic_write_json(manifest_path, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_smoke_loss(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(int(args.seed))
    encoded = [
        torch.randn(int(args.batch_size), 8, 4, 4, requires_grad=False)
        for _ in range(2)
    ]
    decoded = [
        torch.randn(int(args.batch_size), 8, 4, 4, requires_grad=True)
        for _ in range(2)
    ]
    masks = torch.zeros(int(args.batch_size), 1, int(args.input_size), int(args.input_size))
    masks[-1, :, int(args.input_size) // 4 : 3 * int(args.input_size) // 4, int(args.input_size) // 4 : 3 * int(args.input_size) // 4] = 1.0
    result = compute_supervised_losses(encoded, decoded, masks, out_size=args.input_size)
    result["total"].backward()
    report = {
        "version": "v283_smoke_loss_v1",
        "status": "ok",
        "finite": all(
            bool(torch.isfinite(value).all())
            for key, value in result.items()
            if key != "map"
        ),
        "losses": {
            key: float(value.detach().item())
            for key, value in result.items()
            if key != "map"
        },
        "decoder_gradients": all(feature.grad is not None for feature in decoded),
        "encoder_gradients": any(feature.grad is not None for feature in encoded),
    }
    if not report["finite"] or not report["decoder_gradients"] or report["encoder_gradients"]:
        raise RuntimeError(f"Smoke loss contract failed: {report}")
    print(json.dumps(report, indent=2), flush=True)
    return report


def _next_loader_batch(loader: Any, iterator: Any) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _learning_rate_lambda(step: int, target_steps: int, warmup_steps: int = 100) -> float:
    if int(step) < int(warmup_steps):
        return max(1e-3, (int(step) + 1) / max(1, int(warmup_steps)))
    progress = (int(step) - int(warmup_steps)) / max(1, int(target_steps) - int(warmup_steps))
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _immutable_training_config(args: argparse.Namespace, fold_categories: Sequence[str]) -> dict[str, Any]:
    checkpoint = _source_checkpoint(args.source_run_root, args.source_step)
    return {
        "version": "v283_supervised_dinomaly_longtrain_v1",
        "fold_id": int(args.fold_id),
        "partition_key": str(args.partition_key),
        "held_out_categories": sorted(str(item) for item in fold_categories),
        "fold_manifest_sha256": sha256_file(args.fold_manifest),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_step": int(args.source_step),
        "source_dropout": float(args.source_dropout),
        "input_size": int(args.input_size),
        "batch_size": int(args.batch_size),
        "accumulation_steps": int(args.accumulation_steps),
        "target_steps": int(args.target_steps),
        "bottleneck_lr": float(args.bottleneck_lr),
        "decoder_lr": float(args.decoder_lr),
        "normal_loss_weight": float(args.normal_loss_weight),
        "normal_mean_budget": float(args.normal_mean_budget),
        "normal_quantile_budget": float(args.normal_quantile_budget),
        "weight_decay": float(args.weight_decay),
        "amp": bool(args.amp),
    }


def run_train_fold(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("train-fold requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.backends.cudnn.benchmark = True

    rows = load_proxy_rows(args.proxy_index)
    fold_manifest = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
    fold_entry = next(
        item for item in fold_manifest["folds"] if int(item["fold"]) == int(args.fold_id)
    )
    held_out = {str(item) for item in fold_entry["categories"]}
    split = partition_rows_for_fold(
        rows, held_out=held_out, partition_key=str(args.partition_key)
    )
    train_rows = split["train"]
    validation_rows = split["validation"]
    if int(args.limit_train_rows) > 0:
        normal = [row for row in train_rows if int(row.get("label", 0)) == 0]
        anomaly = [row for row in train_rows if int(row.get("label", 0)) == 1]
        half = max(1, int(args.limit_train_rows) // 2)
        train_rows = normal[:half] + anomaly[:half]
    if int(args.limit_validation_rows) > 0:
        normal = [row for row in validation_rows if int(row.get("label", 0)) == 0]
        anomaly = [row for row in validation_rows if int(row.get("label", 0)) == 1]
        half = max(1, int(args.limit_validation_rows) // 2)
        validation_rows = normal[:half] + anomaly[:half]

    run_dir = Path(args.output_root) / f"fold{int(args.fold_id)}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps, args.target_steps)
    immutable_config = _immutable_training_config(args, sorted(held_out))
    atomic_write_json(run_dir / "immutable_config.json", immutable_config)

    model, deps, parameter_counts = _load_source_model(args, device)
    adapter = _import_adapter()
    baseline_path = run_dir / "baseline_validation.json"
    if baseline_path.is_file():
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline = baseline_payload["evaluation"]
        normal_threshold = float(baseline_payload["normal_threshold"])
        baseline_normal_area = float(baseline_payload["baseline_normal_area"])
    else:
        first_baseline = evaluate_model(
            model=model,
            rows=validation_rows,
            input_size=args.input_size,
            device=device,
            batch_size=args.validation_batch_size,
            num_workers=args.num_workers,
        )
        normal_threshold = float(np.quantile(first_baseline["normal_sample_scores"], 0.995))
        second_baseline = evaluate_model(
            model=model,
            rows=validation_rows,
            input_size=args.input_size,
            device=device,
            batch_size=args.validation_batch_size,
            num_workers=args.num_workers,
            normal_threshold=normal_threshold,
        )
        baseline_normal_area = float(second_baseline["normal_area"])
        baseline = _serializable_evaluation(second_baseline)
        atomic_write_json(
            baseline_path,
            {
                "version": "v283_baseline_validation_v1",
                "normal_threshold": normal_threshold,
                "baseline_normal_area": baseline_normal_area,
                "evaluation": baseline,
            },
        )

    train_dataset = RealIADSupervisedDataset(train_rows, input_size=args.input_size)
    sampler = make_balanced_sampler(train_rows, seed=args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(args.num_workers) > 0,
    )
    transform = adapter._build_transform(args.input_size, args.input_size)
    anchor_base = adapter.CsigZipDataset(
        Path(args.train_zip), "Train", transform, include_metadata=True
    )
    anchor_dataset = AnchorTeacherDataset(anchor_base, args.teacher_root)
    anchor_generator = torch.Generator().manual_seed(int(args.seed) + 17)
    anchor_loader = torch.utils.data.DataLoader(
        anchor_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=anchor_generator,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(args.num_workers) > 0,
    )

    optimizer = deps["StableAdamW"](
        [
            {"params": model.bottleneck.parameters(), "lr": float(args.bottleneck_lr)},
            {"params": model.decoder.parameters(), "lr": float(args.decoder_lr)},
        ],
        betas=(0.9, 0.999),
        weight_decay=float(args.weight_decay),
        amsgrad=False,
        eps=1e-10,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_lambda(step, args.target_steps),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    start_step = 0
    resume_path = latest_state_checkpoint(checkpoint_dir, args.target_steps) if args.resume == "auto" else None
    if args.resume not in {"auto", "none"}:
        resume_path = Path(args.resume)
    if resume_path is not None:
        state = torch.load(resume_path, map_location="cpu")
        if state.get("immutable_config") != immutable_config:
            raise ValueError("resume immutable_config mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        sampler.generator.set_state(state["sampler_rng"])
        anchor_generator.set_state(state["anchor_rng"])
        print(f"RESUME {resume_path} step={start_step}", flush=True)
    if start_step >= int(args.target_steps):
        previous = json.loads((run_dir / "train_report.json").read_text(encoding="utf-8"))
        print(json.dumps(previous, indent=2), flush=True)
        return previous

    train_iterator = iter(train_loader)
    anchor_iterator = iter(anchor_loader)
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    validation_reports: list[dict[str, Any]] = []
    for report_path in sorted(run_dir.glob("validation_i*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("step", 0)) <= start_step:
            validation_reports.append(report)
    model.train()
    for step in range(start_step + 1, int(args.target_steps) + 1):
        component_sums: dict[str, float] = {}
        for _micro in range(int(args.accumulation_steps)):
            supervised_batch, train_iterator = _next_loader_batch(train_loader, train_iterator)
            anchor_batch, anchor_iterator = _next_loader_batch(anchor_loader, anchor_iterator)
            supervised_images, masks, _labels, _categories, _groups = supervised_batch
            anchor_images, teacher_maps = anchor_batch
            supervised_images = supervised_images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            anchor_images = anchor_images.to(device, non_blocking=True)
            teacher_maps = teacher_maps.to(device, non_blocking=True)
            joined_images = torch.cat([supervised_images, anchor_images], dim=0)
            supervised_count = int(supervised_images.shape[0])
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                encoded, decoded = model(joined_images)
                supervised_encoded = [feature[:supervised_count] for feature in encoded]
                supervised_decoded = [feature[:supervised_count] for feature in decoded]
                anchor_encoded = [feature[supervised_count:] for feature in encoded]
                anchor_decoded = [feature[supervised_count:] for feature in decoded]
                supervised_losses = compute_supervised_losses(
                    supervised_encoded,
                    supervised_decoded,
                    masks,
                    out_size=args.input_size,
                    normal_loss_weight=float(args.normal_loss_weight),
                    normal_mean_budget=float(args.normal_mean_budget),
                    normal_quantile_budget=float(args.normal_quantile_budget),
                )
                anchor_official = global_cosine_loss(anchor_encoded, anchor_decoded)
                anchor_map = differentiable_anomaly_map(
                    anchor_encoded, anchor_decoded, PATCH_GRID
                ).float()
                distill = anchor_distillation_loss(anchor_map, teacher_maps)
                loss = supervised_losses["total"] + anchor_official + 0.50 * distill
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            scaler.scale(loss / int(args.accumulation_steps)).backward()
            component_values = {
                **{
                    key: float(value.detach().item())
                    for key, value in supervised_losses.items()
                    if key != "map"
                },
                "anchor_official": float(anchor_official.detach().item()),
                "distill": float(distill.detach().item()),
                "combined": float(loss.detach().item()),
            }
            for key, value in component_values.items():
                component_sums[key] = component_sums.get(key, 0.0) + value
        scaler.unscale_(optimizer)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(model.bottleneck.parameters()) + list(model.decoder.parameters()),
                max_norm=1.0,
            ).item()
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if step <= 5 or step % int(args.log_every) == 0 or step in checkpoint_steps:
            means = {
                key: value / int(args.accumulation_steps)
                for key, value in component_sums.items()
            }
            print(
                f"step {step}/{args.target_steps} loss={means['combined']:.6f} "
                f"focal={means['focal']:.6f} dice={means['dice']:.6f} "
                f"rank={means['rank']:.6f} distill={means['distill']:.6f} "
                f"grad={gradient_norm:.6f} elapsed_minutes={(time.time() - started) / 60.0:.2f} "
                f"gpu_max_mib={torch.cuda.max_memory_allocated() / 1024**2:.1f}",
                flush=True,
            )

        if step in checkpoint_steps:
            checkpoint_path = checkpoint_dir / f"state_i{step:06d}.pt"
            state = {
                "version": "v283_supervised_dinomaly_longtrain_v1",
                "step": step,
                "immutable_config": immutable_config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "sampler_rng": sampler.generator.get_state(),
                "anchor_rng": anchor_generator.get_state(),
                "created_unix": time.time(),
            }
            atomic_torch_save(torch, state, checkpoint_path)
            current = evaluate_model(
                model=model,
                rows=validation_rows,
                input_size=args.input_size,
                device=device,
                batch_size=args.validation_batch_size,
                num_workers=args.num_workers,
                normal_threshold=normal_threshold,
            )
            report = build_validation_report(
                step=step,
                current=current,
                baseline=baseline,
                baseline_normal_area=baseline_normal_area,
            )
            report["checkpoint"] = str(checkpoint_path)
            report["fold_id"] = int(args.fold_id)
            report["held_out_categories"] = sorted(held_out)
            atomic_write_json(run_dir / f"validation_i{step:06d}.json", report)
            validation_reports.append(report)
            model.train()

    eligible = [report for report in validation_reports if report["eligible"]]
    best = max(eligible, key=lambda report: float(report["composite_gain"])) if eligible else None
    result = {
        "version": "v283_supervised_dinomaly_longtrain_v1",
        "status": "ok",
        "fold_id": int(args.fold_id),
        "held_out_categories": sorted(held_out),
        "target_steps": int(args.target_steps),
        "parameter_counts": parameter_counts,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "validation_reports": [
            str(run_dir / f"validation_i{int(report['step']):06d}.json")
            for report in validation_reports
        ],
        "best_checkpoint": best["checkpoint"] if best else None,
        "best_validation_step": int(best["step"]) if best else None,
        "best_composite_gain": float(best["composite_gain"]) if best else None,
    }
    atomic_write_json(run_dir / "train_report.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def run_build_folds(args: argparse.Namespace) -> dict[str, Any]:
    result = build_category_folds(
        load_proxy_rows(args.proxy_index), fold_count=args.fold_count
    )
    output_path = Path(args.output)
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous != result:
            raise RuntimeError(
                f"Refusing to overwrite non-identical fold manifest: {output_path}"
            )
    else:
        atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V283 supervised Dinomaly long training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    folds = subparsers.add_parser("build-folds")
    folds.add_argument("--proxy-index", type=Path, default=DEFAULT_PROXY_INDEX)
    folds.add_argument("--output", type=Path, required=True)
    folds.add_argument("--fold-count", type=int, default=4)
    folds.set_defaults(func=run_build_folds)

    smoke = subparsers.add_parser("smoke-loss")
    smoke.add_argument("--input-size", type=int, default=64)
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--seed", type=int, default=28301)
    smoke.set_defaults(func=run_smoke_loss)

    teacher = subparsers.add_parser("precompute-anchor")
    teacher.add_argument("--source-run-root", type=Path, required=True)
    teacher.add_argument("--source-step", type=int, default=DEFAULT_SOURCE_STEP)
    teacher.add_argument("--source-dropout", type=float, required=True)
    teacher.add_argument("--output-root", type=Path, required=True)
    teacher.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    teacher.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHT_DIR)
    teacher.add_argument("--backbone", default="dinov3_vit_base_16")
    teacher.add_argument("--reduce-dim", type=int, default=256)
    teacher.add_argument("--input-size", type=int, default=448)
    teacher.add_argument("--infer-batch-size", type=int, default=4)
    teacher.add_argument("--num-workers", type=int, default=2)
    teacher.add_argument("--device", default="cuda:0")
    teacher.add_argument("--force", action="store_true")
    teacher.set_defaults(func=run_precompute_anchor)

    train = subparsers.add_parser("train-fold")
    train.add_argument("--fold-id", type=int, choices=range(4), required=True)
    train.add_argument("--fold-manifest", type=Path, required=True)
    train.add_argument("--partition-key", default="category")
    train.add_argument("--proxy-index", type=Path, default=DEFAULT_PROXY_INDEX)
    train.add_argument("--source-run-root", type=Path, required=True)
    train.add_argument("--source-step", type=int, default=DEFAULT_SOURCE_STEP)
    train.add_argument("--source-dropout", type=float, required=True)
    train.add_argument("--teacher-root", type=Path, required=True)
    train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    train.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    train.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHT_DIR)
    train.add_argument("--backbone", default="dinov3_vit_base_16")
    train.add_argument("--reduce-dim", type=int, default=256)
    train.add_argument("--input-size", type=int, default=448)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--accumulation-steps", type=int, default=8)
    train.add_argument("--target-steps", type=int, default=30000)
    train.add_argument("--checkpoint-steps", default="5000,10000,20000,30000")
    train.add_argument("--bottleneck-lr", type=float, default=2e-4)
    train.add_argument("--decoder-lr", type=float, default=1e-4)
    train.add_argument("--normal-loss-weight", type=float, default=0.25)
    train.add_argument("--normal-mean-budget", type=float, default=0.01)
    train.add_argument("--normal-quantile-budget", type=float, default=0.10)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--validation-batch-size", type=int, default=4)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--log-every", type=int, default=50)
    train.add_argument("--resume", default="auto")
    train.add_argument("--seed", type=int, default=28301)
    train.add_argument("--limit-train-rows", type=int, default=0)
    train.add_argument("--limit-validation-rows", type=int, default=0)
    train.add_argument("--amp", action="store_true")
    train.set_defaults(func=run_train_fold)

    aggregate = subparsers.add_parser("aggregate-gate")
    aggregate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--require-pass", action="store_true")
    aggregate.set_defaults(func=run_aggregate_gate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
