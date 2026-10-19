from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path.home() / "csig"
DEFAULT_ROUTE_ROOT = ROOT / "outputs" / "v283_supervised_dinomaly_longtrain"
DEFAULT_EVIDENCE_ROOT = DEFAULT_ROUTE_ROOT / "test_evidence"
DEFAULT_WEIGHT_DIR = ROOT / "Dinomaly2_dinov3" / "backbones" / "weights"


def select_checkpoint(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [dict(report) for report in reports if bool(report.get("eligible", False))]
    if not eligible:
        raise ValueError("No eligible validation checkpoint")
    return max(
        eligible,
        key=lambda report: (float(report["composite_gain"]), -int(report["step"])),
    )


def _top_mean(values: np.ndarray, ratio: float) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    count = max(1, int(math.ceil(flat.size * float(ratio))))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def save_float_record(
    path: Path,
    *,
    map_array: np.ndarray,
    category: str,
    sample: str,
    view: str,
    fold: int,
    checkpoint_step: int,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    calibrated = np.asarray(map_array, dtype=np.float32)
    if calibrated.shape != (448, 448):
        raise ValueError(f"Expected a 448x448 map, found {calibrated.shape}")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            calibrated_map=calibrated,
            top001=np.float32(_top_mean(calibrated, 0.001)),
            top005=np.float32(_top_mean(calibrated, 0.005)),
            top010=np.float32(_top_mean(calibrated, 0.010)),
            maximum=np.float32(float(calibrated.max())),
            area020=np.float32(float(np.mean(calibrated > 0.20))),
            area050=np.float32(float(np.mean(calibrated > 0.50))),
            category=np.asarray(str(category)),
            sample=np.asarray(str(sample)),
            view=np.asarray(str(view)),
            fold=np.int8(int(fold)),
            checkpoint_step=np.int64(int(checkpoint_step)),
        )
    os.replace(temporary, target)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _import_adapter() -> Any:
    try:
        from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    except ImportError:
        import v182_dinomaly2_dinov3_csig_official as adapter
    return adapter


def _calibrate_map(raw_map: np.ndarray, lo: float, hi: float, power: float) -> np.ndarray:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo + 1e-8:
        raise ValueError(f"Invalid map calibration lo={lo} hi={hi}")
    normalized = np.clip((np.asarray(raw_map, np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return np.power(normalized, float(power)).astype(np.float32)


def _save_png(path: Path, values: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(target)


def _selected_fold(route_root: Path, fold_id: int) -> dict[str, Any]:
    gate_path = Path(route_root) / "route_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not bool(gate.get("passed", False)):
        raise RuntimeError(f"V283 route gate has not passed: {gate_path}")
    matches = [
        dict(item)
        for item in gate["selected_reports"]
        if int(item["fold_id"]) == int(fold_id)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one selected report for fold {fold_id}, found {len(matches)}")
    return matches[0]


def run_infer_fold(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    adapter = _import_adapter()
    deps = adapter._import_official_modules()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    selected = _selected_fold(args.route_root, args.fold_id)
    checkpoint = Path(selected["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload.get("immutable_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"Checkpoint has no immutable config: {checkpoint}")
    if int(config["fold_id"]) != int(args.fold_id) or int(payload["step"]) != int(selected["step"]):
        raise ValueError("Selected checkpoint identity mismatch")
    model, _trainable = adapter._build_model(
        deps=deps,
        backbone=args.backbone,
        dropout=float(config["source_dropout"]),
        reduce_dim=args.reduce_dim,
        context_aware_recenter=True,
        mask_neighbor_size=0,
        weights_dir=args.weights_dir,
    )
    model.load_state_dict(payload["model"])
    del payload
    model = model.to(device).eval()
    transform = adapter._build_transform(args.input_size, args.input_size)
    train_dataset = adapter.CsigZipDataset(
        Path(args.train_zip),
        "Train",
        transform,
        limit_views=args.calibration_max_views,
        shuffle_seed=args.seed,
        include_metadata=True,
    )
    test_dataset = adapter.CsigZipDataset(
        Path(args.test_zip), args.test_root_name, transform, include_metadata=True
    )
    train_loader = adapter._make_loader(
        train_dataset,
        args.infer_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        deps=deps,
    )
    test_loader = adapter._make_loader(
        test_dataset,
        args.infer_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        deps=deps,
    )
    lo, hi, calibration = adapter._calibrate(
        model=model,
        loader=train_loader,
        deps=deps,
        out_size=args.input_size,
        max_views=args.calibration_max_views,
    )
    fold_root = Path(args.evidence_root) / f"fold{int(args.fold_id)}"
    float_root = fold_root / "float_maps"
    png_root = fold_root / "maps_norm_e1"
    started = time.time()
    saved = 0
    with torch.no_grad():
        for images, categories, samples, views in test_loader:
            images = images.to(device, non_blocking=True)
            raw_maps = adapter._forward_maps(model, images, deps, args.input_size)
            for index, raw_map in enumerate(raw_maps):
                calibrated = _calibrate_map(raw_map, lo, hi, args.norm_power)
                category = str(categories[index])
                sample = str(samples[index])
                view = str(views[index])
                save_float_record(
                    float_root / category / sample / f"{view}.npz",
                    map_array=calibrated,
                    category=category,
                    sample=sample,
                    view=view,
                    fold=args.fold_id,
                    checkpoint_step=int(selected["step"]),
                )
                _save_png(png_root / category / sample / f"{view}_mask.png", calibrated)
                saved += 1
            if saved % 500 == 0:
                print(f"fold{args.fold_id}: saved {saved}/{len(test_dataset)}", flush=True)
    report = {
        "version": "v284_supervised_dinomaly_state_infer_v1",
        "status": "ok",
        "fold_id": int(args.fold_id),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(selected["step"]),
        "saved": saved,
        "test_record_count": len(test_dataset),
        "test_root_name": str(args.test_root_name),
        "elapsed_minutes": (time.time() - started) / 60.0,
        **calibration,
    }
    if saved != len(test_dataset):
        raise RuntimeError(f"Expected {len(test_dataset)} {args.test_root_name} records, saved {saved}")
    _atomic_json(fold_root / "infer_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_build_ensemble(args: argparse.Namespace) -> dict[str, Any]:
    evidence_root = Path(args.evidence_root)
    fold_roots = [evidence_root / f"fold{fold}" / "float_maps" for fold in range(4)]
    reference_paths = sorted(fold_roots[0].rglob("*.npz"))
    if not reference_paths:
        raise RuntimeError("No fold-0 records found")
    expected_count = len(reference_paths)
    for fold_root in fold_roots[1:]:
        fold_count = len(list(fold_root.rglob("*.npz")))
        if fold_count != expected_count:
            raise RuntimeError(
                f"Fold record count mismatch: expected {expected_count}, found {fold_count} in {fold_root}"
            )
    output_root = evidence_root / "ensemble"
    saved = 0
    for reference in reference_paths:
        relative = reference.relative_to(fold_roots[0])
        records = []
        for root in fold_roots:
            with np.load(root / relative) as payload:
                records.append(
                    {
                        "calibrated_map": np.asarray(
                            payload["calibrated_map"], dtype=np.float32
                        ).copy(),
                        "top005": float(payload["top005"]),
                        "category": str(payload["category"]),
                        "sample": str(payload["sample"]),
                        "view": str(payload["view"]),
                    }
                )
        maps = np.stack([record["calibrated_map"] for record in records], axis=0)
        category = records[0]["category"]
        sample = records[0]["sample"]
        view = records[0]["view"]
        mean_map = maps.mean(axis=0, dtype=np.float32)
        median_map = np.median(maps, axis=0).astype(np.float32)
        std_map = maps.std(axis=0, dtype=np.float32)
        agreement_map = np.mean(maps >= 0.5, axis=0, dtype=np.float32)
        target = output_root / "float_maps" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                calibrated_map=median_map,
                mean_map=mean_map,
                median_map=median_map,
                std_map=std_map,
                agreement_map=agreement_map,
                fold_top005=np.asarray(
                    [record["top005"] for record in records], dtype=np.float32
                ),
                category=np.asarray(category),
                sample=np.asarray(sample),
                view=np.asarray(view),
            )
        os.replace(temporary, target)
        _save_png(output_root / "maps_norm_e1" / relative.parent / f"{view}_mask.png", median_map)
        saved += 1
    report = {
        "version": "v284_supervised_dinomaly_ensemble_v1",
        "status": "ok",
        "saved": saved,
        "record_count": expected_count,
        "folds": 4,
    }
    _atomic_json(output_root / "ensemble_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V284 selected V283 checkpoint inference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer-fold")
    infer.add_argument("--fold-id", type=int, choices=range(4), required=True)
    infer.add_argument("--route-root", type=Path, default=DEFAULT_ROUTE_ROOT)
    infer.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    infer.add_argument("--train-zip", type=Path, default=ROOT / "Train.zip")
    infer.add_argument("--test-zip", type=Path, default=ROOT / "Test_A.zip")
    infer.add_argument("--test-root-name", default="Test_A")
    infer.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHT_DIR)
    infer.add_argument("--backbone", default="dinov3_vit_base_16")
    infer.add_argument("--reduce-dim", type=int, default=256)
    infer.add_argument("--input-size", type=int, default=448)
    infer.add_argument("--infer-batch-size", type=int, default=4)
    infer.add_argument("--num-workers", type=int, default=2)
    infer.add_argument("--calibration-max-views", type=int, default=1400)
    infer.add_argument("--norm-power", type=float, default=0.90)
    infer.add_argument("--seed", type=int, default=28401)
    infer.set_defaults(func=run_infer_fold)

    ensemble = subparsers.add_parser("build-ensemble")
    ensemble.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    ensemble.set_defaults(func=run_build_ensemble)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
