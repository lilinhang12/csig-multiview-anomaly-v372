from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path.home() / "csig"
DEFAULT_WEIGHT_DIR = ROOT / "Dinomaly2_dinov3" / "backbones" / "weights"
DEFAULT_EVIDENCE_ROOT = ROOT / "outputs" / "v276_i448_dual_inference" / "evidence"


def state_checkpoint_path(run_root: Path, optimizer_step: int) -> Path:
    return Path(run_root) / "checkpoints" / f"state_i{int(optimizer_step):06d}.pt"


def extract_model_state(
    payload: Any,
    *,
    expected_step: int,
    expected_input_size: int,
    expected_dropout: float | None = None,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    if int(payload.get("optimizer_step", -1)) != int(expected_step):
        raise ValueError(
            f"Checkpoint optimizer_step={payload.get('optimizer_step')} does not match {expected_step}"
        )
    config = payload.get("immutable_config")
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint immutable_config is missing")
    if int(config.get("input_size", -1)) != int(expected_input_size):
        raise ValueError(f"Checkpoint input_size={config.get('input_size')} is not {expected_input_size}")
    if expected_dropout is not None and not math.isclose(
        float(config.get("dropout", -1.0)), float(expected_dropout), rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError(f"Checkpoint dropout={config.get('dropout')} is not {expected_dropout}")
    model = payload.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("Checkpoint model state is missing")
    return model


def _import_adapter() -> Any:
    try:
        from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    except ImportError:
        import v182_dinomaly2_dinov3_csig_official as adapter
    return adapter


def _float_calibrate(raw_map: np.ndarray, lo: float, hi: float, power: float) -> np.ndarray:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo + 1e-8:
        return np.zeros_like(raw_map, dtype=np.float32)
    normalized = np.clip((raw_map.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return np.power(normalized, float(power)).astype(np.float32)


def _save_float_record(
    path: Path,
    *,
    raw_map: np.ndarray,
    calibrated_map: np.ndarray,
    category: str,
    sample: str,
    view: str,
    source_run: str,
    checkpoint_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = calibrated_map.reshape(-1)
    top1_count = max(1, int(math.ceil(flat.size * 0.01)))
    top5_count = max(1, int(math.ceil(flat.size * 0.05)))
    top1 = float(np.partition(flat, flat.size - top1_count)[-top1_count:].mean())
    top5 = float(np.partition(flat, flat.size - top5_count)[-top5_count:].mean())
    np.savez_compressed(
        path,
        raw_map=raw_map.astype(np.float32, copy=False),
        calibrated_map=calibrated_map.astype(np.float32, copy=False),
        top1_percent=np.float32(top1),
        top5_percent=np.float32(top5),
        maximum=np.float32(float(flat.max())),
        anomalous_area=np.float32(float(np.mean(flat >= 0.5))),
        category=np.asarray(category),
        sample=np.asarray(sample),
        view=np.asarray(view),
        source_run=np.asarray(source_run),
        checkpoint_step=np.int64(checkpoint_step),
        input_size=np.int64(448),
    )


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    adapter = _import_adapter()
    deps = adapter._import_official_modules()
    torch = deps["torch"]
    adapter._seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    run_root = Path(args.source_run_root)
    checkpoint = state_checkpoint_path(run_root, args.checkpoint_step)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    evidence_run = Path(args.evidence_root) / args.run_name
    report_path = evidence_run / f"infer_e{int(args.checkpoint_step)}.json"
    if report_path.is_file() and not args.force:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("status") == "ok" and previous.get("saved_mask_count") == previous.get("test_record_count"):
            print(json.dumps(previous, indent=2), flush=True)
            return previous

    transform = adapter._build_transform(args.input_size, args.crop_size)
    train_dataset = adapter.CsigZipDataset(
        Path(args.train_zip),
        "Train",
        transform,
        limit_views=args.calibration_max_views,
        shuffle_seed=args.seed,
        include_metadata=True,
    )
    test_dataset = adapter.CsigZipDataset(
        Path(args.test_zip),
        args.test_root_name,
        transform,
        include_metadata=True,
        category_filter=args.category_filter,
    )
    train_loader = adapter._make_loader(
        train_dataset, args.infer_batch_size, shuffle=False, num_workers=args.num_workers, deps=deps
    )
    test_loader = adapter._make_loader(
        test_dataset, args.infer_batch_size, shuffle=False, num_workers=args.num_workers, deps=deps
    )

    payload = torch.load(checkpoint, map_location="cpu")
    model_state = extract_model_state(
        payload,
        expected_step=args.checkpoint_step,
        expected_input_size=args.input_size,
        expected_dropout=args.dropout,
    )
    model, _trainable = adapter._build_model(
        deps=deps,
        backbone=args.backbone,
        dropout=args.dropout,
        reduce_dim=args.reduce_dim,
        context_aware_recenter=not args.disable_context,
        mask_neighbor_size=args.mask_neighbor_size,
        weights_dir=args.weights_dir,
    )
    model.load_state_dict(model_state)
    del payload, model_state
    model = model.cuda().eval()

    started = time.time()
    lo, hi, calibration = adapter._calibrate(
        model=model,
        loader=train_loader,
        deps=deps,
        out_size=args.crop_size,
        max_views=args.calibration_max_views,
    )
    png_root = evidence_run / f"maps_norm_e{int(args.checkpoint_step)}"
    float_root = evidence_run / f"float_maps_e{int(args.checkpoint_step)}"
    saved_masks = 0
    saved_floats = 0
    with torch.no_grad():
        for images, categories, samples, views in test_loader:
            images = images.cuda(non_blocking=True)
            maps = adapter._forward_maps(model, images, deps, args.crop_size)
            for index, raw_map in enumerate(maps):
                calibrated_float = _float_calibrate(raw_map, lo, hi, args.norm_power)
                final_map = adapter._normalize(raw_map, lo, hi, args.norm_power, args.blur_radius)
                final_map = adapter._resize_map(final_map, args.out_mask)
                category = str(categories[index])
                sample = str(samples[index])
                view = str(views[index])
                adapter._save_gray(png_root / category / sample / f"{view}_mask.png", final_map)
                saved_masks += 1
                if args.save_float:
                    _save_float_record(
                        float_root / category / sample / f"{view}.npz",
                        raw_map=raw_map,
                        calibrated_map=calibrated_float,
                        category=category,
                        sample=sample,
                        view=view,
                        source_run=args.run_name,
                        checkpoint_step=args.checkpoint_step,
                    )
                    saved_floats += 1
            if saved_masks % 500 == 0:
                print(f"saved {saved_masks}/{len(test_dataset)} masks", flush=True)

    report = {
        "version": "v276_i448_state_infer",
        "status": "ok",
        "run_name": args.run_name,
        "source_run_root": str(run_root),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(args.checkpoint_step),
        "input_size": int(args.input_size),
        "crop_size": int(args.crop_size),
        "dropout": float(args.dropout),
        "saved_mask_count": int(saved_masks),
        "test_record_count": int(len(test_dataset)),
        "saved_float_count": int(saved_floats),
        "elapsed_minutes": (time.time() - started) / 60.0,
        **calibration,
    }
    evidence_run.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if saved_masks != len(test_dataset):
        raise RuntimeError(f"Expected {len(test_dataset)} masks, saved {saved_masks}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer V272/V275 DDP state checkpoints at 448 pixels")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--train-zip", type=Path, default=ROOT / "Train.zip")
    parser.add_argument("--test-zip", type=Path, default=ROOT / "Test_A.zip")
    parser.add_argument("--test-root-name", default="Test_A")
    parser.add_argument("--category-filter", default="")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHT_DIR)
    parser.add_argument("--backbone", default="dinov3_vit_base_16")
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--out-mask", type=int, default=448)
    parser.add_argument("--reduce-dim", type=int, default=256)
    parser.add_argument("--mask-neighbor-size", type=int, default=0)
    parser.add_argument("--disable-context", action="store_true")
    parser.add_argument("--infer-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--calibration-max-views", type=int, default=1400)
    parser.add_argument("--norm-power", type=float, default=0.90)
    parser.add_argument("--blur-radius", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=27601)
    parser.add_argument("--save-float", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    run_inference(build_parser().parse_args())
