from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path.home() / "csig"
TRAIN_ZIP = ROOT / "Train.zip"
TEST_ZIP = ROOT / "Test_A.zip"
OUTPUT_ROOT = ROOT / "outputs" / "v182_dinomaly2_dinov3_official"
EVIDENCE_ROOT = OUTPUT_ROOT / "evidence"
OFFICIAL_DINOMALY2_REPO = "https://github.com/guojiajeremy/Dinomaly2"
OFFICIAL_DINOV3_REPO = "https://github.com/facebookresearch/dinov3"
ROUTE_NAME = "v182_dinomaly2_dinov3_official"
OFFICIAL_REAL_IAD_VARIETY_PARAMS = {
    "backbone": "dinov3_vit_base_16",
    "input_size": 280,
    "crop_size": 280,
    "dropout": 0.4,
    "lc": 2,
    "ll": 1,
    "cr": 1,
    "la": 1,
    "topk_ratio": 0.001,
    "target_total_iters": 100000,
    "lr": 0.002,
    "final_lr_ratio": 1.0,
    "loss_factor": 0.10,
}
DEFAULT_CHECKPOINT_EPOCHS = [20, 40, 60, 80]
OFFICIAL_BASE_LARGE_TARGET_LAYERS = [4, 6, 8, 10, 12, 14, 16, 18]
ADAPTER_SAFE_12_BLOCK_TARGET_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]
DINOV3_TARGET_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]


@dataclass(frozen=True)
class CsigImageRecord:
    category: str
    sample: str
    view: str
    member: str

    @property
    def group(self) -> str:
        return f"{self.category}/{self.sample}"


class CsigZipDataset:
    def __init__(
        self,
        zip_path: Path,
        root_name: str,
        transform: Any,
        limit_views: int = 0,
        shuffle_seed: int | None = None,
        include_metadata: bool = False,
        category_filter: str | Sequence[str] | None = None,
    ) -> None:
        self.zip_path = Path(zip_path)
        self.root_name = str(root_name)
        self.transform = transform
        self.include_metadata = bool(include_metadata)
        records = index_csig_zip(self.zip_path, self.root_name)
        categories = _parse_category_filter(category_filter)
        if categories:
            records = [record for record in records if record.category in categories]
        if shuffle_seed is not None:
            rng = random.Random(int(shuffle_seed))
            records = list(records)
            rng.shuffle(records)
        if int(limit_views) > 0:
            records = list(records)[: int(limit_views)]
        if not records:
            suffix = f" for category_filter={','.join(sorted(categories))}" if categories else ""
            raise ValueError(f"No PNG records found under {root_name}/ in {zip_path}{suffix}")
        self.records = records
        self._archive: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _zip(self) -> zipfile.ZipFile:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.zip_path)
        return self._archive

    def __getitem__(self, index: int) -> Any:
        record = self.records[int(index)]
        data = self._zip().read(record.member)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        tensor = self.transform(image)
        if self.include_metadata:
            return tensor, record.category, record.sample, record.view
        return tensor, 0


def index_csig_zip(zip_path: Path, root_name: str) -> list[CsigImageRecord]:
    records: list[CsigImageRecord] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.endswith("/") or not name.lower().endswith(".png"):
                continue
            parts = PurePosixPath(name).parts
            if len(parts) != 4 or parts[0] != root_name:
                continue
            _root, category, sample, file_name = parts
            view = Path(file_name).stem
            if view not in {"0", "1", "2", "3", "4"}:
                continue
            records.append(CsigImageRecord(category=category, sample=sample, view=view, member=name))
    return sorted(records, key=lambda item: (item.category, item.sample, int(item.view), item.member))


def _parse_category_filter(value: str | Sequence[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


def _parse_csv_ints(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _build_transform(input_size: int, crop_size: int) -> Any:
    from torchvision import transforms

    interpolation = getattr(getattr(transforms, "InterpolationMode", transforms), "BICUBIC", Image.BICUBIC)
    return transforms.Compose(
        [
            transforms.Resize((int(input_size), int(input_size)), interpolation=interpolation),
            transforms.CenterCrop(int(crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _embed_config(backbone: str) -> tuple[int, int]:
    lower = str(backbone).lower()
    if "huge" in lower or "7b" in lower:
        return 4096, 32
    if "large" in lower or "vit_l" in lower:
        return 1024, 16
    if "base" in lower or "vit_b" in lower:
        return 768, 12
    return 384, 6


def _target_layers(backbone: str) -> list[int]:
    lower = str(backbone).lower()
    if "dinov3" in lower:
        return list(DINOV3_TARGET_LAYERS)
    if "base" in lower or "large" in lower or "vit_b" in lower or "vit_l" in lower:
        return list(OFFICIAL_BASE_LARGE_TARGET_LAYERS)
    return list(ADAPTER_SAFE_12_BLOCK_TARGET_LAYERS)


def _encoder_block_depth(encoder: Any) -> int | None:
    blocks = getattr(encoder, "blocks", None)
    if blocks is not None:
        try:
            return int(len(blocks))
        except TypeError:
            pass
    for attr in ("n_blocks", "num_blocks", "depth"):
        value = getattr(encoder, attr, None)
        if isinstance(value, int):
            return int(value)
    return None


def _select_target_layers(backbone: str, encoder_depth: int | None) -> tuple[list[int], str]:
    preferred = _target_layers(backbone)
    if encoder_depth is None or max(preferred) < int(encoder_depth):
        return preferred, "official_preferred"

    if int(encoder_depth) >= 10 and max(ADAPTER_SAFE_12_BLOCK_TARGET_LAYERS) < int(encoder_depth):
        return list(ADAPTER_SAFE_12_BLOCK_TARGET_LAYERS), "depth_safe_12_block"

    if int(encoder_depth) >= 8:
        start = int(encoder_depth) - 8
        return list(range(start, int(encoder_depth))), "depth_safe_last8"

    raise ValueError(
        f"Encoder depth {encoder_depth} is too shallow for Dinomaly2 fusion; "
        "need at least 8 feature layers."
    )


def _import_official_modules() -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from models import vit_encoder
    from models.uad import Dinomaly
    from models.vision_transformer import Block as VitBlock
    from models.vision_transformer import LinearAttention2
    from optimizers import StableAdamW
    from utils import WarmupCosineScheduler, cal_anomaly_maps, global_cosine_hm_percent

    return {
        "torch": torch,
        "nn": nn,
        "vit_encoder": vit_encoder,
        "Dinomaly": Dinomaly,
        "VitBlock": VitBlock,
        "LinearAttention2": LinearAttention2,
        "StableAdamW": StableAdamW,
        "WarmupCosineScheduler": WarmupCosineScheduler,
        "cal_anomaly_maps": cal_anomaly_maps,
        "global_cosine_hm_percent": global_cosine_hm_percent,
    }


def _build_model(
    deps: dict[str, Any],
    backbone: str,
    dropout: float,
    reduce_dim: int,
    context_aware_recenter: bool,
    mask_neighbor_size: int,
    weights_dir: Path | str,
) -> tuple[Any, Any]:
    nn = deps["nn"]
    vit_encoder = deps["vit_encoder"]
    embed_dim, num_heads = _embed_config(backbone)
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(backbone, WEIGHTS_DIR=str(weights_dir))
    encoder_depth = _encoder_block_depth(encoder)
    target_layers, layer_mode = _select_target_layers(backbone, encoder_depth)
    bottleneck = nn.ModuleList(
        [
            nn.Sequential(nn.Linear(embed_dim, int(reduce_dim)), nn.Dropout(p=float(dropout))),
            nn.Sequential(
                nn.Linear(int(reduce_dim), embed_dim * 4),
                nn.GELU(),
                nn.Dropout(p=float(dropout)),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(p=float(dropout)),
            ),
        ]
    )
    decoder = nn.ModuleList(
        [
            deps["VitBlock"](
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=partial(deps["LinearAttention2"], eps=1e-8),
            )
            for _ in target_layers
        ]
    )
    model = deps["Dinomaly"](
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=False,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        context_aware_recenter=bool(context_aware_recenter),
        mask_neighbor_size=int(mask_neighbor_size),
        use_get_intermediate=True,
    )
    if hasattr(model, "init_weights"):
        model.init_weights()
    setattr(model, "_v182_encoder_depth", encoder_depth)
    setattr(model, "_v182_target_layers", target_layers)
    setattr(model, "_v182_layer_mode", layer_mode)
    return model, nn.ModuleList([bottleneck, decoder])


def _checkpoint_path(run_dir: Path, run_name: str, epoch: int) -> Path:
    return run_dir / "checkpoints" / f"{run_name}_e{int(epoch)}.pt"


def _make_loader(dataset: Any, batch_size: int, shuffle: bool, num_workers: int, deps: dict[str, Any]) -> Any:
    return deps["torch"].utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=bool(shuffle),
    )


def _to_numpy_maps(anomaly_maps: Any) -> np.ndarray:
    if isinstance(anomaly_maps, (tuple, list)):
        anomaly_maps = anomaly_maps[0]
    if hasattr(anomaly_maps, "detach"):
        anomaly_maps = anomaly_maps.detach().cpu().numpy()
    array = np.asarray(anomaly_maps, dtype=np.float32)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected anomaly maps with 3 dims, got shape {array.shape}")
    return array


def _forward_maps(model: Any, images: Any, deps: dict[str, Any], out_size: int) -> np.ndarray:
    encoded, decoded = model(images)
    maps = deps["cal_anomaly_maps"](encoded, decoded, out_size=int(out_size))
    return _to_numpy_maps(maps)


def _normalize(values: np.ndarray, lo: float, hi: float, power: float, blur_radius: float) -> np.ndarray:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo + 1e-8:
        lo = float(np.percentile(values, 95.0))
        hi = float(np.percentile(values, 99.85))
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if float(blur_radius) > 0.0:
        image = Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L")
        image = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        normalized = np.asarray(image, dtype=np.float32) / 255.0
    return np.power(np.clip(normalized, 0.0, 1.0), float(power)).astype(np.float32)


def _resize_map(mask: np.ndarray, out_size: int) -> np.ndarray:
    if mask.shape == (int(out_size), int(out_size)):
        return mask.astype(np.float32)
    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    image = Image.fromarray(np.rint(np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    resized = image.resize((int(out_size), int(out_size)), resampling)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _save_gray(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


def _calibrate(
    model: Any,
    loader: Any,
    deps: dict[str, Any],
    out_size: int,
    max_views: int,
) -> tuple[float, float, dict[str, Any]]:
    torch = deps["torch"]
    rows: list[tuple[float, float]] = []
    seen = 0
    model.eval()
    with torch.no_grad():
        for images, *_metadata in loader:
            images = images.cuda(non_blocking=True)
            maps = _forward_maps(model, images, deps, out_size)
            for item in maps:
                rows.append((float(np.percentile(item, 95.0)), float(np.percentile(item, 99.85))))
                seen += 1
                if int(max_views) > 0 and seen >= int(max_views):
                    break
            if int(max_views) > 0 and seen >= int(max_views):
                break
    if not rows:
        raise RuntimeError("No calibration maps were produced")
    array = np.asarray(rows, dtype=np.float32)
    lo = float(np.median(array[:, 0]))
    hi = float(np.median(array[:, 1]))
    return lo, hi, {"calibration_views": int(seen), "global_lo": lo, "global_hi": hi}


def run_worker(args: argparse.Namespace) -> None:
    started = time.time()
    _seed_everything(args.seed)
    deps = _import_official_modules()
    torch = deps["torch"]
    torch.backends.cudnn.benchmark = True
    transform = _build_transform(args.input_size, args.crop_size)
    dataset = CsigZipDataset(args.train_zip, "Train", transform, shuffle_seed=args.seed, category_filter=args.category_filter)
    loader = _make_loader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, deps=deps)
    if len(loader) == 0:
        raise RuntimeError("Training loader is empty; lower batch size or inspect Train.zip")

    run_dir = Path(args.output_root) / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoints = set(_parse_csv_ints(args.checkpoint_epochs))
    checkpoints.add(int(args.epochs))
    final_checkpoint = _checkpoint_path(run_dir, args.run_name, int(args.epochs))
    if final_checkpoint.exists() and not args.force_train:
        print(f"final checkpoint exists, skip training: {final_checkpoint}", flush=True)
        return

    model, trainable = _build_model(
        deps=deps,
        backbone=args.backbone,
        dropout=args.dropout,
        reduce_dim=args.reduce_dim,
        context_aware_recenter=not args.disable_context,
        mask_neighbor_size=args.mask_neighbor_size,
        weights_dir=args.weights_dir,
    )
    model = model.cuda()
    optimizer = deps["StableAdamW"](
        [{"params": trainable.parameters()}],
        lr=float(args.lr),
        betas=(0.9, 0.999),
        weight_decay=float(args.weight_decay),
        amsgrad=False,
        eps=1e-10,
    )
    accumulation_steps = max(1, int(args.accumulation_steps))
    effective_batch = int(args.batch_size) * accumulation_steps
    total_optimizer_steps = max(1, int(math.ceil((int(args.epochs) * len(loader)) / accumulation_steps)))
    scheduler = deps["WarmupCosineScheduler"](
        optimizer,
        final_ratio=float(args.final_lr_ratio),
        total_epochs=total_optimizer_steps,
        warmup_epochs=min(100, max(10, total_optimizer_steps // 100)),
    )

    print(
        f"run={args.run_name} source={OFFICIAL_DINOMALY2_REPO} dinov3_source={OFFICIAL_DINOV3_REPO} backbone={args.backbone} "
        f"input={args.input_size} crop={args.crop_size} epochs={args.epochs} batch={args.batch_size} "
        f"accumulation_steps={accumulation_steps} effective_batch={effective_batch} "
        f"dropout={args.dropout} lc=2 ll=1 cr=1 category_filter={args.category_filter or 'ALL'} train_views={len(dataset)} "
        f"weights_dir={args.weights_dir} "
        f"encoder_depth={getattr(model, '_v182_encoder_depth', None)} "
        f"target_layers={getattr(model, '_v182_target_layers', [])} "
        f"layer_mode={getattr(model, '_v182_layer_mode', 'unknown')}",
        flush=True,
    )
    mini_iteration = 0
    optimizer_iteration = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for images, _labels in loader:
            images = images.cuda(non_blocking=True)
            encoded, decoded = model(images)
            percent = min(float(args.percent_max) * optimizer_iteration / max(1, int(args.percent_warmup)), float(args.percent_max))
            loss = deps["global_cosine_hm_percent"](encoded, decoded, p=percent, factor=float(args.loss_factor))
            loss_to_backward = loss / accumulation_steps
            loss_to_backward.backward()
            losses.append(float(loss.item()))
            mini_iteration += 1
            should_step = mini_iteration % accumulation_steps == 0
            if should_step:
                deps["nn"].utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_iteration += 1
        if epoch == int(args.epochs) and mini_iteration % accumulation_steps != 0:
            deps["nn"].utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_iteration += 1
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch {epoch}/{args.epochs} loss={mean_loss:.6f} elapsed_minutes={(time.time() - started) / 60:.2f}", flush=True)
        if epoch in checkpoints:
            checkpoint = _checkpoint_path(run_dir, args.run_name, epoch)
            torch.save(model.state_dict(), checkpoint)
            print(f"saved checkpoint: {checkpoint}", flush=True)


def infer_checkpoints(args: argparse.Namespace) -> None:
    started = time.time()
    _seed_everything(args.seed)
    deps = _import_official_modules()
    torch = deps["torch"]
    transform = _build_transform(args.input_size, args.crop_size)
    train_dataset = CsigZipDataset(
        args.train_zip,
        "Train",
        transform,
        limit_views=args.calibration_max_views,
        shuffle_seed=args.seed,
        include_metadata=True,
        category_filter=args.category_filter,
    )
    test_dataset = CsigZipDataset(args.test_zip, "Test_A", transform, include_metadata=True, category_filter=args.category_filter)
    train_loader = _make_loader(train_dataset, args.infer_batch_size, shuffle=False, num_workers=args.num_workers, deps=deps)
    test_loader = _make_loader(test_dataset, args.infer_batch_size, shuffle=False, num_workers=args.num_workers, deps=deps)
    run_dir = Path(args.output_root) / args.run_name
    checkpoints = _parse_csv_ints(args.checkpoint_epochs)

    for epoch in checkpoints:
        checkpoint = _checkpoint_path(run_dir, args.run_name, epoch)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        model, _trainable = _build_model(
            deps=deps,
            backbone=args.backbone,
            dropout=args.dropout,
            reduce_dim=args.reduce_dim,
            context_aware_recenter=not args.disable_context,
            mask_neighbor_size=args.mask_neighbor_size,
            weights_dir=args.weights_dir,
        )
        model = model.cuda()
        model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
        model.eval()
        lo, hi, calibration = _calibrate(
            model=model,
            loader=train_loader,
            deps=deps,
            out_size=args.crop_size,
            max_views=args.calibration_max_views,
        )
        save_root = Path(args.evidence_root) / args.run_name / f"maps_norm_e{int(epoch)}"
        saved = 0
        with torch.no_grad():
            for images, categories, samples, views in test_loader:
                images = images.cuda(non_blocking=True)
                maps = _forward_maps(model, images, deps, args.crop_size)
                for idx, raw_map in enumerate(maps):
                    normalized = _normalize(raw_map, lo, hi, args.norm_power, args.blur_radius)
                    resized = _resize_map(normalized, args.out_mask)
                    _save_gray(save_root / str(categories[idx]) / str(samples[idx]) / f"{views[idx]}_mask.png", resized)
                    saved += 1
        report = {
            "run_name": args.run_name,
            "epoch": int(epoch),
            "official_repo": OFFICIAL_DINOMALY2_REPO,
            "official_dinov3_repo": OFFICIAL_DINOV3_REPO,
            "backbone": args.backbone,
            "input_size": int(args.input_size),
            "crop_size": int(args.crop_size),
            "dropout": float(args.dropout),
            "category_filter": str(args.category_filter or "ALL"),
            "saved_mask_count": int(saved),
            "test_view_count": int(len(test_dataset)),
            "elapsed_minutes": float((time.time() - started) / 60.0),
            **calibration,
        }
        (Path(args.evidence_root) / args.run_name / f"infer_e{int(epoch)}.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2), flush=True)


def audit_zips(args: argparse.Namespace) -> None:
    train_records = index_csig_zip(args.train_zip, "Train")
    test_records = index_csig_zip(args.test_zip, "Test_A")
    report = {
        "version": ROUTE_NAME,
        "official_repo": OFFICIAL_DINOMALY2_REPO,
        "official_dinov3_repo": OFFICIAL_DINOV3_REPO,
        "official_params": OFFICIAL_REAL_IAD_VARIETY_PARAMS,
        "train_zip": str(args.train_zip),
        "test_zip": str(args.test_zip),
        "train_views": len(train_records),
        "test_views": len(test_records),
        "train_groups": len({record.group for record in train_records}),
        "test_groups": len({record.group for record in test_records}),
        "train_categories": sorted({record.category for record in train_records}),
        "test_categories": sorted({record.category for record in test_records}),
    }
    print(json.dumps(report, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V182 official Dinomaly2 + DINOv3 CSIG adapter.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--run-name", required=True)
        command.add_argument("--seed", type=int, default=9101)
        command.add_argument("--backbone", default=OFFICIAL_REAL_IAD_VARIETY_PARAMS["backbone"])
        command.add_argument("--input-size", type=int, default=OFFICIAL_REAL_IAD_VARIETY_PARAMS["input_size"])
        command.add_argument("--crop-size", type=int, default=OFFICIAL_REAL_IAD_VARIETY_PARAMS["crop_size"])
        command.add_argument("--dropout", type=float, default=OFFICIAL_REAL_IAD_VARIETY_PARAMS["dropout"])
        command.add_argument("--reduce-dim", type=int, default=256)
        command.add_argument("--mask-neighbor-size", type=int, default=0)
        command.add_argument("--disable-context", action="store_true")
        command.add_argument("--weights-dir", type=Path, default=ROOT / "Dinomaly2_dinov3" / "backbones" / "weights")
        command.add_argument("--checkpoint-epochs", default=",".join(str(epoch) for epoch in DEFAULT_CHECKPOINT_EPOCHS))
        command.add_argument("--train-zip", type=Path, default=TRAIN_ZIP)
        command.add_argument("--test-zip", type=Path, default=TEST_ZIP)
        command.add_argument("--output-root", type=Path, default=EVIDENCE_ROOT)
        command.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
        command.add_argument("--num-workers", type=int, default=2)
        command.add_argument("--category-filter", default="")

    worker = sub.add_parser("worker")
    add_common(worker)
    worker.add_argument("--epochs", type=int, default=80)
    worker.add_argument("--batch-size", type=int, default=4)
    worker.add_argument("--accumulation-steps", type=int, default=1)
    worker.add_argument("--lr", type=float, default=OFFICIAL_REAL_IAD_VARIETY_PARAMS["lr"])
    worker.add_argument("--final-lr-ratio", type=float, default=OFFICIAL_REAL_IAD_VARIETY_PARAMS["final_lr_ratio"])
    worker.add_argument("--weight-decay", type=float, default=1e-4)
    worker.add_argument("--percent-max", type=float, default=0.90)
    worker.add_argument("--percent-warmup", type=int, default=1000)
    worker.add_argument("--loss-factor", type=float, default=OFFICIAL_REAL_IAD_VARIETY_PARAMS["loss_factor"])
    worker.add_argument("--force-train", action="store_true")

    infer = sub.add_parser("infer-checkpoints")
    add_common(infer)
    infer.add_argument("--infer-batch-size", type=int, default=6)
    infer.add_argument("--out-mask", type=int, default=448)
    infer.add_argument("--calibration-max-views", type=int, default=1400)
    infer.add_argument("--norm-power", type=float, default=0.90)
    infer.add_argument("--blur-radius", type=float, default=1.0)

    audit = sub.add_parser("audit-zips")
    audit.add_argument("--train-zip", type=Path, default=TRAIN_ZIP)
    audit.add_argument("--test-zip", type=Path, default=TEST_ZIP)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "worker":
        run_worker(args)
        return
    if args.cmd == "infer-checkpoints":
        infer_checkpoints(args)
        return
    if args.cmd == "audit-zips":
        audit_zips(args)
        return
    parser.error(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
