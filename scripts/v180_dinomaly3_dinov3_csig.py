from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path.home() / "csig"
TRAIN_ZIP = ROOT / "Train.zip"
TEST_ZIP = ROOT / "Test_A.zip"
OUTPUT_ROOT = ROOT / "outputs" / "v180_dinomaly3_dinov3"
EVIDENCE_ROOT = OUTPUT_ROOT / "evidence"
OFFICIAL_DINOV3_REPO = "https://github.com/facebookresearch/dinov3"
OFFICIAL_DINOMALY_REPO = "https://github.com/guojiajeremy/Dinomaly"
ROUTE_NAME = "v180_dinomaly3_dinov3"
DEFAULT_CHECKPOINT_ITERS = [1]


@dataclass(frozen=True)
class CsigImageRecord:
    category: str
    sample: str
    view: str
    member: str

    @property
    def group(self) -> str:
        return f"{self.category}/{self.sample}"


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


def _parse_feature_layers(value: str | Sequence[int] | None) -> list[int]:
    if value is None:
        return []
    layers = _parse_csv_ints(value) if isinstance(value, str) else [int(item) for item in value]
    if any(layer < 0 for layer in layers):
        raise ValueError(f"feature layers must be non-negative block indices: {layers}")
    return layers


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


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
        return tensor, record.category


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


def _make_loader(dataset: Any, batch_size: int, shuffle: bool, num_workers: int) -> Any:
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
    )


class MockDinov3Model:
    def __call__(self, images: Any) -> Any:
        import torch

        pooled = torch.nn.functional.avg_pool2d(images, kernel_size=16, stride=16)
        tokens = pooled.flatten(2).transpose(1, 2)
        extras = [tokens, tokens * 0.5, tokens * 0.25]
        return {"x_norm_patchtokens": torch.cat(extras, dim=-1)}

    def eval(self) -> "MockDinov3Model":
        return self

    def to(self, _device: str) -> "MockDinov3Model":
        return self


def _strip_prefix_tokens(tokens: Any) -> tuple[Any, tuple[int, int]]:
    count = int(tokens.shape[1])
    if count <= 1:
        raise RuntimeError("DINOv3 returned only one token; patch tokens are required")
    grid = int(round(math.sqrt(count)))
    if grid * grid == count:
        return tokens, (grid, grid)

    # HF-style outputs can contain cls plus DINOv3 storage/register tokens before patches.
    for prefix_count in (1, 5, 2, 3, 4, 6, 7, 8):
        patch_count = count - prefix_count
        if patch_count <= 1:
            continue
        patch_grid = int(round(math.sqrt(patch_count)))
        if patch_grid * patch_grid == patch_count:
            return tokens[:, prefix_count:, :], (patch_grid, patch_grid)
    raise RuntimeError(f"Cannot infer square patch grid from token count {count}")


def _load_dinov3_model(args: argparse.Namespace, device: str) -> Any:
    backend = str(args.backend)
    if backend == "mock":
        if not bool(args.allow_mock):
            raise RuntimeError("The mock backend is test-only. Pass --allow-mock only in local tests.")
        return MockDinov3Model().to(device).eval()

    if backend == "torchhub":
        import torch

        repo = str(args.dinov3_repo)
        weights = str(args.weights).strip()
        if not weights and not bool(args.allow_official_download):
            raise FileNotFoundError(
                "Missing official DINOv3 weights. Set --weights or V180_DINOV3_WEIGHTS after obtaining access "
                "from the official DINOv3/HuggingFace release. To try the official hub default download, pass "
                "--allow-official-download."
            )
        hub_kwargs: dict[str, Any] = {}
        if weights:
            hub_kwargs["weights"] = weights
        if Path(repo).exists():
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from dinov3.hub import backbones

            builder = getattr(backbones, str(args.model_name))
            model = builder(**hub_kwargs)
        else:
            model = torch.hub.load(repo, str(args.model_name), source="github", **hub_kwargs)
        return model.to(device).eval()

    if backend == "huggingface":
        from transformers import AutoModel

        token = str(args.hf_token).strip() or None
        model = AutoModel.from_pretrained(str(args.hf_model), token=token, trust_remote_code=True)
        return model.to(device).eval()

    raise ValueError(f"Unknown backend: {backend}")


def _extract_patch_tokens(model: Any, images: Any, feature_layers: str | Sequence[int] | None = None) -> tuple[Any, tuple[int, int]]:
    import torch
    import torch.nn.functional as F

    layers = _parse_feature_layers(feature_layers)
    if layers and hasattr(model, "get_intermediate_layers"):
        try:
            output = model.get_intermediate_layers(images, n=layers, return_class_token=False)
        except TypeError:
            output = model.get_intermediate_layers(images, n=layers)
        if not isinstance(output, (tuple, list)) or not output:
            raise RuntimeError(f"DINOv3 did not return intermediate layers for n={layers}")
        pieces = []
        grid: tuple[int, int] | None = None
        for item in output:
            layer_tokens = item[0] if isinstance(item, (tuple, list)) else item
            layer_tokens, layer_grid = _strip_prefix_tokens(layer_tokens)
            if grid is None:
                grid = layer_grid
            elif grid != layer_grid:
                raise RuntimeError(f"DINOv3 intermediate grids differ: {grid} vs {layer_grid}")
            pieces.append(F.normalize(layer_tokens.float(), dim=-1))
        tokens = torch.cat(pieces, dim=-1)
        tokens = F.normalize(tokens.float(), dim=-1)
        if not torch.isfinite(tokens).all():
            raise RuntimeError("DINOv3 produced non-finite multilayer patch tokens")
        return tokens, grid or (0, 0)

    if hasattr(model, "forward_features"):
        output = model.forward_features(images)
    elif hasattr(model, "get_intermediate_layers"):
        output = model.get_intermediate_layers(images, n=1)[0]
    else:
        output = model(images)
    tokens: Any | None = None
    if isinstance(output, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state"):
            if key in output:
                tokens = output[key]
                break
    elif hasattr(output, "last_hidden_state"):
        tokens = output.last_hidden_state
    elif isinstance(output, (tuple, list)) and output:
        tokens = output[-1]
    else:
        tokens = output

    if tokens is None:
        raise RuntimeError("DINOv3 model did not expose patch tokens")
    if not hasattr(tokens, "ndim") or tokens.ndim != 3:
        raise RuntimeError(f"Expected DINOv3 patch tokens with shape B,N,C, got {getattr(tokens, 'shape', None)}")

    tokens, grid = _strip_prefix_tokens(tokens)
    tokens = F.normalize(tokens.float(), dim=-1)
    if not torch.isfinite(tokens).all():
        raise RuntimeError("DINOv3 produced non-finite patch tokens")
    return tokens, grid


def _sample_tokens(tokens: Any, max_patches: int, rng: np.random.Generator) -> np.ndarray:
    array = tokens.detach().cpu().numpy().reshape(-1, int(tokens.shape[-1])).astype(np.float32)
    if array.shape[0] <= int(max_patches):
        return array
    indices = rng.choice(array.shape[0], size=int(max_patches), replace=False)
    return array[np.sort(indices)]


def _memory_path(run_dir: Path, category: str) -> Path:
    return run_dir / "memory" / f"{category}.npz"


def _save_memory(path: Path, memory: np.ndarray, category: str, grid: tuple[int, int], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        memory=np.asarray(memory, dtype=np.float16),
        category=np.asarray([category]),
        grid=np.asarray(grid, dtype=np.int32),
        meta=np.asarray([json.dumps(meta, sort_keys=True)]),
    )


def _load_memory(path: Path) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    memory = np.asarray(data["memory"], dtype=np.float32)
    grid_array = np.asarray(data["grid"], dtype=np.int32)
    meta = json.loads(str(np.asarray(data["meta"])[0]))
    return memory, (int(grid_array[0]), int(grid_array[1])), meta


def _distance_maps(tokens: Any, memory: np.ndarray, chunk_size: int) -> np.ndarray:
    import torch

    device = tokens.device
    mem = torch.from_numpy(np.asarray(memory, dtype=np.float32)).to(device)
    mem = torch.nn.functional.normalize(mem, dim=-1)
    outputs = []
    for batch_tokens in tokens:
        best_chunks = []
        for start in range(0, mem.shape[0], int(chunk_size)):
            similarity = batch_tokens @ mem[start : start + int(chunk_size)].T
            best_chunks.append(similarity.max(dim=1).values)
        best = torch.stack(best_chunks, dim=0).max(dim=0).values
        outputs.append((1.0 - best).clamp(0.0, 2.0).detach().cpu().numpy())
    return np.stack(outputs, axis=0).astype(np.float32)


def _resize_map(mask: np.ndarray, out_size: int) -> np.ndarray:
    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    image = Image.fromarray(np.rint(np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    resized = image.resize((int(out_size), int(out_size)), resampling)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _save_gray(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


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


def run_worker(args: argparse.Namespace) -> None:
    started = time.time()
    _seed_everything(args.seed)
    import torch

    device = "cuda" if torch.cuda.is_available() and str(args.device) == "auto" else str(args.device)
    if device == "auto":
        device = "cpu"
    transform = _build_transform(args.input_size, args.crop_size)
    dataset = CsigZipDataset(
        args.train_zip,
        "Train",
        transform,
        shuffle_seed=args.seed,
        include_metadata=True,
        category_filter=args.category_filter,
    )
    loader = _make_loader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = _load_dinov3_model(args, device)
    run_dir = Path(args.output_root) / args.run_name
    rng = np.random.default_rng(int(args.seed))
    by_category: dict[str, list[np.ndarray]] = {}
    grid_by_category: dict[str, tuple[int, int]] = {}

    print(
        f"run={args.run_name} route={ROUTE_NAME} official_dinov3={OFFICIAL_DINOV3_REPO} "
        f"official_dinomaly={OFFICIAL_DINOMALY_REPO} backend={args.backend} model={args.model_name or args.hf_model} "
        f"input={args.input_size} crop={args.crop_size} train_views={len(dataset)} category_filter={args.category_filter or 'ALL'}",
        flush=True,
    )
    with torch.no_grad():
        for index, (images, categories, _samples, _views) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            tokens, grid = _extract_patch_tokens(model, images, args.feature_layers)
            per_image_cap = max(1, int(args.max_patches_per_category) // max(1, len(dataset)))
            for item_index, category in enumerate(categories):
                sampled = _sample_tokens(tokens[item_index : item_index + 1], per_image_cap, rng)
                by_category.setdefault(str(category), []).append(sampled)
                grid_by_category[str(category)] = grid
            if index % int(args.log_interval) == 0 or index == len(loader):
                print(f"worker views={min(index * args.batch_size, len(dataset))}/{len(dataset)}", flush=True)

    reports = []
    for category, chunks in sorted(by_category.items()):
        memory = np.concatenate(chunks, axis=0).astype(np.float32)
        if memory.shape[0] > int(args.max_patches_per_category):
            indices = rng.choice(memory.shape[0], size=int(args.max_patches_per_category), replace=False)
            memory = memory[np.sort(indices)]
        meta = {
            "version": ROUTE_NAME,
            "category": category,
            "backend": str(args.backend),
            "model_name": str(args.model_name),
            "hf_model": str(args.hf_model),
            "feature_layers": _parse_feature_layers(args.feature_layers),
            "patch_count": int(memory.shape[0]),
            "feature_dim": int(memory.shape[1]),
            "crop_size": int(args.crop_size),
        }
        path = _memory_path(run_dir, category)
        _save_memory(path, memory, category, grid_by_category[category], meta)
        reports.append({**meta, "memory_path": str(path), "grid": list(grid_by_category[category])})
        print(f"saved memory: {path} patches={memory.shape[0]} dim={memory.shape[1]}", flush=True)

    report = {
        "version": ROUTE_NAME,
        "mode": "worker",
        "status": "ok",
        "run_name": args.run_name,
        "official_dinov3_repo": OFFICIAL_DINOV3_REPO,
        "official_dinomaly_repo": OFFICIAL_DINOMALY_REPO,
        "categories": reports,
        "elapsed_minutes": float((time.time() - started) / 60.0),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def _calibrate_category(args: argparse.Namespace, model: Any, memory: np.ndarray, category: str, device: str) -> tuple[float, float, int]:
    import torch

    transform = _build_transform(args.input_size, args.crop_size)
    dataset = CsigZipDataset(
        args.train_zip,
        "Train",
        transform,
        limit_views=args.calibration_max_views,
        shuffle_seed=args.seed,
        include_metadata=True,
        category_filter=None if category == "__global__" else category,
    )
    loader = _make_loader(dataset, args.infer_batch_size, shuffle=False, num_workers=args.num_workers)
    rows: list[tuple[float, float]] = []
    seen = 0
    with torch.no_grad():
        for images, *_metadata in loader:
            images = images.to(device, non_blocking=True)
            tokens, grid = _extract_patch_tokens(model, images, args.feature_layers)
            distances = _distance_maps(tokens, memory, args.memory_chunk_size)
            for item in distances:
                raw = item.reshape(grid)
                rows.append((float(np.percentile(raw, 95.0)), float(np.percentile(raw, 99.85))))
                seen += 1
                if int(args.calibration_max_views) > 0 and seen >= int(args.calibration_max_views):
                    break
            if int(args.calibration_max_views) > 0 and seen >= int(args.calibration_max_views):
                break
    if not rows:
        return 0.0, 1.0, 0
    array = np.asarray(rows, dtype=np.float32)
    return float(np.median(array[:, 0])), float(np.median(array[:, 1])), int(seen)


def infer_checkpoints(args: argparse.Namespace) -> None:
    started = time.time()
    _seed_everything(args.seed)
    import torch

    device = "cuda" if torch.cuda.is_available() and str(args.device) == "auto" else str(args.device)
    if device == "auto":
        device = "cpu"
    model = _load_dinov3_model(args, device)
    transform = _build_transform(args.input_size, args.crop_size)
    test_dataset = CsigZipDataset(
        args.test_zip,
        args.test_root_name,
        transform,
        include_metadata=True,
        category_filter=args.category_filter,
    )
    test_loader = _make_loader(test_dataset, args.infer_batch_size, shuffle=False, num_workers=args.num_workers)
    run_dir = Path(args.output_root) / args.run_name
    memory_paths = sorted((run_dir / "memory").glob("*.npz"))
    if not memory_paths:
        raise FileNotFoundError(f"No V180 memory banks found under {run_dir / 'memory'}")
    memories = {path.stem: _load_memory(path)[0] for path in memory_paths}
    if args.global_memory_path is not None:
        global_memory, _global_grid, _global_meta = _load_memory(args.global_memory_path)
        memories["__global__"] = global_memory
    calibration: dict[str, tuple[float, float, int]] = {}
    for category, memory in memories.items():
        calibration[category] = _calibrate_category(args, model, memory, category, device)
        print(f"calibration category={category} lo={calibration[category][0]:.6f} hi={calibration[category][1]:.6f}", flush=True)

    save_root = Path(args.evidence_root) / args.run_name / "maps_norm_i1"
    saved = 0
    with torch.no_grad():
        for images, categories, samples, views in test_loader:
            images = images.to(device, non_blocking=True)
            tokens, grid = _extract_patch_tokens(model, images, args.feature_layers)
            for item_index, category in enumerate(categories):
                category = str(category)
                memory = memories.get(category, memories.get("__global__"))
                if memory is None:
                    continue
                calibration_key = category if category in calibration else "__global__"
                distances = _distance_maps(tokens[item_index : item_index + 1], memory, args.memory_chunk_size)[0]
                raw = distances.reshape(grid)
                lo, hi, _seen = calibration.get(calibration_key, (float(np.percentile(raw, 95.0)), float(np.percentile(raw, 99.85)), 0))
                normalized = _normalize(raw, lo, hi, args.norm_power, args.blur_radius)
                resized = _resize_map(normalized, args.out_mask)
                _save_gray(save_root / category / str(samples[item_index]) / f"{views[item_index]}_mask.png", resized)
                saved += 1
            if saved % 500 == 0 or saved == len(test_dataset):
                print(f"infer saved={saved}/{len(test_dataset)}", flush=True)

    report = {
        "version": ROUTE_NAME,
        "mode": "infer-checkpoints",
        "status": "ok",
        "run_name": args.run_name,
        "checkpoint_iter": 1,
        "official_dinov3_repo": OFFICIAL_DINOV3_REPO,
        "saved_mask_count": int(saved),
        "test_view_count": int(len(test_dataset)),
        "test_root_name": str(args.test_root_name),
        "categories": sorted(memories),
        "calibration": {
            category: {"lo": values[0], "hi": values[1], "views": values[2]}
            for category, values in calibration.items()
        },
        "elapsed_minutes": float((time.time() - started) / 60.0),
    }
    (Path(args.evidence_root) / args.run_name / "infer_i1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def audit_zips(args: argparse.Namespace) -> None:
    train_records = index_csig_zip(args.train_zip, "Train")
    test_records = index_csig_zip(args.test_zip, "Test_A")
    report = {
        "version": ROUTE_NAME,
        "official_dinov3_repo": OFFICIAL_DINOV3_REPO,
        "official_dinomaly_repo": OFFICIAL_DINOMALY_REPO,
        "train_views": len(train_records),
        "test_views": len(test_records),
        "train_groups": len({record.group for record in train_records}),
        "test_groups": len({record.group for record in test_records}),
        "train_categories": sorted({record.category for record in train_records}),
        "test_categories": sorted({record.category for record in test_records}),
    }
    print(json.dumps(report, indent=2), flush=True)


def smoke_model(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)
    import torch

    device = "cuda" if torch.cuda.is_available() and str(args.device) == "auto" else str(args.device)
    if device == "auto":
        device = "cpu"
    model = _load_dinov3_model(args, device)
    with torch.no_grad():
        images = torch.randn(1, 3, int(args.crop_size), int(args.crop_size), device=device)
        tokens, grid = _extract_patch_tokens(model, images)
    report = {
        "version": ROUTE_NAME,
        "mode": "smoke-model",
        "status": "ok",
        "backend": str(args.backend),
        "official_dinov3_repo": OFFICIAL_DINOV3_REPO,
        "token_shape": list(tokens.shape),
        "grid": list(grid),
        "feature_layers": _parse_feature_layers(args.feature_layers),
        "device": device,
    }
    print(json.dumps(report, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V180 DINOv3/Dinomaly3 CSIG evidence adapter.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--run-name", required=True)
        command.add_argument("--seed", type=int, default=18001)
        command.add_argument("--backend", choices=["torchhub", "huggingface", "mock"], default="torchhub")
        command.add_argument("--dinov3-repo", default=str(ROOT / "DINOv3"))
        command.add_argument("--weights", default="")
        command.add_argument("--model-name", default="dinov3_vitb16")
        command.add_argument("--hf-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
        command.add_argument("--hf-token", default="")
        command.add_argument("--allow-mock", action="store_true")
        command.add_argument("--allow-official-download", action="store_true")
        command.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
        command.add_argument("--input-size", type=int, default=448)
        command.add_argument("--crop-size", type=int, default=448)
        command.add_argument("--feature-layers", default="")
        command.add_argument("--checkpoint-iters", default="1")
        command.add_argument("--train-zip", type=Path, default=TRAIN_ZIP)
        command.add_argument("--test-zip", type=Path, default=TEST_ZIP)
        command.add_argument("--test-root-name", default="Test_A")
        command.add_argument("--global-memory-path", type=Path, default=None)
        command.add_argument("--output-root", type=Path, default=EVIDENCE_ROOT)
        command.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
        command.add_argument("--num-workers", type=int, default=2)
        command.add_argument("--category-filter", default="")

    worker = sub.add_parser("worker")
    add_common(worker)
    worker.add_argument("--batch-size", type=int, default=2)
    worker.add_argument("--max-patches-per-category", type=int, default=50000)
    worker.add_argument("--log-interval", type=int, default=20)

    infer = sub.add_parser("infer-checkpoints")
    add_common(infer)
    infer.add_argument("--infer-batch-size", type=int, default=2)
    infer.add_argument("--out-mask", type=int, default=448)
    infer.add_argument("--calibration-max-views", type=int, default=220)
    infer.add_argument("--memory-chunk-size", type=int, default=8192)
    infer.add_argument("--norm-power", type=float, default=0.90)
    infer.add_argument("--blur-radius", type=float, default=0.8)

    smoke = sub.add_parser("smoke-model")
    add_common(smoke)

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
    if args.cmd == "smoke-model":
        smoke_model(args)
        return
    parser.error(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
