from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


MIN_FREE_BYTES = 200 * 1024**3
EXPECTED_GPU_COUNT = 4
EXPECTED_TRAIN_VIEWS = 5000
EXPECTED_TEST_VIEWS = 3750
EXPECTED_INPUT_SIZE = 448
PATCH_SIZE = 16
ROOT = Path.home() / "csig"
DEFAULT_TRAIN_ZIP = ROOT / "Train.zip"
DEFAULT_TEST_ZIP = ROOT / "Test_A.zip"
DEFAULT_REPO = ROOT / "Dinomaly2_dinov3"
DEFAULT_WEIGHT_PATH = DEFAULT_REPO / "backbones" / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "v272_dinomaly2_dinov3_i448_ddp"
EXPECTED_WEIGHT_SHA256 = "c73f5f4ee2f3b4f7b51694615d6684336fc5ca6e783f57f834ede642c8c5f93d"
ROUTE_NAME = "v272_dinomaly2_dinov3_i448_ddp"


@dataclass(frozen=True)
class TrainingGeometry:
    dataset_size: int
    world_size: int
    per_gpu_batch: int
    accumulation_steps: int
    target_optimizer_steps: int
    effective_batch: int
    local_samples_per_epoch: int
    local_micro_batches_per_epoch: int
    required_micro_batches: int
    full_epochs: int
    final_epoch_micro_batches: int


def compute_training_geometry(
    *,
    dataset_size: int,
    world_size: int,
    per_gpu_batch: int,
    accumulation_steps: int,
    target_optimizer_steps: int,
) -> TrainingGeometry:
    values = {
        "dataset_size": dataset_size,
        "world_size": world_size,
        "per_gpu_batch": per_gpu_batch,
        "accumulation_steps": accumulation_steps,
        "target_optimizer_steps": target_optimizer_steps,
    }
    for name, value in values.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    local_samples = int(math.ceil(int(dataset_size) / int(world_size)))
    local_micro_batches = int(math.ceil(local_samples / int(per_gpu_batch)))
    required_micro_batches = int(target_optimizer_steps) * int(accumulation_steps)
    full_epochs, final_epoch_micro_batches = divmod(required_micro_batches, local_micro_batches)
    return TrainingGeometry(
        dataset_size=int(dataset_size),
        world_size=int(world_size),
        per_gpu_batch=int(per_gpu_batch),
        accumulation_steps=int(accumulation_steps),
        target_optimizer_steps=int(target_optimizer_steps),
        effective_batch=int(world_size) * int(per_gpu_batch) * int(accumulation_steps),
        local_samples_per_epoch=local_samples,
        local_micro_batches_per_epoch=local_micro_batches,
        required_micro_batches=required_micro_batches,
        full_epochs=full_epochs,
        final_epoch_micro_batches=final_epoch_micro_batches,
    )


def validate_patch_alignment(input_size: int, patch_size: int = PATCH_SIZE) -> int:
    if int(input_size) <= 0 or int(patch_size) <= 0:
        raise ValueError("input_size and patch_size must be positive")
    if int(input_size) % int(patch_size) != 0:
        raise ValueError(f"input_size={input_size} must be divisible by patch_size={patch_size}")
    return int(input_size) // int(patch_size)


def evaluate_preflight(
    *,
    free_bytes: int,
    gpu_count: int,
    weight_sha256: str,
    expected_weight_sha256: str,
    train_views: int,
    test_views: int,
    input_size: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    if int(free_bytes) < MIN_FREE_BYTES:
        blockers.append("free_space_below_200_gib")
    if int(gpu_count) != EXPECTED_GPU_COUNT:
        blockers.append("expected_four_cuda_devices")
    if str(weight_sha256).lower() != str(expected_weight_sha256).lower():
        blockers.append("dinov3_weight_hash_mismatch")
    if int(train_views) != EXPECTED_TRAIN_VIEWS:
        blockers.append("unexpected_train_view_count")
    if int(test_views) != EXPECTED_TEST_VIEWS:
        blockers.append("unexpected_test_view_count")
    if int(input_size) != EXPECTED_INPUT_SIZE:
        blockers.append("input_size_must_be_448")
    try:
        patch_grid = validate_patch_alignment(input_size, PATCH_SIZE)
    except ValueError:
        patch_grid = None
        blockers.append("input_size_not_patch_aligned")

    return {
        "ready": not blockers,
        "blockers": blockers,
        "free_bytes": int(free_bytes),
        "minimum_free_bytes": MIN_FREE_BYTES,
        "gpu_count": int(gpu_count),
        "expected_gpu_count": EXPECTED_GPU_COUNT,
        "weight_sha256": str(weight_sha256).lower(),
        "expected_weight_sha256": str(expected_weight_sha256).lower(),
        "train_views": int(train_views),
        "test_views": int(test_views),
        "input_size": int(input_size),
        "patch_size": PATCH_SIZE,
        "patch_grid": patch_grid,
    }


def geometry_as_dict(geometry: TrainingGeometry) -> dict[str, int]:
    return {key: int(value) for key, value in asdict(geometry).items()}


def should_optimizer_step(micro_step: int, accumulation_steps: int) -> bool:
    if int(micro_step) <= 0:
        raise ValueError("micro_step must be positive")
    if int(accumulation_steps) <= 0:
        raise ValueError("accumulation_steps must be positive")
    return int(micro_step) % int(accumulation_steps) == 0


def parse_checkpoint_steps(value: str, target_optimizer_steps: int) -> tuple[int, ...]:
    target = int(target_optimizer_steps)
    if target <= 0:
        raise ValueError("target_optimizer_steps must be positive")
    try:
        steps = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"checkpoint steps must be integers: {value}") from exc
    if not steps:
        raise ValueError("checkpoint steps cannot be empty")
    if any(step <= 0 for step in steps):
        raise ValueError("checkpoint steps must be positive")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("checkpoint steps must be unique and strictly increasing")
    if steps[-1] > target:
        raise ValueError(f"checkpoint step {steps[-1]} exceeds target {target}")
    if target not in steps:
        raise ValueError(f"checkpoint steps must include target {target}")
    return steps


def validate_resume_state(
    state: Mapping[str, Any],
    expected_immutable_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    required = {
        "optimizer_step",
        "micro_step",
        "epoch",
        "micro_in_epoch",
        "immutable_config",
        "model",
        "optimizer",
        "scheduler",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"resume state is missing keys: {', '.join(missing)}")
    actual_config = state["immutable_config"]
    if not isinstance(actual_config, Mapping):
        raise ValueError("resume immutable_config must be a mapping")
    for key, expected in expected_immutable_config.items():
        actual = actual_config.get(key)
        if actual != expected:
            raise ValueError(f"resume {key} mismatch: checkpoint={actual!r} expected={expected!r}")
    optimizer_step = int(state["optimizer_step"])
    micro_step = int(state["micro_step"])
    epoch = int(state["epoch"])
    micro_in_epoch = int(state["micro_in_epoch"])
    if min(optimizer_step, micro_step, epoch, micro_in_epoch) < 0:
        raise ValueError("resume counters cannot be negative")
    accumulation_steps = int(expected_immutable_config.get("accumulation_steps", 1))
    if micro_step != optimizer_step * accumulation_steps:
        raise ValueError(
            "resume micro_step must equal optimizer_step * accumulation_steps at checkpoint boundaries"
        )
    return state


def state_checkpoint_path(checkpoint_dir: Path, optimizer_step: int) -> Path:
    return Path(checkpoint_dir) / f"state_i{int(optimizer_step):06d}.pt"


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def _count_csig_views(path: Path, root_name: str) -> int:
    count = 0
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith("/") or not name.lower().endswith(".png"):
                continue
            parts = PurePosixPath(name).parts
            if len(parts) != 4 or parts[0] != str(root_name):
                continue
            if Path(parts[-1]).stem in {"0", "1", "2", "3", "4"}:
                count += 1
    return count


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _build_preflight_report(args: argparse.Namespace) -> dict[str, Any]:
    train_zip = Path(args.train_zip)
    test_zip = Path(args.test_zip)
    weight_path = Path(args.weight_path)
    output = Path(args.output)
    observed_errors: list[str] = []

    try:
        free_bytes = int(shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free)
    except OSError as exc:
        free_bytes = 0
        observed_errors.append(f"disk_usage_failed:{exc}")

    try:
        import torch

        gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda)
        gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
    except Exception as exc:
        gpu_count = 0
        torch_version = "unavailable"
        cuda_version = "unavailable"
        gpu_names = []
        observed_errors.append(f"torch_cuda_probe_failed:{exc}")

    try:
        train_views = _count_csig_views(train_zip, "Train")
    except Exception as exc:
        train_views = 0
        observed_errors.append(f"train_zip_audit_failed:{exc}")
    try:
        test_views = _count_csig_views(test_zip, "Test_A")
    except Exception as exc:
        test_views = 0
        observed_errors.append(f"test_zip_audit_failed:{exc}")
    try:
        weight_sha256 = _sha256_file(weight_path)
    except Exception as exc:
        weight_sha256 = ""
        observed_errors.append(f"weight_hash_failed:{exc}")

    report = evaluate_preflight(
        free_bytes=free_bytes,
        gpu_count=gpu_count,
        weight_sha256=weight_sha256,
        expected_weight_sha256=args.expected_weight_sha256,
        train_views=train_views,
        test_views=test_views,
        input_size=args.input_size,
    )
    report.update(
        {
            "version": ROUTE_NAME,
            "status": "ready" if report["ready"] and not observed_errors else "blocked",
            "observed_errors": observed_errors,
            "train_zip": str(train_zip),
            "test_zip": str(test_zip),
            "weight_path": str(weight_path),
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "gpu_names": gpu_names,
            "created_unix": time.time(),
        }
    )
    if observed_errors:
        report["ready"] = False
        report["blockers"] = [*report["blockers"], *observed_errors]
    return report


def run_preflight(args: argparse.Namespace) -> None:
    report = _build_preflight_report(args)
    _atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["ready"]:
        raise SystemExit(2)


def _import_adapter() -> Any:
    try:
        from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    except ImportError:
        import v182_dinomaly2_dinov3_csig_official as adapter
    return adapter


def _resolve_resume_checkpoint(checkpoint_dir: Path, resume: str, target_step: int) -> Path | None:
    value = str(resume).strip()
    if value.lower() in {"", "none", "off", "false"}:
        return None
    if value.lower() != "auto":
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return path
    candidates: list[tuple[int, Path]] = []
    for path in Path(checkpoint_dir).glob("state_i*.pt"):
        try:
            step = int(path.stem.split("_i", 1)[1])
        except (IndexError, ValueError):
            continue
        if step <= int(target_step):
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _optimizer_to_device(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


def freeze_non_trainable_parameters(model: Any, trainable: Any) -> dict[str, int]:
    trainable_ids = {id(parameter) for parameter in trainable.parameters()}
    all_count = 0
    trainable_count = 0
    frozen_count = 0
    for parameter in model.parameters():
        all_count += 1
        enabled = id(parameter) in trainable_ids
        parameter.requires_grad_(enabled)
        if enabled:
            trainable_count += 1
        else:
            frozen_count += 1
    if trainable_count != len(trainable_ids):
        raise RuntimeError(
            f"Official optimizer contains {len(trainable_ids)} parameters but only {trainable_count} belong to model"
        )
    if trainable_count == 0:
        raise RuntimeError("Official optimizer parameter set is empty")
    return {"all": all_count, "trainable": trainable_count, "frozen": frozen_count}


def _capture_rng_state(torch: Any, np: Any, local_rank: int) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(local_rank),
    }


def _restore_rng_state(torch: Any, np: Any, local_rank: int, state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], local_rank)


def select_rank_rng_state(state: Mapping[str, Any], rank: int) -> Mapping[str, Any]:
    states = state.get("rng_by_rank")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise ValueError("checkpoint does not contain rng_by_rank")
    if int(rank) < 0 or int(rank) >= len(states):
        raise ValueError(f"checkpoint has no RNG state for rank {rank}")
    selected = states[int(rank)]
    if not isinstance(selected, Mapping):
        raise ValueError(f"checkpoint RNG state for rank {rank} is invalid")
    return selected


def _atomic_torch_save(torch: Any, payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _canonical_resume_position(epoch: int, micro_in_epoch: int, loader_length: int) -> tuple[int, int]:
    if int(micro_in_epoch) >= int(loader_length):
        return int(epoch) + 1, 0
    return int(epoch), int(micro_in_epoch)


def _immutable_training_config(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    return {
        "route": ROUTE_NAME,
        "world_size": int(world_size),
        "per_gpu_batch": int(args.per_gpu_batch),
        "accumulation_steps": int(args.accumulation_steps),
        "input_size": int(args.input_size),
        "crop_size": int(args.crop_size),
        "patch_size": PATCH_SIZE,
        "backbone": str(args.backbone),
        "dropout": float(args.dropout),
        "reduce_dim": int(args.reduce_dim),
        "target_optimizer_steps": int(args.target_optimizer_steps),
        "lr": float(args.lr),
        "final_lr_ratio": float(args.final_lr_ratio),
        "weight_decay": float(args.weight_decay),
        "percent_max": float(args.percent_max),
        "percent_warmup": int(args.percent_warmup),
        "loss_factor": float(args.loss_factor),
        "category_filter": str(args.category_filter),
        "weight_sha256": str(args.expected_weight_sha256).lower(),
    }


def run_train_ddp(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if min(local_rank, rank, world_size) < 0:
        raise RuntimeError("train-ddp must be launched with torchrun")
    if world_size != int(args.expected_world_size):
        raise RuntimeError(f"WORLD_SIZE={world_size}, expected {args.expected_world_size}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError(f"CUDA device count {torch.cuda.device_count()} is below world size {world_size}")
    if int(args.input_size) != int(args.crop_size):
        raise ValueError("V272 requires input_size == crop_size")
    validate_patch_alignment(args.input_size, PATCH_SIZE)
    checkpoint_steps = parse_checkpoint_steps(args.checkpoint_steps, args.target_optimizer_steps)

    initialized = False
    try:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        initialized = True
        adapter = _import_adapter()
        adapter._seed_everything(args.seed)
        torch.backends.cudnn.benchmark = True

        transform = adapter._build_transform(args.input_size, args.crop_size)
        dataset = adapter.CsigZipDataset(
            Path(args.train_zip),
            "Train",
            transform,
            category_filter=args.category_filter,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(args.seed),
            drop_last=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.per_gpu_batch),
            sampler=sampler,
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
            drop_last=False,
            persistent_workers=int(args.num_workers) > 0,
        )
        geometry = compute_training_geometry(
            dataset_size=len(dataset),
            world_size=world_size,
            per_gpu_batch=args.per_gpu_batch,
            accumulation_steps=args.accumulation_steps,
            target_optimizer_steps=args.target_optimizer_steps,
        )
        if len(loader) != geometry.local_micro_batches_per_epoch:
            raise RuntimeError(
                f"DataLoader length {len(loader)} differs from geometry {geometry.local_micro_batches_per_epoch}"
            )

        run_dir = Path(args.output_root) / str(args.run_name)
        checkpoint_dir = run_dir / "checkpoints"
        if rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        deps = adapter._import_official_modules()
        model, trainable = adapter._build_model(
            deps=deps,
            backbone=args.backbone,
            dropout=args.dropout,
            reduce_dim=args.reduce_dim,
            context_aware_recenter=not args.disable_context,
            mask_neighbor_size=args.mask_neighbor_size,
            weights_dir=Path(args.weight_path).parent,
        )
        parameter_counts = freeze_non_trainable_parameters(model, trainable)
        model = model.to(device)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        adapter._seed_everything(args.seed + rank)
        optimizer = deps["StableAdamW"](
            [{"params": trainable.parameters()}],
            lr=float(args.lr),
            betas=(0.9, 0.999),
            weight_decay=float(args.weight_decay),
            amsgrad=False,
            eps=1e-10,
        )
        scheduler = deps["WarmupCosineScheduler"](
            optimizer,
            final_ratio=float(args.final_lr_ratio),
            total_epochs=int(args.target_optimizer_steps),
            warmup_epochs=min(100, max(10, int(args.target_optimizer_steps) // 100)),
        )
        immutable_config = _immutable_training_config(args, world_size)

        optimizer_step = 0
        micro_step = 0
        epoch = 0
        resume_micro_in_epoch = 0
        resume_path = _resolve_resume_checkpoint(checkpoint_dir, args.resume, args.target_optimizer_steps)
        if resume_path is not None:
            state = torch.load(resume_path, map_location="cpu")
            validate_resume_state(state, immutable_config)
            ddp_model.module.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            _optimizer_to_device(optimizer, device)
            scheduler.load_state_dict(state["scheduler"])
            optimizer_step = int(state["optimizer_step"])
            micro_step = int(state["micro_step"])
            epoch = int(state["epoch"])
            resume_micro_in_epoch = int(state["micro_in_epoch"])
            if "rng_by_rank" in state:
                _restore_rng_state(torch, np, local_rank, select_rank_rng_state(state, rank))
            elif "rng" in state:
                _restore_rng_state(torch, np, local_rank, state["rng"])
            if rank == 0:
                print(f"RESUME path={resume_path} optimizer_step={optimizer_step}", flush=True)

        if optimizer_step >= int(args.target_optimizer_steps):
            if rank == 0:
                print(f"COMPLETE checkpoint already reached target={args.target_optimizer_steps}", flush=True)
                _atomic_json(
                    run_dir / "train_report.json",
                    {
                        "version": ROUTE_NAME,
                        "status": "ok",
                        "resume_complete": True,
                        "run_name": str(args.run_name),
                        "optimizer_step": optimizer_step,
                        "micro_step": micro_step,
                        "epoch": epoch,
                        "immutable_config": immutable_config,
                        "geometry": geometry_as_dict(geometry),
                    },
                )
            dist.barrier()
            return

        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        accumulated_losses: list[float] = []
        if rank == 0:
            print(
                "START "
                f"run={args.run_name} world_size={world_size} input={args.input_size} "
                f"patch_grid={validate_patch_alignment(args.input_size)} dataset={len(dataset)} "
                f"effective_batch={geometry.effective_batch} target_steps={args.target_optimizer_steps} "
                f"parameter_counts={json.dumps(parameter_counts, sort_keys=True)} "
                f"geometry={json.dumps(geometry_as_dict(geometry), sort_keys=True)}",
                flush=True,
            )

        while optimizer_step < int(args.target_optimizer_steps):
            sampler.set_epoch(epoch)
            ddp_model.train()
            skip_before = resume_micro_in_epoch
            resume_micro_in_epoch = 0
            for batch_index, (images, _labels) in enumerate(loader):
                if batch_index < skip_before:
                    continue
                images = images.to(device, non_blocking=True)
                next_micro_step = micro_step + 1
                sync_gradients = should_optimizer_step(next_micro_step, args.accumulation_steps)
                sync_context = contextlib.nullcontext() if sync_gradients else ddp_model.no_sync()
                with sync_context:
                    encoded, decoded = ddp_model(images)
                    percent = min(
                        float(args.percent_max)
                        * optimizer_step
                        / max(1, int(args.percent_warmup)),
                        float(args.percent_max),
                    )
                    loss = deps["global_cosine_hm_percent"](
                        encoded,
                        decoded,
                        p=percent,
                        factor=float(args.loss_factor),
                    )
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(
                            f"Non-finite loss at rank={rank} optimizer_step={optimizer_step} micro_step={next_micro_step}"
                        )
                    (loss / int(args.accumulation_steps)).backward()
                micro_step = next_micro_step
                accumulated_losses.append(float(loss.detach().item()))

                if not sync_gradients:
                    continue

                deps["nn"].utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                local_mean = sum(accumulated_losses) / max(1, len(accumulated_losses))
                accumulated_losses.clear()
                reduced_loss = torch.tensor(local_mean, device=device, dtype=torch.float64)
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss /= world_size
                if rank == 0 and (
                    optimizer_step <= 5
                    or optimizer_step % int(args.log_every) == 0
                    or optimizer_step in checkpoint_steps
                ):
                    print(
                        f"step {optimizer_step}/{args.target_optimizer_steps} epoch={epoch} "
                        f"micro_in_epoch={batch_index + 1}/{len(loader)} "
                        f"loss={float(reduced_loss.item()):.6f} "
                        f"elapsed_minutes={(time.time() - started) / 60.0:.2f} "
                        f"gpu_max_mib={torch.cuda.max_memory_allocated(local_rank) / 1024**2:.1f}",
                        flush=True,
                    )

                if optimizer_step in checkpoint_steps:
                    dist.barrier()
                    local_rng_state = _capture_rng_state(torch, np, local_rank)
                    rng_by_rank: list[Any] | None = [None] * world_size if rank == 0 else None
                    dist.gather_object(local_rng_state, rng_by_rank, dst=0)
                    if rank == 0:
                        next_epoch, next_micro_in_epoch = _canonical_resume_position(
                            epoch, batch_index + 1, len(loader)
                        )
                        payload = {
                            "version": ROUTE_NAME,
                            "optimizer_step": optimizer_step,
                            "micro_step": micro_step,
                            "epoch": next_epoch,
                            "micro_in_epoch": next_micro_in_epoch,
                            "immutable_config": immutable_config,
                            "geometry": geometry_as_dict(geometry),
                            "model": ddp_model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "rng_by_rank": rng_by_rank,
                            "created_unix": time.time(),
                        }
                        checkpoint_path = state_checkpoint_path(checkpoint_dir, optimizer_step)
                        _atomic_torch_save(torch, payload, checkpoint_path)
                        print(f"CHECKPOINT {checkpoint_path}", flush=True)
                    dist.barrier()

                if optimizer_step >= int(args.target_optimizer_steps):
                    break
            epoch += 1

        dist.barrier()
        if rank == 0:
            report = {
                "version": ROUTE_NAME,
                "status": "ok",
                "run_name": str(args.run_name),
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "epoch": epoch,
                "elapsed_minutes": (time.time() - started) / 60.0,
                "immutable_config": immutable_config,
                "geometry": geometry_as_dict(geometry),
                "parameter_counts": parameter_counts,
                "checkpoints": [
                    str(state_checkpoint_path(checkpoint_dir, step)) for step in checkpoint_steps
                ],
            }
            _atomic_json(run_dir / "train_report.json", report)
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V272 patch-aligned Dinomaly2 + DINOv3 DDP route")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    preflight.add_argument("--test-zip", type=Path, default=DEFAULT_TEST_ZIP)
    preflight.add_argument("--weight-path", type=Path, default=DEFAULT_WEIGHT_PATH)
    preflight.add_argument("--expected-weight-sha256", default=EXPECTED_WEIGHT_SHA256)
    preflight.add_argument("--input-size", type=int, default=EXPECTED_INPUT_SIZE)
    preflight.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train-ddp")
    train.add_argument("--run-name", required=True)
    train.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    train.add_argument("--weight-path", type=Path, default=DEFAULT_WEIGHT_PATH)
    train.add_argument("--expected-weight-sha256", default=EXPECTED_WEIGHT_SHA256)
    train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "evidence")
    train.add_argument("--backbone", default="dinov3_vit_base_16")
    train.add_argument("--input-size", type=int, default=EXPECTED_INPUT_SIZE)
    train.add_argument("--crop-size", type=int, default=EXPECTED_INPUT_SIZE)
    train.add_argument("--dropout", type=float, default=0.4)
    train.add_argument("--reduce-dim", type=int, default=256)
    train.add_argument("--mask-neighbor-size", type=int, default=0)
    train.add_argument("--disable-context", action="store_true")
    train.add_argument("--per-gpu-batch", type=int, default=1)
    train.add_argument("--accumulation-steps", type=int, default=4)
    train.add_argument("--expected-world-size", type=int, default=4)
    train.add_argument("--target-optimizer-steps", type=int, default=100000)
    train.add_argument("--checkpoint-steps", default="20000,40000,60000,80000,100000")
    train.add_argument("--lr", type=float, default=0.002)
    train.add_argument("--final-lr-ratio", type=float, default=1.0)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--percent-max", type=float, default=0.90)
    train.add_argument("--percent-warmup", type=int, default=1000)
    train.add_argument("--loss-factor", type=float, default=0.10)
    train.add_argument("--category-filter", default="")
    train.add_argument("--seed", type=int, default=27201)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--log-every", type=int, default=100)
    train.add_argument("--resume", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        run_preflight(args)
        return
    if args.command == "train-ddp":
        run_train_ddp(args)
        return
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
