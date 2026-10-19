from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image


ROOT = Path.home() / "csig"
DEFAULT_EVIDENCE_ROOT = ROOT / "outputs" / "b榜_v276_dinov3_i448" / "evidence"
DEFAULT_SOURCES = (
    "b榜_v276_fold3_i80000:80000",
    "b榜_v276_v272_i100000:100000",
    "b榜_v276_v275_i080000:80000",
    "b榜_v276_v275_i100000:100000",
)


def fuse_maps(maps: Iterable[np.ndarray]) -> np.ndarray:
    values = [np.asarray(item, dtype=np.float32) for item in maps]
    if len(values) < 3:
        raise ValueError("At least three evidence maps are required for robust fusion")
    shape = values[0].shape
    if shape != (448, 448) or any(value.shape != shape for value in values):
        raise ValueError("All evidence maps must be 448x448")
    return np.median(np.stack(values, axis=0), axis=0).astype(np.float32)


def score_group(maps: Iterable[np.ndarray], ratio: float = 0.005) -> float:
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    view_scores: list[float] = []
    for item in maps:
        values = np.asarray(item, dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Group score maps must be finite and nonempty")
        count = max(1, int(math.ceil(values.size * float(ratio))))
        view_scores.append(float(np.partition(values, values.size - count)[-count:].mean()))
    ordered = np.sort(np.asarray(view_scores, dtype=np.float32))[::-1]
    top_two = float(ordered[: min(2, ordered.size)].mean())
    return float(0.55 * ordered[0] + 0.30 * top_two + 0.15 * ordered.mean())


def rank_percentiles(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values, key=lambda key: (float(values[key]), key))
    if len(ordered) == 1:
        return {ordered[0]: 0.5}
    return {key: index / (len(ordered) - 1) for index, key in enumerate(ordered)}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _test_index(test_zip: Path, root_name: str) -> dict[str, tuple[str, ...]]:
    groups: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(test_zip) as archive:
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if len(parts) != 4 or parts[0] != root_name or Path(parts[-1]).suffix.lower() != ".png":
                continue
            view = Path(parts[-1]).stem
            if view in {"0", "1", "2", "3", "4"}:
                groups[f"{parts[1]}/{parts[2]}"].add(view)
    result = {group: tuple(sorted(views, key=int)) for group, views in groups.items()}
    invalid = {group: views for group, views in result.items() if views != ("0", "1", "2", "3", "4")}
    if invalid:
        raise ValueError(f"Test archive has incomplete multiview groups: {sorted(invalid)[:5]}")
    if not result:
        raise ValueError(f"No {root_name} PNG groups found in {test_zip}")
    return dict(sorted(result.items()))


def _parse_sources(values: Iterable[str], evidence_root: Path) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    for value in values:
        run_name, separator, step_text = str(value).rpartition(":")
        if not separator or not run_name:
            raise ValueError(f"Invalid source {value!r}; expected RUN_NAME:STEP")
        step = int(step_text)
        path = Path(evidence_root) / run_name / f"float_maps_e{step}"
        if not path.is_dir():
            raise FileNotFoundError(path)
        output.append((run_name, path))
    if len(output) < 3:
        raise ValueError("At least three source routes are required")
    return output


def _load_map(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        result = np.asarray(data["calibrated_map"], dtype=np.float32).copy()
    if result.shape != (448, 448) or not np.isfinite(result).all():
        raise ValueError(f"Invalid calibrated map: {path}")
    return np.clip(result, 0.0, 1.0)


def _write_csv(path: Path, scores: Mapping[str, float]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_folder", "anomaly_score"])
        writer.writeheader()
        for group in sorted(scores):
            writer.writerow({"group_folder": group, "anomaly_score": f"{scores[group]:.8f}"})


def audit_submission_dir(package_dir: Path, expected_groups: set[str]) -> dict[str, Any]:
    package_dir = Path(package_dir)
    csv_path = package_dir / "submission.csv"
    rows: list[dict[str, str]] = []
    csv_error = ""
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["group_folder", "anomaly_score"]:
                raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
            rows = list(reader)
    except Exception as error:
        csv_error = str(error)
    groups = [row.get("group_folder", "") for row in rows]
    score_errors: list[str] = []
    for row in rows:
        try:
            score = float(row["anomaly_score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("outside [0, 1]")
        except Exception as error:
            score_errors.append(f"{row.get('group_folder', '')}: {error}")
    actual_masks = {
        str(path.relative_to(package_dir / "predicted_masks")).replace("\\", "/")
        for path in (package_dir / "predicted_masks").rglob("*_mask.png")
    }
    expected_masks = {
        f"{group}/{view}_mask.png" for group in expected_groups for view in range(5)
    }
    invalid_masks: list[str] = []
    for relative in sorted(actual_masks.intersection(expected_masks)):
        with Image.open(package_dir / "predicted_masks" / relative) as image:
            if image.mode != "L" or image.size != (448, 448):
                invalid_masks.append(relative)
    missing_masks = sorted(expected_masks.difference(actual_masks))
    extra_masks = sorted(actual_masks.difference(expected_masks))
    duplicate_groups = sorted({group for group in groups if groups.count(group) > 1})
    missing_groups = sorted(expected_groups.difference(groups))
    extra_groups = sorted(set(groups).difference(expected_groups))
    passed = not any((csv_error, score_errors, missing_masks, extra_masks, invalid_masks, duplicate_groups, missing_groups, extra_groups))
    return {
        "passed": passed,
        "csv_error": csv_error,
        "csv_rows": len(rows),
        "expected_groups": len(expected_groups),
        "duplicate_groups": duplicate_groups,
        "missing_groups": missing_groups,
        "extra_groups": extra_groups,
        "score_errors": score_errors,
        "mask_count": len(actual_masks),
        "expected_masks": len(expected_masks),
        "missing_masks": missing_masks,
        "extra_masks": extra_masks,
        "invalid_masks": invalid_masks,
    }


def _zip_package(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {zip_path}")
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir).as_posix())


def build_submission(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    zip_path = Path(args.zip_path)
    if output_dir.exists() or zip_path.exists():
        raise FileExistsError("Output directory or ZIP already exists; choose a new versioned target")
    groups = _test_index(Path(args.test_zip), args.test_root_name)
    sources = _parse_sources(args.source, Path(args.evidence_root))
    output_dir.mkdir(parents=True)
    mask_root = output_dir / "predicted_masks"
    raw_scores: dict[str, float] = {}
    for index, (group, views) in enumerate(groups.items(), start=1):
        category, sample = group.split("/", 1)
        fused_group: list[np.ndarray] = []
        for view in views:
            maps = [_load_map(root / category / sample / f"{view}.npz") for _, root in sources]
            fused = fuse_maps(maps)
            fused_group.append(fused)
            target = mask_root / category / sample / f"{view}_mask.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.rint(fused * 255.0).astype(np.uint8), mode="L").save(target)
        raw_scores[group] = score_group(fused_group, ratio=args.score_top_ratio)
        if index % 100 == 0:
            print(f"fused_groups={index}/{len(groups)}", flush=True)
    by_category: dict[str, dict[str, float]] = defaultdict(dict)
    for group, value in raw_scores.items():
        by_category[group.split("/", 1)[0]][group] = value
    scores: dict[str, float] = {}
    for values in by_category.values():
        scores.update(rank_percentiles(values))
    _write_csv(output_dir / "submission.csv", scores)
    audit = audit_submission_dir(output_dir, expected_groups=set(groups))
    _atomic_json(output_dir / "audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"Submission audit failed: {audit}")
    manifest = {
        "version": "b榜_dinov3_longrun_robust_median_v1",
        "test_zip": str(Path(args.test_zip)),
        "test_zip_sha256": _sha256(Path(args.test_zip)),
        "test_root_name": args.test_root_name,
        "source_routes": [{"run_name": name, "float_root": str(root)} for name, root in sources],
        "fusion": "per-pixel median of calibrated maps",
        "image_score": {"five_view_pool": "top_mean", "ratio": float(args.score_top_ratio), "normalization": "within_category_rank_percentile"},
        "groups": len(groups),
        "masks": len(groups) * 5,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    _zip_package(output_dir, zip_path)
    manifest["package_path"] = str(zip_path)
    manifest["package_sha256"] = _sha256(zip_path)
    manifest["package_bytes"] = zip_path.stat().st_size
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and audit one B-leaderboard DINOv3 submission package")
    parser.add_argument("--test-zip", type=Path, default=ROOT / "Test_B.zip")
    parser.add_argument("--test-root-name", default="Test_B")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--score-top-ratio", type=float, default=0.005)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.source:
        args.source = list(DEFAULT_SOURCES)
    print(json.dumps(build_submission(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
