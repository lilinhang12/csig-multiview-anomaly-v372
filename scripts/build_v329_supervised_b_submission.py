from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

try:
    from scripts.build_b_submission import _test_index, audit_submission_dir
except ModuleNotFoundError:
    from build_b_submission import _test_index, audit_submission_dir


ROOT = Path.home() / "csig"
DEFAULT_EVIDENCE_ROOT = (
    ROOT / "outputs" / "b榜_v326_supervised" / "evidence" / "ensemble" / "float_maps"
)
DEFAULT_GATE_PATH = ROOT / "outputs" / "v323_monotonic_calibration" / "route_gate.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _top_mean(values: np.ndarray, ratio: float) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("Score map must be finite and nonempty")
    count = max(1, int(math.ceil(flat.size * float(ratio))))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def score_group(maps: Iterable[np.ndarray], ratio: float = 0.005) -> float:
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    view_scores = []
    for item in maps:
        values = np.clip(np.asarray(item, dtype=np.float32), 0.0, 1.0)
        view_scores.append(
            0.60 * _top_mean(values, min(float(ratio), 0.001))
            + 0.30 * _top_mean(values, float(ratio))
            + 0.10 * float(values.max())
        )
    if len(view_scores) != 5:
        raise ValueError(f"Expected exactly five views, found {len(view_scores)}")
    ordered = np.sort(np.asarray(view_scores, dtype=np.float64))[::-1]
    score = 0.60 * ordered[0] + 0.25 * ordered[:2].mean() + 0.15 * ordered.mean()
    return float(np.clip(score, 0.0, 1.0))


def _load_map(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        values = np.asarray(payload["calibrated_map"], dtype=np.float32).copy()
    if values.shape != (448, 448) or not np.isfinite(values).all():
        raise ValueError(f"Invalid calibrated map: {path}")
    return np.clip(values, 0.0, 1.0)


def _write_csv(path: Path, scores: Mapping[str, float]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_folder", "anomaly_score"])
        for group in sorted(scores):
            writer.writerow([group, f"{float(scores[group]):.10f}"])


def _write_clean_zip(package_dir: Path, zip_path: Path) -> int:
    members = [package_dir / "submission.csv", *sorted((package_dir / "predicted_masks").rglob("*_mask.png"))]
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            archive.write(path, path.relative_to(package_dir).as_posix())
    return len(members)


def build_submission(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    zip_path = Path(args.zip_path)
    if output_dir.exists() or zip_path.exists():
        raise FileExistsError("Output directory or ZIP already exists; choose a versioned target")

    gate_path = Path(args.gate_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not bool(gate.get("passed", False)):
        raise RuntimeError(f"Supervised validation gate did not pass: {gate_path}")

    groups = _test_index(Path(args.test_zip), str(args.test_root_name))
    evidence_root = Path(args.evidence_root)
    expected_paths = [
        evidence_root / group / f"{view}.npz"
        for group, views in groups.items()
        for view in views
    ]
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing supervised evidence maps: {missing[:5]}")

    output_dir.mkdir(parents=True)
    mask_root = output_dir / "predicted_masks"
    scores: dict[str, float] = {}
    for index, (group, views) in enumerate(groups.items(), start=1):
        maps = []
        for view in views:
            values = _load_map(evidence_root / group / f"{view}.npz")
            maps.append(values)
            target = mask_root / group / f"{view}_mask.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.rint(values * 255.0).astype(np.uint8), mode="L").save(target)
        scores[group] = score_group(maps, ratio=float(args.score_top_ratio))
        if index % 100 == 0 or index == len(groups):
            print(f"built_groups={index}/{len(groups)}", flush=True)

    _write_csv(output_dir / "submission.csv", scores)
    audit = audit_submission_dir(output_dir, expected_groups=set(groups))
    _atomic_json(output_dir / "audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"Submission directory audit failed: {audit}")

    zip_entries = _write_clean_zip(output_dir, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP integrity check failed at {bad_member}")
    manifest = {
        "version": "b榜_v329_supervised_continuous_v1",
        "development_gate": str(gate_path),
        "development_gate_sha256": _sha256(gate_path),
        "test_zip": str(Path(args.test_zip)),
        "test_zip_sha256": _sha256(Path(args.test_zip)),
        "test_root_name": str(args.test_root_name),
        "evidence_root": str(evidence_root),
        "fusion": "median ensemble of four validation-selected supervised Dinomaly folds",
        "image_score": {
            "five_view_pool": "0.60*max + 0.25*top2_mean + 0.15*mean",
            "view_pool": "0.60*top0.1% + 0.30*top0.5% + 0.10*max",
            "normalization": "continuous_raw_multiview",
            "ratio": float(args.score_top_ratio),
        },
        "groups": len(groups),
        "masks": len(expected_paths),
        "zip_entries": zip_entries,
        "package_path": str(zip_path),
        "package_sha256": _sha256(zip_path),
        "package_bytes": zip_path.stat().st_size,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the validation-gated supervised B榜 package")
    parser.add_argument("--test-zip", type=Path, default=ROOT / "Test_B.zip")
    parser.add_argument("--test-root-name", default="Test_B")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--gate-path", type=Path, default=DEFAULT_GATE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--score-top-ratio", type=float, default=0.005)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(build_submission(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
