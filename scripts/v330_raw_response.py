"""Raw supervised responses and the shared, non-clipping export contract."""
from __future__ import annotations

import json
import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.v283_supervised_dinomaly_longtrain import sha256_file, top_mean

SCHEMA = 'v330_raw_map_v2'
HALF_RESPONSE = 0.10  # Same reference response as the supervised training logits.


def checked_raw(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if not raw.size or not np.isfinite(raw).all() or np.min(raw) < -1e-6 or np.max(raw) > 2.000001:
        raise ValueError('Raw mean cosine distances must be finite and within [0, 2]')
    return np.maximum(raw, 0.0)


def encode_response(values: np.ndarray) -> np.ndarray:
    raw = checked_raw(values)
    return raw / (raw + HALF_RESPONSE)


def encode_png(raw_map: np.ndarray) -> np.ndarray:
    if np.shape(raw_map) != (448, 448):
        raise ValueError('Expected a 448x448 raw map')
    return np.rint(255.0 * encode_response(raw_map)).astype(np.uint8)


def score_views(raw_view_scores: Sequence[float]) -> float:
    if len(raw_view_scores) != 5:
        raise ValueError('Expected exactly five raw view scores')
    values = np.sort(checked_raw(np.asarray(raw_view_scores)))
    return float(encode_response(values[-2:].mean()))


def audit_scores(scores: Mapping[str, float]) -> dict[str, Any]:
    categories: dict[str, list[float]] = defaultdict(list)
    for group, score in scores.items():
        categories[group.split('/', 1)[0]].append(float(score))
    values = np.asarray(list(scores.values()), dtype=np.float64)
    all_tied = sorted(cat for cat, group in categories.items() if len(group) > 1 and len(set(group)) == 1)
    valid = bool(values.size and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)))
    ones = int(np.sum(values == 1.0))
    return {
        'passed': valid and not all_tied and ones == 0,
        'rows': int(values.size), 'unique': len(set(values.tolist())),
        'exact_ones': ones, 'all_tied_categories': all_tied,
        'minimum': float(values.min()) if values.size else None,
        'maximum': float(values.max()) if values.size else None,
    }


def histogram_metrics(hist: np.ndarray) -> dict[str, float]:
    values = np.asarray(hist, dtype=np.float64)
    if values.shape != (2, 256) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Expected finite nonnegative 2x256 pixel histograms')
    negative, positive = values
    n, p = negative.sum(), positive.sum()
    if n <= 0 or p <= 0:
        raise ValueError('Pixel metrics require both labels')
    tp, fp = np.cumsum(positive[::-1]), np.cumsum(negative[::-1])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / p
    auc = np.sum(positive * (np.cumsum(negative) - 0.5 * negative)) / (p * n)
    ap = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    return {
        'auroc': float(auc), 'ap': float(ap),
        'best_f1': float(np.max(2 * tp / np.maximum(p + tp + fp, 1))),
        'positive_fraction': float(p / (n + p)),
    }


def current_pipeline_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    names = ('v330_raw_response.py', 'v330_replay.py', 'build_v330_raw_b_submission.py',
             'v283_supervised_dinomaly_longtrain.py', 'v182_dinomaly2_dinov3_csig_official.py')
    return {name: sha256_file(directory / name) for name in names}


def record_contract_id(contract: Mapping[str, Any]) -> str:
    keys = ('fold_id', 'step', 'checkpoint_sha256', 'score_scale', 'pipeline_hashes', 'test_zip_sha256')
    payload = {key: contract[key] for key in keys}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_record(path: Path, raw_map: np.ndarray, *, category: str, sample: str, view: str, fold_id: int, step: int,
                contract_id: str) -> None:
    raw = checked_raw(raw_map).astype(np.float32)
    if raw.shape != (448, 448):
        raise ValueError('Expected a 448x448 raw map')
    if not contract_id:
        raise ValueError('A nonempty raw evidence contract is required')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('xb') as handle:
        np.savez_compressed(handle, raw_map=raw, schema=np.asarray(SCHEMA), category=np.asarray(category),
                            sample=np.asarray(sample), view=np.asarray(view), fold_id=np.int8(fold_id),
                            step=np.int64(step), contract_id=np.asarray(contract_id), raw_top005=np.float64(top_mean(raw, 0.005)))
    os.replace(temporary, path)


def load_record(path: Path, *, category: str, sample: str, view: str, fold_id: int,
                expected_step: int, expected_contract_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if 'raw_map' not in data or 'schema' not in data or str(data['schema']) != SCHEMA:
            raise ValueError(f'Missing untruncated raw evidence: {path}')
        if any(str(data[key]) != str(value) for key, value in
               {'category': category, 'sample': sample, 'view': view, 'fold_id': fold_id,
                'step': expected_step, 'contract_id': expected_contract_id}.items()):
            raise ValueError(f'Raw evidence identity mismatch: {path}')
        result = checked_raw(data['raw_map']).astype(np.float32)
    if result.shape != (448, 448):
        raise ValueError(f'Invalid raw shape: {path}')
    return result


def selected_contract(route_root: Path, fold_id: int) -> dict[str, Any]:
    gate_path = Path(route_root) / 'route_gate.json'
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    if gate.get('passed') is not True:
        raise ValueError('Source model selection gate did not pass')
    matches = [s for s in gate['selected_reports'] if s['fold_id'] == fold_id]
    if len(matches) != 1:
        raise ValueError('Expected one selected fold')
    selected = matches[0]
    report_path = Path(selected['source_root']) / f'fold{fold_id}' / f"validation_i{selected['step']:06d}.json"
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('eligible') is not True or any(report[k] != selected[k] for k in ('fold_id', 'step', 'checkpoint', 'held_out_categories')):
        raise ValueError('Selected validation identity mismatch')
    if 'calibration' in report:
        calibration = report['calibration']
        if not isinstance(calibration, Mapping) or not {'score_scale', 'source_checkpoint_sha256'} <= calibration.keys():
            raise ValueError('Incomplete calibration identity')
        scale = float(calibration['score_scale'])
        expected_hash = str(calibration['source_checkpoint_sha256'])
        if len(expected_hash) != 64 or any(c not in '0123456789abcdef' for c in expected_hash):
            raise ValueError('Invalid calibration checkpoint hash')
    else:
        scale = 1.0
        expected_hash = sha256_file(Path(selected['checkpoint']))
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('Invalid selected score_scale')
    return {
        **selected, 'score_scale': scale, 'validation_report': str(report_path),
        'validation_sha256': sha256_file(report_path), 'gate_sha256': sha256_file(gate_path),
        'expected_checkpoint_sha256': expected_hash,
        'map_method': 'trainer.differentiable_anomaly_map', 'encoding': 'raw/(raw+0.10)',
    }
