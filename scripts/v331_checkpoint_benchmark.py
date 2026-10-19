"""Compare existing checkpoints on full-pixel category-held-out development data."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts.v330_raw_response import checked_raw
from scripts.v330_replay import ExportMetrics, development_rows, load_model

ROOT = Path.home() / 'csig'
FIELDS = (('image', 'auroc'), ('image', 'ap'), ('pixel', 'auroc'), ('pixel', 'ap'), ('pixel', 'best_f1'))


def _matrix(metrics: dict[str, Any], categories: list[str]) -> np.ndarray:
    if set(metrics['per_category']) != set(categories):
        raise ValueError('Development category coverage changed')
    matrix = np.asarray([[metrics['per_category'][cat][level][key] for level, key in FIELDS]
                         for cat in categories], dtype=np.float64)
    if not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError('Nonfinite or out-of-range checkpoint metric')
    return matrix


def _balanced(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * matrix[:, :2].mean(axis=1) + 0.5 * matrix[:, 2:].mean(axis=1)


def compare_candidate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    categories = sorted(baseline['per_category'])
    if not categories:
        raise ValueError('No categories')
    old, new = _matrix(baseline, categories), _matrix(candidate, categories)
    delta = new.mean(axis=0) - old.mean(axis=0)
    gains = _balanced(new) - _balanced(old)
    checks = {
        'image_auc_loss_at_most_001': bool(delta[0] >= -0.01),
        'pixel_ap_nonregressing': bool(delta[3] >= 0),
        'pixel_f1_loss_at_most_0005': bool(delta[4] >= -0.005),
        'category_nonregression_at_least_080': bool(np.mean(gains >= 0) >= 0.8),
        'balanced_gain_at_least_0005': bool(np.mean(gains) >= 0.005),
    }
    return {
        'passed': all(checks.values()), 'checks': checks,
        'balanced_score': float(_balanced(new).mean()), 'balanced_gain': float(gains.mean()),
        'category_nonregression': float(np.mean(gains >= 0)),
        'category_gains': dict(zip(categories, gains.tolist())),
        'metric_delta': {f'{l}_{k}': float(d) for (l, k), d in zip(FIELDS, delta)},
    }


def select_candidate(anchor: dict[str, Any], candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [anchor, *[c for c in candidates if c['passed'] is True]]
    return max(eligible, key=lambda c: (float(c['balanced_score']), -int(c['step'])))


def run(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(33101 + args.fold_id)
    output = Path(args.output_root) / f'fold{args.fold_id}'
    if output.exists():
        raise FileExistsError(output)
    path = Path(args.baseline_root) / 'development' / f'fold{args.fold_id}' / 'report.json'
    baseline_report = json.loads(path.read_text(encoding='utf-8'))
    if baseline_report['status'] != 'ok' or not baseline_report['development_only']:
        raise ValueError('Completed development baseline required')
    baseline = baseline_report['evaluation']['corrected']
    contract = baseline_report['contract']
    proxy_index = trainer.DEFAULT_PROXY_INDEX
    if trainer.sha256_file(proxy_index) != contract['development_index_sha256']:
        raise ValueError('Baseline development data changed')
    rows = development_rows(trainer.load_proxy_rows(proxy_index), contract['held_out_categories'],
                            approved_root=proxy_index.resolve().parent / 'dataset')
    if len(rows) != baseline_report['saved']:
        raise ValueError('Baseline development row count changed')
    dataset = trainer.RealIADSupervisedDataset(rows, input_size=448)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2, shuffle=False,
                                         pin_memory=True, persistent_workers=True)
    candidate_root = ROOT / 'outputs/v321_supervised_dinomaly_normal_fp' / f'fold{args.fold_id}' / 'checkpoints'
    candidates = [candidate_root / f'state_i{s:06d}.pt' for s in (5000, 10000, 20000, 30000)]
    if any(not p.is_file() for p in candidates):
        raise FileNotFoundError('The full requested checkpoint grid is not available')
    output.mkdir(parents=True)
    identity = {'phase': 'candidate_benchmark', 'development_only': True, 'fold_id': args.fold_id,
                'baseline_report_sha256': trainer.sha256_file(path),
                'benchmark_sha256': trainer.sha256_file(Path(__file__))}
    anchor = dict(name='v330_anchor', checkpoint=contract['checkpoint'], step=contract['step'], passed=True,
                  balanced_score=float(_balanced(_matrix(baseline, sorted(baseline['per_category']))).mean()))
    results = []
    for checkpoint in candidates:
        if checkpoint == Path(contract['checkpoint']):
            continue
        started = time.time()
        step = int(checkpoint.stem.split('i')[-1])
        candidate_contract = {**contract, 'checkpoint': str(checkpoint), 'step': step, 'score_scale': 1.0,
                              'expected_checkpoint_sha256': trainer.sha256_file(checkpoint)}
        model, _ = load_model(candidate_contract)
        metrics, seen = ExportMetrics(), 0
        with torch.inference_mode():
            for images, masks, _labels, _categories, _groups in loader:
                encoded, decoded = model(images.cuda(non_blocking=True))
                maps = checked_raw(trainer.differentiable_anomaly_map(encoded, decoded, 448).cpu().numpy()[:, 0]).astype(np.float32)
                for i, raw in enumerate(maps):
                    row = rows[seen]
                    # Only the corrected slot is reported; the stored V330 replay is the baseline.
                    metrics.add(category=row['category'], group=row['group'], view=str(row['view']), label=int(row['label']),
                                mask=masks[i, 0].numpy() > 0.5, raw=raw, legacy=np.zeros_like(raw))
                    seen += 1
                if seen % 500 == 0 or seen == len(rows):
                    print(f'fold{args.fold_id} step={step} views={seen}/{len(rows)}', flush=True)
        evaluated = metrics.finish()
        report = {**identity, 'name': f'v321_i{step:06d}', 'step': step, 'checkpoint': str(checkpoint),
                  'status': 'ok', 'views': seen, 'expected': len(rows), 'contract': candidate_contract,
                  'metrics': evaluated['corrected'], 'elapsed_minutes': (time.time() - started) / 60,
                  **compare_candidate(baseline, evaluated['corrected'])}
        trainer.atomic_write_json(output / f'candidate_i{step:06d}.json', report)
        results.append(report)
        print(json.dumps({k: report[k] for k in ('fold_id', 'step', 'balanced_score', 'balanced_gain', 'passed')}), flush=True)
        del model, encoded, decoded, maps
        torch.cuda.empty_cache()
    selected = select_candidate(anchor, results)
    trainer.atomic_write_json(output / 'selection.json', {**identity, 'status': 'ok', 'selected': selected,
                              'candidate_count': len(results), 'anchor': anchor,
                              'limitation': 'Development checkpoint selection only; not a new training run or a B score.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold-id', type=int, choices=range(4), required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    run(parser.parse_args())
