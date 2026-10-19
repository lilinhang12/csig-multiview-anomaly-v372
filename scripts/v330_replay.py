"""Replay frozen models with identical raw responses for development and export."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts.v284_supervised_dinomaly_state_infer import _calibrate_map
from scripts.v330_raw_response import (checked_raw, current_pipeline_hashes, encode_png, histogram_metrics,
                                       record_contract_id, save_record, score_views, selected_contract)

ROOT = Path.home() / 'csig'


def development_rows(rows: Sequence[Mapping[str, Any]], held_out: Sequence[str],
                     approved_root: Path | None = None) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if row['category'] in set(held_out) and row['phase'] == 'test']
    if not selected:
        raise ValueError('Empty held-out development partition')
    for row in selected:
        for key in ('image_path', 'mask_path'):
            if not row.get(key):
                continue
            resolved = Path(row[key]).resolve()
            if 'test_b' in str(resolved).replace('\\', '/').lower():
                raise ValueError('Test_B is forbidden in labelled development evaluation')
            if approved_root is not None and not resolved.is_relative_to(approved_root.resolve()):
                raise ValueError(f'Development provenance violation: {key}')
    return selected


class ExportMetrics:
    def __init__(self) -> None:
        self.hists = {name: defaultdict(lambda: np.zeros((2, 256), dtype=np.int64))
                      for name in ('legacy', 'corrected')}
        self.groups: dict[str, dict[str, Any]] = {}
        self.views = 0

    def add(self, *, category: str, group: str, view: str, label: int, mask: np.ndarray,
            legacy: np.ndarray, raw: np.ndarray) -> None:
        item = self.groups.setdefault(group, {'category': category, 'label': 0, 'views': {}})
        if item['category'] != category or view in item['views']:
            raise ValueError('Category mismatch or duplicate development view')
        if np.shape(mask) != (448, 448) or np.shape(legacy) != (448, 448):
            raise ValueError('Invalid development shape')
        if not np.isfinite(legacy).all() or np.min(legacy) < 0 or np.max(legacy) > 1:
            raise ValueError('Invalid legacy response')
        pngs = {'legacy': np.rint(legacy * 255).astype(np.uint8), 'corrected': encode_png(raw)}
        labels = np.asarray(mask, dtype=np.uint16).reshape(-1)
        for name, png in pngs.items():
            indices = labels * 256 + png.reshape(-1)
            self.hists[name][category] += np.bincount(indices, minlength=512).reshape(2, 256)
        old_score = (0.6 * trainer.top_mean(legacy, 0.001) +
                     0.3 * trainer.top_mean(legacy, 0.005) + 0.1 * float(legacy.max()))
        item['views'][view] = {'legacy': old_score, 'corrected': trainer.top_mean(raw, 0.005)}
        item['label'] = max(item['label'], int(label))
        self.views += 1

    def finish(self) -> dict[str, Any]:
        if any(len(g['views']) != 5 for g in self.groups.values()):
            raise ValueError('Expected five views in every development group')
        report: dict[str, Any] = {'views': self.views, 'pixel_count': self.views * 448 * 448,
                                  'categories': len(self.hists['legacy']), 'groups': len(self.groups)}
        for name in self.hists:
            category_metrics = {}
            for category, hist in sorted(self.hists[name].items()):
                labels, scores = [], []
                for item in self.groups.values():
                    if item['category'] != category:
                        continue
                    values = [v[name] for v in item['views'].values()]
                    if name == 'legacy':
                        ordered = np.sort(values)[::-1]
                        score = float(0.6 * ordered[0] + 0.25 * ordered[:2].mean() + 0.15 * ordered.mean())
                    else:
                        score = score_views(values)
                    labels.append(item['label'])
                    scores.append(score)
                category_metrics[category] = {
                    'image': trainer.metric_triplet(np.asarray(labels), np.asarray(scores)),
                    'pixel': histogram_metrics(hist), 'unique_group_scores': len(set(scores)),
                    'groups': len(scores), 'pixels_at_255_fraction': float(hist[:, -1].sum() / hist.sum()),
                }
            report[name] = {
                'per_category': category_metrics,
                'macro': {level: {key: float(np.mean([m[level][key] for m in category_metrics.values()]))
                                  for key in ('auroc', 'ap', 'best_f1')}
                          for level in ('image', 'pixel')},
            }
        return report


def load_model(contract: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch

    deps = adapter._import_official_modules()
    digest = trainer.sha256_file(Path(contract['checkpoint']))
    expected = contract['expected_checkpoint_sha256']
    if expected != digest:
        raise ValueError('Checkpoint hash differs from validation contract')
    payload = torch.load(contract['checkpoint'], map_location='cpu', weights_only=False)
    config = payload['immutable_config']
    if config['fold_id'] != contract['fold_id'] or payload['step'] != contract['step'] or config['input_size'] != 448:
        raise ValueError('Checkpoint identity/config mismatch')
    if set(config['held_out_categories']) != set(contract['held_out_categories']):
        raise ValueError('Checkpoint holdout differs from gate')
    if config.get('partition_key') != 'category':
        raise ValueError('Checkpoint was not trained with a category holdout')
    model, _ = adapter._build_model(deps, backbone='dinov3_vit_base_16', dropout=float(config['source_dropout']),
                                    reduce_dim=256, context_aware_recenter=True, mask_neighbor_size=0,
                                    weights_dir=ROOT / 'Dinomaly2_dinov3/backbones/weights')
    model.load_state_dict(payload['model'], strict=True)
    contract['checkpoint_sha256'] = digest
    contract['input_size'] = 448
    contract['pipeline_hashes'] = current_pipeline_hashes()
    return model.cuda().eval(), deps


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.time()
    target = Path(args.output_root) / args.phase / f'fold{args.fold_id}'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite run: {target}')
    contract = selected_contract(args.route_root, args.fold_id)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(33001 + args.fold_id)
    model, _deps = load_model(contract)
    metrics = ExportMetrics()
    rows: list[dict[str, Any]] = []
    if args.phase == 'development':
        rows = development_rows(trainer.load_proxy_rows(args.proxy_index), contract['held_out_categories'],
                                approved_root=Path(args.proxy_index).resolve().parent / 'dataset')
        dataset = trainer.RealIADSupervisedDataset(rows, input_size=448)
        old_path = Path(args.legacy_root) / f'fold{args.fold_id}' / 'infer_report.json'
        old_report = json.loads(old_path.read_text(encoding='utf-8'))
        if old_report['checkpoint'] != contract['checkpoint'] or old_report['checkpoint_step'] != contract['step']:
            raise ValueError('Legacy calibration uses a different checkpoint')
        contract['legacy_calibration_sha256'] = trainer.sha256_file(old_path)
        contract['development_index_sha256'] = trainer.sha256_file(args.proxy_index)
    else:
        dataset = adapter.CsigZipDataset(args.test_zip, 'Test_B', adapter._build_transform(448, 448), include_metadata=True)
        contract['test_zip_sha256'] = trainer.sha256_file(args.test_zip)
        contract['record_contract_id'] = record_contract_id(contract)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=False, pin_memory=True, persistent_workers=args.num_workers > 0)
    target.mkdir(parents=True)
    trainer.atomic_write_json(target / 'contract.json', contract)
    saved, saturated, pixel_count = 0, 0, 0
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch[0].cuda(non_blocking=True)
                encoded, decoded = model(images)
                unscaled = trainer.differentiable_anomaly_map(encoded, decoded, 448).cpu().numpy()[:, 0]
                raw_maps = checked_raw(unscaled * contract['score_scale']).astype(np.float32)
                for i, raw in enumerate(raw_maps):
                    if args.phase == 'development':
                        row = rows[saved]
                        old = _calibrate_map(unscaled[i], old_report['global_lo'], old_report['global_hi'], 0.9)
                        metrics.add(category=row['category'], group=row['group'], view=str(row['view']),
                                    label=int(row['label']), mask=batch[1][i, 0].numpy() > 0.5, legacy=old, raw=raw)
                    else:
                        category, sample, view = (str(batch[k][i]) for k in (1, 2, 3))
                        save_record(target / 'raw_maps' / category / sample / f'{view}.npz', raw,
                                    category=category, sample=sample, view=view, fold_id=args.fold_id, step=contract['step'],
                                    contract_id=contract['record_contract_id'])
                    png = encode_png(raw)
                    saturated += int(np.count_nonzero(png == 255))
                    pixel_count += png.size
                    saved += 1
                if saved % 100 == 0 or saved == len(dataset):
                    print(f'{args.phase} fold{args.fold_id} views={saved}/{len(dataset)}', flush=True)
        if saved != len(dataset):
            raise RuntimeError('Incomplete inference')
        report = {'status': 'ok', 'phase': args.phase, 'fold_id': args.fold_id, 'saved': saved,
                  'expected': len(dataset), 'contract': contract, 'pixels_at_255': saturated,
                  'pixel_count': pixel_count, 'elapsed_minutes': (time.time() - started) / 60}
        if args.phase == 'development':
            report['evaluation'] = metrics.finish()
            report['development_only'] = True
            report['metric_scope'] = 'category-held-out single-fold, all pixels at exported uint8 precision'
        trainer.atomic_write_json(target / 'report.json', report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('evaluation', 'contract')}), flush=True)
        return report
    except Exception as error:
        trainer.atomic_write_json(target / 'failure.json', {'error': repr(error), 'saved': saved})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['development', 'test'])
    parser.add_argument('--fold-id', type=int, choices=range(4), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--route-root', type=Path, default=ROOT / 'outputs/v323_monotonic_calibration')
    parser.add_argument('--legacy-root', type=Path, default=ROOT / 'outputs/b榜_v326_supervised/evidence')
    parser.add_argument('--proxy-index', type=Path, default=trainer.DEFAULT_PROXY_INDEX)
    parser.add_argument('--test-zip', type=Path, default=ROOT / 'Test_B.zip')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
