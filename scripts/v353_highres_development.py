"""Clean category-held-out evaluation of high-resolution memory evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts import v342_clean_longtrain as longtrain
from scripts.v330_raw_response import score_views
from scripts.v337_clean_gate import _validate_evaluation
from scripts.v339_verify_clean_metrics import evaluation_drift
from scripts.v344_longrun_submission import inference_hashes, load_verified_bundle
from scripts.v180_dinomaly3_dinov3_csig import _extract_patch_tokens
from scripts.v348_clean_memory_core import fuse_memory, normal_quantile_scale, resize_raw_map
from scripts.v349_clean_memory_bank import load_verified_bank, preprocessing_environment_identity, source_inputs
from scripts.v350_scaled_evidence import ScaledEvidenceMetrics
from scripts.v351_source_calibration import SCHEMA as CALIBRATION_SCHEMA
from scripts.v351_source_calibration import ResidentCosineMemory

ROOT = (Path.home() / 'csig')
GATE = ROOT / 'outputs/v339_independent_clean_replay_final_20260910/clean_selection_gate.json'
BANK_ROOT = ROOT / 'outputs/v352_clean_memory_bank_env_20260911'
CALIBRATION_ROOT = ROOT / 'outputs/v351_source_calibration_20260911'
BRANCHES = ('reference', 'memory_i560', 'memory_i672', 'fusion_i560', 'fusion_i672')


def canonical_view(value):
    text = str(value)
    if text in {'0', '1', '2', '3', '4'}:
        return text
    if len(text) == 2 and text[0] == 'C' and text[1] in {'1', '2', '3', '4', '5'}:
        return str(int(text[1]) - 1)
    raise ValueError(f'Unsupported development view identifier: {value!r}')


def branch_maps(reference, memory, *, scale):
    reference = np.asarray(reference, dtype=np.float32)
    memory = np.asarray(memory, dtype=np.float32)
    if reference.shape != (448, 448) or memory.shape != (448, 448):
        raise ValueError('Reference and memory maps must both be448x448')
    if not np.isfinite(reference).all() or not np.isfinite(memory).all():
        raise ValueError('Reference and memory maps must be finite')
    fused = fuse_memory(reference, memory, scale=float(scale))
    return {'reference': reference.copy(), 'memory': memory.copy(), 'fusion': fused}


def branch_map_for_name(maps, name):
    if name not in BRANCHES or name not in maps:
        raise ValueError(f'Unknown evidence branch: {name}')
    return maps[name]


def primary_reference_score(raw):
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != (448, 448) or not np.isfinite(values).all():
        raise ValueError('Expected a finite448x448 reference map')
    flat = values.reshape(-1)
    count = max(1, int(np.ceil(flat.size * .005)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def validate_development_report(report):
    if (report.get('status') != 'ok' or report.get('branches') != list(BRANCHES)
            or report.get('categories') != 11 or report.get('folds') != 4
            or report.get('test_b_accessed') is not False
            or set(report.get('metrics', {})) != set(BRANCHES)):
        raise ValueError('Incomplete or contaminated high-resolution development report')
    return True


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _calibration(path, expected_kind):
    result = _read(path)
    if (result.get('status') != 'ok' or result.get('version') != CALIBRATION_SCHEMA
            or result.get('kind') != expected_kind or result.get('replay_identical') is not True
            or result.get('partition_sha256') != '5afe17f53ddde0d8d146d07d6e5414e72fa1b40a561c29f33100bf238ceabcee'
            or result.get('train_zip_sha256') != 'a62414e7aa11b32c73ef97f8513a5425ede5510e72e4f13893b31da8f8d62c23'):
        raise ValueError(f'Invalid calibration manifest: {path}')
    calibration = result.get('calibration', {})
    if calibration.get('views') != 1000 or calibration.get('pixels') != 1000 * 448 * 448:
        raise ValueError(f'Calibration coverage mismatch: {path}')
    if not np.isfinite(float(calibration.get('q99', 0))) or float(calibration['q99']) <= 1e-8:
        raise ValueError(f'Degenerate calibration: {path}')
    return result


def _selected_models():
    bundle = load_verified_bundle(
        ROOT / 'outputs/v343_verified_longrun_20260911',
        GATE,
    )
    return bundle


def _fold_worker(fold, selected, development, banks, calibrations):
    import torch

    torch.cuda.set_device(fold)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = Path(selected['checkpoint'])
    if trainer.sha256_file(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Selected reference checkpoint changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    deps = adapter._import_official_modules()
    model, _ = adapter._build_model(
        deps, backbone='dinov3_vit_base_16', dropout=.4, reduce_dim=256,
        context_aware_recenter=True, mask_neighbor_size=0,
        weights_dir=ROOT / 'Dinomaly2_dinov3/backbones/weights',
    )
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.to(f'cuda:{fold}').eval()
    model.requires_grad_(False)
    encoder = model.encoder
    memories = {name: ResidentCosineMemory(value, device=f'cuda:{fold}', chunk_size=8192)
                for name, value in banks.items()}
    scales = {
        'memory_i560': normal_quantile_scale(
            calibrations['reference'][fold]['calibration']['q99'],
            calibrations['memory_i560']['calibration']['q99'],
        ),
        'memory_i672': normal_quantile_scale(
            calibrations['reference'][fold]['calibration']['q99'],
            calibrations['memory_i672']['calibration']['q99'],
        ),
    }
    reference_dataset = trainer.RealIADSupervisedDataset(development, 448)
    dataset560 = trainer.RealIADSupervisedDataset(development, 560)
    dataset672 = trainer.RealIADSupervisedDataset(development, 672)
    loader_kwargs = dict(batch_size=8, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    reference_loader = torch.utils.data.DataLoader(reference_dataset, **loader_kwargs)
    loader560 = torch.utils.data.DataLoader(dataset560, **loader_kwargs)
    loader672 = torch.utils.data.DataLoader(dataset672, **loader_kwargs)
    metrics = {name: ScaledEvidenceMetrics() for name in BRANCHES}
    seen = 0
    with torch.inference_mode():
        for base, high560, high672 in zip(reference_loader, loader560, loader672):
            reference_encoded, reference_decoded = model(base[0].to(f'cuda:{fold}', non_blocking=True))
            reference_maps = trainer.differentiable_anomaly_map(
                reference_encoded, reference_decoded, 448,
            ).float().cpu().numpy()[:, 0]
            token560, grid560 = _extract_patch_tokens(
                encoder, high560[0].to(f'cuda:{fold}', non_blocking=True), [3, 7, 11],
            )
            token672, grid672 = _extract_patch_tokens(
                encoder, high672[0].to(f'cuda:{fold}', non_blocking=True), [3, 7, 11],
            )
            if grid560 != (35, 35) or grid672 != (42, 42):
                raise ValueError('High-resolution patch grid changed')
            memory_maps = {
                'memory_i560': np.stack([resize_raw_map(raw.reshape(grid560))
                                         for raw in memories['memory_i560'].distances(token560)]),
                'memory_i672': np.stack([resize_raw_map(raw.reshape(grid672))
                                         for raw in memories['memory_i672'].distances(token672)]),
            }
            for index in range(len(base[0])):
                row = development[seen + index]
                reference = reference_maps[index]
                branch = {
                    'reference': reference,
                    'memory_i560': memory_maps['memory_i560'][index],
                    'memory_i672': memory_maps['memory_i672'][index],
                }
                branch['fusion_i560'] = branch_maps(reference, branch['memory_i560'], scale=scales['memory_i560'])['fusion']
                branch['fusion_i672'] = branch_maps(reference, branch['memory_i672'], scale=scales['memory_i672'])['fusion']
                score = primary_reference_score(reference)
                mask = base[1][index, 0].cpu().numpy() > .5
                for name in BRANCHES:
                    metrics[name].add(
                        category=row['category'], group=row['group'], view=canonical_view(row['view']),
                        label=int(row['label']), mask=mask, raw=branch_map_for_name(branch, name),
                        reference_view_score=score,
                    )
            seen += len(base[0])
            if seen % 400 == 0 or seen == len(development):
                print(f'v353_fold{fold}_views={seen}/{len(development)}', flush=True)
    if seen != len(development):
        raise ValueError('Development loader alignment ended early')
    return {
        'fold': fold, 'categories': len({row['category'] for row in development}),
        'views': seen, 'metrics': {name: metrics[name].finish() for name in BRANCHES},
        'scales': scales, 'checkpoint': selected,
        'runtime_preprocess': preprocessing_environment_identity(),
    }


def run(args):
    import torch

    output = Path(args.output_root).resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output.exists():
        raise ValueError('A new output directory under the project outputs is required')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError('Four GPUs are required for the four-fold evaluation')
    train, _, groups, partition = source_inputs()
    if len(groups) != 1000 or partition['bank_views'] != 4000 or partition['calibration_views'] != 1000:
        raise ValueError('Frozen official Train partition is incomplete')
    banks = {}
    bank_reports = {}
    for size in (560, 672):
        bank, report = load_verified_bank(BANK_ROOT / f'i{size}')
        banks[f'memory_i{size}'] = bank
        bank_reports[f'memory_i{size}'] = report
    calibrations = {
        'reference': [_calibration(CALIBRATION_ROOT / f'reference_fold{fold}' / 'manifest.json', 'reference')
                      for fold in range(4)],
        'memory_i560': _calibration(CALIBRATION_ROOT / 'memory_i560' / 'manifest.json', 'memory'),
        'memory_i672': _calibration(CALIBRATION_ROOT / 'memory_i672' / 'manifest.json', 'memory'),
    }
    bundle = _selected_models()
    developments = []
    for fold in range(4):
        _, _, development, _, _, _ = longtrain.preflight(SimpleNamespace(gate=GATE, fold_id=fold))
        developments.append(development)
    output.mkdir(parents=True)
    started = time.monotonic()
    jobs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fold, record in enumerate(bundle['records']):
            jobs.append(pool.submit(_fold_worker, fold, record['selected'], developments[fold], banks, calibrations))
        folds = [job.result() for job in jobs]
    folds.sort(key=lambda item: item['fold'])
    for fold_report in folds:
        (output / f"fold{fold_report['fold']}.json").write_text(
            json.dumps(fold_report, indent=2, allow_nan=False), encoding='utf-8',
        )
    aggregate = {name: [] for name in BRANCHES}
    for fold_report in folds:
        for name in BRANCHES:
            aggregate[name].append(fold_report['metrics'][name])
    metrics = {}
    for name in BRANCHES:
        per_category = {}
        for fold_metric in aggregate[name]:
            per_category.update(fold_metric['metrics']['per_category'])
        if len(per_category) != 11:
            raise ValueError('Expected exactly11 clean development categories')
        metrics[name] = {
            'per_category': per_category,
            'macro': {
                level: {key: float(np.mean([item[level][key] for item in per_category.values()]))
                        for key in ('auroc', 'ap', 'best_f1')}
                for level in ('image', 'pixel')
            },
        }
    report = {
        'status': 'ok', 'version': 'v353_clean_highres_development_v1',
        'branches': list(BRANCHES), 'categories': 11, 'folds': 4,
        'test_b_accessed': False, 'development_only': True,
        'train_zip_sha256': trainer.sha256_file(train),
        'partition': partition,
        'partition_sha256': hashlib.sha256(json.dumps(partition, sort_keys=True).encode()).hexdigest(),
        'bank_reports': {name: {key: value for key, value in value.items()
                                if key in ('status', 'version', 'bank_sha256', 'trace', 'metadata')}
                        for name, value in bank_reports.items()},
        'calibrations': {
            'reference': [item['identity'] for item in calibrations['reference']],
            'memory_i560': calibrations['memory_i560']['identity'],
            'memory_i672': calibrations['memory_i672']['identity'],
        },
        'metrics': metrics, 'folds_detail': folds,
        'code_hashes': {**inference_hashes(),
                        'v353_highres_development.py': trainer.sha256_file(Path(__file__))},
        'elapsed_seconds': time.monotonic() - started,
        'scope': 'Clean 11-category held-out development evidence; not B performance or submission selection.',
    }
    validate_development_report(report)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': 'ok', 'branches': list(BRANCHES), 'categories': 11,
                      'macros': metrics, 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    run(parser.parse_args())
