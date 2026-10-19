"""Paired source-scale validation with reserved normal groups; never B labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts import v342_clean_longtrain as longtrain
from scripts.v339_verify_clean_metrics import evaluation_drift
from scripts.v344_longrun_submission import load_verified_bundle
from scripts.v348_clean_memory_core import fuse_memory, normal_quantile_scale
from scripts.v350_scaled_evidence import ScaledEvidenceMetrics
from scripts.v353_highres_development import canonical_view
from scripts import v360_seen_reference_validation as prior

ROOT = (Path.home() / 'csig')
INDEX_SHA = 'a1f8a6e7f4a813488119813ba1ee38bee4f692534154abe6ad7ab21eab705177'
PRIOR_SHA = '79f3373f55c627c2723592c150e5f48a4b1c85eb50a881dad97141f572647731'
PRODUCER_SHA = '55c78ab3eb2865e55ec66a6661ec146958da9b5ed60606660e2954bf588a3543'
VARIANTS = ('reference', 'category_calibrated', 'deployment_scale')


def category_partitions(rows, category, split):
    by_phase = {'train': defaultdict(list), 'test': defaultdict(list)}
    for row in rows:
        if row['category'] != category:
            continue
        if row['phase'] not in by_phase or not row['group'].startswith(category + '/'):
            raise ValueError('Wrong category phase or object identity')
        if row['phase'] == 'train' and (row['label'] != 0 or row.get('mask_path')):
            raise ValueError('Reference source must be exclusively normal')
        by_phase[row['phase']][row['group']].append(row)
    for groups in by_phase.values():
        for group, members in groups.items():
            if len(members) != 5 or {canonical_view(r['view']) for r in members} != set('01234'):
                raise ValueError('Missing or duplicate five-view members')
            groups[group] = sorted(members, key=lambda r: int(canonical_view(r['view'])))
    train, test = by_phase['train'], by_phase['test']
    if len(train) != 25 or len(test) != 80 or set(train) & set(test):
        raise ValueError('Normal source and evaluation populations must be disjoint25/80 objects')
    expected = prior.split_seen_groups({g: tuple('01234') for g in train})
    if split != expected:
        raise ValueError('Frozen source split changed')
    return {'bank': [r for g in split['bank_groups'] for r in train[g]],
            'calibration': [r for g in split['calibration_groups'] for r in train[g]],
            'normal_audit': [r for g in split['audit_groups'] for r in train[g]],
            'evaluation': [r for g in test for r in test[g]]}


def decoded_rgb448_sha256(path):
    with Image.open(path) as image:
        pixels = image.convert('RGB').resize((448, 448), Image.Resampling.BICUBIC)
        return hashlib.sha256(pixels.tobytes()).hexdigest()


def require_pixel_disjoint(hashes):
    if set(hashes) != {'bank', 'calibration', 'normal_audit', 'evaluation'} or any(not h for h in hashes.values()):
        raise ValueError('All four decoded-image partitions are required')
    owners = {}
    for phase, members in hashes.items():
        for digest in members.values():
            if digest in owners and owners[digest] != phase:
                raise ValueError(f'Decoded pixel overlap between {owners[digest]} and {phase}')
            owners[digest] = phase
    return {'passed': True, 'hash_method': 'RGB448_BICUBIC_SHA256',
            'partition_counts': {k: len(v) for k, v in hashes.items()},
            'limits': 'Exact normalized-pixel duplicate check, not a near-duplicate guarantee'}


def policy_maps(reference, memory, *, category_ref_q99, memory_q99, deployed_ref_q99):
    return {'reference': np.asarray(reference, dtype=np.float32),
            'category_calibrated': fuse_memory(reference, memory,
                scale=normal_quantile_scale(category_ref_q99, memory_q99)),
            'deployment_scale': fuse_memory(reference, memory,
                scale=normal_quantile_scale(deployed_ref_q99, memory_q99))}


def normal_audit_metrics(raw, *, reference_q99):
    values = np.asarray(raw, dtype=np.float32)
    if (values.shape != (25, 448, 448) or not np.isfinite(values).all() or np.any(values < 0)
            or not np.isfinite(reference_q99) or reference_q99 <= 0):
        raise ValueError('Expected25 reserved normal views and a fixed positive calibration threshold')
    areas = (values > reference_q99).mean(axis=(1, 2))
    return {'views': 25, 'pixels': int(values.size), 'reference_q99': float(reference_q99),
            'pixel_fraction_above_reference_q99': float(areas.mean()),
            'view_area_p95': float(np.quantile(areas, .95)), 'raw_mean': float(values.mean()),
            'scope': 'False-positive pixel area on known-normal reserved images at a source-fixed threshold; not image FPR'}


def score_policies(rows, references, memories, masks, **scales):
    if not (len(rows) == len(references) == len(memories) == len(masks)):
        raise ValueError('Evaluation stream lengths differ')
    accumulators = {name: ScaledEvidenceMetrics() for name in VARIANTS}
    for row, reference, memory, mask in zip(rows, references, memories, masks):
        view_score = trainer.top_mean(reference, .005)
        for name, raw in policy_maps(reference, memory, **scales).items():
            accumulators[name].add(category=row['category'], group=row['group'],
                view=canonical_view(row['view']), label=int(row['label']), mask=mask,
                raw=raw, reference_view_score=view_score)
    return {name: value.finish() for name, value in accumulators.items()}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _code_hashes():
    names = ('v364_calibration_alignment.py', 'v360_seen_reference_validation.py',
             'v351_source_calibration.py', 'v350_scaled_evidence.py', 'v348_clean_memory_core.py',
             'v330_raw_response.py', 'v283_supervised_dinomaly_longtrain.py', 'v353_highres_development.py',
             'v180_dinomaly3_dinov3_csig.py', 'v182_dinomaly2_dinov3_csig_official.py')
    return {str(Path(__file__).with_name(n)): trainer.sha256_file(Path(__file__).with_name(n)) for n in names}


def run(args):
    import torch

    output = (Path(args.output_root) / f'fold{args.fold}').resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('New project output folder required')
    prior_report_path = ROOT / 'outputs/v360_seen_reference_validation_560_20260912/report.json'
    producer_path = ROOT / 'outputs/v362_category_memory_B_560_package_only_retry_20260912/manifest.json'
    source_hashes = {str(prior.INDEX): INDEX_SHA, str(prior_report_path): PRIOR_SHA,
                     str(producer_path): PRODUCER_SHA, **_code_hashes()}
    if any(trainer.sha256_file(Path(p)) != h for p, h in source_hashes.items()):
        raise ValueError('Frozen source report, index or code differs')
    previous = _read(prior_report_path)
    deployed_q99 = float(_read(producer_path)['provenance']['reference_calibration_q99'])
    bundle = load_verified_bundle(ROOT / 'outputs/v343_verified_longrun_20260911', prior.GATE)
    record = bundle['records'][args.fold]; selected = record['selected']
    rows = trainer.load_proxy_rows(prior.INDEX)
    held = sorted(selected['evaluation']['metrics']['per_category'])
    old_fold = next(f for f in previous['folds_detail'] if f['fold'] == args.fold)
    if set(old_fold['details']) != set(held):
        raise ValueError('Held-out categories changed')
    source_hashes.update(bundle['selection_record_hashes'])
    source_hashes[selected['checkpoint']] = selected['checkpoint_sha256']
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(36401 + args.fold)
    model, deps = prior._model(Path(selected['checkpoint']), 0)
    runtime = longtrain.runtime_source_hashes(deps, model)
    if runtime != record['long_config']['runtime_source_hashes']:
        raise ValueError('Reference runtime differs from trained checkpoint')
    approved = Path(record['long_config']['approved_image_root']).resolve()
    output.mkdir(parents=True)
    print(f'fold{args.fold}: model_loaded', flush=True)
    results = {}
    started = time.monotonic()
    for category in held:
        split = old_fold['details'][category]['split']
        partitions = category_partitions(rows, category, split)
        file_hashes, pixels = {}, {}
        for name, members in partitions.items():
            pixels[name] = {}
            for row in members:
                for field in ('image_path', 'mask_path'):
                    if row.get(field):
                        path = Path(row[field])
                        if not path.resolve().is_relative_to(approved) or 'test_b' in str(path).lower():
                            raise ValueError('Unapproved auxiliary image or mask')
                        file_hashes[str(path)] = trainer.sha256_file(path)
                pixels[name][row['image_path']] = decoded_rgb448_sha256(row['image_path'])
        duplicate_audit = require_pixel_disjoint(pixels)
        print(f'fold{args.fold}: {category} pixel_partitions_verified', flush=True)
        memory = prior._bank(model, partitions['bank'], 560)
        ref_cal, mem_cal, _, _ = prior._maps(model, partitions['calibration'], memory, 560)
        rq, mq = prior.source_q99(ref_cal), prior.source_q99(mem_cal)
        old_cal = old_fold['details'][category]['calibration_q99']
        if abs(old_cal['reference'] - rq) > 1e-6 or abs(old_cal['memory'] - mq) > 1e-6:
            raise ValueError('Calibration does not reproduce frozen seen simulation')
        refs, mems, masks, _ = prior._maps(model, partitions['evaluation'], memory, 560)
        scales = dict(category_ref_q99=rq, memory_q99=mq, deployed_ref_q99=deployed_q99)
        scored = score_policies(partitions['evaluation'], refs, mems, masks, **scales)
        drifts = {}
        for variant, old_name in [('reference', 'reference'), ('category_calibrated', 'fusion')]:
            drifts[variant] = evaluation_drift(old_fold['details'][category]['metrics'][old_name], scored[variant])
            if drifts[variant] > 1e-6:
                raise ValueError('Existing reference/category-calibrated metrics did not reproduce')
        query_digest = hashlib.sha256(refs.astype('<f4').tobytes() + mems.astype('<f4').tobytes()).hexdigest()
        del refs, mems, masks
        normal_ref, normal_mem, normal_masks, _ = prior._maps(model, partitions['normal_audit'], memory, 560)
        if any(np.asarray(m).any() for m in normal_masks):
            raise ValueError('A reserved normal audit image has an anomaly mask')
        normal_maps = {name: [] for name in VARIANTS}
        for ref, mem in zip(normal_ref, normal_mem):
            for name, values in policy_maps(ref, mem, **scales).items():
                normal_maps[name].append(values)
        normal_stats = {name: normal_audit_metrics(np.stack(values), reference_q99=rq)
                        for name, values in normal_maps.items()}
        if any(trainer.sha256_file(Path(p)) != h for p, h in file_hashes.items()):
            raise ValueError('Auxiliary data changed during category evaluation')
        category_result = {'category': category, 'split': split, 'calibration': scales,
            'duplicate_audit': duplicate_audit, 'input_file_hashes': file_hashes,
            'metrics': scored, 'reference_replay_max_drift': drifts, 'reserved_normal': normal_stats,
            'memory_sha256': hashlib.sha256(memory.tobytes()).hexdigest(),
            'query_raw_stream_sha256': query_digest}
        with (output / f'{category}.json').open('x', encoding='utf-8') as f:
            json.dump(category_result, f, indent=2, allow_nan=False)
        results[category] = category_result
        print(f'fold{args.fold}: {category} complete', flush=True)
    if any(trainer.sha256_file(Path(p)) != h for p, h in source_hashes.items()):
        raise ValueError('Bound source changed during validation')
    if longtrain.runtime_source_hashes(deps, model) != runtime:
        raise ValueError('Model runtime changed')
    result = {'status': 'ok', 'version': 'v364_calibration_alignment_v1', 'fold': args.fold,
        'variants': list(VARIANTS), 'input_size': 560, 'category_count': len(held),
        'details': results, 'source_hashes': source_hashes, 'runtime_source_hashes': runtime,
        'deployed_reference_q99': deployed_q99, 'elapsed_seconds': time.monotonic() - started,
        'test_b_labels_accessed': False, 'quality_status': 'development_comparison_only',
        'scope': 'Paired single-fold auxiliary simulation with source-scale alignment and reserved normal audit; not a B ensemble score'}
    with (output / 'report.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f'fold{args.fold}: report_written', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=range(4), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    run(args)
