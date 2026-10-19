"""Train-normal category calibration of the actual four-model median ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.v340_clean_submission import combine_raw
from scripts.v348_clean_memory_core import fuse_memory, normal_quantile_scale
from scripts.v355_seen_category_calibration import category_quantile

ROOT = (Path.home() / 'csig')
PRODUCER = ROOT / 'outputs/v362_category_memory_B_560_package_only_retry_20260912/manifest.json'
PRODUCER_SHA = '55c78ab3eb2865e55ec66a6661ec146958da9b5ed60606660e2954bf588a3543'
SCHEMA = 'v365_actual_ensemble_category_calibration_v1'


def response_digest(raw):
    return hashlib.sha256(np.asarray(raw, dtype='<f4').tobytes(order='C')).hexdigest()


def checked_members(members):
    if len(members) != 20 or len(set(members)) != 20 or members != sorted(members):
        raise ValueError('Expected20 ordered unique official Train calibration views')
    if any(not re.fullmatch(r'Train/[^/\\:.]+/S\d{4}/[0-4]\.png', member) for member in members):
        raise ValueError('Expected normal Train members only')
    groups = defaultdict(list)
    for index, member in enumerate(members):
        groups['/'.join(member.split('/')[1:3])].append(index)
    if len(groups) != 4 or len({x.split('/')[1] for x in members}) != 1:
        raise ValueError('Expected four objects from one category')
    if any([members[i].rsplit('/', 1)[1] for i in rows] != [f'{v}.png' for v in range(5)]
           for rows in groups.values()):
        raise ValueError('Incomplete five-view object')
    return dict(groups)


def evaluate_category(fold_maps, memory, members, old_reference_q99):
    groups = checked_members(members)
    reference = combine_raw(fold_maps)
    memory = np.asarray(memory, np.float32)
    if reference.shape != (20, 448, 448) or memory.shape != reference.shape:
        raise ValueError('Expected20 normal448 maps per branch')
    reference_q99 = category_quantile(reference)
    memory_q99 = category_quantile(memory)
    normal_quantile_scale(old_reference_q99, memory_q99)
    crossfit = []
    for held, held_indices in groups.items():
        fit_indices = [i for group, rows in groups.items() if group != held for i in rows]
        rq = category_quantile(reference[fit_indices])
        mq = category_quantile(memory[fit_indices])
        fractions = {}
        for name, scale in [('category', normal_quantile_scale(rq, mq)),
                            ('old_scale', normal_quantile_scale(old_reference_q99, mq))]:
            fractions[f'{name}_pixel_fraction'] = float(np.mean([
                np.mean(fuse_memory(reference[i], memory[i], scale=scale) > rq)
                for i in held_indices]))
        crossfit.append({'held_group': held, 'fit_groups': [g for g in groups if g != held],
                         'reference_q99': rq, 'memory_q99': mq,
                         'reference_pixel_fraction': float(np.mean(reference[held_indices] > rq)),
                         **fractions})
    return {'views': 20, 'groups': 4, 'members': members,
            'reference_q99': reference_q99, 'memory_q99': memory_q99,
            'scale': normal_quantile_scale(reference_q99, memory_q99),
            'old_reference_q99': float(old_reference_q99),
            'median_fold_q99': float(np.median([category_quantile(x) for x in fold_maps])),
            'fold_response_sha256': [response_digest(x) for x in fold_maps],
            'ensemble_response_sha256': response_digest(reference),
            'memory_response_sha256': response_digest(memory),
            'leave_one_group_out': crossfit,
            'crossfit_macro': {key: float(np.mean([item[key] for item in crossfit])) for key in
                              ('reference_pixel_fraction', 'old_scale_pixel_fraction', 'category_pixel_fraction')},
            'audit_scope': 'normal_calibration_group_crossfit_not_independent_holdout'}


def verify_replay(first, second):
    if first != second:
        raise ValueError('Exact ensemble/category response replay differs')


def load_models(bundle):
    import torch
    from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    from scripts import v342_clean_longtrain as longtrain
    from scripts.v283_supervised_dinomaly_longtrain import sha256_file

    deps = adapter._import_official_modules()
    models = []
    for fold, record in enumerate(bundle['records']):
        choice = record['selected']
        path = Path(choice['checkpoint'])
        if sha256_file(path) != choice['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if choice['kind'] == 'source_fallback':
            longtrain.validate_source(record['source'], state, choice['checkpoint_sha256'], fold)
        elif (state['immutable_config'] != record['long_config']
              or state['baseline_sha256'] != record['baseline_sha256']
              or state['relative_additional_step'] != choice['additional_steps']
              or state['total_optimizer_step'] != choice['total_steps']):
            raise ValueError('Long-run model metadata mismatch')
        model, _ = adapter._build_model(deps, backbone='dinov3_vit_base_16', dropout=.4, reduce_dim=256,
                                        context_aware_recenter=True, mask_neighbor_size=0,
                                        weights_dir=ROOT / 'Dinomaly2_dinov3/backbones/weights')
        model.load_state_dict(state['model'], strict=True)
        del state
        model = model.to(f'cuda:{fold}').eval()
        model.requires_grad_(False)
        if longtrain.runtime_source_hashes(deps, model) != record['long_config']['runtime_source_hashes']:
            raise ValueError('Model runtime changed')
        models.append(model)
    return models


def infer_normal_category(models, reference_dataset, high_dataset, indices, resident, pool):
    import torch
    from scripts.v180_dinomaly3_dinov3_csig import _extract_patch_tokens
    from scripts.v283_supervised_dinomaly_longtrain import differentiable_anomaly_map
    from scripts.v348_clean_memory_core import resize_raw_map

    def worker(fold, images):
        torch.cuda.set_device(fold)
        with torch.inference_mode():
            enc, dec = models[fold](images.to(f'cuda:{fold}'))
            return differentiable_anomaly_map(enc, dec, 448).float().cpu().numpy()[:, 0]

    folds = [[] for _ in range(4)]
    memory_maps = []
    for offset in range(0, len(indices), 4):
        current = indices[offset:offset + 4]
        images = torch.stack([reference_dataset[i][0] for i in current])
        high = torch.stack([high_dataset[i][0] for i in current])
        pending = [pool.submit(worker, f, images) for f in range(4)]
        for fold, result in enumerate(pending):
            folds[fold].append(result.result())
        with torch.inference_mode():
            tokens, grid = _extract_patch_tokens(models[0].encoder, high.to('cuda:0'), [3, 7, 11])
            if grid != (35, 35):
                raise ValueError('Expected560 feature grid')
            distances = resident.distances(tokens)
            memory_maps.extend(resize_raw_map(value.reshape(grid)) for value in distances)
    return [np.concatenate(items) for items in folds], np.stack(memory_maps)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(args):
    import torch
    from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    from scripts.v283_supervised_dinomaly_longtrain import sha256_file
    from scripts.v344_longrun_submission import load_verified_bundle, inference_hashes
    from scripts.v349_clean_memory_bank import source_inputs, preprocessing_environment_identity
    from scripts.v351_source_calibration import ResidentCosineMemory
    from scripts.v356_seen_memory_submission import GATE, _load_category_memory

    output = Path(args.output_root).resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output.exists():
        raise ValueError('A new project output is required')
    if torch.cuda.device_count() < 4 or any(torch.cuda.mem_get_info(i)[0] < 8 * 2**30 for i in range(4)):
        raise RuntimeError('Four GPUs with8GiB free required')
    torch.set_num_threads(4)
    torch.manual_seed(36501)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if sha256_file(PRODUCER) != PRODUCER_SHA:
        raise ValueError('Frozen V362 producer changed')
    producer = _read(PRODUCER)['provenance']
    train, _, _, partition = source_inputs()
    bundle = load_verified_bundle(ROOT / 'outputs/v343_verified_longrun_20260911', GATE)
    if [r['selected'] for r in bundle['records']] != producer['selected_checkpoints']:
        raise ValueError('Models differ from V362')
    bank_root = ROOT / 'outputs/v352_clean_memory_bank_env_20260911/i560'
    cal_path = ROOT / 'outputs/v355_seen_category_calibration_560_20260912/seen_category_i560/manifest.json'
    bank, category_indices, bank_report = _load_category_memory(bank_root)
    if (bank_report['bank_sha256'] != producer['bank_sha256']
            or sha256_file(bank_root / 'manifest.json') != producer['bank_manifest_sha256']
            or sha256_file(cal_path) != producer['category_calibration_sha256']):
        raise ValueError('Frozen bank/calibration identities differ from V362')
    old_calibration = _read(cal_path)
    models = load_models(bundle)
    reference_dataset = adapter.CsigZipDataset(train, 'Train', adapter._build_transform(448, 448), include_metadata=True)
    high_dataset = adapter.CsigZipDataset(train, 'Train', adapter._build_transform(560, 560), include_metadata=True)
    if reference_dataset.records != high_dataset.records:
        raise ValueError('Reference/high-resolution dataset order differs')
    cal_groups = set(partition['calibration_groups'])
    grouped = defaultdict(list)
    for index, record in enumerate(reference_dataset.records):
        if record.group in cal_groups:
            grouped[record.category].append(index)
    if sorted(grouped) != sorted(category_indices) or len(grouped) != 50 or any(len(v) != 20 for v in grouped.values()):
        raise ValueError('Expected50 categories and1000 frozen normal views')
    paths = [PRODUCER, cal_path, bank_root / 'memory.npz', bank_root / 'manifest.json', GATE,
             *[Path(r['selected']['checkpoint']) for r in bundle['records']],
             *[Path(p) for p in bundle['selection_record_hashes']],
             *[PROJECT / 'scripts' / name for name in inference_hashes()], Path(__file__),
             PROJECT / 'scripts/v355_seen_category_calibration.py',
             PROJECT / 'scripts/v348_clean_memory_core.py', PROJECT / 'scripts/v349_clean_memory_bank.py',
             PROJECT / 'scripts/v351_source_calibration.py', PROJECT / 'scripts/v356_seen_memory_submission.py']
    inputs = {str(path): sha256_file(path) for path in paths}
    output.mkdir()
    started = time.monotonic()
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for pass_number in (1, 2):
            for category, indices in sorted(grouped.items()):
                resident = ResidentCosineMemory(bank[category_indices[category]], device='cuda:0', chunk_size=8192)
                folds, memory = infer_normal_category(models, reference_dataset, high_dataset, indices, resident, pool)
                members = [reference_dataset.records[i].member for i in indices]
                result = evaluate_category(folds, memory, members, producer['reference_calibration_q99'])
                drift = abs(result['memory_q99'] - old_calibration['categories_detail'][category]['q99'])
                if drift > 1e-6:
                    raise ValueError(f'Actual inference encoder memory calibration differs: {category} {drift}')
                result['published_memory_q99_abs_drift'] = drift
                if pass_number == 1:
                    results[category] = result
                    (output / f'{category}.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
                else:
                    verify_replay(results[category], result)
                del folds, memory, resident
                print(f'pass={pass_number} category={category} done={list(sorted(grouped)).index(category)+1}/50', flush=True)
    source_inputs()
    if any(sha256_file(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError('Source files changed during calibration')
    report = {'status': 'ok', 'version': SCHEMA, 'reference_input_size': 448, 'memory_input_size': 560,
              'categories': 50, 'views_per_category': 20, 'views_per_pass': 1000, 'passes': 2,
              'response_replay': 'exact_all_maps_all_categories', 'categories_detail': results,
              'source_files_sha256': inputs, 'partition': partition, 'producer_sha256': PRODUCER_SHA,
              'selected_checkpoints': producer['selected_checkpoints'],
              'bank_sha256': producer['bank_sha256'], 'preprocess_environment': preprocessing_environment_identity(),
              'crossfit_macros': {k: float(np.mean([r['crossfit_macro'][k] for r in results.values()]))
                                  for k in next(iter(results.values()))['crossfit_macro']},
              'reference_q99_range': [min(r['reference_q99'] for r in results.values()), max(r['reference_q99'] for r in results.values())],
              'elapsed_seconds': time.monotonic() - started,
              'quality_status': 'normal_only_calibration_and_crossfit_not_defect_quality_validation',
              'test_b_accessed': False, 'new_training': False}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('categories_detail', 'source_files_sha256', 'partition', 'preprocess_environment', 'selected_checkpoints')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    run(parser.parse_args())
