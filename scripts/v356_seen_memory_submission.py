"""B-blind submission builder using category memory for seen categories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(PROJECT))

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts import v342_clean_longtrain as longtrain
from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts import v343_longrun_selection as selection
from scripts.v180_dinomaly3_dinov3_csig import _extract_patch_tokens
from scripts.build_b_submission import _test_index
from scripts.v283_supervised_dinomaly_longtrain import sha256_file, top_mean
from scripts.v330_raw_response import audit_scores, score_views
from scripts.v340_clean_submission import combine_raw
from scripts.v344_longrun_submission import load_verified_bundle, require_full_b_index
from scripts.v348_clean_memory_core import encode_evidence, fuse_memory, normal_quantile_scale
from scripts.v349_clean_memory_bank import (load_verified_bank, preprocessing_environment_identity,
                                             source_inputs)
from scripts.v350_scaled_evidence import SCHEMA as SCALED_SCHEMA
from scripts.v351_source_calibration import ResidentCosineMemory
from scripts.v355_seen_category_calibration import category_feature_row_indices
from scripts.v353_highres_development import _calibration

ROOT = (Path.home() / 'csig')
GATE = ROOT / 'outputs/v339_independent_clean_replay_final_20260910/clean_selection_gate.json'
CALIBRATION_ROOT = ROOT / 'outputs/v351_source_calibration_20260911'
SUPPORTED_INPUT_SIZES = (560, 672)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def required_free_gib(omit_raw):
    return 2 if omit_raw else 10


def primary_score(raw):
    return top_mean(np.asarray(raw, dtype=np.float32), .005)


def memory_route(category, seen_categories):
    return 'category' if str(category) in set(seen_categories) else 'global'


def build_view_output(reference, memory, *, scale):
    fused = fuse_memory(reference, memory, scale=float(scale)).astype(np.float32)
    return {'mask_raw': fused, 'mask_png': encode_evidence(fused),
            'reference_view_score': primary_score(reference)}


def write_mask(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encode_evidence(np.asarray(raw, dtype=np.float32)), mode='L').save(path)


def validate_submission_contract(report):
    if (report.get('status') != 'ok' or report.get('version') != 'v356_seen_memory_submission_v1'
            or (report.get('groups'), report.get('masks'), report.get('entries')) != (2250, 11250, 11251)
            or report.get('mask_schema') != SCALED_SCHEMA
            or report.get('classification_source') != 'reference_ensemble_top005_top2'
            or report.get('test_b_labels_accessed') is not False
            or report.get('score_audit', {}).get('passed') is not True
            or report.get('score_audit', {}).get('rows') != 2250
            or report.get('score_audit', {}).get('unique') != 2250):
        raise ValueError('Invalid V356 submission contract')
    return True


def _write_zip(package, target):
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(Path(package).rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(package).as_posix())


def build_stream_package(groups, batches, output, provenance, save_raw=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    require_full_b_index(groups)
    expected = {(group, view) for group, views in groups.items() for view in views}
    package = output / 'package'; raw_root = output / 'raw'
    package.mkdir(parents=True); seen = set(); scores = defaultdict(dict)
    for categories, samples, views, maps, reference_scores in batches:
        if not (len(categories) == len(samples) == len(views) == len(maps) == len(reference_scores)):
            raise ValueError('B batch alignment error')
        for category, sample, view, raw, ref_score in zip(categories, samples, views, maps, reference_scores):
            category, sample, view = str(category), str(sample), str(view)
            key = (f'{category}/{sample}', view)
            if key not in expected or key in seen:
                raise ValueError('Unknown or duplicate B view')
            raw = np.asarray(raw, dtype=np.float32)
            if raw.shape != (448, 448) or not np.isfinite(raw).all() or np.any(raw < 0):
                raise ValueError('Invalid scaled mask response')
            write_mask(package / 'predicted_masks' / key[0] / f'{view}_mask.png', raw)
            if save_raw:
                raw_path = raw_root / key[0] / f'{view}.npz'; raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(raw_path, raw_evidence=raw, reference_view_score=np.float64(ref_score),
                                    schema=np.asarray(SCALED_SCHEMA), category=np.asarray(category),
                                    sample=np.asarray(sample), view=np.asarray(view))
            scores[key[0]][view] = float(ref_score); seen.add(key)
        if len(seen) % 500 == 0 or len(seen) == len(expected):
            print(f'v356_saved={len(seen)}/{len(expected)}', flush=True)
    if seen != expected:
        raise ValueError('Incomplete V356 coverage')
    csv_path = package / 'submission.csv'
    with csv_path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['group_folder', 'anomaly_score'])
        for group in sorted(groups):
            if set(scores[group]) != set(groups[group]): raise ValueError('Incomplete group scores')
            writer.writerow([group, f'{score_views([scores[group][v] for v in groups[group]]):.12f}'])
    with csv_path.open(encoding='utf-8', newline='') as handle:
        numeric = audit_scores({row['group_folder']: float(row['anomaly_score'])
                                for row in csv.DictReader(handle)})
    if not numeric['passed']: raise ValueError('V356 scores are degenerate')
    staging = output / 'submission.zip.part'; zip_path = output / 'submission.zip'
    _write_zip(package, staging)
    with zipfile.ZipFile(staging) as archive:
        if len(archive.namelist()) != 11251 or archive.testzip() is not None:
            raise ValueError('V356 ZIP membership/CRC mismatch')
    os.link(staging, zip_path); staging.unlink()
    report = {'status': 'ok', 'version': 'v356_seen_memory_submission_v1',
              'zip_path': str(zip_path), 'sha256': sha256_file(zip_path), 'bytes': zip_path.stat().st_size,
              'groups': 2250, 'masks': 11250, 'entries': 11251, 'mask_schema': SCALED_SCHEMA,
              'classification_source': 'reference_ensemble_top005_top2', 'test_b_labels_accessed': False,
              'raw_evidence_persisted': bool(save_raw),
              'score_audit': numeric, 'provenance': provenance,
              'policy': 'V344 reference ensemble scores; 672 category memory for seen categories and global memory for unseen categories; fixed 90/10 mask fusion',
              'limitations': ['B score is unknown until upload.', 'High-resolution category-memory quality is not independently measurable on B labels.']}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    validate_submission_contract(report)
    return report


def _load_category_memory(bank_root):
    bank, bank_report = load_verified_bank(bank_root)
    with np.load(Path(bank_root) / 'memory.npz', allow_pickle=False) as data:
        members = data['members'].astype(str).tolist()
    indices = category_feature_row_indices(members)
    if set(indices) != set(bank_report['metadata']['partition']['categories']):
        raise ValueError('Seen category bank coverage mismatch')
    return bank, indices, bank_report


def run(args):
    import torch

    output = Path(args.output_root).resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output.exists() or args.input_size not in SUPPORTED_INPUT_SIZES:
        raise ValueError('A new V356 output directory is required')
    if shutil.disk_usage(ROOT).free < required_free_gib(args.omit_raw) * 1024 ** 3:
        raise RuntimeError(f'{required_free_gib(args.omit_raw)}GiB free disk required for V356 output')
    groups = _test_index(ROOT / 'Test_B.zip', 'Test_B'); require_full_b_index(groups)
    bundle = load_verified_bundle(ROOT / 'outputs/v343_verified_longrun_20260911', GATE)
    bank_root = ROOT / 'outputs/v352_clean_memory_bank_env_20260911' / f'i{args.input_size}'
    category_calibration_path = ROOT / 'outputs/v355_seen_category_calibration_560_20260912' / f'seen_category_i{args.input_size}' / 'manifest.json'
    if args.input_size == 672:
        category_calibration_path = ROOT / 'outputs/v355_seen_category_calibration_retry_20260912/seen_category_i672/manifest.json'
    global_bank, category_indices, bank_report = _load_category_memory(bank_root)
    with np.load(bank_root / 'memory.npz', allow_pickle=False) as data:
        members = data['members'].astype(str).tolist()
    seen_categories = sorted(category_indices)
    category_bank = {category: global_bank[rows] for category, rows in category_indices.items()}
    global_memory = ResidentCosineMemory(global_bank, device='cuda:0', chunk_size=8192)
    category_memory = {category: ResidentCosineMemory(value, device='cuda:0', chunk_size=8192)
                       for category, value in category_bank.items()}
    del global_bank, category_bank
    category_calibration = _read(category_calibration_path)
    global_calibration = _calibration(CALIBRATION_ROOT / f'memory_i{args.input_size}/manifest.json', 'memory')
    reference_calibrations = [_calibration(CALIBRATION_ROOT / f'reference_fold{i}/manifest.json', 'reference')
                              for i in range(4)]
    reference_q99 = float(np.median([item['calibration']['q99'] for item in reference_calibrations]))
    q99 = {category: float(item['q99']) for category, item in category_calibration['categories_detail'].items()}
    q99['__global__'] = float(global_calibration['calibration']['q99'])
    deps = adapter._import_official_modules(); models = []
    for fold, record in enumerate(bundle['records']):
        choice = record['selected']; checkpoint = Path(choice['checkpoint'])
        if sha256_file(checkpoint) != choice['checkpoint_sha256']: raise ValueError('Reference checkpoint changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model, _ = adapter._build_model(deps, backbone='dinov3_vit_base_16', dropout=.4, reduce_dim=256,
                                        context_aware_recenter=True, mask_neighbor_size=0,
                                        weights_dir=ROOT / 'Dinomaly2_dinov3/backbones/weights')
        model.load_state_dict(state['model'], strict=True); del state
        models.append(model.to(f'cuda:{fold}').eval()); models[-1].requires_grad_(False)
    reference_dataset = adapter.CsigZipDataset(ROOT / 'Test_B.zip', 'Test_B', adapter._build_transform(448, 448), include_metadata=True)
    high_dataset = adapter.CsigZipDataset(ROOT / 'Test_B.zip', 'Test_B', adapter._build_transform(args.input_size, args.input_size), include_metadata=True)
    kwargs = dict(batch_size=4, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    ref_loader = torch.utils.data.DataLoader(reference_dataset, **kwargs)
    high_loader = torch.utils.data.DataLoader(high_dataset, **kwargs)
    test_sha = sha256_file(ROOT / 'Test_B.zip')
    def worker(fold, images):
        torch.cuda.set_device(fold)
        with torch.inference_mode():
            encoded, decoded = models[fold](images.to(f'cuda:{fold}', non_blocking=True))
            return trainer.differentiable_anomaly_map(encoded, decoded, 448).float().cpu().numpy()[:, 0]
    def batches():
        with ThreadPoolExecutor(max_workers=4) as pool:
            for base, high in zip(ref_loader, high_loader):
                futures = [pool.submit(worker, fold, base[0]) for fold in range(4)]
                references = combine_raw([future.result() for future in futures])
                tokens, grid = _extract_patch_tokens(models[0].encoder,
                    high[0].to('cuda:0', non_blocking=True), [3, 7, 11])
                expected_grid = (args.input_size // 16, args.input_size // 16)
                if grid != expected_grid: raise ValueError('B high-resolution grid changed')
                raw_maps = []
                reference_scores = []
                for index, category in enumerate(high[1]):
                    category = str(category); route = memory_route(category, seen_categories)
                    resident = category_memory[category] if route == 'category' else global_memory
                    distances = resident.distances(tokens[index:index + 1])[0].reshape(grid)
                    memory_map = __import__('scripts.v348_clean_memory_core', fromlist=['resize_raw_map']).resize_raw_map(distances)
                    scale = normal_quantile_scale(reference_q99, q99[category] if route == 'category' else q99['__global__'])
                    result = build_view_output(references[index], memory_map, scale=scale)
                    raw_maps.append(result['mask_raw']); reference_scores.append(result['reference_view_score'])
                yield list(high[1]), list(high[2]), list(high[3]), raw_maps, reference_scores
    provenance = {'kind': 'v356_seen_category_memory', 'test_zip_sha256': test_sha,
                  'input_size': args.input_size,
                  'bank_sha256': bank_report['bank_sha256'], 'bank_manifest_sha256': sha256_file(bank_root / 'manifest.json'),
                  'category_calibration_sha256': sha256_file(category_calibration_path),
                  'reference_calibration_q99': reference_q99, 'seen_categories': seen_categories,
                  'preprocess_environment': preprocessing_environment_identity(),
                  'selected_checkpoints': [record['selected'] for record in bundle['records']],
                  'selection_record_hashes': bundle['selection_record_hashes'],
                  'code_hashes': {'v356_seen_memory_submission.py': sha256_file(Path(__file__)),
                                  'v350_scaled_evidence.py': sha256_file(PROJECT / 'scripts/v350_scaled_evidence.py'),
                                  'v348_clean_memory_core.py': sha256_file(PROJECT / 'scripts/v348_clean_memory_core.py')},
                  'test_b_labels_accessed': False}
    report = build_stream_package(groups, batches(), output, provenance, save_raw=not args.omit_raw)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--input-size', type=int, choices=SUPPORTED_INPUT_SIZES, default=672)
    parser.add_argument('--omit-raw', action='store_true')
    run(parser.parse_args())
