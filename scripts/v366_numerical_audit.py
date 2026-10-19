"""Bounded independent V366 replay: 50 fixed seen objects, no training or B labels.

Run only after the final candidate exists and its producer has released all four
GPUs. Writes one small report; never builds an archive or persists raw responses.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
from PIL import Image
from scripts import v366_release_audit as parent
from scripts.v283_supervised_dinomaly_longtrain import sha256_file

ROOT = (Path.home() / 'csig')
FROZEN = ROOT / 'outputs/v362_category_memory_B_560_package_only_retry_20260912'
PRODUCER_SHA = '55c78ab3eb2865e55ec66a6661ec146958da9b5ed60606660e2954bf588a3543'
DONOR = ROOT / 'outputs/v365_ensemble_category_calibration_20260912/manifest.json'
DONOR_SHA = '08c7e4b90085b321138edb19117e46fdd8b859478343d0ec05d0b7b6d30a9186'
OVERLAY = ROOT / 'outputs/v365_clean_calibration_overlay_20260912.json'
OVERLAY_SHA = 'e797669fe2793f8a764da26629e907f7c00093fc025640eb173ed34722b119c1'
OLD_CAL = ROOT / 'outputs/v355_seen_category_calibration_560_20260912/seen_category_i560/manifest.json'
OLD_CAL_SHA = '928fd5d069a7088a7064bf87e6efd653663914b8d18f200d697d4211d80777f6'
BANK = ROOT / 'outputs/v352_clean_memory_bank_env_20260911/i560'
CONTRACT_SHA = parent.CONTRACT_SHA


def verify_sources(*tables):
    sources = {}
    for table in tables:
        for path, digest in table.items():
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError(f'Invalid source identity: {path}')
            if path in sources and sources[path] != digest:
                raise ValueError(f'Conflicting source identity: {path}')
            sources[path] = digest
    for path, digest in sources.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f'Source identity changed: {path}')
    return sources


def read_verified(path, digest):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f'Source identity changed: {path}')
    return json.loads(payload, object_pairs_hook=unique)


def selected_batches(records, contract):
    selected = set(parent.sample_groups(contract, CONTRACT_SHA))
    expected = [(g, v) for g in sorted(contract['groups']) for v in contract['groups'][g]]
    if len(records) != len(expected):
        raise ValueError('Incomplete full B dataset index')
    for record, (group, view) in zip(records, expected):
        actual = tuple(getattr(record, k, None) for k in ('group', 'view', 'category', 'sample', 'member'))
        if actual != (group, view, *group.split('/'), f'Test_B/{group}/{view}.png'):
            raise ValueError('Full B dataset metadata/order differs from contract')
    # Retain neighbouring, non-target views so model batch arithmetic is unchanged.
    for start in range(0, len(records), 4):
        indices = list(range(start, min(start + 4, len(records))))
        positions = [p for p, row in enumerate(indices) if records[row].group in selected]
        if positions:
            yield indices, positions


def audit_core(candidate, baseline, contract, responses, scales):
    groups = parent.sample_groups(contract, CONTRACT_SHA)
    if set(scales) != set(contract['seen_categories']):
        raise ValueError('Scale category coverage differs from contract')
    payload = baseline.read('submission.csv')
    if candidate.read('submission.csv') != payload:
        raise ValueError('Classification CSV bytes changed')
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8')))
    rows = list(reader)
    if (reader.fieldnames != ['group_folder', 'anomaly_score'] or len(rows) != len(contract['groups'])
            or any(set(r) != {'group_folder', 'anomaly_score'} for r in rows)
            or len({r['group_folder'] for r in rows}) != len(rows)
            or {r['group_folder'] for r in rows} != set(contract['groups'])):
        raise ValueError('Classification CSV group coverage mismatch')
    scores = {r['group_folder']: float(r['anomaly_score']) for r in rows}
    if not all(math.isfinite(s) and 0 <= s <= 1 for s in scores.values()):
        raise ValueError('Invalid classification score')
    expected = {(g, v) for g in groups for v in '01234'}
    observed, view_scores = set(), {g: {} for g in groups}
    for group, view, reference, memory in responses:
        key = group, view
        if key not in expected or key in observed:
            raise ValueError(f'Unknown or duplicate sampled view: {key}')
        name = f'predicted_masks/{group}/{view}_mask.png'
        for label, archive, scale_key in [('candidate', candidate, 'scale'), ('baseline', baseline, 'old_scale')]:
            pixels = parent.independent_png_pixels(reference, memory, scales[group.split('/')[0]][scale_key])
            if sum(info.filename == name for info in archive.infolist()) != 1:
                raise ValueError(f'Missing or duplicate sampled PNG: {name}')
            with Image.open(io.BytesIO(archive.read(name))) as image:
                if image.format != 'PNG' or image.mode != 'L' or image.size != (448, 448):
                    raise ValueError(f'Invalid sampled PNG: {name}')
                decoded = np.array(image)
            errors = np.abs(pixels.astype(np.int16) - decoded.astype(np.int16))
            if np.any(errors):
                raise ValueError(f'{label} pixel mismatch: {name}; pixels={np.count_nonzero(errors)}, max={errors.max()}')
        flat = np.asarray(reference, np.float32).ravel()
        count = math.ceil(flat.size * .005)
        view_scores[group][view] = float(np.partition(flat, flat.size - count)[-count:].mean())
        observed.add(key)
    if observed != expected:
        raise ValueError(f'Incomplete sampled view coverage: {len(observed)}/250')
    details = {}
    for group in groups:
        top = np.sort(np.maximum(np.array(list(view_scores[group].values()), np.float64), 0))[-2:].mean()
        actual = float(top / (top + .1))
        error = abs(actual - scores[group])
        if not math.isfinite(error) or error > 1e-6:
            raise ValueError(f'Classification group score mismatch: {group}; error={error}')
        details[group] = {'baseline_score': scores[group], 'recomputed_score': actual, 'abs_error': error}
    return dict(status='passed', sample_groups=groups, sampled_classes=50, target_views=250,
                candidate_masks_checked=250, baseline_masks_checked=250,
                candidate_mismatched_pixels=0, baseline_mismatched_pixels=0,
                score_groups_checked=50, score_tolerance=1e-6,
                score_max_abs_error=max(r['abs_error'] for r in details.values()), group_scores=details,
                sampling_seed_sha256=CONTRACT_SHA, scope='Fixed numerical spot audit, not B quality validation',
                new_training=False, test_b_labels_accessed=False, raw_evidence_persisted=False)


def checked_scales(frozen, donor, overlay, old, seen):
    if any(set(record['categories_detail']) != seen for record in (donor, overlay, old)):
        raise ValueError('Calibration category coverage mismatch')
    result = {}
    for category in sorted(seen):
        item = overlay['categories_detail'][category]
        old_q99 = old['categories_detail'][category]['q99']
        values = [item['reference_q99'], item['memory_q99'], frozen['reference_calibration_q99'], old_q99]
        if not all(math.isfinite(v) and 1e-8 < v <= 2.000001 for v in values):
            raise ValueError(f'Invalid calibration q99: {category}')
        if (item['scale'] != values[0] / values[1]
                or abs(donor['categories_detail'][category]['memory_q99'] - old_q99) > 1e-6):
            raise ValueError(f'Calibration q99/scale mismatch: {category}')
        result[category] = {'scale': item['scale'], 'old_scale': values[2] / old_q99}
    return result


def bounded_responses(context):
    import os
    import subprocess
    import torch
    import torch.nn.functional as functional
    from scripts.v180_dinomaly3_dinov3_csig import CsigZipDataset, _build_transform, _extract_patch_tokens
    from scripts.v283_supervised_dinomaly_longtrain import differentiable_anomaly_map
    from scripts.v351_source_calibration import ResidentCosineMemory
    from scripts.v365_ensemble_category_calibration import load_models

    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    if any(int(pid.strip()) != os.getpid() for pid in active.splitlines() if pid.strip()):
        raise RuntimeError('Numerical audit requires idle GPUs; another compute process is active')
    environment = context['environment']
    if torch.cuda.device_count() != 4 or any(torch.cuda.mem_get_info(i)[0] < 8 * 2**30 for i in range(4)):
        raise RuntimeError('Four idle GPUs with8GiB free required')
    for fold in range(4):
        if (torch.cuda.get_device_name(fold) != environment['cuda_device']
                or list(torch.cuda.get_device_capability(fold)) != environment['cuda_capability']):
            raise ValueError('GPU differs from frozen numerical environment')
    torch.set_num_threads(4)
    torch.manual_seed(36501)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    reference = CsigZipDataset(ROOT / 'Test_B.zip', 'Test_B', _build_transform(448, 448), include_metadata=True)
    high = CsigZipDataset(ROOT / 'Test_B.zip', 'Test_B', _build_transform(560, 560), include_metadata=True)
    try:
        batches = list(selected_batches(reference.records, context['contract']))
        if batches != list(selected_batches(high.records, context['contract'])):
            raise ValueError('Reference/high dataset index mismatch')
        models = load_models(context['bundle'])
        resident, resident_category = None, None
        def images_at(dataset, indices):
            images = []
            for row in indices:
                image, category, sample, view = dataset[row]
                record = dataset.records[row]
                if (category, sample, view) != (record.category, record.sample, record.view):
                    raise ValueError('Loaded image metadata changed')
                images.append(image)
            return torch.stack(images)
        def worker(fold, images):
            torch.cuda.set_device(fold)
            with torch.inference_mode():
                enc, dec = models[fold](images.to(f'cuda:{fold}', non_blocking=True))
                return differentiable_anomaly_map(enc, dec, 448).float().cpu().numpy()[:, 0]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for indices, positions in batches:
                images, high_images = images_at(reference, indices), images_at(high, indices)
                pending = [pool.submit(worker, fold, images) for fold in range(4)]
                raw = np.stack([future.result() for future in pending]).astype(np.float32)
                if (raw.shape != (4, len(indices), 448, 448) or not np.isfinite(raw).all()
                        or raw.min() < -1e-6 or raw.max() > 2.000001):
                    raise ValueError('Invalid four-model raw batch')
                median = np.median(np.maximum(raw, 0), axis=0).astype(np.float32)
                with torch.inference_mode():
                    tokens, grid = _extract_patch_tokens(models[0].encoder, high_images.to('cuda:0'), [3, 7, 11])
                    if grid != (35, 35):
                        raise ValueError('Expected560 feature grid')
                    for position in positions:
                        record = reference.records[indices[position]]
                        if record.category != resident_category:
                            resident = None
                            resident = ResidentCosineMemory(context['bank'][context['category_rows'][record.category]],
                                                          device='cuda:0', chunk_size=8192)
                            resident_category = record.category
                        distance = resident.distances(tokens[position:position + 1])[0].reshape(grid)
                        memory = functional.interpolate(torch.from_numpy(distance)[None, None], size=(448, 448),
                                                        mode='bilinear', align_corners=True)[0, 0].numpy()
                        yield record.group, record.view, median[position], memory
        context['batch_count'] = len(batches)
        context['inferred_views_including_neighbours'] = sum(len(rows) for rows, _ in batches)
    finally:
        for dataset in (reference, high):
            if dataset._archive is not None:
                dataset._archive.close()


def main(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.contract_sha256 != CONTRACT_SHA:
        raise ValueError('Only the immutable official directory contract is permitted')
    fixed = {str(args.candidate): args.candidate_sha256, str(args.producer_audit): args.producer_audit_sha256,
             str(args.contract): CONTRACT_SHA, str(FROZEN / 'submission.zip'): parent.BASELINE_SHA,
             str(FROZEN / 'manifest.json'): PRODUCER_SHA, str(DONOR): DONOR_SHA,
             str(OVERLAY): OVERLAY_SHA, str(OLD_CAL): OLD_CAL_SHA}
    verify_sources(fixed)
    contract = read_verified(args.contract, CONTRACT_SHA)
    audit = read_verified(args.producer_audit, args.producer_audit_sha256)
    frozen_manifest = read_verified(FROZEN / 'manifest.json', PRODUCER_SHA)
    frozen = frozen_manifest['provenance']
    donor, old = read_verified(DONOR, DONOR_SHA), read_verified(OLD_CAL, OLD_CAL_SHA)
    parent.sample_groups(contract, CONTRACT_SHA)
    provenance = audit['provenance']
    if (audit.get('status') != 'candidate_unscored' or audit.get('sha256') != args.candidate_sha256
            or audit.get('bytes') != args.candidate.stat().st_size
            or any(audit.get(k) is not False for k in ('new_training', 'test_b_labels_accessed', 'raw_evidence_persisted', 'auto_upload'))
            or frozen_manifest.get('sha256') != parent.BASELINE_SHA
            or contract['seen_source_sha256'] != parent.BASELINE_SHA
            or contract['producer_manifest_sha256'] != PRODUCER_SHA
            or frozen.get('test_b_labels_accessed') is not False):
        raise ValueError('Candidate/frozen producer identity mismatch')
    from scripts.build_b_submission import _test_index
    from scripts.v344_longrun_submission import load_verified_bundle
    from scripts.v349_clean_memory_bank import source_inputs, preprocessing_environment_identity, PARTITION_SHA
    from scripts.v348_memory_smoke import TRAIN_SHA, WEIGHT_SHA
    from scripts.v356_seen_memory_submission import GATE, _load_category_memory
    from scripts import v365_clean_calibration_overlay as clean

    fixed.update({str(ROOT / 'Test_B.zip'): frozen['test_zip_sha256'], str(BANK / 'memory.npz'): frozen['bank_sha256'],
                  str(BANK / 'manifest.json'): frozen['bank_manifest_sha256'], str(clean.AUDIT_PATH): clean.AUDIT_SHA})
    sources = verify_sources(fixed, donor['source_files_sha256'], audit['source_files_sha256'])
    if (contract['test_zip_sha256'] != frozen['test_zip_sha256'] or contract['train_zip_sha256'] != TRAIN_SHA
            or _test_index(ROOT / 'Test_B.zip', 'Test_B') != {g: tuple(v) for g, v in contract['groups'].items()}):
        raise ValueError('Actual Test_B index/archive differs from immutable contract')
    overlay = clean.load_verified_overlay(OVERLAY, OVERLAY_SHA, donor, DONOR_SHA)
    train, weight, _, partition = source_inputs()
    bundle = load_verified_bundle(ROOT / 'outputs/v343_verified_longrun_20260911', GATE)
    selected = [record['selected'] for record in bundle['records']]
    bank, category_rows, bank_report = _load_category_memory(BANK)
    environment = preprocessing_environment_identity()
    seen = set(contract['seen_categories'])
    if (len(selected) != 4 or any(record['selected_checkpoints'] != selected for record in (frozen, donor, overlay, provenance))
            or any(record['bank_sha256'] != frozen['bank_sha256'] for record in (donor, overlay, old, provenance, bank_report))
            or any(record['preprocess_environment'] != environment for record in (frozen, donor, old, provenance))
            or bundle['selection_record_hashes'] != frozen['selection_record_hashes']
            or donor['partition'] != partition or set(partition['categories']) != seen
            or set(category_rows) != seen or set(frozen['seen_categories']) != seen
            or bank_report['metadata']['input_size'] != 560 or bank_report['trace']['grid'] != [35, 35]
            or frozen['category_calibration_sha256'] != OLD_CAL_SHA
            or old['bank_manifest_sha256'] != frozen['bank_manifest_sha256']
            or overlay['effective_calibration_views'] != 995 or overlay['excluded_groups'] != ['toy_tire/S0020']):
        raise ValueError('Frozen calibration/model/bank/environment identity mismatch')
    required = dict(baseline_sha256=parent.BASELINE_SHA, producer_sha256=PRODUCER_SHA, calibration_sha256=DONOR_SHA,
                    clean_overlay_sha256=OVERLAY_SHA, reference_input_size=448, memory_input_size=560, global_dataset_batch_size=4)
    if any(provenance.get(k) != v for k, v in required.items()):
        raise ValueError('Candidate numerical provenance mismatch')
    scales = checked_scales(frozen, donor, overlay, old, seen)
    if set(audit['scales']) != seen or any(audit['scales'][c].get(k) != v for c in scales for k, v in scales[c].items()):
        raise ValueError('Candidate category scales differ from independently pinned scales')
    numerical = {str(train): TRAIN_SHA, str(weight): WEIGHT_SHA, str(GATE): bundle['source_gate_sha256'],
                 str(ROOT / 'outputs/v348_memory_smoke_reviewed_20260911/source_partition.json'): PARTITION_SHA,
                 **{r['checkpoint']: r['checkpoint_sha256'] for r in selected}, **bundle['selection_record_hashes']}
    for name in ('v366_numerical_audit.py', 'v366_release_audit.py', 'v180_dinomaly3_dinov3_csig.py',
                 'v365_clean_calibration_overlay.py'):
        numerical[str(PROJECT / 'scripts' / name)] = sha256_file(PROJECT / 'scripts' / name)
    sources = verify_sources(sources, numerical, old['runtime_source_hashes'], bank_report['metadata']['runtime_source_hashes'],
                             *[r['long_config']['runtime_source_hashes'] for r in bundle['records']])
    context = dict(contract=contract, bundle=bundle, bank=bank, category_rows=category_rows, environment=environment)
    responses = bounded_responses(context)
    try:
        with zipfile.ZipFile(args.candidate) as candidate, zipfile.ZipFile(FROZEN / 'submission.zip') as baseline:
            report = audit_core(candidate, baseline, contract, responses, scales)
    finally:
        responses.close()
        verify_sources(sources)
        if preprocessing_environment_identity() != environment:
            raise ValueError('Numerical environment changed during audit')
    report.update(version='v366_independent_numerical_audit_v1', candidate_sha256=args.candidate_sha256,
                  producer_audit_sha256=args.producer_audit_sha256, global_dataset_batch_size=4,
                  batch_count=context['batch_count'], inferred_views_including_neighbours=context['inferred_views_including_neighbours'],
                  sources_before_sha256=sources, sources_after_sha256=sources, source_hashes_unchanged=True)
    payload = json.dumps(report, indent=2, allow_nan=False)
    if len(payload.encode('utf-8')) > 1_000_000:
        raise ValueError('Numerical report exceeds1MB budget')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(payload)
    return report


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'producer-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('candidate-sha256', 'producer-audit-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--contract', type=Path, default=ROOT / 'outputs/v366_source_contract.json')
    parser.add_argument('--contract-sha256', default=CONTRACT_SHA)
    return parser


if __name__ == '__main__':
    try:
        result = main(argument_parser().parse_args())
        print(json.dumps({k: result[k] for k in ('status', 'sampled_classes', 'target_views', 'score_max_abs_error')}))
    except (ValueError, OSError, RuntimeError, KeyError, zipfile.BadZipFile) as error:
        print(json.dumps({'status': 'failed', 'error': str(error), 'new_training': False}), flush=True)
        raise SystemExit(1)
