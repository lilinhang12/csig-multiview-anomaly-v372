"""Fixed 50-object V372 density spot replay; run only after the producer exits.

Rebuild normal banks/calibration, then infer 250 target views plus batch neighbours.
Writes only a new JSON report: no training, labels, raw disk cache or ZIP export.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
from PIL import Image
from scripts import v366_release_audit as parent
from scripts import v372_dense_inference as density
from scripts.v366_numerical_audit import read_verified, verify_sources, selected_batches
from scripts.v283_supervised_dinomaly_longtrain import sha256_file
from scripts.v348_clean_memory_core import normal_quantile_scale, resize_raw_map

ROOT, CONTRACT_SHA = density.ROOT, parent.CONTRACT_SHA
REFERENCE_Q99 = 0.04275937380269168


def checked_sources(actual, required):
    density.v366.require_sources(actual, required)
    return verify_sources(required, actual)


def checked_scales(records, old, seen):
    if set(records) != set(seen) or set(old['categories_detail']) != set(seen):
        raise ValueError('Calibration category coverage mismatch')
    scales = {}
    for category, record in records.items():
        q99 = old['categories_detail'][category]['q99']
        old_scale = normal_quantile_scale(REFERENCE_Q99, q99)
        new_scale = normal_quantile_scale(REFERENCE_Q99, record['dense_q99'])
        hashes = record['calibration_hashes']
        if (record['reference_q99'] != REFERENCE_Q99 or record['old_q99'] != q99
                or record['old_scale'] != old_scale or record['new_scale'] != new_scale
                or not math.isfinite(record['sparse_q99']) or abs(record['sparse_q99'] - q99) > 1e-6
                or record['dense_q99'] > record['sparse_q99'] or record['global_calibration_batch_size'] != 8
                or record['calibration_replay_exact'] is not True or len(hashes) != 2 or hashes[0] != hashes[1]
                or set(hashes[0]) != {'sparse', 'dense'}
                or any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in hashes[0].values())):
            raise ValueError(f'Prepared calibration q99/scale/hash mismatch: {category}')
        scales[category] = dict(old_scale=old_scale, new_scale=new_scale)
    return scales


def verify_calibration(members, sparse, dense, record):
    density.v365.checked_members(members)
    if (members != record['calibration_members'] or any(np.shape(a) != (20, 448, 448) for a in (sparse, dense))
            or np.any(dense > sparse)):
        raise ValueError('Normal calibration member/shape/minimum mismatch')
    hashes = {}
    for label, raw in (('sparse', sparse), ('dense', dense)):
        if (density.category_quantile(raw) != record[label + '_q99']
                or np.max(raw) > 2.000001):
            raise ValueError(f'Normal calibration q99 mismatch: {label}')
        hashes[label] = density.v365.response_digest(raw)
    if record['calibration_hashes'] != [hashes, hashes]:
        raise ValueError('Normal calibration response hashes differ from producer')
    return hashes


def verify_sample(responses, expected_scores, scales):
    """Rows are (group, view, reference, sparse, dense, candidate PNG, baseline PNG)."""
    if not expected_scores or any(not math.isfinite(s) or not 0 <= s <= 1 for s in expected_scores.values()):
        raise ValueError('Invalid sampled classification score')
    expected = {(g, v) for g in expected_scores for v in '01234'}
    observed, scores = set(), {g: [] for g in expected_scores}
    for group, view, reference, sparse, dense, candidate, baseline in responses:
        key = group, view
        if key not in expected or key in observed:
            raise ValueError(f'Unknown or duplicate sampled view: {key}')
        arrays = [np.asarray(a, np.float32) for a in (reference, sparse, dense)]
        if (any(a.shape != (448, 448) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 2.000001
                for a in arrays) or np.any(arrays[2] > arrays[1] + 1e-6)):
            raise ValueError('Invalid sampled response values or dense minimum')
        scale = scales[group.split('/')[0]]
        for label, payload, memory, factor in [('candidate', candidate, dense, scale['new_scale']),
                                               ('baseline', baseline, sparse, scale['old_scale'])]:
            pixels = parent.independent_png_pixels(reference, memory, factor)
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != 'PNG' or image.mode != 'L' or image.size != (448, 448):
                    raise ValueError(f'Invalid {label} sampled PNG: {key}')
                decoded = np.array(image)
            errors = np.abs(pixels.astype(np.int16) - decoded.astype(np.int16))
            if np.any(errors):
                raise ValueError(f'{label} pixel mismatch: {key}; pixels={np.count_nonzero(errors)}, max={errors.max()}')
        flat = arrays[0].ravel()
        count = math.ceil(flat.size * .005)
        scores[group].append(float(np.partition(flat, flat.size - count)[-count:].mean()))
        observed.add(key)
    if observed != expected:
        raise ValueError(f'Incomplete sampled view coverage: {len(observed)}/{len(expected)}')
    errors = {}
    for group, values in scores.items():
        top = np.sort(np.asarray(values, np.float64))[-2:].mean()
        error = abs(float(top / (top + .1)) - expected_scores[group])
        if not math.isfinite(error) or error > 1e-6:
            raise ValueError(f'Classification group score mismatch: {group}; error={error}')
        errors[group] = error
    return dict(status='passed', target_views=len(expected), candidate_masks_checked=len(observed),
                baseline_masks_checked=len(observed), candidate_mismatched_pixels=0, baseline_mismatched_pixels=0,
                score_groups_checked=len(scores), score_tolerance=1e-6, score_max_abs_error=max(errors.values()),
                group_score_abs_errors=errors, new_training=False, test_b_labels_accessed=False,
                raw_evidence_persisted=False, raw_disk_cache=False, zip_created=False)


def audit_core(candidate, baseline, contract, responses, scales):
    groups = parent.sample_groups(contract, CONTRACT_SHA)
    if set(scales) != set(contract['seen_categories']):
        raise ValueError('Scale category coverage differs from fixed sample')
    names = [Counter(a.namelist()) for a in (candidate, baseline)]
    if any(n['submission.csv'] != 1 for n in names):
        raise ValueError('Missing or duplicate classification CSV')
    payload = baseline.read('submission.csv')
    if candidate.read('submission.csv') != payload:
        raise ValueError('Classification CSV bytes changed')
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8')))
    if reader.fieldnames != ['group_folder', 'anomaly_score']:
        raise ValueError('Classification CSV columns changed')
    scores = {}
    for row in reader:
        group = row['group_folder']
        if group in groups:
            if group in scores or set(row) != {'group_folder', 'anomaly_score'}:
                raise ValueError('Duplicate or invalid sampled classification row')
            scores[group] = float(row['anomaly_score'])
    if set(scores) != set(groups):
        raise ValueError('Missing sampled classification groups')
    def payloads():
        for group, view, reference, sparse, dense in responses:
            if group not in groups or view not in tuple('01234'):
                raise ValueError('Unknown sampled view')
            name = f'predicted_masks/{group}/{view}_mask.png'
            if any(n[name] != 1 for n in names):
                raise ValueError(f'Missing or duplicate sampled PNG: {name}')
            yield group, view, reference, sparse, dense, candidate.read(name), baseline.read(name)
    result = verify_sample(payloads(), scores, scales)
    result.update(sample_groups=groups, sampled_classes=len(groups), sampling_seed_sha256=CONTRACT_SHA,
                  sampled_members=[f'Test_B/{g}/{v}.png' for g in groups for v in '01234'],
                  selected_by_test_score=False, csv_byte_identical=True)
    return result


def memory_maps(tokens, sparse, extra):
    old_maps, new_maps = [], []
    for p in range(len(tokens)):
        old = sparse.distances(tokens[p:p + 1])[0]
        additional = extra.distances(tokens[p:p + 1])[0]
        old_maps.append(resize_raw_map(old.reshape(35, 35)))
        new_maps.append(resize_raw_map(np.minimum(old, additional).reshape(35, 35)))
    return np.stack(old_maps), np.stack(new_maps)


def bounded_responses(context, category_records):
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    if any(int(pid.strip()) != os.getpid() for pid in active.splitlines() if pid.strip()):
        raise RuntimeError('Numerical audit requires idle GPUs; another compute process is active')
    import torch
    from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    from scripts.v180_dinomaly3_dinov3_csig import _extract_patch_tokens
    from scripts.v283_supervised_dinomaly_longtrain import differentiable_anomaly_map
    from scripts.v351_source_calibration import ResidentCosineMemory

    environment = context['environment']
    if torch.cuda.device_count() != 4 or any(torch.cuda.mem_get_info(i)[0] < 8 * 2**30 for i in range(4)):
        raise RuntimeError('Four idle GPUs with8GiB free required')
    if any(torch.cuda.get_device_name(i) != environment['cuda_device'] or
           list(torch.cuda.get_device_capability(i)) != environment['cuda_capability'] for i in range(4)):
        raise ValueError('GPU differs from frozen numerical environment')
    torch.set_num_threads(4)
    torch.manual_seed(34801)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    models = density.v365.load_models(context['bundle'])
    def check_models():
        if len(models) != 4 or any(any(m.training for m in model.modules()) or
                any(p.requires_grad or p.dtype != torch.float32 or p.device != torch.device(f'cuda:{i}')
                    for p in model.parameters()) for i, model in enumerate(models)):
            raise ValueError('Expected four frozen FP32 models')
    check_models()
    datasets = []
    def dataset(path, root, size):
        data = adapter.CsigZipDataset(path, root, adapter._build_transform(size, size), include_metadata=True)
        datasets.append(data)
        return data
    def images_at(data, rows):
        images = []
        for row in rows:
            image, category, sample, view = data[row]
            r = data.records[row]
            if (category, sample, view) != (r.category, r.sample, r.view):
                raise ValueError('Loaded image metadata changed')
            images.append(image)
        return torch.stack(images)
    def extract(data, rows):
        with torch.inference_mode():
            tokens, grid = _extract_patch_tokens(models[0].encoder, images_at(data, rows).to('cuda:0'), [3, 7, 11])
        if grid != (35, 35) or tuple(tokens.shape) != (len(rows), 1225, 2304) or tokens.dtype != torch.float32:
            raise ValueError('Expected FP32 encoder0 layers3/7/11 at560')
        return tokens
    def worker(fold, images):
        torch.cuda.set_device(fold)
        with torch.inference_mode():
            enc, dec = models[fold](images.to(f'cuda:{fold}', non_blocking=True))
            return differentiable_anomaly_map(enc, dec, 448).float().cpu().numpy()[:, 0]
    try:
        normal = dataset(context['train'], 'Train', 560)
        bank_indices = density.normal_indices(normal.records, context['partition'], 'bank')
        cal_indices = density.normal_indices(normal.records, context['partition'], 'calibration')
        if [normal.records[i].member for i in bank_indices] != context['bank_members']:
            raise ValueError('Original NPZ bank member order changed')
        reference, high = [dataset(ROOT / 'Test_B.zip', 'Test_B', size) for size in (448, 560)]
        batches = list(selected_batches(reference.records, context['contract']))
        if batches != list(selected_batches(high.records, context['contract'])):
            raise ValueError('Reference/high dataset index mismatch')
        context['batch_count'] = context['inferred_views_including_neighbours'] = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for category, record in context['category_records'].items():
                sparse_bank = context['bank'][context['category_rows'][category]]
                extra_bank, trace = density.build_extra(category, normal.records, bank_indices, sparse_bank,
                                                        lambda rows: extract(normal, rows))
                if any(record.get(k) != v for k, v in trace.items()):
                    raise ValueError(f'Normal bank trace/bytes differ from producer: {category}')
                sparse, extra = [ResidentCosineMemory(a, device='cuda:0', chunk_size=8192) for a in (sparse_bank, extra_bank)]
                members, old_cal, dense_cal = [], [], []
                for rows, positions in density.category_batches(normal.records, cal_indices, category):
                    pair = memory_maps(extract(normal, rows)[positions], sparse, extra)
                    old_cal.extend(pair[0]); dense_cal.extend(pair[1])
                    members.extend(normal.records[rows[p]].member for p in positions)
                hashes = verify_calibration(members, np.stack(old_cal), np.stack(dense_cal), record)
                category_records[category] = dict(bank_replay_exact=True, calibration_passes_recomputed=1,
                    calibration_hashes=hashes, reference_numerator=REFERENCE_Q99,
                    sparse_sha256=trace['sparse_sha256'], extra_sha256=trace['extra_sha256'])
                del old_cal, dense_cal, pair
                for rows, targets in batches:
                    positions = [p for p in targets if reference.records[rows[p]].category == category]
                    if not positions:
                        continue
                    images = images_at(reference, rows)
                    futures = [pool.submit(worker, fold, images) for fold in range(4)]
                    raw = np.stack([f.result() for f in futures]).astype(np.float32)
                    if (raw.shape != (4, len(rows), 448, 448) or not np.isfinite(raw).all()
                            or raw.min() < -1e-6 or raw.max() > 2.000001):
                        raise ValueError('Invalid four-model reference batch')
                    median = np.median(np.maximum(raw, 0), axis=0).astype(np.float32)
                    old, new = memory_maps(extract(high, rows)[positions], sparse, extra)
                    context['batch_count'] += 1
                    context['inferred_views_including_neighbours'] += len(rows)
                    for i, p in enumerate(positions):
                        r = reference.records[rows[p]]
                        yield r.group, r.view, median[p], old[i], new[i]
                del sparse, extra, sparse_bank, extra_bank
                print(f'numerical_category={category} bank=exact calibration=exact sampled_views=5', flush=True)
    finally:
        for data in datasets:
            if data._archive is not None:
                data._archive.close()
        check_models()


def main(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    frozen = density.v366
    fixed = {str(args.candidate): args.candidate_sha256, str(args.producer_audit): args.producer_audit_sha256,
             str(args.contract): CONTRACT_SHA, str(frozen.BASELINE): parent.BASELINE_SHA,
             str(frozen.PRODUCER): frozen.PRODUCER_SHA, str(frozen.OLD_CALIBRATION): frozen.LEGACY_CALIBRATION_SHA}
    verify_sources(fixed)
    contract = read_verified(args.contract, CONTRACT_SHA)
    audit = read_verified(args.producer_audit, args.producer_audit_sha256)
    groups = parent.sample_groups(contract, CONTRACT_SHA)
    context = density.preflight()
    provenance, replay = audit['provenance'], audit['replay']
    if (audit.get('status') != 'candidate_unscored' or audit.get('sha256') != args.candidate_sha256
            or audit.get('version') != 'v372_dense_v362_candidate_v1' or audit.get('bytes') != args.candidate.stat().st_size
            or any(audit.get(k) is not False for k in ('new_training', 'test_b_labels_accessed', 'raw_evidence_persisted', 'auto_upload'))
            or any(provenance.get(k) != v for k, v in context['provenance'].items())
            or provenance.get('source_hashes_after') != context['sources']
            or contract['seen_source_sha256'] != parent.BASELINE_SHA or contract['producer_manifest_sha256'] != frozen.PRODUCER_SHA
            or contract['test_zip_sha256'] != context['producer']['test_zip_sha256']
            or contract['train_zip_sha256'] != density.v349.TRAIN_SHA
            or {g: tuple(v) for g, v in contract['groups'].items()} != context['index']['groups']
            or set(contract['seen_categories']) != set(context['index']['seen'])
            or replay.get('old_scale_masks_checked') != 6250 or replay.get('reference_score_groups_checked') != 1250
            or any(replay.get(k) != 0 for k in ('old_scale_mismatched_masks', 'old_scale_mismatched_pixels',
                                              'old_scale_max_abs_error', 'reference_score_mismatched_groups'))
            or not 0 <= replay.get('reference_score_max_abs_error', float('nan')) <= 1e-6):
        raise ValueError('Candidate/frozen producer provenance or replay mismatch')
    sources = checked_sources(audit['source_files_sha256'], context['sources'])
    code = {str(PROJECT / 'scripts' / n): sha256_file(PROJECT / 'scripts' / n) for n in
            ('v372_numerical_audit.py', 'v366_numerical_audit.py', 'v366_release_audit.py')}
    sources = verify_sources(fixed, sources, code)
    scales = checked_scales(audit['category_records'], context['old'], contract['seen_categories'])
    context.update(contract=contract, category_records={c: audit['category_records'][c] for c in sorted(scales)})
    records = {}
    responses = bounded_responses(context, records)
    try:
        with zipfile.ZipFile(args.candidate) as candidate, zipfile.ZipFile(frozen.BASELINE) as baseline:
            report = audit_core(candidate, baseline, contract, responses, scales)
    finally:
        responses.close()
        after = verify_sources(sources)
        if density.v349.preprocessing_environment_identity() != context['environment']:
            raise ValueError('Numerical environment changed during audit')
    if set(records) != set(scales) or len(groups) != 50 or report['target_views'] != 250:
        raise ValueError('Incomplete50-category numerical replay')
    report.update(version='v372_independent_numerical_audit_v1', candidate_sha256=args.candidate_sha256,
        producer_audit_sha256=args.producer_audit_sha256, baseline_sha256=parent.BASELINE_SHA,
        reference_numerator=REFERENCE_Q99, category_records=records, global_normal_batch_size=8,
        global_dataset_batch_size=4, batch_count=context['batch_count'],
        inferred_views_including_neighbours=context['inferred_views_including_neighbours'],
        sources_before_sha256=sources, sources_after_sha256=after, source_hashes_unchanged=True,
        scope='Fixed numerical spot audit, not Test_B quality validation', auto_upload=False)
    payload = json.dumps(report, indent=2, allow_nan=False)
    if len(payload.encode()) > 1_000_000:
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
    return parser


if __name__ == '__main__':
    result = main(argument_parser().parse_args())
    print(json.dumps({k: result[k] for k in ('status', 'sampled_classes', 'target_views', 'score_max_abs_error')}))
