"""V366 candidate_unscored: clean-calibrated seen memory, frozen unseen/CSV.

No training, upload, B labels, extraction or raw cache. A verified clean overlay
is mandatory: toy_tire uses both q99 values fitted without its duplicate object;
other categories use the original V365 values. Old-scale replay always uses
the original V362 numerator AND denominator. Unseen PNG/CSV bytes are frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.v283_supervised_dinomaly_longtrain import sha256_file, top_mean
from scripts.v330_raw_response import score_views
from scripts.v340_clean_submission import combine_raw
from scripts.v348_clean_memory_core import encode_evidence, fuse_memory, normal_quantile_scale
from scripts.v357_v356_release import decode_png
from scripts.v363_route_alignment import validate_members, validate_png, _check_csv, _expected

ROOT = (Path.home() / 'csig')
FROZEN_ROOT = ROOT / 'outputs/v362_category_memory_B_560_package_only_retry_20260912'
BASELINE = FROZEN_ROOT / 'submission.zip'
BASELINE_SHA = 'a99079d7cced250a8f4a0043135d676f91cf600cb1cb287a85e35e1a0213363e'
PRODUCER = FROZEN_ROOT / 'manifest.json'
PRODUCER_SHA = '55c78ab3eb2865e55ec66a6661ec146958da9b5ed60606660e2954bf588a3543'
CALIBRATION = ROOT / 'outputs/v365_ensemble_category_calibration_20260912/manifest.json'
OLD_CALIBRATION = ROOT / 'outputs/v355_seen_category_calibration_560_20260912/seen_category_i560/manifest.json'
BANK_ROOT = ROOT / 'outputs/v352_clean_memory_bank_env_20260911/i560'
LEGACY_CALIBRATION_SHA = '928fd5d069a7088a7064bf87e6efd653663914b8d18f200d697d4211d80777f6'
LEGACY_V355_SHA = 'cb1beb0f233d21ab7ba7f340b001e2c1844fe6bc1aa9ac295493ad67e342268f'
LEGACY_BANK_SHA = 'dcb55dc7ae0ed7bf1a771c22359af8623df363fcb43d300091717eb83ad99c31'
VERSION = 'v366_calibrated_seen_submission_v1'
ZIP_BUDGET = 980_000_000
REPORT_BUDGET = 10_000_000


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid SHA256 digest')
    return value


def read_verified_json(path, expected_sha):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != _digest(expected_sha):
        raise ValueError(f'Source digest mismatch: {path}')
    return json.loads(payload, object_pairs_hook=unique)


def unchanged(sources):
    for path, expected in sources.items():
        if sha256_file(Path(path)) != _digest(expected):
            raise ValueError(f'Source changed or digest mismatch: {path}')


def make_index(groups, seen):
    from scripts.v363_route_alignment import validate_contract

    return validate_contract({
        'version': 'v363_route_source_contract_v1', 'train_groups': 1000,
        'groups': groups, 'seen_categories': list(seen),
        **{key: '0' * 64 for key in ('train_zip_sha256', 'test_zip_sha256',
                                   'baseline_sha256', 'seen_source_sha256', 'producer_manifest_sha256')}})


def validate_records(reference, high, index):
    expected = [(g, v) for g, views in index['groups'].items() for v in views]
    for records in (reference, high):
        if len(records) != len(expected):
            raise ValueError('Incomplete B dataset')
        for record, (group, view) in zip(records, expected):
            if (getattr(record, 'group', None) != group or getattr(record, 'view', None) != view
                    or getattr(record, 'category', None) != group.split('/')[0]
                    or getattr(record, 'sample', None) != group.split('/')[1]
                    or getattr(record, 'member', None) != f'Test_B/{group}/{view}.png'):
                raise ValueError('B dataset metadata/order differs from the blind index')


def seen_batches(records, seen):
    # Keep the original full-dataset batch boundaries, including mixed batches.
    for offset in range(0, len(records), 4):
        indices = list(range(offset, min(offset + 4, len(records))))
        positions = [i for i, row in enumerate(indices) if records[row].category in seen]
        if positions:
            yield indices, positions


def calibration_metadata_provenance(old, producer, cal, old_sha, bank_report):
    if old.get('input_size') == 560 and old.get('patch_grid') == [35, 35]:
        return {'legacy_metadata_exception': False}
    bank_report = bank_report or {}
    details = cal.get('categories_detail', {})
    legacy = (
        old_sha == producer.get('category_calibration_sha256') == LEGACY_CALIBRATION_SHA
        and old.get('code_hashes', {}).get('v355_seen_category_calibration.py') == LEGACY_V355_SHA
        and old.get('bank_sha256') == producer.get('bank_sha256') == bank_report.get('bank_sha256') == LEGACY_BANK_SHA
        and old.get('input_size') == 672 and 'patch_grid' not in old
        and bank_report.get('metadata', {}).get('input_size') == 560
        and bank_report.get('trace', {}).get('grid') == [35, 35]
        and cal.get('memory_input_size') == 560 and cal.get('passes') == 2
        and cal.get('response_replay') == 'exact_all_maps_all_categories'
        and len(details) == 50 and set(details) == set(old.get('categories_detail', {}))
        and all(item.get('memory_q99') == old['categories_detail'][c]['q99']
                and item.get('published_memory_q99_abs_drift') == 0 for c, item in details.items()))
    if not legacy:
        raise ValueError('Unrecognized frozen calibration resolution/metadata mismatch')
    return {'legacy_metadata_exception': True, 'frozen_manifest_sha256': old_sha,
            'historical_v355_sha256': LEGACY_V355_SHA, 'bank_sha256': LEGACY_BANK_SHA,
            'recorded_input_size': 672, 'recorded_patch_grid': None,
            'verified_bank_input_size': 560, 'verified_bank_grid': [35, 35],
            'memory_q99_replay': 'exact_all50categories', 'source_manifest_unmodified': True}


def validate_calibration(cal, producer, old, partition, selected, environment, *, old_sha=None, bank_report=None):
    from scripts.v355_seen_category_calibration import validate_category_calibration
    from scripts.v365_ensemble_category_calibration import SCHEMA, checked_members

    seen = producer['seen_categories']
    required = {'status': 'ok', 'version': SCHEMA, 'reference_input_size': 448,
                'memory_input_size': 560, 'categories': 50, 'views_per_category': 20,
                'views_per_pass': 1000, 'passes': 2,
                'response_replay': 'exact_all_maps_all_categories', 'producer_sha256': PRODUCER_SHA}
    if (any(cal.get(key) != value for key, value in required.items())
            or cal.get('test_b_accessed') is not False or cal.get('new_training') is not False
            or len(seen) != 50 or len(set(seen)) != 50
            or set(cal.get('categories_detail', {})) != set(seen)
            or cal.get('partition') != partition or partition.get('categories') != sorted(seen)
            or len(selected) != 4 or cal.get('selected_checkpoints') != selected
            or producer.get('selected_checkpoints') != selected
            or cal.get('bank_sha256') != producer['bank_sha256']
            or producer.get('input_size') != 560
            or any(record.get('preprocess_environment') != environment for record in (cal, producer, old))):
        raise ValueError('Calibration schema, partition, producer, model, bank or environment mismatch')
    sources = cal.get('source_files_sha256')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Calibration lacks source identities')
    for path, value in sources.items():
        if not isinstance(path, str) or not path.startswith('/'):
            raise ValueError('Calibration source paths must be absolute')
        _digest(value)
    validate_category_calibration(old, seen)
    if (old.get('bank_sha256') != producer['bank_sha256']
            or old.get('bank_manifest_sha256') != producer['bank_manifest_sha256']):
        raise ValueError('Frozen memory calibration identity differs')
    calibration_metadata_provenance(old, producer, cal, old_sha, bank_report)
    normal_groups = partition['calibration_groups']
    if (len(normal_groups) != 200 or len(set(normal_groups)) != 200
            or set(normal_groups) & set(partition['bank_groups'])):
        raise ValueError('Normal calibration partition overlaps or is incomplete')
    scales = {}
    for category in sorted(seen):
        item = cal['categories_detail'][category]
        members = item.get('members', [])
        checked_members(members)
        expected = [f'Train/{g}/{v}.png' for g in normal_groups
                    if g.split('/')[0] == category for v in range(5)]
        if members != expected or item.get('views') != 20 or item.get('groups') != 4:
            raise ValueError(f'Calibration members are misrouted: {category}')
        rq = item['reference_q99']
        mq = item['memory_q99']
        old_mq = old['categories_detail'][category]['q99']
        old_rq = producer['reference_calibration_q99']
        measured_scale = normal_quantile_scale(rq, mq)
        drift = abs(mq - old_mq)
        if (item.get('old_reference_q99') != old_rq or item.get('scale') != measured_scale
                or not math.isfinite(drift) or drift > 1e-6
                or item.get('published_memory_q99_abs_drift') != drift):
            raise ValueError(f'Calibration q99/scale or published memory drift mismatch: {category}')
        normal_quantile_scale(item['median_fold_q99'], mq)
        folds = item.get('fold_response_sha256', [])
        if len(folds) != 4:
            raise ValueError('Expected all four fold response identities')
        for value in [*folds, item.get('ensemble_response_sha256'), item.get('memory_response_sha256')]:
            _digest(value)
        # Only the numerator changes, even if V365's memory replay drift is tiny.
        scales[category] = {'reference_q99': rq, 'memory_q99': old_mq,
                            'old_reference_q99': old_rq, 'scale': normal_quantile_scale(rq, old_mq),
                            'old_scale': normal_quantile_scale(old_rq, old_mq)}
    return scales


class ReplayError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('Frozen V362 replay failed: ' + json.dumps(report, allow_nan=False, sort_keys=True))


class ReplayAudit:
    def __init__(self, frozen_scores):
        self.frozen_scores = frozen_scores
        self.scores = defaultdict(dict)
        self.report = dict(old_scale_masks_checked=0, old_scale_mismatched_masks=0,
                           old_scale_mismatched_pixels=0, old_scale_max_abs_error=0,
                           reference_score_groups_checked=0, reference_score_mismatched_groups=0,
                           reference_score_max_abs_error=0., reference_score_tolerance=1e-6)

    def add(self, group, view, score, replayed, archived):
        if group not in self.frozen_scores or view not in '01234' or len(view) != 1 or view in self.scores[group]:
            raise ValueError('Unknown or duplicate replay view')
        if not math.isfinite(float(score)) or not 0 <= score <= 2.000001:
            raise ValueError('Invalid reference view score')
        if any(np.shape(x) != (448, 448) or x.dtype != np.uint8 for x in (replayed, archived)):
            raise ValueError('Replay requires encoded448x448 uint8 maps')
        errors = np.abs(replayed.astype(np.int16) - archived.astype(np.int16))
        count = int(np.count_nonzero(errors))
        self.report['old_scale_masks_checked'] += 1
        self.report['old_scale_mismatched_masks'] += int(count != 0)
        self.report['old_scale_mismatched_pixels'] += count
        self.report['old_scale_max_abs_error'] = max(self.report['old_scale_max_abs_error'], int(errors.max()))
        self.scores[group][view] = float(score)

    def finish(self):
        if set(self.scores) != set(self.frozen_scores) or any(set(v) != set('01234') for v in self.scores.values()):
            raise ValueError('Incomplete frozen V362 replay')
        report = dict(self.report)
        for group, values in self.scores.items():
            error = abs(score_views([values[v] for v in '01234']) - self.frozen_scores[group])
            report['reference_score_groups_checked'] += 1
            report['reference_score_mismatched_groups'] += int(error > 1e-6)
            report['reference_score_max_abs_error'] = max(report['reference_score_max_abs_error'], error)
        if report['old_scale_mismatched_masks'] or report['reference_score_mismatched_groups']:
            raise ReplayError(report)
        return report


def _png(pixels):
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    return buffer.getvalue()


def stream_candidate(base, target, index, responses, scales):
    expected = _expected(index)
    validate_members(base, expected)
    csv_bytes = base.read('submission.csv')
    _check_csv(csv_bytes, index)
    if set(scales) != set(index['seen']):
        raise ValueError('Expected exactly50 seen-category scales')
    frozen_scores = {row['group_folder']: float(row['anomaly_score']) for row in
                     csv.DictReader(io.StringIO(csv_bytes.decode('utf-8')))
                     if row['group_folder'].split('/')[0] in index['seen']}
    replay = ReplayAudit(frozen_scores)
    expected_seen = {(g, v) for g, views in index['groups'].items()
                     if g.split('/')[0] in index['seen'] for v in views}
    target.writestr('submission.csv', csv_bytes)
    for group, views in index['groups'].items():
        if group.split('/')[0] in index['seen']:
            continue
        for view in views:
            name = f'predicted_masks/{group}/{view}_mask.png'
            payload = base.read(name)
            validate_png(payload)
            target.writestr(name, payload)
    written, hashes = set(), {}
    for group, view, reference, memory in responses:
        key = (group, view)
        if key not in expected_seen or key in written:
            raise ValueError(f'Unknown, duplicate or unseen/misrouted candidate view: {key}')
        name = f'predicted_masks/{group}/{view}_mask.png'
        scale = scales[group.split('/')[0]]
        original = base.read(name)
        validate_png(original)
        old_pixels = encode_evidence(fuse_memory(reference, memory, scale=scale['old_scale']))
        replay.add(group, view, top_mean(np.asarray(reference, np.float32), .005), old_pixels, decode_png(original))
        payload = _png(encode_evidence(fuse_memory(reference, memory, scale=scale['scale'])))
        validate_png(payload)
        target.writestr(name, payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
        written.add(key)
        if len(written) % 500 == 0:
            print(f'candidate_unscored_seen_masks={len(written)}/6250', flush=True)
    if written != expected_seen:
        raise ValueError(f'Incomplete seen mask coverage: {len(written)}/{len(expected_seen)}')
    return {'seen_sha256': hashes, 'replay': replay.finish()}


def audit_archive(archive, base, index, seen_sha256):
    expected = _expected(index)
    for source in (archive, base):
        validate_members(source, expected)
    expected_seen = {f'predicted_masks/{g}/{v}_mask.png' for g, views in index['groups'].items()
                     if g.split('/')[0] in index['seen'] for v in views}
    if set(seen_sha256) != expected_seen:
        raise ValueError('Candidate seen digest coverage mismatch')
    csv_bytes = archive.read('submission.csv')
    if csv_bytes != base.read('submission.csv'):
        raise ValueError('Frozen CSV payload changed')
    stats = _check_csv(csv_bytes, index)
    counts = Counter()
    tree = hashlib.sha256()
    for name in sorted(expected - {'submission.csv'}):
        payload = archive.read(name)
        validate_png(payload)
        if name in expected_seen:
            if hashlib.sha256(payload).hexdigest() != _digest(seen_sha256[name]):
                raise ValueError(f'Seen mask differs from its assigned inference response: {name}')
            counts['seen'] += 1
        else:
            if payload != base.read(name):
                raise ValueError(f'Unseen mask bytes changed: {name}')
            counts['unseen'] += 1
        tree.update(name.encode() + b'\0' + hashlib.sha256(payload).digest())
    if counts != {'seen': 6250, 'unseen': 5000}:
        raise ValueError('Unexpected final mask population')
    return {'groups': 2250, 'masks': 11250, 'entries': 11251, 'seen_masks': 6250, 'unseen_masks': 5000,
            'csv_byte_exact': True, 'unseen_byte_exact': True, 'crc': 'ok', 'png_internal_crc': 'ok',
            'mode': 'L', 'dimensions': [448, 448], 'score_audit': stats,
            'assignment_tree_sha256': tree.hexdigest(), 'quality_status': 'candidate_unscored'}


class BoundedWriter:
    """Reject a ZIP write before its physical file can cross the disk budget."""
    def __init__(self, handle, limit):
        self.handle, self.limit = handle, limit

    def write(self, payload):
        if self.handle.tell() + len(payload) > self.limit:
            raise ValueError('Candidate ZIP disk budget exceeded')
        return self.handle.write(payload)

    def __getattr__(self, name):
        return getattr(self.handle, name)


def require_sources(actual, required):
    for path, expected in required.items():
        if actual.get(path) != _digest(expected):
            raise ValueError(f'Calibration omitted or changed a required source identity: {path}')
    return dict(actual)


def merge_sources(*tables):
    merged = {}
    for table in tables:
        for path, expected in table.items():
            if path in merged and merged[path] != expected:
                raise ValueError(f'Conflicting source identities: {path}')
            merged[path] = _digest(expected)
    return merged


def validate_producer(manifest):
    from scripts.v356_seen_memory_submission import validate_submission_contract

    validate_submission_contract(manifest)
    producer = manifest['provenance']
    if (manifest.get('sha256') != BASELINE_SHA or producer.get('input_size') != 560
            or producer.get('test_b_labels_accessed') is not False):
        raise ValueError('Not the frozen560 V362 blind producer')
    _digest(producer.get('test_zip_sha256'))
    return producer


def historical_calibration_sources(old):
    hashes = old['code_hashes']
    name = 'v355_seen_category_calibration.py'
    if name not in hashes:
        raise ValueError('Historical V355 source identity is missing')
    # The old report predates the metadata-only V355 fix. Hash its preserved
    # producer, without executing it or weakening checks on any other source.
    historical = ROOT / 'outputs/v365_historical_v355_before_metadata_fix.py'
    sources = {str(PROJECT / 'scripts' / n): d for n, d in hashes.items() if n != name}
    sources[str(historical)] = hashes[name]
    unchanged(sources)
    return sources


def load_clean_scales(path, expected_sha, calibration, calibration_sha, original):
    from scripts import v365_clean_calibration_overlay as clean

    sources = {str(path): _digest(expected_sha), str(clean.AUDIT_PATH): clean.AUDIT_SHA,
               str(Path(clean.__file__)): sha256_file(Path(clean.__file__))}
    unchanged(sources)
    # Keep strict JSON parsing too; the shared helper independently reconstructs
    # the complete overlay from the pinned audit and original two-pass donor.
    requested = read_verified_json(path, expected_sha)
    overlay = clean.load_verified_overlay(path, expected_sha, calibration, calibration_sha)
    if (overlay != requested or set(overlay['categories_detail']) != set(original)
            or overlay['effective_calibration_views'] != 995
            or overlay['excluded_groups'] != ['toy_tire/S0020']
            or overlay['unchanged_categories'] != 49
            or overlay['donor_crossfit_categories_in_summary'] != 49):
        raise ValueError('Clean overlay differs from the reviewed995-view exclusion contract')
    scales = {}
    for category, prior in original.items():
        item = overlay['categories_detail'][category]
        scales[category] = {**prior, **item, 'old_memory_q99': prior['memory_q99'],
                            'old_reference_q99': prior['old_reference_q99'], 'old_scale': prior['old_scale']}
    unchanged(sources)
    return scales, overlay, sources


def preflight(args):
    """Verify all donor identities before model loading or any B image access."""
    _digest(args.calibration_sha256)
    _digest(args.clean_overlay_sha256)
    producer = validate_producer(read_verified_json(PRODUCER, PRODUCER_SHA))
    calibration = read_verified_json(args.calibration, args.calibration_sha256)
    old = read_verified_json(OLD_CALIBRATION, producer['category_calibration_sha256'])
    fixed = {str(PRODUCER): PRODUCER_SHA, str(BASELINE): BASELINE_SHA,
             str(ROOT / 'Test_B.zip'): producer['test_zip_sha256'],
             str(args.calibration): args.calibration_sha256,
             str(OLD_CALIBRATION): producer['category_calibration_sha256'],
             str(BANK_ROOT / 'memory.npz'): producer['bank_sha256'],
             str(BANK_ROOT / 'manifest.json'): producer['bank_manifest_sha256']}
    unchanged(fixed)

    from scripts.build_b_submission import _test_index
    from scripts.v344_longrun_submission import load_verified_bundle, inference_hashes
    from scripts.v349_clean_memory_bank import source_inputs, preprocessing_environment_identity, PARTITION_SHA
    from scripts.v356_seen_memory_submission import GATE, _load_category_memory
    from scripts.v348_memory_smoke import TRAIN_SHA, WEIGHT_SHA

    train, weight, train_groups, partition = source_inputs()
    environment = preprocessing_environment_identity()
    bundle = load_verified_bundle(ROOT / 'outputs/v343_verified_longrun_20260911', GATE)
    selected = [record['selected'] for record in bundle['records']]
    bank, rows, bank_report = _load_category_memory(BANK_ROOT)
    scales = validate_calibration(calibration, producer, old, partition, selected, environment,
                                  old_sha=producer['category_calibration_sha256'], bank_report=bank_report)
    metadata = calibration_metadata_provenance(old, producer, calibration,
                                               producer['category_calibration_sha256'], bank_report)
    scales, overlay, overlay_sources = load_clean_scales(
        args.clean_overlay, args.clean_overlay_sha256, calibration, args.calibration_sha256, scales)
    if bundle['selection_record_hashes'] != producer['selection_record_hashes']:
        raise ValueError('Frozen selection records changed')
    code = {str(PROJECT / 'scripts' / name): value for name, value in inference_hashes().items()}
    for name in ('v365_ensemble_category_calibration.py', 'v355_seen_category_calibration.py',
                 'v348_clean_memory_core.py', 'v349_clean_memory_bank.py',
                 'v351_source_calibration.py', 'v356_seen_memory_submission.py'):
        path = PROJECT / 'scripts' / name
        code[str(path)] = sha256_file(path)
    required = merge_sources(
        {path: fixed[path] for path in (str(PRODUCER), str(OLD_CALIBRATION),
                                      str(BANK_ROOT / 'memory.npz'), str(BANK_ROOT / 'manifest.json'))},
        {str(GATE): bundle['source_gate_sha256']}, code,
        {r['checkpoint']: r['checkpoint_sha256'] for r in selected}, bundle['selection_record_hashes'])
    donor_sources = require_sources(calibration['source_files_sha256'], required)
    sources = merge_sources(fixed, donor_sources, overlay_sources,
                            {str(PROJECT / 'scripts' / n): d for n, d in producer['code_hashes'].items()},
                            historical_calibration_sources(old),
                            old['runtime_source_hashes'],
                            {str(train): TRAIN_SHA, str(weight): WEIGHT_SHA,
                             str(ROOT / 'outputs/v348_memory_smoke_reviewed_20260911/source_partition.json'): PARTITION_SHA,
                             str(Path(__file__)): sha256_file(Path(__file__)),
                             **{str(PROJECT / 'scripts' / name): sha256_file(PROJECT / 'scripts' / name)
                                for name in ('v363_route_alignment.py', 'v357_v356_release.py')}})
    unchanged(sources)
    if (bank_report['bank_sha256'] != producer['bank_sha256']
            or bank_report['metadata']['input_size'] != 560
            or set(rows) != set(producer['seen_categories'])
            or set(rows) != {g.split('/')[0] for g in train_groups}):
        raise ValueError('Bank/Train-seen identities differ from V362')
    sources = merge_sources(sources, bank_report['metadata']['runtime_source_hashes'],
                            *[r['long_config']['runtime_source_hashes'] for r in bundle['records']])
    index = make_index(_test_index(ROOT / 'Test_B.zip', 'Test_B'), producer['seen_categories'])
    unchanged(sources)
    return {'producer': producer, 'calibration': calibration, 'clean_overlay': overlay, 'scales': scales, 'index': index,
            'bundle': bundle, 'bank': bank, 'category_rows': rows, 'sources': sources,
            'environment': environment, 'historical_calibration_metadata': metadata}


def gpu_responses(context):
    import torch
    from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    from scripts.v180_dinomaly3_dinov3_csig import _extract_patch_tokens
    from scripts.v283_supervised_dinomaly_longtrain import differentiable_anomaly_map
    from scripts.v348_clean_memory_core import resize_raw_map
    from scripts.v351_source_calibration import ResidentCosineMemory
    from scripts.v365_ensemble_category_calibration import load_models

    models = load_models(context['bundle'])
    test_zip = ROOT / 'Test_B.zip'
    reference = adapter.CsigZipDataset(test_zip, 'Test_B', adapter._build_transform(448, 448), include_metadata=True)
    high = adapter.CsigZipDataset(test_zip, 'Test_B', adapter._build_transform(560, 560), include_metadata=True)
    validate_records(reference.records, high.records, context['index'])
    residents = {}

    def worker(fold, images):
        torch.cuda.set_device(fold)
        with torch.inference_mode():
            enc, dec = models[fold](images.to(f'cuda:{fold}', non_blocking=True))
            return differentiable_anomaly_map(enc, dec, 448).float().cpu().numpy()[:, 0]

    def images_at(dataset, indices):
        images = []
        for row in indices:
            image, category, sample, view = dataset[row]
            record = dataset.records[row]
            if (category, sample, view) != (record.category, record.sample, record.view):
                raise ValueError('Loaded B image metadata changed')
            images.append(image)
        return torch.stack(images)

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            def infer_batch(indices):
                images = images_at(reference, indices)
                high_images = images_at(high, indices)
                pending = [pool.submit(worker, fold, images) for fold in range(4)]
                references = combine_raw([future.result() for future in pending])
                memory = {}
                categories = {high.records[row].category for row in indices} & context['index']['seen']
                # Keep at most the categories in this full batch resident, not a raw-map cache.
                for category in list(residents):
                    if category not in categories:
                        del residents[category]
                for category in sorted(categories):
                    if category not in residents:
                        residents[category] = ResidentCosineMemory(
                            context['bank'][context['category_rows'][category]], device='cuda:0', chunk_size=8192)
                with torch.inference_mode():
                    tokens, grid = _extract_patch_tokens(models[0].encoder,
                                                        high_images.to('cuda:0', non_blocking=True), [3, 7, 11])
                    if grid != (35, 35):
                        raise ValueError('Expected560 feature grid')
                    for position, row in enumerate(indices):
                        category = high.records[row].category
                        if category in categories:
                            distances = residents[category].distances(tokens[position:position + 1])[0]
                            memory[position] = resize_raw_map(distances.reshape(grid))
                return references, memory

            yield from generate_responses(reference.records, context['index']['seen'], infer_batch)
    finally:
        for dataset in (reference, high):
            if dataset._archive is not None:
                dataset._archive.close()


def generate_responses(records, seen, infer_batch):
    for indices, positions in seen_batches(records, seen):
        references, memory_maps = infer_batch(indices)
        if np.shape(references) != (len(indices), 448, 448) or set(memory_maps) != set(positions):
            raise ValueError('Inference batch shape/routing mismatch')
        for position in positions:
            record = records[indices[position]]
            yield record.group, record.view, references[position], memory_maps[position]


def build_candidate(baseline, output, index, responses, scales, sources, *, provenance, final_check):
    """Write one bounded ZIP, audit it while staged, then publish without overwrite.

    Failure leaves only new .part files for diagnosis. Existing files are never
    removed/replaced; successful cleanup only unlinks this call's staging links.
    """
    baseline, output = Path(baseline), Path(output)
    staging = output.with_name(output.name + '.part')
    report_path = output.with_name(output.name + '.audit.json')
    report_stage = report_path.with_name(report_path.name + '.part')
    for path in (output, staging, report_path, report_stage):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if str(baseline) not in sources:
        raise ValueError('Frozen archive must be included in source identities')
    unchanged(sources)
    with baseline.open('rb') as base_file, zipfile.ZipFile(base_file) as base:
        with staging.open('xb') as handle:
            writer = BoundedWriter(handle, ZIP_BUDGET)
            with zipfile.ZipFile(writer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as target:
                streamed = stream_candidate(base, target, index, responses, scales)
            handle.flush()
            os.fsync(handle.fileno())
        before_audit = sha256_file(staging)
        with staging.open('rb') as handle, zipfile.ZipFile(handle) as archive:
            audit = audit_archive(archive, base, index, streamed['seen_sha256'])
    final_check()
    unchanged(sources)
    if sha256_file(staging) != before_audit:
        raise ValueError('Staged archive changed during the final audit')
    report = {
        **audit, 'status': 'candidate_unscored', 'version': VERSION, 'archive': str(output),
        'sha256': before_audit, 'bytes': staging.stat().st_size,
        'replay': streamed['replay'], 'source_files_sha256': sources, 'provenance': provenance,
        'scales': scales, 'new_training': False, 'test_b_labels_accessed': False,
        'raw_evidence_persisted': False, 'auto_upload': False,
        'recommended_replacement_for_online_best': False,
        'policy': 'Only50 seen memory scales change using the verified clean overlay; '
                  'toy_tire uses both q99 values from15 retained views, others use20. '
                  '90/10 fusion, CSV and5000 unseen PNG payloads are unchanged. '
                  'Old replay uses only the original V362 q99 values.',
        'limitations': ['Candidate is unscored; no guaranteed B quality improvement.',
                        'Score/map replay covers6250 seen views; unseen payloads are byte-exact.',
                        'The original toy_tire crossfit is not a clean independent audit.']}
    payload = json.dumps(report, indent=2, allow_nan=False).encode('utf-8')
    if len(payload) > REPORT_BUDGET:
        raise ValueError('Candidate audit report disk budget exceeded')
    with report_stage.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    # The final ZIP is the commit marker. Hard links fail if any old file exists.
    os.link(report_stage, report_path)
    os.link(staging, output)
    staging.unlink()
    report_stage.unlink()
    return report


def run(args):
    output = Path(args.output_root).resolve()
    output_parent = (ROOT / 'outputs').resolve()
    if (not output.is_relative_to(output_parent) or output == output_parent
            or output.exists() or output.is_symlink()):
        raise ValueError('A new output directory under the project outputs is required')
    if shutil.disk_usage(output.parent).free < ZIP_BUDGET + REPORT_BUDGET:
        raise RuntimeError('990MB free required; total newly written data stays below1GB')
    context = preflight(args)
    import torch
    from scripts.v349_clean_memory_bank import source_inputs, preprocessing_environment_identity

    if torch.cuda.device_count() < 4 or any(torch.cuda.mem_get_info(i)[0] < 8 * 2**30 for i in range(4)):
        raise RuntimeError('Four GPUs with8GiB free required')
    environment = context['environment']
    def final_check():
        source_inputs()
        if preprocessing_environment_identity() != environment:
            raise ValueError('Inference environment changed')
        for fold in range(4):
            if (torch.cuda.get_device_name(fold) != environment['cuda_device']
                    or list(torch.cuda.get_device_capability(fold)) != environment['cuda_capability']):
                raise ValueError('A fold GPU differs from the frozen producer environment')

    for fold in range(4):
        if (torch.cuda.get_device_name(fold) != environment['cuda_device']
                or list(torch.cuda.get_device_capability(fold)) != environment['cuda_capability']):
            raise ValueError('A fold GPU differs from the frozen producer environment')
    torch.set_num_threads(4)
    torch.manual_seed(36501)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    provenance = {'baseline_sha256': BASELINE_SHA, 'producer_sha256': PRODUCER_SHA,
                  'calibration_sha256': args.calibration_sha256,
                  'clean_overlay_sha256': args.clean_overlay_sha256,
                  'selected_checkpoints': context['producer']['selected_checkpoints'],
                  'bank_sha256': context['producer']['bank_sha256'],
                  'preprocess_environment': environment, 'source_calibration_views': 1000,
                  'source_calibration_passes': 2, 'source_response_replay': 'exact_all_maps_all_categories',
                  'effective_calibration_views': context['clean_overlay']['effective_calibration_views'],
                  'excluded_calibration_groups': context['clean_overlay']['excluded_groups'],
                  'historical_calibration_metadata': context['historical_calibration_metadata'],
                  'reference_input_size': 448, 'memory_input_size': 560, 'global_dataset_batch_size': 4,
                  'toy_tire_original_crossfit_is_clean_independent_audit': False}
    output.mkdir()
    responses = gpu_responses(context)
    try:
        report = build_candidate(BASELINE, output / 'candidate_unscored.zip', context['index'], responses,
                                 context['scales'], context['sources'], provenance=provenance, final_check=final_check)
    finally:
        responses.close()
    print(json.dumps({key: report[key] for key in ('status', 'archive', 'sha256', 'bytes', 'replay')
                      if key in report}, allow_nan=False), flush=True)
    return report


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, default=CALIBRATION)
    parser.add_argument('--calibration-sha256', required=True)
    parser.add_argument('--clean-overlay', type=Path, required=True)
    parser.add_argument('--clean-overlay-sha256', required=True)
    return parser


if __name__ == '__main__':
    try:
        run(argument_parser().parse_args())
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        print(json.dumps({'status': 'candidate_unscored', 'published': False, 'error': str(error),
                          'replay': getattr(error, 'report', None)}, allow_nan=False), flush=True)
        raise SystemExit(1)
