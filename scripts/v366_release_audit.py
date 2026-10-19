"""Independent parsing and byte-preservation audit for the V366 candidate ZIP."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.v283_supervised_dinomaly_longtrain import sha256_file

BASELINE_SHA = 'a99079d7cced250a8f4a0043135d676f91cf600cb1cb287a85e35e1a0213363e'
CONTRACT_SHA = 'f7fcd1e335fccd99db468e13e3ea5548be22f24277696895b4f3d1cb8e625f5b'


def validated_index(contract):
    groups, seen = contract['groups'], contract['seen_categories']
    if len(groups) != 2250 or len(seen) != 50 or len(set(seen)) != 50:
        raise ValueError('Incorrect object/category coverage')
    for group, views in groups.items():
        if (not re.fullmatch(r'[^/\\:\x00]+/S\d{4}', group) or group.split('/')[0] in ('.', '..')
                or list(views) != list('01234')):
            raise ValueError('Invalid object path or incomplete five-view index')
    categories = {g.split('/')[0] for g in groups}
    counts = Counter('seen' if g.split('/')[0] in seen else 'unseen' for g in groups)
    if len(categories) != 100 or not set(seen) <= categories or counts != {'seen': 1250, 'unseen': 1000}:
        raise ValueError('Incorrect seen/unseen partition')
    return dict(sorted(groups.items())), set(seen)


def sample_groups(contract, seed_sha):
    groups, seen = validated_index(contract)
    if not re.fullmatch('[0-9a-f]{64}', seed_sha):
        raise ValueError('Sampling seed must be a frozen source digest')
    return [min((g for g in groups if g.split('/')[0] == category),
                key=lambda g: hashlib.sha256((seed_sha + ':' + g).encode()).digest())
            for category in sorted(seen)]


def independent_png_pixels(reference, memory, scale):
    a, b = np.asarray(reference, np.float64), np.asarray(memory, np.float64)
    if (a.shape != (448, 448) or b.shape != a.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or min(a.min(), b.min()) < -1e-6
            or max(a.max(), b.max()) > 2.000001 or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Invalid reference/memory values')
    # Preserve the producer's explicit float32 fusion boundary, independently of its encoder.
    with np.errstate(over='ignore', invalid='ignore'):
        fused = (.9 * np.maximum(a, 0) + .1 * scale * np.maximum(b, 0)).astype(np.float32).astype(np.float64)
    if not np.isfinite(fused).all():
        raise ValueError('Nonfinite fusion result')
    return np.rint(255.0 * (fused / (fused + .1))).astype(np.uint8)


def inspect_archives(candidate, baseline, contract):
    groups, seen = validated_index(contract)
    expected = {'submission.csv', *{f'predicted_masks/{g}/{v}_mask.png' for g in groups for v in '01234'}}
    for archive in (candidate, baseline):
        names = archive.namelist()
        if len(names) != 11251 or len(set(names)) != len(names) or set(names) != expected:
            raise ValueError('ZIP member coverage, duplicate or path mismatch')
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC mismatch')
    payload = candidate.read('submission.csv')
    if payload != baseline.read('submission.csv'):
        raise ValueError('Classification CSV bytes changed')
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8')))
    if reader.fieldnames != ['group_folder', 'anomaly_score']:
        raise ValueError('CSV columns changed')
    rows = list(reader)
    if (len(rows) != 2250 or any(set(r) != {'group_folder', 'anomaly_score'} for r in rows)
            or len({r['group_folder'] for r in rows}) != 2250
            or {r['group_folder'] for r in rows} != set(groups)):
        raise ValueError('CSV object coverage mismatch')
    scores = [float(r['anomaly_score']) for r in rows]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
        raise ValueError('Invalid classification score')
    counts = Counter()
    tree = hashlib.sha256()
    for number, name in enumerate(sorted(expected - {'submission.csv'}), 1):
        png = candidate.read(name)
        category = name.split('/')[1]
        original = baseline.read(name)
        if category not in seen and png != original:
            raise ValueError(f'Unseen PNG bytes changed: {name}')
        with Image.open(io.BytesIO(png)) as image:
            if image.format != 'PNG' or image.mode != 'L' or image.size != (448, 448):
                raise ValueError(f'Incorrect PNG format/mode/size: {name}')
            image.verify()
        with Image.open(io.BytesIO(png)) as image:
            image.load()
            low, high = image.getextrema()
            counts['all_black_masks'] += int(high == 0)
            counts['constant_masks'] += int(low == high)
        counts['png_verified'] += 1
        counts['seen_changed_masks'] += int(category in seen and png != original)
        counts['unseen_byte_identical_masks'] += int(category not in seen)
        tree.update(name.encode() + b'\0' + hashlib.sha256(png).digest())
        if number % 2500 == 0:
            print(f'independent_png_verified={number}/11250', flush=True)
    return {'groups': 2250, 'entries': 11251, **dict(counts), 'csv_byte_identical': True,
            'csv_score_range': [min(scores), max(scores)], 'mode': 'L', 'dimensions': [448, 448],
            'assignment_tree_sha256': tree.hexdigest(),
            'scope': 'Export correctness and frozen-byte comparison, not B accuracy'}


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    fingerprints = {args.candidate: args.candidate_sha256, args.producer_audit: args.producer_audit_sha256,
                    args.baseline: BASELINE_SHA, args.contract: CONTRACT_SHA}
    for path, digest in fingerprints.items():
        if sha256_file(path) != digest:
            raise ValueError(f'Source identity mismatch: {path}')
    contract = json.loads(args.contract.read_text(encoding='utf-8'))
    producer = json.loads(args.producer_audit.read_text(encoding='utf-8'))
    if (contract['seen_source_sha256'] != BASELINE_SHA or producer.get('status') != 'candidate_unscored'
            or producer.get('sha256') != args.candidate_sha256
            or producer.get('bytes') != args.candidate.stat().st_size
            or producer.get('new_training') is not False or producer.get('auto_upload') is not False
            or producer.get('test_b_labels_accessed') is not False):
        raise ValueError('Candidate provenance mismatch')
    replay = producer['replay']
    if (replay.get('old_scale_masks_checked') != 6250 or replay.get('reference_score_groups_checked') != 1250
            or any(replay.get(k) != 0 for k in ('old_scale_mismatched_masks', 'old_scale_mismatched_pixels',
                                               'old_scale_max_abs_error', 'reference_score_mismatched_groups'))
            or not 0 <= replay.get('reference_score_max_abs_error', float('nan')) <= 1e-6):
        raise ValueError('Incomplete or failed producer numerical replay')
    with zipfile.ZipFile(args.candidate) as c, zipfile.ZipFile(args.baseline) as b:
        report = inspect_archives(c, b, contract)
    if report['assignment_tree_sha256'] != producer['assignment_tree_sha256']:
        raise ValueError('Independent assignment tree differs from producer audit')
    for path, digest in fingerprints.items():
        if sha256_file(path) != digest:
            raise ValueError('An input changed during independent audit')
    report.update(status='passed', candidate_sha256=args.candidate_sha256,
                  source_files_sha256={str(p): h for p, h in fingerprints.items()},
                  numerical_sample_groups=sample_groups(contract, CONTRACT_SHA),
                  numerical_sample_status='planned_not_yet_executed',
                  code_sha256=sha256_file(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_files_sha256', 'numerical_sample_groups')}, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--producer-audit', type=Path, required=True)
    parser.add_argument('--producer-audit-sha256', required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
