"""Byte-preserving seen/unseen ablation; does not claim improved model quality."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile
import zlib

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.v283_supervised_dinomaly_longtrain import sha256_file
from scripts.v330_raw_response import audit_scores
from scripts.v357_v356_release import decode_png

VERSION = 'v363_seen_unseen_ablation_v1'


def validate_png(payload):
    if not isinstance(payload, bytes) or not payload.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Invalid PNG signature')
    position, chunks = 8, []
    while position < len(payload):
        if position + 12 > len(payload):
            raise ValueError('Truncated PNG chunk')
        length = int.from_bytes(payload[position:position + 4], 'big')
        kind = payload[position + 4:position + 8]
        end = position + 12 + length
        if end > len(payload):
            raise ValueError('PNG chunk length exceeds payload')
        stored_crc = int.from_bytes(payload[end - 4:end], 'big')
        if zlib.crc32(payload[position + 4:end - 4]) != stored_crc:
            raise ValueError('PNG internal chunk checksum failed')
        if kind == b'IHDR' and (chunks or length != 13):
            raise ValueError('Invalid or repeated PNG header')
        chunks.append(kind)
        position = end
        if kind == b'IEND':
            if length != 0 or position != len(payload):
                raise ValueError('Invalid PNG ending or trailing bytes')
            break
    if not chunks or chunks[0] != b'IHDR' or chunks[-1] != b'IEND' or b'IDAT' not in chunks:
        raise ValueError('PNG lacks required header, image data or ending')
    decode_png(payload)


def validate_contract(contract):
    if (contract.get('version') != 'v363_route_source_contract_v1'
            or contract.get('train_groups') != 1000):
        raise ValueError('Incorrect Train-derived route contract')
    for key in ('train_zip_sha256', 'test_zip_sha256', 'baseline_sha256',
                'seen_source_sha256', 'producer_manifest_sha256'):
        if not isinstance(contract.get(key), str) or not re.fullmatch('[0-9a-f]{64}', contract[key]):
            raise ValueError('Missing or invalid source digest')
    groups, seen = contract.get('groups'), contract.get('seen_categories')
    if not isinstance(groups, dict) or len(groups) != 2250:
        raise ValueError('Exactly2250 B objects are required')
    if (not isinstance(seen, list) or len(seen) != 50 or any(not isinstance(c, str) for c in seen)
            or len(set(seen)) != 50):
        raise ValueError('Exactly50 distinct Train-seen categories are required')
    for group, views in groups.items():
        if (not isinstance(group, str) or not re.fullmatch(r'[^/\\:\x00]+/S\d{4}', group)
                or group.split('/')[0] in ('.', '..')
                or not isinstance(views, (list, tuple)) or tuple(views) != tuple('01234')):
            raise ValueError('Unsafe path or incomplete ordered five-view object')
    categories = {g.split('/')[0] for g in groups}
    if len(categories) != 100 or not set(seen) <= categories:
        raise ValueError('Seen and unseen populations do not match the B index')
    counts = Counter('seen' if g.split('/')[0] in seen else 'unseen' for g in groups)
    if counts != {'seen': 1250, 'unseen': 1000}:
        raise ValueError('Unexpected complete B seen/unseen group counts')
    return {'groups': {g: tuple(groups[g]) for g in sorted(groups)}, 'seen': frozenset(seen)}


def source_for(group, index):
    if group not in index['groups']:
        raise ValueError('Unknown B object')
    return 'seen_source' if group.split('/')[0] in index['seen'] else 'baseline'


def _expected(index):
    return {'submission.csv', *{f'predicted_masks/{g}/{v}_mask.png'
                               for g, views in index['groups'].items() for v in views}}


def validate_members(archive, expected):
    names = archive.namelist()
    if len(names) != len(expected) or len(set(names)) != len(names) or set(names) != expected:
        raise ValueError('ZIP contains missing, duplicate or unexpected entries')
    if archive.testzip() is not None:
        raise ValueError('ZIP failed CRC validation')


def _check_csv(payload, index):
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8')))
    if reader.fieldnames != ['group_folder', 'anomaly_score']:
        raise ValueError('Incorrect CSV columns')
    rows = list(reader)
    values = {row['group_folder']: float(row['anomaly_score']) for row in rows}
    if len(rows) != 2250 or len(values) != 2250 or set(values) != set(index['groups']):
        raise ValueError('CSV does not cover every B object exactly once')
    stats = audit_scores(values)
    if not stats['passed']:
        raise ValueError('CSV contains invalid or degenerate scores')
    return stats


def _sources(baseline, seen_source, contract_path, contract_sha256):
    paths = [Path(p).resolve() for p in (baseline, seen_source, contract_path)]
    if sha256_file(paths[2]) != contract_sha256:
        raise ValueError('Source contract digest mismatch')
    contract = json.loads(paths[2].read_text(encoding='utf-8'))
    index = validate_contract(contract)
    fingerprints = {str(paths[2]): contract_sha256, str(paths[0]): contract['baseline_sha256'],
                    str(paths[1]): contract['seen_source_sha256']}
    if len(fingerprints) != 3:
        raise ValueError('Baseline, seen source and contract must be distinct files')
    _unchanged(fingerprints)
    return paths, index, fingerprints


def _unchanged(fingerprints):
    if any(sha256_file(Path(path)) != value for path, value in fingerprints.items()):
        raise ValueError('A source file changed or has an unexpected digest')


def build(baseline, seen_source, contract_path, contract_sha256, output):
    output = Path(output).resolve()
    staging = output.with_suffix(output.suffix + '.part')
    if output.exists() or staging.exists():
        raise FileExistsError('Output or staging already exists; source artifacts are preserved')
    paths, index, fingerprints = _sources(baseline, seen_source, contract_path, contract_sha256)
    counts = Counter()
    expected = _expected(index)
    with zipfile.ZipFile(paths[0]) as base, zipfile.ZipFile(paths[1]) as candidate:
        validate_members(base, expected); validate_members(candidate, expected)
        csv_bytes = base.read('submission.csv')
        if candidate.read('submission.csv') != csv_bytes:
            raise ValueError('V362 CSV differs from the online baseline')
        stats = _check_csv(csv_bytes, index)
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(staging, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as result:
            result.writestr('submission.csv', csv_bytes)
            for number, (group, views) in enumerate(index['groups'].items(), 1):
                role = source_for(group, index)
                origin = candidate if role == 'seen_source' else base
                counts[role + '_groups'] += 1
                for view in views:
                    name = f'predicted_masks/{group}/{view}_mask.png'
                    payload = origin.read(name)
                    validate_png(payload)
                    result.writestr(name, payload)
                    counts[role + '_masks'] += 1
                if number % 250 == 0 or number == 2250:
                    print(f'route_objects_written={number}/2250', flush=True)
    with zipfile.ZipFile(staging) as result:
        validate_members(result, expected)
    _unchanged(fingerprints)
    os.link(staging, output)
    staging.unlink()
    return {'version': VERSION, 'archive': str(output), 'sha256': sha256_file(output),
            'bytes': output.stat().st_size, 'counts': dict(counts), 'score_audit': stats,
            'source_files_sha256': fingerprints, 'quality_status': 'controlled_ablation_unscored',
            'recommended_replacement_for_online_best': False}


def verify(output, baseline, seen_source, contract_path, contract_sha256):
    paths, index, fingerprints = _sources(baseline, seen_source, contract_path, contract_sha256)
    output = Path(output).resolve(); digest = sha256_file(output)
    counts = Counter(); changed_base = changed_seen = 0
    assignment_tree = hashlib.sha256()
    expected = _expected(index)
    with zipfile.ZipFile(output) as archive, zipfile.ZipFile(paths[0]) as base, zipfile.ZipFile(paths[1]) as seen:
        for z in (archive, base, seen):
            validate_members(z, expected)
        csv_bytes = archive.read('submission.csv')
        if csv_bytes != base.read('submission.csv') or csv_bytes != seen.read('submission.csv'):
            raise ValueError('Classification CSV changed during the route correction')
        stats = _check_csv(csv_bytes, index)
        for i, name in enumerate(sorted(expected - {'submission.csv'}), 1):
            _, category, sample, _ = name.split('/')
            group = category + '/' + sample
            role = 'seen_source' if category in index['seen'] else 'baseline'
            candidate, original = seen.read(name), base.read(name)
            delivered = archive.read(name)
            if delivered != (candidate if role == 'seen_source' else original):
                raise ValueError('Mask is not byte-identical to its assigned source')
            validate_png(delivered)
            counts[role] += 1
            changed_base += delivered != original
            changed_seen += delivered != candidate
            assignment_tree.update(name.encode() + b'\0' + role.encode() + b'\0' + hashlib.sha256(delivered).digest())
            if i % 2500 == 0 or i == 11250:
                print(f'route_masks_verified={i}/11250', flush=True)
    if counts != {'baseline': 5000, 'seen_source': 6250}:
        raise ValueError('Incorrect seen/unseen mask coverage')
    _unchanged(fingerprints)
    if sha256_file(output) != digest:
        raise ValueError('Output archive changed during verification')
    return {'passed': True, 'version': VERSION, 'archive': str(output), 'sha256': digest,
            'bytes': output.stat().st_size, 'groups': 2250, 'masks': 11250, 'entries': 11251,
            'csv_identical_to_baseline': True, 'all_masks_equal_to_assigned_source': True,
            'seen_masks': counts['seen_source'], 'unseen_masks': counts['baseline'],
            'changed_masks_from_v362': changed_seen, 'changed_masks_from_v344': changed_base,
            'crc': 'ok', 'mode': 'L', 'dimensions': [448, 448],
            'assignment_tree_sha256': assignment_tree.hexdigest(), 'score_audit': stats,
            'source_files_sha256': fingerprints, 'auditor_sha256': sha256_file(Path(__file__)),
            'quality_status': 'controlled_ablation_unscored',
            'recommended_replacement_for_online_best': False,
            'scope': 'Byte-exact partition routing and file integrity, not model-quality or leaderboard certification'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--seen-source', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    build(args.baseline, args.seen_source, args.contract, args.contract_sha256, args.output)
    report = verify(args.output, args.baseline, args.seen_source, args.contract, args.contract_sha256)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False), flush=True)
