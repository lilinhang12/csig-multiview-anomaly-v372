"""Exclude exact bank/calibration duplicate objects without rewriting donor reports."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.v283_supervised_dinomaly_longtrain import sha256_file
from scripts.v348_clean_memory_core import normal_quantile_scale
from scripts.v348_memory_smoke import TRAIN_SHA
from scripts.v349_clean_memory_bank import PARTITION_SHA
from scripts.v365_ensemble_category_calibration import SCHEMA as DONOR_SCHEMA, checked_members

SCHEMA = 'v365_duplicate_excluded_calibration_overlay_v1'
AUDIT_PATH = (Path.home() / 'csig' / 'outputs/v365_normal_decoded_partition_audit_20260912.json')
AUDIT_SHA = '7e48a53480506253dc00a007814c707106cee75e4b6315d13f50410605c70d5f'


def derive_overlay(calibration, audit, calibration_sha256, audit_sha256):
    if any(not isinstance(s, str) or not re.fullmatch('[0-9a-f]{64}', s)
           for s in (calibration_sha256, audit_sha256)):
        raise ValueError('Expected source identities')
    if (calibration.get('status') != 'ok' or calibration.get('version') != DONOR_SCHEMA
            or calibration.get('categories') != 50 or calibration.get('views_per_pass') != 1000
            or calibration.get('passes') != 2
            or calibration.get('response_replay') != 'exact_all_maps_all_categories'):
        raise ValueError('Incomplete repeated calibration donor')
    details = audit.get('cross_partition_duplicate_details', [])
    if (audit.get('train_sha256') != TRAIN_SHA or audit.get('partition_sha256') != PARTITION_SHA
            or audit.get('groups') != 1000 or audit.get('views') != 5000
            or audit.get('cross_partition_duplicate_sets') != len(details)
            or audit.get('status') != ('overlap_found' if details else 'ok')):
        raise ValueError('Invalid duplicate audit/source partition')
    rows = calibration['categories_detail']
    if len(rows) != 50:
        raise ValueError('Incomplete category coverage')
    partition = calibration['partition']
    by_category = {c: checked_members(row['members']) for c, row in rows.items()}
    excluded = defaultdict(set)
    for duplicate in details:
        entries = duplicate['members']
        if {item['partition'] for item in entries} != {'bank', 'calibration'}:
            raise ValueError('Expected a cross-partition duplicate set')
        for item in entries:
            member = item['member']
            parts = member.split('/')
            if (len(parts) != 4 or parts[0] != 'Train' or parts[1] not in rows
                    or parts[3] not in [f'{v}.png' for v in range(5)]):
                raise ValueError('Unrecognized duplicate member')
            group = '/'.join(parts[1:3])
            if group not in partition[item['partition'] + '_groups']:
                raise ValueError('Duplicate membership conflicts with frozen partition')
            if item['partition'] == 'calibration':
                if member not in rows[parts[1]]['members']:
                    raise ValueError('Duplicate calibration member absent from donor')
                excluded[parts[1]].add(group)
    result = {}
    for category, row in sorted(rows.items()):
        omit = sorted(excluded[category])
        if len(omit) > 1:
            raise ValueError('Multiple excluded groups in one category require recomputation')
        if set(by_category[category]) - set(partition['calibration_groups']):
            raise ValueError('Donor calibration groups differ from partition')
        fit = sorted(set(by_category[category]) - set(omit))
        rq, mq = row['reference_q99'], row['memory_q99']
        if omit:
            candidates = [x for x in row['leave_one_group_out'] if x['held_group'] == omit[0]]
            if len(candidates) != 1 or candidates[0]['fit_groups'] != fit:
                raise ValueError('Leaveout fit contains excluded object or wrong groups')
            rq, mq = candidates[0]['reference_q99'], candidates[0]['memory_q99']
        result[category] = {'reference_q99': float(rq), 'memory_q99': float(mq),
                            'scale': normal_quantile_scale(rq, mq), 'calibration_views': len(fit) * 5,
                            'calibration_groups': fit, 'excluded_groups': omit,
                            'donor': 'leave_one_group_out' if omit else 'full_category'}
    unaffected = [rows[c]['crossfit_macro'] for c in rows if not excluded[c]]
    return {'status': 'ok', 'version': SCHEMA, 'categories': 50,
            'source_calibration_sha256': calibration_sha256, 'duplicate_audit_sha256': audit_sha256,
            'categories_detail': result, 'bank_sha256': calibration['bank_sha256'],
            'selected_checkpoints': calibration['selected_checkpoints'],
            'effective_calibration_views': sum(x['calibration_views'] for x in result.values()),
            'unchanged_categories': len(unaffected),
            'excluded_groups': sorted(g for groups in excluded.values() for g in groups),
            'donor_crossfit_categories_in_summary': len(unaffected),
            'donor_crossfit_macros_unaffected_categories': {
                k: float(np.mean([x[k] for x in unaffected])) for k in unaffected[0]} if unaffected else {},
            'quality_status': 'normal_only_duplicate_exclusion_not_defect_quality_validation',
            'limits': ['Affected-category old crossfit metrics are not clean after excluding its duplicate group.',
                       'Unaffected-category old-scale numerator was frozen using the original full calibration set.',
                       'No Test_B quality labels or new training are used.']}


def load_verified_overlay(path, expected_sha256, calibration, calibration_sha256,
                          audit_path=AUDIT_PATH, audit_sha256=AUDIT_SHA):
    if sha256_file(Path(path)) != expected_sha256 or sha256_file(Path(audit_path)) != audit_sha256:
        raise ValueError('Overlay or duplicate audit digest mismatch')
    audit = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    overlay = json.loads(Path(path).read_text(encoding='utf-8'))
    if overlay != derive_overlay(calibration, audit, calibration_sha256, audit_sha256):
        raise ValueError('Clean overlay does not reproduce the source-derived scales')
    return overlay


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if (sha256_file(args.calibration) != args.calibration_sha256
            or sha256_file(args.audit) != args.audit_sha256):
        raise ValueError('Source digest mismatch')
    cal = json.loads(args.calibration.read_text(encoding='utf-8'))
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    result = derive_overlay(cal, audit, args.calibration_sha256, args.audit_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    load_verified_overlay(args.output, sha256_file(args.output), cal, args.calibration_sha256,
                          args.audit, args.audit_sha256)
    print(json.dumps({'status': result['status'], 'effective_calibration_views': result['effective_calibration_views'],
                      'excluded_groups': result['excluded_groups'], 'sha256': sha256_file(args.output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--calibration-sha256', required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--audit-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
