"""Produce a new auxiliary manifest excluding every target-category exposure."""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from scripts.v283_supervised_dinomaly_longtrain import atomic_write_json, build_category_folds, sha256_file


def filter_clean_rows(rows: Sequence[dict[str, Any]], target_categories: set[str],
                       matched_record_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids = [str(r['record_id']) for r in rows]
    if len(ids) != len(set(ids)) or not matched_record_ids <= set(ids):
        raise ValueError('Duplicate source IDs or unknown audit match')
    for row in rows:
        groups[str(row['group'])].append(row)
    for group, members in groups.items():
        if (len(members) != 5 or {str(r['view']) for r in members} != {'C1', 'C2', 'C3', 'C4', 'C5'}
                or len({r['category'] for r in members}) != 1 or len({r['phase'] for r in members}) != 1):
            raise ValueError(f'Invalid external five-view group: {group}')
    overlap_groups = {r['group'] for r in rows if r['record_id'] in matched_record_ids}
    excluded_categories = sorted({r['category'] for r in rows} & target_categories)
    clean = [dict(r) for r in rows if r['category'] not in target_categories and r['group'] not in overlap_groups]
    if not clean:
        raise ValueError('No clean auxiliary records remain')
    return clean, {
        'policy': 'exclude_all_target_categories_and_any_exact_overlap_object_group',
        'original_records': len(rows), 'retained_records': len(clean), 'excluded_records': len(rows) - len(clean),
        'excluded_categories': excluded_categories, 'excluded_exact_overlap_groups': sorted(overlap_groups),
        'retained_categories': sorted({r['category'] for r in clean}), 'category_count': len({r['category'] for r in clean}),
        'retained_groups': len({r['group'] for r in clean}), 'retained_phase_counts': dict(Counter(r['phase'] for r in clean)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_root)
    if out.exists():
        raise FileExistsError('Clean manifests must use a new output version')
    index, target_zip = Path(args.proxy_index), Path(args.test_zip)
    audit = json.loads(Path(args.overlap_audit).read_text(encoding='utf-8'))
    source_sha, test_sha = sha256_file(index), sha256_file(target_zip)
    if audit['proxy_index_sha256'] != source_sha or audit['test_zip_sha256'] != test_sha or audit['test_labels_accessed'] is not False:
        raise ValueError('Image-overlap audit identity mismatch')
    with zipfile.ZipFile(target_zip) as archive:
        target_categories = set()
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if len(parts) == 4 and parts[0] == 'Test_B' and parts[-1] in {f'{v}.png' for v in range(5)}:
                target_categories.add(parts[1])
    if len(target_categories) != 100:
        raise ValueError('Unexpected target-category count')
    with index.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    matches = {str(r['record_id']) for m in audit['matches'] for r in m['external_records']}
    clean, report = filter_clean_rows(rows, target_categories, matches)
    approved = index.resolve().parent / 'dataset'
    for row in clean:
        for key in ('image_path', 'mask_path'):
            if row.get(key) and not Path(row[key]).resolve().is_relative_to(approved):
                raise ValueError('Clean data source provenance violation')
    folds = build_category_folds(clean, 4)
    out.mkdir(parents=True)
    clean_path = out / 'clean_proxy_index.jsonl'
    with clean_path.open('x', encoding='utf-8') as handle:
        for row in clean:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    atomic_write_json(out / 'category_folds.json', folds)
    report.update({'status': 'ok', 'source_index': str(index), 'source_index_sha256': source_sha,
                   'test_zip_sha256': test_sha, 'overlap_audit_sha256': sha256_file(args.overlap_audit),
                   'clean_index': str(clean_path), 'clean_index_sha256': sha256_file(clean_path),
                   'approved_image_root': str(approved), 'test_labels_accessed': False,
                   'do_not_reuse_supervised_weights': True,
                   'allowed_initialization': 'Train-only unsupervised V272, plus authorized foundation weights',
                   'limitations': ['Exact image and category exclusion does not rule out all semantic/near-duplicate relations.']})
    atomic_write_json(out / 'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'excluded_exact_overlap_groups'}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proxy-index', type=Path, required=True)
    parser.add_argument('--test-zip', type=Path, required=True)
    parser.add_argument('--overlap-audit', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    run(parser.parse_args())
