"""Camera-bank routing and fixed comparison gates for V367 development only."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from scripts.v353_highres_development import canonical_view

VARIANTS = ('all_view', 'same_view', 'mixed_view')


def view_feature_indices(bank_rows):
    if len(bank_rows) != 80:
        raise ValueError('Expected80 normal bank images')
    categories = {r['category'] for r in bank_rows}
    if len(categories) != 1:
        raise ValueError('Bank must contain one category')
    groups, indices = defaultdict(set), defaultdict(list)
    for i, row in enumerate(bank_rows):
        view = canonical_view(row['view'])
        if (row['phase'] != 'train' or row['label'] != 0 or row.get('mask_path')
                or not row['group'].startswith(row['category'] + '/') or view in groups[row['group']]):
            raise ValueError('Invalid normal bank phase/label/group/view')
        groups[row['group']].add(view)
        indices[view].extend(range(i * 50, (i + 1) * 50))
    if len(groups) != 16 or any(v != set('01234') for v in groups.values()):
        raise ValueError('Bank must contain16 complete five-view objects')
    if set(indices) != set('01234') or any(len(rows) != 800 for rows in indices.values()):
        raise ValueError('Camera bank must contain800 traced features')
    return dict(sorted(indices.items()))


def memory_variants(all_maps, same_maps):
    all_maps, same_maps = np.asarray(all_maps, np.float32), np.asarray(same_maps, np.float32)
    if (all_maps.ndim != 3 or all_maps.shape[1:] != (448, 448) or not len(all_maps)
            or same_maps.shape != all_maps.shape or not np.isfinite(all_maps).all()
            or not np.isfinite(same_maps).all() or min(all_maps.min(), same_maps.min()) < 0
            or max(all_maps.max(), same_maps.max()) > 2.000001):
        raise ValueError('Expected matching finite448 distance maps')
    if np.any(same_maps < all_maps - 1e-6):
        raise ValueError('Same-view subset distance is smaller than full bank distance')
    return {'all_view': all_maps, 'same_view': same_maps,
            'mixed_view': (.5 * all_maps + .5 * same_maps).astype(np.float32)}


def promotion_screen(rows):
    if (len(rows) != 11 or len({r['category'] for r in rows}) != 11
            or {r['fold'] for r in rows} != set(range(4))):
        raise ValueError('Expected11 distinct categories spanning four held folds')
    for row in rows:
        for name in ('baseline', 'candidate'):
            values = [row[name][metric] for metric in ('ap', 'auroc', 'best_f1', 'normal')]
            if not all(np.isfinite(x) and 0 <= x <= 1 for x in values):
                raise ValueError('Invalid category quality values')
    delta = {key: [r['candidate'][key] - r['baseline'][key] for r in rows]
             for key in ('ap', 'auroc', 'best_f1', 'normal')}
    means = {key: float(np.mean(values)) for key, values in delta.items()}
    folds = {fold: {key: float(np.mean([r['candidate'][key] - r['baseline'][key]
                                      for r in rows if r['fold'] == fold])) for key in delta}
             for fold in range(4)}
    positive_folds = sum(r['ap'] > 0 for r in folds.values())
    nonregressing_categories = sum(x >= 0 for x in delta['ap'])
    checks = {'pixel_ap_gain': means['ap'] >= .005,
              'fold_consistency': positive_folds >= 3,
              'category_nonregression': nonregressing_categories >= 8,
              'pixel_auroc_nonregression': means['auroc'] >= -.002,
              'pixel_f1_nonregression': means['best_f1'] >= -.002,
              'normal_area_nonregression': means['normal'] <= .001,
              'normal_fold_nonregression': max(r['normal'] for r in folds.values()) <= .0025}
    return {'eligible': all(checks.values()), 'checks': checks, 'macro_deltas': means,
            'fold_deltas': folds, 'positive_ap_folds': positive_folds,
            'nonregressing_ap_categories': nonregressing_categories,
            'scope': 'Further-validation screen on existing auxiliary data, not B score or independent confirmation'}


def summarize_reports(reports):
    if len(reports) != 4 or {r['fold'] for r in reports} != set(range(4)) or any(r['status'] != 'ok' for r in reports):
        raise ValueError('All four successful fold reports required')
    categories, cases = set(), []
    values = {name: [] for name in VARIANTS}
    for report in reports:
        if not report['details'] or categories & set(report['details']):
            raise ValueError('Empty or overlapping held-fold categories')
        categories.update(report['details'])
        for category, row in sorted(report['details'].items()):
            case = {'category': category, 'fold': report['fold']}
            for variant in VARIANTS:
                pixel = row['metrics'][variant]['metrics']['per_category'][category]['pixel']
                item = {**{k: pixel[k] for k in ('ap', 'auroc', 'best_f1')},
                        'normal': row['reserved_normal'][variant]['pixel_fraction_above_reference_q99']}
                case[variant] = item
                values[variant].append(item)
            cases.append(case)
    screens = {variant: promotion_screen([{'category': c['category'], 'fold': c['fold'],
               'baseline': c['all_view'], 'candidate': c[variant]} for c in cases])
               for variant in VARIANTS if variant != 'all_view'}
    return {'status': 'ok', 'version': 'v367_view_memory_summary_v1', 'categories': len(categories),
            'macros': {name: {k: float(np.mean([row[k] for row in rows]))
                             for k in ('ap', 'auroc', 'best_f1', 'normal')} for name, rows in values.items()},
            'screens': screens, 'category_results': cases, 'new_training': False,
            'scope': 'Seen-reference auxiliary simulation, not independent ensemble/zero-shot/B score'}
