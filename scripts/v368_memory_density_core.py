"""Nested deterministic normal-feature sampling and fixed development screening."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from scripts.v360_seen_reference_validation import _sample_indices
from scripts.v367_view_memory_core import promotion_screen

VARIANTS = ('sparse', 'dense')


def sampling_indices(member, token_count):
    if (not isinstance(member, str) or not re.fullmatch(r'aux/[^/\\:\x00]+/(?:OK/)?[^/\\:\x00]+/[0-4]\.png', member)
            or any(p in ('.', '..') for p in member.split('/'))
            or not isinstance(token_count, (int, np.integer)) or isinstance(token_count, bool)
            or token_count < 200):
        raise ValueError('Expected an auxiliary five-view member and at least200 token positions')
    original = _sample_indices(member, int(token_count))
    remaining = np.setdiff1d(np.arange(token_count), original, assume_unique=True)
    seed = int.from_bytes(hashlib.sha256(('36801:' + member).encode()).digest()[:8], 'little')
    additional = np.sort(np.random.default_rng(seed).choice(remaining, 150, replace=False))
    return {'sparse': original, 'additional': additional}


def nested_distances(sparse, additional):
    sparse, additional = np.asarray(sparse, np.float32), np.asarray(additional, np.float32)
    if (sparse.ndim != 2 or sparse.shape != additional.shape or not sparse.size
            or not np.isfinite(sparse).all() or not np.isfinite(additional).all()
            or min(sparse.min(), additional.min()) < 0 or max(sparse.max(), additional.max()) > 2.000001):
        raise ValueError('Expected finite aligned nearest-neighbour distances')
    return {'sparse': sparse, 'dense': np.minimum(sparse, additional)}


def summarize_reports(reports):
    if len(reports) != 4 or {r['fold'] for r in reports} != set(range(4)) or any(r['status'] != 'ok' for r in reports):
        raise ValueError('All four successful held-fold reports required')
    seen, cases = set(), []
    for report in reports:
        if not report['details'] or seen & set(report['details']):
            raise ValueError('Empty or overlapping held-class reports')
        seen.update(report['details'])
        for category, row in sorted(report['details'].items()):
            case = {'category': category, 'fold': report['fold']}
            for variant in VARIANTS:
                pixel = row['metrics'][variant]['metrics']['per_category'][category]['pixel']
                case[variant] = {**{k: pixel[k] for k in ('ap', 'auroc', 'best_f1')},
                    'normal': row['reserved_normal'][variant]['pixel_fraction_above_reference_q99']}
            cases.append(case)
    screen = promotion_screen([{'category': r['category'], 'fold': r['fold'],
                                'baseline': r['sparse'], 'candidate': r['dense']} for r in cases])
    return {'version': 'v368_nested_memory_density_summary_v1', 'status': 'ok', 'categories': len(seen),
            'macros': {name: {key: float(np.mean([r[name][key] for r in cases]))
                             for key in ('ap', 'auroc', 'best_f1', 'normal')} for name in VARIANTS},
            'screen': screen, 'category_results': cases, 'new_training': False,
            'scope': 'Seen-reference auxiliary comparison, not independent ensemble or zero-shot/B quality'}
