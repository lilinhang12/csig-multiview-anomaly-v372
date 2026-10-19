"""Development-only scaled memory evidence; not the bounded V330 cosine schema."""
from __future__ import annotations

import re
import io
import zipfile
from numbers import Integral
from pathlib import Path

import numpy as np

from scripts.v283_supervised_dinomaly_longtrain import metric_triplet
from scripts.v330_raw_response import histogram_metrics, score_views
from scripts.v348_clean_memory_core import encode_evidence


SCHEMA = 'v350_scaled_memory_evidence_v1'
VIEWS = frozenset('01234')
SHAPE = (448, 448)
RECORD_FIELDS = frozenset({'raw_evidence', 'reference_view_score', 'schema', 'category', 'sample', 'view', 'contract_id'})


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Expected nonempty {name} identifier')
    return value


def _view(value):
    if (not isinstance(value, (str, Integral)) or isinstance(value, bool)
            or str(value) not in VIEWS):
        raise ValueError('Expected view ID 0..4')
    return str(value)


def _identity(category, sample, view, contract_id):
    if not isinstance(contract_id, str) or re.fullmatch(r'[0-9a-fA-F]{64}', contract_id) is None:
        raise ValueError('Expected a 64hex contract_id')
    return dict(schema=SCHEMA, category=_identifier(category, 'category'),
                sample=_identifier(sample, 'sample'), view=_view(view), contract_id=contract_id)


def _evidence(values):
    raw = np.asarray(values)
    if (raw.shape != SHAPE or raw.dtype.kind not in 'fiu'
            or not np.isfinite(raw).all() or np.any(raw < 0)):
        raise ValueError('Expected finite nonnegative numeric 448x448 evidence')
    return raw


def _reference_score(value):
    score = np.asarray(value)
    if (score.shape != () or score.dtype.kind not in 'fiu' or not np.isfinite(score)
            or not 0 <= score <= 2):
        raise ValueError('Expected a finite scalar reference raw top005 score in [0, 2]')
    return float(score)


def save_scaled_record(path, raw_evidence, *, category, sample, view, contract_id, reference_view_score):
    """Create one record exclusively; the destination's parent must already exist."""
    identity = _identity(category, sample, view, contract_id)
    score = _reference_score(reference_view_score)
    with np.errstate(over='ignore', invalid='ignore'):
        raw = _evidence(raw_evidence).astype(np.float32)
    _evidence(raw)
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, raw_evidence=raw, reference_view_score=np.float64(score),
                            **{key: np.asarray(value) for key, value in identity.items()})


def load_scaled_record(path, *, category, sample, view, contract_id):
    identity = _identity(category, sample, view, contract_id)
    with Path(path).open('rb') as handle:
        payload = handle.read()
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile as error:
        raise ValueError('Scaled evidence is not a valid NPZ archive') from error
    if len(names) != len(set(names)) or set(names) != {f'{field}.npy' for field in RECORD_FIELDS}:
        raise ValueError('Scaled evidence NPZ fields are not exact and unique')
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        if set(data.files) != RECORD_FIELDS:
            raise ValueError('Missing scaled evidence fields')
        for key, expected in identity.items():
            value = data[key]
            if value.shape != () or value.dtype.kind != 'U' or value.item() != expected:
                raise ValueError(f'Scaled evidence identity/schema mismatch: {key}')
        raw = data['raw_evidence']
        if raw.dtype != np.dtype(np.float32):
            raise ValueError('Persisted scaled evidence must have float32 dtype')
        return {'raw_evidence': _evidence(raw),
                'reference_view_score': _reference_score(data['reference_view_score'])}


def _binary(values, shape, name):
    result = np.asarray(values)
    if (result.shape != shape or result.dtype.kind not in 'buif'
            or np.any((result != 0) & (result != 1))):
        raise ValueError(f'Expected binary 0/1 {name} with shape {shape}')
    return result


class ScaledEvidenceMetrics:
    """All-pixel histograms with frozen reference raw top005 image scores."""

    def __init__(self):
        self.hists = {}
        self.groups = {}
        self.views = 0

    def add(self, *, category, group, view, label, mask, raw, reference_view_score):
        category = _identifier(category, 'category')
        group = _identifier(group, 'group')
        view = _view(view)
        item = self.groups.get(group)
        if item is not None and (item['category'] != category or view in item['views']):
            raise ValueError('Category mismatch or duplicate development view')
        label = int(_binary(label, (), 'label'))
        mask = _binary(mask, SHAPE, 'mask')
        score = _reference_score(reference_view_score)
        png = encode_evidence(_evidence(raw))
        indices = mask.astype(np.int64).reshape(-1) * 256 + png.reshape(-1)
        hist = np.bincount(indices, minlength=512).reshape(2, 256)

        # Commit only validated views; group keys remain opaque full object IDs.
        if item is None:
            item = self.groups[group] = {'category': category, 'label': 0, 'views': {}}
        self.hists.setdefault(category, np.zeros((2, 256), dtype=np.int64))[:] += hist
        item['views'][view] = score
        item['label'] = max(item['label'], label)
        self.views += 1

    def finish(self):
        if not self.groups:
            raise ValueError('Cannot evaluate an empty development collection')
        if any(set(item['views']) != VIEWS for item in self.groups.values()):
            raise ValueError('Expected exactly five views in every development group')
        per_category = {}
        for category, hist in sorted(self.hists.items()):
            items = [item for item in self.groups.values() if item['category'] == category]
            labels = np.asarray([item['label'] for item in items])
            scores = [score_views(list(item['views'].values())) for item in items]
            per_category[category] = {
                'image': metric_triplet(labels, np.asarray(scores)),
                'pixel': histogram_metrics(hist), 'groups': len(items),
                'unique_group_scores': len(set(scores)),
                'pixels_at_255_fraction': float(hist[:, 255].sum() / hist.sum()),
            }
        return {
            'views': self.views, 'pixel_count': self.views * 448 * 448,
            'categories': len(per_category), 'groups': len(self.groups), 'prediction_scale': 1.0,
            'metrics': {
                'per_category': per_category,
                'macro': {level: {key: float(np.mean([item[level][key] for item in per_category.values()]))
                                  for key in ('auroc', 'ap', 'best_f1')}
                          for level in ('image', 'pixel')},
            },
            'classification_source': 'reference_raw_top005_top2', 'scope': 'development-only',
        }
