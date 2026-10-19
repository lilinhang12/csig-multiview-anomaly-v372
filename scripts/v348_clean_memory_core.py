"""Group-isolated source calibration and floating-point patch-memory evidence."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from numbers import Integral

import numpy as np

from scripts.v330_raw_response import HALF_RESPONSE, checked_raw


def partition_train_groups(groups, *, seed=34801, calibration_groups_per_category=4):
    if (not groups or not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0
            or not isinstance(calibration_groups_per_category, Integral)
            or isinstance(calibration_groups_per_category, bool) or calibration_groups_per_category < 1):
        raise ValueError('Invalid normal-source partition configuration')
    categories = defaultdict(list)
    for group, views in sorted(groups.items()):
        if (not re.fullmatch(r'[^/\\\x00]+/S\d{4}', group)
                or group.split('/')[0] in ('.', '..') or tuple(views) != ('0', '1', '2', '3', '4')):
            raise ValueError('Expected complete, ordered five-view source objects')
        categories[group.split('/')[0]].append(group)
    rng = np.random.default_rng(int(seed))
    bank, calibration = [], []
    for category, objects in sorted(categories.items()):
        if len(objects) <= calibration_groups_per_category:
            raise ValueError(f'Insufficient bank/calibration objects for {category}')
        chosen = set(rng.choice(len(objects), size=calibration_groups_per_category, replace=False).tolist())
        bank.extend(obj for i, obj in enumerate(objects) if i not in chosen)
        calibration.extend(obj for i, obj in enumerate(objects) if i in chosen)
    encoded = json.dumps([[key, list(groups[key])] for key in sorted(groups)], separators=(',', ':')).encode()
    return dict(version='v348_normal_group_partition_v1', seed=int(seed),
                calibration_groups_per_category=int(calibration_groups_per_category),
                categories=sorted(categories), bank_groups=sorted(bank), calibration_groups=sorted(calibration),
                bank_views=5 * len(bank), calibration_views=5 * len(calibration),
                input_groups_sha256=hashlib.sha256(encoded).hexdigest())


def validate_partition(groups, partition):
    try:
        expected = partition_train_groups(groups, seed=partition['seed'],
                                          calibration_groups_per_category=partition['calibration_groups_per_category'])
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete group partition contract') from error
    if partition != expected:
        raise ValueError('Normal bank/calibration partition changed or overlaps')


def resize_raw_map(grid):
    import torch
    import torch.nn.functional as functional

    raw = checked_raw(grid).astype(np.float32)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or min(raw.shape) < 2:
        raise ValueError('Expected a square raw patch-distance grid')
    resized = functional.interpolate(torch.from_numpy(raw)[None, None], size=(448, 448),
                                     mode='bilinear', align_corners=True)
    return resized[0, 0].numpy().astype(np.float32, copy=False)


def normal_quantile_scale(reference_q99, memory_q99):
    reference, memory = float(reference_q99), float(memory_q99)
    if not all(math.isfinite(value) and 1e-8 < value <= 2.000001 for value in (reference, memory)):
        raise ValueError('Degenerate source-normal q99; no query-adaptive fallback is permitted')
    return reference / memory


def _positive_evidence(values):
    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (448, 448) or not np.isfinite(raw).all() or np.min(raw) < 0:
        raise ValueError('Expected finite nonnegative448x448 evidence')
    return raw


def fuse_memory(reference, memory, *, scale):
    reference = checked_raw(reference)
    memory = checked_raw(memory)
    if reference.shape != (448, 448) or memory.shape != reference.shape:
        raise ValueError('Both source maps must be448x448')
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError('Invalid source calibration scale')
    with np.errstate(over='ignore', invalid='ignore'):
        fused = (.9 * reference + .1 * scale * memory).astype(np.float32)
    _positive_evidence(fused)
    return fused


def encode_evidence(values):
    # Source-scaled evidence is no longer bounded by the cosine-distance maximum2.
    raw = _positive_evidence(values)
    return np.rint(255. * (raw / (raw + HALF_RESPONSE))).astype(np.uint8)
