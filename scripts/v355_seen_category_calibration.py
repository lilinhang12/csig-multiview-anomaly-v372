"""Train-only category-normal calibration for the B seen-category branch."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts import v342_clean_longtrain as longtrain
from scripts.v180_dinomaly3_dinov3_csig import CsigZipDataset, _extract_patch_tokens
from scripts.v283_supervised_dinomaly_longtrain import sha256_file
from scripts.v348_memory_smoke import ROOT, TRAIN_SHA, WEIGHT_SHA
from scripts.v349_clean_memory_bank import PARTITION_SHA, preprocessing_environment_identity, source_inputs
from scripts.v351_source_calibration import ResidentCosineMemory

SCHEMA = 'v355_seen_category_calibration_v1'
PARTITION_ROOT = ROOT / 'outputs/v348_memory_smoke_reviewed_20260911/source_partition.json'
PATCHES_PER_IMAGE = 50


def calibration_resolution(input_size):
    if input_size not in (560, 672):
        raise ValueError('Unsupported calibration resolution')
    return {'input_size': int(input_size), 'patch_grid': [int(input_size) // 16] * 2}


def category_row_indices(members):
    result = defaultdict(list)
    for index, member in enumerate(np.asarray(members).tolist()):
        parts = str(member).split('/')
        if len(parts) != 4 or parts[0] != 'Train' or not parts[1] or parts[1] in ('.', '..'):
            raise ValueError('Invalid normal Train bank member')
        result[parts[1]].append(index)
    return {category: indices for category, indices in sorted(result.items())}


def category_feature_row_indices(members, patches_per_image=PATCHES_PER_IMAGE):
    if not isinstance(patches_per_image, (int, np.integer)) or patches_per_image <= 0:
        raise ValueError('Invalid patches_per_image')
    member_rows = category_row_indices(members)
    return {category: [row * int(patches_per_image) + offset
                       for row in rows for offset in range(int(patches_per_image))]
            for category, rows in member_rows.items()}


def category_quantile(values):
    raw = np.asarray(values, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (448, 448) or not raw.size or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('Expected finite nonnegative B-normal448x448 maps')
    q99 = float(np.quantile(raw, .99, method='linear'))
    if not np.isfinite(q99) or q99 <= 1e-8:
        raise ValueError('Degenerate category-normal q99')
    return q99


def validate_category_calibration(report, categories):
    if (report.get('status') != 'ok' or report.get('version') != SCHEMA
            or report.get('categories') != len(categories) or report.get('views_per_category') != 20
            or list(report.get('categories_detail', {})) != sorted(categories)):
        raise ValueError('Incomplete category calibration')
    for category in sorted(categories):
        item = report['categories_detail'].get(category, {})
        if item.get('views') != 20 or not np.isfinite(float(item.get('q99', 0))) or float(item['q99']) <= 1e-8:
            raise ValueError(f'Invalid category calibration: {category}')
    return True


def _load_bank(bank_root):
    with np.load(Path(bank_root) / 'memory.npz', allow_pickle=False) as data:
        memory = np.asarray(data['memory'], dtype=np.float16)
        members = data['members'].astype(str).tolist()
    categories = category_feature_row_indices(members)
    if len(categories) != 50 or any(len(indices) != 4000 for indices in categories.values()):
        raise ValueError('Expected4000 traced bank features per seen category')
    return memory, members, categories


def run(args):
    import torch

    resolution = calibration_resolution(args.input_size)
    bank_root = (ROOT / 'outputs/v352_clean_memory_bank_env_20260911' / f'i{args.input_size}').resolve()
    output = (Path(args.output_root) / f'seen_category_i{args.input_size}').resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output.exists():
        raise ValueError('A new category calibration output is required')
    train, _, groups, partition = source_inputs()
    if sha256_file(bank_root / 'memory.npz') != args.bank_sha256:
        raise ValueError('Category bank hash differs from the requested immutable input')
    memory, members, category_indices = _load_bank(bank_root)
    if set(partition['calibration_groups']) & set(partition['bank_groups']):
        raise ValueError('Bank/calibration group overlap')
    cal_members = [f'Train/{group}/{view}.png' for group in partition['calibration_groups'] for view in range(5)]
    if len(cal_members) != 1000:
        raise ValueError('Expected1000 calibration views')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 12 * 1024 ** 3:
        raise RuntimeError('An available GPU with12GiB free is required')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark = False;torch.backends.cudnn.deterministic = True
    deps = adapter._import_official_modules()
    full, _ = adapter._build_model(deps, backbone='dinov3_vit_base_16', dropout=.4, reduce_dim=256,
                                   context_aware_recenter=True, mask_neighbor_size=0,
                                   weights_dir=ROOT / 'Dinomaly2_dinov3/backbones/weights')
    encoder = full.encoder.to('cuda:0').eval();encoder.requires_grad_(False);del full
    runtime = longtrain.runtime_source_hashes(deps, encoder)
    transform = adapter._build_transform(args.input_size, args.input_size)
    dataset = CsigZipDataset(train, 'Train', transform, include_metadata=True)
    indices = [i for i, record in enumerate(dataset.records) if record.group in set(partition['calibration_groups'])]
    if [dataset.records[i].member for i in indices] != cal_members:
        raise ValueError('Calibration loader order differs from frozen partition')
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=8,
                                         shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    residents = {}
    for category, row_indices in category_indices.items():
        residents[category] = ResidentCosineMemory(memory[row_indices], device='cuda:0', chunk_size=8192)
    results = {}
    started = time.monotonic()
    for pass_number in (1, 2):
        values = defaultdict(list);seen = 0
        with torch.inference_mode():
            for images, categories, samples, views in loader:
                tokens, grid = _extract_patch_tokens(encoder, images.to('cuda:0', non_blocking=True), [3, 7, 11])
                if tuple(grid) != tuple(resolution['patch_grid']):
                    raise ValueError('Calibration grid changed')
                for index, (category, sample, view) in enumerate(zip(categories, samples, views)):
                    if f'Train/{category}/{sample}/{view}.png' != cal_members[seen + index]:
                        raise ValueError('Calibration metadata changed')
                    raw = residents[str(category)].distances(tokens[index:index + 1])[0].reshape(grid)
                    from scripts.v348_clean_memory_core import resize_raw_map
                    values[str(category)].append(resize_raw_map(raw))
                seen += len(categories)
        if seen != 1000:
            raise ValueError('Calibration pass ended early')
        if pass_number == 1:
            results = {category: category_quantile(np.stack(items)) for category, items in values.items()}
        elif results != {category: category_quantile(np.stack(items)) for category, items in values.items()}:
            raise ValueError('Category calibration replay differs')
        print(f'category_calibration_pass{pass_number}=1000/1000', flush=True)
    source_inputs()
    report = {'status': 'ok', 'version': SCHEMA, **resolution, 'categories': len(results),
              'views_per_category': 20, 'categories_detail': {
                  category: {'views': 20, 'q99': q99} for category, q99 in sorted(results.items())},
              'bank_sha256': args.bank_sha256, 'bank_manifest_sha256': sha256_file(bank_root / 'manifest.json'),
              'train_zip_sha256': TRAIN_SHA, 'weights_sha256': WEIGHT_SHA, 'partition_sha256': PARTITION_SHA,
              'partition_member_list_sha256': hashlib.sha256(json.dumps(cal_members, separators=(',', ':')).encode()).hexdigest(),
              'preprocess_environment': preprocessing_environment_identity(), 'runtime_source_hashes': runtime,
              'code_hashes': {name: sha256_file(Path(__file__).with_name(name)) for name in
                              ['v355_seen_category_calibration.py', 'v351_source_calibration.py',
                               'v349_clean_memory_bank.py', 'v348_clean_memory_core.py']},
              'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
              'elapsed_seconds': time.monotonic() - started,
              'scope': 'Official normal Train category calibration only; no Test_B labels or quality claim.'}
    validate_category_calibration(report, list(results))
    output.mkdir(parents=True)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': 'ok', 'categories': 50, 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--bank-sha256', required=True)
    parser.add_argument('--input-size', type=int, choices=(560, 672), default=672)
    run(parser.parse_args())
