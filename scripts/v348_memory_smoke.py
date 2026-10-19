"""Offline P100 smoke using disjoint normal Train objects, never Test_B."""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts.build_b_submission import _test_index
from scripts.v180_dinomaly3_dinov3_csig import _distance_maps, _extract_patch_tokens
from scripts.v283_supervised_dinomaly_longtrain import sha256_file
from scripts.v348_clean_memory_core import partition_train_groups, resize_raw_map, validate_partition

ROOT = (Path.home() / 'csig')
TRAIN_SHA = 'a62414e7aa11b32c73ef97f8513a5425ede5510e72e4f13893b31da8f8d62c23'
WEIGHT_SHA = 'c73f5f4ee2f3b4f7b51694615d6684336fc5ca6e783f57f834ede642c8c5f93d'


def run(output_root):
    import torch

    output = Path(output_root).resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output.exists():
        raise ValueError('A new output directory under the project outputs is required')
    train = ROOT / 'Train.zip'
    weights_dir = ROOT / 'Dinomaly2_dinov3/backbones/weights'
    weights = weights_dir / 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    if sha256_file(train) != TRAIN_SHA or sha256_file(weights) != WEIGHT_SHA:
        raise ValueError('Normal Train or foundation-weight identity changed')
    groups = _test_index(train, 'Train')
    partition = partition_train_groups(groups)
    validate_partition(groups, partition)
    if len(groups) != 1000 or len(partition['categories']) != 50:
        raise ValueError('Unexpected official normal Train cardinality')
    category = partition['categories'][0]
    bank_group = next(g for g in partition['bank_groups'] if g.split('/')[0] == category)
    query_group = next(g for g in partition['calibration_groups'] if g.split('/')[0] == category)
    if bank_group == query_group:
        raise ValueError('Smoke query entered the normal memory bank')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 12 * 1024 ** 3:
        raise RuntimeError('An available P100 with at least12GiB free is required')
    torch.set_num_threads(4)
    torch.manual_seed(34801)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    deps = adapter._import_official_modules()
    model, _ = adapter._build_model(deps, backbone='dinov3_vit_base_16', dropout=.4, reduce_dim=256,
                                    context_aware_recenter=True, mask_neighbor_size=0, weights_dir=weights_dir)
    encoder = model.encoder.to('cuda:0').eval()
    encoder.requires_grad_(False)
    del model
    output.mkdir(parents=True)
    source_record = {'train_zip_sha256': TRAIN_SHA, 'weights_sha256': WEIGHT_SHA, 'partition': partition}
    with (output / 'source_partition.json').open('x', encoding='utf-8') as handle:
        json.dump(source_record, handle, indent=2)
    rows = []
    with zipfile.ZipFile(train) as archive:
        for size in (560, 672):
            torch.cuda.reset_peak_memory_stats(0)
            started = time.monotonic()
            transform = adapter._build_transform(size, size)
            rng = np.random.default_rng(34801)
            memory_parts, members = [], []

            def image_tensor(group, view):
                with archive.open(f'Train/{group}/{view}.png') as handle, Image.open(handle) as image:
                    return transform(image.convert('RGB')).unsqueeze(0).to('cuda:0')

            with torch.inference_mode():
                for view in groups[bank_group]:
                    tokens, grid = _extract_patch_tokens(encoder, image_tensor(bank_group, view), [3, 7, 11])
                    if grid != (size // 16, size // 16) or tokens.shape[-1] != 2304:
                        raise ValueError('Unexpected physical DINOv3 grid or feature dimension')
                    indices = np.sort(rng.choice(tokens.shape[1], size=64, replace=False))
                    memory_parts.append(tokens[0, indices].float().cpu().numpy())
                    members.append({'group': bank_group, 'view': view, 'patch_indices': indices.tolist()})
                memory = np.concatenate(memory_parts, axis=0).astype(np.float32)
                distance_ranges = []
                target = output / f'i{size}'
                target.mkdir()
                for view in groups[query_group]:
                    tokens, grid = _extract_patch_tokens(encoder, image_tensor(query_group, view), [3, 7, 11])
                    if grid != (size // 16, size // 16) or tokens.shape[-1] != 2304:
                        raise ValueError('Query feature geometry changed')
                    distances = _distance_maps(tokens, memory, chunk_size=128)[0].reshape(grid)
                    raw = resize_raw_map(distances)
                    np.savez_compressed(target / f'{view}.npz', raw_grid=distances, raw_map=raw,
                                        group=np.asarray(query_group), view=np.asarray(view))
                    distance_ranges.append([float(raw.min()), float(raw.max())])
            torch.cuda.synchronize()
            row = dict(input_size=size, patch_grid=list(grid), feature_dim=2304, memory_patches=len(memory),
                       bank_groups=[bank_group], query_groups=[query_group], saved_views=5,
                       members=members, response_ranges=distance_ranges, group_disjoint=True,
                       seconds=time.monotonic() - started,
                       peak_cuda_mib=torch.cuda.max_memory_allocated(0) / 1024 ** 2)
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != 'members'}), flush=True)
    if sha256_file(train) != TRAIN_SHA or sha256_file(weights) != WEIGHT_SHA:
        raise ValueError('Smoke input identity changed during execution')
    names = ['v348_memory_smoke.py', 'v348_clean_memory_core.py',
             'v180_dinomaly3_dinov3_csig.py', 'v182_dinomaly2_dinov3_csig_official.py']
    result = dict(status='ok', version='v348_real_normal_memory_smoke_v1', capacity_smoke_only=True,
                  calibration_not_for_production=True, torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(0), train_zip_sha256=TRAIN_SHA, weights_sha256=WEIGHT_SHA,
                  code_hashes={name: sha256_file(Path(__file__).with_name(name)) for name in names},
                  partition_sha256=sha256_file(output / 'source_partition.json'), resolutions=rows,
                  scope='Real normal-source feature extraction and numeric smoke only; no quality metrics or B access.')
    with (output / 'smoke.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print('v348_memory_smoke_complete', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    run(parser.parse_args().output_root)
