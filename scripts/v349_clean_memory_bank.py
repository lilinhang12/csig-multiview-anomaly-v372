"""Traceable normal-only560/672 DINOv3 banks with held-out calibration objects."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts.build_b_submission import _test_index
from scripts.v180_dinomaly3_dinov3_csig import CsigZipDataset, _extract_patch_tokens
from scripts.v283_supervised_dinomaly_longtrain import sha256_file
from scripts.v342_clean_longtrain import runtime_source_hashes
from scripts.v348_clean_memory_core import validate_partition
from scripts.v348_memory_smoke import ROOT, TRAIN_SHA, WEIGHT_SHA

SCHEMA = 'v349_normal_patch_bank_v1'
PATCHES_PER_IMAGE = 50
PARTITION_SHA = '5afe17f53ddde0d8d146d07d6e5414e72fa1b40a561c29f33100bf238ceabcee'
CODE_FILES = ('v349_clean_memory_bank.py', 'v348_clean_memory_core.py', 'v348_memory_smoke.py',
              'v342_clean_longtrain.py', 'v283_supervised_dinomaly_longtrain.py',
              'v182_dinomaly2_dinov3_csig_official.py', 'v180_dinomaly3_dinov3_csig.py', 'build_b_submission.py')


def member_patch_indices(member, token_count):
    if (not isinstance(member, str) or not re.fullmatch(r'Train/[^/\\\x00]+/S\d{4}/[0-4]\.png', member)
            or member.split('/')[1] in ('.', '..') or not isinstance(token_count, (int, np.integer))
            or token_count < PATCHES_PER_IMAGE):
        raise ValueError('Expected a normal Train member with enough patches')
    seed = int.from_bytes(hashlib.sha256(('34801:' + member).encode()).digest()[:8], 'little')
    return np.sort(np.random.default_rng(seed).choice(token_count, PATCHES_PER_IMAGE, replace=False))


def expected_bank_members(partition):
    return [f'Train/{group}/{view}.png' for group in partition['bank_groups'] for view in range(5)]


def validate_bank_trace(memory, image_indices, patch_indices, members, *, groups, partition, input_size):
    validate_partition(groups, partition)
    if input_size not in (560, 672) or list(members) != expected_bank_members(partition):
        raise ValueError('Bank member coverage/order or input grid mismatch')
    count = len(members) * PATCHES_PER_IMAGE
    if (memory.shape != (count, 2304) or memory.dtype != np.float16
            or np.shape(image_indices) != (count,) or np.shape(patch_indices) != (count,)
            or np.asarray(image_indices).dtype.kind not in 'iu' or np.asarray(patch_indices).dtype.kind not in 'iu'):
        raise ValueError('Invalid bank feature/trace shape or dtype')
    if not np.array_equal(image_indices, np.repeat(np.arange(len(members)), PATCHES_PER_IMAGE)):
        raise ValueError('Feature rows differ from source-image order')
    tokens = (input_size // 16) ** 2
    for index, member in enumerate(members):
        start = index * PATCHES_PER_IMAGE
        if not np.array_equal(patch_indices[start:start + PATCHES_PER_IMAGE], member_patch_indices(member, tokens)):
            raise ValueError('Patch indices differ from fixed per-member sampling')
    for start in range(0, count, 4096):
        chunk = np.asarray(memory[start:start + 4096], np.float32)
        if not np.isfinite(chunk).all() or not np.allclose(np.linalg.norm(chunk, axis=1), 1., atol=.003, rtol=0):
            raise ValueError('Bank contains nonfinite or unnormalized features')
    return dict(images=len(members), patches=count, feature_dim=2304, dtype='float16',
                objects=len(partition['bank_groups']), grid=[input_size // 16] * 2)


def code_hashes():
    return {name: sha256_file(Path(__file__).with_name(name)) for name in CODE_FILES}


def preprocessing_environment_identity():
    import sys as python_sys
    import torch
    import torchvision
    from PIL import Image
    from torchvision import transforms

    source_objects = (transforms.Resize, transforms.CenterCrop, transforms.ToTensor,
                      transforms.Normalize, transforms.functional.resize,
                      transforms.functional.to_tensor, transforms.functional.normalize,
                      Image)
    source_paths = sorted({Path(inspect.getsourcefile(item)).resolve() for item in source_objects
                           if inspect.getsourcefile(item)})
    source_digest = hashlib.sha256()
    for path in source_paths:
        source_digest.update(str(path).encode('utf-8') + b'\0')
        source_digest.update(bytes.fromhex(sha256_file(path)))
    capability = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    return {
        'python_version': python_sys.version,
        'torch_version': str(torch.__version__),
        'torchvision_version': str(torchvision.__version__),
        'pillow_version': importlib.metadata.version('Pillow'),
        'numpy_version': np.__version__,
        'cuda_version': str(torch.version.cuda),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_capability': capability,
        'preprocess_source_paths': [str(path) for path in source_paths],
        'preprocess_source_sha256': source_digest.hexdigest(),
    }


def source_inputs():
    train = ROOT / 'Train.zip'
    weight = ROOT / 'Dinomaly2_dinov3/backbones/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    partition_path = ROOT / 'outputs/v348_memory_smoke_reviewed_20260911/source_partition.json'
    if sha256_file(train) != TRAIN_SHA or sha256_file(weight) != WEIGHT_SHA or sha256_file(partition_path) != PARTITION_SHA:
        raise ValueError('Normal source, foundation weights or reviewed partition changed')
    record = json.loads(partition_path.read_text(encoding='utf-8'))
    if record['train_zip_sha256'] != TRAIN_SHA or record['weights_sha256'] != WEIGHT_SHA:
        raise ValueError('Partition source identity mismatch')
    groups = _test_index(train, 'Train')
    partition = record['partition']
    validate_partition(groups, partition)
    if (len(groups) != 1000 or len(partition['categories']) != 50
            or partition['bank_views'] != 4000 or partition['calibration_views'] != 1000):
        raise ValueError('Full official normal source partition is required')
    return train, weight, groups, partition


def load_verified_bank(run_root):
    root = Path(run_root)
    report = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if report.get('status') != 'ok' or report.get('version') != SCHEMA:
        raise ValueError('Incomplete normal bank')
    meta = report['metadata']
    if meta['code_hashes'] != code_hashes():
        raise ValueError('Bank implementation differs from its producer')
    _, _, groups, partition = source_inputs()
    if (meta['partition'] != partition or meta['partition_sha256'] != PARTITION_SHA
            or meta['train_zip_sha256'] != TRAIN_SHA or meta['weights_sha256'] != WEIGHT_SHA
            or meta['feature_layers'] != [3, 7, 11] or meta['patches_per_image'] != 50):
        raise ValueError('Bank source/feature contract mismatch')
    if meta.get('preprocess_environment') != preprocessing_environment_identity():
        raise ValueError('Bank preprocessing environment differs from the current runtime')
    if any(sha256_file(Path(p)) != value for p, value in meta['runtime_source_hashes'].items()):
        raise ValueError('Encoder runtime code changed')
    path = root / 'memory.npz'
    if sha256_file(path) != report['bank_sha256']:
        raise ValueError('Bank payload changed')
    with np.load(path, allow_pickle=False) as data:
        if str(data['schema']) != SCHEMA or json.loads(str(data['metadata'])) != meta:
            raise ValueError('Serialized bank metadata mismatch')
        memory = data['memory']
        images, patches = data['image_indices'], data['patch_indices']
        members = data['members'].tolist()
    stats = validate_bank_trace(memory, images, patches, members, groups=groups,
                                partition=partition, input_size=meta['input_size'])
    if stats != report['trace'] or len(memory) != 200000:
        raise ValueError('Full bank trace differs from published report')
    return memory, report


def run(args):
    import torch

    output = (Path(args.output_root) / f'i{args.input_size}').resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output.exists() or args.input_size not in (560, 672):
        raise ValueError('New560/672 output under the project is required')
    if shutil.disk_usage(ROOT).free < 5 * 1024 ** 3:
        raise RuntimeError('5GiB free disk required before normal-bank extraction')
    train, weight, groups, partition = source_inputs()
    hashes = code_hashes()
    preprocess_environment = preprocessing_environment_identity()
    members = expected_bank_members(partition)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 12 * 1024 ** 3:
        raise RuntimeError('An available GPU with12GiB free is required')
    torch.set_num_threads(4)
    torch.manual_seed(34801)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    deps = adapter._import_official_modules()
    model, _ = adapter._build_model(deps, backbone='dinov3_vit_base_16', dropout=.4, reduce_dim=256,
                                    context_aware_recenter=True, mask_neighbor_size=0, weights_dir=weight.parent)
    encoder = model.encoder.to('cuda:0').eval()
    encoder.requires_grad_(False)
    del model
    runtime_hashes = runtime_source_hashes(deps, encoder)
    transform = adapter._build_transform(args.input_size, args.input_size)
    dataset = CsigZipDataset(train, 'Train', transform, include_metadata=True)
    allowed = set(partition['bank_groups'])
    indices = [i for i, record in enumerate(dataset.records) if record.group in allowed]
    if [dataset.records[i].member for i in indices] != members:
        raise ValueError('Actual dataset differs from source-bank partition')
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=8,
                    shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    output.mkdir(parents=True)
    count = len(members) * PATCHES_PER_IMAGE
    memory = np.empty((count, 2304), dtype=np.float16)
    image_indices = np.repeat(np.arange(len(members), dtype=np.int32), PATCHES_PER_IMAGE)
    patch_indices = np.empty(count, dtype=np.uint16)
    cursor = 0
    started = time.monotonic()
    with torch.inference_mode():
        for images, categories, samples, views in loader:
            tokens, grid = _extract_patch_tokens(encoder, images.to('cuda:0', non_blocking=True), [3, 7, 11])
            if grid != (args.input_size // 16,) * 2 or tokens.shape[-1] != 2304:
                raise ValueError('DINOv3 feature geometry mismatch')
            if cursor == 0 and any(float((tokens[..., a:a + 768] - tokens[..., b:b + 768]).abs().max()) < 1e-6
                                   for a, b in ((0, 768), (768, 1536))):
                raise ValueError('Intermediate-layer features are duplicated')
            for i, (category, sample, view) in enumerate(zip(categories, samples, views)):
                member = f'Train/{category}/{sample}/{view}.png'
                if member != members[cursor]:
                    raise ValueError('Loaded image identity/order changed')
                chosen = member_patch_indices(member, int(tokens.shape[1]))
                start = cursor * PATCHES_PER_IMAGE
                memory[start:start + PATCHES_PER_IMAGE] = tokens[i, chosen].float().cpu().numpy().astype(np.float16)
                patch_indices[start:start + PATCHES_PER_IMAGE] = chosen
                cursor += 1
            if cursor % 256 == 0 or cursor == len(members):
                print(f'bank_i{args.input_size}_images={cursor}/{len(members)}', flush=True)
        if cursor != len(members):
            raise ValueError('Normal bank extraction stopped early')
        replay = []
        for index in (0, len(members) // 2, len(members) - 1):
            image = dataset[indices[index]][0].unsqueeze(0).to('cuda:0')
            tokens, _ = _extract_patch_tokens(encoder, image, [3, 7, 11])
            chosen = member_patch_indices(members[index], int(tokens.shape[1]))
            actual = tokens[0, chosen].float().cpu().numpy()
            saved = memory[index * 50:(index + 1) * 50].astype(np.float32)
            drift = float(np.abs(saved - actual).max())
            if not np.isfinite(drift) or drift > 5e-4:
                raise ValueError('Sampled features cannot be reproduced from their source members')
            replay.append({'member': members[index], 'max_abs_float16_error': drift})
    trace = validate_bank_trace(memory, image_indices, patch_indices, members, groups=groups,
                                partition=partition, input_size=args.input_size)
    if code_hashes() != hashes or runtime_source_hashes(deps, encoder) != runtime_hashes:
        raise ValueError('Bank source code changed during extraction')
    source_inputs()
    meta = dict(input_size=args.input_size, feature_layers=[3, 7, 11], patches_per_image=50,
                sampling='sha256(34801:member) first8 little-endian seed; PCG64 choice without replacement',
                normalization='per-layer_l2_concat_l2; float16_storage; renormalize_fp32_for_matching',
                train_zip_sha256=TRAIN_SHA, weights_sha256=WEIGHT_SHA, partition_sha256=PARTITION_SHA,
                partition=partition, code_hashes=hashes, runtime_source_hashes=runtime_hashes,
                preprocess_environment=preprocess_environment,
                torch_version=str(torch.__version__), numpy_version=np.__version__, transform=repr(transform))
    path = output / 'memory.npz'
    with path.open('xb') as handle:
        np.savez_compressed(handle, memory=memory, image_indices=image_indices, patch_indices=patch_indices,
                            members=np.asarray(members), schema=np.asarray(SCHEMA),
                            metadata=np.asarray(json.dumps(meta, sort_keys=True)))
    with np.load(path, allow_pickle=False) as stored:
        if not np.array_equal(stored['memory'], memory):
            raise ValueError('Stored bank differs from extracted features')
    result = dict(status='ok', version=SCHEMA, bank_sha256=sha256_file(path), metadata=meta,
                  trace=trace, feature_replay=replay, elapsed_seconds=time.monotonic() - started,
                  bytes=path.stat().st_size, peak_cuda_mib=torch.cuda.max_memory_allocated(0) / 1024 ** 2,
                  scope='Official normal Train memory only; calibration and quality validation remain separate')
    with (output / 'manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status', 'bank_sha256', 'trace', 'elapsed_seconds', 'bytes', 'peak_cuda_mib')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-size', type=int, choices=(560, 672), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    run(parser.parse_args())
