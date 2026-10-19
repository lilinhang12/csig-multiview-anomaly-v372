"""Reproducible source-normal q99 calibration for fixed reference/memory branches."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts import v342_clean_longtrain as longtrain
from scripts.v180_dinomaly3_dinov3_csig import CsigZipDataset, _extract_patch_tokens
from scripts.v330_raw_response import checked_raw
from scripts.v344_longrun_submission import inference_hashes, load_verified_bundle
from scripts.v348_clean_memory_core import resize_raw_map
from scripts.v348_memory_smoke import ROOT, TRAIN_SHA, WEIGHT_SHA
from scripts.v349_clean_memory_bank import (PARTITION_SHA, code_hashes as bank_code_hashes,
                                            load_verified_bank, preprocessing_environment_identity, source_inputs)

SCHEMA = 'v351_normal_source_calibration_v1'


class ResidentCosineMemory:
    def __init__(self, memory, *, device='cuda:0', chunk_size=8192):
        import torch
        import torch.nn.functional as functional

        array = np.asarray(memory, dtype=np.float32)
        if (array.ndim != 2 or not array.size or not np.isfinite(array).all()
                or np.any(np.linalg.norm(array, axis=1) <= 1e-8)
                or not isinstance(chunk_size, int) or chunk_size <= 0):
            raise ValueError('Invalid memory features or chunk size')
        self.memory = functional.normalize(torch.from_numpy(array).to(device), dim=-1)
        self.chunk_size = chunk_size

    def distances(self, tokens):
        import torch
        import torch.nn.functional as functional

        if (tokens.ndim != 3 or tokens.shape[-1] != self.memory.shape[-1]
                or tokens.device != self.memory.device or not torch.isfinite(tokens).all()
                or torch.any(torch.linalg.vector_norm(tokens.float(), dim=-1) <= 1e-8)):
            raise ValueError('Invalid query features/device')
        tokens = functional.normalize(tokens.float(), dim=-1)
        output = []
        with torch.inference_mode():
            for query in tokens:
                best = torch.full((query.shape[0],), -torch.inf, device=query.device)
                for start in range(0, len(self.memory), self.chunk_size):
                    scores = query @ self.memory[start:start + self.chunk_size].T
                    best = torch.maximum(best, scores.max(dim=1).values)
                output.append((1. - best).clamp(0., 2.).cpu().numpy())
        return np.stack(output).astype(np.float32)


def collect_calibration(members, batches):
    if (not members or len(set(members)) != len(members)
            or any(not re.fullmatch(r'Train/[^/\\\x00]+/S\d{4}/[0-4]\.png', m)
                   or m.split('/')[1] in ('.', '..') for m in members)):
        raise ValueError('Calibration requires unique normal Train members')
    pool = np.empty((len(members), 448, 448), dtype=np.float32)
    digest = hashlib.sha256()
    cursor = 0
    for actual_members, values in batches:
        if (not actual_members or list(actual_members) != members[cursor:cursor + len(actual_members)]
                or np.shape(values) != (len(actual_members), 448, 448)):
            raise ValueError('Calibration member ordering/shape changed')
        raw = checked_raw(values).astype('<f4')
        pool[cursor:cursor + len(actual_members)] = raw
        for member, image in zip(actual_members, raw):
            digest.update(member.encode('utf-8') + b'\0')
            digest.update(image.tobytes(order='C'))
        cursor += len(actual_members)
    if cursor != len(members):
        raise ValueError('Incomplete normal calibration stream')
    q99 = float(np.quantile(pool, .99, method='linear'))
    if not np.isfinite(q99) or q99 <= 1e-8:
        raise ValueError('Degenerate source-normal calibration')
    return dict(views=cursor, pixels=int(pool.size), q99=q99, quantile=.99, quantile_method='linear',
                minimum=float(pool.min()), maximum=float(pool.max()),
                response_stream_sha256=digest.hexdigest(), dtype='float32')


def code_hashes():
    return {**bank_code_hashes(), **inference_hashes(),
            'v351_source_calibration.py': trainer.sha256_file(Path(__file__))}


def calibration_environment_identity():
    return preprocessing_environment_identity()


def _reference(fold):
    import torch

    bundle = load_verified_bundle(ROOT / 'outputs/v343_verified_longrun_20260911',
                                 ROOT / 'outputs/v339_independent_clean_replay_final_20260910/clean_selection_gate.json')
    record = bundle['records'][fold]
    choice = record['selected']
    state = torch.load(choice['checkpoint'], map_location='cpu', weights_only=False)
    if trainer.sha256_file(Path(choice['checkpoint'])) != choice['checkpoint_sha256']:
        raise ValueError('Reference checkpoint changed')
    if choice['kind'] == 'source_fallback':
        longtrain.validate_source(record['source'], state, choice['checkpoint_sha256'], fold)
    elif (state['immutable_config'] != record['long_config'] or state['baseline_sha256'] != record['baseline_sha256']
          or state['relative_additional_step'] != choice['additional_steps'] or state['total_optimizer_step'] != choice['total_steps']):
        raise ValueError('Reference checkpoint metadata mismatch')
    deps = adapter._import_official_modules()
    model, _ = adapter._build_model(deps, backbone='dinov3_vit_base_16', dropout=.4, reduce_dim=256,
            context_aware_recenter=True, mask_neighbor_size=0, weights_dir=ROOT / 'Dinomaly2_dinov3/backbones/weights')
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.to('cuda:0').eval()
    model.requires_grad_(False)
    runtime = longtrain.runtime_source_hashes(deps, model)
    if runtime != record['long_config']['runtime_source_hashes']:
        raise ValueError('Reference runtime differs from validated model')
    identity = dict(fold_id=fold, checkpoint=choice['checkpoint'], checkpoint_sha256=choice['checkpoint_sha256'],
                    total_steps=choice['total_steps'], selection_record_hashes=bundle['selection_record_hashes'])
    return model, deps, runtime, identity


def run(args):
    import torch

    if args.kind == 'reference' and (args.fold is None or args.bank_root is not None):
        raise ValueError('Reference calibration requires a fold and no memory bank')
    if args.kind == 'memory' and (args.bank_root is None or args.fold is not None):
        raise ValueError('Memory calibration requires a bank and no reference fold')
    train, weight, groups, partition = source_inputs()
    hashes = code_hashes()
    preprocess_environment = calibration_environment_identity()
    members = [f'Train/{group}/{view}.png' for group in partition['calibration_groups'] for view in range(5)]
    if len(members) != 1000:
        raise ValueError('All1,000 source calibration views are required')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 12 * 1024 ** 3:
        raise RuntimeError('An available GPU with12GiB free is required')
    torch.set_num_threads(4)
    torch.manual_seed(34801)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    resident = None
    if args.kind == 'reference':
        model, deps, runtime, identity = _reference(args.fold)
        size, name = 448, f'reference_fold{args.fold}'
    else:
        bank_root = Path(args.bank_root).resolve()
        if not bank_root.is_relative_to(ROOT / 'outputs'):
            raise ValueError('Bank must be in the approved output tree')
        memory, report = load_verified_bank(bank_root)
        size = report['metadata']['input_size']
        name = f'memory_i{size}'
        deps = adapter._import_official_modules()
        full, _ = adapter._build_model(deps, backbone='dinov3_vit_base_16', dropout=.4, reduce_dim=256,
                      context_aware_recenter=True, mask_neighbor_size=0, weights_dir=weight.parent)
        model = full.encoder.to('cuda:0').eval()
        model.requires_grad_(False)
        del full
        runtime = longtrain.runtime_source_hashes(deps, model)
        if runtime != report['metadata']['runtime_source_hashes']:
            raise ValueError('Calibration encoder differs from bank producer')
        if report['metadata'].get('preprocess_environment') != preprocess_environment:
            raise ValueError('Calibration preprocessing environment differs from bank producer')
        resident = ResidentCosineMemory(memory)
        del memory
        identity = dict(bank_root=str(bank_root), bank_sha256=report['bank_sha256'],
                        bank_manifest_sha256=trainer.sha256_file(bank_root / 'manifest.json'))
    output = (Path(args.output_root) / name).resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output.exists():
        raise ValueError('A new project calibration output is required')
    transform = adapter._build_transform(size, size)
    dataset = CsigZipDataset(train, 'Train', transform, include_metadata=True)
    allowed = set(partition['calibration_groups'])
    indices = [i for i, record in enumerate(dataset.records) if record.group in allowed]
    if [dataset.records[i].member for i in indices] != members:
        raise ValueError('Actual calibration views differ from the frozen partition')
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=8,
                        shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    output.mkdir(parents=True)
    started = time.monotonic()

    def batches(pass_number):
        seen = 0
        with torch.inference_mode():
            for images, categories, samples, views in loader:
                images = images.to('cuda:0', non_blocking=True)
                if resident is None:
                    encoded, decoded = model(images)
                    maps = trainer.differentiable_anomaly_map(encoded, decoded, 448).float().cpu().numpy()[:, 0]
                else:
                    tokens, grid = _extract_patch_tokens(model, images, [3, 7, 11])
                    if grid != (size // 16,) * 2:
                        raise ValueError('Calibration patch grid changed')
                    maps = np.stack([resize_raw_map(raw.reshape(grid)) for raw in resident.distances(tokens)])
                current = [f'Train/{c}/{s}/{v}.png' for c, s, v in zip(categories, samples, views)]
                yield current, maps
                seen += len(current)
                if seen % 200 == 0:
                    print(f'calibration_{name}_pass{pass_number}={seen}/1000', flush=True)

    first = collect_calibration(members, batches(1))
    second = collect_calibration(members, batches(2))
    if first != second:
        raise ValueError('Full source-calibration replay does not reproduce exactly')
    source_inputs()
    if code_hashes() != hashes or longtrain.runtime_source_hashes(deps, model) != runtime:
        raise ValueError('Calibration implementation changed during the run')
    if resident is None:
        if trainer.sha256_file(Path(identity['checkpoint'])) != identity['checkpoint_sha256'] or any(
                trainer.sha256_file(Path(p)) != h for p, h in identity['selection_record_hashes'].items()):
            raise ValueError('Reference selection changed during calibration')
    elif (trainer.sha256_file(Path(identity['bank_root']) / 'memory.npz') != identity['bank_sha256']
          or trainer.sha256_file(Path(identity['bank_root']) / 'manifest.json') != identity['bank_manifest_sha256']):
        raise ValueError('Memory bank changed during calibration')
    result = dict(status='ok', version=SCHEMA, kind=args.kind, input_size=size, identity=identity,
                  calibration=first, replay_identical=True, calibration_members=members,
                  member_list_sha256=hashlib.sha256(json.dumps(members, separators=(',', ':')).encode()).hexdigest(),
                  train_zip_sha256=TRAIN_SHA, weights_sha256=WEIGHT_SHA, partition_sha256=PARTITION_SHA,
                  code_hashes=hashes, runtime_source_hashes=runtime, torch_version=str(torch.__version__),
                  numpy_version=np.__version__, preprocess_environment=preprocess_environment,
                  transform=repr(transform),
                  elapsed_seconds=time.monotonic() - started,
                  peak_cuda_mib=torch.cuda.max_memory_allocated(0) / 1024 ** 2,
                  scope='Source-normal calibration only; no target-label fitting or quality claim')
    with (output / 'manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({'status': 'ok', 'name': name, 'q99': first['q99'],
                      'replay_identical': True, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('reference', 'memory'), required=True)
    parser.add_argument('--fold', type=int, choices=range(4))
    parser.add_argument('--bank-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    run(parser.parse_args())
