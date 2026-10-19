"""Fixed held-fold view-memory experiment; auxiliary data only, no training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import v364_calibration_alignment as baseline
from scripts.v180_dinomaly3_dinov3_csig import _extract_patch_tokens
from scripts.v348_clean_memory_core import fuse_memory, normal_quantile_scale, resize_raw_map
from scripts.v350_scaled_evidence import ScaledEvidenceMetrics
from scripts.v351_source_calibration import ResidentCosineMemory
from scripts.v367_view_memory_core import VARIANTS, memory_variants, view_feature_indices

trainer, prior, ROOT = baseline.trainer, baseline.prior, baseline.ROOT
BASELINE_SHAS = (
    '08e6d025246cd09630dd5d0821991cb9eb69df60f6f41a311a51e3630eaa4f8f',
    'a94b1081ea519c9c4161f929b4e592f3e1b7e5f3ba012819f24d1b2d29a95e74',
    'd6e7a930082d78b5e3ec7d7654d1d8b3ac71996288cc53798d518c4ad0d5feed',
    '057e90f34bbc23aed5a7bdb36385407eb6dfe7a26ef16229dbe212f698aa17a4')


class ViewMemory:
    def __init__(self, memory, bank_rows, *, device):
        import torch
        indices = view_feature_indices(bank_rows)
        memory = np.asarray(memory)
        if memory.ndim != 2 or memory.shape[0] != 4000:
            raise ValueError('Expected the original4000 feature category memory')
        self.category = bank_rows[0]['category']
        self.all_view = ResidentCosineMemory(memory, device=device, chunk_size=8192)
        self.view_columns = {v: torch.tensor(ii, dtype=torch.long, device=device) for v, ii in indices.items()}

    def distances(self, tokens, rows):
        import torch
        import torch.nn.functional as functional
        if tokens.ndim != 3 or not 1 <= len(rows) == len(tokens) <= 8:
            raise ValueError('Expected aligned query rows in batches of at most8')
        if any(row['category'] != self.category for row in rows):
            raise ValueError('Query category differs from category memory')
        views = [baseline.canonical_view(row['view']) for row in rows]
        memory = self.all_view.memory
        if (tokens.device != memory.device or tokens.shape[-1] != memory.shape[-1]
                or not torch.isfinite(tokens).all()
                or torch.any(torch.linalg.vector_norm(tokens.float(), dim=-1) <= 1e-8)):
            raise ValueError('Invalid query features/device')
        tokens = functional.normalize(tokens.float(), dim=-1)
        all_values, same_values = [], []
        with torch.inference_mode():
            for query, view in zip(tokens, views):
                # Share the original4000-column GEMM; a separate800-column GEMM
                # rounds differently on P100 and can break subset monotonicity.
                similarities = query @ memory.T
                all_best = similarities.max(dim=1).values
                same_best = similarities.index_select(1, self.view_columns[view]).max(dim=1).values
                all_values.append((1. - all_best).clamp(0., 2.).cpu().numpy())
                same_values.append((1. - same_best).clamp(0., 2.).cpu().numpy())
        all_maps, same_maps = np.stack(all_values), np.stack(same_values)
        if np.any(same_maps < all_maps - 1e-6):
            raise ValueError('Same-view subset distance is smaller than all-view distance')
        return all_maps, same_maps


def _maps(model, rows, resident):
    import torch
    kw = dict(batch_size=8, shuffle=False, num_workers=prior.DATA_LOADER_WORKERS, pin_memory=True)
    loaders = [torch.utils.data.DataLoader(trainer.RealIADSupervisedDataset(rows, size), **kw)
               for size in (448, 560)]
    refs = np.empty((len(rows), 448, 448), np.float32)
    variants = {name: np.empty_like(refs) for name in VARIANTS}
    masks, labels, offset = [], [], 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for base, high in zip(*loaders, strict=True):
            current = rows[offset:offset + len(base[0])]
            for batch in (base, high):
                if (list(batch[3]) != [r['category'] for r in current]
                        or list(batch[4]) != [r['group'] for r in current]
                        or batch[2].tolist() != [int(r['label']) for r in current]):
                    raise ValueError('Paired loader identities changed')
            e, d = model(base[0].to(device))
            ref = trainer.differentiable_anomaly_map(e, d, 448).float().cpu().numpy()[:, 0]
            tokens, grid = _extract_patch_tokens(model.encoder, high[0].to(device), [3, 7, 11])
            if grid != (35, 35):
                raise ValueError('Expected frozen560 patch grid')
            all_maps, same_maps = resident.distances(tokens, current)
            raw = memory_variants(*[np.stack([resize_raw_map(x.reshape(grid)) for x in batch])
                                    for batch in (all_maps, same_maps)])
            refs[offset:offset + len(current)] = ref
            for name in VARIANTS:
                variants[name][offset:offset + len(current)] = raw[name]
            masks.extend(base[1][:, 0].cpu().numpy() > .5)
            labels.extend(base[2].tolist())
            offset += len(current)
            del e, d, tokens
    if offset != len(rows):
        raise ValueError('Incomplete paired inference stream')
    return refs, variants, masks, labels


def score_policies(rows, refs, variantmaps, masks, reference_q99, memory_q99dict):
    if (set(variantmaps) != set(VARIANTS) or set(memory_q99dict) != set(VARIANTS)
            or not len(rows) or any(len(x) != len(rows) for x in [refs, masks, *variantmaps.values()])):
        raise ValueError('Evaluation streams or policy keys differ')
    scales = {name: normal_quantile_scale(reference_q99, memory_q99dict[name]) for name in VARIANTS}
    metrics = {name: ScaledEvidenceMetrics() for name in VARIANTS}
    for i, row in enumerate(rows):
        score = trainer.top_mean(refs[i], .005)
        for name in VARIANTS:
            metrics[name].add(category=row['category'], group=row['group'],
                view=baseline.canonical_view(row['view']), label=int(row['label']), mask=masks[i],
                raw=fuse_memory(refs[i], variantmaps[name][i], scale=scales[name]), reference_view_score=score)
    result = {name: metric.finish() for name, metric in metrics.items()}
    for name in VARIANTS:
        for category, item in result[name]['metrics']['per_category'].items():
            if item['image'] != result['all_view']['metrics']['per_category'][category]['image']:
                raise ValueError('Reference classification changed between policies')
    return result


def query_stream_sha256(refs, memories):
    digest = hashlib.sha256()
    for stream in (refs, memories):
        for raw in stream:
            digest.update(np.ascontiguousarray(raw, dtype='<f4'))
    return digest.hexdigest()


def baseline_replay(frozen, actual):
    for field in ('memory_sha256', 'query_raw_stream_sha256'):
        if frozen[field] != actual[field]:
            raise ValueError(f'Frozen baseline replay mismatch: {field}')
    expected = {'reference_q99': frozen['calibration']['category_ref_q99'],
                'memory_q99': frozen['calibration']['memory_q99']}
    current = {'reference_q99': actual['calibration']['reference_q99'],
               'memory_q99': actual['calibration']['memory_q99']['all_view']}
    drifts = {'calibration': baseline.evaluation_drift(expected, current),
              'metrics': baseline.evaluation_drift(frozen['metrics']['category_calibrated'], actual['metrics']['all_view']),
              'reserved_normal': baseline.evaluation_drift(frozen['reserved_normal']['category_calibrated'], actual['reserved_normal']['all_view'])}
    if max(drifts.values()) > 1e-6:
        raise ValueError(f'Frozen baseline replay drift: {drifts}')
    return drifts


def _verify_hashes(hashes):
    if any(trainer.sha256_file(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('Bound input/source/code hash changed')


def _input_audit(partitions, approved, expected):
    files, pixels = {}, {}
    for phase, rows in partitions.items():
        pixels[phase] = {}
        for row in rows:
            for field in ('image_path', 'mask_path'):
                if row.get(field):
                    path = Path(row[field])
                    if (not path.resolve().is_relative_to(approved) or 'test_b' in str(path.resolve()).lower()
                            or path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                        raise ValueError('Unapproved auxiliary image or mask')
                    files[str(path)] = trainer.sha256_file(path)
            pixels[phase][row['image_path']] = baseline.decoded_rgb448_sha256(row['image_path'])
    if files != expected:
        raise ValueError('Auxiliary data differ from frozen baseline')
    return baseline.require_pixel_disjoint(pixels), pixels


def run(args):
    import torch
    output = (Path(args.output_root) / f'fold{args.fold}').resolve()
    if args.fold not in range(4) or output.exists() or not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('New project output folder and one held fold required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU per held-fold process; device is cuda:0')
    frozen_path = ROOT / f'outputs/v364_calibration_alignment_20260912/fold{args.fold}/report.json'
    prior_path = ROOT / 'outputs/v360_seen_reference_validation_560_20260912/report.json'
    sources = {str(frozen_path): BASELINE_SHAS[args.fold], str(prior.INDEX): baseline.INDEX_SHA,
               str(prior_path): baseline.PRIOR_SHA}
    _verify_hashes(sources)
    frozen, previous = baseline._read(frozen_path), baseline._read(prior_path)
    if frozen['status'] != 'ok' or frozen['fold'] != args.fold:
        raise ValueError('Wrong frozen baseline fold')
    sources.update(frozen['source_hashes'])
    _verify_hashes(sources)
    bundle = baseline.load_verified_bundle(ROOT / 'outputs/v343_verified_longrun_20260911', prior.GATE)
    record = bundle['records'][args.fold]
    selected = record['selected']
    held = sorted(selected['evaluation']['metrics']['per_category'])
    old_fold = next(f for f in previous['folds_detail'] if f['fold'] == args.fold)
    if set(held) != set(frozen['details']) or set(held) != set(old_fold['details']):
        raise ValueError('Eligible held categories changed')
    sources.update(bundle['selection_record_hashes'])
    sources.update({selected['checkpoint']: selected['checkpoint_sha256'], str(prior.GATE): bundle['source_gate_sha256']})
    codes = {**baseline._code_hashes(), **{str(Path(__file__).with_name(n)): trainer.sha256_file(Path(__file__).with_name(n))
             for n in ('v367_view_memory_validation.py', 'v367_view_memory_core.py', 'v339_verify_clean_metrics.py',
                       'v342_clean_longtrain.py', 'v344_longrun_submission.py')}}
    _verify_hashes(sources)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(36401 + args.fold)
    model, deps = prior._model(Path(selected['checkpoint']), 0)
    runtime = baseline.longtrain.runtime_source_hashes(deps, model)
    if runtime != record['long_config']['runtime_source_hashes'] or runtime != frozen['runtime_source_hashes']:
        raise ValueError('Reference runtime differs from frozen checkpoint')
    rows = trainer.load_proxy_rows(prior.INDEX)
    approved = Path(record['long_config']['approved_image_root']).resolve()
    output.mkdir(parents=True)
    results, started = {}, time.monotonic()
    for category in held:
        old = frozen['details'][category]
        split = old_fold['details'][category]['split']
        if split != old['split']:
            raise ValueError('Frozen source partition changed')
        parts = baseline.category_partitions(rows, category, split)
        audit, pixels = _input_audit(parts, approved, old['input_file_hashes'])
        memory = prior._bank(model, parts['bank'], 560)
        memory_sha = hashlib.sha256(memory.tobytes()).hexdigest()
        if memory_sha != old['memory_sha256']:
            raise ValueError('Frozen baseline memory replay mismatch')
        resident = ViewMemory(memory, parts['bank'], device=next(model.parameters()).device)
        del memory
        ref, maps, masks, _ = _maps(model, parts['calibration'], resident)
        rq, mq = prior.source_q99(ref), {name: prior.source_q99(maps[name]) for name in VARIANTS}
        del ref, maps, masks
        ref, maps, masks, _ = _maps(model, parts['evaluation'], resident)
        metrics = score_policies(parts['evaluation'], ref, maps, masks, rq, mq)
        query_sha = query_stream_sha256(ref, maps['all_view'])
        del ref, maps, masks
        ref, maps, masks, labels = _maps(model, parts['normal_audit'], resident)
        if any(labels) or any(np.asarray(mask).any() for mask in masks):
            raise ValueError('Reserved audit images must remain normal')
        normal = {name: baseline.normal_audit_metrics(np.stack([
            fuse_memory(r, m, scale=normal_quantile_scale(rq, mq[name])) for r, m in zip(ref, maps[name])]),
            reference_q99=rq) for name in VARIANTS}
        del ref, maps, masks, resident
        repeated, after_pixels = _input_audit(parts, approved, old['input_file_hashes'])
        if repeated != audit or after_pixels != pixels:
            raise ValueError('Decoded auxiliary partitions changed')
        result = {'category': category, 'split': split, 'metrics': metrics, 'reserved_normal': normal,
                  'calibration': {'reference_q99': rq, 'memory_q99': mq}, 'duplicate_audit': audit,
                  'input_file_hashes': old['input_file_hashes'], 'memory_sha256': memory_sha,
                  'query_raw_stream_sha256': query_sha}
        result['frozenbaseline_replaydrift'] = baseline_replay(old, result)
        with (output / f'{category}.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        results[category] = result
        print(f'fold{args.fold}: {category} complete', flush=True)
    _verify_hashes(sources)
    _verify_hashes(codes)
    if baseline.longtrain.runtime_source_hashes(deps, model) != runtime:
        raise ValueError('Model runtime changed')
    result = {'status': 'ok', 'version': 'v367_view_memory_validation_v1', 'fold': args.fold,
              'details': results, 'variants': list(VARIANTS), 'input_size': 560, 'batch_size': 8,
              'source_hashes': sources, 'code_hashes': codes, 'runtime_source_hashes': runtime,
              'frozenbaseline_replaydrift': max(max(r['frozenbaseline_replaydrift'].values()) for r in results.values()),
              'elapsed_seconds': time.monotonic() - started, 'test_b_labels_accessed': False,
              'test_b_zip_accessed': False, 'quality_status': 'development_comparison_only',
              'scope': 'Seen-reference held-fold auxiliary simulation, not zero-shot or B ensemble evidence'}
    with (output / 'report.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=range(4), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    run(parser.parse_args())
