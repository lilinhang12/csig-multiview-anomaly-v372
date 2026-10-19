"""Nested normal-memory density on frozen held folds; no training or B inference."""
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
from scripts import v367_view_memory_validation as v367
from scripts.v180_dinomaly3_dinov3_csig import _extract_patch_tokens
from scripts.v348_clean_memory_core import fuse_memory, normal_quantile_scale, resize_raw_map
from scripts.v350_scaled_evidence import ScaledEvidenceMetrics
from scripts.v351_source_calibration import ResidentCosineMemory
from scripts.v368_memory_density_core import VARIANTS, nested_distances, sampling_indices

trainer, prior, ROOT = baseline.trainer, baseline.prior, baseline.ROOT
BASELINE_SHAS = v367.BASELINE_SHAS
_input_audit, _verify_hashes = v367._input_audit, v367._verify_hashes
query_stream_sha256 = v367.query_stream_sha256


def _loader(rows, size):
    import torch
    return torch.utils.data.DataLoader(trainer.RealIADSupervisedDataset(rows, size),
        batch_size=8, shuffle=False, num_workers=prior.DATA_LOADER_WORKERS, pin_memory=True)


def _check_batch(batch, rows):
    if (not 1 <= len(rows) == len(batch[0]) <= 8
            or list(batch[3]) != [r['category'] for r in rows]
            or list(batch[4]) != [r['group'] for r in rows]
            or batch[2].tolist() != [int(r['label']) for r in rows]):
        raise ValueError('Paired loader identities changed')


def _bank(model, rows):
    import torch
    v367.view_feature_indices(rows)
    pieces = {'sparse': [], 'additional': []}
    offset = 0
    with torch.inference_mode():
        for batch in _loader(rows, 560):
            current = rows[offset:offset + len(batch[0])]
            _check_batch(batch, current)
            tokens, grid = _extract_patch_tokens(model.encoder,
                batch[0].to(next(model.parameters()).device), [3, 7, 11])
            if grid != (35, 35) or tokens.shape[:2] != (len(current), 1225):
                raise ValueError('Expected frozen560 bank patch grid')
            for i, row in enumerate(current):
                member = f"aux/{row['group']}/{baseline.canonical_view(row['view'])}.png"
                for name, indices in sampling_indices(member, int(tokens.shape[1])).items():
                    # Keep V360's gather -> float32 CPU -> concatenate -> float16 order.
                    pieces[name].append(tokens[i, indices].float().cpu().numpy())
            offset += len(current)
    if offset != 80:
        raise ValueError('Incomplete80-view normal bank stream')
    return {name: np.concatenate(values, axis=0).astype(np.float16) for name, values in pieces.items()}


class NestedMemory:
    def __init__(self, sparse, additional, bank_rows, *, device):
        v367.view_feature_indices(bank_rows)
        sparse, additional = np.asarray(sparse), np.asarray(additional)
        if (sparse.ndim != 2 or additional.ndim != 2 or sparse.shape[0] != 4000
                or additional.shape != (12000, sparse.shape[1])):
            raise ValueError('Expected aligned4000 sparse and12000 additional features')
        self.category = bank_rows[0]['category']
        self.sparse = ResidentCosineMemory(sparse, device=device, chunk_size=8192)
        self.additional = ResidentCosineMemory(additional, device=device, chunk_size=8192)

    def distances(self, tokens, rows):
        if tokens.ndim != 3 or not 1 <= len(rows) == len(tokens) <= 8 or tokens.shape[1] == 0:
            raise ValueError('Expected aligned query rows in batches of at most8')
        if any(row['category'] != self.category for row in rows):
            raise ValueError('Query category differs from category memory')
        for row in rows:
            baseline.canonical_view(row['view'])
        # Never reshape or replace the original4000-column ResidentCosineMemory GEMM.
        return nested_distances(self.sparse.distances(tokens), self.additional.distances(tokens))


def _maps(model, rows, resident):
    import torch
    refs = np.empty((len(rows), 448, 448), np.float32)
    variants = {name: np.empty_like(refs) for name in VARIANTS}
    masks, labels, offset = [], [], 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for base, high in zip(_loader(rows, 448), _loader(rows, 560), strict=True):
            current = rows[offset:offset + len(base[0])]
            _check_batch(base, current)
            _check_batch(high, current)
            encoded, decoded = model(base[0].to(device))
            ref = trainer.differentiable_anomaly_map(encoded, decoded, 448).float().cpu().numpy()[:, 0]
            tokens, grid = _extract_patch_tokens(model.encoder, high[0].to(device), [3, 7, 11])
            if grid != (35, 35) or tokens.shape[:2] != (len(current), 1225):
                raise ValueError('Expected frozen560 query patch grid')
            raw = resident.distances(tokens, current)
            refs[offset:offset + len(current)] = ref
            for name in VARIANTS:
                variants[name][offset:offset + len(current)] = np.stack([
                    resize_raw_map(values.reshape(grid)) for values in raw[name]])
            masks.extend(base[1][:, 0].cpu().numpy() > .5)
            labels.extend(base[2].tolist())
            offset += len(current)
            del encoded, decoded, tokens
    if not offset or offset != len(rows):
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
    for category, item in result['sparse']['metrics']['per_category'].items():
        if item['image'] != result['dense']['metrics']['per_category'][category]['image']:
            raise ValueError('Reference classification changed between policies')
    return result


def baseline_replay(frozen, actual):
    mapped = {**actual, 'calibration': {'reference_q99': actual['calibration']['reference_q99'],
              'memory_q99': {'all_view': actual['calibration']['memory_q99']['sparse']}},
              'metrics': {'all_view': actual['metrics']['sparse']},
              'reserved_normal': {'all_view': actual['reserved_normal']['sparse']}}
    return v367.baseline_replay(frozen, mapped)


def _code_hashes():
    names = ('v368_memory_density_validation.py', 'v368_memory_density_core.py',
             'v367_view_memory_validation.py', 'v367_view_memory_core.py',
             'v339_verify_clean_metrics.py', 'v342_clean_longtrain.py', 'v344_longrun_submission.py')
    return {**baseline._code_hashes(), **{str(Path(__file__).with_name(name)):
            trainer.sha256_file(Path(__file__).with_name(name)) for name in names}}


def _smoke(model, calibration_rows, sparse, resident):
    rows = calibration_rows[:8]
    if len(rows) != 8 or any(row['label'] != 0 or row.get('mask_path') for row in rows):
        raise ValueError('Smoke requires eight normal calibration views')
    refs, maps, masks, labels = _maps(model, rows, resident)
    old_refs, old_maps, old_masks, old_labels = prior._maps(model, rows, sparse, 560)
    if (refs.tobytes() != old_refs.tobytes() or maps['sparse'].tobytes() != old_maps.tobytes()
            or not np.array_equal(masks, old_masks) or labels != old_labels):
        raise ValueError('Smoke legacy reference/distance identity failed')
    if np.any(maps['dense'] > maps['sparse']):
        raise ValueError('Smoke nested distance relation failed')
    return {'calibration_views': 8, 'legacy_reference_identical': True, 'legacy_distance_identical': True,
            'nested_distance_relation': True, 'query_raw_stream_sha256': query_stream_sha256(refs, maps['sparse'])}


def run(args):
    import torch
    smoke = bool(getattr(args, 'smoke', False))
    output = (Path(args.output_root) / f"fold{args.fold}{'_smoke' if smoke else ''}").resolve()
    if args.fold not in range(4) or output.exists() or not output.is_relative_to((ROOT / 'outputs').resolve()):
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
    prior.validate_seen_report(previous)
    if not held or set(held) != set(frozen['details']) or set(held) != set(old_fold['details']):
        raise ValueError('Eligible held categories changed')
    sources.update(bundle['selection_record_hashes'])
    sources.update({selected['checkpoint']: selected['checkpoint_sha256'], str(prior.GATE): bundle['source_gate_sha256']})
    codes = _code_hashes()
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
    for category in held[:1] if smoke else held:
        old = frozen['details'][category]
        split = old_fold['details'][category]['split']
        if split != old['split']:
            raise ValueError('Frozen source partition changed')
        parts = baseline.category_partitions(rows, category, split)
        audit, pixels = _input_audit(parts, approved, old['input_file_hashes'])
        banks = _bank(model, parts['bank'])
        memory_sha = hashlib.sha256(banks['sparse'].tobytes()).hexdigest()
        if memory_sha != old['memory_sha256']:
            raise ValueError('Frozen baseline memory replay mismatch')
        result = {'category': category, 'split': split, 'duplicate_audit': audit,
                  'input_file_hashes': old['input_file_hashes'], 'memory_sha256': memory_sha,
                  'additional_memory_sha256': hashlib.sha256(banks['additional'].tobytes()).hexdigest()}
        resident = NestedMemory(**banks, bank_rows=parts['bank'], device=next(model.parameters()).device)
        if smoke:
            result.update(_smoke(model, parts['calibration'], banks['sparse'], resident))
            del banks, resident
        else:
            del banks
            ref, maps, masks, labels = _maps(model, parts['calibration'], resident)
            if any(labels) or any(np.asarray(mask).any() for mask in masks):
                raise ValueError('Calibration images must remain normal')
            rq, mq = prior.source_q99(ref), {name: prior.source_q99(maps[name]) for name in VARIANTS}
            if (abs(rq - old['calibration']['category_ref_q99']) > 1e-6
                    or abs(mq['sparse'] - old['calibration']['memory_q99']) > 1e-6):
                raise ValueError('Frozen baseline calibration replay drift')
            del ref, maps, masks
            ref, maps, masks, _ = _maps(model, parts['evaluation'], resident)
            metrics = score_policies(parts['evaluation'], ref, maps, masks, rq, mq)
            query_sha = query_stream_sha256(ref, maps['sparse'])
            del ref, maps, masks
            ref, maps, masks, labels = _maps(model, parts['normal_audit'], resident)
            if any(labels) or any(np.asarray(mask).any() for mask in masks):
                raise ValueError('Reserved audit images must remain normal')
            normal = {name: baseline.normal_audit_metrics(np.stack([
                fuse_memory(r, mem, scale=normal_quantile_scale(rq, mq[name]))
                for r, mem in zip(ref, maps[name], strict=True)]), reference_q99=rq) for name in VARIANTS}
            del ref, maps, masks, resident
            result.update(metrics=metrics, reserved_normal=normal,
                calibration={'reference_q99': rq, 'memory_q99': mq}, query_raw_stream_sha256=query_sha)
            result['frozenbaseline_replaydrift'] = baseline_replay(old, result)
        repeated, after_pixels = _input_audit(parts, approved, old['input_file_hashes'])
        if repeated != audit or after_pixels != pixels:
            raise ValueError('Decoded auxiliary partitions changed')
        with (output / f'{category}.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        results[category] = result
        print(f'fold{args.fold}: {category} complete', flush=True)
    _verify_hashes(sources)
    _verify_hashes(codes)
    if baseline.longtrain.runtime_source_hashes(deps, model) != runtime:
        raise ValueError('Model runtime changed')
    result = {'status': 'smoke_ok' if smoke else 'ok', 'version': 'v368_memory_density_validation_v1',
              'fold': args.fold, 'details': results, 'variants': list(VARIANTS), 'input_size': 560, 'batch_size': 8,
              'memory_features': {'sparse': 4000, 'additional': 12000, 'dense': 16000},
              'source_hashes': sources, 'code_hashes': codes, 'runtime_source_hashes': runtime,
              'elapsed_seconds': time.monotonic() - started, 'test_b_labels_accessed': False,
              'test_b_zip_accessed': False, 'new_training': False, 'quality_status': 'development_comparison_only',
              'scope': 'Seen-normal-reference held-fold auxiliary simulation, not independent ensemble or zero-shot/B quality'}
    if not smoke:
        result['frozenbaseline_replaydrift'] = max(max(r['frozenbaseline_replaydrift'].values()) for r in results.values())
    with (output / 'report.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=range(4), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Check eight calibration views of the first held category')
    run(parser.parse_args())
