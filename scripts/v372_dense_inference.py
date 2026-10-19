"""Bounded V362 density candidate: frozen numerator, normal-only calibration."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
from scripts import v349_clean_memory_bank as v349
from scripts import v365_ensemble_category_calibration as v365
from scripts import v366_calibrated_seen_submission as v366
from scripts.v283_supervised_dinomaly_longtrain import sha256_file
from scripts.v348_clean_memory_core import normal_quantile_scale, resize_raw_map
from scripts.v351_source_calibration import ResidentCosineMemory
from scripts.v355_seen_category_calibration import category_quantile, validate_category_calibration

ROOT = v366.ROOT
REFERENCE_Q99 = 0.04275937380269168
V368 = ROOT / 'outputs/v368_nested_density_20260912/summary.json'
V368_SHA = '3e095ce644ec2d5cec9c4e4916d8b4de81c4d4132fa39865f3645ce09544b3ef'
V369 = ROOT / 'outputs/v369_visa_confirm_20260913/manifest.json'
V369_SHA = '7e457f42cfd07406d4b26b8a9963ecb9d8420c590f27d537bb084ddf8063eb53'


def extra_patch_indices(member, token_count=1225):
    original = v349.member_patch_indices(member, token_count)
    if token_count < 200:
        raise ValueError('Insufficient normal patch positions')
    seed = int.from_bytes(hashlib.sha256(('36801:' + member).encode()).digest()[:8], 'little')
    remaining = np.setdiff1d(np.arange(token_count), original)
    return np.sort(np.random.default_rng(seed).choice(remaining, 150, replace=False))


def normal_indices(records, partition, kind):
    count = {'bank': 80, 'calibration': 20}[kind]
    allowed = set(partition[kind + '_groups'])
    indices = [i for i, record in enumerate(records) if record.group in allowed]
    expected = [f'Train/{g}/{v}.png' for g in partition[kind + '_groups'] for v in range(5)]
    if (len(indices) != 50 * count or len(set(expected)) != len(expected)
            or expected != sorted(expected) or [records[i].member for i in indices] != expected
            or Counter(records[i].category for i in indices) != dict.fromkeys(partition['categories'], count)
            or any(r.member != f'Train/{r.category}/{r.sample}/{r.view}.png'
                   or r.group != f'{r.category}/{r.sample}' for r in (records[i] for i in indices))):
        raise ValueError('Global normal partition/order/calibration shape mismatch')
    return indices


def category_batches(records, indices, category):
    # Slice the original filtered global population BEFORE selecting a category.
    for offset in range(0, len(indices), 8):
        batch = indices[offset:offset + 8]
        positions = [p for p, row in enumerate(batch) if records[row].category == category]
        if positions:
            yield batch, positions


def build_extra(category, records, indices, sparse, extract):
    if sparse.shape != (4000, 2304) or sparse.dtype != np.float16:
        raise ValueError('Expected original4000 float16 sparse rows')
    extra = np.empty((12000, 2304), np.float16)
    members, old_indices, new_indices, batches = [], [], [], []
    for batch, positions in category_batches(records, indices, category):
        tokens = extract(batch)
        if tuple(tokens.shape) != (8, 1225, 2304) or positions != list(range(8)):
            raise ValueError('Original global bank batch8 geometry changed')
        batches.append(batch)
        for p in positions:
            member = records[batch[p]].member
            old, new = v349.member_patch_indices(member, 1225), extra_patch_indices(member)
            actual = tokens[p, old].float().cpu().numpy().astype(np.float16)
            start = len(members)
            if actual.tobytes() != sparse[start * 50:(start + 1) * 50].tobytes():
                raise ValueError(f'Original sparse bank replay differs: {member}')
            extra[start * 150:(start + 1) * 150] = tokens[p, new].float().cpu().numpy().astype(np.float16)
            members.append(member)
            old_indices.extend(old)
            new_indices.extend(new)
    if len(members) != 80 or members != sorted(set(members)) or not np.isfinite(extra).all():
        raise ValueError('Incomplete or invalid normal extra bank')
    return extra, dict(bank_members=members, sparse_sha256=hashlib.sha256(sparse.tobytes()).hexdigest(),
        extra_sha256=hashlib.sha256(extra.tobytes()).hexdigest(), sparse_replay='exact_all4000_float16_rows',
        bank_trace={'global_batch_size': 8, 'global_dataset_batches': batches, 'grid': [35, 35],
                    'sparse_rows': 4000, 'extra_rows': 12000, 'feature_layers': [3, 7, 11],
                    'original_seed_prefix': '34801:', 'extra_seed_prefix': '36801:',
                    'sparse_patch_indices_sha256': hashlib.sha256(np.asarray(old_indices, '<u2').tobytes()).hexdigest(),
                    'extra_patch_indices_sha256': hashlib.sha256(np.asarray(new_indices, '<u2').tobytes()).hexdigest()})


def memory_maps(tokens, sparse, extra):
    old_maps, dense_maps = [], []
    for p in range(len(tokens)):
        old = sparse.distances(tokens[p:p + 1])[0]
        additional = extra.distances(tokens[p:p + 1])[0]
        old_maps.append(resize_raw_map(old.reshape(35, 35)))
        dense_maps.append(resize_raw_map(np.minimum(old, additional).reshape(35, 35)))
    return np.stack(old_maps), np.stack(dense_maps)


def calibration_record(members, passes, old_q99):
    v365.checked_members(members)
    summaries = []
    for sparse, dense in passes:
        if (len(summaries) >= 2 or any(np.shape(a) != (20, 448, 448) for a in (sparse, dense))
                or np.any(dense > sparse)):
            raise ValueError('Invalid calibration pass count, global shape or dense minimum')
        q = [category_quantile(a) for a in (sparse, dense)]
        if not np.isfinite(old_q99) or abs(q[0] - old_q99) > 1e-6:
            raise ValueError('Original V355 sparse normal q99 replay differs')
        summaries.append({'sparse_q99': q[0], 'dense_q99': q[1],
                          'hashes': {k: v365.response_digest(a) for k, a in zip(('sparse', 'dense'), (sparse, dense))}})
    if len(summaries) != 2 or summaries[0] != summaries[1]:
        raise ValueError('Exact two-pass normal calibration replay differs')
    first = summaries[0]
    return dict(calibration_members=members, reference_q99=REFERENCE_Q99, old_q99=float(old_q99),
        sparse_q99=first['sparse_q99'], dense_q99=first['dense_q99'],
        old_scale=normal_quantile_scale(REFERENCE_Q99, old_q99),
        new_scale=normal_quantile_scale(REFERENCE_Q99, first['dense_q99']),
        calibration_hashes=[s['hashes'] for s in summaries], calibration_replay_exact=True,
        published_memory_q99_abs_drift=abs(first['sparse_q99'] - old_q99), global_calibration_batch_size=8)


def calibrate(category, records, indices, sparse, extra, extract, old_q99):
    batches = list(category_batches(records, indices, category))
    members = [records[rows[p]].member for rows, positions in batches for p in positions]
    def passes():
        for _ in range(2):
            old, dense = [], []
            for rows, positions in batches:
                pair = memory_maps(extract(rows)[positions], sparse, extra)
                old.extend(pair[0])
                dense.extend(pair[1])
            yield np.stack(old), np.stack(dense)
    return calibration_record(members, passes(), old_q99)


def category_responses(records, category, infer, scales):
    for indices, positions in v366.seen_batches(records, {category}):
        references, memory = infer(indices, positions)
        for p in positions:
            row = records[indices[p]]
            yield dict(group=row.group, view=row.view, reference=references[p], sparse=memory[p][0],
                       dense=memory[p][1], old_scale=scales['old_scale'], new_scale=scales['new_scale'])


def evidence_sources():
    density = v366.read_verified_json(V368, V368_SHA)
    visa = v366.read_verified_json(V369, V369_SHA)
    if (density.get('status') != 'ok' or density.get('screen', {}).get('eligible') is not True
            or visa.get('status') != 'ok' or visa.get('screen', {}).get('passed') is not True):
        raise ValueError('Both pinned V368/V369 density gates must pass')
    return visa, v366.merge_sources({str(V368): V368_SHA, str(V369): V369_SHA}, density['source_reports_sha256'])


def validate_frozen(producer, old, bank_report, selected, partition, environment):
    validate_category_calibration(old, partition['categories'])
    if (producer['reference_calibration_q99'] != REFERENCE_Q99
            or producer['category_calibration_sha256'] != v366.LEGACY_CALIBRATION_SHA
            or producer['selected_checkpoints'] != selected or len(selected) != 4
            or producer['seen_categories'] != partition['categories'] or len(partition['categories']) != 50
            or bank_report['metadata']['partition'] != partition
            or any(r.get('preprocess_environment') != environment for r in (producer, old, bank_report['metadata']))
            or producer['bank_sha256'] != old['bank_sha256'] or old['bank_sha256'] != bank_report['bank_sha256']
            or producer['bank_manifest_sha256'] != old['bank_manifest_sha256']
            or producer['input_size'] != 560 or bank_report['metadata']['input_size'] != 560
            or bank_report['trace']['grid'] != [35, 35]
            or old.get('input_size') != 672 or 'patch_grid' in old
            or old['code_hashes'].get('v355_seen_category_calibration.py') != v366.LEGACY_V355_SHA
            or old['bank_sha256'] != v366.LEGACY_BANK_SHA):
        raise ValueError('Frozen V362/V355 normal calibration, model or bank contract changed')


def source_snapshot(sources):
    actual = {p: sha256_file(Path(p)) for p in sources}
    if actual != sources:
        raise ValueError('Source changed during V372 inference')
    return actual


def preflight():
    from scripts.build_b_submission import _test_index
    from scripts.v344_longrun_submission import load_verified_bundle, inference_hashes
    from scripts.v356_seen_memory_submission import GATE, _load_category_memory

    producer = v366.validate_producer(v366.read_verified_json(v366.PRODUCER, v366.PRODUCER_SHA))
    old = v366.read_verified_json(v366.OLD_CALIBRATION, v366.LEGACY_CALIBRATION_SHA)
    fixed = {str(v366.PRODUCER): v366.PRODUCER_SHA, str(v366.BASELINE): v366.BASELINE_SHA,
             str(ROOT / 'Test_B.zip'): producer['test_zip_sha256'],
             str(v366.OLD_CALIBRATION): v366.LEGACY_CALIBRATION_SHA,
             str(v366.BANK_ROOT / 'memory.npz'): producer['bank_sha256'],
             str(v366.BANK_ROOT / 'manifest.json'): producer['bank_manifest_sha256']}
    v366.unchanged(fixed)
    visa, evidence = evidence_sources()
    train, weight, _, partition = v349.source_inputs()
    bundle = load_verified_bundle(ROOT / 'outputs/v343_verified_longrun_20260911', GATE)
    selected = [r['selected'] for r in bundle['records']]
    bank, rows, bank_report = _load_category_memory(v366.BANK_ROOT)
    environment = v349.preprocessing_environment_identity()
    validate_frozen(producer, old, bank_report, selected, partition, environment)
    if (bundle['selection_record_hashes'] != producer['selection_record_hashes']
            or visa['selected_checkpoints'] != selected or set(rows) != set(partition['categories'])):
        raise ValueError('Frozen selection/category evidence changed')
    code = {str(PROJECT / 'scripts' / n): d for n, d in inference_hashes().items()}
    code.update({str(Path(m.__file__)): sha256_file(Path(m.__file__)) for m in (v349, v365)})
    current_v355 = PROJECT / 'scripts' / 'v355_seen_category_calibration.py'
    code[str(current_v355)] = sha256_file(current_v355)
    required = v366.merge_sources(code, {str(v366.PRODUCER): v366.PRODUCER_SHA, str(GATE): bundle['source_gate_sha256']},
        bundle['selection_record_hashes'], {r['checkpoint']: r['checkpoint_sha256'] for r in selected},
        *[r['long_config']['runtime_source_hashes'] for r in bundle['records']])
    v366.require_sources(visa['source_hashes'], required)
    configs = {str(Path(r['long_root']) / f"fold{r['fold_id']}" / 'immutable_config.json'): r['config_sha256']
               for r in bundle['records']}
    sources = v366.merge_sources(fixed, evidence, required, configs, v366.historical_calibration_sources(old),
        old['runtime_source_hashes'], bank_report['metadata']['runtime_source_hashes'],
        {str(PROJECT / 'scripts' / n): d for n, d in producer['code_hashes'].items()},
        {str(train): v349.TRAIN_SHA, str(weight): v349.WEIGHT_SHA,
         str(ROOT / 'outputs/v348_memory_smoke_reviewed_20260911/source_partition.json'): v349.PARTITION_SHA},
        {str(PROJECT / 'scripts' / n): sha256_file(PROJECT / 'scripts' / n) for n in
         ('v372_dense_inference.py', 'v372_dense_export.py', 'v366_calibrated_seen_submission.py',
          'v363_route_alignment.py', 'v357_v356_release.py', 'v355_seen_category_calibration.py')})
    # The historical V355 hash belongs to its preserved copy, never today's path.
    with np.load(v366.BANK_ROOT / 'memory.npz', allow_pickle=False) as data:
        members = data['members'].astype(str).tolist()
    provenance = dict(reference_q99=REFERENCE_Q99, selected_checkpoints=selected, new_training=False,
        reference_input_size=448, memory_input_size=560, global_dataset_batch_size=4,
        global_normal_batch_size=8, feature_layers=[3, 7, 11], fusion_weights=[.9, .1],
        calibration_producer='v372_normal_only_original_V355_global_batch8',
        legacy_metadata_exception={'sha256': v366.LEGACY_CALIBRATION_SHA, 'recorded_input_size': 672,
                                   'verified_bank_input_size': 560, 'verified_bank_grid': [35, 35]},
        original_partition_retained=True, calibration_is_independent_quality_evidence=False,
        known_duplicate_group_retained='toy_tire/S0020', source_hashes_before=source_snapshot(sources),
        preprocess_environment=environment, test_b_labels_accessed=False)
    return dict(producer=producer, old=old, train=train, partition=partition, bank=bank, category_rows=rows,
                bank_members=members, bundle=bundle, sources=sources, provenance=provenance, environment=environment,
                index=v366.make_index(_test_index(ROOT / 'Test_B.zip', 'Test_B'), producer['seen_categories']))


def check_resources():
    import torch
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('1GiB free disk required')
    if torch.cuda.device_count() < 4 or any(torch.cuda.mem_get_info(i)[0] < 8 * 2**30 for i in range(4)):
        raise RuntimeError('Four GPUs with8GiB free required')
    torch.set_num_threads(4)
    torch.manual_seed(34801)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def final_check(context):
    import torch
    v349.source_inputs()
    environment = context['environment']
    if (v349.preprocessing_environment_identity() != environment
            or any(torch.cuda.get_device_name(i) != environment['cuda_device']
                   or list(torch.cuda.get_device_capability(i)) != environment['cuda_capability'] for i in range(4))):
        raise ValueError('Frozen inference environment changed')
    context['provenance']['source_hashes_after'] = source_snapshot(context['sources'])


def check_models(models, bundle):
    import torch
    from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    from scripts.v342_clean_longtrain import runtime_source_hashes
    deps = adapter._import_official_modules()
    if len(models) != 4:
        raise ValueError('Expected four frozen models')
    for i, model in enumerate(models):
        if (any(m.training for m in model.modules()) or any(p.requires_grad or p.dtype != torch.float32
                or p.device != torch.device(f'cuda:{i}') for p in model.parameters())
                or runtime_source_hashes(deps, model) != bundle['records'][i]['long_config']['runtime_source_hashes']):
            raise ValueError('Frozen FP32 model/runtime configuration changed')


def gpu_responses(context, category_records, *, smoke=False):
    import torch
    from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    from scripts.v180_dinomaly3_dinov3_csig import _extract_patch_tokens
    from scripts.v283_supervised_dinomaly_longtrain import differentiable_anomaly_map

    models = v365.load_models(context['bundle'])
    check_models(models, context['bundle'])
    datasets = []
    def dataset(path, root, size):
        result = adapter.CsigZipDataset(path, root, adapter._build_transform(size, size), include_metadata=True)
        datasets.append(result)
        return result
    def images_at(data, indices):
        images = []
        for row in indices:
            image, category, sample, view = data[row]
            r = data.records[row]
            if (category, sample, view) != (r.category, r.sample, r.view):
                raise ValueError('Loaded image metadata changed')
            images.append(image)
        return torch.stack(images)
    def extract(data, indices):
        with torch.inference_mode():
            tokens, grid = _extract_patch_tokens(models[0].encoder, images_at(data, indices).to('cuda:0'), [3, 7, 11])
        if grid != (35, 35) or tuple(tokens.shape) != (len(indices), 1225, 2304) or tokens.dtype != torch.float32:
            raise ValueError('Expected FP32 encoder0 layers3/7/11 at560')
        return tokens
    def worker(fold, images):
        torch.cuda.set_device(fold)
        with torch.inference_mode():
            enc, dec = models[fold](images.to(f'cuda:{fold}', non_blocking=True))
            return differentiable_anomaly_map(enc, dec, 448).float().cpu().numpy()[:, 0]
    try:
        normal = dataset(context['train'], 'Train', 560)
        bank_indices = normal_indices(normal.records, context['partition'], 'bank')
        cal_indices = normal_indices(normal.records, context['partition'], 'calibration')
        if [normal.records[i].member for i in bank_indices] != context['bank_members']:
            raise ValueError('Original NPZ bank member order changed')
        reference = high = None
        with ThreadPoolExecutor(max_workers=4) as pool:
            for category in sorted(context['index']['seen']):
                sparse_bank = context['bank'][context['category_rows'][category]]
                extra_bank, trace = build_extra(category, normal.records, bank_indices, sparse_bank, lambda rows: extract(normal, rows))
                sparse, extra = [ResidentCosineMemory(a, device='cuda:0', chunk_size=8192) for a in (sparse_bank, extra_bank)]
                record = calibrate(category, normal.records, cal_indices, sparse, extra,
                                   lambda rows: extract(normal, rows), context['old']['categories_detail'][category]['q99'])
                category_records[category] = {**trace, **record}
                print(f'normal_category={category} sparse_replay=exact calibration_replay=exact', flush=True)
                if smoke:
                    return
                if reference is None:
                    reference, high = [dataset(ROOT / 'Test_B.zip', 'Test_B', size) for size in (448, 560)]
                    v366.validate_records(reference.records, high.records, context['index'])
                def infer(indices, positions):
                    images = images_at(reference, indices)
                    pending = [pool.submit(worker, fold, images) for fold in range(4)]
                    refs = v366.combine_raw([f.result() for f in pending])
                    old, dense = memory_maps(extract(high, indices)[positions], sparse, extra)
                    return refs, {p: (old[i], dense[i]) for i, p in enumerate(positions)}
                yield from category_responses(reference.records, category, infer, record)
                del sparse, extra, sparse_bank, extra_bank
    finally:
        for data in datasets:
            if data._archive is not None:
                data._archive.close()
        check_models(models, context['bundle'])


def run(args):
    output = Path(args.output_root).resolve()
    if not output.is_relative_to((ROOT / 'outputs').resolve()) or output.exists():
        raise ValueError('A new output directory under outputs is required')
    check_resources()
    context = preflight()
    output.mkdir(parents=True)
    records = {}
    responses = gpu_responses(context, records, smoke=args.smoke)
    try:
        if args.smoke:
            for _ in responses:
                raise ValueError('Smoke must not yield any Test_B response')
            final_check(context)
            if list(records) != sorted(context['index']['seen'])[:1]:
                raise ValueError('Smoke requires exactly the first normal category')
            report = dict(status='smoke_ok', sources=context['sources'], category_records=records,
                          selected_checkpoints=context['producer']['selected_checkpoints'], new_training=False,
                          test_b_images_accessed=False, provenance=context['provenance'],
                          report_path=str(output / 'smoke_manifest.json'))
            with (output / 'smoke_manifest.json').open('x', encoding='utf-8') as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
        else:
            from scripts.v372_dense_export import build_candidate
            report = build_candidate(v366.BASELINE, output / 'candidate_unscored.zip', context['index'], responses,
                context['sources'], provenance=context['provenance'], category_records=records,
                final_check=lambda: final_check(context))
    finally:
        responses.close()
    print(json.dumps({'status': report['status'], 'output': str(output / ('smoke_manifest.json' if args.smoke else 'candidate_unscored.zip'))}), flush=True)
    return report


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    return parser


if __name__ == '__main__':
    run(argument_parser().parse_args())
