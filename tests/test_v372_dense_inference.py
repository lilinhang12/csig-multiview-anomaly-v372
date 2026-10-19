"""CPU contracts for the bounded V372 producer; no production data access."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    name = 'scripts.v372_dense_inference'
    assert importlib.util.find_spec(name) is not None, 'V372 inference is not implemented'
    return importlib.import_module(name)


@pytest.fixture(scope='module')
def normal():
    from scripts.v348_clean_memory_core import partition_train_groups
    groups = {f'part{c:03d}/S{g:04d}': tuple('01234') for c in range(50) for g in range(1, 21)}
    partition = partition_train_groups(groups)
    records = [SimpleNamespace(category=g.split('/')[0], sample=g.split('/')[1], group=g,
                               view=v, member=f'Train/{g}/{v}.png') for g in groups for v in groups[g]]
    return records, partition


def test_extra_sampling_preserves_real_v349_prefix_and_uses_normal_member_seed():
    from scripts.v349_clean_memory_bank import member_patch_indices
    mod = module()
    for member in ('Train/3_adapter/S0001/0.png', 'Train/DVD_switch/S0019/4.png'):
        original = member_patch_indices(member, 1225)
        seed = int.from_bytes(hashlib.sha256(('36801:' + member).encode()).digest()[:8], 'little')
        expected = np.sort(np.random.default_rng(seed).choice(np.setdiff1d(np.arange(1225), original), 150, replace=False))
        extra = mod.extra_patch_indices(member, 1225)
        np.testing.assert_array_equal(extra, expected)
        dense = np.concatenate((original, extra))
        np.testing.assert_array_equal(dense[:50], original)
        assert len(set(dense)) == 200 and np.all(np.diff(extra) > 0)
        np.testing.assert_array_equal(extra, mod.extra_patch_indices(member, 1225))


@pytest.mark.parametrize('member,count', [('Test_B/part/S0001/0.png', 1225),
                                         ('Train/../S0001/0.png', 1225), ('Train/part/S0001/0.png', 199)])
def test_extra_sampling_rejects_non_normal_or_insufficient_grid(member, count):
    with pytest.raises(ValueError):
        module().extra_patch_indices(member, count)


def test_original_global_bank_and_calibration_boundaries(normal):
    mod = module()
    records, partition = normal
    bank = mod.normal_indices(records, partition, 'bank')
    cal = mod.normal_indices(records, partition, 'calibration')
    assert len(bank) == 4000 and len(cal) == 1000 and not set(bank) & set(cal)
    assert list(mod.category_batches(records, bank, 'part001')) == [(bank[i:i + 8], list(range(8))) for i in range(80, 160, 8)]
    batches = list(mod.category_batches(records, cal, 'part001'))
    assert batches == [(cal[16:24], [4, 5, 6, 7]), (cal[24:32], list(range(8))), (cal[32:40], list(range(8)))]
    assert records[batches[0][0][0]].category == 'part000'
    assert [records[rows[p]].member for rows, positions in batches for p in positions] == [records[i].member for i in cal[20:40]]


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'order', 'member', 'category', 'shape'])
def test_global_normal_population_fails_closed(normal, bad):
    records, partition = copy.deepcopy(normal)
    cal_group = partition['calibration_groups'][0]
    i = next(i for i, r in enumerate(records) if r.group == cal_group)
    if bad == 'missing': records.pop(i)
    elif bad == 'duplicate': records[i + 1] = records[i]
    elif bad == 'order': records[i:i + 2] = reversed(records[i:i + 2])
    elif bad == 'member': records[i].member = records[i].member.replace('Train/', 'Test_B/')
    elif bad == 'category': records[i].category = 'wrong'
    elif bad == 'shape': partition['calibration_groups'].pop()
    with pytest.raises(ValueError, match='normal|partition|calibration'):
        module().normal_indices(records, partition, 'calibration')


def test_build_extra_replays_every_sparse_row_and_keeps_bank_batch8(normal):
    import torch
    from scripts.v349_clean_memory_bank import member_patch_indices
    mod = module()
    records, partition = normal
    indices = mod.normal_indices(records, partition, 'bank')
    own = indices[80:160]
    tokens = torch.arange(1, 1226, dtype=torch.float32).reshape(1225, 1).expand(-1, 2304) / 60000
    sparse = np.concatenate([tokens[member_patch_indices(records[i].member, 1225)].numpy().astype(np.float16) for i in own])
    calls = []
    def extract(rows):
        calls.append(rows)
        return tokens.unsqueeze(0).expand(len(rows), -1, -1)
    extra, trace = mod.build_extra('part001', records, indices, sparse, extract)
    assert calls == [indices[i:i + 8] for i in range(80, 160, 8)]
    assert extra.shape == (12000, 2304) and extra.dtype == np.float16
    np.testing.assert_array_equal(extra[:150], tokens[mod.extra_patch_indices(records[own[0]].member, 1225)].numpy().astype(np.float16))
    assert trace['sparse_replay'] == 'exact_all4000_float16_rows'
    assert trace['sparse_sha256'] == hashlib.sha256(sparse.tobytes()).hexdigest()
    assert trace['extra_sha256'] == hashlib.sha256(extra.tobytes()).hexdigest()
    sparse[-1, -1] = np.float16(1)
    with pytest.raises(ValueError, match='sparse.*replay'):
        mod.build_extra('part001', records, indices, sparse, extract)


def test_dense_min_uses_unchanged_resident_sparse_matrix():
    import torch
    from scripts.v351_source_calibration import ResidentCosineMemory
    from scripts.v348_clean_memory_core import resize_raw_map
    mod = module()
    rng = np.random.default_rng(372)
    sparse = ResidentCosineMemory(rng.normal(size=(4000, 4)), device='cpu', chunk_size=8192)
    extra = ResidentCosineMemory(rng.normal(size=(12000, 4)), device='cpu', chunk_size=8192)
    query = torch.tensor(rng.normal(size=(1, 1225, 4)), dtype=torch.float32)
    original = sparse.distances(query)[0]
    additional = extra.distances(query)[0]
    old, dense = mod.memory_maps(query, sparse, extra)
    np.testing.assert_array_equal(old[0], resize_raw_map(original.reshape(35, 35)))
    np.testing.assert_array_equal(dense[0], resize_raw_map(np.minimum(original, additional).reshape(35, 35)))
    assert np.all(dense <= old)


@pytest.fixture
def cal_maps():
    members = [f'Train/part000/S{g:04d}/{v}.png' for g in range(1, 5) for v in range(5)]
    sparse = np.full((20, 448, 448), .25, np.float32)
    dense = np.full_like(sparse, .125)
    return members, sparse, dense


def test_calibration_exact_replay_and_frozen_numerator(cal_maps):
    mod = module()
    members, sparse, dense = cal_maps
    record = mod.calibration_record(members, [(sparse, dense), (sparse.copy(), dense.copy())], .2500001)
    assert record['reference_q99'] == 0.04275937380269168
    assert record['old_scale'] == mod.REFERENCE_Q99 / .2500001
    assert record['new_scale'] == mod.REFERENCE_Q99 / .125
    assert record['old_q99'] == .2500001 and record['sparse_q99'] == .25 and record['dense_q99'] == .125
    assert record['calibration_hashes'][0] == record['calibration_hashes'][1]
    assert len(record['calibration_hashes'][0]['dense']) == 64
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize('bad', ['pixel', 'q99', 'old_drift', 'shape', 'nan', 'negative', 'larger', 'one_pass', 'third_pass', 'members'])
def test_calibration_rejects_inexact_or_invalid_inputs(cal_maps, bad):
    mod = module()
    members, sparse, dense = cal_maps
    second = dense.copy()
    old = .25
    if bad == 'pixel': second[0, 0, 0] = .124
    elif bad == 'q99': second[:] = .124
    elif bad == 'old_drift': old += 2e-6
    elif bad == 'shape': sparse = sparse[:19]
    elif bad == 'nan': dense[0, 0, 0] = np.nan
    elif bad == 'negative': dense[0, 0, 0] = -1
    elif bad == 'larger': dense[0, 0, 0] = .3
    elif bad == 'members': members[0] = members[0].replace('Train/', 'Test_B/')
    passes = [(sparse, dense), (sparse, second)]
    if bad == 'one_pass': passes.pop()
    elif bad == 'third_pass': passes.append(passes[-1])
    with pytest.raises(ValueError):
        mod.calibration_record(members, passes, old)


def test_test_b_batches_include_neighbours_but_only_requested_category():
    mod = module()
    records = [SimpleNamespace(category=c, group=f'{c}/S{i // 5:04d}', view=str(i % 5))
               for c in ('a', 'b', 'c') for i in range(125)]
    calls = []
    def infer(indices, positions):
        calls.append((indices, positions))
        return np.arange(len(indices)), {p: (p + .1, p + .01) for p in positions}
    responses = list(mod.category_responses(records, 'b', infer, {'old_scale': 2., 'new_scale': 3.}))
    assert calls[0] == ([124, 125, 126, 127], [1, 2, 3])
    assert calls[-1] == ([248, 249, 250, 251], [0, 1])
    assert len(responses) == 125 and all(r['group'].startswith('b/') for r in responses)
    assert responses[0] == {'group': 'b/S0000', 'view': '0', 'reference': 1, 'sparse': 1.1,
                            'dense': 1.01, 'old_scale': 2., 'new_scale': 3.}


def test_gate_reports_are_both_pinned_and_passed(monkeypatch):
    mod = module()
    calls = []
    reports = [{'status': 'ok', 'screen': {'eligible': True}, 'source_reports_sha256': {}},
               {'status': 'ok', 'screen': {'passed': True}, 'source_hashes': {}}]
    def read(path, sha):
        calls.append((path, sha))
        return copy.deepcopy(reports[len(calls) - 1])
    monkeypatch.setattr(mod.v366, 'read_verified_json', read)
    _, sources = mod.evidence_sources()
    assert {sha for _, sha in calls} == {mod.V368_SHA, mod.V369_SHA}
    assert set(sources.values()) == {mod.V368_SHA, mod.V369_SHA}
    for index, key in ((0, 'eligible'), (1, 'passed')):
        calls.clear()
        reports[index]['screen'][key] = False
        with pytest.raises(ValueError, match='gate'):
            mod.evidence_sources()
        reports[index]['screen'][key] = True


def test_smoke_stops_after_first_normal_category_without_export(monkeypatch, tmp_path):
    mod = module()
    context = {'sources': {}, 'producer': {'selected_checkpoints': [{'fold_id': i} for i in range(4)]},
               'provenance': {}, 'index': {'groups': {}, 'seen': frozenset({'a', 'b'})}}
    events = []
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    (tmp_path / 'outputs').mkdir()
    monkeypatch.setattr(mod, 'check_resources', lambda: events.append('resources'))
    monkeypatch.setattr(mod, 'preflight', lambda: context)
    monkeypatch.setattr(mod, 'final_check', lambda ctx: events.append('checked'))
    def responses(ctx, records, *, smoke=False):
        assert smoke
        events.append('normal')
        records['a'] = {'old_scale': 2., 'new_scale': 3.}
        return
        yield  # Keep the production iterator contract.
    monkeypatch.setattr(mod, 'gpu_responses', responses)
    args = mod.argument_parser().parse_args(['--output-root', str(tmp_path / 'outputs' / 'new'), '--smoke'])
    report = mod.run(args)
    assert report['status'] == 'smoke_ok' and report['new_training'] is False
    assert len(report['selected_checkpoints']) == 4 and list(report['category_records']) == ['a']
    assert events == ['resources', 'normal', 'checked']
    assert sorted(p.name for p in args.output_root.iterdir()) == ['smoke_manifest.json']
    assert json.loads((args.output_root / 'smoke_manifest.json').read_text()) == report
    with pytest.raises(ValueError, match='new|output'):
        mod.run(args)


def test_final_source_change_prevents_smoke_manifest(monkeypatch, tmp_path):
    mod = module()
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    (tmp_path / 'outputs').mkdir()
    monkeypatch.setattr(mod, 'check_resources', lambda: None)
    monkeypatch.setattr(mod, 'preflight', lambda: {'sources': {}, 'producer': {}, 'provenance': {}})
    monkeypatch.setattr(mod, 'gpu_responses', lambda *a, **k: (_ for _ in ()))
    def changed(ctx):
        raise ValueError('Source changed')
    monkeypatch.setattr(mod, 'final_check', changed)
    args = SimpleNamespace(output_root=tmp_path / 'outputs' / 'new', smoke=True)
    with pytest.raises(ValueError, match='Source changed'):
        mod.run(args)
    assert not (args.output_root / 'smoke_manifest.json').exists()


def test_real_gpu_smoke_orchestration_never_constructs_test_b(monkeypatch, normal):
    from scripts import v182_dinomaly2_dinov3_csig_official as adapter
    mod = module()
    records, partition = normal
    events = []
    def dataset(path, root, transform, **kwargs):
        assert root == 'Train' and transform == (560, 560)
        events.append('Train560')
        return SimpleNamespace(records=records, _archive=SimpleNamespace(close=lambda: events.append('close')))
    monkeypatch.setattr(adapter, 'CsigZipDataset', dataset)
    monkeypatch.setattr(adapter, '_build_transform', lambda a, b: (a, b))
    monkeypatch.setattr(mod.v365, 'load_models', lambda _: [SimpleNamespace(encoder=None)] * 4)
    monkeypatch.setattr(mod, 'check_models', lambda *a: events.append('models_checked'))
    monkeypatch.setattr(mod, 'ResidentCosineMemory', lambda *a, **k: object())
    def extra(category, actual, indices, sparse, extract):
        assert category == 'part000' and actual is records and len(indices) == 4000
        events.append('extra')
        return np.zeros((1, 1), np.float16), {'sparse_replay': 'exact_all4000_float16_rows'}
    def calibration(category, actual, indices, sparse, extra, extract, old_q99):
        assert category == 'part000' and len(indices) == 1000 and old_q99 == .25
        events.append('cal20_twice')
        return {'old_scale': 2., 'new_scale': 3.}
    monkeypatch.setattr(mod, 'build_extra', extra)
    monkeypatch.setattr(mod, 'calibrate', calibration)
    context = dict(bundle={}, train=Path('Train.zip'), partition=partition,
                   index={'seen': frozenset(partition['categories'])}, bank=np.zeros((1, 1)),
                   bank_members=[records[i].member for i in mod.normal_indices(records, partition, 'bank')],
                   category_rows=dict.fromkeys(partition['categories'], [0]),
                   old={'categories_detail': {c: {'q99': .25} for c in partition['categories']}})
    category_records = {}
    assert list(mod.gpu_responses(context, category_records, smoke=True)) == []
    assert list(category_records) == ['part000']
    assert events == ['models_checked', 'Train560', 'extra', 'cal20_twice', 'close', 'models_checked']


def test_calibrate_extracts_complete_global_batches_twice(monkeypatch, normal):
    import torch
    mod = module()
    records, partition = normal
    indices = mod.normal_indices(records, partition, 'calibration')
    calls, queries = [], []
    def extract(rows):
        calls.append(rows)
        return torch.tensor(rows)
    def maps(tokens, *args):
        queries.extend(tokens.tolist())
        return np.full((len(tokens), 448, 448), .25, np.float32), np.full((len(tokens), 448, 448), .125, np.float32)
    monkeypatch.setattr(mod, 'memory_maps', maps)
    record = mod.calibrate('part001', records, indices, None, None, extract, .25)
    assert calls == [indices[16:24], indices[24:32], indices[32:40]] * 2
    assert queries == indices[20:40] * 2
    assert record['calibration_replay_exact'] is True


def test_frozen_legacy_metadata_is_narrowly_pinned(normal):
    mod = module()
    _, partition = normal
    env = {'cuda_device': 'P100'}
    selected = [{'fold_id': i} for i in range(4)]
    producer = dict(reference_calibration_q99=mod.REFERENCE_Q99, input_size=560,
        category_calibration_sha256=mod.v366.LEGACY_CALIBRATION_SHA, selected_checkpoints=selected,
        seen_categories=partition['categories'], bank_sha256=mod.v366.LEGACY_BANK_SHA,
        bank_manifest_sha256='b' * 64, preprocess_environment=env)
    old = dict(status='ok', version='v355_seen_category_calibration_v1', categories=50, views_per_category=20,
        categories_detail={c: {'views': 20, 'q99': .25} for c in partition['categories']}, input_size=672,
        bank_sha256=producer['bank_sha256'], bank_manifest_sha256=producer['bank_manifest_sha256'],
        preprocess_environment=env, code_hashes={'v355_seen_category_calibration.py': mod.v366.LEGACY_V355_SHA})
    bank = dict(metadata={'partition': partition, 'preprocess_environment': env, 'input_size': 560},
                trace={'grid': [35, 35]}, bank_sha256=producer['bank_sha256'])
    mod.validate_frozen(producer, old, bank, selected, partition, env)
    for target, key, value in [(producer, 'reference_calibration_q99', .05),
        (producer, 'category_calibration_sha256', 'a' * 64), (old, 'input_size', 560),
        (old, 'code_hashes', {'v355_seen_category_calibration.py': 'a' * 64}),
        (bank['metadata'], 'input_size', 672), (bank['trace'], 'grid', [42, 42]),
        (producer, 'selected_checkpoints', selected[::-1])]:
        original = target[key]
        target[key] = value
        with pytest.raises(ValueError, match='Frozen'):
            mod.validate_frozen(producer, old, bank, selected, partition, env)
        target[key] = original


def test_source_snapshot_records_actual_hash_and_rejects_change(monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, 'sha256_file', lambda _: 'a' * 64)
    assert mod.source_snapshot({'source': 'a' * 64}) == {'source': 'a' * 64}
    with pytest.raises(ValueError, match='Source changed'):
        mod.source_snapshot({'source': 'b' * 64})


def test_current_v355_source_is_bound_to_passed_v369_report():
    import inspect
    source = inspect.getsource(module().preflight)
    assert source.index("'v355_seen_category_calibration.py'") < source.index('v366.require_sources')


def test_smoke_rejects_empty_normal_iterator(monkeypatch, tmp_path):
    mod = module()
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    (tmp_path / 'outputs').mkdir()
    monkeypatch.setattr(mod, 'check_resources', lambda: None)
    monkeypatch.setattr(mod, 'preflight', lambda: {'sources': {}, 'producer': {'selected_checkpoints': []},
                        'provenance': {}, 'index': {'seen': frozenset({'a'})}})
    monkeypatch.setattr(mod, 'gpu_responses', lambda *a, **k: (_ for _ in ()))
    monkeypatch.setattr(mod, 'final_check', lambda _: None)
    with pytest.raises(ValueError, match='first normal category'):
        mod.run(SimpleNamespace(output_root=tmp_path / 'outputs' / 'new', smoke=True))
