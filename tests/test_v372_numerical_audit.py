"""CPU-only contracts for the bounded spot auditor, using one five-view object."""
import hashlib
import importlib
import io
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import v366_release_audit as parent


def module():
    name = 'scripts.v372_numerical_audit'
    assert importlib.util.find_spec(name) is not None, 'V372 numerical auditor is not implemented'
    return importlib.import_module(name)


def png(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    return stream.getvalue()


@pytest.fixture
def sample():
    scales = {'part': {'old_scale': .3, 'new_scale': .6}}
    rows, values = [], []
    for view in '01234':
        ref = np.full((448, 448), .01 + int(view) * .02, np.float32)
        ref.ravel()[:1003] += np.float32(.08)
        sparse, dense = np.full_like(ref, .4), np.full_like(ref, .1)
        rows.append(('part/S0001', view, ref, sparse, dense,
                     png(parent.independent_png_pixels(ref, dense, .6)),
                     png(parent.independent_png_pixels(ref, sparse, .3))))
        values.append(float(np.sort(ref.ravel())[-1004:].mean()))
    top = sum(sorted(values)[-2:]) / 2
    return rows, {'part/S0001': top / (top + .1)}, scales


def test_five_views_use_distinct_old_new_maps_and_top_two_score(sample):
    result = module().verify_sample(*sample)
    assert result['target_views'] == result['candidate_masks_checked'] == result['baseline_masks_checked'] == 5
    assert result['candidate_mismatched_pixels'] == result['baseline_mismatched_pixels'] == 0
    assert result['score_groups_checked'] == 1 and result['score_max_abs_error'] <= 1e-6
    assert result['new_training'] is result['test_b_labels_accessed'] is False


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'unknown', 'view'])
def test_target_coverage_fails_closed(sample, damage):
    rows, scores, scales = sample
    if damage == 'missing':
        rows.pop()
    elif damage == 'duplicate':
        rows.append(rows[0])
    else:
        row = list(rows[0])
        row[0 if damage == 'unknown' else 1] = 'other/S0001' if damage == 'unknown' else '5'
        rows[0] = tuple(row)
    with pytest.raises(ValueError, match='coverage|Unknown|duplicate'):
        module().verify_sample(rows, scores, scales)


@pytest.mark.parametrize('branch,damage', [('candidate', 'pixel'), ('baseline', 'pixel'),
                                        ('candidate', 'map'), ('baseline', 'map'),
                                        ('candidate', 'scale'), ('baseline', 'scale')])
def test_separate_map_scale_and_pixel_mismatches(sample, branch, damage):
    rows, scores, scales = sample
    row = list(rows[0])
    payload_index = 5 if branch == 'candidate' else 6
    if damage == 'pixel':
        with Image.open(io.BytesIO(row[payload_index])) as image:
            pixels = np.array(image)
        pixels[0, 0] += 1
        row[payload_index] = png(pixels)
    elif damage == 'map':
        row[4 if branch == 'candidate' else 3] = np.full((448, 448), .2, np.float32)
    else:
        scales['part']['new_scale' if branch == 'candidate' else 'old_scale'] = .9
    rows[0] = tuple(row)
    with pytest.raises(ValueError, match=branch + ' pixel mismatch'):
        module().verify_sample(rows, scores, scales)


@pytest.mark.parametrize('delta,passes', [(5e-7, True), (2e-6, False), (float('nan'), False)])
def test_score_tolerance(sample, delta, passes):
    rows, scores, scales = sample
    scores['part/S0001'] += delta
    if passes:
        assert module().verify_sample(rows, scores, scales)['score_max_abs_error'] <= 1e-6
    else:
        with pytest.raises(ValueError, match='score'):
            module().verify_sample(rows, scores, scales)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 3.])
def test_invalid_raw_maps(sample, value):
    sample[0][0][4][0, 0] = value
    with pytest.raises(ValueError):
        module().verify_sample(*sample)


def test_fixed_sampling_has_no_score_or_query_input():
    contract = {'seen_categories': [f'part{i:03d}' for i in range(50)],
                'groups': {f'part{c:03d}/S{g:04d}': list('01234')
                           for c in range(100) for g in range(25 if c < 50 else 20)}}
    records = [SimpleNamespace(group=g, view=v, category=g.split('/')[0], sample=g.split('/')[1],
                               member=f'Test_B/{g}/{v}.png')
               for g in sorted(contract['groups']) for v in contract['groups'][g]]
    batches = list(module().selected_batches(records, contract))
    groups = parent.sample_groups(contract, parent.CONTRACT_SHA)
    actual = [(records[rows[p]].group, records[rows[p]].view) for rows, positions in batches for p in positions]
    assert len(actual) == 250 and set(actual) == {(g, v) for g in groups for v in '01234'}
    assert len(groups) == 50 and sum(len(rows) for rows, _ in batches) <= 400
    assert any(len(rows) > len(positions) for rows, positions in batches)
    assert all(rows == list(range(rows[0], rows[0] + 4)) and rows[0] % 4 == 0 for rows, _ in batches)
    with pytest.raises(TypeError):
        list(module().selected_batches(records, contract, seed_sha='a' * 64))
    records[0].member = 'Test_A/wrong.png'
    with pytest.raises(ValueError, match='metadata'):
        list(module().selected_batches(records, contract))


@pytest.fixture
def calibration():
    from scripts.v372_dense_inference import calibration_record
    members = [f'Train/part/S{g:04d}/{v}.png' for g in range(4) for v in '01234']
    sparse = np.full((20, 448, 448), .25, np.float32)
    dense = np.full_like(sparse, .125)
    record = calibration_record(members, [(sparse, dense), (sparse, dense)], .2500001)
    return members, sparse, dense, record


def test_single_calibration_pass_matches_pinned_first_pass(calibration):
    members, sparse, dense, record = calibration
    assert module().verify_calibration(members, sparse, dense, record) == record['calibration_hashes'][0]
    dense[0, 0, 0] = .12  # q99 stays identical; byte identity must still fail.
    with pytest.raises(ValueError, match='calibration'):
        module().verify_calibration(members, sparse, dense, record)


@pytest.mark.parametrize('damage', ['numerator', 'old', 'new', 'q99', 'passes', 'missing', 'digest'])
def test_prepared_scales_are_source_bound(calibration, damage):
    record = deepcopy(calibration[3])
    old = {'categories_detail': {'part': {'q99': record['old_q99']}}}
    assert module().checked_scales({'part': record}, old, {'part'})['part'] == {
        'old_scale': record['old_scale'], 'new_scale': record['new_scale']}
    if damage == 'numerator': record['reference_q99'] = .043
    elif damage == 'old': record['old_scale'] += 1e-5
    elif damage == 'new': record['new_scale'] += 1e-5
    elif damage == 'q99': record['dense_q99'] = float('nan')
    elif damage == 'passes': record['calibration_hashes'] = record['calibration_hashes'][:1]
    elif damage == 'missing': old['categories_detail'] = {}
    else: record['calibration_hashes'] = [{'sparse': 'x', 'dense': 'x'}] * 2
    with pytest.raises(ValueError, match='[Cc]alibration|q99|scale'):
        module().checked_scales({'part': record}, old, {'part'})


def test_required_sources_cannot_be_omitted_changed_or_conflicted(tmp_path):
    mod = module()
    path = tmp_path / 'source.json'
    path.write_bytes(b'{"x":1}')
    required = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    assert mod.checked_sources(required, required) == required
    for bad in ({}, {str(path): 'a' * 64}, {str(path): 'invalid'}):
        with pytest.raises(ValueError, match='source|identity'):
            mod.checked_sources(bad, required)
    path.write_bytes(b'{"x":2}')
    with pytest.raises(ValueError, match='changed'):
        mod.checked_sources(required, required)
    with pytest.raises(ValueError, match='Conflicting'):
        mod.verify_sources(required, {str(path): 'b' * 64})


def test_pinned_json_rejects_wrong_hash_and_duplicate_keys(tmp_path):
    path = tmp_path / 'audit.json'
    path.write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(ValueError, match='identity'):
        module().read_verified(path, 'a' * 64)
    with pytest.raises(ValueError, match='Duplicate'):
        module().read_verified(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_busy_gpu_rejected_before_models_or_datasets(monkeypatch):
    import subprocess
    from scripts import v365_ensemble_category_calibration as v365
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: '987654321\n')
    monkeypatch.setattr(v365, 'load_models', lambda *a: pytest.fail('Must check idle GPUs before models'))
    with pytest.raises(RuntimeError, match='idle GPUs'):
        next(module().bounded_responses({}, {}))


def test_cli_and_new_report_only(tmp_path):
    mod = module()
    args = mod.argument_parser().parse_args([
        '--candidate', 'candidate.zip', '--candidate-sha256', 'a' * 64,
        '--producer-audit', 'audit.json', '--producer-audit-sha256', 'b' * 64,
        '--output', str(tmp_path / 'report.json')])
    assert args.contract == mod.ROOT / 'outputs/v366_source_contract.json'
    assert mod.CONTRACT_SHA == parent.CONTRACT_SHA
    args.output.write_text('{}')
    with pytest.raises(FileExistsError):
        mod.main(args)
    assert json.loads(args.output.read_text()) == {}


@pytest.mark.parametrize('damage', [None, 'missing_png', 'duplicate_png', 'csv'])
def test_archive_wrapper_reads_only_target_members(sample, monkeypatch, damage):
    mod = module()
    rows, scores, scales = sample
    monkeypatch.setattr(parent, 'sample_groups', lambda contract, seed: list(scores))
    csv_bytes = ('group_folder,anomaly_score\npart/S0001,' + str(scores['part/S0001']) + '\n').encode()
    archives, reads = [], []
    for index in (5, 6):
        payloads = {'submission.csv': csv_bytes,
                    **{f'predicted_masks/{g}/{v}_mask.png': row[index] for row in rows for g, v in [row[:2]]}}
        names = [*payloads, 'unrelated_not_a_png']
        if index == 5:
            if damage == 'missing_png': names.pop(1)
            elif damage == 'duplicate_png': names.append(names[1])
            elif damage == 'csv': payloads['submission.csv'] += b'\n'
        def read(name, payloads=payloads):
            reads.append(name)
            return payloads[name]
        archives.append(SimpleNamespace(namelist=lambda names=names: names, read=read))
    args = (*archives, {'seen_categories': ['part']}, (row[:5] for row in rows), scales)
    if damage:
        with pytest.raises(ValueError, match='Missing|duplicate|CSV'):
            mod.audit_core(*args)
    else:
        result = mod.audit_core(*args)
        assert result['target_views'] == 5 and result['selected_by_test_score'] is False
    assert 'unrelated_not_a_png' not in reads


@pytest.mark.parametrize('damage', [None, 'during', 'source', 'provenance', 'replay'])
def test_main_pins_inputs_and_checks_again_before_publication(tmp_path, monkeypatch, calibration, damage):
    mod = module()
    def write(name, payload):
        path = tmp_path / name
        path.write_bytes(payload)
        return path, hashlib.sha256(payload).hexdigest()
    source, digest = write('runtime.py', b'original')
    required = {str(source): digest}
    baseline, baseline_sha = write('baseline.zip', b'fixture')
    frozen, frozen_sha = write('frozen.json', b'{}')
    old_path, old_sha = write('old.json', b'{}')
    candidate, candidate_sha = write('candidate.zip', b'fixture_candidate')
    for obj, key, value in [(mod.parent, 'BASELINE_SHA', baseline_sha),
            (mod.density.v366, 'BASELINE', baseline), (mod.density.v366, 'PRODUCER', frozen),
            (mod.density.v366, 'PRODUCER_SHA', frozen_sha), (mod.density.v366, 'OLD_CALIBRATION', old_path),
            (mod.density.v366, 'LEGACY_CALIBRATION_SHA', old_sha)]:
        monkeypatch.setattr(obj, key, value)
    cats = [f'part{i:03d}' for i in range(50)]
    groups = [c + '/S0001' for c in cats]
    contract = dict(groups={g: list('01234') for g in groups}, seen_categories=cats,
                    seen_source_sha256=baseline_sha, producer_manifest_sha256=frozen_sha,
                    test_zip_sha256='a' * 64, train_zip_sha256=mod.density.v349.TRAIN_SHA)
    contract_path, contract_sha = write('contract.json', json.dumps(contract).encode())
    monkeypatch.setattr(mod, 'CONTRACT_SHA', contract_sha)
    monkeypatch.setattr(parent, 'sample_groups', lambda contract, seed: groups)
    provenance = dict(reference_q99=mod.REFERENCE_Q99, source_hashes_before=required, new_training=False)
    context = dict(sources=required, provenance=provenance, environment={},
                   producer={'test_zip_sha256': 'a' * 64}, index={'groups': {g: tuple('01234') for g in groups}, 'seen': cats},
                   old={'categories_detail': {c: {'q99': calibration[3]['old_q99']} for c in cats}})
    replay = dict(old_scale_masks_checked=6250, reference_score_groups_checked=1250,
                  old_scale_mismatched_masks=0, old_scale_mismatched_pixels=0, old_scale_max_abs_error=0,
                  reference_score_mismatched_groups=0, reference_score_max_abs_error=0.)
    audit = dict(status='candidate_unscored', version='v372_dense_v362_candidate_v1', sha256=candidate_sha,
                 bytes=candidate.stat().st_size, source_files_sha256=required,
                 provenance={**provenance, 'source_hashes_after': required}, replay=replay,
                 category_records=dict.fromkeys(cats, calibration[3]), new_training=False,
                 test_b_labels_accessed=False, raw_evidence_persisted=False, auto_upload=False)
    if damage == 'source': audit['source_files_sha256'] = {}
    elif damage == 'provenance': audit['provenance']['reference_q99'] = .05
    elif damage == 'replay': audit['replay']['reference_score_groups_checked'] = 1249
    audit_path, audit_sha = write('audit.json', json.dumps(audit).encode())
    monkeypatch.setattr(mod.density, 'preflight', lambda: context)
    monkeypatch.setattr(mod.density.v349, 'preprocessing_environment_identity', lambda: {})
    def responses(ctx, records):
        assert damage not in ('source', 'provenance', 'replay'), 'Invalid inputs must fail before GPU work'
        records.update(dict.fromkeys(cats, {}))
        ctx.update(batch_count=100, inferred_views_including_neighbours=400)
        if damage == 'during': source.write_bytes(b'changed')
        return
        yield
    monkeypatch.setattr(mod, 'bounded_responses', responses)
    def audit_core(candidate, baseline, contract, responses, scales):
        assert list(responses) == []
        return {'status': 'passed', 'target_views': 250}
    monkeypatch.setattr(mod, 'audit_core', audit_core)
    from contextlib import nullcontext
    monkeypatch.setattr(mod.zipfile, 'ZipFile', lambda path: nullcontext())
    args = SimpleNamespace(candidate=candidate, candidate_sha256=candidate_sha, producer_audit=audit_path,
                           producer_audit_sha256=audit_sha, contract=contract_path, output=tmp_path / 'report.json')
    if damage:
        with pytest.raises(ValueError, match='changed|required source|provenance|replay'):
            mod.main(args)
        assert not args.output.exists()
    else:
        report = mod.main(args)
        assert report['sources_before_sha256'] == report['sources_after_sha256']
        assert report['reference_numerator'] == .04275937380269168
        assert str(args.candidate) in report['sources_before_sha256']
        assert json.loads(args.output.read_text()) == report
