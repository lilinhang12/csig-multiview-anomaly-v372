"""Independent artifact-integrity audit for the V356 B-blind archive."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_b_submission import _test_index
from scripts.v283_supervised_dinomaly_longtrain import atomic_write_json, sha256_file
from scripts.v330_raw_response import audit_scores, score_views
from scripts.v344_longrun_submission import require_full_b_index
from scripts.v348_clean_memory_core import encode_evidence
from scripts.v350_scaled_evidence import SCHEMA as SCALED_SCHEMA


RAW_FIELDS = frozenset({'raw_evidence', 'reference_view_score', 'schema', 'category', 'sample', 'view'})
SCOPE = 'Artifact identity and raw/CSV/PNG integrity only; not model provenance or quality certification'


def recompute_group_score(values):
    if len(values) != 5 or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError('Exactly five finite reference view scores are required')
    return score_views(values)


def decode_png(payload):
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != 'PNG' or image.mode != 'L' or image.size != (448, 448):
                raise ValueError('PNG must have format PNG, mode L, and dimensions 448x448')
            image.load()
            return np.asarray(image).copy()
    except (OSError, SyntaxError) as error:
        raise ValueError('Invalid PNG payload') from error


def validate_raw_png(raw, png):
    raw = np.asarray(raw)
    png = np.asarray(png)
    if raw.shape != (448, 448) or raw.dtype != np.dtype(np.float32) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('Invalid stored scaled raw evidence')
    if png.shape != (448, 448) or png.dtype != np.dtype(np.uint8):
        raise ValueError('Invalid decoded PNG array')
    if not np.array_equal(png, encode_evidence(raw)):
        raise ValueError('PNG differs from stored scaled evidence encoding')


def _text_field(data, name, expected):
    value = np.asarray(data[name])
    if value.shape != () or value.dtype.kind != 'U' or value.item() != expected:
        raise ValueError(f'Raw evidence identity mismatch: {name}')


def load_raw_record(path, group, view):
    payload = Path(path).read_bytes()
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != {f'{field}.npy' for field in RAW_FIELDS}:
                raise ValueError('Raw evidence NPZ fields are not exact and unique')
    except zipfile.BadZipFile as error:
        raise ValueError('Raw evidence is not a valid NPZ archive') from error
    category, sample = str(group).split('/', 1)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            if len(data.files) != len(RAW_FIELDS) or set(data.files) != RAW_FIELDS:
                raise ValueError('Raw evidence NPZ fields are not exact and unique')
            _text_field(data, 'schema', SCALED_SCHEMA)
            _text_field(data, 'category', category)
            _text_field(data, 'sample', sample)
            _text_field(data, 'view', str(view))
            raw = np.asarray(data['raw_evidence'])
            if raw.shape != (448, 448) or raw.dtype != np.dtype(np.float32) or not np.isfinite(raw).all() or np.any(raw < 0):
                raise ValueError('Invalid stored scaled raw evidence')
            ref = np.asarray(data['reference_view_score'])
            if ref.shape != () or ref.dtype.kind not in 'fiu' or not np.isfinite(ref).item() or not 0 <= float(ref) <= 2:
                raise ValueError('Invalid reference view score')
            return {'raw_evidence': raw, 'reference_view_score': float(ref)}
    except (OSError, ValueError, TypeError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError('Invalid raw evidence NPZ') from error


def validate_release_manifest(manifest):
    if (manifest.get('status') != 'ok' or manifest.get('version') != 'v356_seen_memory_submission_v1'
            or (manifest.get('groups'), manifest.get('masks'), manifest.get('entries')) != (2250, 11250, 11251)
            or manifest.get('mask_schema') != SCALED_SCHEMA
            or manifest.get('classification_source') != 'reference_ensemble_top005_top2'
            or manifest.get('test_b_labels_accessed') is not False
            or manifest.get('score_audit', {}).get('passed') is not True
            or manifest.get('score_audit', {}).get('rows') != 2250
            or manifest.get('score_audit', {}).get('unique') != 2250):
        raise ValueError('V356 manifest contract failed')
    return True


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _publish_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    atomic_write_json(temporary, payload)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(path) from error
    finally:
        temporary.unlink(missing_ok=True)


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    root = Path(args.run_root).resolve()
    manifest_path = root / 'manifest.json'
    archive_path = root / 'submission.zip'
    manifest_hash_before = sha256_file(manifest_path)
    archive_hash_before = sha256_file(archive_path)
    manifest = _read(manifest_path)
    validate_release_manifest(manifest)
    provenance = manifest.get('provenance', {})
    test_zip_path = Path(args.test_zip)
    test_zip_hash_before = sha256_file(test_zip_path)
    if (provenance.get('test_b_labels_accessed') is not False
            or test_zip_hash_before != provenance.get('test_zip_sha256')
            or archive_hash_before != manifest.get('sha256')
            or archive_path.stat().st_size != manifest.get('bytes')):
        raise ValueError('Test_B archive identity or blind-access contract failed')
    groups = _test_index(test_zip_path, 'Test_B')
    require_full_b_index(groups)
    expected_masks = {f'predicted_masks/{group}/{view}_mask.png' for group, views in groups.items() for view in views}
    expected_raw = {f'{group}/{view}.npz' for group, views in groups.items() for view in views}
    if (manifest.get('groups'), manifest.get('masks'), manifest.get('entries')) != (
            len(groups), len(expected_masks), len(expected_masks) + 1):
        raise ValueError('V356 manifest coverage mismatch')
    if manifest.get('score_audit', {}).get('rows') != len(groups):
        raise ValueError('V356 manifest score coverage mismatch')
    raw_root = root / 'raw'
    actual_raw = {path.relative_to(raw_root).as_posix() for path in raw_root.rglob('*') if path.is_file()} if raw_root.is_dir() else set()
    if actual_raw != expected_raw:
        raise ValueError('Raw evidence view coverage mismatch')

    csv_scores = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {'submission.csv', *expected_masks}:
            raise ValueError('B archive membership mismatch')
        if archive.testzip() is not None:
            raise ValueError('B archive CRC mismatch')
        reader = csv.DictReader(io.StringIO(archive.read('submission.csv').decode('utf-8')))
        if reader.fieldnames != ['group_folder', 'anomaly_score']:
            raise ValueError('CSV header mismatch')
        rows = list(reader)
        if len(rows) != len(groups) or len({row['group_folder'] for row in rows}) != len(groups) or set(row['group_folder'] for row in rows) != set(groups):
            raise ValueError('CSV group coverage mismatch')
        try:
            csv_scores = {row['group_folder']: float(row['anomaly_score']) for row in rows}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('CSV score is not numeric') from error
        for number, (group, views) in enumerate(groups.items(), 1):
            reference_scores = []
            for view in views:
                record = load_raw_record(raw_root / group / f'{view}.npz', group, view)
                validate_raw_png(record['raw_evidence'], decode_png(archive.read(f'predicted_masks/{group}/{view}_mask.png')))
                reference_scores.append(record['reference_view_score'])
            score = recompute_group_score(reference_scores)
            if not np.isfinite(csv_scores[group]) or abs(float(f'{score:.12f}') - csv_scores[group]) > 1e-12:
                raise ValueError('CSV score does not match five reference view scores')
            if number % 250 == 0 or number == len(groups):
                print(f'v357_release_verified={number}/{len(groups)}', flush=True)

    numeric = audit_scores(csv_scores)
    if not numeric['passed']:
        raise ValueError('V356 CSV score audit failed')
    if manifest.get('score_audit', {}).get('unique') != numeric['unique']:
        raise ValueError('V356 manifest score audit mismatch')
    if (sha256_file(archive_path) != archive_hash_before
            or archive_path.stat().st_size != manifest.get('bytes')
            or sha256_file(manifest_path) != manifest_hash_before
            or sha256_file(test_zip_path) != test_zip_hash_before):
        raise ValueError('Archive or manifest changed during audit')
    result = {
        'status': 'ok', 'passed': True, 'version': 'v357_v356_release_audit_v1', 'run_root': str(root),
        'zip_path': str(archive_path), 'sha256': archive_hash_before, 'bytes': archive_path.stat().st_size,
        'groups': len(groups), 'masks': len(expected_masks), 'entries': len(expected_masks) + 1,
        'crc': 'ok', 'mode': 'L', 'dimensions': [448, 448], 'score_audit': numeric,
        'raw_png_equality': True, 'raw_csv_equality': True, 'model_provenance_verified': False,
        'quality_status': manifest.get('quality_status', 'experimental_not_promoted'),
        'manifest_sha256': manifest_hash_before, 'scope': SCOPE,
    }
    _publish_new(output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--test-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
