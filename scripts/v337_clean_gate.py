"""Select clean-trained checkpoints and report exact category-macro development evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.v283_supervised_dinomaly_longtrain import atomic_write_json, build_category_folds, load_proxy_rows, sha256_file
from scripts.v331_checkpoint_benchmark import FIELDS, _balanced, _matrix, compare_candidate
from scripts.v336_clean_train import SOURCE_SHA, validate_clean_report


def _validate_evaluation(value: dict, categories: list[str], views: int) -> np.ndarray:
    if value['views'] != views or value['pixel_count'] != views * 448 * 448 or views <= 0:
        raise ValueError('Incomplete full-pixel development evaluation')
    if value.get('prediction_scale') != 1.0 or value.get('categories') != len(categories) or value.get('groups',0) * 5 != views:
        raise ValueError('Contradictory evaluation scale/category/group metadata')
    counts=[value['metrics']['per_category'][c].get('groups',0) for c in categories]
    if any(n <= 0 for n in counts) or sum(counts) != value['groups']:
        raise ValueError('Contradictory per-category group counts')
    return _matrix(value['metrics'], categories)


def choose_checkpoint(fold: int, categories: list[str], baseline: dict, reports: Sequence[dict]) -> dict | None:
    categories = sorted(categories)
    _validate_evaluation(baseline, categories, baseline['views'])
    eligible = []
    for report in reports:
        if (report['status'] != 'ok' or report['development_only'] is not True or report['fold_id'] != fold
                or int(report['step']) <= 0):
            raise ValueError('Invalid clean checkpoint report')
        _validate_evaluation(report['evaluation'], categories, baseline['views'])
        comparison = compare_candidate(baseline['metrics'], report['evaluation']['metrics'])
        if comparison['passed']:
            eligible.append({**report, 'comparison': comparison, 'baseline': baseline, 'held_out_categories': categories})
    return max(eligible, key=lambda r: (r['comparison']['balanced_score'], -r['step'])) if eligible else None


def route_decision(selected: Sequence[dict]) -> dict[str, Any]:
    if len(selected) != 4 or {s['fold_id'] for s in selected} != set(range(4)):
        raise ValueError('Four distinct clean fold selections are required')
    categories: set[str] = set()
    before, after, fold_gains = [], [], []
    for item in sorted(selected, key=lambda s: s['fold_id']):
        held = sorted(item['held_out_categories'])
        if not held or categories.intersection(held):
            raise ValueError('Overlapping or empty clean held-out categories')
        categories.update(held)
        old = _validate_evaluation(item['baseline'], held, item['baseline']['views'])
        new = _validate_evaluation(item['evaluation'], held, item['baseline']['views'])
        before.extend(old); after.extend(new)
        fold_gains.append(float(np.mean(_balanced(new) - _balanced(old))))
    old, new = np.asarray(before), np.asarray(after)
    gains = _balanced(new) - _balanced(old)
    delta = np.mean(new, axis=0) - np.mean(old, axis=0)
    checks = {
        'every_fold_gain_positive': bool(np.all(np.asarray(fold_gains) > 0)),
        'mean_balanced_gain_at_least_002': bool(np.mean(gains) >= .02),
        'category_nonregression_at_least_080': bool(np.mean(gains >= 0) >= .8),
        'worst_quartile_at_least_minus0005': bool(np.quantile(gains,.25) >= -.005),
        'pixel_ap_nonregression': bool(delta[3] >= 0),
        'image_auc_loss_at_most_001': bool(delta[0] >= -.01),
        'pixel_f1_loss_at_most_0005': bool(delta[4] >= -.005),
    }
    return {'passed': all(checks.values()), 'checks': checks, 'category_count': len(categories),
        'fold_gains': fold_gains, 'mean_balanced_gain': float(np.mean(gains)),
        'category_nonregression': float(np.mean(gains >= 0)),
        'baseline_macro': {f'{l}_{k}': float(v) for (l,k),v in zip(FIELDS,np.mean(old,axis=0))},
        'macro': {f'{l}_{k}': float(v) for (l,k),v in zip(FIELDS,np.mean(new,axis=0))},
        'scope': 'clean auxiliary category-held-out development selection; not B score or global optimum'}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.v339_verify_clean_metrics import verify_replay

    root, clean, output = Path(args.training_root), Path(args.clean_root), Path(args.output)
    if output.exists(): raise FileExistsError(output)
    clean_report_path = clean / 'report.json'
    clean_report = json.loads(clean_report_path.read_text(encoding='utf-8'))
    index = clean / 'clean_proxy_index.jsonl'
    rows = load_proxy_rows(index)
    validate_clean_report(clean_report, sha256_file(index), rows)
    folds_path = clean / 'category_folds.json'
    folds = json.loads(folds_path.read_text(encoding='utf-8'))
    if folds != build_category_folds(rows,4): raise ValueError('Clean fold identity changed')
    selected, missing = [], []
    for fold in range(4):
        folder = root / f'fold{fold}'
        terminal = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        config_path = folder / 'immutable_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        expected_fold = next(f for f in folds['folds'] if f['fold'] == fold)
        if config['target_steps'] != 1000 or config['checkpoint_steps'] != [500,1000]:
            raise ValueError('The clean pilot gate requires complete500/1000-step training')
        if (terminal['status'] != 'ok' or terminal.get('clean_training') is not True or terminal['steps'] != config['target_steps']
                or terminal['config'] != config or config['fold_id'] != fold or config['partition_key'] != 'category'
                or config['source_checkpoint_sha256'] != SOURCE_SHA
                or config['clean_index_sha256'] != sha256_file(index)
                or config['clean_report_sha256'] != sha256_file(clean_report_path)
                or config['fold_manifest_sha256'] != sha256_file(folds_path)
                or set(config['held_out_categories']) != set(expected_fold['categories'])
                or set(config['training_categories']) != set(clean_report['retained_categories']) - set(expected_fold['categories'])
                or set(config['excluded_target_categories']) != set(clean_report['excluded_categories'])
                or config['input_size'] != 448 or config['source_dropout'] != .4 or config['prediction_scale'] != 1.0
                or config['driver_sha256'] != sha256_file(Path(__file__).with_name('v336_clean_train.py'))):
            raise ValueError('Clean training provenance or completion mismatch')
        required_dependencies = {'v332_reachable_normal.py','v332_train_pilot.py','v330_raw_response.py',
                                 'v330_replay.py','v283_supervised_dinomaly_longtrain.py'}
        if set(config['dependency_hashes']) != required_dependencies:
            raise ValueError('Missing or changed training dependency hashes')
        for name,digest in config['dependency_hashes'].items():
            if Path(name).name != name or sha256_file(Path(__file__).parent/name) != digest:
                raise ValueError('Training/evaluation implementation changed')
        baseline = json.loads((folder/'baseline.json').read_text(encoding='utf-8'))
        if baseline['views'] != expected_fold['test_images']:
            raise ValueError('Clean baseline view coverage mismatch')
        expected_group_counts={cat:len({r['group'] for r in rows if r['category']==cat and r['phase']=='test'})
                               for cat in expected_fold['categories']}
        if any(baseline['metrics']['per_category'][cat]['groups']!=count for cat,count in expected_group_counts.items()):
            raise ValueError('Clean baseline per-category group count mismatch')
        reports = []
        if terminal['validation_steps'] != config['checkpoint_steps']:
            raise ValueError('Missing requested checkpoint evaluations')
        for step in config['checkpoint_steps']:
            path = folder/f'validation_i{step:06d}.json'
            report = json.loads(path.read_text(encoding='utf-8'))
            checkpoint = folder/'checkpoints'/f'state_i{step:06d}.pt'
            if (report['step'] != step or Path(report['checkpoint']).resolve() != checkpoint.resolve()
                    or sha256_file(checkpoint) != report['checkpoint_sha256']):
                raise ValueError('Checkpoint identity changed since clean evaluation')
            if any(report['evaluation']['metrics']['per_category'][cat]['groups']!=count for cat,count in expected_group_counts.items()):
                raise ValueError('Checkpoint per-category group count mismatch')
            reports.append({**report,'validation_report':str(path),'validation_sha256':sha256_file(path)})
        replay_path=Path(args.replay_root)/f'fold{fold}'/'replay.json'
        verified=verify_replay(replay_path,folder,index)
        if verified['fold_id']!=fold:raise ValueError('Independent replay fold mismatch')
        best = choose_checkpoint(fold,expected_fold['categories'],baseline,reports)
        if best is None:
            missing.append(fold)
        else:
            selected.append({**best,'immutable_config':config,'immutable_config_sha256':sha256_file(config_path),
                             'independent_replay':str(replay_path),'independent_replay_sha256':sha256_file(replay_path)})
    decision = route_decision(selected) if not missing else {'passed':False,'missing_eligible_folds':missing}
    result = {**decision,'version':'v337_clean_route_gate_v1','selected_reports':selected,
              'clean_root':str(clean),'training_root':str(root),'clean_index_sha256':sha256_file(index),
              'clean_report_sha256':sha256_file(clean_report_path),'script_sha256':sha256_file(Path(__file__))}
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_write_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='selected_reports'},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,required=True)
    parser.add_argument('--clean-root',type=Path,required=True)
    parser.add_argument('--replay-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
