"""Independently reproduce frozen clean-training metrics before checkpoint selection."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts.v332_train_pilot import evaluate, partition_rows
from scripts.v336_clean_train import ROOT, SOURCE_ROOT, SOURCE_SHA, validate_initialization, validate_clean_report


def evaluation_drift(expected: Any, actual: Any) -> float:
    if isinstance(expected, dict):
        if not isinstance(actual,dict) or expected.keys()!=actual.keys():
            raise ValueError('Reproduced evaluation structure differs')
        return max((evaluation_drift(expected[k],actual[k]) for k in expected),default=0.0)
    if isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not isinstance(actual,(int,float)) or not math.isfinite(float(expected)) or not math.isfinite(float(actual)):
            raise ValueError('Invalid numeric evaluation evidence')
        return abs(float(expected)-float(actual))
    if expected != actual:raise ValueError('Evaluation protocol metadata changed')
    return 0.0


def replay_hashes() -> dict[str,str]:
    folder=Path(__file__).resolve().parent
    names=['v339_verify_clean_metrics.py','v337_clean_gate.py','v336_clean_train.py','v335_clean_data.py',
           'v331_checkpoint_benchmark.py','v332_train_pilot.py','v332_reachable_normal.py',
           'v330_replay.py','v330_raw_response.py','v283_supervised_dinomaly_longtrain.py',
           'v276_i448_state_infer.py','v182_dinomaly2_dinov3_csig_official.py']
    return {name:trainer.sha256_file(folder/name) for name in names}


def run(args: argparse.Namespace) -> dict[str,Any]:
    import torch

    folder=Path(args.training_root)/f'fold{args.fold_id}'
    output=Path(args.output_root)/f'fold{args.fold_id}'/'replay.json'
    if output.exists():raise FileExistsError(output)
    terminal_path=folder/'report.json';config_path=folder/'immutable_config.json'
    terminal=json.loads(terminal_path.read_text(encoding='utf-8'))
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if (terminal.get('status')!='ok' or terminal.get('clean_training') is not True or terminal.get('steps')!=1000
            or config.get('target_steps')!=1000 or config.get('checkpoint_steps')!=[500,1000] or terminal['config']!=config):
        raise ValueError('Independent replay requires completed1000-step training')
    clean=Path(args.clean_root);index=clean/'clean_proxy_index.jsonl'
    data_report=json.loads((clean/'report.json').read_text(encoding='utf-8'))
    rows=trainer.load_proxy_rows(index)
    validate_clean_report(data_report,trainer.sha256_file(index),rows)
    if config['clean_index_sha256']!=trainer.sha256_file(index) or config['fold_id']!=args.fold_id:
        raise ValueError('Replay clean-data/fold identity mismatch')
    held=set(config['held_out_categories'])
    _,development=partition_rows(rows,held,Path(data_report['approved_image_root']))
    code_hashes=replay_hashes()
    source=SOURCE_ROOT/'checkpoints/state_i100000.pt'
    digest=trainer.sha256_file(source)
    payload=torch.load(source,map_location='cpu',weights_only=False)
    validate_initialization(payload,digest);del payload
    torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.manual_seed(33601+args.fold_id)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    source_args=SimpleNamespace(source_run_root=SOURCE_ROOT,source_step=100000,input_size=448,source_dropout=.4,
                               backbone='dinov3_vit_base_16',reduce_dim=256,weights_dir=ROOT/'Dinomaly2_dinov3/backbones/weights')
    model,_,_=trainer._load_source_model(source_args,torch.device('cuda:0'))
    loader=torch.utils.data.DataLoader(trainer.RealIADSupervisedDataset(development,448),batch_size=4,num_workers=2,
                                      shuffle=False,pin_memory=True,persistent_workers=True)
    baseline_path=folder/'baseline.json'
    baseline_hash=trainer.sha256_file(baseline_path)
    recorded=json.loads(baseline_path.read_text(encoding='utf-8'))
    baseline=evaluate(model,development,loader,1.0)
    drift=evaluation_drift(recorded,baseline)
    if drift>1e-6:raise ValueError(f'Initializer baseline reproduction drift={drift}')
    print(f'replayed fold{args.fold_id} initializer drift={drift}',flush=True)
    checkpoints={}
    for step in (500,1000):
        path=folder/'checkpoints'/f'state_i{step:06d}.pt'
        report_path=folder/f'validation_i{step:06d}.json'
        record=json.loads(report_path.read_text(encoding='utf-8'))
        checkpoint_sha=trainer.sha256_file(path)
        if record['checkpoint_sha256']!=checkpoint_sha or record['step']!=step:
            raise ValueError('Checkpoint identity differs from recorded evaluation')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['immutable_config']!=config or state['step']!=step:raise ValueError('Checkpoint training config differs')
        model.load_state_dict(state['model'],strict=True);del state
        current=evaluate(model,development,loader,1.0)
        difference=evaluation_drift(record['evaluation'],current)
        if difference>1e-6:raise ValueError(f'Checkpoint{step} metric reproduction drift={difference}')
        checkpoints[str(step)]={'checkpoint_sha256':checkpoint_sha,'recorded_report_sha256':trainer.sha256_file(report_path),
                                'evaluation':current,'max_abs_drift':difference}
        print(f'replayed fold{args.fold_id} checkpoint{step} drift={difference}',flush=True)
    if replay_hashes()!=code_hashes or trainer.sha256_file(baseline_path)!=baseline_hash:
        raise ValueError('Code or baseline changed during independent replay')
    result={'version':'v339_independent_clean_metrics_v1','status':'ok','fold_id':args.fold_id,
            'source_initializer_sha256':SOURCE_SHA,'config_sha256':trainer.sha256_file(config_path),
            'training_report_sha256':trainer.sha256_file(terminal_path),'clean_index_sha256':trainer.sha256_file(index),
            'code_hashes':code_hashes,'baseline':{'recorded_sha256':baseline_hash,'evaluation':baseline,'max_abs_drift':drift},
            'checkpoints':checkpoints}
    output.parent.mkdir(parents=True,exist_ok=True)
    trainer.atomic_write_json(output,result)
    return result


def verify_replay(path: Path, folder: Path, clean_index: Path) -> dict[str,Any]:
    report=json.loads(path.read_text(encoding='utf-8'))
    if (report.get('version')!='v339_independent_clean_metrics_v1' or report.get('status')!='ok'
            or report.get('source_initializer_sha256')!=SOURCE_SHA or report.get('code_hashes')!=replay_hashes()
            or report['config_sha256']!=trainer.sha256_file(folder/'immutable_config.json')
            or report['training_report_sha256']!=trainer.sha256_file(folder/'report.json')
            or report['clean_index_sha256']!=trainer.sha256_file(clean_index)):
        raise ValueError('Independent metric replay identity or scoring code mismatch')
    if report['baseline']['recorded_sha256']!=trainer.sha256_file(folder/'baseline.json'):
        raise ValueError('Initializer baseline hash mismatch')
    recorded=json.loads((folder/'baseline.json').read_text(encoding='utf-8'))
    if evaluation_drift(recorded,report['baseline']['evaluation'])>1e-6:raise ValueError('Initializer baseline mismatch')
    if set(report['checkpoints'])!={'500','1000'}:raise ValueError('Incomplete independent checkpoint replay')
    for step in (500,1000):
        item=report['checkpoints'][str(step)]
        original=folder/f'validation_i{step:06d}.json'
        if item['checkpoint_sha256']!=trainer.sha256_file(folder/'checkpoints'/f'state_i{step:06d}.pt') or item['recorded_report_sha256']!=trainer.sha256_file(original):
            raise ValueError('Checkpoint identity changed after independent replay')
        if evaluation_drift(json.loads(original.read_text())['evaluation'],item['evaluation'])>1e-6:
            raise ValueError('Independent checkpoint evaluation mismatch')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,required=True)
    parser.add_argument('--clean-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--fold-id',type=int,choices=range(4),required=True)
    run(parser.parse_args())
