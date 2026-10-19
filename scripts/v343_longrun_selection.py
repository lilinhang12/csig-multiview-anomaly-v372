"""Select long-run candidates only after reproducing their full development metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts import v342_clean_longtrain as longtrain
from scripts.v331_checkpoint_benchmark import FIELDS, _balanced, _matrix, compare_candidate
from scripts.v337_clean_gate import _validate_evaluation
from scripts.v339_verify_clean_metrics import evaluation_drift

STEPS=(1000,2000,3000,5000)


def choose_long_checkpoint(source, reports):
    if len(reports)!=4 or {r['relative_additional_step'] for r in reports}!=set(STEPS):
        raise ValueError('All1000/2000/3000/5000 checkpoint evaluations are required')
    categories=sorted(source['held_out_categories'])
    initial=source['evaluation']
    _validate_evaluation(initial,categories,initial['views'])
    best=dict(fold_id=source['fold_id'],kind='source_fallback',checkpoint=source['checkpoint'],
              checkpoint_sha256=source['checkpoint_sha256'],additional_steps=0,total_steps=source['step'],
              baseline=initial,evaluation=initial,
              balanced_score=float(_balanced(_matrix(initial['metrics'],categories)).mean()))
    for report in sorted(reports,key=lambda r:r['relative_additional_step']):
        step=report['relative_additional_step']
        if (report['status']!='ok' or report['development_only'] is not True or report['fold_id']!=source['fold_id']
                or report['step']!=step or report['total_optimizer_step']!=source['step']+step):
            raise ValueError('Invalid additional-step/fold evidence')
        evaluated=report['evaluation']
        _validate_evaluation(evaluated,categories,initial['views'])
        if any(evaluated['metrics']['per_category'][c]['groups']!=initial['metrics']['per_category'][c]['groups'] for c in categories):
            raise ValueError('Per-category development groups changed')
        comparison=compare_candidate(initial['metrics'],evaluated['metrics'])
        if comparison['passed'] and comparison['balanced_score']>best['balanced_score']:
            best=dict(fold_id=source['fold_id'],kind='longrun_candidate',checkpoint=report['checkpoint'],
                      checkpoint_sha256=report['checkpoint_sha256'],additional_steps=step,total_steps=source['step']+step,
                      baseline=initial,evaluation=evaluated,comparison=comparison,balanced_score=comparison['balanced_score'])
    return best


def summarize_selections(selections):
    if len(selections)!=4 or {s['fold_id'] for s in selections}!=set(range(4)):
        raise ValueError('Four completed fold selections required')
    old,new,seen=[],[],set()
    for selected in sorted(selections,key=lambda s:s['fold_id']):
        categories=sorted(selected['baseline']['metrics']['per_category'])
        if not categories or seen.intersection(categories):raise ValueError('Duplicated or empty heldout categories')
        seen.update(categories)
        views=selected['baseline']['views']
        old.extend(_validate_evaluation(selected['baseline'],categories,views))
        new.extend(_validate_evaluation(selected['evaluation'],categories,views))
    old,new=np.asarray(old),np.asarray(new)
    return dict(category_count=len(seen),macro={f'{l}_{k}':float(v) for (l,k),v in zip(FIELDS,new.mean(axis=0))},
                baseline_macro={f'{l}_{k}':float(v) for (l,k),v in zip(FIELDS,old.mean(axis=0))},
                mean_balanced_gain=float((_balanced(new)-_balanced(old)).mean()),
                scope='clean category-heldout development selection; not B performance or independent ensemble validation')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_terminal(terminal,config,fold,gate):
    if (terminal.get('status')!='ok' or terminal.get('relative_additional_step')!=5000
            or terminal.get('total_optimizer_step')!=6000 or terminal.get('validation_steps')!=list(STEPS)
            or terminal.get('immutable_config')!=config or config.get('target_steps')!=5000
            or config.get('checkpoint_steps')!=list(STEPS) or terminal.get('fold_id')!=fold
            or config.get('fold_id')!=fold or Path(config['source_gate']).resolve()!=Path(gate).resolve()):
        raise ValueError('Full5k additional training identity/completion mismatch')


def validate_saved_inputs(config,hashes):
    if not hashes or any(config.get(key)!=value for key,value in hashes.items()):
        raise ValueError('Current source/input hashes differ from the saved training contract')


def run_fold(args):
    import torch

    output=Path(args.output_root)/f'fold{args.fold_id}'/'selection.json'
    if output.exists():raise FileExistsError(output)
    folder=Path(args.long_root)/f'fold{args.fold_id}'
    terminal=_read(folder/'report.json');config=_read(folder/'immutable_config.json')
    validate_terminal(terminal,config,args.fold_id,args.gate)
    source_args=SimpleNamespace(gate=Path(config['source_gate']),fold_id=args.fold_id)
    source,training,development,approved,files,hashes=longtrain.preflight(source_args)
    validate_saved_inputs(config,hashes)
    baseline_path=folder/'baseline.json'
    baseline_sha=trainer.sha256_file(baseline_path)
    if terminal['baseline_sha256']!=baseline_sha:raise ValueError('Baseline identity changed')
    baseline=_read(baseline_path);longtrain.validate_baseline(baseline,config,source)
    longtrain.verify_inputs(files,hashes,config,source,folder,trainer.sha256_file(folder/'immutable_config.json'),baseline_sha)
    if longtrain.auxiliary_file_hashes(training+development,approved)!=config['auxiliary_file_hashes']:
        raise ValueError('Auxiliary content set changed')
    reports=[]
    for step in STEPS:
        report=_read(folder/f'validation_i{step:06d}.json')
        checkpoint=folder/'checkpoints'/f'state_i{step:06d}.pt'
        if (Path(report['checkpoint']).resolve()!=checkpoint.resolve() or report['checkpoint_sha256']!=trainer.sha256_file(checkpoint)
                or report['immutable_config']!=config or report['baseline_sha256']!=baseline_sha):
            raise ValueError('Checkpoint/data identity differs from training evidence')
        reports.append(report)
    chosen=choose_long_checkpoint(source,reports)
    torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    original_args=SimpleNamespace(source_run_root=longtrain.clean.SOURCE_ROOT,source_step=100000,input_size=448,source_dropout=.4,
        backbone='dinov3_vit_base_16',reduce_dim=256,weights_dir=longtrain.clean.ROOT/'Dinomaly2_dinov3/backbones/weights')
    model,deps,_=trainer._load_source_model(original_args,torch.device('cuda:0'))
    loader=torch.utils.data.DataLoader(trainer.RealIADSupervisedDataset(development,448),batch_size=4,num_workers=2,
        shuffle=False,pin_memory=True,persistent_workers=True)
    replayed=[]
    for step,path,expected in [(0,Path(source['checkpoint']),source['evaluation'])]+[
        (report['step'],Path(report['checkpoint']),report['evaluation']) for report in reports]:
        state=torch.load(path,map_location='cpu',weights_only=False)
        if step==0:
            longtrain.validate_source(source,state,trainer.sha256_file(path),args.fold_id)
        elif state['immutable_config']!=config or state['baseline_sha256']!=baseline_sha or state['relative_additional_step']!=step:
            raise ValueError('Replayed checkpoint metadata mismatch')
        model.load_state_dict(state['model'],strict=True);del state
        actual=longtrain.pilot.evaluate(model,development,loader,1.0)
        drift=evaluation_drift(expected,actual)
        if drift>1e-6:raise ValueError(f'Longrun metric reproduction drift={drift} at additional step{step}')
        replayed.append({'additional_step':step,'checkpoint_sha256':trainer.sha256_file(path),'drift':drift})
        print(f'longrun_replay fold={args.fold_id} additional={step} drift={drift}',flush=True)
    if longtrain.runtime_source_hashes(deps,model)!=config['runtime_source_hashes']:
        raise ValueError('Replayed runtime implementation differs from training')
    longtrain.verify_inputs(files,hashes,config,source,folder,trainer.sha256_file(folder/'immutable_config.json'),baseline_sha)
    result={'status':'ok','version':'v343_verified_longrun_selection_v1','fold_id':args.fold_id,
            'selected':chosen,'independent_replay':replayed,'source_gate_sha256':trainer.sha256_file(args.gate),
            'training_report_sha256':trainer.sha256_file(folder/'report.json'),'baseline_sha256':baseline_sha,
            'config_sha256':trainer.sha256_file(folder/'immutable_config.json'),'long_root':str(Path(args.long_root)),
            'selection_script_sha256':trainer.sha256_file(Path(__file__)),'training_code_hashes':longtrain.code_hashes(),
            'long_config':config,'source':source}
    output.parent.mkdir(parents=True,exist_ok=True);trainer.atomic_write_json(output,result)
    print(json.dumps({'fold':args.fold_id,'kind':chosen['kind'],'additional_steps':chosen['additional_steps']}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold-id',type=int,choices=range(4),required=True)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--long-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    run_fold(parser.parse_args())
