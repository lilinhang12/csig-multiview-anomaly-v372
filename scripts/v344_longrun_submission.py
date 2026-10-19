"""Prediction adapter for independently selected clean long-run or fallback models."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts import v342_clean_longtrain as longtrain
from scripts.build_b_submission import _test_index
from scripts.v340_clean_submission import build_stream_package, combine_raw, validate_gate
from scripts.v343_longrun_selection import (STEPS, choose_long_checkpoint, summarize_selections,
                                            validate_saved_inputs, validate_terminal)


def validate_selection_record(record,source,reports,config,gate_sha):
    if (record.get('version')!='v343_verified_longrun_selection_v1' or record.get('status')!='ok'
            or record.get('fold_id')!=source['fold_id'] or record.get('source_gate_sha256')!=gate_sha
            or record.get('source')!=source or record.get('long_config')!=config
            or config.get('source_gate_sha256')!=gate_sha):
        raise ValueError('Long-run selection/source contract mismatch')
    expected=choose_long_checkpoint(source,reports)
    if record.get('selected')!=expected:
        raise ValueError('The selected long-run checkpoint is not the verified best eligible choice')
    replay=record.get('independent_replay',[])
    expected_hashes={0:source['checkpoint_sha256'],**{r['step']:r['checkpoint_sha256'] for r in reports}}
    if len(replay)!=5 or {r.get('additional_step') for r in replay}!=set(expected_hashes):
        raise ValueError('Incomplete independent replay')
    for entry in replay:
        drift=entry.get('drift',float('nan'))
        if (not isinstance(drift,(int,float)) or not math.isfinite(drift) or not 0<=drift<=1e-6
                or entry.get('checkpoint_sha256')!=expected_hashes[entry['additional_step']]):
            raise ValueError('Independent checkpoint replay identity/drift mismatch')
    return expected


def require_full_b_index(groups):
    if len(groups)!=2250 or any(tuple(v)!=('0','1','2','3','4') for v in groups.values()):
        raise ValueError('Full B requires2250 complete five-view groups')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inference_hashes():
    return {**longtrain.code_hashes(),**{name:trainer.sha256_file(Path(__file__).with_name(name)) for name in
                                       ['v343_longrun_selection.py','v344_longrun_submission.py']}}


def load_verified_bundle(selection_root,gate_path):
    selection_root,gate_path=Path(selection_root),Path(gate_path)
    gate=validate_gate(gate_path);gate_sha=trainer.sha256_file(gate_path)
    records=[];hashes={}
    for fold in range(4):
        path=selection_root/f'fold{fold}'/'selection.json';record=_read(path)
        if record['training_code_hashes']!=longtrain.code_hashes() or record['selection_script_sha256']!=trainer.sha256_file(Path(__file__).with_name('v343_longrun_selection.py')):
            raise ValueError('Training/selection implementation changed')
        folder=Path(record['long_root'])/f'fold{fold}'
        config=_read(folder/'immutable_config.json');terminal=_read(folder/'report.json')
        validate_terminal(terminal,config,fold,gate_path)
        source,training,development,approved,files,current=longtrain.preflight(SimpleNamespace(gate=gate_path,fold_id=fold))
        validate_saved_inputs(config,current)
        expected_hashes={'config_sha256':folder/'immutable_config.json',
                         'training_report_sha256':folder/'report.json','baseline_sha256':folder/'baseline.json'}
        if any(record[k]!=trainer.sha256_file(p) for k,p in expected_hashes.items()):
            raise ValueError('Long-run evidence changed after independent replay')
        longtrain.verify_inputs(files,current,config,source,folder,record['config_sha256'],record['baseline_sha256'])
        if longtrain.auxiliary_file_hashes(training+development,approved)!=config['auxiliary_file_hashes']:
            raise ValueError('Auxiliary file set changed')
        reports=[]
        for step in STEPS:
            report=_read(folder/f'validation_i{step:06d}.json')
            checkpoint=folder/'checkpoints'/f'state_i{step:06d}.pt'
            if (report['baseline_sha256']!=record['baseline_sha256'] or report['immutable_config']!=config
                    or Path(report['checkpoint']).resolve()!=checkpoint.resolve()
                    or report['checkpoint_sha256']!=trainer.sha256_file(checkpoint)):
                raise ValueError('Replayed checkpoint/report identity changed')
            reports.append(report)
        validate_selection_record(record,source,reports,config,gate_sha)
        records.append(record);hashes[str(path)]=trainer.sha256_file(path)
    return {'source_gate':gate,'source_gate_sha256':gate_sha,'records':records,
            'selection_record_hashes':hashes,'summary':summarize_selections([r['selected'] for r in records])}


def run(args):
    import torch

    output=Path(args.output_root)
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<20*1024**3:raise RuntimeError('20GiB free required for raw responses/package')
    bundle=load_verified_bundle(args.selection_root,args.gate)
    if torch.cuda.device_count()<4:raise RuntimeError('Four GPUs required')
    groups=_test_index(args.test_zip,'Test_B');require_full_b_index(groups)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    deps=adapter._import_official_modules();models=[]
    for fold,record in enumerate(bundle['records']):
        choice=record['selected'];checkpoint=Path(choice['checkpoint'])
        if trainer.sha256_file(checkpoint)!=choice['checkpoint_sha256']:raise ValueError('Selected model hash changed')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if choice['kind']=='source_fallback':
            longtrain.validate_source(record['source'],state,choice['checkpoint_sha256'],fold)
        elif (state['immutable_config']!=record['long_config'] or state['baseline_sha256']!=record['baseline_sha256']
                or state['relative_additional_step']!=choice['additional_steps'] or state['total_optimizer_step']!=choice['total_steps']):
            raise ValueError('Long-run checkpoint metadata mismatch')
        model,_=adapter._build_model(deps,backbone='dinov3_vit_base_16',dropout=.4,reduce_dim=256,
                  context_aware_recenter=True,mask_neighbor_size=0,weights_dir=longtrain.clean.ROOT/'Dinomaly2_dinov3/backbones/weights')
        model.load_state_dict(state['model'],strict=True);del state
        model=model.to(f'cuda:{fold}').eval();model.requires_grad_(False)
        if longtrain.runtime_source_hashes(deps,model)!=record['long_config']['runtime_source_hashes']:
            raise ValueError('Actual inference model code differs from training')
        models.append(model)
    test_sha=trainer.sha256_file(args.test_zip)
    dataset=adapter.CsigZipDataset(args.test_zip,'Test_B',adapter._build_transform(448,448),include_metadata=True)
    if len(dataset)!=11250:raise ValueError('Full B view count mismatch')
    loader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    code_hashes=inference_hashes()
    def worker(fold,images):
        torch.cuda.set_device(fold)
        with torch.inference_mode():
            encoded,decoded=models[fold](images.to(f'cuda:{fold}',non_blocking=True))
            return trainer.differentiable_anomaly_map(encoded,decoded,448).float().cpu().numpy()[:,0]
    def batches():
        with ThreadPoolExecutor(max_workers=4) as pool:
            for images,categories,samples,views in loader:
                futures=[pool.submit(worker,f,images) for f in range(4)]
                yield categories,samples,views,combine_raw([future.result() for future in futures])
        if trainer.sha256_file(args.gate)!=bundle['source_gate_sha256'] or any(trainer.sha256_file(Path(p))!=h for p,h in bundle['selection_record_hashes'].items()):
            raise ValueError('Selection evidence changed during inference')
        for name,digest in code_hashes.items():
            if trainer.sha256_file(Path(__file__).with_name(name))!=digest:raise ValueError('Inference code changed')
    provenance={'kind':'clean_longrun_or_fallback_ensemble','gate_path':str(args.gate),
                'gate_sha256':bundle['source_gate_sha256'],'selection_root':str(args.selection_root),
                'selection_record_hashes':bundle['selection_record_hashes'],'summary':bundle['summary'],
                'selected_checkpoints':[r['selected'] for r in bundle['records']],
                'test_zip_sha256':test_sha,'code_hashes':code_hashes}
    return build_stream_package(groups,batches(),output,provenance)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-root',type=Path,required=True)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--test-zip',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    run(parser.parse_args())
