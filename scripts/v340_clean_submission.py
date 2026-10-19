"""Stream a frozen clean four-model ensemble into an audited B submission."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts.build_b_submission import _test_index, audit_submission_dir
from scripts.build_v329_supervised_b_submission import _write_clean_zip
from scripts.v330_raw_response import audit_scores, checked_raw, encode_png, score_views
from scripts.v336_clean_train import ROOT, SOURCE_SHA
from scripts.v337_clean_gate import route_decision
from scripts.v339_verify_clean_metrics import evaluation_drift, replay_hashes, verify_replay


def combine_raw(maps: list[np.ndarray]) -> np.ndarray:
    if len(maps)!=4:raise ValueError('Four model outputs required')
    arrays=[checked_raw(value).astype(np.float32) for value in maps]
    shape=arrays[0].shape
    if len(shape)!=3 or shape[1:]!=(448,448) or any(a.shape!=shape for a in arrays):
        raise ValueError('Invalid raw batch geometry')
    return np.median(np.stack(arrays),axis=0).astype(np.float32)


def build_stream_package(groups: Mapping[str,tuple], batches: Iterable[tuple], output: Path,
                          provenance: dict[str,Any]) -> dict[str,Any]:
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    for group,views in groups.items():
        parts=group.split('/')
        if len(parts)!=2 or any(p in ('','.','..') or '\\' in p or ':' in p for p in parts) or tuple(views)!=('0','1','2','3','4'):
            raise ValueError('Unsafe or incomplete group index')
    if not groups:raise ValueError('Empty test group index')
    expected={(group,view) for group,views in groups.items() for view in views}
    seen=set(); scores_by_group:dict[str,dict[str,float]]=defaultdict(dict)
    package=output/'package';package.mkdir(parents=True)
    raw_root=output/'raw'
    for categories,samples,views,maps in batches:
        if not (len(categories)==len(samples)==len(views)==len(maps)):
            raise ValueError('Batch metadata alignment error')
        for category,sample,view,raw in zip(categories,samples,views,maps):
            category,sample,view=str(category),str(sample),str(view)
            group=f'{category}/{sample}'
            key=(group,view)
            if key not in expected or key in seen:raise ValueError('Unknown or duplicate test view')
            raw=checked_raw(raw).astype(np.float32)
            png=encode_png(raw)
            target=package/'predicted_masks'/group/f'{view}_mask.png'
            target.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(png).save(target)
            raw_path=raw_root/group/f'{view}.npz'
            raw_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(raw_path,raw_map=raw,category=np.asarray(category),sample=np.asarray(sample),view=np.asarray(view))
            scores_by_group[group][view]=trainer.top_mean(raw,.005)
            seen.add(key)
        if len(seen)%500==0 or len(seen)==len(expected):print(f'clean_ensemble_saved={len(seen)}/{len(expected)}',flush=True)
    if seen!=expected:raise ValueError('Incomplete clean ensemble coverage')
    scores={group:score_views([scores_by_group[group][v] for v in groups[group]]) for group in sorted(groups)}
    csv_path=package/'submission.csv'
    with csv_path.open('x',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['group_folder','anomaly_score'])
        writer.writerows((g,f'{s:.12f}') for g,s in scores.items())
    numeric_audit=audit_scores({g:float(f'{s:.12f}') for g,s in scores.items()})
    structural=audit_submission_dir(package,expected_groups=set(groups))
    trainer.atomic_write_json(output/'audit.json',{'structural':structural,'scores':numeric_audit})
    if not structural['passed'] or not numeric_audit['passed']:raise ValueError('Clean package failed audit')
    staging,zip_path=output/'submission.zip.part',output/'submission.zip'
    _write_clean_zip(package,staging)
    with zipfile.ZipFile(staging) as archive:
        names=archive.namelist()
        target_names={'submission.csv',*{f'predicted_masks/{g}/{v}_mask.png' for g,v in expected}}
        if len(names)!=len(target_names) or set(names)!=target_names or archive.testzip() is not None:
            raise ValueError('ZIP entries/CRC mismatch')
        for name in sorted(target_names-{'submission.csv'}):
            with archive.open(name) as stream,Image.open(stream) as image:
                image.load()
                if image.size!=(448,448) or image.mode!='L':raise ValueError('Invalid packaged PNG')
    digest=trainer.sha256_file(staging)
    os.link(staging,zip_path);staging.unlink()
    result={'version':'v340_clean_stream_ensemble_v1','zip_path':str(zip_path),'sha256':digest,
            'bytes':zip_path.stat().st_size,'groups':len(groups),'masks':len(expected),'score_audit':numeric_audit,
            'structural_audit':structural,'provenance':provenance,
            'policy':'median raw four-model responses; top0.5% per view then top2-of-five; x/(x+0.10) mapping',
            'limitations':['B score is unknown; no automated online submission.',
                           'Category-held-out development selection does not independently evaluate four-model ensemble.']}
    trainer.atomic_write_json(output/'manifest.json',result)
    print(json.dumps({k:result[k] for k in ('zip_path','sha256','groups','masks','score_audit')}),flush=True)
    return result


def verify_selected_metrics(selection: dict[str,Any], attested: dict[str,Any]) -> None:
    checkpoint=attested['checkpoints'][str(selection['step'])]
    if (checkpoint['checkpoint_sha256']!=selection['checkpoint_sha256']
            or evaluation_drift(selection['baseline'],attested['baseline']['evaluation'])>1e-6
            or evaluation_drift(selection['evaluation'],checkpoint['evaluation'])>1e-6):
        raise ValueError('Selected metrics differ from independent replay')


def validate_gate(path: Path) -> dict[str,Any]:
    gate=json.loads(path.read_text(encoding='utf-8'))
    if gate.get('version')!='v337_clean_route_gate_v1' or gate.get('passed') is not True:
        raise ValueError('A passed clean selection gate is required')
    if gate['script_sha256']!=trainer.sha256_file(Path(__file__).with_name('v337_clean_gate.py')):
        raise ValueError('Selection implementation changed')
    selected=sorted(gate['selected_reports'],key=lambda r:r['fold_id'])
    recomputed=route_decision(selected)
    if not recomputed['passed'] or evaluation_drift(recomputed['macro'],gate['macro'])>1e-12:
        raise ValueError('Clean route metrics no longer pass or agree')
    clean=Path(gate['clean_root']);index=clean/'clean_proxy_index.jsonl'
    if gate['clean_index_sha256']!=trainer.sha256_file(index) or gate['clean_report_sha256']!=trainer.sha256_file(clean/'report.json'):
        raise ValueError('Clean data changed since selection')
    for selection in selected:
        folder=Path(gate['training_root'])/f"fold{selection['fold_id']}"
        replay=Path(selection['independent_replay'])
        if trainer.sha256_file(replay)!=selection['independent_replay_sha256']:raise ValueError('Metric replay changed')
        attested=verify_replay(replay,folder,index)
        verify_selected_metrics(selection,attested)
        config=selection['immutable_config']
        if config['source_checkpoint_sha256']!=SOURCE_SHA or config['prediction_scale']!=1.0:
            raise ValueError('Invalid clean checkpoint ancestry or scale')
    return gate


def run(args: argparse.Namespace) -> dict[str,Any]:
    import torch

    gate=validate_gate(args.gate)
    output=Path(args.output_root)
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<20*1024**3:raise RuntimeError('At least20GiB free required for raw ensemble and package')
    if torch.cuda.device_count()<4:raise RuntimeError('Four GPUs required for the frozen ensemble')
    deps=adapter._import_official_modules()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    selections=sorted(gate['selected_reports'],key=lambda r:r['fold_id'])
    models=[]
    for i,selection in enumerate(selections):
        checkpoint=Path(selection['checkpoint'])
        if trainer.sha256_file(checkpoint)!=selection['checkpoint_sha256']:raise ValueError('Selected model hash changed')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        config=selection['immutable_config']
        if payload['step']!=selection['step'] or payload['immutable_config']!=config:raise ValueError('Checkpoint config mismatch')
        model,_=adapter._build_model(deps,backbone='dinov3_vit_base_16',dropout=.4,reduce_dim=256,
                                     context_aware_recenter=True,mask_neighbor_size=0,weights_dir=ROOT/'Dinomaly2_dinov3/backbones/weights')
        model.load_state_dict(payload['model'],strict=True);del payload
        model=model.to(f'cuda:{i}').eval();model.requires_grad_(False);models.append(model)
    test_sha=trainer.sha256_file(args.test_zip)
    groups=_test_index(args.test_zip,'Test_B')
    dataset=adapter.CsigZipDataset(args.test_zip,'Test_B',adapter._build_transform(448,448),include_metadata=True)
    loader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    if len(dataset)!=len(groups)*5:raise ValueError('Test index count mismatch')
    code_fingerprints=replay_hashes()
    def infer_one(i:int,images:Any) -> np.ndarray:
        torch.cuda.set_device(i)
        with torch.inference_mode():
            encoded,decoded=models[i](images.to(f'cuda:{i}',non_blocking=True))
            return trainer.differentiable_anomaly_map(encoded,decoded,448).float().cpu().numpy()[:,0]
    def batches():
        with ThreadPoolExecutor(max_workers=4) as pool:
            for images,categories,samples,views in loader:
                futures=[pool.submit(infer_one,i,images) for i in range(4)]
                raw=combine_raw([future.result() for future in futures])
                yield categories,samples,views,raw
        if replay_hashes()!=code_fingerprints:raise ValueError('Producer/scoring code changed during prediction')
    provenance={'gate_path':str(args.gate),'gate_sha256':trainer.sha256_file(args.gate),
        'clean_index_sha256':gate['clean_index_sha256'],'test_zip_sha256':test_sha,
        'selected_checkpoints':[{'fold':s['fold_id'],'step':s['step'],'sha256':s['checkpoint_sha256']} for s in selections],
        'development_macro':gate['macro'],'producer_hashes':code_fingerprints,
        'stream_inference_sha256':trainer.sha256_file(Path(__file__)),
        'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda}
    return build_stream_package(groups,batches(),output,provenance)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--test-zip',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    run(parser.parse_args())
