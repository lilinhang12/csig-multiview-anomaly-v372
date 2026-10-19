"""Clean auxiliary seen-category validation with disjoint normal references."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts import v342_clean_longtrain as longtrain
from scripts.v283_supervised_dinomaly_longtrain import sha256_file
from scripts.build_b_submission import _test_index
from scripts.v348_clean_memory_core import fuse_memory, normal_quantile_scale, resize_raw_map
from scripts.v350_scaled_evidence import ScaledEvidenceMetrics
from scripts.v351_source_calibration import ResidentCosineMemory
from scripts.v353_highres_development import canonical_view, primary_reference_score
from scripts.v344_longrun_submission import load_verified_bundle

ROOT = (Path.home() / 'csig')
INDEX = ROOT / 'outputs/v335_clean_auxiliary_20260910/clean_proxy_index.jsonl'
GATE = ROOT / 'outputs/v339_independent_clean_replay_final_20260910/clean_selection_gate.json'
BRANCHES = ['reference', 'category_memory', 'fusion']
DATA_LOADER_WORKERS = 0
SUPPORTED_INPUT_SIZES = (560, 672)


def split_seen_groups(groups, *, seed=36001):
    rng = np.random.default_rng(int(seed))
    if not groups:
        raise ValueError('Empty seen-category groups')
    if any(tuple(v) != ('0', '1', '2', '3', '4') for v in groups.values()):
        raise ValueError('Seen groups must contain five ordered views')
    if len(groups) != 25:
        raise ValueError('Seen-category validation requires25 complete groups')
    keys = sorted(groups)
    chosen = set(rng.choice(len(keys), size=9, replace=False).tolist())
    calibration = set(list(sorted(chosen))[:4])
    audit = chosen - calibration
    return {'seed': int(seed), 'bank_groups': [key for i, key in enumerate(keys) if i not in chosen],
            'calibration_groups': [key for i, key in enumerate(keys) if i in calibration],
            'audit_groups': [key for i, key in enumerate(keys) if i in audit],
            'bank_views': 80, 'calibration_views': 20, 'audit_views': 25}


def source_q99(values):
    raw = np.asarray(values, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (448, 448) or not raw.size or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('Expected finite nonnegative448x448 maps')
    value = float(np.quantile(raw, .99, method='linear'))
    if not np.isfinite(value) or value <= 1e-8:
        raise ValueError('Degenerate source q99')
    return value


def validate_seen_report(report):
    if (report.get('status') != 'ok' or report.get('categories') != 11 or report.get('groups_per_category') != 25
            or report.get('branches') != BRANCHES or report.get('test_b_labels_accessed') is not False):
        raise ValueError('Invalid seen-category report')
    return True


def _sample_indices(member, token_count):
    seed = int.from_bytes(hashlib.sha256(('36001:' + member).encode()).digest()[:8], 'little')
    return np.sort(np.random.default_rng(seed).choice(token_count, 50, replace=False))


def _rows_by_group(rows):
    result = {}
    for row in rows:
        result.setdefault(row['group'], []).append(row)
    for group in result:
        result[group] = sorted(result[group], key=lambda r: int(canonical_view(r['view'])))
        if len(result[group]) != 5:
            raise ValueError('Incomplete auxiliary group')
    return result


def _model(checkpoint, fold):
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    deps = adapter._import_official_modules()
    model, _ = adapter._build_model(deps, backbone='dinov3_vit_base_16', dropout=.4, reduce_dim=256,
        context_aware_recenter=True, mask_neighbor_size=0,
        weights_dir=ROOT / 'Dinomaly2_dinov3/backbones/weights')
    model.load_state_dict(state['model'], strict=True); del state
    model = model.to(f'cuda:{fold}').eval(); model.requires_grad_(False)
    return model, deps


def _bank(model, rows, input_size):
    import torch
    ds = trainer.RealIADSupervisedDataset(rows, input_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=DATA_LOADER_WORKERS, pin_memory=True)
    pieces=[]
    with torch.inference_mode():
        for images, masks, labels, categories, group_ids in loader:
            tokens, grid = __import__('scripts.v180_dinomaly3_dinov3_csig',fromlist=['_extract_patch_tokens'])._extract_patch_tokens(model.encoder, images.to(next(model.parameters()).device), [3,7,11])
            if grid != (input_size // 16, input_size // 16): raise ValueError('Auxiliary bank grid mismatch')
            for i,row in enumerate(rows[len(pieces)*0:len(pieces)*0]): pass
            offset = sum(x.shape[0] for x in pieces) // 50
            for i in range(len(images)):
                member=f"aux/{rows[offset+i]['group']}/{canonical_view(rows[offset+i]['view'])}.png"
                pieces.append(tokens[i,_sample_indices(member,int(tokens.shape[1]))].float().cpu().numpy())
    return np.concatenate(pieces,axis=0).astype(np.float16)


def _maps(model, rows, memory, input_size):
    import torch
    ds448=trainer.RealIADSupervisedDataset(rows,448); ds672=trainer.RealIADSupervisedDataset(rows,input_size)
    kw=dict(batch_size=8,shuffle=False,num_workers=DATA_LOADER_WORKERS,pin_memory=True)
    l448=torch.utils.data.DataLoader(ds448,**kw); l672=torch.utils.data.DataLoader(ds672,**kw)
    resident=ResidentCosineMemory(memory,device=next(model.parameters()).device,chunk_size=8192)
    refs=[]; mems=[]; masks=[]; labels=[]
    with torch.inference_mode():
        for base,high in zip(l448,l672):
            e,d=model(base[0].to(next(model.parameters()).device)); ref=trainer.differentiable_anomaly_map(e,d,448).float().cpu().numpy()[:,0]
            tok,grid=__import__('scripts.v180_dinomaly3_dinov3_csig',fromlist=['_extract_patch_tokens'])._extract_patch_tokens(model.encoder,high[0].to(next(model.parameters()).device),[3,7,11])
            mm=np.stack([resize_raw_map(x.reshape(grid)) for x in resident.distances(tok)])
            refs.extend(ref);mems.extend(mm);masks.extend((base[1][:,0].cpu().numpy()>.5));labels.extend(base[2].tolist())
    return np.asarray(refs,np.float32),np.asarray(mems,np.float32),masks,labels


def _fold(fold, selected, rows, input_size):
    import torch
    torch.cuda.set_device(fold);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model,deps=_model(Path(selected['checkpoint']),fold)
    by_group=_rows_by_group([r for r in rows if r['phase']=='train'])
    test_by_group=_rows_by_group([r for r in rows if r['phase']=='test'])
    held=set(selected['baseline']['metrics']['per_category'])
    details={}
    for category in sorted(held):
        cat_groups={g:tuple(str(canonical_view(r['view'])) for r in rs) for g,rs in by_group.items() if g.startswith(category+'/')}
        split=split_seen_groups(cat_groups)
        bank_rows=[row for g in split['bank_groups'] for row in by_group[g]]
        cal_rows=[row for g in split['calibration_groups'] for row in by_group[g]]
        eval_rows=[row for g in test_by_group if g.startswith(category+'/') for row in test_by_group[g]]
        memory=_bank(model,bank_rows,input_size)
        ref_cal,mem_cal,_,_=_maps(model,cal_rows,memory,input_size)
        rq,mq=source_q99(ref_cal),source_q99(mem_cal);scale=normal_quantile_scale(rq,mq)
        ref,mem,masks,labels=_maps(model,eval_rows,memory,input_size)
        metric={name:ScaledEvidenceMetrics() for name in BRANCHES}
        for i,row in enumerate(eval_rows):
            primary=primary_reference_score(ref[i]);fused=fuse_memory(ref[i],mem[i],scale=scale)
            for name,raw in [('reference',ref[i]),('category_memory',mem[i]),('fusion',fused)]:
                metric[name].add(category=category,group=row['group'],view=canonical_view(row['view']),label=int(row['label']),mask=masks[i],raw=raw,reference_view_score=primary)
        details[category]={'split':split,'calibration_q99':{'reference':rq,'memory':mq,'scale':scale},'metrics':{name:metric[name].finish() for name in BRANCHES}}
        print(f'v360_fold{fold}_{category}_done',flush=True)
    del model
    return {'fold':fold,'categories':len(details),'groups_per_category':80,'details':details}


def run(args):
    import torch
    output=Path(args.output_root).resolve()
    if not output.is_relative_to(ROOT/'outputs') or output.exists() or args.input_size not in SUPPORTED_INPUT_SIZES: raise ValueError('New output under outputs required')
    if not torch.cuda.is_available() or torch.cuda.device_count()<4: raise RuntimeError('Four GPUs required')
    rows=[json.loads(line) for line in INDEX.read_text().splitlines() if line.strip()]
    cats=sorted({r['category'] for r in rows}); assert len(cats)==11
    bundle=load_verified_bundle(ROOT/'outputs/v343_verified_longrun_20260911',GATE)
    output.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(_fold,fold,bundle['records'][fold]['selected'],rows,args.input_size) for fold in range(4)]
        folds=[f.result() for f in fs]
    folds.sort(key=lambda x:x['fold'])
    metrics={name:{} for name in BRANCHES}
    for name in BRANCHES:
        for fold in folds: metrics[name].update({cat:d['metrics'][name]['metrics']['per_category'][cat] for cat,d in folds[fold['fold']]['details'].items()})
    report={'status':'ok','version':'v360_seen_reference_validation_v1','input_size':args.input_size,'categories':11,'groups_per_category':25,'branches':BRANCHES,
            'folds':4,'test_b_labels_accessed':False,'development_only':True,'index_sha256':sha256_file(INDEX),
            'metrics':metrics,'folds_detail':folds,'scope':'Clean auxiliary seen-reference simulation; not B score or zero-shot evidence.'}
    validate_seen_report(report)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':'ok','output':str(output/'report.json')}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--input-size',type=int,choices=SUPPORTED_INPUT_SIZES,default=672);run(p.parse_args())
