"""Fresh clean-category training from a fixed Train-only unsupervised checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts.v276_i448_state_infer import extract_model_state
from scripts.v331_checkpoint_benchmark import compare_candidate
from scripts.v332_train_pilot import evaluate, partition_rows, training_loss, training_mode, validate_teacher

ROOT = Path.home() / 'csig'
SOURCE_ROOT = ROOT / 'outputs/v272_dinomaly2_dinov3_i448_ddp/evidence/v272_d3b_i448_ddp_seed27201'
SOURCE_SHA = 'e1b4cfc9c0b7f50c8ff738d586a45b3ec520e9c8d9ebaa5d8c5932696c0d3620'
TEACHER_ROOT = ROOT / 'outputs/v332_teacher_cache_verified_20260910'


def validate_initialization(payload: dict, digest: str) -> None:
    config = payload.get('immutable_config', {})
    if digest != SOURCE_SHA or config.get('route') != 'v272_dinomaly2_dinov3_i448_ddp':
        raise ValueError('Only the pinned unsupervised initializer is allowed')
    if config.get('crop_size') != 448 or config.get('reduce_dim') != 256 or config.get('backbone') != 'dinov3_vit_base_16':
        raise ValueError('Initializer architecture mismatch')
    extract_model_state(payload, expected_step=100000, expected_input_size=448, expected_dropout=.4)


def validate_clean_report(report: dict, clean_sha: str, rows: list[dict]) -> None:
    if (report.get('status') != 'ok' or report.get('do_not_reuse_supervised_weights') is not True
            or report.get('test_labels_accessed') is not False
            or report.get('policy') != 'exclude_all_target_categories_and_any_exact_overlap_object_group'
            or report.get('clean_index_sha256') != clean_sha or report.get('retained_records') != len(rows)):
        raise ValueError('Invalid or changed clean-data contract')
    categories = {r['category'] for r in rows}
    if categories != set(report['retained_categories']) or categories & set(report['excluded_categories']):
        raise ValueError('Excluded target categories entered clean training')


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not 0 < args.steps <= 1000:
        raise ValueError('This initial clean pilot is bounded to1000 steps')
    checkpoints = trainer.parse_checkpoint_steps(args.checkpoint_steps, args.steps)
    if args.steps == 1000 and checkpoints != (500, 1000):
        raise ValueError('Clean1000-step pilot requires500/1000 checkpoints')
    clean_root = Path(args.clean_root)
    output = Path(args.output_root) / f'fold{args.fold_id}'
    if output.exists():
        raise FileExistsError(output)
    report_path, index = clean_root / 'report.json', clean_root / 'clean_proxy_index.jsonl'
    data_report = json.loads(report_path.read_text(encoding='utf-8'))
    rows = trainer.load_proxy_rows(index)
    validate_clean_report(data_report, trainer.sha256_file(index), rows)
    folds_path = clean_root / 'category_folds.json'
    folds = json.loads(folds_path.read_text(encoding='utf-8'))
    if folds != trainer.build_category_folds(rows, 4):
        raise ValueError('Clean category fold manifest mismatch')
    heldout = set(next(f['categories'] for f in folds['folds'] if f['fold'] == args.fold_id))
    train_rows, dev_rows = partition_rows(rows, heldout, Path(data_report['approved_image_root']))
    torch.cuda.set_device(0); torch.set_num_threads(4)
    seed = 33601 + args.fold_id
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    checkpoint = SOURCE_ROOT / 'checkpoints/state_i100000.pt'
    source_sha = trainer.sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_initialization(payload, source_sha)
    del payload
    source_args = SimpleNamespace(source_run_root=SOURCE_ROOT, source_step=100000, input_size=448, source_dropout=.4,
                                  backbone='dinov3_vit_base_16', reduce_dim=256,
                                  weights_dir=ROOT / 'Dinomaly2_dinov3/backbones/weights')
    model, deps, counts = trainer._load_source_model(source_args, torch.device('cuda:0'))
    anchor_base = adapter.CsigZipDataset(ROOT / 'Train.zip', 'Train', adapter._build_transform(448, 448), include_metadata=True)
    teacher_manifest = json.loads((TEACHER_ROOT / 'teacher_manifest.json').read_text())
    teacher_files = sorted(TEACHER_ROOT.rglob('*.npy'))
    train_sha = trainer.sha256_file(ROOT / 'Train.zip')
    validate_teacher(teacher_manifest, {'source_checkpoint_sha256': source_sha, 'source_dropout': .4,
                                        'held_out_categories': list(heldout)}, anchor_base.records,
                      [p.relative_to(TEACHER_ROOT).as_posix() for p in teacher_files], train_sha)
    tree = hashlib.sha256()
    for path in teacher_files:
        values = np.load(path, allow_pickle=False)
        if values.shape != (28, 28) or not np.isfinite(values).all():
            raise ValueError('Invalid teacher map')
        tree.update(path.relative_to(TEACHER_ROOT).as_posix().encode())
        tree.update(bytes.fromhex(trainer.sha256_file(path)))
    if teacher_manifest.get('teacher_tree_sha256') != tree.hexdigest():
        raise ValueError('Teacher content changed')
    output.mkdir(parents=True)
    config = dict(version='v336_clean_train_v1', fold_id=args.fold_id, partition_key='category',
                  held_out_categories=sorted(heldout), training_categories=sorted({r['category'] for r in train_rows}),
                  excluded_target_categories=data_report['excluded_categories'], input_size=448, source_dropout=.4,
                  source_checkpoint=str(checkpoint), source_checkpoint_sha256=source_sha,
                  prediction_scale=1.0, clean_index_sha256=trainer.sha256_file(index), clean_report_sha256=trainer.sha256_file(report_path),
                  fold_manifest_sha256=trainer.sha256_file(folds_path), train_zip_sha256=train_sha,
                  teacher_manifest_sha256=trainer.sha256_file(TEACHER_ROOT / 'teacher_manifest.json'),
                  seed=seed, batch_size=4, accumulation_steps=2, target_steps=args.steps, checkpoint_steps=list(checkpoints),
                  bottleneck_lr=5e-5, decoder_lr=2e-5, weight_decay=1e-4, amp=False, encoder_mode='eval',
                  normal_penalty_space='raw_distance', normal_mean_budget=.006, normal_quantile_budget=.06,
                  driver_sha256=trainer.sha256_file(Path(__file__)),
                  dependency_hashes={name: trainer.sha256_file(Path(__file__).parent / name) for name in
                    ['v332_reachable_normal.py','v332_train_pilot.py','v330_raw_response.py','v330_replay.py','v283_supervised_dinomaly_longtrain.py']})
    trainer.atomic_write_json(output / 'immutable_config.json', config)
    step, started = 0, time.time()
    try:
        sampler = trainer.make_balanced_sampler(train_rows, seed)
        generator = torch.Generator().manual_seed(seed + 17)
        kwargs = dict(batch_size=4, num_workers=2, pin_memory=True, persistent_workers=True)
        train_loader = torch.utils.data.DataLoader(trainer.RealIADSupervisedDataset(train_rows, 448), sampler=sampler, drop_last=True, **kwargs)
        anchor_loader = torch.utils.data.DataLoader(trainer.AnchorTeacherDataset(anchor_base, TEACHER_ROOT), shuffle=True,
                                                    generator=generator, drop_last=True, **kwargs)
        dev_loader = torch.utils.data.DataLoader(trainer.RealIADSupervisedDataset(dev_rows, 448), shuffle=False, **kwargs)
        initial = evaluate(model, dev_rows, dev_loader, 1.0)
        trainer.atomic_write_json(output / 'baseline.json', initial)
        optimizer = deps['StableAdamW']([{'params':model.bottleneck.parameters(),'lr':5e-5},
                                        {'params':model.decoder.parameters(),'lr':2e-5}], betas=(.9,.999),weight_decay=1e-4,amsgrad=False,eps=1e-10)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: trainer._learning_rate_lambda(s, args.steps))
        train_it, anchor_it = iter(train_loader), iter(anchor_loader)
        training_mode(model)
        results = []
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.0
            for _ in range(2):
                sup, train_it = trainer._next_loader_batch(train_loader, train_it)
                normal, anchor_it = trainer._next_loader_batch(anchor_loader, anchor_it)
                loss = training_loss(model, tuple(x.cuda(non_blocking=True) for x in sup[:2]),
                                      tuple(x.cuda(non_blocking=True) for x in normal[:2]), 'raw_distance')
                (loss / 2).backward(); mean_loss += float(loss.detach()) / 2
            grad = torch.nn.utils.clip_grad_norm_(list(model.bottleneck.parameters())+list(model.decoder.parameters()),1.0)
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite clean training gradient')
            optimizer.step();scheduler.step()
            if step<=3 or step%25==0:print(f'clean fold{args.fold_id} step={step}/{args.steps} loss={mean_loss:.6f} minutes={(time.time()-started)/60:.2f}',flush=True)
            if step in checkpoints:
                path=output/'checkpoints'/f'state_i{step:06d}.pt'
                trainer.atomic_torch_save(torch,dict(version='v336_clean_train_v1',step=step,model=model.state_dict(),
                  optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),immutable_config=config,
                  python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all(),sampler_rng=sampler.generator.get_state(),anchor_rng=generator.get_state()),path)
                evaluated=evaluate(model,dev_rows,dev_loader,1.0)
                result=dict(status='ok',development_only=True,fold_id=args.fold_id,step=step,evaluation=evaluated,
                    comparison=compare_candidate(initial['metrics'],evaluated['metrics']),checkpoint=str(path),checkpoint_sha256=trainer.sha256_file(path))
                trainer.atomic_write_json(output/f'validation_i{step:06d}.json',result)
                results.append(result);training_mode(model)
                print(json.dumps({'fold':args.fold_id,'step':step,'comparison':result['comparison']}),flush=True)
        result=dict(status='ok',pilot_only=True,clean_training=True,fold_id=args.fold_id,steps=args.steps,
                    parameter_counts=counts,config=config,validation_steps=[r['step'] for r in results],elapsed_minutes=(time.time()-started)/60)
        trainer.atomic_write_json(output/'report.json',result)
        return result
    except Exception as error:
        trainer.atomic_write_json(output/'failure.json',dict(status='failed',step=step,error=repr(error)))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold-id',type=int,choices=range(4),required=True)
    parser.add_argument('--clean-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--checkpoint-steps',default='500,1000')
    return parser


if __name__=='__main__':
    run(build_parser().parse_args())
