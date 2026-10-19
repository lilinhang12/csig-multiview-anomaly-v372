"""Bounded paired fine-tuning pilot; never reads competition test images or labels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import v182_dinomaly2_dinov3_csig_official as adapter
from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts import v330_replay as replay
from scripts import v332_reachable_normal as loss_module
from scripts.v330_raw_response import checked_raw
from scripts.v331_checkpoint_benchmark import compare_candidate

ROOT = Path.home() / 'csig'


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if (args.fold_id not in (0, 2) or args.arm not in loss_module.ARMS or not 0 < args.steps <= 1000
            or args.batch_size <= 0 or args.accumulation_steps <= 0 or args.num_workers < 0):
        raise ValueError('Invalid bounded pilot arguments')
    steps = trainer.parse_checkpoint_steps(args.checkpoint_steps, args.steps)
    if args.steps == 1000 and steps != (500, 1000):
        raise ValueError('Full pilots require exactly checkpoint-steps 500,1000')
    return steps


def validate_source_report(report: Mapping[str, Any], fold: int) -> dict[str, Any]:
    if (report.get('status') != 'ok' or report.get('phase') != 'test' or report.get('fold_id') != fold
            or report.get('saved', 0) <= 0 or report.get('saved') != report.get('expected')):
        raise ValueError('A completed matching source report is required')
    contract = copy.deepcopy(report['contract'])
    if (contract.get('fold_id') != fold or not contract.get('checkpoint') or contract.get('step', 0) <= 0
            or not _digest(contract.get('checkpoint_sha256'))
            or contract.get('expected_checkpoint_sha256') != contract['checkpoint_sha256']):
        raise ValueError('Source checkpoint identity is invalid')
    scale = float(contract.get('score_scale', 0))
    cats = contract.get('held_out_categories', [])
    hashes = contract.get('pipeline_hashes', {})
    if (not math.isfinite(scale) or not 0 < scale <= 1 or not cats or len(cats) != len(set(cats))
            or not hashes or not all(_digest(h) for h in hashes.values())):
        raise ValueError('Invalid source scale, held-out categories or pipeline hashes')
    if fold == 0 and contract.get('selection_kind') != 'full_pixel_checkpoint_comparison':
        raise ValueError('Fold0 must use the promoted checkpoint selection')
    return contract


def partition_rows(rows: Sequence[Mapping[str, Any]], held_out: set[str], approved_root: Path) -> tuple[list, list]:
    split = trainer.partition_rows_for_fold(rows, held_out=held_out)
    for row in split['train'] + split['validation']:
        if not row.get('image_path'):
            raise ValueError('Missing development image path')
    loss_module.validate_training_rows(split['train'], held_out, approved_root)
    validation = replay.development_rows(rows, list(held_out), approved_root=approved_root)
    if {r['category'] for r in validation} != held_out:
        raise ValueError('Incomplete held-out category coverage')
    return split['train'], validation


def validate_teacher(manifest: Mapping[str, Any], source_config: Mapping[str, Any], records: Sequence[Any],
                      keys: Sequence[str], train_sha: str) -> None:
    if manifest.get('train_zip_sha256') != train_sha:
        raise ValueError('Train.zip identity missing or changed in teacher train_zip_sha256')
    if (manifest.get('status') != 'ok' or manifest.get('saved') != 5000 or len(records) != 5000
            or manifest.get('source_checkpoint_sha256') != source_config['source_checkpoint_sha256']
            or manifest.get('input_size') != 448 or manifest.get('patch_grid') != 28
            or manifest.get('source_dropout') != source_config['source_dropout']):
        raise ValueError('Teacher source ancestry, configuration or coverage mismatch')
    expected: set[str] = set()
    views: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        if (record.category in set(source_config['held_out_categories']) or record.view not in {'0', '1', '2', '3', '4'}
                or record.member != f'Train/{record.category}/{record.sample}/{record.view}.png'):
            raise ValueError('Invalid Train member or held-out category overlap')
        key = f'{record.category}/{record.sample}/{record.view}.npy'
        if key in expected:
            raise ValueError('Duplicate Train view')
        expected.add(key)
        views[(record.category, record.sample)].add(record.view)
    if (set(keys) != expected or len(keys) != len(expected)
            or any(v != {'0', '1', '2', '3', '4'} for v in views.values())):
        raise ValueError('Teacher key or five-view coverage mismatch')


def immutable_config(args: argparse.Namespace, contract: Mapping[str, Any], source: Mapping[str, Any],
                      hashes: Mapping[str, str]) -> dict[str, Any]:
    validate_args(args)
    if (source.get('fold_id') != args.fold_id or source.get('partition_key') != 'category'
            or source.get('input_size') != 448 or set(source.get('held_out_categories', [])) != set(contract['held_out_categories'])
            or not math.isfinite(float(source.get('source_dropout', float('nan'))))
            or not _digest(source.get('source_checkpoint_sha256'))):
        raise ValueError('Source immutable config mismatch')
    return {
        'version': 'v332_paired_pilot_v1', 'pilot_only': True, 'fold_id': args.fold_id,
        'arm': args.arm, 'seed': loss_module.paired_seed(args.fold_id, args.arm),
        'partition_key': 'category', 'held_out_categories': sorted(contract['held_out_categories']),
        'input_size': 448, 'source_dropout': source['source_dropout'],
        'source_checkpoint': contract['checkpoint'], 'source_checkpoint_sha256': contract['checkpoint_sha256'],
        'original_source_checkpoint_sha256': source['source_checkpoint_sha256'],
        'prediction_scale': contract['score_scale'], 'source_pipeline_hashes': copy.deepcopy(contract['pipeline_hashes']),
        'steps': args.steps, 'checkpoint_steps': list(validate_args(args)), 'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps, 'num_workers': args.num_workers,
        'bottleneck_lr': 5e-5, 'decoder_lr': 2e-5, 'betas': [.9, .999], 'weight_decay': 1e-4,
        'eps': 1e-10, 'amsgrad': False, 'clip_norm': 1.0, 'distillation_weight': .5,
        'normal_loss_weight': .5, 'normal_mean_budget': .006, 'normal_quantile_budget': .060,
        'encoder_mode': 'eval', 'amp': False, **dict(hashes),
    }


def validate_pair(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if {k: v for k, v in first.items() if k != 'arm'} != {k: v for k, v in second.items() if k != 'arm'}:
        raise ValueError('Paired arms have different source/data/seed/hyperparameters')


def training_mode(model: Any) -> None:
    model.train()
    model.encoder.eval()


def training_loss(model: Any, supervised: tuple, anchor: tuple, arm: str) -> Any:
    import torch

    images, masks = supervised[:2]
    anchors, teacher = anchor[:2]
    encoded, decoded = model(torch.cat([images, anchors], dim=0))
    n = len(images)
    se, sd = [v[:n] for v in encoded], [v[:n] for v in decoded]
    ae, ad = [v[n:] for v in encoded], [v[n:] for v in decoded]
    loss = (loss_module.supervised_losses(se, sd, masks, out_size=448, penalty_space=arm)['total']
            + trainer.global_cosine_loss(ae, ad)
            + .5 * trainer.anchor_distillation_loss(trainer.differentiable_anomaly_map(ae, ad, 28), teacher))
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError('Nonfinite paired pilot loss')
    return loss


def evaluate(model: Any, rows: Sequence[Mapping[str, Any]], loader: Any, scale: float) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    model.eval()
    metrics, seen = replay.ExportMetrics(), 0
    with torch.inference_mode():
        for batch in loader:
            encoded, decoded = model(batch[0].to(device, non_blocking=True))
            maps = checked_raw(trainer.differentiable_anomaly_map(encoded, decoded, 448).float().cpu().numpy()[:, 0] * scale).astype(np.float32)
            for i, raw in enumerate(maps):
                if seen >= len(rows):
                    raise ValueError('Incomplete development row alignment')
                row = rows[seen]
                metrics.add(category=row['category'], group=row['group'], view=str(row['view']), label=int(row['label']),
                            mask=batch[1][i, 0].cpu().numpy() > .5, legacy=np.zeros_like(raw), raw=raw)
                seen += 1
    if seen != len(rows):
        raise ValueError('Incomplete development evaluation')
    report = metrics.finish()
    return {**{k: report[k] for k in ('views', 'pixel_count', 'categories', 'groups')},
            'metrics': report['corrected'], 'legacy_slot': 'unused_zero_placeholder', 'prediction_scale': scale}


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoints = validate_args(args)
    output = Path(args.output_root) / args.arm / f'fold{args.fold_id}'
    output.mkdir(parents=True, exist_ok=False)
    step, started = 0, time.time()
    try:
        source = json.loads(Path(args.source_report).read_text(encoding='utf-8'))
        contract = validate_source_report(source, args.fold_id)
        import torch

        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        seed = loss_module.paired_seed(args.fold_id, args.arm)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        payload = torch.load(contract['checkpoint'], map_location='cpu', weights_only=False)
        source_config = payload['immutable_config']
        if payload['step'] != contract['step']:
            raise ValueError('Source checkpoint step differs from report')
        del payload
        proxy, train_zip = trainer.DEFAULT_PROXY_INDEX, ROOT / 'Train.zip'
        rows = trainer.load_proxy_rows(proxy)
        train_rows, dev_rows = partition_rows(rows, set(contract['held_out_categories']), proxy.parent / 'dataset')
        teacher_root = Path(args.teacher_root).resolve()
        teacher_path = teacher_root / 'teacher_manifest.json'
        manifest = json.loads(teacher_path.read_text(encoding='utf-8'))
        train_sha = trainer.sha256_file(train_zip)
        anchor_base = adapter.CsigZipDataset(train_zip, 'Train', adapter._build_transform(448, 448), include_metadata=True)
        teacher_files = sorted(teacher_root.rglob('*.npy'))
        validate_teacher(manifest, source_config, anchor_base.records,
                          [p.relative_to(teacher_root).as_posix() for p in teacher_files], train_sha)
        tree = hashlib.sha256()
        for path in teacher_files:
            values = np.load(path, allow_pickle=False)
            if values.shape != (28, 28) or not np.isfinite(values).all():
                raise ValueError('Invalid teacher map')
            tree.update(path.relative_to(teacher_root).as_posix().encode())
            tree.update(bytes.fromhex(trainer.sha256_file(path)))
        if manifest.get('teacher_tree_sha256') != tree.hexdigest():
            raise ValueError('Teacher content tree identity mismatch')
        hashes = {'source_report_sha256': trainer.sha256_file(args.source_report), 'proxy_index_sha256': trainer.sha256_file(proxy),
                  'train_zip_sha256': train_sha, 'teacher_manifest_sha256': trainer.sha256_file(teacher_path),
                  'loss_module_sha256': trainer.sha256_file(Path(loss_module.__file__)),
                  'driver_sha256': trainer.sha256_file(Path(__file__)), 'trainer_sha256': trainer.sha256_file(Path(trainer.__file__))}
        config = immutable_config(args, contract, source_config, hashes)
        trainer.atomic_write_json(output / 'immutable_config.json', config)
        other_arm = next(arm for arm in loss_module.ARMS if arm != args.arm)
        other_config = Path(args.output_root) / other_arm / f'fold{args.fold_id}' / 'immutable_config.json'
        if other_config.exists():
            validate_pair(json.loads(other_config.read_text()), config)
        model, deps = replay.load_model(contract)
        counts = trainer.configure_trainable_parameters(model)
        train_ds = trainer.RealIADSupervisedDataset(train_rows, input_size=448)
        sampler = trainer.make_balanced_sampler(train_rows, seed)
        kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                      persistent_workers=args.num_workers > 0)
        train_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, drop_last=True, **kwargs)
        generator = torch.Generator().manual_seed(seed + 17)
        anchor_loader = torch.utils.data.DataLoader(trainer.AnchorTeacherDataset(anchor_base, teacher_root),
                                                    shuffle=True, generator=generator, drop_last=True, **kwargs)
        dev_loader = torch.utils.data.DataLoader(trainer.RealIADSupervisedDataset(dev_rows, 448), shuffle=False, **kwargs)
        if not len(train_loader) or not len(anchor_loader):
            raise ValueError('Empty training loader')
        baseline = evaluate(model, dev_rows, dev_loader, contract['score_scale'])
        trainer.atomic_write_json(output / 'baseline.json', baseline)
        optimizer = deps['StableAdamW']([
            {'params': model.bottleneck.parameters(), 'lr': config['bottleneck_lr']},
            {'params': model.decoder.parameters(), 'lr': config['decoder_lr']}],
            betas=tuple(config['betas']), weight_decay=config['weight_decay'], amsgrad=False, eps=1e-10)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: trainer._learning_rate_lambda(s, args.steps))
        train_it, anchor_it = iter(train_loader), iter(anchor_loader)
        training_mode(model)
        reports = []
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.0
            for _ in range(args.accumulation_steps):
                supervised, train_it = trainer._next_loader_batch(train_loader, train_it)
                anchor, anchor_it = trainer._next_loader_batch(anchor_loader, anchor_it)
                supervised_gpu = tuple(v.cuda(non_blocking=True) for v in supervised[:2])
                anchor_gpu = tuple(v.cuda(non_blocking=True) for v in anchor[:2])
                loss = training_loss(model, supervised_gpu, anchor_gpu, args.arm)
                (loss / args.accumulation_steps).backward()
                mean_loss += float(loss.detach()) / args.accumulation_steps
            grad = torch.nn.utils.clip_grad_norm_(list(model.bottleneck.parameters()) + list(model.decoder.parameters()), 1.0)
            if not bool(torch.isfinite(grad).item()):
                raise FloatingPointError('Nonfinite pilot gradient')
            optimizer.step(); scheduler.step()
            if step <= 3 or step % 25 == 0:
                print(f'pilot {args.arm} fold{args.fold_id} step={step}/{args.steps} loss={mean_loss:.6f} grad={float(grad):.6f} minutes={(time.time()-started)/60:.2f}', flush=True)
            if step in checkpoints:
                checkpoint = output / 'checkpoints' / f'state_i{step:06d}.pt'
                state = dict(version='v332_paired_pilot_v1', model=model.state_dict(), optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), immutable_config=config, step=step,
                             python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all(), sampler_rng=sampler.generator.get_state(),
                             anchor_rng=generator.get_state())
                trainer.atomic_torch_save(torch, state, checkpoint)
                current = evaluate(model, dev_rows, dev_loader, contract['score_scale'])
                report = dict(status='ok', development_only=True, fold_id=args.fold_id, arm=args.arm, step=step,
                              evaluation=current, comparison=compare_candidate(baseline['metrics'], current['metrics']),
                              checkpoint=str(checkpoint), checkpoint_sha256=trainer.sha256_file(checkpoint))
                trainer.atomic_write_json(output / f'validation_i{step:06d}.json', report)
                reports.append(report)
                print(json.dumps({'fold': args.fold_id, 'arm': args.arm, 'step': step, 'comparison': report['comparison']}), flush=True)
                training_mode(model)
        result = dict(status='ok', pilot_only=True, fold_id=args.fold_id, arm=args.arm, steps=args.steps,
                      parameter_counts=counts, source_config=config, reports=[r['step'] for r in reports],
                      elapsed_minutes=(time.time() - started) / 60)
        trainer.atomic_write_json(output / 'report.json', result)
        return result
    except Exception as error:
        trainer.atomic_write_json(output / 'failure.json', {'status': 'failed', 'step': step, 'error': repr(error)})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold-id', type=int, choices=(0, 2), required=True)
    parser.add_argument('--arm', choices=loss_module.ARMS, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--teacher-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--checkpoint-steps', default='500,1000')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation-steps', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
