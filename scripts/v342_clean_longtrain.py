"""Bounded clean finetuning from a replay-attested 1k model; never opens Test_B.

Run with python -m scripts.v342_clean_longtrain --fold-id 0 --gate GATE
--output-root NEW_VERSION. A smoke run additionally needs --steps 2 --checkpoint-steps 2.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import v283_supervised_dinomaly_longtrain as trainer
from scripts import v332_train_pilot as pilot
from scripts import v336_clean_train as clean
from scripts import v339_verify_clean_metrics as replay
from scripts import v340_clean_submission as submission
from scripts.v337_clean_gate import _validate_evaluation

META = dict(version='v342_clean_longtrain_v1', stage='clean_long_finetuning',
            development_only=True, pilot_only=False, clean_training=True, exact_resume=False,
            scope='clean auxiliary category-held-out finetune selection; not B score or global optimum',
            test_images_accessed=False, test_labels_accessed=False)


def validate_args(args):
    if type(args.fold_id) is not int or args.fold_id not in range(4) or type(args.steps) is not int or not 0 < args.steps <= 5000:
        raise ValueError('Four folds and 1..5000 additional optimizer steps only')
    steps = trainer.parse_checkpoint_steps(args.checkpoint_steps, args.steps)
    if args.steps == 5000 and steps != (1000, 2000, 3000, 5000):
        raise ValueError('Full finetuning requires exactly 1000,2000,3000,5000')
    return steps


def validate_source(selected, payload, digest, fold):
    source = selected.get('immutable_config', {})
    required = dict(version='v336_clean_train_v1', fold_id=fold, partition_key='category',
                    input_size=448, source_dropout=.4, prediction_scale=1.0,
                    source_checkpoint_sha256=clean.SOURCE_SHA, target_steps=1000,
                    checkpoint_steps=[500, 1000], amp=False, encoder_mode='eval', normal_penalty_space='raw_distance')
    if (selected.get('status') != 'ok' or selected.get('development_only') is not True
            or type(selected.get('fold_id')) is not int or selected['fold_id'] != fold
            or type(selected.get('step')) is not int or selected['step'] != 1000
            or not pilot._digest(digest) or digest != selected.get('checkpoint_sha256')
            or payload.get('version') != 'v336_clean_train_v1' or type(payload.get('step')) is not int
            or payload['step'] != selected['step'] or payload.get('immutable_config') != source or not payload.get('model')
            or any(source.get(k) != v for k, v in required.items())
            or Path(source.get('source_checkpoint', '')).resolve() != (clean.SOURCE_ROOT / 'checkpoints/state_i100000.pt').resolve()):
        raise ValueError('Selected clean 1000-step checkpoint identity/config mismatch')
    held, trained, excluded = (set(source.get(k, [])) for k in
                              ('held_out_categories', 'training_categories', 'excluded_target_categories'))
    if not held or not trained or held != set(selected.get('held_out_categories', [])) or held & trained or (held | trained) & excluded:
        raise ValueError('Selected clean category isolation mismatch')


def code_hashes():
    return {**replay.replay_hashes(), **{name: trainer.sha256_file(Path(__file__).with_name(name))
                                      for name in ('v342_clean_longtrain.py', 'v340_clean_submission.py')}}


def auxiliary_file_hashes(rows, approved_root):
    approved = Path(approved_root).resolve()
    paths = set()
    for row in rows:
        if not row.get('image_path'):
            raise ValueError('Missing approved auxiliary image')
        for key in ('image_path', 'mask_path'):
            if not row.get(key):
                continue
            path = Path(row[key]).absolute()
            if not path.resolve().is_relative_to(approved) or 'test_b' in str(path.resolve()).lower():
                raise ValueError('Auxiliary file is outside the approved source')
            paths.add(path)
    if not paths:
        raise ValueError('Empty auxiliary content manifest')
    return {str(path): trainer.sha256_file(path) for path in sorted(paths)}


def runtime_source_hashes(deps, model):
    paths = set()
    for item in [*deps.values(), *[type(module) for module in model.modules()]]:
        try:
            source = inspect.getsourcefile(item)
        except TypeError:
            continue
        if source:
            paths.add(Path(source).resolve())
    optimizer_source = inspect.getsourcefile(deps['StableAdamW'])
    if not optimizer_source:
        raise ValueError('Cannot identify actual optimizer implementation')
    paths.add(Path(optimizer_source).resolve())
    repo = (clean.ROOT / 'Dinomaly2_dinov3').resolve()
    if repo.is_dir():
        paths.update(p.resolve() for p in repo.rglob('*.py'))
    if not paths:
        raise ValueError('Empty runtime source manifest')
    return {str(path): trainer.sha256_file(path) for path in sorted(paths)}


def verify_materialized_inputs(config):
    for field in ('auxiliary_file_hashes', 'runtime_source_hashes'):
        expected = config.get(field)
        if not expected:
            raise ValueError(f'Missing {field} content manifest')
        for name, digest in expected.items():
            path = Path(name)
            if not path.is_file() or trainer.sha256_file(path) != digest:
                raise ValueError(f'{field} changed: {name}')


def immutable_config(args, selected, hashes, approved_root):
    source = selected['immutable_config']
    return dict(META, fold_id=args.fold_id, source_gate=str(args.gate), source_checkpoint=selected['checkpoint'],
                source_step=selected['step'], source_immutable_config=copy.deepcopy(source),
                partition_key='category', held_out_categories=list(source['held_out_categories']),
                training_categories=list(source['training_categories']), excluded_target_categories=list(source['excluded_target_categories']),
                approved_image_root=str(approved_root), input_size=448, source_dropout=.4, prediction_scale=1.0,
                seed=34201 + args.fold_id, optimizer_reset=True, rng_reset=True, batch_size=4, accumulation_steps=2,
                target_steps=args.steps, checkpoint_steps=list(validate_args(args)), num_workers=2,
                bottleneck_lr=2.5e-5, decoder_lr=1e-5, weight_decay=1e-4, betas=[.9, .999], eps=1e-10,
                amsgrad=False, scheduler='LambdaLR_warmup_cosine_floor_0.05', warmup_steps=100, clip_norm=1.0,
                amp=False, precision='fp32', encoder_mode='eval', normal_penalty_space='raw_distance',
                normal_mean_budget=.006, normal_quantile_budget=.06, normal_loss_weight=.5,
                normal_train_cosine_weight=1.0, distillation_weight=.5, code_hashes=code_hashes(), **hashes)


def bind_baseline(config, selected, evaluated):
    if (config['selected_source_checkpoint_sha256'] != selected['checkpoint_sha256']
            or config['source_immutable_config'] != selected['immutable_config'] or config['source_step'] != selected['step']):
        raise ValueError('Baseline initializer binding mismatch')
    _validate_evaluation(evaluated, config['held_out_categories'], selected['evaluation']['views'])
    drift = replay.evaluation_drift(selected['evaluation'], evaluated)
    if drift > 1e-6:
        raise ValueError(f'Selected initializer baseline reproduction drift={drift}')
    return dict(META, immutable_config=copy.deepcopy(config), evaluation=copy.deepcopy(evaluated), max_abs_drift=drift)


def validate_baseline(baseline, config, selected):
    if baseline != bind_baseline(config, selected, baseline['evaluation']):
        raise ValueError('Baseline payload/config changed')


def progress(config, baseline_sha, step):
    return dict(META, fold_id=config['fold_id'], step=step, relative_additional_step=step,
                total_optimizer_step=config['source_step'] + step, baseline_sha256=baseline_sha)


def candidate(config, baseline_sha, step, checkpoint, digest, baseline, evaluated):
    _validate_evaluation(evaluated, config['held_out_categories'], baseline['views'])
    comparison = clean.compare_candidate(baseline['metrics'], evaluated['metrics'])
    return dict(progress(config, baseline_sha, step), status='ok', immutable_config=config,
                checkpoint=str(checkpoint), checkpoint_sha256=digest, evaluation=evaluated, comparison=comparison,
                eligible=step == 0 or comparison['passed'], fallback=step == 0)


def preflight(args):
    gate_sha = trainer.sha256_file(args.gate)
    gate = submission.validate_gate(args.gate)
    selected, = [s for s in gate['selected_reports'] if s['fold_id'] == args.fold_id]
    root, source = Path(gate['clean_root']), selected['immutable_config']
    folder = Path(gate['training_root']) / f'fold{args.fold_id}'
    if selected['step'] != 1000 or Path(selected['checkpoint']).resolve() != (folder / 'checkpoints/state_i001000.pt').resolve():
        raise ValueError('Gate must select the matching clean 1000-step checkpoint')
    files = dict(source_gate_sha256=args.gate, selected_source_checkpoint_sha256=Path(selected['checkpoint']),
                 source_immutable_config_sha256=folder / 'immutable_config.json',
                 original_source_checkpoint_sha256=clean.SOURCE_ROOT / 'checkpoints/state_i100000.pt',
                 clean_index_sha256=root / 'clean_proxy_index.jsonl', clean_report_sha256=root / 'report.json',
                 fold_manifest_sha256=root / 'category_folds.json', train_zip_sha256=clean.ROOT / 'Train.zip',
                 teacher_manifest_sha256=clean.TEACHER_ROOT / 'teacher_manifest.json')
    hashes = {k: trainer.sha256_file(p) for k, p in files.items()}
    expected = dict(source_gate_sha256=gate_sha, selected_source_checkpoint_sha256=selected['checkpoint_sha256'],
                    source_immutable_config_sha256=selected['immutable_config_sha256'], original_source_checkpoint_sha256=clean.SOURCE_SHA)
    expected.update({k: source[k] for k in ('clean_index_sha256', 'clean_report_sha256', 'fold_manifest_sha256',
                                          'train_zip_sha256', 'teacher_manifest_sha256')})
    if hashes != expected or json.loads(files['source_immutable_config_sha256'].read_text(encoding='utf-8')) != source:
        raise ValueError('Gate/source/clean/Train/teacher input identity changed')
    report = json.loads(files['clean_report_sha256'].read_text(encoding='utf-8'))
    rows = trainer.load_proxy_rows(files['clean_index_sha256'])
    clean.validate_clean_report(report, hashes['clean_index_sha256'], rows)
    folds = json.loads(files['fold_manifest_sha256'].read_text(encoding='utf-8'))
    if folds != trainer.build_category_folds(rows, 4):
        raise ValueError('Clean fold manifest changed')
    held = set(next(f['categories'] for f in folds['folds'] if f['fold'] == args.fold_id))
    approved = Path(report['approved_image_root'])
    training, development = pilot.partition_rows(rows, held, approved)
    if (held != set(source['held_out_categories']) or set(source['training_categories']) != {r['category'] for r in training}
            or set(source['excluded_target_categories']) != set(report['excluded_categories'])
            or len(development) != selected['evaluation']['views']):
        raise ValueError('Clean source partition or evaluation coverage changed')
    return selected, training, development, approved, files, hashes


def teacher_tree():
    tree = hashlib.sha256()
    for path in sorted(clean.TEACHER_ROOT.rglob('*.npy')):
        if not path.resolve().is_relative_to(clean.TEACHER_ROOT.resolve()):
            raise ValueError('Teacher path escaped the pinned cache')
        values = np.load(path, allow_pickle=False)
        if values.shape != (28, 28) or not np.isfinite(values).all():
            raise ValueError('Invalid teacher map')
        tree.update(path.relative_to(clean.TEACHER_ROOT).as_posix().encode())
        tree.update(bytes.fromhex(trainer.sha256_file(path)))
    return tree.hexdigest()


def make_optimizer(model, deps, steps):
    import torch
    optimizer = deps['StableAdamW']([{'params': model.bottleneck.parameters(), 'lr': 2.5e-5},
                                   {'params': model.decoder.parameters(), 'lr': 1e-5}],
                                  betas=(.9, .999), weight_decay=1e-4, eps=1e-10, amsgrad=False)
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: trainer._learning_rate_lambda(s, steps, 100))


def optimizer_step(model, optimizer, scheduler, train_loader, anchor_loader, train_it, anchor_it):
    import torch
    optimizer.zero_grad(set_to_none=True)
    device, mean_loss = next(model.parameters()).device, 0.0
    for _ in range(2):
        supervised, train_it = trainer._next_loader_batch(train_loader, train_it)
        anchor, anchor_it = trainer._next_loader_batch(anchor_loader, anchor_it)
        loss = pilot.training_loss(model, tuple(x.to(device, non_blocking=True) for x in supervised[:2]),
                                   tuple(x.to(device, non_blocking=True) for x in anchor[:2]), 'raw_distance')
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Nonfinite clean finetuning loss')
        (loss / 2).backward()
        mean_loss += float(loss.detach()) / 2
    grad = torch.nn.utils.clip_grad_norm_(list(model.bottleneck.parameters()) + list(model.decoder.parameters()), 1.0)
    if not torch.isfinite(grad).item():
        raise FloatingPointError('Nonfinite clean finetuning gradient')
    optimizer.step()
    scheduler.step()
    return train_it, anchor_it, mean_loss, float(grad)


def verify_inputs(files, hashes, config, selected, output, config_sha, baseline_sha):
    if ({k: trainer.sha256_file(p) for k, p in files.items()} != hashes or code_hashes() != config['code_hashes']
            or trainer.sha256_file(output / 'immutable_config.json') != config_sha
            or trainer.sha256_file(output / 'baseline.json') != baseline_sha or teacher_tree() != config['teacher_tree_sha256']):
        raise ValueError('Immutable inputs/code/baseline changed during finetuning')
    verify_materialized_inputs(config)
    validate_baseline(json.loads((output / 'baseline.json').read_text(encoding='utf-8')), config, selected)


def checkpoint_state(model, optimizer, scheduler, sampler, generator, config, baseline_sha, step):
    import torch
    return dict(progress(config, baseline_sha, step), model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), immutable_config=config, python_rng=random.getstate(),
                numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == 'cuda' else [],
                sampler_rng=sampler.generator.get_state(), anchor_rng=generator.get_state())


def run(args):
    checkpoints = validate_args(args)
    output = Path(args.output_root) / f'fold{args.fold_id}'
    output.mkdir(parents=True, exist_ok=False)
    step, started, baseline_sha, config = 0, time.monotonic(), None, None
    try:
        selected, training, development, approved, files, hashes = preflight(args)
        import torch
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        source = torch.load(files['original_source_checkpoint_sha256'], map_location='cpu', weights_only=False)
        clean.validate_initialization(source, hashes['original_source_checkpoint_sha256'])
        del source
        source_args = SimpleNamespace(source_run_root=clean.SOURCE_ROOT, source_step=100000, input_size=448, source_dropout=.4,
                                      backbone='dinov3_vit_base_16', reduce_dim=256, weights_dir=clean.ROOT / 'Dinomaly2_dinov3/backbones/weights')
        model, deps, counts = trainer._load_source_model(source_args, torch.device('cuda:0'))
        payload = torch.load(files['selected_source_checkpoint_sha256'], map_location='cpu', weights_only=False)
        validate_source(selected, payload, trainer.sha256_file(files['selected_source_checkpoint_sha256']), args.fold_id)
        model.load_state_dict(payload['model'], strict=True)
        del payload
        model.float()
        anchor_base = clean.adapter.CsigZipDataset(clean.ROOT / 'Train.zip', 'Train', clean.adapter._build_transform(448, 448), include_metadata=True)
        manifest = json.loads(files['teacher_manifest_sha256'].read_text(encoding='utf-8'))
        keys = [p.relative_to(clean.TEACHER_ROOT).as_posix() for p in sorted(clean.TEACHER_ROOT.rglob('*.npy'))]
        pilot.validate_teacher(manifest, selected['immutable_config'], anchor_base.records, keys, hashes['train_zip_sha256'])
        tree = teacher_tree()
        if tree != manifest.get('teacher_tree_sha256'):
            raise ValueError('Pinned teacher content tree changed')
        config = immutable_config(args, selected, dict(hashes, teacher_tree_sha256=tree), approved)
        config['input_files'] = {k: str(p) for k, p in files.items()}
        config['auxiliary_file_hashes'] = auxiliary_file_hashes(training + development, approved)
        config['runtime_source_hashes'] = runtime_source_hashes(deps, model)
        config['framework_versions'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                        'numpy': np.__version__}
        print(f'Pinned {len(config["auxiliary_file_hashes"])} auxiliary files and '
              f'{len(config["runtime_source_hashes"])} runtime source files', flush=True)
        trainer.atomic_write_json(output / 'immutable_config.json', config)
        config_sha = trainer.sha256_file(output / 'immutable_config.json')
        sampler = trainer.make_balanced_sampler(training, config['seed'])
        generator = torch.Generator().manual_seed(config['seed'] + 17)
        kwargs = dict(batch_size=4, num_workers=2, pin_memory=True, persistent_workers=True)
        train_loader = torch.utils.data.DataLoader(trainer.RealIADSupervisedDataset(training, 448), sampler=sampler, drop_last=True, **kwargs)
        anchor_loader = torch.utils.data.DataLoader(trainer.AnchorTeacherDataset(anchor_base, clean.TEACHER_ROOT), shuffle=True,
                                                    generator=generator, drop_last=True, **kwargs)
        dev_loader = torch.utils.data.DataLoader(trainer.RealIADSupervisedDataset(development, 448), shuffle=False, **kwargs)
        if not len(train_loader) or not len(anchor_loader):
            raise ValueError('Empty clean training loader')
        baseline = bind_baseline(config, selected, pilot.evaluate(model, development, dev_loader, 1.0))
        trainer.atomic_write_json(output / 'baseline.json', baseline)
        baseline_sha = trainer.sha256_file(output / 'baseline.json')
        integrity = (files, hashes, config, selected, output, config_sha, baseline_sha)
        verify_inputs(*integrity)
        initial = baseline['evaluation']
        best = candidate(config, baseline_sha, 0, selected['checkpoint'], selected['checkpoint_sha256'], initial, initial)
        # This is a new stochastic experiment, not restoration of a V336 iterator/optimizer.
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        optimizer, scheduler = make_optimizer(model, deps, args.steps)
        train_it, anchor_it = iter(train_loader), iter(anchor_loader)
        pilot.training_mode(model)
        results = []
        for next_step in range(1, args.steps + 1):
            train_it, anchor_it, loss, grad = optimizer_step(model, optimizer, scheduler, train_loader, anchor_loader, train_it, anchor_it)
            step = next_step
            if step <= 3 or step % 25 == 0:
                print(f'clean_long fold={args.fold_id} additional={step}/{args.steps} total={selected["step"] + step} '
                      f'loss={loss:.6f} grad={grad:.6f} minutes={(time.monotonic()-started)/60:.2f}', flush=True)
            if step in checkpoints:
                evaluated = pilot.evaluate(model, development, dev_loader, 1.0)
                verify_inputs(*integrity)
                path = output / 'checkpoints' / f'state_i{step:06d}.pt'
                state = checkpoint_state(model, optimizer, scheduler, sampler, generator, config, baseline_sha, step)
                trainer.atomic_torch_save(torch, state, path)
                result = candidate(config, baseline_sha, step, path, trainer.sha256_file(path), initial, evaluated)
                trainer.atomic_write_json(output / f'validation_i{step:06d}.json', result)
                results.append(step)
                if result['comparison']['passed'] and result['comparison']['balanced_score'] > best['comparison']['balanced_score']:
                    best = result
                pilot.training_mode(model)
                print(json.dumps(dict(fold=args.fold_id, step=step, elapsed_minutes=(time.monotonic()-started)/60,
                                      metrics=evaluated['metrics'], comparison=result['comparison'])), flush=True)
        verify_inputs(*integrity)
        terminal = dict(progress(config, baseline_sha, step), status='ok', immutable_config=config, parameter_counts=counts,
                        validation_steps=results, selected=best, baseline_fallback=dict(checkpoint=selected['checkpoint'],
                        checkpoint_sha256=selected['checkpoint_sha256'], baseline_sha256=baseline_sha), elapsed_minutes=(time.monotonic()-started)/60)
        trainer.atomic_write_json(output / 'report.json', terminal)
        return terminal
    except BaseException as error:
        failure = dict(META, status='failed', fold_id=args.fold_id, relative_additional_step=step, baseline_sha256=baseline_sha,
                       total_optimizer_step=None if config is None else config['source_step'] + step,
                       immutable_config=config, error=repr(error), elapsed_minutes=(time.monotonic()-started)/60)
        trainer.atomic_write_json(output / 'failure.json', failure)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--fold-id', type=int, choices=range(4), required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--checkpoint-steps', default='1000,2000,3000,5000')
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
