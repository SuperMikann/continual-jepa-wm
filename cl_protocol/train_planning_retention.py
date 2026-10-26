"""Small, controlled T1->T2 pilot: finetune / pixel replay / planner KD.

Uses the native LeWM training configuration and data pipeline, and the exact
teacher architecture. No adapter growth or historical auxiliary losses.
Planner KD keeps native GoalMSE's detached goal, but supplies a differentiable
history embedding to LeWM.rollout. Check numerical parity AND gradient flow
on the server before training. Final checkpoints use native save_pretrained.
"""

import argparse
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path


def valid_windows(episodes, steps, allowed, span):
    """Require every step in the window, never bridge gaps/episode boundaries."""
    by_episode = {}
    for episode, step in zip(episodes, steps):
        by_episode.setdefault(int(episode), set()).add(int(step))
    return [
        (int(ep), s)
        for ep in allowed
        for s in sorted(by_episode[int(ep)])
        if all(s + offset in by_episode[int(ep)] for offset in range(span + 1))
    ]


@contextmanager
def planning_mode(model):
    # Restore individual flags; model.train() would overwrite frozen submodules.
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, flag in flags:
            module.training = flag


def preference_loss(student_cost, teacher_cost, temperature=1.0):
    import torch
    import torch.nn.functional as F

    def logp(cost):
        cost = cost.float()
        centered = cost - cost.mean(-1, keepdim=True)
        scale = centered.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        return F.log_softmax(-centered / scale / temperature, dim=-1)

    t_logp = logp(teacher_cost.detach())
    s_logp = logp(student_cost)
    return (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()


def score_plans(model, pixels, goal, candidates, history_actions, differentiable):
    """Same scores as native evaluator, optionally retain history gradients.

Native rollout overwrites info['action'] with candidates[:,:,:H]. The manual
initial encoding must do the same, even though historical actions are provided
by the protocol. This pilot preserves that deployment convention.
"""
    import torch
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator

    batch, count = candidates.shape[:2]
    history = pixels.shape[1]
    info = {
        'pixels': pixels[:, None].expand(batch, count, *pixels.shape[1:]),
        'goal': goal[:, None, None].expand(batch, count, 1, *goal.shape[1:]),
        'action': history_actions[:, None].expand(batch, count, *history_actions.shape[1:]),
    }
    if differentiable:
        initial = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        initial['action'] = candidates[:, 0, :history]
        emb = model.encode(initial)['emb']
        info['emb'] = emb[:, None].expand(batch, count, *emb.shape[1:])
    return ShootingCostEvaluator(model, GoalMSE()).get_cost(info, candidates.clone())


def load_memory_batch(bank, indices, device):
    import torch
    from pred_error import img_transform

    transform = img_transform()
    pixels = torch.stack([
        torch.stack([transform(frame) for frame in bank['pixels'][int(i)]])
        for i in indices
    ]).to(device)
    return {
        'pixels': pixels[:, :-1], 'goal': pixels[:, -1],
        'action': bank['actions'][indices].to(device),
        'candidates': bank['candidates'][indices].to(device),
        'teacher_costs': bank['teacher_costs'][indices].to(device),
    }


def build_memory(model, args, device):
    import numpy as np
    import torch
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from pred_error import episode_col

    dataset = swm.data.load_dataset(args.old_dataset, keys_to_cache=['pixels', 'action'])
    episodes = np.asarray(dataset.get_col_data(episode_col(dataset))).reshape(-1)
    steps = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    unique = np.unique(episodes)
    if len(unique) <= args.holdout:
        raise ValueError('No memory episodes remain after reserving the probe pool.')
    allowed = unique[:-args.holdout]
    history = int(getattr(model.predictor, 'num_frames', 3))
    if history != 3 or args.horizon < history:
        raise ValueError('This pilot requires native LeWM history=3, horizon>=3.')
    span = max(args.goal_offset, (history + 1) * args.frameskip - 1)
    if args.goal_offset <= history * args.frameskip:
        raise ValueError('Goal must follow the replay target frame.')
    pool = valid_windows(episodes, steps, allowed, span)
    if len(pool) < args.memory_size:
        raise ValueError('Not enough old-task memory windows.')
    rng = np.random.default_rng(args.seed)
    scenes = [pool[i] for i in rng.choice(len(pool), args.memory_size, replace=False)]
    all_actions = np.asarray(dataset.get_col_data('action'))
    scaler = StandardScaler().fit(all_actions[np.isfinite(all_actions).all(axis=1)])
    action_dim = all_actions.shape[1] * args.frameskip
    pixels, actions = [], []
    for ep, start in scenes:
        chunk = dataset.load_chunk(np.array([ep]), np.array([start]), np.array([start + span + 1]))[0]
        frames = []
        for offset in [0, args.frameskip, 2 * args.frameskip, 3 * args.frameskip, args.goal_offset]:
            raw = torch.as_tensor(np.asarray(chunk['pixels'][offset])).clone()
            if raw.ndim != 3 or raw.dtype != torch.uint8:
                raise ValueError(f'Memory requires raw uint8 images, got {raw.shape}/{raw.dtype}.')
            if raw.shape[0] != 3 and raw.shape[-1] == 3:
                raw = raw.permute(2, 0, 1).contiguous()
            if raw.shape[0] != 3:
                raise ValueError('Expected RGB image.')
            frames.append(raw)
        raw_actions = np.asarray(chunk['action'][:4 * args.frameskip])
        if not np.isfinite(raw_actions).all():
            raise ValueError(f'Non-finite memory actions: ep={ep}, start={start}.')
        pixels.append(torch.stack(frames))
        actions.append(torch.from_numpy(scaler.transform(raw_actions).astype('float32').reshape(4, action_dim)))
    generator = torch.Generator().manual_seed(args.seed + 100)
    bank = {
        'pixels': torch.stack(pixels), 'actions': torch.stack(actions),
        'candidates': torch.randn(args.memory_size, args.candidates, args.horizon, action_dim, generator=generator),
        'teacher_costs': torch.zeros(args.memory_size, args.candidates),
        'manifest': {
            'teacher': args.teacher, 'dataset': args.old_dataset, 'seed': args.seed,
            'memory_size': args.memory_size, 'frameskip': args.frameskip,
            'goal_offset': args.goal_offset, 'horizon': args.horizon,
            'candidates': args.candidates, 'holdout': args.holdout,
            'reserved_episodes': unique[-args.holdout:].tolist(),
            'scenes': scenes, 'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'candidate_source': 'initial Gaussian population, not CEM-refined',
            'goal_gradient': 'detached, same as native GoalMSE',
        },
    }
    with planning_mode(model), torch.no_grad():
        for i in range(args.memory_size):
            batch = load_memory_batch(bank, [i], device)
            costs = score_plans(model, batch['pixels'][:, :3], batch['goal'], batch['candidates'], batch['action'][:, :3], False)
            if not torch.isfinite(costs).all() or costs.std(unbiased=False) < 1e-6:
                raise ValueError(f'Invalid/flat teacher costs in memory scene {i}.')
            bank['teacher_costs'][i] = costs[0].cpu()
            if (i + 1) % 16 == 0:
                print(f'[MEMORY {i + 1}/{args.memory_size}]', flush=True)
    bank['manifest']['tensor_bytes'] = sum(
        v.numel() * v.element_size() for v in bank.values() if torch.is_tensor(v)
    )
    bank['manifest']['pixel_bytes'] = bank['pixels'].numel() * bank['pixels'].element_size()
    bank['manifest']['score_label_and_candidate_bytes'] = sum(
        bank[key].numel() * bank[key].element_size() for key in ('teacher_costs', 'candidates')
    )
    return bank


def gradient_check(model, bank, device):
    import torch

    batch = load_memory_batch(bank, [0], device)
    inputs = (batch['pixels'][:, :3], batch['goal'], batch['candidates'], batch['action'][:, :3])
    with planning_mode(model):
        with torch.no_grad():
            native = score_plans(model, *inputs, False)
            retained = score_plans(model, *inputs, True)
        torch.testing.assert_close(retained, native, rtol=1e-5, atol=1e-4)
        costs = score_plans(model, *inputs, True)
        # Teacher==student at initialization gives KL~0; reversed labels are
        # ONLY for this gradient check, never saved as training targets.
        loss = preference_loss(costs, native.flip(-1))
        parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        gradients = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
        norms = {}
        for prefix in ('encoder', 'predictor'):
            value = sum(float(g.float().square().sum()) for (name, _), g in zip(parameters, gradients) if name.startswith(prefix) and g is not None)
            norms[prefix] = math.sqrt(value)
            if not math.isfinite(norms[prefix]) or norms[prefix] <= 0:
                raise RuntimeError(f'No finite nonzero gradient reaches {prefix}: {norms}.')
        same_loss = float(preference_loss(retained, native))
        if abs(same_loss) > 1e-5:
            raise RuntimeError(f'Identical-model KL check failed: {same_loss}.')
    print(f'[GRAD_CHECK_OK] native_cost_match; encoder/predictor_grad_norm={norms}', flush=True)
    return {'native_cost_max_abs_diff': float((retained - native).abs().max()), 'gradient_norm': norms, 'same_model_kl': same_loss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['prepare', 'finetune', 'replay', 'score_kd'])
    parser.add_argument('--teacher', default='pusht_seed/weights_epoch_30.pt')
    parser.add_argument('--old-dataset', default='pusht_expert_train.lance')
    parser.add_argument('--new-dataset', default='pusht_t2.lance')
    parser.add_argument('--memory', default='results/p1_pusht_memory_seed0.pt')
    parser.add_argument('--run-name')
    parser.add_argument('--config-dir', default='scripts/train/config')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--memory-size', type=int, default=64)
    parser.add_argument('--old-batch', type=int, default=2)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--frameskip', type=int, default=5)
    parser.add_argument('--goal-offset', type=int, default=25)
    parser.add_argument('--holdout', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--kd-weight', type=float, default=0.01)
    parser.add_argument('--replay-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.memory_size, args.old_batch, args.holdout, args.frameskip) < 1 or args.candidates < 2 or args.temperature <= 0:
        parser.error('Invalid pilot sizes/temperature.')
    if args.old_batch > args.memory_size or args.kd_weight < 0 or args.replay_weight < 0:
        parser.error('old-batch must fit the memory; loss weights must be nonnegative.')
    if args.mode != 'prepare' and not args.run_name:
        parser.error('--run-name is required for training.')
    print(f'[START] mode={args.mode}; steps={args.steps}; teacher={args.teacher}', flush=True)

    import numpy as np
    import torch
    import hydra
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from omegaconf import OmegaConf, open_dict
    from stable_worldmodel.data import column_normalizer
    from stable_worldmodel.wm.loss import SIGReg
    from stable_worldmodel.wm.utils import save_pretrained

    pl.seed_everything(args.seed, workers=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('Run this training pilot on the AutoDL GPU.')
    teacher_path = Path(args.teacher)
    if not teacher_path.is_absolute():
        teacher_path = Path(os.environ['STABLEWM_HOME']) / 'checkpoints' / teacher_path
    model_config = OmegaConf.load(teacher_path.parent / 'config.json')
    model = swm.wm.utils.load_pretrained(str(teacher_path)).to(device)
    model.requires_grad_(True)
    if type(model.predictor).__name__ != 'Predictor':
        raise ValueError('Pilot requires the vanilla Predictor checkpoint; do not change architecture yet.')
    memory_path = Path(args.memory).resolve()
    if args.mode == 'prepare':
        if memory_path.exists():
            raise FileExistsError(f'Memory already exists: {memory_path}; reuse it or choose another path.')
        bank = build_memory(model, args, device)
        check = gradient_check(model, bank, device)
        bank['manifest']['gradient_check'] = check
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank, memory_path)
        memory_path.with_suffix('.json').write_text(json.dumps(bank['manifest'], ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[PREPARE_DONE] memory={memory_path}; bytes={bank["manifest"]["tensor_bytes"]}', flush=True)
        return

    bank = torch.load(memory_path, map_location='cpu', weights_only=False)
    expected = ('teacher', 'seed', 'old_dataset', 'memory_size', 'frameskip', 'goal_offset', 'horizon', 'candidates', 'holdout')
    for key in expected:
        stored = bank['manifest']['dataset' if key == 'old_dataset' else key]
        if stored != getattr(args, key):
            raise ValueError(f'Memory mismatch for {key}: {stored} != {getattr(args, key)}.')
    gradient_check(model, bank, device)
    run_dir = Path(os.environ['STABLEWM_HOME']) / 'checkpoints' / args.run_name
    if run_dir.exists():
        raise FileExistsError(f'Run already exists: {run_dir}; choose a new run-name.')
    with hydra.initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()), version_base=None):
        cfg = hydra.compose(config_name='lewm', overrides=[f'data.dataset.name={args.new_dataset}'])
    with open_dict(cfg):
        cfg.seed = args.seed
        cfg.model = model_config
        cfg.loader.batch_size = args.batch_size
        cfg.loader.num_workers = 0
        cfg.loader.prefetch_factor = None
        cfg.loader.persistent_workers = False
    if int(cfg.wm.history_size) != 3 or int(cfg.wm.num_preds) != 1 or int(cfg.img_size) != 224:
        raise ValueError('Pilot expects native history=3, num_preds=1, image_size=224.')
    if int(cfg.data.dataset.frameskip) != args.frameskip:
        raise ValueError('Training/memory frameskip differ.')
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=os.environ.get('LOCAL_DATASET_DIR'), **dataset_cfg)
    dt = spt.data
    image = dt.transforms.Compose(
        dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source='pixels', target='pixels'),
        dt.transforms.Resize(cfg.img_size, source='pixels', target='pixels'),
    )
    transforms = [image]
    for col in cfg.data.dataset.keys_to_load:
        if not col.startswith('pixels'):
            transforms.append(column_normalizer(dataset, col, col))
    dataset.transform = dt.transforms.Compose(*transforms)
    if args.frameskip * dataset.get_dim('action') != int(model_config.action_encoder.input_dim):
        raise ValueError('T1/T2 action dimensions do not match.')
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = dt.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=generator)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, generator=generator)
    val = torch.utils.data.DataLoader(val_set, **{**cfg.loader, 'shuffle': False, 'drop_last': False})
    if len(train) == 0:
        raise ValueError('Empty new-task training loader.')
    memory_rng = np.random.default_rng(args.seed + 200)
    trace = []

    def forward(module, batch, stage):
        batch['action'] = torch.nan_to_num(batch['action'], 0.0)
        output = module.model.encode(batch)
        emb = output['emb']
        pred = module.model.predict(emb[:, :3], output['act_emb'][:, :3])
        pred_loss = (pred - emb[:, 1:]).square().mean()
        sigreg = module.sigreg(emb.transpose(0, 1))
        loss = pred_loss + cfg.loss.sigreg.weight * sigreg
        old_loss = loss.new_zeros(())
        is_training = module.training and torch.is_grad_enabled()
        if is_training and args.mode != 'finetune':
            indices = memory_rng.choice(args.memory_size, args.old_batch, replace=False)
            old = load_memory_batch(bank, indices, emb.device)
            with planning_mode(module.model):
                if args.mode == 'replay':
                    replay = module.model.encode({'pixels': old['pixels'], 'action': old['action']})
                    prediction = module.model.predict(replay['emb'][:, :3], replay['act_emb'][:, :3])
                    old_loss = (prediction - replay['emb'][:, 1:]).square().mean()
                    loss = loss + args.replay_weight * old_loss
                else:
                    costs = score_plans(module.model, old['pixels'][:, :3], old['goal'], old['candidates'], old['action'][:, :3], True)
                    old_loss = preference_loss(costs, old['teacher_costs'], args.temperature)
                    loss = loss + args.kd_weight * old_loss
        output.update(loss=loss, pred_loss=pred_loss, sigreg_loss=sigreg, old_loss=old_loss)
        module.log_dict({f'{stage}/{key}': output[key].detach() for key in ('loss', 'pred_loss', 'sigreg_loss', 'old_loss')}, on_step=True, sync_dist=True)
        if is_training:
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss.')
            values = {key: float(output[key].detach()) for key in ('loss', 'pred_loss', 'sigreg_loss', 'old_loss')}
            trace.append({'step': int(module.global_step) + 1, **values})
            if len(trace) == 1 or len(trace) % 20 == 0:
                print(f'[TRAIN {args.mode} {len(trace)}/{args.steps}] {values}', flush=True)
        return output

    # Native optimizer/SIGReg. All three arms share a step-based short-run
    # schedule; this screening run is not a reproduction of the 30-epoch chain.
    optimizers = {'model_opt': {
        'modules': 'model', 'optimizer': dict(cfg.optimizer),
        'scheduler': {'type': 'LinearWarmupCosineAnnealingLR', 'warmup_steps': max(1, int(0.01 * args.steps)), 'max_steps': args.steps},
        'interval': 'step',
    }}
    module = spt.Module(model=model, sigreg=SIGReg(**cfg.loss.sigreg.kwargs), forward=forward, optim=optimizers)
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.update(max_steps=args.steps, max_epochs=math.ceil(args.steps / len(train)), accumulate_grad_batches=1, limit_train_batches=1.0, limit_val_batches=1)
    for key in ('callbacks', 'logger', 'enable_checkpointing', 'num_sanity_val_steps'):
        trainer_cfg.pop(key, None)
    trainer = pl.Trainer(**trainer_cfg, logger=False, enable_checkpointing=False, num_sanity_val_steps=0)
    run_dir.mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / 'training_config.yaml')
    # Reset all RNGs after setup/checks, for identical new-data order and SIGReg
    # sampling across arms; the memory RNG remains independent.
    pl.seed_everything(args.seed, workers=True)
    manager = spt.Manager(trainer=trainer, module=module, data=dt.DataModule(train=train, val=val), ckpt_path=None)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    manager()
    training_seconds = time.perf_counter() - started
    if trainer.global_step != args.steps:
        raise RuntimeError(f'Incomplete run: {trainer.global_step}/{args.steps} optimizer steps.')
    filename = f'weights_step_{args.steps}.pt'
    save_pretrained(module.model, run_name=args.run_name, config=model_config, filename=filename)
    result = {'config': vars(args), 'global_step': trainer.global_step, 'checkpoint': str(run_dir / filename), 'training_seconds': training_seconds, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'memory_manifest': bank['manifest'], 'trace': trace}
    (run_dir / 'pilot_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'[TRAIN_DONE] {args.mode}; steps={trainer.global_step}; checkpoint={run_dir / filename}', flush=True)


if __name__ == '__main__':
    main()
