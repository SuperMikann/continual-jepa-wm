"""Inspect native CEM candidates after the P1 pilot, without training.

Searches are guided by the old model and score-KD model on the same offline
inputs and native solver settings. This is not an environment success test.
"""
import argparse
import json
import os
import time
from pathlib import Path


def load_probe_scenes(bank, count, seed, device):
    import numpy as np
    import torch
    import stable_worldmodel as swm
    from pred_error import episode_col, img_transform
    from train_planning_retention import valid_windows
    manifest = bank['manifest']
    dataset = swm.data.load_dataset(manifest['dataset'], keys_to_cache=['pixels', 'action'])
    episodes = np.asarray(dataset.get_col_data(episode_col(dataset))).reshape(-1)
    steps = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    reserved = manifest['reserved_episodes']
    span = max(manifest['goal_offset'], 3 * manifest['frameskip'] - 1)
    pool = valid_windows(episodes, steps, reserved, span)
    if len(pool) < count:
        raise ValueError('Not enough complete probe windows.')
    rng = np.random.default_rng(seed)
    transform = img_transform()
    mean = np.asarray(manifest['scaler_mean'])
    scale = np.asarray(manifest['scaler_scale'])
    frame_skip = manifest['frameskip']
    scenes = []
    for pick in rng.choice(len(pool), count, replace=False):
        ep, start = pool[int(pick)]
        chunk = dataset.load_chunk(
            np.array([ep]), np.array([start]), np.array([start + span + 1])
        )[0]
        raw = np.asarray(chunk['action'][:3 * frame_skip])
        if not np.isfinite(raw).all():
            raise ValueError('Non-finite probe actions.')
        actions = ((raw - mean) / scale).astype('float32').reshape(3, -1)
        window = torch.stack([transform(chunk['pixels'][i * frame_skip]) for i in range(3)])
        goal = transform(chunk['pixels'][manifest['goal_offset']])
        scenes.append({
            'group': 'probe', 'ep': ep, 'start': start,
            'pixels': window[None].to(device), 'goal': goal[None].to(device),
            'action': torch.from_numpy(actions)[None].to(device),
        })
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/root/autodl-tmp/stable-worldmodel')
    parser.add_argument('--memory', default='results/p1_pusht_memory_seed0.pt')
    parser.add_argument('--sources', default='results/p1_memory_coverage_audit_sources.json')
    parser.add_argument('--out', default='results/p1_native_cem_retention.json')
    parser.add_argument('--old-policy', default='pusht_seed/weights_epoch_30.pt')
    parser.add_argument('--score-policy', default='p1_pusht_score_kd_s0_300/weights_step_300.pt')
    parser.add_argument('--other-policies', nargs='*', default=[
        'p1_pusht_finetune_s0_300/weights_step_300.pt',
        'p1_pusht_replay_s0_300/weights_step_300.pt',
    ])
    parser.add_argument('--memory-scenes', type=int, default=4)
    parser.add_argument('--probe-scenes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.memory_scenes, args.probe_scenes) < 1:
        parser.error('Use positive scene counts for both groups.')
    root = Path(args.root).resolve()

    def path_for(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    out = path_for(args.out)
    population_out = out.with_name(out.stem + '_populations.pt')
    for path in (out, population_out):
        if path.exists():
            raise FileExistsError(f'Keep previous outputs; choose another --out: {path}')
    print('[START] native CEM inspection, no training/environment rollout', flush=True)

    import numpy as np
    import torch
    import gymnasium as gym
    import hydra
    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from stable_worldmodel.planning import ShootingCostEvaluator
    from planning_retention_probe import score_metrics, aggregate
    from train_planning_retention import load_memory_batch, score_plans
    if not torch.cuda.is_available():
        raise RuntimeError('Run this inspection on the AutoDL GPU.')
    device = 'cuda'
    sources = json.loads(path_for(args.sources).read_text(encoding='utf-8'))
    with hydra.initialize_config_dir(config_dir=str(Path(sources['config_root']).resolve()), version_base=None):
        cfg = hydra.compose(config_name='pusht', overrides=[f'seed={args.seed}'])
    solver_cfg = OmegaConf.to_container(cfg.solver, resolve=True)
    solver_cfg.pop('cost', None)
    if solver_cfg['_target_'] != 'stable_worldmodel.planning.solver.CEMSolver':
        raise ValueError('Expected the native continuous CEMSolver.')
    if int(solver_cfg['batch_size']) != 1 or str(solver_cfg['device']) != 'cuda':
        raise ValueError('This small inspection expects native batch_size=1 and CUDA.')
    if cfg.get('bf16', False) or cfg.get('compile', False):
        raise ValueError('Inspect the current uncompiled float32 pilot only.')
    plan_cfg = OmegaConf.to_container(cfg.plan_config, resolve=True)
    if plan_cfg['horizon'] != plan_cfg['receding_horizon']:
        raise ValueError('Warm-start leftovers need a separate protocol.')
    bank = torch.load(path_for(args.memory), map_location='cpu', weights_only=False)
    manifest = bank['manifest']
    if manifest['teacher'] != args.old_policy:
        raise ValueError('Memory and old model do not match.')
    if (plan_cfg['horizon'] != manifest['horizon']
        or plan_cfg['action_block'] != manifest['frameskip']
        or int(cfg.eval.goal_offset_steps) != manifest['goal_offset']):
        raise ValueError('Native planning and P1 temporal settings differ.')
    if args.memory_scenes > len(bank['pixels']):
        raise ValueError('Requested too many memory scenes.')
    policies = [args.old_policy, args.score_policy] + args.other_policies
    if len(set(policies)) != len(policies):
        raise ValueError('Checkpoint names must be unique.')
    cache = Path(os.environ['STABLEWM_HOME']) / 'checkpoints'
    for policy in policies:
        checkpoint = Path(policy)
        checkpoint = checkpoint if checkpoint.is_absolute() else cache / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    horizon = int(plan_cfg['horizon'])
    action_dim = bank['candidates'].shape[-1]
    if action_dim % int(plan_cfg['action_block']):
        raise ValueError('Action block dimensions are inconsistent.')
    raw_action_dim = action_dim // int(plan_cfg['action_block'])
    native_plan = swm.PlanConfig(**plan_cfg)
    iterations = int(solver_cfg['n_steps'])
    capture_steps = sorted({0, iterations // 2, iterations - 1})

    class TraceCost(torch.nn.Module):
        def __init__(self, evaluator):
            super().__init__()
            self.evaluator = evaluator
            self.calls = 0
            self.populations = []

        def get_cost(self, info, candidates):
            costs = self.evaluator.get_cost(info, candidates)
            if self.calls in capture_steps:
                self.populations.append({
                    'iteration': self.calls + 1,
                    'candidates': candidates.detach().cpu().clone(),
                    'guide_costs': costs.detach().cpu().clone(),
                })
            self.calls += 1
            return costs

    def make_solver(cost, seed):
        settings = {**solver_cfg, 'seed': seed}
        solver = hydra.utils.instantiate(settings, cost=cost)
        # The inspected native solver only reads this batched Box's shape.
        # No environment is created and no physical action is executed.
        space = gym.spaces.Box(-np.inf, np.inf, shape=(1, raw_action_dim), dtype=np.float32)
        solver.configure(action_space=space, n_envs=1, config=native_plan)
        if solver.action_dim != action_dim:
            raise ValueError('Configured native solver action dimension differs.')
        return solver

    class QuadraticCost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), device=device), requires_grad=False)

        def get_cost(self, info, candidates):
            return candidates.square().sum(dim=(2, 3))

    # Meaningful integration check: observing candidates must not change search.
    plain = make_solver(QuadraticCost(), args.seed)
    traced_cost = TraceCost(QuadraticCost())
    traced = make_solver(traced_cost, args.seed)
    dummy = {'state': torch.zeros(1, 1)}
    plain_result = plain.solve(dummy.copy())
    traced_result = traced.solve(dummy.copy())
    torch.testing.assert_close(plain_result['actions'], traced_result['actions'], rtol=0, atol=0)
    if traced_cost.calls != iterations or len(traced_cost.populations) != len(capture_steps):
        raise RuntimeError('Native CEM observation check failed.')
    print('[TRACE_CHECK_OK] recording leaves native search output unchanged', flush=True)
    del plain, traced, traced_cost

    rng = np.random.default_rng(args.seed)
    scenes = []
    for index in rng.choice(len(bank['pixels']), args.memory_scenes, replace=False):
        index = int(index)
        batch = load_memory_batch(bank, [index], device)
        ep, start = manifest['scenes'][index]
        scenes.append({
            'group': 'memory', 'memory_index': index, 'ep': int(ep), 'start': int(start),
            'pixels': batch['pixels'][:, :3], 'goal': batch['goal'], 'action': batch['action'][:, :3],
        })
    scenes += load_probe_scenes(bank, args.probe_scenes, args.seed + 100, device)

    captures = []
    returned = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def load_model(policy):
        model = swm.wm.utils.load_pretrained(policy).to(device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if int(getattr(model.predictor, 'num_frames', 3)) != 3:
            raise ValueError('This inspection expects LeWM history=3.')
        return model

    for guide in (args.old_policy, args.score_policy):
        model = load_model(guide)
        objective = hydra.utils.instantiate(cfg.objective)
        if (getattr(objective, 'pred_key', None), getattr(objective, 'goal_key', None)) != (
            'predicted_emb', 'goal_emb'
        ):
            raise ValueError('Expected unchanged native GoalMSE keys.')
        for i, scene in enumerate(scenes):
            trace = TraceCost(ShootingCostEvaluator(model, objective))
            solver = make_solver(trace, args.seed + i)
            info = {
                'pixels': scene['pixels'], 'goal': scene['goal'][:, None],
                'action': scene['action'],
            }
            result = solver.solve(info, init_action=None)
            if trace.calls != iterations:
                raise RuntimeError('Unexpected number of native CEM cost calls.')
            returned[(guide, i)] = result['actions'].clone()
            for population in trace.populations:
                captures.append({'guide': guide, 'scene_index': i, **population})
            print(f'[SEARCH {guide} {i + 1}/{len(scenes)}] {scene["group"]} ep={scene["ep"]} start={scene["start"]}', flush=True)
            del solver, trace
        del model
        torch.cuda.empty_cache()
    for i in range(len(scenes)):
        initial = [p for p in captures if p['scene_index'] == i and p['iteration'] == 1]
        if len(initial) != 2 or not torch.equal(initial[0]['candidates'], initial[1]['candidates']):
            raise RuntimeError('The two guides did not receive identical initial populations.')

    old_costs = []
    scored = []
    returned_costs = {}
    parity_differences = {}
    for policy in policies:
        model = load_model(policy)
        parity = 0.0
        with torch.inference_mode():
            for p_index, population in enumerate(captures):
                scene = scenes[population['scene_index']]
                costs = score_plans(
                    model, scene['pixels'], scene['goal'],
                    population['candidates'].to(device), scene['action'], False,
                ).detach().cpu()
                if not torch.isfinite(costs).all():
                    raise ValueError('Non-finite CEM population scores.')
                if population['guide'] == policy:
                    difference = float((costs - population['guide_costs']).abs().max())
                    parity = max(parity, difference)
                    torch.testing.assert_close(costs, population['guide_costs'], rtol=1e-4, atol=1e-3)
                values = costs[0].tolist()
                if policy == args.old_policy:
                    old_costs.append(values)
                reference = old_costs[p_index]
                measures = score_metrics(reference, values, int(solver_cfg['topk']), 1.0)
                best = min(range(len(values)), key=lambda k: values[k])
                measures['old_absolute_choice_gap'] = reference[best] - min(reference)
                scored.append({
                    'policy': policy, 'guide': population['guide'],
                    'scene_index': population['scene_index'], 'group': scene['group'],
                    'iteration': population['iteration'], **measures,
                    'old_costs': reference, 'new_costs': values,
                })
            for i, scene in enumerate(scenes):
                plans = torch.stack([
                    returned[(guide, i)][0] for guide in (args.old_policy, args.score_policy)
                ])[None].to(device)
                values = score_plans(
                    model, scene['pixels'], scene['goal'], plans, scene['action'], False,
                )[0].detach().cpu().tolist()
                returned_costs[(policy, i)] = values
        parity_differences[policy] = parity if policy in (args.old_policy, args.score_policy) else None
        print('[SCORE_DONE]', policy, flush=True)
        del model
        torch.cuda.empty_cache()

    plan_comparison = []
    for i, scene in enumerate(scenes):
        old_returned = returned_costs[(args.old_policy, i)]
        first = next(p_index for p_index, p in enumerate(captures)
                     if p['scene_index'] == i and p['iteration'] == 1 and p['guide'] == args.old_policy)
        reference = np.asarray(old_costs[first], dtype=float)
        initial_std = float(reference.std())
        gap = old_returned[1] - old_returned[0]
        plan_comparison.append({
            'scene_index': i, 'group': scene['group'],
            'old_cost_of_old_cem_plan': old_returned[0],
            'old_cost_of_score_kd_cem_plan': old_returned[1],
            'old_score_gap_between_returned_plans': gap,
            'old_score_gap_initial_std': gap / initial_std if initial_std > 1e-6 else None,
            'old_initial_population_std': initial_std,
            'returned_plan_scores': {p: returned_costs[(p, i)] for p in policies},
        })
    summary = []
    keys = [
        'teacher_student_tau', 'top1_agreement', 'topk_overlap',
        'teacher_regret_std', 'preference_kl', 'old_absolute_choice_gap',
        'teacher_cost_std', 'teacher_cost_range',
    ]
    for guide in (args.old_policy, args.score_policy):
        for group in ('memory', 'probe'):
            for iteration in [step + 1 for step in capture_steps]:
                for policy in policies:
                    rows = [r for r in scored if r['guide'] == guide and r['group'] == group
                            and r['iteration'] == iteration and r['policy'] == policy]
                    summary.append({
                        'guide': guide, 'group': group, 'iteration': iteration, 'policy': policy,
                        **{key: aggregate(rows, key) for key in keys},
                    })
    plan_summary = {
        group: {key: aggregate([r for r in plan_comparison if r['group'] == group], key)
                for key in ('old_score_gap_between_returned_plans', 'old_score_gap_initial_std')}
        for group in ('memory', 'probe')
    }
    torch.cuda.synchronize()
    result = {
        'config': vars(args), 'native_solver_config': solver_cfg,
        'native_plan_config': plan_cfg, 'capture_iterations': [x + 1 for x in capture_steps],
        'scene_metadata': [{k: v for k, v in s.items() if k not in ('pixels', 'goal', 'action')} for s in scenes],
        'trace_check_passed': True, 'initial_populations_equal': True,
        'guide_cost_reproduction_max_abs_diff': parity_differences,
        'summary': summary, 'per_population': scored,
        'returned_plan_summary': plan_summary, 'returned_plan_per_scene': plan_comparison,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'population_file': str(population_out),
        'limitation': 'Offline memory/probe inputs, not visited execution states or physical success. Old-model scores are a reference, not ground truth. Ranking can be unstable when final costs are nearly flat; inspect absolute gaps and cost ranges.',
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'scene_metadata': result['scene_metadata'],
        'captures': captures,
        'returned_plans': [
            {'guide': guide, 'scene_index': i, 'actions': actions}
            for (guide, i), actions in returned.items()
        ],
    }, population_out)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('[RETURNED_PLAN_SUMMARY]', json.dumps(plan_summary, ensure_ascii=False), flush=True)
    print(f'[NATIVE_CEM_DONE] scenes={len(scenes)}; searches={2 * len(scenes)}; out={out}', flush=True)


if __name__ == '__main__':
    main()
