"""Prepare a P1 memory variant containing native CEM-refined candidates.

Keep the same old image windows and eight candidates per window: four original
Gaussian candidates plus two from iteration 16 and two from iteration 30.
Only the old model guides searches. The training loss and budget stay unchanged.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/root/autodl-tmp/stable-worldmodel')
    parser.add_argument('--memory', default='results/p1_pusht_memory_seed0.pt')
    parser.add_argument('--out', default='results/p1_pusht_memory_cem_seed0.pt')
    parser.add_argument('--config-dir', default='scripts/plan/config')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root).resolve()

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    output = resolve(args.out)
    manifest_path = output.with_suffix('.json')
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f'Keep previous memories; choose another --out: {output}')
    print('[START] prepare 4 original + 2 middle + 2 late candidates per memory scene', flush=True)

    import numpy as np
    import torch
    import gymnasium as gym
    import hydra
    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from stable_worldmodel.planning import ShootingCostEvaluator
    from train_planning_retention import load_memory_batch, score_plans, gradient_check

    if not torch.cuda.is_available():
        raise RuntimeError('Run memory preparation on the AutoDL GPU.')
    device = 'cuda'
    original = torch.load(resolve(args.memory), map_location='cpu', weights_only=False)
    manifest = original['manifest']
    if original['candidates'].shape[:3] != (64, 8, 5):
        raise ValueError('This controlled variant requires the original 64 x 8 x 5 memory.')
    if manifest['candidate_source'] != 'initial Gaussian population, not CEM-refined':
        raise ValueError('Input must be the original Gaussian memory, not a derived memory.')
    if len(manifest['scenes']) != 64:
        raise ValueError('Memory scene metadata is incomplete.')
    if any(int(ep) in set(manifest['reserved_episodes']) for ep, _ in manifest['scenes']):
        raise ValueError('Probe episodes must not enter training memory.')

    with hydra.initialize_config_dir(config_dir=str(resolve(args.config_dir)), version_base=None):
        cfg = hydra.compose(config_name='pusht', overrides=[f'seed={args.seed}'])
    solver_cfg = OmegaConf.to_container(cfg.solver, resolve=True)
    solver_cfg.pop('cost', None)
    expected = {
        '_target_': 'stable_worldmodel.planning.solver.CEMSolver',
        'batch_size': 1, 'num_samples': 300, 'n_steps': 30, 'topk': 30,
        'var_scale': 1.0, 'device': 'cuda',
    }
    if any(solver_cfg.get(key) != value for key, value in expected.items()):
        raise ValueError('Native CEM settings differ from the completed diagnostic.')
    if cfg.get('bf16', False) or cfg.get('compile', False):
        raise ValueError('Use the uncompiled float32 pilot.')
    plan_cfg = OmegaConf.to_container(cfg.plan_config, resolve=True)
    if (plan_cfg['horizon'] != manifest['horizon']
        or plan_cfg['receding_horizon'] != plan_cfg['horizon']
        or plan_cfg['action_block'] != manifest['frameskip']
        or int(cfg.eval.goal_offset_steps) != manifest['goal_offset']):
        raise ValueError('Planning and memory temporal settings differ.')
    model = swm.wm.utils.load_pretrained(manifest['teacher']).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    objective = hydra.utils.instantiate(cfg.objective)
    if (getattr(objective, 'pred_key', None), getattr(objective, 'goal_key', None)) != (
        'predicted_emb', 'goal_emb'
    ):
        raise ValueError('Use the same native GoalMSE as the completed diagnostic.')
    action_dim = original['candidates'].shape[-1]
    if action_dim % int(plan_cfg['action_block']):
        raise ValueError('Invalid action block dimension.')
    space = gym.spaces.Box(
        -np.inf, np.inf, shape=(1, action_dim // int(plan_cfg['action_block'])), dtype=np.float32
    )
    native_plan = swm.PlanConfig(**plan_cfg)

    class CaptureCost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.evaluator = ShootingCostEvaluator(model, objective)
            self.calls = 0
            self.captured = {}

        def get_cost(self, info, candidates):
            costs = self.evaluator.get_cost(info, candidates)
            iteration = self.calls + 1
            if iteration in (16, 30):
                self.captured[iteration] = (candidates.detach().cpu().clone(), costs.detach().cpu().clone())
            self.calls += 1
            return costs

    # Same storage size, pixels, historical actions and ordering as the old bank.
    bank = {
        'pixels': original['pixels'], 'actions': original['actions'],
        'candidates': original['candidates'].clone(),
        'teacher_costs': original['teacher_costs'].clone(),
        'manifest': copy.deepcopy(manifest),
    }
    selection_rng = np.random.default_rng(args.seed + 1000)
    choices = []
    reproduction = 0.0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(64):
            batch = load_memory_batch(original, [index], device)
            # Check original labels before deriving the variant.
            costs = score_plans(
                model, batch['pixels'][:, :3], batch['goal'],
                batch['candidates'], batch['action'][:, :3], False,
            )
            torch.testing.assert_close(costs.cpu(), batch['teacher_costs'].cpu(), rtol=1e-4, atol=1e-3)
            capture = CaptureCost()
            solver = hydra.utils.instantiate({**solver_cfg, 'seed': args.seed + index}, cost=capture)
            solver.configure(action_space=space, n_envs=1, config=native_plan)
            if solver.action_dim != action_dim:
                raise ValueError('Native action dimension does not match memory.')
            solver.solve({
                'pixels': batch['pixels'][:, :3], 'goal': batch['goal'][:, None],
                'action': batch['action'][:, :3],
            }, init_action=None)
            if capture.calls != 30 or set(capture.captured) != {16, 30}:
                raise RuntimeError('Native search did not produce both requested populations.')
            scene_choices = {}
            for iteration, destination in ((16, slice(4, 6)), (30, slice(6, 8))):
                population, native_costs = capture.captured[iteration]
                picked = selection_rng.choice(population.shape[1], 2, replace=False)
                scene_choices[str(iteration)] = picked.tolist()
                bank['candidates'][index, destination] = population[0, picked]
                bank['teacher_costs'][index, destination] = native_costs[0, picked]
            derived = load_memory_batch(bank, [index], device)
            rescored = score_plans(
                model, derived['pixels'][:, :3], derived['goal'],
                derived['candidates'], derived['action'][:, :3], False,
            ).cpu()
            torch.testing.assert_close(rescored, derived['teacher_costs'].cpu(), rtol=1e-4, atol=1e-3)
            if not torch.isfinite(rescored).all() or float(rescored.std(unbiased=False)) < 1e-6:
                raise ValueError(f'Non-finite or flat labels in memory scene {index}.')
            reproduction = max(reproduction, float((rescored - derived['teacher_costs'].cpu()).abs().max()))
            choices.append({'memory_index': index, 'solver_seed': args.seed + index, 'population_indices': scene_choices})
            print(f'[CEM_MEMORY {index + 1}/64]', flush=True)
            del capture, solver
    torch.testing.assert_close(bank['candidates'][:, :4], original['candidates'][:, :4], rtol=0, atol=0)
    torch.testing.assert_close(bank['teacher_costs'][:, :4], original['teacher_costs'][:, :4], rtol=0, atol=0)
    bank['manifest'].update(
        candidate_source='4 original Gaussian + 2 native CEM iteration 16 + 2 native CEM iteration 30',
        parent_memory=str(resolve(args.memory)), native_solver_config=solver_cfg,
        native_plan_config=plan_cfg, search_seed=args.seed, candidate_selection=choices,
        selection_rule='Uniform without replacement within each old-guided population; no new-model or probe labels.',
        native_cost_reproduction_max_abs_diff=reproduction,
        preparation_seconds=time.perf_counter() - started,
        preparation_peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    )
    # The original check used a different candidate bank; run it again on this one.
    model.requires_grad_(True)
    bank['manifest']['gradient_check'] = gradient_check(model, bank, device)
    bank['manifest']['tensor_bytes'] = sum(
        value.numel() * value.element_size() for value in bank.values() if torch.is_tensor(value)
    )
    if bank['manifest']['tensor_bytes'] != manifest['tensor_bytes']:
        raise RuntimeError('Candidate variant changed memory tensor storage size.')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output)
    manifest_path.write_text(json.dumps(bank['manifest'], ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'[CEM_MEMORY_DONE] memory={output}; scenes=64; candidates=8; tensor_bytes={bank["manifest"]["tensor_bytes"]}', flush=True)


if __name__ == '__main__':
    main()
