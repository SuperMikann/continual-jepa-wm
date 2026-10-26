"""Measure native locomotion reward from image-goal MPC or a SAC expert.

This keeps eval_wm.py's dataset sampling and fixed evaluation budget. For model
checkpoints it also keeps the image/action preprocessing, GoalMSE, and CEM.
The SAC expert reads physical state and acts without a planner. No pose-match
termination is used in either case.
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')

import numpy as np

PROTOCOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTOCOL))

import stable_worldmodel as swm

from drift_world import DriftWorld, make_darken
from eval_task import load_seq


def scalar(info, key, index):
    if key not in info:
        raise KeyError(f'Missing physical measurement {key!r}')
    value = np.asarray(info[key][index]).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError(f'Invalid {key} for environment {index}: {value}')
    return float(value[0])


def aggregate(records, key):
    values = np.asarray([r[key] for r in records], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite {key}')
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'n': int(len(values)),
    }


def build_world_class(kind, variation_values, darken, out_path, config):
    physical_task = 'walk' if kind == 'walker' else 'run'

    class LocomotionWorld(DriftWorld):
        def __init__(self, *args, **kwargs):
            # The native planning config uses Walker qpos_match. Its target
            # image remains in the input, but physical reward comes from walk.
            kwargs['task'] = physical_task
            kwargs['variation_values'] = variation_values
            kwargs['image_transform'] = make_darken(darken)
            super().__init__(*args, **kwargs)
            self._traces = [[] for _ in range(self.num_envs)]

        def _run(self, *args, on_step=None, **kwargs):
            if kwargs.get('max_steps') is None:
                raise ValueError('Locomotion evaluation needs a fixed step budget')
            previous_reward = [
                float(env.unwrapped._cumulative_reward)
                for env in self.envs.envs
            ]
            previous_time = [
                float(env.unwrapped.env.physics.data.time)
                for env in self.envs.envs
            ]

            def record_step(world, mask):
                if on_step is not None:
                    on_step(world, mask)
                active = np.ones(world.num_envs, dtype=bool) if mask is None else mask
                for i in np.flatnonzero(active):
                    raw = world.envs.envs[i].unwrapped
                    total_reward = float(raw._cumulative_reward)
                    reward = total_reward - previous_reward[i]
                    previous_reward[i] = total_reward
                    wrapped_reward = float(world.rewards[i])
                    info_reward = scalar(world.infos, 'reward', i)
                    if not np.isclose(reward, wrapped_reward, atol=1e-4, rtol=1e-5):
                        raise ValueError('Outer reward differs from native DMC reward')
                    if not np.isclose(reward, info_reward, atol=1e-4, rtol=1e-5):
                        raise ValueError('Info reward differs from native DMC reward')

                    now = float(raw.env.physics.data.time)
                    dt = now - previous_time[i]
                    previous_time[i] = now
                    if not np.isfinite(dt) or dt <= 0:
                        raise ValueError(f'Invalid simulation time increment: {dt}')
                    step = {
                        'reward': reward,
                        'speed': scalar(world.infos, 'speed', i),
                        'dt_seconds': dt,
                        'terminated': bool(world.terminateds[i]),
                        'truncated': bool(world.truncateds[i]),
                    }
                    if kind == 'walker':
                        step['torso_height'] = scalar(world.infos, 'torso_height', i)
                        step['torso_upright'] = scalar(world.infos, 'torso_upright', i)
                    if 'goal_qpos' in world.infos:
                        qpos = np.asarray(world.infos['qpos'][i]).reshape(-1)
                        goal = np.asarray(world.infos['goal_qpos'][i]).reshape(-1)
                        if qpos.shape != goal.shape or not np.isfinite(qpos).all():
                            raise ValueError('Invalid qpos/goal_qpos for pose record')
                        step['qpos_rmse'] = float(np.sqrt(np.mean((qpos - goal) ** 2)))
                    self._traces[i].append(step)

            return super()._run(*args, on_step=record_step, **kwargs)

        def evaluate(self, *args, **kwargs):
            # set_target_qpos belongs to the pose-match task and is invalid
            # under walk/run. State restoration still uses the native callable.
            specs = kwargs.get('callables') or []
            kwargs['callables'] = [
                spec for spec in specs if spec['method'] != 'set_target_qpos'
            ]
            if not any(spec['method'] == 'set_state' for spec in kwargs['callables']):
                raise ValueError('Dataset state restoration is required')
            budget = int(kwargs['eval_budget'])
            native_result = super().evaluate(*args, **kwargs)
            records = []
            for i, trace in enumerate(self._traces):
                if len(trace) != budget:
                    raise ValueError(
                        f'Environment {i}: {len(trace)} of {budget} steps; '
                        'fixed-duration comparison would be invalid'
                    )
                duration = sum(row['dt_seconds'] for row in trace)
                item = {
                    'episode_idx': int(kwargs['episodes_idx'][i]),
                    'start_step': int(kwargs['start_steps'][i]),
                    'return': float(sum(row['reward'] for row in trace)),
                    'mean_speed': float(sum(
                        row['speed'] * row['dt_seconds'] for row in trace
                    ) / duration),
                    'simulation_seconds': float(duration),
                    'environment_steps': len(trace),
                    'ended_early': any(
                        row['terminated'] or row['truncated']
                        for row in trace[:-1]
                    ),
                }
                if kind == 'walker':
                    item['mean_torso_height'] = float(np.mean([
                        row['torso_height'] for row in trace
                    ]))
                    item['mean_torso_upright'] = float(np.mean([
                        row['torso_upright'] for row in trace
                    ]))
                    item['standing_time_fraction'] = float(sum(
                        row['dt_seconds']
                        for row in trace
                        if row['torso_height'] >= 1.2
                        and row['torso_upright'] >= 0.8
                    ) / duration)
                if 'qpos_rmse' in trace[-1]:
                    item['final_qpos_rmse'] = trace[-1]['qpos_rmse']
                records.append(item)

            if any(record['ended_early'] for record in records):
                raise ValueError('Physical task ended before the fixed budget')
            config['eval_budget'] = budget
            config['goal_offset_steps'] = int(kwargs['goal_offset'])
            summary_keys = ['return', 'mean_speed', 'simulation_seconds']
            if kind == 'walker':
                summary_keys.extend([
                    'mean_torso_height', 'mean_torso_upright',
                    'standing_time_fraction',
                ])
            summary = {key: aggregate(records, key) for key in summary_keys}
            payload = {
                'config': config,
                'physical_task': physical_task,
                'decision_rule': config['decision_rule'],
                'reward_source': 'native DMC cumulative reward delta, cross-checked against outer reward',
                'native_pose_success_unused': float(native_result['success_rate']),
                'summary': summary,
                'per_scene': records,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding='utf-8',
            )
            print('SEMANTIC_SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)
            return {'semantic_summary': summary, 'semantic_json': str(out_path)}

    return LocomotionWorld


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', required=True, choices=('C_walker', 'D_cheetah'))
    parser.add_argument('--task', required=True, choices=('T1', 'T2', 'T3'))
    parser.add_argument('--policy', required=True, help='Checkpoint path, random, or expert')
    parser.add_argument('--expert-policy', type=Path, help='SAC expert_policy.zip, required for expert')
    parser.add_argument('--vecnorm', type=Path, help='SAC vec_normalize.pkl, required for expert')
    parser.add_argument('--num-eval', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--eval-dataset', help='Optional independent evaluation dataset')
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/stable-worldmodel'))
    args = parser.parse_args()
    if args.num_eval <= 0:
        parser.error('--num-eval must be positive')
    root = args.root.resolve()
    out = args.out.resolve()
    if out.exists():
        parser.error(f'Keep existing result; choose another --out: {out}')
    seq = load_seq(PROTOCOL / 'tasks.yaml', args.sequence)
    task = seq['tasks'][args.task]
    kind = 'walker' if args.sequence == 'C_walker' else 'cheetah'
    eval_script = root / 'scripts' / 'plan' / 'eval_wm.py'
    if not eval_script.is_file():
        raise FileNotFoundError(eval_script)
    if args.policy == 'expert':
        if args.expert_policy is None or args.vecnorm is None:
            parser.error('expert requires --expert-policy and --vecnorm')
        for source in (args.expert_policy, args.vecnorm):
            if not source.is_file():
                raise FileNotFoundError(source)
    elif args.policy != 'random':
        checkpoint = Path(args.policy)
        if not checkpoint.is_absolute():
            checkpoint = Path(os.environ['STABLEWM_HOME']) / 'checkpoints' / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    config = {
        'sequence': args.sequence,
        'task': args.task,
        'policy': args.policy,
        'expert_policy': str(args.expert_policy.resolve()) if args.policy == 'expert' else None,
        'vecnorm': str(args.vecnorm.resolve()) if args.policy == 'expert' else None,
        'decision_rule': ('sac_expert_state_policy' if args.policy == 'expert'
                          else 'random_actions' if args.policy == 'random'
                          else 'goal_mse_cem'),
        'dataset': args.eval_dataset or task['dataset'],
        'variation_values': task.get('variation_values') or {},
        'num_eval': args.num_eval,
        'seed': args.seed,
        'eval_budget': None,
        'goal_offset_steps': None,
        'standing_thresholds': {'torso_height': 1.2, 'torso_upright': 0.8}
        if kind == 'walker' else None,
    }
    original_world = swm.World
    original_random_policy = swm.policy.RandomPolicy
    original_argv = sys.argv
    if args.policy == 'expert':
        # eval_wm.py's random branch avoids loading a world-model checkpoint.
        # Substitute the already-used collection SAC adapter only for this run.
        from collect_task import SACPolicy
        swm.policy.RandomPolicy = lambda: SACPolicy(
            str(args.expert_policy), str(args.vecnorm)
        )
    swm.World = build_world_class(
        kind, config['variation_values'], task.get('darken', 1.0), out, config
    )
    native_text = out.with_suffix('.native.txt')
    sys.argv = [
        str(eval_script), '--config-name', seq['eval_config'],
        f'policy={"random" if args.policy == "expert" else args.policy}',
        f'eval.dataset_name={config["dataset"]}',
        f'eval.num_eval={args.num_eval}',
        f'seed={args.seed}',
        f'output.filename={native_text}',
    ]
    try:
        runpy.run_path(str(eval_script), run_name='__main__')
    finally:
        sys.argv = original_argv
        swm.World = original_world
        swm.policy.RandomPolicy = original_random_policy
    if not out.is_file() or not native_text.is_file():
        raise RuntimeError('Expected both semantic JSON and native evaluation log')
    print('[SEMANTIC_DONE]', out, flush=True)


if __name__ == '__main__':
    main()
