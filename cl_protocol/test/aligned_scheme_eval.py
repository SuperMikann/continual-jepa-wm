"""Audit image-goal candidate ranking with observed history and real rollouts aligned.

The original scheme_eval.py conditions on dataset frames at t, t+fs, t+2fs,
but replays every candidate from t. This makes its measured outcome differ from
the history seen by the model. Here the first two action blocks are fixed from
the dataset; only actions after the last observed frame are varied and replayed.
This is a diagnostic of candidate prediction, not a native CEM policy run.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

import numpy as np
import torch
import yaml
from sklearn import preprocessing

PROTOCOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTOCOL))

import stable_worldmodel as swm
from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator

from drift_world import DriftWorld, make_darken
from scheme_eval import BakedPolicy, episode_col, img_transform, kendall_tau, true_cost


def summarize(records, key):
    values = np.asarray([record[key] for record in records], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite {key}')
    return {'mean': float(values.mean()), 'std': float(values.std()), 'n': len(values)}


class NativeReturnWorld(DriftWorld):
    """Accumulate each candidate's original DMC reward during a fixed rollout."""

    def _run(self, *args, on_step=None, **kwargs):
        budget = kwargs.get('max_steps')
        if budget is None:
            raise ValueError('Native return needs a fixed evaluation budget')
        previous = np.array([
            float(env.unwrapped._cumulative_reward) for env in self.envs.envs
        ])
        self.candidate_returns = np.zeros(self.num_envs, dtype=np.float64)
        counts = np.zeros(self.num_envs, dtype=np.int64)

        def record_step(world, mask):
            if on_step is not None:
                on_step(world, mask)
            active = np.ones(world.num_envs, dtype=bool) if mask is None else mask
            for i in np.flatnonzero(active):
                total = float(world.envs.envs[i].unwrapped._cumulative_reward)
                reward = total - previous[i]
                previous[i] = total
                outer = float(world.rewards[i])
                info_reward = np.asarray(world.infos['reward'][i]).reshape(-1)
                if info_reward.size != 1 or not np.isfinite(info_reward[0]):
                    raise ValueError(f'Invalid reward info for candidate {i}')
                if not (np.isclose(reward, outer, atol=1e-4, rtol=1e-5)
                        and np.isclose(reward, float(info_reward[0]), atol=1e-4, rtol=1e-5)):
                    raise ValueError(f'Native/outer/info reward mismatch for candidate {i}')
                self.candidate_returns[i] += reward
                counts[i] += 1

        result = super()._run(*args, on_step=record_step, **kwargs)
        if not np.all(counts == budget):
            raise ValueError(f'Candidates ended before {budget} steps: {counts.tolist()}')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', required=True, choices=('B_reacher', 'C_walker', 'D_cheetah'))
    parser.add_argument('--task', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--tasks', type=Path, default=PROTOCOL / 'tasks.yaml')
    parser.add_argument('--num-scenes', type=int, default=5)
    parser.add_argument('--num-candidates', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--frameskip', type=int, default=5)
    parser.add_argument('--goal-offset', type=int, default=25)
    parser.add_argument('--holdout', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clip-to-action-space', action='store_true',
                        help='Clamp future raw actions to environment bounds, then restandardize for model scoring')
    parser.add_argument('--replay-dataset-future', action='store_true',
                        help='Use candidate 0 to replay the recorded future actions and check endpoint alignment')
    parser.add_argument('--future-noise-std', type=float, default=None,
                        help='Sample all future candidates near recorded actions, with this standardized noise scale')
    parser.add_argument('--record-native-reward', action='store_true',
                        help='Use walk/run task and record each candidate original DMC reward')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if min(args.num_scenes, args.num_candidates, args.frameskip, args.holdout) < 1:
        parser.error('scene/candidate/frameskip/holdout counts must be positive')
    if args.num_candidates < 2 or args.horizon < 3:
        parser.error('at least two candidates and three horizon blocks are required')
    if args.goal_offset != args.horizon * args.frameskip:
        parser.error('goal offset must equal horizon × frameskip')
    if args.future_noise_std is not None and not np.isfinite(args.future_noise_std):
        parser.error('future noise scale must be finite')
    if args.future_noise_std is not None and args.future_noise_std <= 0:
        parser.error('future noise scale must be positive')
    if args.record_native_reward and args.sequence not in ('C_walker', 'D_cheetah'):
        parser.error('native reward recording supports Walker and Cheetah only')
    out = args.out.resolve()
    if out.exists():
        parser.error(f'Choose a new output path; existing result is preserved: {out}')

    spec = yaml.safe_load(args.tasks.read_text(encoding='utf-8'))['sequences'][args.sequence]
    task = spec['tasks'][args.task]
    dataset = swm.data.load_dataset(task['dataset'], keys_to_cache=['pixels', 'action'])
    columns = set(dataset.column_names) | set(getattr(dataset, '_schema_names', ()))
    if not {'qpos', 'qvel'} <= columns:
        raise ValueError('Dataset must expose qpos and qvel for exact state restoration')
    ep_col = episode_col(dataset)
    episodes = np.asarray(dataset.get_col_data(ep_col))
    steps = np.asarray(dataset.get_col_data('step_idx'))
    unique_episodes = np.unique(episodes)
    if len(unique_episodes) < args.holdout:
        raise ValueError(f'Only {len(unique_episodes)} episodes, holdout={args.holdout}')
    episode_length = {
        ep: int(steps[episodes == ep].max()) + 1 for ep in unique_episodes
    }
    pool = []
    # Keep the scene pool fixed when comparing 5/10/15-step futures. A shorter
    # goal offset must not change which episode/start pairs seed=0 selects.
    pool_goal_offset = max(args.goal_offset, 25)
    for ep in unique_episodes[-args.holdout:]:
        latest_start = episode_length[ep] - pool_goal_offset - 1
        pool.extend((int(ep), int(step)) for step in np.unique(steps[episodes == ep])
                    if step <= latest_start)
    if len(pool) < args.num_scenes:
        raise ValueError(f'Only {len(pool)} valid scenes')
    generator = np.random.default_rng(args.seed)
    scenes = [pool[i] for i in generator.choice(len(pool), args.num_scenes, replace=False)]

    all_actions = np.asarray(dataset.get_col_data('action'))
    if all_actions.ndim != 2:
        raise ValueError(f'Expected 2D action column, got {all_actions.shape}')
    action_dim = all_actions.shape[1]
    block_dim = action_dim * args.frameskip
    scaler = preprocessing.StandardScaler()
    scaler.fit(all_actions[np.isfinite(all_actions).all(axis=1)])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False)
    history = int(getattr(model.predictor, 'num_frames', 3))
    if history != 3:
        raise ValueError(f'This audit expects three observation frames, got {history}')
    fixed_blocks = history - 1
    future_blocks = args.horizon - fixed_blocks
    future_steps = future_blocks * args.frameskip
    evaluator = ShootingCostEvaluator(model, GoalMSE())
    torch_generator = torch.Generator().manual_seed(args.seed)
    # Draw the full 15-step noise once, then use the same prefix for shorter
    # horizons so candidate i remains paired across 5/10/15-step tests.
    future = torch.randn(1, args.num_candidates, max(3, future_blocks), block_dim,
                         generator=torch_generator)[:, :, :future_blocks]
    future_noise = future.clone()

    world_class = NativeReturnWorld if args.record_native_reward else DriftWorld
    task_override = (
        {'task': 'walk' if args.sequence == 'C_walker' else 'run'}
        if args.record_native_reward else {}
    )
    world = world_class(
        spec['env'], num_envs=args.num_candidates, image_shape=(224, 224),
        max_episode_steps=spec['max_episode_steps'],
        image_transform=make_darken(task.get('darken', 1.0)),
        variation_values=task.get('variation_values') or {},
        **task_override,
    )
    baked = BakedPolicy()
    world.set_policy(baked)
    action_low = np.asarray(baked.env.single_action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(baked.env.single_action_space.high, dtype=np.float32).reshape(-1)
    if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
        raise ValueError(f'Unexpected action bounds: {action_low.shape}, {action_high.shape}')
    if args.clip_to_action_space and not (
        np.isfinite(action_low).all() and np.isfinite(action_high).all()
    ):
        raise ValueError('Cannot clip future actions: environment bounds are not finite')
    raw_future = scaler.inverse_transform(
        future[0].reshape(-1, action_dim).numpy()
    ).astype(np.float32)
    outside = (raw_future < action_low) | (raw_future > action_high)
    print(f'[ACTION_RANGE] low={action_low.tolist()} high={action_high.tolist()} '
          f'gaussian_outside_fraction={outside.mean():.6f} '
          f'clip_to_action_space={args.clip_to_action_space} '
          f'future_noise_std={args.future_noise_std}', flush=True)
    if args.clip_to_action_space:
        raw_future = np.clip(raw_future, action_low, action_high)
        future = torch.from_numpy(
            scaler.transform(raw_future).astype(np.float32)
            .reshape(1, args.num_candidates, future_blocks, block_dim)
        )
    raw_future = raw_future.reshape(args.num_candidates, future_steps, action_dim)
    # Pose-match termination is irrelevant to endpoint image fidelity. Do not
    # install a target pose; only restore qpos/qvel at the last observed frame.
    callables = [{'method': 'set_state', 'args': {
        'qpos': {'value': 'qpos'}, 'qvel': {'value': 'qvel'},
    }}]
    transform = img_transform()
    records = []
    candidate_outside_fractions = []
    try:
        for index, (ep, start) in enumerate(scenes, 1):
            chunk = dataset.load_chunk(
                np.array([ep]), np.array([start]),
                np.array([start + args.goal_offset + 1]),
            )[0]
            pixels = chunk['pixels']
            raw_actions = chunk['action'].numpy() if hasattr(chunk['action'], 'numpy') \
                else np.asarray(chunk['action'])
            history_frames = torch.stack([
                transform(pixels[i * args.frameskip]) for i in range(history)
            ])
            goal = transform(pixels[args.goal_offset])
            history_actions = scaler.transform(
                raw_actions[:history * args.frameskip]
            ).astype(np.float32).reshape(history, block_dim)
            fixed = torch.from_numpy(history_actions[:fixed_blocks])[None, None]
            scene_future = future
            scene_raw_future = raw_future
            candidate_outside_fraction = float(outside.mean())
            recorded_future = None
            if args.replay_dataset_future or args.future_noise_std is not None:
                recorded_future = np.asarray(
                    raw_actions[fixed_blocks * args.frameskip:args.goal_offset],
                    dtype=np.float32,
                )
                if recorded_future.shape != (future_steps, action_dim):
                    raise ValueError(f'Unexpected recorded future shape: {recorded_future.shape}')
            if args.future_noise_std is not None:
                center = torch.from_numpy(
                    scaler.transform(recorded_future).astype(np.float32)
                    .reshape(1, 1, future_blocks, block_dim)
                )
                scene_future = center + args.future_noise_std * future_noise
                scene_raw_future = scaler.inverse_transform(
                    scene_future.reshape(-1, action_dim).numpy()
                ).astype(np.float32).reshape(
                    args.num_candidates, future_steps, action_dim
                )
                candidate_outside_fraction = float(np.mean(
                    (scene_raw_future < action_low) | (scene_raw_future > action_high)
                ))
                if args.clip_to_action_space:
                    scene_raw_future = np.clip(
                        scene_raw_future, action_low, action_high
                    )
                    scene_future = torch.from_numpy(
                        scaler.transform(
                            scene_raw_future.reshape(-1, action_dim)
                        ).astype(np.float32).reshape(
                            1, args.num_candidates, future_blocks, block_dim
                        )
                    )
            candidate_outside_fractions.append(candidate_outside_fraction)
            if args.replay_dataset_future:
                scene_raw_future = scene_raw_future.copy()
                scene_raw_future[0] = recorded_future
                scene_future = scene_future.clone()
                scene_future[0, 0] = torch.from_numpy(
                    scaler.transform(recorded_future).astype(np.float32)
                    .reshape(future_blocks, block_dim)
                )
            candidates = torch.cat([
                fixed.expand(1, args.num_candidates, fixed_blocks, block_dim),
                scene_future,
            ], dim=2)
            info = {
                'pixels': history_frames[None, None].expand(
                    1, args.num_candidates, history, *history_frames.shape[1:]
                ).clone(),
                'goal': goal[None, None, None].expand(
                    1, args.num_candidates, 1, *goal.shape
                ).clone(),
                'action': torch.from_numpy(history_actions)[None, None].expand(
                    1, args.num_candidates, history, block_dim
                ).clone(),
            }
            with torch.no_grad():
                predicted_cost = evaluator.get_cost(
                    {key: value.to(device) for key, value in info.items()},
                    candidates.to(device),
                )[0].detach().cpu().numpy()

            baked.set_actions([scene_raw_future[i] for i in range(args.num_candidates)])
            rollout_start = start + fixed_blocks * args.frameskip
            world.evaluate(
                dataset=dataset,
                episodes_idx=[ep] * args.num_candidates,
                start_steps=[rollout_start] * args.num_candidates,
                goal_offset=future_steps,
                eval_budget=future_steps,
                callables=callables,
            )
            achieved = world.infos['pixels'][:, -1]
            if achieved.ndim == 5:
                achieved = achieved[:, -1]
            goal_np = np.asarray(
                pixels[args.goal_offset].permute(1, 2, 0).detach().cpu().numpy()
            )
            if np.issubdtype(goal_np.dtype, np.floating):
                goal_np = np.clip(goal_np * 255.0, 0, 255)
            actual_cost = true_cost(model, achieved, goal_np.astype(np.uint8), device)
            tau = kendall_tau(predicted_cost, actual_cost)
            record = {
                'episode_idx': ep, 'start_step': start,
                'rollout_start_step': rollout_start,
                'goal_step': start + args.goal_offset,
                'tau': float(tau),
                'top1_agreement': int(np.argmin(predicted_cost) == np.argmin(actual_cost)),
                'mean_absolute_cost_error': float(np.mean(np.abs(predicted_cost - actual_cost))),
                'predicted_cost_range': float(np.ptp(predicted_cost)),
                'actual_cost_range': float(np.ptp(actual_cost)),
                'predicted_costs': predicted_cost.tolist(),
                'actual_costs': actual_cost.tolist(),
            }
            if args.replay_dataset_future:
                replay_pixels = np.asarray(achieved[0], dtype=np.float32)
                if replay_pixels.shape != goal_np.shape:
                    raise ValueError(f'Replay image shape {replay_pixels.shape} != goal {goal_np.shape}')
                record['replay_pixel_mae'] = float(np.mean(
                    np.abs(replay_pixels - goal_np.astype(np.float32))
                ))
                record['replay_actual_cost'] = float(actual_cost[0])
                record['replay_predicted_cost'] = float(predicted_cost[0])
                print(f'[REPLAY {index}/{args.num_scenes}] ep={ep} start={start} '
                      f'pixel_mae={record["replay_pixel_mae"]:.4f} '
                      f'actual_cost={record["replay_actual_cost"]:.4f} '
                      f'predicted_cost={record["replay_predicted_cost"]:.4f}', flush=True)
            if args.record_native_reward:
                native_returns = world.candidate_returns.copy()
                reward_best = float(native_returns.max())
                record['native_returns'] = native_returns.tolist()
                record['native_return_range'] = float(np.ptp(native_returns))
                record['tau_predicted_cost_vs_negative_return'] = float(
                    kendall_tau(predicted_cost, -native_returns)
                )
                record['tau_actual_cost_vs_negative_return'] = float(
                    kendall_tau(actual_cost, -native_returns)
                )
                record['predicted_choice_reward_gap'] = float(
                    reward_best - native_returns[np.argmin(predicted_cost)]
                )
                record['actual_choice_reward_gap'] = float(
                    reward_best - native_returns[np.argmin(actual_cost)]
                )
                print(f'[NATIVE_REWARD {index}/{args.num_scenes}] '
                      f'predicted_tau={record["tau_predicted_cost_vs_negative_return"]:+.3f} '
                      f'actual_tau={record["tau_actual_cost_vs_negative_return"]:+.3f} '
                      f'return_range={record["native_return_range"]:.3f}', flush=True)
            records.append(record)
            print(f'[SCENE {index}/{args.num_scenes}] ep={ep} start={start} '
                  f'rollout={rollout_start} goal={record["goal_step"]} '
                  f'tau={tau:+.3f} top1={record["top1_agreement"]}', flush=True)
    finally:
        world.close()

    payload = {
        'config': vars(args) | {
            'tasks': str(args.tasks), 'out': str(out),
            'dataset': task['dataset'], 'fixed_blocks': fixed_blocks,
            'future_blocks': future_blocks, 'future_steps': future_steps,
            'pool_goal_offset': pool_goal_offset,
            'action_low': action_low.tolist(), 'action_high': action_high.tolist(),
            'gaussian_reference_outside_fraction': float(outside.mean()),
            'sampled_action_outside_fraction': float(np.mean(candidate_outside_fractions)),
        },
        'interpretation': (
            'Aligned future-candidate image-goal ranking with candidate native reward; not native CEM'
            if args.record_native_reward else
            'Aligned future-candidate image-goal ranking; not native CEM or physical reward'
        ),
        'summary': {
            'tau': summarize(records, 'tau'),
            'top1_agreement': summarize(records, 'top1_agreement'),
            'mean_absolute_cost_error': summarize(records, 'mean_absolute_cost_error'),
            'predicted_cost_range': summarize(records, 'predicted_cost_range'),
            'actual_cost_range': summarize(records, 'actual_cost_range'),
        },
        'per_scene': records,
    }
    if args.replay_dataset_future:
        payload['summary'].update({
            'replay_pixel_mae': summarize(records, 'replay_pixel_mae'),
            'replay_actual_cost': summarize(records, 'replay_actual_cost'),
            'replay_predicted_cost': summarize(records, 'replay_predicted_cost'),
        })
    if args.record_native_reward:
        for key in (
            'native_return_range', 'tau_predicted_cost_vs_negative_return',
            'tau_actual_cost_vs_negative_return', 'predicted_choice_reward_gap',
            'actual_choice_reward_gap',
        ):
            payload['summary'][key] = summarize(records, key)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('[SAMPLED_ACTION_RANGE]', payload['config']['sampled_action_outside_fraction'], flush=True)
    print('[ALIGNED_SUMMARY]', json.dumps(payload['summary'], ensure_ascii=False), flush=True)
    print('[ALIGNED_DONE]', out, flush=True)


if __name__ == '__main__':
    main()
