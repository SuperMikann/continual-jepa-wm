"""Physics interventions and native evaluation workers."""
import json
import os
from pathlib import Path
import runpy
import sys

from .common import CASES, emit, save, configuration, validate


def apply_condition(raw, condition):
    """Apply multipliers to original physical parameters, once per reset.

    The inner step receives transformed actions; the outer info wrapper
    still records the agent's command, which is the world model's action input.
    """
    import numpy as np

    physics = raw.env.physics
    model = physics.model
    if (model.nq, model.nv, model.nu) != (2, 2, 2):
        raise ValueError('This screen requires the two-joint Reacher model')
    previous = getattr(raw, '_dynamics_screen', None)
    if previous is None or previous['physics'] is not physics:
        original_step = raw.step if previous is None else previous['original_step']
        previous = dict(physics=physics, original_step=original_step,
                        damping=np.array(model.dof_damping, copy=True),
                        gear=np.array(model.actuator_gear, copy=True))
    model.dof_damping[:] = previous['damping'] * condition['damping']
    model.actuator_gear[:] = previous['gear']
    model.actuator_gear[:, 0] *= condition['motor']
    expected_damping = previous['damping'] * condition['damping']
    expected_gear = previous['gear'][:, 0] * condition['motor']
    if not (np.allclose(model.dof_damping, expected_damping)
            and np.allclose(model.actuator_gear[:, 0], expected_gear)):
        raise RuntimeError('Physical parameter application failed')
    angle = np.deg2rad(condition['rotation_deg'])
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    stats = dict(command_scalars=0, command_outside=0, mapping_outside=0)
    previous['stats'] = stats
    previous['last_command'] = None
    raw._dynamics_screen = previous

    def step(command):
        command = np.asarray(command)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError(f'Invalid active Reacher command: {command}')
        previous['last_command'] = command.copy()
        stats['command_scalars'] += command.size
        stats['command_outside'] += int((np.abs(command) > 1).sum())
        applied = command
        if condition['rotation_deg'] != 0:
            # Rotate the bounded motor command; clip the rotated result again.
            applied = rotation @ np.clip(command, -1, 1)
            stats['mapping_outside'] += int((np.abs(applied) > 1).sum())
            applied = np.clip(applied, -1, 1).astype(command.dtype)
        return previous['original_step'](applied)

    raw.step = step
    return dict(damping=np.asarray(model.dof_damping).tolist(),
                motor_gear=np.asarray(model.actuator_gear[:, 0]).tolist(),
                rotation_deg=condition['rotation_deg'])


def build_world(base_world, case, config, output):
    import numpy as np

    class ScreenWorld(base_world):
        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            self._applied = [apply_condition(e.unwrapped, CASES[case])
                             for e in self.envs.envs]
            emit('PHYSICS_APPLIED', dict(case=case, environments=len(self._applied),
                                         first=self._applied[0]))
            return result

        def _run(self, *args, on_step=None, **kwargs):
            self._traces = [[] for _ in range(self.num_envs)]
            self._initial = []
            last_time = []
            for e in self.envs.envs:
                raw = e.unwrapped
                physics = raw.env.physics
                goal = np.asarray(raw.env.task.target_qpos).reshape(-1)
                if goal.shape != (2,) or not np.isfinite(goal).all():
                    raise ValueError('Missing physical goal joint configuration')
                self._initial.append(dict(qpos=physics.data.qpos.copy().tolist(),
                                          qvel=physics.data.qvel.copy().tolist(),
                                          goal_qpos=goal.tolist()))
                last_time.append(float(physics.data.time))
            progress = 0

            def record(world, mask):
                nonlocal progress
                if on_step is not None:
                    on_step(world, mask)
                active = np.ones(self.num_envs, dtype=bool) if mask is None else mask
                stepped = False
                for i in np.flatnonzero(active):
                    raw = self.envs.envs[i].unwrapped
                    physics = raw.env.physics
                    now = float(physics.data.time)
                    if np.isclose(now, last_time[i], atol=1e-12, rtol=0):
                        continue  # The upstream callback also fires at resets.
                    if now < last_time[i]:
                        raise RuntimeError('Unexpected episode reset during screen')
                    last_time[i] = now
                    stepped = True
                    command = raw._dynamics_screen['last_command']
                    history = np.asarray(world.infos['action'][i]).reshape(-1, 2)[-1]
                    if not np.allclose(history, command, equal_nan=False):
                        raise RuntimeError('Action history must store commands, not mapped actions')
                    qpos = np.asarray(physics.data.qpos)
                    goal = np.asarray(self._initial[i]['goal_qpos'])
                    error = float(np.sqrt(np.mean((qpos - goal) ** 2)))
                    if not np.isfinite(error):
                        raise ValueError('Non-finite physical error')
                    self._traces[i].append(dict(qpos_rmse=error,
                                                success=bool(world.terminateds[i]),
                                                truncated=bool(world.truncateds[i])))
                if stepped:
                    progress += 1
                    if progress % 10 == 0:
                        emit('PROGRESS', dict(case=case, step=progress,
                                              successes=int(world.terminateds.sum())))

            return super()._run(*args, on_step=record, **kwargs)

        def evaluate(self, *args, **kwargs):
            if args or kwargs.get('dataset') is None:
                raise ValueError('Only the native dataset-driven evaluation is supported')
            if config.get('scene_manifest'):
                scenes = json.loads(Path(config['scene_manifest']).read_text(encoding='utf-8'))
                if len(scenes['scenes']) != config['num_eval']:
                    raise ValueError('Scene manifest size differs from the evaluation size')
                kwargs['episodes_idx'] = [s['episode_idx'] for s in scenes['scenes']]
                kwargs['start_steps'] = [s['start_step'] for s in scenes['scenes']]
            kwargs['video'] = None
            kwargs['reset_mode'] = 'wait'
            budget = int(kwargs['eval_budget'])
            if budget != 50 or int(kwargs['goal_offset']) != 25:
                raise ValueError('Keep the existing 50-step budget and 25-step goal')
            methods = {s['method'] for s in kwargs.get('callables') or []}
            if not {'set_state', 'set_target_qpos'} <= methods:
                raise ValueError('Both physical state and goal restoration are required')
            plan = self.policy.cfg
            planning = {k: getattr(plan, k) for k in
                        ('horizon', 'receding_horizon', 'action_block', 'warm_start')}
            if any(planning[k] != 5 for k in ('horizon', 'receding_horizon', 'action_block')):
                raise ValueError('Keep the archived horizon/replanning/action-block configuration')
            native = super().evaluate(*args, **kwargs)
            flags = np.asarray(native['episode_successes'], dtype=bool)
            rows = []
            for i, trace in enumerate(self._traces):
                if not trace or len(trace) > budget or any(x['truncated'] for x in trace):
                    raise RuntimeError(f'Invalid scene trace {i}')
                first_hit = next((j + 1 for j, row in enumerate(trace) if row['success']), None)
                if bool(first_hit) != bool(flags[i]) or (first_hit is None and len(trace) != budget):
                    raise RuntimeError('Trace success/step budget differs from the native result')
                if first_hit is not None and first_hit != len(trace):
                    raise RuntimeError('Successful environments must stop at their first hit')
                rows.append(dict(episode_idx=int(kwargs['episodes_idx'][i]),
                                 start_step=int(kwargs['start_steps'][i]),
                                 initial=self._initial[i], success=bool(flags[i]),
                                 first_success_step=first_hit, active_steps=len(trace),
                                 steps_to_success_or_budget=first_hit or budget,
                                 last_active_qpos_rmse=trace[-1]['qpos_rmse']))
            rate = 100 * float(flags.mean())
            if not np.isclose(rate, native['success_rate']):
                raise RuntimeError('Native success rate differs from per-scene flags')
            hits = [r['first_success_step'] for r in rows if r['success']]
            states = [e.unwrapped._dynamics_screen['stats'] for e in self.envs.envs]
            scalars = sum(s['command_scalars'] for s in states)
            summary = dict(success_rate=rate,
                           successful_scene_count=len(hits),
                           median_success_step=float(np.median(hits)) if hits else None,
                           mean_steps_to_success_or_budget=float(np.mean([
                               r['steps_to_success_or_budget'] for r in rows])),
                           mean_last_active_qpos_rmse=float(np.mean([
                               r['last_active_qpos_rmse'] for r in rows])),
                           command_outside_fraction=sum(s['command_outside'] for s in states)/scalars,
                           mapping_clip_fraction=sum(s['mapping_outside'] for s in states)/scalars)
            payload = dict(schema=1, status='evaluated', config=config, case=case,
                           condition=CASES[case], planning=planning, eval_budget=budget,
                           goal_offset=25, applied_parameters=self._applied,
                           notes=['Exploratory scenes from the existing T1 dataset; not an independent final test.',
                                  'Native first-hit success; successful environments stop early.',
                                  'Last active error is not a full 50-step endpoint error.',
                                  'Fixed command rotation, not an exact JEPA-TTT reproduction.'],
                           summary=summary, per_scene=rows)
            save(output, payload)
            emit('CASE_RESULT', dict(case=case, **summary))
            return native

    return ScreenWorld


def worker(args):
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    os.chdir(args.root)
    sys.path.insert(0, str(args.root / 'cl_protocol'))
    import numpy as np
    import torch
    import stable_worldmodel as swm
    from drift_world import DriftWorld
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = args.out_dir / f'{args.case}.json'
    pending = args.out_dir / f'{args.case}.pending.json'
    if output.exists():
        raise FileExistsError(output)
    original = swm.World
    original_argv = sys.argv
    swm.World = build_world(DriftWorld, args.case, configuration(args), pending)
    script = args.root / 'scripts/plan/eval_wm.py'
    sys.argv = [str(script), '--config-name', 'reacher', f'policy={args.policy}',
                f'eval.dataset_name={args.dataset}', f'eval.num_eval={args.num_eval}',
                f'seed={args.seed}', '++compile=false',
                f'output.filename={args.out_dir / (args.case + ".native.txt")}']
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        swm.World = original
        sys.argv = original_argv
    payload = json.loads(pending.read_text(encoding='utf-8'))
    if not (args.out_dir / (args.case + '.native.txt')).is_file():
        raise RuntimeError('The native evaluation entrypoint did not finish writing its result')
    payload['status'] = 'complete'
    validate(payload, args, args.case)
    save(output, payload)
    emit('CASE_DONE', args.case)


def prediction_worker(args):
    import numpy as np
    import torch
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, str(args.root/'cl_protocol'))
    from pred_error import img_transform
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    home = Path(os.environ['STABLEWM_HOME'])
    for name in [cfg['scaler_dataset'], *cfg['datasets'].values()]:
        if not (home/'datasets'/name/'_versions').is_dir():
            raise FileNotFoundError(name)
    source = swm.data.load_dataset(cfg['scaler_dataset'], keys_to_cache=['action'])
    action = np.asarray(source.get_col_data('action'))
    scaler = StandardScaler().fit(action[np.isfinite(action).all(axis=1)])
    del source, action
    model = swm.wm.utils.load_pretrained(cfg['policy']).to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if getattr(model.predictor, 'num_frames', 3) != 3:
        raise ValueError('Prediction windows require the archived three-frame history')
    transform = img_transform()
    output = dict(status='running', config=cfg, results={})
    with torch.inference_mode():
        for case, name in cfg['datasets'].items():
            dataset = swm.data.load_dataset(name, keys_to_cache=['action'])
            windows = json.loads(Path(cfg['windows'][case]).read_text(encoding='utf-8'))
            errors, targets, per_window = [], [], []
            for offset in range(0, len(windows), cfg['batch']):
                rows = windows[offset:offset+cfg['batch']]
                chunks = dataset.load_chunk(
                    np.array([r['episode_idx'] for r in rows]),
                    np.array([r['start_step'] for r in rows]),
                    np.array([r['start_step']+16 for r in rows]))
                pixels, actions, goals = [], [], []
                for chunk in chunks:
                    act = np.asarray(chunk['action'])[:15]
                    if act.shape != (15, 2) or np.isinf(act).any():
                        raise ValueError('Invalid prediction-window actions')
                    act = np.nan_to_num(act, nan=0.0)  # Same reset-action convention as native training.
                    actions.append(torch.from_numpy(scaler.transform(act).astype('float32').reshape(3, 10)))
                    pixels.append(torch.stack([transform(chunk['pixels'][i]) for i in (0, 5, 10)]))
                    goals.append(transform(chunk['pixels'][15]))
                encoded = model.encode(dict(pixels=torch.stack(pixels).cuda(),
                                             action=torch.stack(actions).cuda()))
                predicted = model.predict(encoded['emb'], encoded['act_emb'])[:, -1]
                target = model.encode(dict(pixels=torch.stack(goals)[:, None].cuda()))['emb'][:, 0]
                mse = (predicted-target).square().mean(-1).float().cpu().numpy()
                if not np.isfinite(mse).all() or len(mse) != len(rows):
                    raise ValueError('Non-finite or incomplete prediction scores')
                errors.extend(mse.tolist())
                targets.append(target.float().cpu().numpy())
                per_window.extend(dict(row, mse=float(value)) for row, value in zip(rows, mse))
            target_array = np.concatenate(targets, axis=0)
            variance = float(target_array.var(axis=0).mean())
            result = dict(mse_mean=float(np.mean(errors)), mse_std=float(np.std(errors)),
                          n_windows=len(errors), target_variance=variance,
                          normalized_mse=float(np.mean(errors)/variance) if variance > 1e-12 else None,
                          per_window=per_window)
            output['results'][case] = result
            emit('PREDICTION_RESULT', dict(policy=cfg['policy'], case=case,
                                         **{k: v for k, v in result.items() if k != 'per_window'}))
            del dataset
    output['status'] = 'complete'
    save(Path(cfg['output']), output)
    emit('PREDICTION_DONE', cfg['output'])
