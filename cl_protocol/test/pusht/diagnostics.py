"""Read-only model, candidate-ranking and history diagnostics."""
import json
import os
from pathlib import Path
import subprocess
import sys

from .training import load_t2_step_control

# The deployment transforms are used only to check the installed source here.
_review_root = str(Path(__file__).resolve().parents[3])
if _review_root not in sys.path:
    sys.path.insert(0, _review_root)
from implementation.planning_history import policy_history_source, future_rollout_source

def probe_prediction_batch(model, pixels, actions, swapped_actions):
    """Inspect last-step predictions in one frozen model's own coordinates."""
    import torch

    with torch.inference_mode():
        history = actions.size(1)
        emb = model.encode({'pixels': pixels})['emb']
        context, target = emb[:, :history], emb[:, history]
        current = context[:, -1]
        prediction = model.predict(context, model.action_encoder(actions))[:, -1]
        effective_delta, true_delta = prediction-current, target-current
        values = dict(prediction_mse=(prediction-target).square().mean(-1),
                      copy_mse=true_delta.square().mean(-1),
                      predicted_delta_mse=effective_delta.square().mean(-1))
        denom = effective_delta.norm(dim=-1)*true_delta.norm(dim=-1)
        values['delta_cosine_valid'] = denom > 1e-10
        values['delta_cosine'] = (effective_delta*true_delta).sum(-1)/denom.clamp_min(1e-10)
        result = dict(values=values, target=target.detach().cpu())
        if hasattr(model.predictor, 'last_delta'):
            raw = model.predictor.last_delta[:, -1].detach().clone()
            mask = model.predictor.last_mask[:, -1].detach().clone()
            difference = (current+mask*raw-prediction).abs().max()
            if not torch.allclose(current+mask*raw, prediction, rtol=1e-5, atol=1e-6):
                raise RuntimeError('Native residual reconstruction failed')
            result.update(mask=mask.cpu(), raw_delta=raw.cpu(),
                          reconstruction_max_abs_diff=float(difference))
        swapped = model.predict(context, model.action_encoder(swapped_actions))[:, -1]
        values['swapped_action_mse'] = (swapped-target).square().mean(-1)
        values['action_response_mse'] = (swapped-prediction).square().mean(-1)
        values['action_input_change_mse'] = (swapped_actions[:, -1]-actions[:, -1]).square().mean(-1)
        if any(not torch.isfinite(v).all() for v in values.values()) or not torch.isfinite(target).all():
            raise ValueError('Non-finite probe measurement')
        result['values'] = {k: v.detach().cpu() for k, v in values.items()}
        return result


def summarize_prediction_probe(batches):
    import torch

    values = {k: torch.cat([b['values'][k] for b in batches]) for k in batches[0]['values']}
    targets = torch.cat([b['target'] for b in batches])
    pred, copy = values['prediction_mse'], values['copy_mse']
    pred_mean, copy_mean = float(pred.mean()), float(copy.mean())
    variance = float(targets.var(0, unbiased=False).mean())
    response = float(values['action_response_mse'].mean())
    valid = values['delta_cosine_valid']
    summary = dict(windows=len(pred), prediction_mse=pred_mean, copy_mse=copy_mean,
        prediction_to_copy_ratio=pred_mean/copy_mean if copy_mean > 1e-12 else None,
        improvement_over_copy=copy_mean-pred_mean,
        better_than_copy_fraction=float((pred < copy).float().mean()),
        target_cross_window_variance=variance,
        copy_mse_to_target_variance=copy_mean/variance if variance > 1e-12 else None,
        true_delta_rms=copy_mean**.5,
        predicted_delta_rms=float(values['predicted_delta_mse'].mean())**.5,
        delta_cosine_mean=float(values['delta_cosine'][valid].mean()) if valid.any() else None,
        delta_cosine_valid_windows=int(valid.sum()),
        swapped_action_mse=float(values['swapped_action_mse'].mean()),
        swapped_minus_original_mse=float((values['swapped_action_mse']-pred).mean()),
        action_response_mse=response,
        action_response_to_motion_ratio=response/copy_mean if copy_mean > 1e-12 else None,
        action_input_change_mse=float(values['action_input_change_mse'].mean()),
        unchanged_action_fraction=float((values['action_input_change_mse'] <= 1e-12).float().mean()))
    if 'mask' in batches[0]:
        mask = torch.cat([b['mask'] for b in batches]).float().flatten()
        raw = torch.cat([b['raw_delta'] for b in batches]).float()
        if not torch.isfinite(mask).all() or not torch.isfinite(raw).all():
            raise ValueError('Non-finite residual intermediate')
        q = torch.quantile(mask, torch.tensor([.1, .5, .9])).tolist()
        summary['mask'] = dict(mean=float(mask.mean()), p10=q[0], median=q[1], p90=q[2],
            fraction_below_0p1=float((mask < .1).float().mean()),
            fraction_above_0p9=float((mask > .9).float().mean()),
            raw_delta_rms=float(raw.square().mean())**.5,
            effective_delta_rms=summary['predicted_delta_rms'],
            reconstruction_max_abs_diff=max(b['reconstruction_max_abs_diff'] for b in batches))
    per_window = [{k: bool(v[i]) if v.dtype == torch.bool else float(v[i])
                   for k, v in values.items()} for i in range(len(pred))]
    return summary, per_window


def run_t2_bn_probe(root, home, tag, comparison_tag, windows, batch):
    """Bounded inspection of the two existing models; no optimizer or environment."""
    import numpy as np
    import torch

    comparison = json.loads((root/'results'/comparison_tag/'summary.json').read_text())
    rows = comparison['rows']
    if (comparison.get('status') != 'complete' or comparison.get('dataset') != 'pusht_t2.lance'
            or [row['arm'] for row in rows] != ['ordinary', 'delta']):
        raise ValueError('Expected the completed ordinary/delta BN comparison')
    for row in rows:
        _, cfg, _, _ = load_t2_step_control(root, home, comparison['num_eval'], Path(row['policy']).parent.name)
        if (cfg.data.dataset.name != 'pusht_t2.lance' or cfg.data.dataset.frameskip != 5
                or cfg.wm.history_size != 3 or cfg.wm.num_preds != 1
                or cfg.model.projector.norm_fn._target_ != 'torch.nn.BatchNorm1d'
                or cfg.model.pred_proj._target_ != 'torch.nn.Identity'
                or not cfg.model.predictor._target_.endswith('.'+('Predictor' if row['arm'] == 'ordinary' else 'DiffPredictor'))
                or any(cfg.get(k) for k in ('fixed_action_stats', 'warm_start', 'old_path_diagnostic'))):
            raise ValueError('Probe source does not match the accepted BN/Identity model protocol')
    data_path = home/'datasets/pusht_t2.lance'
    if not (data_path/'_versions').is_dir():
        raise FileNotFoundError(data_path)
    output = root/'results'/tag
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='preparing', dataset='pusht_t2.lance', windows=windows, batch=batch,
        seed=0, rows=[], data_independence='unverified; sampled diagnostic windows, not a held-out result',
        action_intervention='Only the latest 5-step action block is replaced by another sampled window; '
                            'history actions and all images remain fixed. No alternative-action ground truth.',
        coordinate_comparison='Each model uses its own encoder; use within-model copy ratios, not raw cross-model MSE ranking.',
        normalization='Native column_normalizer on all original T2 action rows, matching this training protocol.',
        scope='128-window default; eval float32, no gradients, optimizer, checkpoint writes, or simulator.')
    def save():
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    save()
    try:
        os.environ['STABLEWM_HOME'] = str(home)
        import stable_worldmodel as swm
        sys.path.insert(0, str(root/'cl_protocol'))
        from pred_error import img_transform

        dataset = swm.data.load_dataset(str(data_path), frameskip=1, num_steps=1,
            keys_to_load=['pixels', 'action'], keys_to_cache=['action'])
        lengths = np.asarray(dataset.lengths, dtype=int)
        eligible = np.flatnonzero(lengths >= 20)
        if windows > len(eligible):
            raise ValueError('Need one distinct valid episode per probe window')
        rng = np.random.default_rng(0)
        episodes = rng.choice(eligible, windows, replace=False)
        starts = np.array([rng.integers(0, lengths[e]-20+1) for e in episodes])
        scenes = [dict(episode_ordinal=int(e), start_step=int(s)) for e, s in zip(episodes, starts)]
        order = rng.permutation(windows)
        donors = np.empty(windows, dtype=int)
        donors[order] = np.roll(order, 1)
        summary.update(scenes=scenes, donor_window_indices=donors.tolist())
        tf = img_transform()
        normalize = swm.data.column_normalizer(dataset, 'action', 'action')
        pixels, actions = [], []
        print('[BN_PROBE_START]', json.dumps(dict(windows=windows, policies=[r['policy'] for r in rows])), flush=True)
        for start in range(0, windows, batch):
            stop = min(windows, start+batch)
            chunks = dataset.load_chunk(episodes[start:stop], starts[start:stop], starts[start:stop]+16)
            if len(chunks) != stop-start:
                raise ValueError('Incomplete diagnostic batch')
            for chunk in chunks:
                raw = torch.as_tensor(chunk['action'])[:15].float()
                if len(chunk['pixels']) != 16 or raw.shape != (15, 2) or not torch.isfinite(raw).all():
                    raise ValueError('Incomplete or non-finite diagnostic window')
                pixels.append(torch.stack([tf(chunk['pixels'][i]) for i in (0, 5, 10, 15)]))
                normalized = normalize({'action': raw})['action']
                actions.append(torch.as_tensor(normalized).float().reshape(3, 10))
        pixels, actions = torch.stack(pixels), torch.stack(actions)
        swapped_actions = actions.clone()
        swapped_actions[:, -1] = actions[torch.as_tensor(donors), -1]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        summary.update(device=device, status='inspecting')
        save()
        for row in rows:
            model = swm.wm.utils.load_pretrained(str(home/'checkpoints'/row['policy'])).to(device).float().eval()
            model.requires_grad_(False)
            before_versions = {n: p._version for n, p in model.named_parameters()}
            before_buffers = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
            batches = []
            for start in range(0, windows, batch):
                end = min(windows, start+batch)
                batches.append(probe_prediction_batch(model, pixels[start:end].to(device),
                    actions[start:end].to(device), swapped_actions[start:end].to(device)))
                print('[BN_PROBE_PROGRESS]', row['arm'], end, '/', windows, flush=True)
            unchanged = (before_versions == {n: p._version for n, p in model.named_parameters()}
                         and all(torch.equal(b.detach().cpu(), before_buffers[n]) for n, b in model.named_buffers()))
            if not unchanged:
                raise RuntimeError('Model parameters or buffers changed during inspection')
            metrics, per_window = summarize_prediction_probe(batches)
            summary['rows'].append(dict(arm=row['arm'], policy=row['policy'], summary=metrics,
                per_window=per_window, parameter_versions_and_buffers_unchanged=unchanged))
            save()
            short = {k: metrics[k] for k in ('prediction_mse', 'copy_mse', 'prediction_to_copy_ratio',
                'better_than_copy_fraction', 'delta_cosine_mean', 'copy_mse_to_target_variance',
                'swapped_minus_original_mse', 'action_response_to_motion_ratio', 'unchanged_action_fraction')}
            print('[BN_PROBE_RESULT]', row['arm'], json.dumps(short), flush=True)
            if 'mask' in metrics:
                print('[BN_PROBE_MASK]', json.dumps(metrics['mask']), flush=True)
            del model, batches
        summary['status'] = 'complete'
        save()
        print('[BN_PROBE_DONE]', output/'summary.json', flush=True)
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        save()
        raise


def embedding_rollout(model, context, past_actions, future_actions, true_future=None):
    """Future blocks leave the last context frame; teacher forcing uses identical timing."""
    import torch

    with torch.inference_mode():
        history = context.size(1)
        if past_actions.size(1) != history-1:
            raise ValueError('Expected one executed block between each pair of context frames')
        all_actions = model.action_encoder(torch.cat([past_actions, future_actions], dim=1))
        states, predictions = list(context.unbind(1)), []
        limit = int(model.predictor.num_frames)
        for step in range(future_actions.size(1)):
            lo = max(0, len(states)-limit)
            pred = model.predict(torch.stack(states[lo:], 1), all_actions[:, lo:len(states)])[:, -1]
            predictions.append(pred)
            states.append(pred if true_future is None else true_future[:, step])
        return torch.stack(predictions, 1)


def native_policy_history_probe(swm, plan, world_options, policy_class=None):
    """Observe installed World/Policy history shapes with a zero-plan solver, no model."""
    import torch

    class ShapeSolver:
        def configure(self, *, action_space, n_envs, config):
            self.n_envs = n_envs
            self.horizon = config.horizon
            import math
            self.action_dim = math.prod(action_space.shape[1:])*config.action_block
            self.frames = []
            self.action_frames = []

        def solve(self, info_dict, init_action=None):
            self.frames.append(int(info_dict['pixels'].shape[1]))
            past = info_dict.get('action_history')
            self.action_frames.append(0 if past is None else int(past.shape[1]))
            return {'actions': torch.zeros(self.n_envs, self.horizon, self.action_dim)}

        __call__ = solve

    solver = ShapeSolver()
    policy = (policy_class or swm.policy.WorldModelPolicy)(solver=solver, config=plan)
    world = swm.World(**world_options)
    try:
        world.set_policy(policy)
        world.reset(seed=42)
        raw_frames = int(world.infos['pixels'].shape[1])
        for _ in range(plan.receding_horizon*plan.action_block+1):
            action = policy.get_action(world.infos)
            _, world.rewards, world.terminateds, world.truncateds, world.infos = world.envs.step(action)
    finally:
        world.close()
    if len(solver.frames) != 2 or solver.frames[-1] not in (1, 3):
        raise RuntimeError(f'Unexpected native policy context shapes: {solver.frames}')
    return dict(configured_history_len=getattr(plan, 'history_len', None),
                raw_world_info_frames=raw_frames, solver_input_frames=solver.frames,
                solver_action_history_blocks=solver.action_frames,
                probe='Native World/WorldModelPolicy with a zero-plan solver; shape inspection only')


def expand_planning_info(info, samples):
    return {k: v.unsqueeze(1).expand(v.size(0), samples, *v.shape[1:]) for k, v in info.items()}


def ranking_tau(x, y):
    """Kendall tau-b; constant populations are reported as undefined, not zero."""
    import numpy as np

    x, y = np.asarray(x), np.asarray(y)
    a, b = np.triu_indices(len(x), 1)
    dx, dy = np.sign(x[a]-x[b]), np.sign(y[a]-y[b])
    denominator = float(np.sqrt(np.count_nonzero(dx)*np.count_nonzero(dy)))
    return float((dx*dy).sum()/denominator) if denominator else None


def planning_rank_metrics(predicted, actual_latent, physical_distance, successes):
    import numpy as np

    predicted, actual_latent, distance = map(np.asarray, (predicted, actual_latent, physical_distance))
    if (predicted.ndim != 1 or predicted.shape != actual_latent.shape or predicted.shape != distance.shape
            or len(successes) != len(predicted) or len(predicted) < 2
            or not all(np.isfinite(a).all() for a in (predicted, actual_latent, distance))):
        raise ValueError('Invalid candidate ranking arrays')
    choice, image_choice = int(predicted.argmin()), int(actual_latent.argmin())
    return dict(predicted_vs_actual_latent_tau=ranking_tau(predicted, actual_latent),
        predicted_vs_physical_distance_tau=ranking_tau(predicted, distance),
        actual_latent_vs_physical_distance_tau=ranking_tau(actual_latent, distance),
        predicted_cost_range=float(np.ptp(predicted)), actual_latent_cost_range=float(np.ptp(actual_latent)),
        physical_distance_range=float(np.ptp(distance)),
        selected_physical_gap=float(distance[choice]-distance.min()),
        actual_image_choice_physical_gap=float(distance[image_choice]-distance.min()),
        selected_final_success=bool(successes[choice]), selected_index=choice,
        actual_image_selected_index=image_choice)


def pusht_replay(env, initial, goal, prefix, future, variation, darken, seed):
    """Recreate a shared prefix for every candidate, including hidden physics state."""
    import numpy as np

    seen, original = [], env._set_state
    def observe(state):
        seen.append(np.asarray(state).copy())
        return original(state)
    env._set_state = observe
    try:
        obs, _ = env.reset(seed=seed, options={'state': initial.copy(), 'goal_state': goal.copy(),
                                             'variation_values': variation})
    finally:
        env._set_state = original
    if not seen or not np.array_equal(seen[-1], initial) or not np.array_equal(env.goal_state, goal):
        raise RuntimeError('Installed PushT reset did not apply the supplied initial and goal states')
    for key, value in variation.items():
        space = env.variation_space
        for part in key.split('.'):
            space = space[part]
        if not np.array_equal(np.asarray(space.value), value):
            raise RuntimeError(f'PushT variation was not applied: {key}')
    def frame():
        return np.clip(np.asarray(env.render(), dtype=np.float32)*darken, 0, 255).astype(np.uint8)
    frames, ever = [frame()], False
    for step, action in enumerate(prefix, 1):
        obs, _, terminated, truncated, _ = env.step(action)
        if truncated:
            raise RuntimeError('Unexpected prefix truncation')
        ever |= bool(terminated)
        if step % 5 == 0:
            frames.append(frame())
    for action in future:
        obs, _, terminated, truncated, _ = env.step(action)
        if truncated:
            raise RuntimeError('Candidate ended before its fixed horizon')
        ever |= bool(terminated)
    success, distance = env.eval_state(goal, obs['state'])
    return np.stack(frames), frame(), float(distance), bool(success), ever


def run_t2_planning_probe(root, home, tag, comparison_tag, windows, batch, scene_count, sample_count):
    """Multi-step dataset prediction plus shared, physically replayed native CEM candidates."""
    print('[BN_PLAN_START]', flush=True)
    import inspect
    import numpy as np
    import torch
    import gymnasium as gym
    import hydra
    import yaml
    from omegaconf import OmegaConf
    from sklearn.preprocessing import StandardScaler

    os.environ['STABLEWM_HOME'] = str(home)
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import stable_worldmodel as swm
    from stable_worldmodel.planning import ShootingCostEvaluator, GoalMSE
    sys.path.insert(0, str(root/'cl_protocol'))
    from pred_error import img_transform
    comparison = json.loads((root/'results'/comparison_tag/'summary.json').read_text())
    rows = comparison['rows']
    if comparison.get('status') != 'complete' or [r['arm'] for r in rows] != ['ordinary', 'delta']:
        raise ValueError('Expected the completed BN ordinary/delta comparison')
    for row in rows:
        _, source, _, _ = load_t2_step_control(root, home, comparison['num_eval'], Path(row['policy']).parent.name)
        if (source.data.dataset.name != 'pusht_t2.lance' or source.wm.history_size != 3
                or source.wm.num_preds != 1 or source.data.dataset.frameskip != 5
                or source.model.projector.norm_fn._target_ != 'torch.nn.BatchNorm1d'
                or source.model.pred_proj._target_ != 'torch.nn.Identity'
                or not source.model.predictor._target_.endswith('.'+('Predictor' if row['arm'] == 'ordinary' else 'DiffPredictor'))):
            raise ValueError('Source is not the accepted BN/Identity model')
    data_path = home/'datasets/pusht_t2.lance'
    if not (data_path/'_versions').is_dir():
        raise FileNotFoundError(data_path)
    with hydra.initialize_config_dir(config_dir=str(root/'scripts/plan/config'), version_base=None):
        cfg = hydra.compose(config_name='pusht', overrides=['seed=42'])
    if (cfg.plan_config.horizon != 5 or cfg.plan_config.action_block != 5
            or cfg.plan_config.receding_horizon != 5 or cfg.get('bf16', False) or cfg.get('compile', False)
            or cfg.solver._target_ != 'stable_worldmodel.planning.solver.CEMSolver'
            or cfg.solver.batch_size != 1 or sample_count > cfg.solver.num_samples):
        raise ValueError('Expected native float32 CEM with five blocks, batch_size=1 and unchanged solver budget')
    device = str(cfg.solver.device)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('The native CEM config requires the AutoDL GPU')
    output = root/'results'/tag
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='preparing', windows=windows, scenes=scene_count, samples_per_population=sample_count,
        policies={r['arm']: r['policy'] for r in rows}, dataset=str(data_path),
        comparison_summary=str(root/'results'/comparison_tag/'summary.json'),
        timeline=dict(context_frames_at_raw_offsets=[0, 5, 10], current_raw_offset=10,
                      future_raw_offsets=[15, 20, 25, 30, 35], goal_raw_offset=35,
                      candidate_future_steps=25, window_seed=0, search_seed_base=42),
        native_plan_config=OmegaConf.to_container(cfg.plan_config, resolve=True),
        native_solver_config=OmegaConf.to_container(cfg.solver, resolve=True),
        multi_step=[], ranking=[], returned_plans=[], per_scene=[],
        scope='Existing checkpoints only; paired diagnostic windows, not independent held-out or closed-loop results. '
              'Real replay starts from recorded observable state then recreates the same 10-step prefix for each plan. '
              'Native environment distance is ground truth; each encoder latent distance is measured separately. '
              'Fixed-horizon replay continues after success and records final/ever success; no training or checkpoint writes.')
    def save():
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    save()
    env = None
    try:
        plan = swm.PlanConfig(**summary['native_plan_config'])
        world_options = {**OmegaConf.to_container(cfg.world, resolve=False),
                         'num_envs': 1, 'max_episode_steps': 200, 'image_shape': (224, 224)}
        summary['policy_history_probe'] = native_policy_history_probe(swm, plan, world_options)
        history = summary['policy_history_probe']['solver_input_frames'][-1]
        print('[BN_PLAN_HISTORY]', json.dumps(summary['policy_history_probe']), flush=True)
        summary['rollout_source'] = None
        dataset = swm.data.load_dataset(str(data_path), frameskip=1, num_steps=1,
            keys_to_load=['pixels', 'action', 'state'], keys_to_cache=['action', 'state'])
        rng = np.random.default_rng(0)
        eligible = np.flatnonzero(np.asarray(dataset.lengths) >= 36)
        if windows > len(eligible) or scene_count > windows:
            raise ValueError('Not enough distinct episodes for the bounded probe')
        episodes = rng.choice(eligible, windows, replace=False)
        starts = np.array([rng.integers(0, dataset.lengths[e]-35) for e in episodes])
        summary['window_metadata'] = [dict(episode_ordinal=int(e), start_step=int(s)) for e, s in zip(episodes, starts)]
        raw_stats = np.asarray(dataset.get_col_data('action'))
        if np.isinf(raw_stats).any():
            raise ValueError('Non-finite action statistics')
        scaler = StandardScaler().fit(raw_stats[~np.isnan(raw_stats).any(axis=1)])
        normalize = swm.data.column_normalizer(dataset, 'action', 'action')
        summary['action_mean'], summary['action_scale'] = scaler.mean_.tolist(), scaler.scale_.tolist()
        tf = img_transform()
        pixels, actions, scene_data = [], [], []
        for start in range(0, windows, batch):
            stop = min(windows, start+batch)
            chunks = dataset.load_chunk(episodes[start:stop], starts[start:stop], starts[start:stop]+36)
            if len(chunks) != stop-start:
                raise ValueError('Incomplete multi-step window batch')
            for offset, chunk in enumerate(chunks):
                raw = torch.as_tensor(chunk['action'])[:35].float()
                if raw.shape != (35, 2) or not torch.isfinite(raw).all() or len(chunk['pixels']) != 36:
                    raise ValueError('Invalid multi-step window')
                standardized = torch.as_tensor(normalize({'action': raw})['action']).float()
                # Native float32 statistics and sklearn float64 accumulation differ slightly.
                torch.testing.assert_close(standardized, torch.as_tensor(scaler.transform(raw.numpy())).float(), rtol=1e-4, atol=1e-5)
                pixels.append(torch.stack([tf(chunk['pixels'][i]) for i in range(0, 36, 5)]))
                actions.append(standardized.reshape(7, 10))
                if start+offset < scene_count:
                    scene_data.append(dict(initial=np.asarray(chunk['state'][0], dtype=np.float64),
                        goal=np.asarray(chunk['state'][35], dtype=np.float64), prefix=raw[:10].numpy().copy()))
        pixels, actions = torch.stack(pixels), torch.stack(actions)
        print('[BN_PLAN_DATA_READY]', windows, 'windows', flush=True)
        models, snapshots = {}, {}
        for row in rows:
            model = swm.wm.utils.load_pretrained(str(home/'checkpoints'/row['policy'])).to(device).float().eval()
            model.requires_grad_(False)
            models[row['arm']] = model
            snapshots[row['arm']] = ({n: p._version for n, p in model.named_parameters()},
                                    {n: b.detach().cpu().clone() for n, b in model.named_buffers()})
            summary['rollout_source'] = inspect.getsource(type(model).rollout)
        with torch.inference_mode():
            for arm, model in models.items():
                for h in (1, 3):
                    collected = []
                    for start in range(0, windows, batch):
                        emb = model.encode({'pixels': pixels[start:start+batch].to(device)})['emb']
                        a = actions[start:start+batch].to(device)
                        context, target = emb[:, 3-h:3], emb[:, 3:8]
                        auto = embedding_rollout(model, context, a[:, 3-h:2], a[:, 2:7])
                        teacher = embedding_rollout(model, context, a[:, 3-h:2], a[:, 2:7], target)
                        collected.append(torch.stack([(auto-target).square().mean(-1),
                            (teacher-target).square().mean(-1), (context[:, -1:]-target).square().mean(-1)], 1).cpu())
                    errors = torch.cat(collected).mean(0)
                    if not torch.isfinite(errors).all():
                        raise ValueError('Non-finite multi-step error')
                    record = dict(arm=arm, history_frames=h, autoregressive_mse=errors[0].tolist(),
                        teacher_forced_mse=errors[1].tolist(), copy_mse=errors[2].tolist(),
                        autoregressive_to_copy=[float(a/c) if c > 1e-12 else None for a, c in zip(errors[0], errors[2])],
                        teacher_forced_to_copy=[float(a/c) if c > 1e-12 else None for a, c in zip(errors[1], errors[2])])
                    summary['multi_step'].append(record)
                    print('[BN_PLAN_MULTISTEP]', json.dumps(record), flush=True)
                    save()
        spec = yaml.safe_load((root/'cl_protocol/tasks.yaml').read_text(encoding='utf-8'))['sequences']['A_pusht']
        task = spec['tasks']['T2']
        variation = {k: np.asarray(v, dtype=np.float32) for k, v in task['variation_values'].items()}
        env = gym.make(spec['env'], render_mode='rgb_array').unwrapped
        summary['physical_distance_source'] = inspect.getsource(type(env).eval_state)
        evaluators = {arm: ShootingCostEvaluator(model, GoalMSE()) for arm, model in models.items()}
        solver_settings = dict(summary['native_solver_config'])
        solver_settings.pop('cost', None)
        class TraceCost(torch.nn.Module):
            def __init__(self, evaluator):
                super().__init__()
                self.evaluator, self.calls, self.populations = evaluator, 0, {}

            def get_cost(self, info, candidates):
                costs = self.evaluator.get_cost(info, candidates)
                if self.calls in (0, int(solver_settings['n_steps'])-1):
                    self.populations[self.calls] = (candidates.detach().cpu().clone(), costs.detach().cpu().clone())
                self.calls += 1
                return costs

        for scene_index, scene in enumerate(scene_data):
            histories, _, _, _, _ = pusht_replay(env, **scene, future=[], variation=variation,
                darken=task.get('darken', 1.), seed=42+scene_index)
            hist = torch.stack([tf(torch.as_tensor(f).permute(2, 0, 1)) for f in histories[-history:]])[None].to(device)
            goal = pixels[scene_index, -1:][None].to(device)
            past = actions[scene_index:scene_index+1, 3-history:2].to(device)
            info = dict(pixels=hist, goal=goal, action=torch.zeros(1, history, 10, device=device), action_history=past)
            captures, returned = {}, {}
            # Verify the installed rollout's temporal convention against explicit future-block unrolling.
            with torch.inference_mode():
                test_actions = actions[scene_index:scene_index+1, 2:7].to(device)
                for arm, model in models.items():
                    context = model.encode({'pixels': hist})['emb']
                    expected = embedding_rollout(model, context, past, test_actions)
                    native = model.rollout(expand_planning_info(info, 1), test_actions[:, None])['predicted_emb']
                    if native.size(2) != history+5:
                        raise RuntimeError(f'Installed native rollout predicts {native.size(2)-history} future states, expected 5; temporal protocol mismatch')
                    torch.testing.assert_close(native[:, 0, -5:], expected, rtol=1e-4, atol=1e-4)
                    goal_emb = model.encode({'pixels': goal})['emb'][:, 0]
                    native_cost = evaluators[arm].get_cost(expand_planning_info(info, 1), test_actions[:, None])
                    manual_cost = (expected[:, -1]-goal_emb).square().sum(-1)
                    torch.testing.assert_close(native_cost[:, 0], manual_cost, rtol=1e-4, atol=1e-4)
            print('[BN_PLAN_PARITY_OK]', scene_index+1, flush=True)
            for arm in models:
                trace = TraceCost(evaluators[arm])
                solver = hydra.utils.instantiate({**solver_settings, 'seed': 42+scene_index}, cost=trace)
                solver.configure(action_space=gym.spaces.Box(-1., 1., shape=(1, 2), dtype=np.float32), n_envs=1, config=plan)
                result = solver.solve(info.copy(), init_action=None)
                if trace.calls != solver_settings['n_steps']:
                    raise RuntimeError('Native CEM call count changed')
                captures[arm], returned[arm] = trace.populations, result['actions'][0]
            if not torch.equal(captures['ordinary'][0][0], captures['delta'][0][0]):
                raise RuntimeError('Native CEM initial populations differ')
            groups = []
            for name, source_arm, iteration in [('initial', 'ordinary', 0),
                    ('ordinary_final', 'ordinary', solver_settings['n_steps']-1),
                    ('delta_final', 'delta', solver_settings['n_steps']-1)]:
                population, costs = captures[source_arm][iteration]
                required = {0, int(costs[0].argmin())}
                if name == 'initial':
                    required.add(int(captures['delta'][0][1][0].argmin()))
                selected = sorted(required)
                if len(selected) > sample_count:
                    raise ValueError('Candidate sample count must include current mean and both guide best choices')
                pool = np.setdiff1d(np.arange(population.size(1)), selected)
                selected += rng.choice(pool, sample_count-len(selected), replace=False).tolist()
                groups.append((name, population[0, selected]))
            groups.append(('returned', torch.stack([returned['ordinary'], returned['delta']])))
            plans = torch.cat([p for _, p in groups])
            raw_plans = scaler.inverse_transform(plans.reshape(-1, 2).numpy()).reshape(len(plans), 25, 2)
            endpoints, distances, successes, ever = [], [], [], []
            for candidate, future in enumerate(raw_plans):
                repeat_history, endpoint, distance, success, reached = pusht_replay(env, **scene, future=future,
                    variation=variation, darken=task.get('darken', 1.), seed=42+scene_index)
                if not np.array_equal(histories, repeat_history):
                    raise RuntimeError('Candidate replay did not reproduce the same physical history')
                endpoints.append(tf(torch.as_tensor(endpoint).permute(2, 0, 1)))
                distances.append(distance);successes.append(success);ever.append(reached)
            image_tensor = torch.stack(endpoints)
            scored = {}
            with torch.inference_mode():
                for arm, model in models.items():
                    pred = evaluators[arm].get_cost(expand_planning_info(info, len(plans)), plans[None].to(device))[0].cpu().numpy()
                    actual = []
                    goal_emb = model.encode({'pixels': goal})['emb'][:, 0]
                    for start in range(0, len(plans), batch):
                        achieved = model.encode({'pixels': image_tensor[start:start+batch, None].to(device)})['emb'][:, 0]
                        actual.extend((achieved-goal_emb).square().sum(-1).cpu().tolist())
                    scored[arm] = (pred, np.asarray(actual))
                    offset = 0
                    for name, population in groups:
                        end = offset+len(population)
                        metrics = planning_rank_metrics(pred[offset:end], np.asarray(actual)[offset:end],
                            distances[offset:end], successes[offset:end])
                        record = dict(scene=scene_index, arm=arm, population=name, **metrics)
                        summary['ranking'].append(record)
                        print('[BN_PLAN_RANK]', json.dumps(record), flush=True)
                        offset = end
            summary['returned_plans'].append(dict(scene=scene_index,
                ordinary_distance=distances[-2], delta_distance=distances[-1],
                ordinary_final_success=successes[-2], delta_final_success=successes[-1],
                ordinary_ever_success=ever[-2], delta_ever_success=ever[-1]))
            summary['per_scene'].append(dict(scene=scene_index, window=summary['window_metadata'][scene_index],
                candidate_groups=[dict(name=name, count=len(p)) for name, p in groups],
                standardized_plans=plans.tolist(), physical_distances=distances, final_successes=successes,
                ever_successes=ever, raw_action_outside_fraction=float(np.mean(np.abs(raw_plans) > 1)),
                model_costs={arm: dict(predicted=p.tolist(), actual_image=a.tolist()) for arm, (p, a) in scored.items()}))
            print('[BN_PLAN_SCENE_DONE]', scene_index+1, '/', scene_count, flush=True)
            save()
        for arm, model in models.items():
            versions, buffers = snapshots[arm]
            if (versions != {n: p._version for n, p in model.named_parameters()}
                    or not all(torch.equal(b.detach().cpu(), buffers[n]) for n, b in model.named_buffers())):
                raise RuntimeError('Probe changed model parameters or buffers')
        summary.update(status='complete', parameter_versions_and_buffers_unchanged=True)
        save()
        print('[BN_PLAN_DONE]', output/'summary.json', flush=True)
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        save()
        print('[BN_PLAN_FAILED]', str(exc), flush=True)
        raise
    finally:
        if env is not None:
            env.close()


def validate_history_config(actual_cfg, plan_dict, expected_solver, history):
    """Resolve saved Hydra references against their complete root before comparing values."""
    from omegaconf import OmegaConf

    saved = OmegaConf.create(actual_cfg)
    actual = dict(plan_config=OmegaConf.to_container(saved.plan_config, resolve=True),
                  solver=OmegaConf.to_container(saved.solver, resolve=True))
    expected = dict(plan_config={**plan_dict, 'history_len': history}, solver=expected_solver)
    differences = {section: dict(expected=expected[section], actual=actual[section])
                   for section in expected if actual[section] != expected[section]}
    if differences:
        raise ValueError('History planning settings differ after resolving references: '+json.dumps(differences))
    return actual


def run_t2_history(root, home, tag, comparison_tag, num_eval, *, reuse_first_cell_from=None):
    """Evaluate the installed rollout on the fixed four-cell history comparison."""
    print('[HISTORY_START]', flush=True)
    import inspect
    import gc
    import numpy as np
    import torch
    import hydra
    import yaml
    from omegaconf import OmegaConf
    import stable_worldmodel as swm
    from stable_worldmodel.planning import ShootingCostEvaluator, GoalMSE

    comparison = json.loads((root/'results'/comparison_tag/'summary.json').read_text())
    rows = comparison['rows']
    if comparison['status'] != 'complete' or [r['arm'] for r in rows] != ['ordinary', 'delta']:
        raise ValueError('Expected the completed BN ordinary/delta comparison')
    if not torch.cuda.is_available():
        raise RuntimeError('The history comparison requires the AutoDL GPU')
    with hydra.initialize_config_dir(config_dir=str(root/'scripts/plan/config'), version_base=None):
        cfg = hydra.compose(config_name='pusht', overrides=['seed=42', f'eval.num_eval={num_eval}'])
    if (cfg.plan_config.horizon != 5 or cfg.plan_config.action_block != 5 or cfg.plan_config.receding_horizon != 5
            or cfg.eval.goal_offset_steps != 25 or cfg.eval.eval_budget != 50
            or cfg.get('compile', False) or cfg.get('bf16', False)
            or cfg.solver._target_ != 'stable_worldmodel.planning.solver.CEMSolver'):
        raise ValueError('Native planning settings changed from the completed comparison')
    output = root/'results'/tag
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='preflight', num_eval=num_eval, seed=42, rows=[], comparisons=[],
        scope='Existing BN/Identity checkpoints; four fixed cells on the same native T2 scenes. No training. '
              'Installed policy history and LeWM.rollout are checked without modifying source.')
    def save():
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    save()
    try:
        plan_dict = OmegaConf.to_container(cfg.plan_config, resolve=True)
        plan3 = swm.PlanConfig(**{**plan_dict, 'history_len': 3})
        shape = native_policy_history_probe(swm, plan3, {**OmegaConf.to_container(cfg.world, resolve=False),
            'num_envs': 1, 'max_episode_steps': 200, 'image_shape': (224, 224)})
        summary['policy_before'] = shape
        policy_path = root/'stable_worldmodel/policy.py'
        policy_original = policy_path.read_text(encoding='utf-8')
        policy_fixed = policy_history_source(policy_original)
        if (shape['solver_input_frames'] != [1, 3]
                or shape['solver_action_history_blocks'] != [0, 2]
                or policy_fixed != policy_original):
            raise ValueError('Install the confirmed history repair first: python implementation/install.py --root <SWM_ROOT> --apply')
        summary['policy_preflight'] = shape
        print('[HISTORY_POLICY_OK]', json.dumps(shape), flush=True)
        source_path = root/'stable_worldmodel/wm/lewm/lewm.py'
        original_source = source_path.read_text(encoding='utf-8')
        fixed_source = future_rollout_source(original_source)
        if fixed_source != original_source:
            raise ValueError('Install the confirmed rollout repair first: python implementation/install.py --root <SWM_ROOT> --apply')
        device = str(cfg.solver.device)
        generator = torch.Generator().manual_seed(20261024)
        pixels = torch.randn(1, 3, 3, 224, 224, generator=generator).to(device)
        candidates = torch.randn(1, 4, 5, 10, generator=generator).to(device)
        past = torch.randn(1, 2, 10, generator=generator).to(device)
        preflight = []
        for row in rows:
            _, source, _, _ = load_t2_step_control(root, home, comparison['num_eval'], Path(row['policy']).parent.name)
            if (source.data.dataset.name != 'pusht_t2.lance' or source.wm.history_size != 3
                    or source.wm.num_preds != 1 or source.data.dataset.frameskip != 5
                    or source.model.projector.norm_fn._target_ != 'torch.nn.BatchNorm1d'
                    or source.model.pred_proj._target_ != 'torch.nn.Identity'
                    or not source.model.predictor._target_.endswith('.'+('Predictor' if row['arm'] == 'ordinary' else 'DiffPredictor'))):
                raise ValueError('Expected the accepted BN/Identity models')
            model = swm.wm.utils.load_pretrained(str(home/'checkpoints'/row['policy'])).to(device).float().eval()
            model.requires_grad_(False)
            if Path(inspect.getfile(type(model))).resolve() != source_path.resolve():
                raise ValueError('Loaded model is not from the expected installed LeWM source file')
            versions = {n: p._version for n,p in model.named_parameters()}
            buffers = {n: b.detach().cpu().clone() for n,b in model.named_buffers()}
            with torch.inference_mode():
                single = dict(pixels=pixels[:, -1:])
                old = model.rollout(expand_planning_info(single, 4), candidates)['predicted_emb']
                new = model.rollout(expand_planning_info(single, 4), candidates)['predicted_emb']
                torch.testing.assert_close(new, old, rtol=0, atol=0)
                maximum = 0.
                for h in (1, 3):
                    info = dict(pixels=pixels[:, -h:], action_history=past[:, 3-h:])
                    context = model.encode({'pixels': info['pixels']})['emb']
                    actual = model.rollout(expand_planning_info(info, 4), candidates)['predicted_emb']
                    expected = embedding_rollout(model, context.expand(4, -1, -1),
                        info['action_history'].expand(4, -1, -1), candidates[0])
                    if actual.shape != (1, 4, h+5, context.size(-1)):
                        raise ValueError('Repaired rollout returned the wrong future length')
                    torch.testing.assert_close(actual[0, :, -5:], expected, rtol=1e-4, atol=1e-4)
                    maximum = max(maximum, float((actual[0, :, -5:]-expected).abs().max()))
            if versions != {n:p._version for n,p in model.named_parameters()} or not all(
                    torch.equal(b.detach().cpu(), buffers[n]) for n,b in model.named_buffers()):
                raise RuntimeError('Preflight changed model weights or buffers')
            record = dict(arm=row['arm'], one_frame_bitwise_unchanged=True,
                          future_steps_for_h1_and_h3=5, explicit_unroll_max_abs_diff=maximum,
                          parameter_versions_and_buffers_unchanged=True)
            preflight.append(record)
            print('[HISTORY_ROLLOUT_OK]', json.dumps(record), flush=True)
            del model
        summary['rollout_preflight'] = preflight
        if source_path.read_text(encoding='utf-8') != original_source or policy_path.read_text(encoding='utf-8') != policy_original:
            raise RuntimeError('Installed source changed during preflight')
        summary['verified_model_source'] = str(source_path)
        summary['verified_policy_source'] = str(policy_path)
        print('[HISTORY_SOURCE_VERIFIED]', source_path, flush=True)
        gc.collect()
        torch.cuda.empty_cache()

        data_path = home/'datasets/pusht_t2.lance'
        if not (data_path/'_versions').is_dir():
            raise FileNotFoundError(data_path)
        dataset = swm.data.load_dataset(str(data_path), frameskip=1, num_steps=1,
            keys_to_load=['action'], keys_to_cache=['action'])
        names = set(dataset.column_names) | set(getattr(dataset, '_schema_names', ()))
        episodes = np.asarray(dataset.get_col_data('episode_idx' if 'episode_idx' in names else 'ep_idx'))
        steps = np.asarray(dataset.get_col_data('step_idx'))
        maximum_step = {}
        for e, s in zip(episodes, steps):
            maximum_step[int(e)] = max(maximum_step.get(int(e), 0), int(s))
        valid = np.flatnonzero(steps <= np.asarray([maximum_step[int(e)]-25 for e in episodes]))
        if num_eval > len(valid):
            raise ValueError('Not enough native evaluation starting points')
        selected = np.sort(valid[np.random.default_rng(42).choice(len(valid), size=num_eval, replace=False)])
        scenes = [dict(episode_idx=int(episodes[i]), start_step=int(steps[i])) for i in selected]
        manifest = dict(dataset='pusht_t2.lance', episode_lengths=list(map(int, dataset.lengths)),
                        scenes=scenes, goal_offset=25, eval_budget=50)
        scene_path = output/'scenes.json'
        scene_path.write_text(json.dumps(manifest, indent=2)+'\n')
        summary['scene_manifest'] = str(scene_path)
        env = dict(os.environ, STABLEWM_HOME=str(home), PYTHONUNBUFFERED='1',
                   MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa')
        expected_solver = OmegaConf.to_container(cfg.solver, resolve=True)
        reuse = None
        if reuse_first_cell_from is not None:
            prior = root/'results'/reuse_first_cell_from
            old = json.loads((prior/'summary.json').read_text())
            if (old.get('status') != 'failed' or old.get('num_eval') != num_eval or old.get('seed') != 42
                    or old.get('error') != 'History cell changed planning settings beyond history_len'
                    or json.loads((prior/'scenes.json').read_text()) != manifest
                    or (prior/'policy_after.py').read_text(encoding='utf-8') != policy_fixed
                    or (prior/'lewm_after.py').read_text(encoding='utf-8') != fixed_source):
                raise ValueError('Prior first cell does not match this configuration-check failure, scenes and installed sources')
            records = [json.loads(line[len('CELL_RESULT '):])
                for line in (prior/'ordinary_h1.log').read_text(encoding='utf-8').splitlines()
                if line.startswith('CELL_RESULT ')]
            if len(records) != 1:
                raise ValueError('Prior first cell lacks a unique completed evaluation receipt')
            reuse = (prior, records[0])
            summary['reused_first_cell_from'] = str(prior)
        for row in rows:
            for h in (1, 3):
                cell = f"{row['arm']}_h{h}"
                summary['status'] = 'evaluating_'+cell
                save()
                print('[HISTORY_CELL_START]', cell, flush=True)
                scene_result = output/(cell+'_scenes.json')
                command = [sys.executable, '-u', str(root/'cl_protocol/eval_task.py'),
                    '--sequence', 'A_pusht', '--task', 'T2', '--policy', row['policy'],
                    '--num-eval', str(num_eval), '--seed', '42', '--scene-manifest', str(scene_path),
                    '--scene-result', str(scene_result), '--extra', f'++plan_config.history_len={h}']
                records = []
                reused = reuse is not None and cell == 'ordinary_h1'
                if reused:
                    prior, record = reuse
                    scene_result.write_text((prior/'ordinary_h1_scenes.json').read_text())
                    records, code = [record], 0
                    (output/(cell+'.log')).write_text('REUSED_FROM '+str(prior)+'\nCELL_RESULT '+json.dumps(record)+'\n')
                else:
                    with (output/(cell+'.log')).open('w', encoding='utf-8') as log:
                        with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1) as proc:
                            for line in proc.stdout:
                                log.write(line)
                                print(line, end='', flush=True)
                                if line.startswith('CELL_RESULT '):
                                    records.append(json.loads(line[len('CELL_RESULT '):]))
                            code = proc.wait()
                if code != 0 or len(records) != 1:
                    raise RuntimeError(f'{cell} evaluation failed; see {cell}.log')
                record = records[0]
                if any(record.get(k) != v for k,v in dict(policy=row['policy'], task='T2',
                        dataset='pusht_t2.lance', num_eval=num_eval, seed=42,
                        variation_values={'goal.position': [100, 100]}, darken=.5).items()):
                    raise ValueError('History cell evaluation protocol changed')
                behavior = json.loads(scene_result.read_text())
                if (behavior['status'] != 'complete' or behavior['scenes'] != scenes
                        or len(behavior['successes']) != num_eval
                        or abs(100*sum(behavior['successes'])/num_eval-record['success_rate']) > 1e-7):
                    raise ValueError('History cell returned different scenes or inconsistent success flags')
                with Path(record['result_path']).open('rb') as f:
                    f.seek(record['result_start_offset'])
                    fresh = f.read().decode('utf-8')
                actual_cfg = yaml.safe_load(fresh.split('==== CONFIG ====')[1].split('==== RESULTS ====')[0])
                verified_config = validate_history_config(actual_cfg, plan_dict, expected_solver, h)
                summary['rows'].append(dict(arm=row['arm'], history_frames=h, policy=row['policy'],
                    success_rate=record['success_rate'], previous_h1_rate=row['success_rate'],
                    successes=behavior['successes'], scene_result=str(scene_result),
                    reused_from=str(reuse[0]) if reused else None, verified_planning_config=verified_config))
                save()
                if reused:
                    print('[HISTORY_CELL_REUSED]', cell, record['success_rate'], flush=True)
                print('[HISTORY_CELL_DONE]', json.dumps({k:v for k,v in summary['rows'][-1].items()
                    if k not in ('successes', 'verified_planning_config')}), flush=True)
        for arm in ('ordinary', 'delta'):
            a,b = [next(r for r in summary['rows'] if r['arm'] == arm and r['history_frames'] == h) for h in (1,3)]
            diff = 100*(np.asarray(b['successes'], dtype=float)-np.asarray(a['successes'], dtype=float))
            rng = np.random.default_rng(0)
            ci = np.percentile(diff[rng.integers(0, num_eval, (10000, num_eval))].mean(1), [2.5,97.5])
            summary['comparisons'].append(dict(arm=arm, h1=a['success_rate'], h3=b['success_rate'],
                change_pp=float(diff.mean()), paired_bootstrap95_pp=ci.tolist(),
                scope='Scene uncertainty for these fixed checkpoints; not training-seed uncertainty'))
        summary['status'] = 'complete'
        save()
        print('[HISTORY_DONE]', json.dumps(dict(summary=str(output/'summary.json'), comparisons=summary['comparisons'])), flush=True)
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        save()
        print('[HISTORY_FAILED]', str(exc), flush=True)
        raise
