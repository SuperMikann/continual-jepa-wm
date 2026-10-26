"""PushT training configurations and controlled single-task comparisons."""
import json
import os
from pathlib import Path
import subprocess
import sys


def prepare_protocol(home, output, num_eval):
    import numpy as np
    import stable_worldmodel as swm
    dataset = swm.data.load_dataset('pusht_expert_train.lance', frameskip=5,
                                   num_steps=4, keys_to_load=['action'], keys_to_cache=['action'])
    lengths = np.asarray(dataset.lengths, dtype=int)
    if len(lengths) < 600:
        raise ValueError('Need at least 600 episodes for the fixed 200 validation/200 test split')
    rng = np.random.default_rng(20261023)
    order = rng.permutation(len(lengths))
    validation, test, train = order[:200], order[200:400], order[400:]
    manifest = dict(dataset='pusht_expert_train.lance', split_seed=20261023,
                    episode_lengths=lengths.tolist(), train_episodes=sorted(train.tolist()),
                    validation_episodes=sorted(validation.tolist()), test_episodes=sorted(test.tolist()),
                    indexing='native dataset episode ordinal used by clip_indices and load_chunk')
    split_path = output / 'episode_split.json'
    split_path.write_text(json.dumps(manifest, indent=2) + '\n')
    actions = np.asarray(dataset.get_col_data('action'), dtype=np.float64)
    if len(actions) != int(lengths.sum()):
        raise ValueError('Action rows and episode lengths mismatch')
    train_mask = np.repeat(np.isin(np.arange(len(lengths)), train), lengths)
    values = actions[train_mask]
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        raise ValueError('No finite training actions')
    stats = dict(mean=values.mean(axis=0).tolist(), scale=np.maximum(values.std(axis=0), 1e-8).tolist(),
                 source=f'Training episodes only, manifest {split_path}', fitted_rows=len(values))
    valid = sorted(int(ep) for ep in test if lengths[ep] > 25)
    if not 1 <= num_eval <= len(valid):
        raise ValueError('Not enough distinct held-out test episodes')
    scene_rng = np.random.default_rng(42)
    picked = scene_rng.choice(valid, num_eval, replace=False)
    scenes = [dict(episode_idx=int(ep), start_step=int(scene_rng.integers(0, lengths[ep]-25))) for ep in picked]
    (output / 'scenes.json').write_text(json.dumps(dict(dataset=manifest['dataset'],
        episode_lengths=lengths.tolist(), scenes=scenes, goal_offset=25, eval_budget=50), indent=2) + '\n')
    print('[COMPARE_SPLIT_READY]', json.dumps(dict(train_episodes=len(train), validation_episodes=200,
          test_episodes=200, behavior_scenes=num_eval, new_dataset_copies=0)), flush=True)
    return stats


def prepare_configs(root, output, tag, *, mode='smoke', steps=3000, batch=64, stats=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    if not OmegaConf.has_resolver('eval'):
        # Native config uses addition of history_size and num_preds only.
        OmegaConf.register_new_resolver('eval', lambda s: sum(int(x.strip()) for x in s.split('+')))
    with initialize_config_dir(config_dir=str(root / 'scripts/train/config'), version_base=None):
        base = compose(config_name='lewm', overrides=['data=pusht', f'subdir={tag}'])
    configs = []
    for arm, predictor in [('absolute', 'Predictor'), ('delta', 'DiffPredictor')]:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        run = f'{tag}_{arm}'
        with open_dict(cfg):
            cfg.output_model_name = cfg.subdir = run
            cfg.seed = 3072
            cfg.mechanism_version = 2
            cfg.smoke_steps = 20
            if mode == 'compare':
                cfg.smoke_steps = 0
                cfg.comparison_steps = steps
                cfg.episode_split = str(output / 'episode_split.json')
                cfg.save_epoch_interval = steps + 1
            cfg.trainer.max_epochs = 1
            cfg.trainer.accelerator = 'gpu'
            cfg.trainer.devices = 1
            cfg.trainer.precision = 'bf16-mixed'
            cfg.num_workers = 2
            cfg.loader.num_workers = 2
            cfg.loader.batch_size = 16
            cfg.loader.persistent_workers = True
            cfg.loader.prefetch_factor = 2
            cfg.wandb.enabled = False
            cfg.data.dataset.name = 'pusht_expert_train.lance'
            cfg.wm.history_size = 3
            cfg.wm.num_preds = 1
            cfg.data.dataset.frameskip = 5
            cfg.model.predictor = dict(
                _target_=f'stable_worldmodel.wm.lewm.module.{predictor}',
                num_frames=3, input_dim=cfg.embed_dim, hidden_dim=cfg.embed_dim,
                output_dim=cfg.embed_dim, depth=6, heads=16, mlp_dim=2048,
                dim_head=64, dropout=0.1, emb_dropout=0.0)
            cfg.loss = dict(sigreg=OmegaConf.to_container(base.loss.sigreg),
                            delta_weight=0.01 if arm == 'delta' else 0.0,
                            sparse_weight=0.001 if arm == 'delta' else 0.0)
            for key in ('warm_start', 'anchor_path', 'freeze_encoder', 'freeze_dynamics',
                        'freeze_old_adapters', 'old_path_diagnostic', 'fixed_action_stats'):
                cfg.pop(key, None)
            if mode == 'compare':
                cfg.loader.batch_size = batch
                cfg.data.dataset.keys_to_load = ['pixels', 'action']
                cfg.data.dataset.keys_to_cache = ['action']
                stats_path = output / f'{arm}_action_stats.json'
                stats_path.write_text(json.dumps(dict(stats, policy=f'{run}/weights_compare.pt'), indent=2) + '\n')
                cfg.fixed_action_stats = str(stats_path)
        path = output / f'{arm}.yaml'
        OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), path)
        configs.append((arm, run, path))
    return configs


def evaluate_arm(root, output, home, env, arm, run, num_eval):
    policy = f'{run}/weights_compare.pt'
    result_path = output / f'{arm}_behavior.json'
    command = [sys.executable, '-u', str(root/'cl_protocol/eval_task.py'),
               '--sequence', 'A_pusht', '--task', 'T1', '--policy', policy,
               '--num-eval', str(num_eval), '--seed', '42',
               '--action-stats', str(output/f'{arm}_action_stats.json'),
               '--scene-manifest', str(output/'scenes.json'), '--scene-result', str(result_path),
               '--extra', 'compile=false', '--extra', 'bf16=false']
    subprocess.run(command, cwd=root, env=env, check=True)
    result = json.loads(result_path.read_text())
    expected = json.loads((output/'scenes.json').read_text())['scenes']
    if result['status'] != 'complete' or result['scenes'] != expected or len(result['successes']) != num_eval:
        raise RuntimeError('Behavior receipt does not match fixed scenes')
    print('[COMPARE_BEHAVIOR_OK]', arm, result['success_rate'], flush=True)
    return result


def paired_comparison(rows):
    import numpy as np
    a, b = [np.asarray(r['behavior']['successes'], dtype=float) for r in rows]
    diff = 100 * (b-a)
    rng = np.random.default_rng(0)
    ci = np.percentile(diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(1), [2.5,97.5])
    return dict(delta_minus_absolute_pp=float(diff.mean()), paired_bootstrap95_pp=ci.tolist(),
                scope='Scene uncertainty only; one training seed, fixed initial training budget')


def prepare_t2_baseline(root, output, tag, *, arm=None, reference_config=None, schedule=None, structure=None, bn_delta=False):
    """Prepare T2 configurations; BN residual comparison explicitly avoids V2 rewrites."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', lambda s: sum(int(x.strip()) for x in s.split('+')))
    if schedule is not None and (schedule not in ('epoch', 'step') or arm is not None):
        raise ValueError('Schedule comparison uses ordinary legacy structure only')
    if structure is not None and (structure not in ('projector_ln', 'no_pred_proj')
                                  or schedule != 'step' or arm is not None):
        raise ValueError('Structure comparison requires one ordinary-predictor factor and step scheduling')
    if bn_delta and (arm is not None or structure is not None or schedule != 'step'):
        raise ValueError('BN delta uses the ordinary BN/Identity control and step scheduling')
    if arm is not None or schedule is not None:
        if (arm is not None and arm not in ('absolute', 'delta')) or reference_config is None:
            raise ValueError('T2 arms require the completed baseline training config')
        cfg = OmegaConf.load(reference_config)
        if (cfg.data.dataset.name != 'pusht_t2.lance' or cfg.trainer.max_epochs != 30
                or cfg.loader.batch_size != 128 or cfg.train_split != 0.9
                or cfg.seed != 3072 or int(cfg.get('mechanism_version', 1)) != 1
                or not cfg.model.predictor._target_.endswith('.Predictor')):
            raise ValueError('Reference is not the accepted T2 ordinary baseline')
        if any(cfg.get(k) for k in ('warm_start', 'smoke_steps', 'comparison_steps',
                                   'episode_split', 'fixed_action_stats', 'anchor_path',
                                   'old_path_diagnostic')):
            raise ValueError('Reference contains an unexpected training protocol override')
    else:
        with initialize_config_dir(config_dir=str(root/'scripts/train/config'), version_base=None):
            cfg = compose(config_name='lewm', overrides=['data=pusht', f'subdir={tag}'])
    with open_dict(cfg):
        cfg.output_model_name = cfg.subdir = tag
        cfg.mechanism_version = 2 if arm else 1
        cfg.data.dataset.name = 'pusht_t2.lance'
        cfg.trainer.max_epochs = 30
        cfg.trainer.max_steps = -1
        cfg.loader.batch_size = 128
        cfg.train_split = 0.9
        cfg.seed = 3072
        cfg.model.predictor._target_ = ('stable_worldmodel.wm.lewm.module.DiffPredictor'
                                       if arm == 'delta' or bn_delta else 'stable_worldmodel.wm.lewm.module.Predictor')
        cfg.loss.delta_weight = 0.01 if arm == 'delta' or bn_delta else 0.0
        cfg.loss.sparse_weight = 0.001 if arm == 'delta' or bn_delta else 0.0
        cfg.wandb.enabled = False
        cfg.save_epoch_interval = 10
        for key in ('warm_start', 'smoke_steps', 'comparison_steps', 'episode_split',
                    'fixed_action_stats', 'anchor_path', 'old_path_diagnostic'):
            cfg.pop(key, None)
        if schedule is not None:
            cfg.scheduler_interval = schedule
            cfg.seed_initialization = True
            cfg.record_lr_trace = True
            expected_projection = ('torch.nn.Identity' if bn_delta
                                   else 'stable_worldmodel.wm.lewm.module.MLP')
            if (cfg.model.projector.norm_fn._target_ != 'torch.nn.BatchNorm1d'
                    or cfg.model.pred_proj._target_ != expected_projection):
                raise ValueError(f'Expected BN image projector and prediction projection {expected_projection}')
        if structure == 'projector_ln':
            cfg.model.projector.norm_fn._target_ = 'torch.nn.LayerNorm'
        elif structure == 'no_pred_proj':
            cfg.model.pred_proj = {'_target_': 'torch.nn.Identity'}
    path = output/'baseline.yaml'
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), path)
    return path


def run_t2_baseline(root, home, tag, num_eval, *, arm=None, reference_config=None, schedule=None, structure=None, bn_delta=False):
    """One 30-epoch baseline and T2 evaluation, then stop for inspection."""
    data = home/'datasets/pusht_t2.lance'
    if not (data/'_versions').is_dir():
        raise FileNotFoundError(f'Required original T2 dataset missing: {data}; no substitute/download')
    if (home/'checkpoints'/tag).exists():
        raise FileExistsError(f'Checkpoint output already exists: {tag}')
    output = root/'results'/tag
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='preparing', dataset=str(data), task='T2', epochs=30,
                   batch=128, seed=3072, num_eval=num_eval, mechanism_version=2 if arm else 1,
                   arm=arm, reference_config=str(reference_config) if reference_config else None,
                   scheduler_interval=schedule, seed_initialization=True if schedule else None,
                   scope='T2 30-epoch run; window split and same-dataset evaluation retained, '
                         'not independent episode holdout or exact historical random-state replay')
    if structure is not None:
        receipt['structure_factor'] = structure
    if bn_delta:
        receipt.update(arm='delta', residual_protocol='BN image projector; Identity output projection; '
                       'complete prediction = context + mask * delta; no V2 normalization rewrite')
    def save():
        (output/'summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    try:
        config = prepare_t2_baseline(root, output, tag, arm=arm, reference_config=reference_config,
                                     schedule=schedule, structure=structure, bn_delta=bn_delta)
        env = dict(os.environ, STABLEWM_HOME=str(home), PYTHONUNBUFFERED='1',
                   MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa')
        env.pop('LOCAL_DATASET_DIR', None)
        receipt['status'] = 'training'
        save()
        print('[T2_BASELINE_START]', json.dumps(receipt), flush=True)
        subprocess.run([sys.executable, '-u', str(root/'scripts/train/lewm.py'),
                        '--config-path', str(output), '--config-name', config.stem],
                       cwd=root, env=env, check=True)
        policy = f'{tag}/weights_epoch_30.pt'
        if not (home/'checkpoints'/policy).is_file():
            raise FileNotFoundError(f'Final 30-epoch checkpoint missing: {policy}')
        if schedule is not None:
            trace_path = home/'checkpoints'/tag/'learning_rate_trace.json'
            trace = json.loads(trace_path.read_text())
            if (trace['status'] != 'complete' or trace['epochs_completed'] != 30
                    or trace['scheduler_interval'] != schedule or not trace['records']
                    or not trace['schedulers']
                    or any(s['interval'] != schedule or s['frequency'] != 1 for s in trace['schedulers'])):
                raise RuntimeError('Actual scheduler trace does not match the requested 30-epoch arm')
            receipt['learning_rate_trace'] = str(trace_path)
            receipt['global_step'] = trace['global_step']
        receipt.update(status='evaluating', policy=policy)
        save()
        command = [sys.executable, '-u', str(root/'cl_protocol/eval_task.py'),
                   '--sequence', 'A_pusht', '--task', 'T2', '--policy', policy,
                   '--num-eval', str(num_eval), '--seed', '42']
        records = []
        with (output/'evaluation.log').open('w', encoding='utf-8') as log:
            with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                  errors='replace', bufsize=1) as proc:
                for line in proc.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
                    if line.startswith('CELL_RESULT '):
                        records.append(json.loads(line[len('CELL_RESULT '):]))
                code = proc.wait()
        if code != 0:
            raise RuntimeError(f'T2 evaluation exited {code}; see evaluation.log')
        if len(records) != 1 or any(records[0].get(k) != v for k, v in
                dict(policy=policy, task='T2', dataset='pusht_t2.lance', num_eval=num_eval, seed=42).items()):
            raise RuntimeError('Missing or mismatched T2 evaluation receipt')
        receipt.update(status='complete', behavior=records[0])
        save()
        print('[T2_BASELINE_DONE]', json.dumps(receipt), flush=True)
        return receipt
    except Exception as exc:
        receipt.update(status='failed', error=str(exc))
        save()
        raise


def run_t2_v2(root, home, tag, num_eval, baseline_tag, *, scheduler_only=False):
    arms = ('epoch', 'step') if scheduler_only else ('absolute', 'delta')
    label = 'T2_SCHEDULE' if scheduler_only else 'T2_V2'
    baseline_dir = root/'results'/baseline_tag
    baseline = json.loads((baseline_dir/'summary.json').read_text())
    if (baseline.get('status') != 'complete' or baseline.get('mechanism_version') != 1
            or baseline.get('epochs') != 30 or baseline.get('batch') != 128
            or baseline.get('num_eval') != num_eval or baseline.get('task') != 'T2'
            or baseline['behavior'].get('dataset') != 'pusht_t2.lance'
            or baseline['behavior'].get('seed') != 42):
        raise ValueError('Completed baseline receipt does not match the T2 comparison')
    reference = home/'checkpoints'/Path(baseline['policy']).parent/'config.yaml'
    if not reference.is_file():
        raise FileNotFoundError(reference)
    for name in (tag, *(tag+'_'+arm for arm in arms)):
        if (root/'results'/name).exists() or (home/'checkpoints'/name).exists():
            raise FileExistsError(f'Output already exists: {name}; no overwrite or implicit resume')
    output = root/'results'/tag
    output.mkdir(parents=True)
    summary = dict(status='preparing', dataset='pusht_t2.lance', task='T2',
                   epochs=30, batch=128, seed=3072, eval_seed=42, num_eval=num_eval,
                   baseline_policy=baseline['policy'], baseline_success_rate=baseline['behavior']['success_rate'],
                   rows=[], scope='V2 absolute versus V2 delta, single training seed; '
                   'V1 is a reference only. Native same-dataset evaluation, no independent holdout claim.')
    if scheduler_only:
        summary['scope'] = ('Original ordinary predictor, BN and output MLP; global initialization seed set '
                            'in both arms; only scheduler interval changes. Warmup/max_steps numeric '
                            'arguments stay identical to isolate the previous switch. Same-dataset evaluation.')
    def save():
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    save()
    try:
        # Prepare both configurations before spending GPU time.
        for arm in arms:
            folder = output/arm
            folder.mkdir()
            prepare_t2_baseline(root, folder, tag+'_'+arm, arm=None if scheduler_only else arm,
                                schedule=arm if scheduler_only else None, reference_config=reference)
        for arm in arms:
            summary['status'] = 'running_'+arm
            save()
            print(f'[{label}_ARM_START]', arm, flush=True)
            row = run_t2_baseline(root, home, tag+'_'+arm, num_eval,
                                  arm=None if scheduler_only else arm, reference_config=reference,
                                  schedule=arm if scheduler_only else None)
            for key in ('dataset', 'num_eval', 'seed', 'variation_values', 'darken'):
                if row['behavior'].get(key) != baseline['behavior'].get(key):
                    raise ValueError(f'Evaluation protocol changed from baseline: {key}')
            summary['rows'].append(dict(arm=arm, policy=row['policy'],
                                       success_rate=row['behavior']['success_rate'],
                                       receipt=str(root/'results'/(tag+'_'+arm)/'summary.json')))
            if scheduler_only:
                summary['rows'][-1].update(global_step=row['global_step'],
                                           learning_rate_trace=row['learning_rate_trace'])
            save()
            print(f'[{label}_ARM_DONE]', json.dumps(summary['rows'][-1]), flush=True)
        if scheduler_only and summary['rows'][0]['global_step'] != summary['rows'][1]['global_step']:
            raise RuntimeError('Schedule arms completed different numbers of optimizer updates')
        difference_key = 'step_minus_epoch_pp' if scheduler_only else 'delta_minus_absolute_pp'
        summary[difference_key] = summary['rows'][1]['success_rate']-summary['rows'][0]['success_rate']
        summary['status'] = 'complete'
        save()
        print(f'[{label}_DONE]', json.dumps(summary), flush=True)
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        save()
        raise


def load_t2_step_control(root, home, num_eval, control_tag):
    """Read the completed ordinary control and its actual scheduler receipt."""
    from omegaconf import OmegaConf

    control = json.loads((root/'results'/control_tag/'summary.json').read_text())
    behavior = control['behavior']
    if (control.get('status') != 'complete' or control.get('mechanism_version') != 1
            or control.get('epochs') != 30 or control.get('batch') != 128
            or control.get('seed') != 3072 or control.get('scheduler_interval') != 'step'
            or control.get('seed_initialization') is not True
            or control.get('task') != 'T2' or control.get('num_eval') != num_eval
            or any(behavior.get(k) != v for k, v in dict(task='T2', dataset='pusht_t2.lance',
                                                        num_eval=num_eval, seed=42).items())
            or behavior.get('policy') != control.get('policy')):
        raise ValueError('Control must be the completed matched-seed ordinary T2 step arm')
    checkpoint_dir = home/'checkpoints'/Path(control['policy']).parent
    if not (home/'checkpoints'/control['policy']).is_file():
        raise FileNotFoundError('Completed control checkpoint is missing')
    reference = checkpoint_dir/'config.yaml'
    cfg = OmegaConf.load(reference)
    if (cfg.get('scheduler_interval') != 'step' or cfg.get('seed_initialization') is not True
            or cfg.get('record_lr_trace') is not True):
        raise ValueError('Control config does not match its seed and scheduler receipt')
    trace = json.loads((checkpoint_dir/'learning_rate_trace.json').read_text())
    if (trace.get('status') != 'complete' or trace.get('epochs_completed') != 30
            or trace.get('scheduler_interval') != 'step' or not trace.get('records')
            or trace.get('global_step', 0) <= 0 or trace['global_step'] != control.get('global_step')
            or not trace.get('schedulers')
            or any(s['interval'] != 'step' or s['frequency'] != 1 for s in trace['schedulers'])):
        raise ValueError('Control LR trace is incomplete or inconsistent')
    return control, cfg, reference, trace


def run_t2_structure(root, home, tag, num_eval, control_tag):
    """Reuse the completed step control; train exactly two single-factor arms."""
    from omegaconf import OmegaConf

    control, cfg, reference, trace = load_t2_step_control(root, home, num_eval, control_tag)
    behavior = control['behavior']
    arms = ('projector_ln', 'no_pred_proj')
    for name in (tag, *(tag+'_'+arm for arm in arms)):
        if (root/'results'/name).exists() or (home/'checkpoints'/name).exists():
            raise FileExistsError(f'Output already exists: {name}; no overwrite or implicit resume')
    output = root/'results'/tag
    output.mkdir(parents=True)
    summary = dict(status='preparing', dataset='pusht_t2.lance', task='T2', epochs=30,
                   batch=128, seed=3072, eval_seed=42, num_eval=num_eval,
                   control_policy=control['policy'], control_success_rate=behavior['success_rate'],
                   control_receipt=str(root/'results'/control_tag/'summary.json'),
                   expected_updates=trace['global_step'], rows=[],
                   scope='Original ordinary predictor; projector_ln changes only image-projector BN to LN; '
                         'no_pred_proj changes only prediction output MLP to Identity. Both use step scheduling '
                         'and fixed seed; reuse completed control. Same-dataset T2 evaluation, single training seed.')
    def save():
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    save()
    try:
        # Validate both exact configuration changes before starting GPU work.
        baseline = OmegaConf.to_container(cfg, resolve=True)
        for arm in arms:
            folder = output/arm
            folder.mkdir()
            path = prepare_t2_baseline(root, folder, tag+'_'+arm, reference_config=reference,
                                       schedule='step', structure=arm)
            candidate = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            candidate['output_model_name'] = baseline['output_model_name']
            candidate['subdir'] = baseline['subdir']
            if arm == 'projector_ln':
                candidate['model']['projector']['norm_fn'] = baseline['model']['projector']['norm_fn']
            else:
                candidate['model']['pred_proj'] = baseline['model']['pred_proj']
            if candidate != baseline:
                raise ValueError(f'Unexpected configuration change beyond the requested factor: {arm}')
        for arm in arms:
            summary['status'] = 'running_'+arm
            save()
            print('[T2_STRUCTURE_ARM_START]', arm, flush=True)
            row = run_t2_baseline(root, home, tag+'_'+arm, num_eval, reference_config=reference,
                                  schedule='step', structure=arm)
            if row['global_step'] != summary['expected_updates']:
                raise RuntimeError('Structure arm completed a different number of updates from control')
            for key in ('dataset', 'num_eval', 'seed', 'variation_values', 'darken', 'model_protocol'):
                if row['behavior'].get(key) != behavior.get(key):
                    raise ValueError(f'Evaluation protocol changed from control: {key}')
            summary['rows'].append(dict(arm=arm, policy=row['policy'],
                success_rate=row['behavior']['success_rate'],
                change_from_control_pp=row['behavior']['success_rate']-behavior['success_rate'],
                global_step=row['global_step'], learning_rate_trace=row['learning_rate_trace'],
                receipt=str(root/'results'/(tag+'_'+arm)/'summary.json')))
            save()
            print('[T2_STRUCTURE_ARM_DONE]', json.dumps(summary['rows'][-1]), flush=True)
        summary['status'] = 'complete'
        save()
        print('[T2_STRUCTURE_DONE]', json.dumps(summary), flush=True)
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        save()
        raise


def run_t2_bn_delta(root, home, tag, num_eval, control_tag):
    """Compare a BN/Identity residual predictor with the already completed ordinary arm."""
    from omegaconf import OmegaConf

    control, cfg, reference, trace = load_t2_step_control(root, home, num_eval, control_tag)
    behavior = control['behavior']
    if (cfg.model.projector.norm_fn._target_ != 'torch.nn.BatchNorm1d'
            or cfg.model.pred_proj._target_ != 'torch.nn.Identity'):
        raise ValueError('BN delta requires the completed BN/Identity ordinary control, not the LN or MLP arm')
    # Existing DiffPredictor already returns context + mask * delta. With the
    # outer projection fixed to Identity, this is also the complete prediction.
    # The existing delta L1 then equals final prediction L1, not an independent constraint.
    for name in (tag, tag+'_delta'):
        if (root/'results'/name).exists() or (home/'checkpoints'/name).exists():
            raise FileExistsError(f'Output already exists: {name}; no overwrite or implicit resume')
    output = root/'results'/tag
    output.mkdir(parents=True)
    summary = dict(status='preparing', dataset='pusht_t2.lance', task='T2', epochs=30,
                   batch=128, seed=3072, eval_seed=42, num_eval=num_eval,
                   expected_updates=trace['global_step'], rows=[dict(arm='ordinary', reused=True,
                       policy=control['policy'], success_rate=behavior['success_rate'],
                       receipt=str(root/'results'/control_tag/'summary.json'))],
                   scope='BN image projector and Identity prediction projection in both arms; ordinary control reused. '
                         'Train DiffPredictor with delta_weight=.01 and sparse_weight=.001, otherwise unchanged. '
                         'Single training seed, native same-dataset T2 evaluation; no continual-retention claim.')
    def save():
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    save()
    try:
        config = prepare_t2_baseline(root, output, tag+'_delta', reference_config=reference,
                                     schedule='step', bn_delta=True)
        baseline = OmegaConf.to_container(cfg, resolve=True)
        candidate = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
        candidate['output_model_name'] = baseline['output_model_name']
        candidate['subdir'] = baseline['subdir']
        candidate['model']['predictor']['_target_'] = baseline['model']['predictor']['_target_']
        for key in ('delta_weight', 'sparse_weight'):
            candidate['loss'][key] = baseline['loss'][key]
        if candidate != baseline:
            raise ValueError('Unexpected configuration change beyond the delta predictor and its losses')
        summary['status'] = 'training_delta'
        save()
        print('[T2_BN_DELTA_START]', json.dumps(summary), flush=True)
        row = run_t2_baseline(root, home, tag+'_delta', num_eval, reference_config=reference,
                              schedule='step', bn_delta=True)
        if row['global_step'] != summary['expected_updates']:
            raise RuntimeError('Delta arm completed a different number of updates from ordinary control')
        for key in ('dataset', 'num_eval', 'seed', 'variation_values', 'darken', 'model_protocol'):
            if row['behavior'].get(key) != behavior.get(key):
                raise ValueError(f'Evaluation protocol changed from ordinary control: {key}')
        summary['rows'].append(dict(arm='delta', reused=False, policy=row['policy'],
            success_rate=row['behavior']['success_rate'], global_step=row['global_step'],
            learning_rate_trace=row['learning_rate_trace'],
            receipt=str(root/'results'/(tag+'_delta')/'summary.json')))
        summary['delta_minus_ordinary_pp'] = row['behavior']['success_rate']-behavior['success_rate']
        summary['status'] = 'complete'
        save()
        print('[T2_BN_DELTA_DONE]', json.dumps(summary), flush=True)
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        save()
        raise
