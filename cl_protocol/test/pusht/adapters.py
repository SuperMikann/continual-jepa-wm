"""Fixed-capacity adapter and corrected-gate comparison."""
import json
import os
from pathlib import Path
import subprocess
import sys

from .diagnostics import expand_planning_info, native_policy_history_probe, validate_history_config


def adapter_initial_state(state, arm):
    """Explicit uniform-bank conversion; each arm starts from the same function."""
    import torch
    sd = {k: v.detach().clone() for k, v in state.items()}
    if arm not in ('finetune', 'uniform', 'gated'):
        raise ValueError('Unknown adapter comparison arm')
    a, b = sd['predictor.adapter_A'], sd['predictor.adapter_B']
    scale = sd.get('predictor.adapter_scale', a.new_tensor(1.0))
    if a.shape[0] < 2 or a.shape[0] != b.shape[0]:
        raise ValueError('The routing comparison needs an existing bank with at least two slots')
    if arm == 'finetune':
        # A uniform linear bank is exactly another linear delta head. Keep its
        # learned function when removing its factorization, rather than discarding it.
        sd['predictor.delta_head.weight'] += scale * torch.einsum('kdr,krf->df', a, b) / a.shape[0]
        for k in list(sd):
            if k.startswith(('predictor.adapter_', 'predictor.gate.', 'predictor.gate_protocol_version')):
                del sd[k]
    else:
        # The old gate was unused. Zero logits give exactly uniform routing at t=0.
        sd['predictor.gate.2.weight'].zero_()
        sd['predictor.gate.2.bias'].zero_()
        sd['predictor.gate_protocol_version'] = torch.tensor(2, dtype=torch.long)
    return sd


def adapter_stage_config(base, model_config, run, initial, stats_path, epochs):
    from omegaconf import OmegaConf, open_dict
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    with open_dict(cfg):
        for k in ('anchor_path', 'freeze_encoder', 'freeze_dynamics', 'freeze_old_adapters',
                  'old_path_diagnostic', 'episode_split', 'smoke_steps', 'comparison_steps',
                  'fixed_action_normalization'):
            cfg.pop(k, None)
        cfg.model = model_config
        cfg.output_model_name = cfg.subdir = run
        cfg.mechanism_version = 1  # Routing repair is separate from the V2 LN rewrite.
        cfg.warm_start = str(initial)
        cfg.fixed_action_stats = str(stats_path)
        cfg.scheduler_interval = 'step'
        cfg.seed_initialization = cfg.record_lr_trace = True
        cfg.seed = 3072
        cfg.train_split = .9
        cfg.data.dataset.name = 'pusht_t2.lance'
        cfg.data.dataset.keys_to_load = ['pixels', 'action']
        cfg.data.dataset.keys_to_cache = ['action']
        cfg.trainer.max_epochs = epochs
        cfg.trainer.max_steps = -1
        cfg.trainer.devices = 1
        cfg.trainer.accelerator = 'gpu'
        cfg.trainer.precision = 'bf16-mixed'
        cfg.loader.batch_size = 128
        cfg.num_workers = cfg.loader.num_workers = 6
        cfg.loader.persistent_workers = True
        cfg.loader.prefetch_factor = 3
        cfg.wandb.enabled = False
        cfg.save_epoch_interval = epochs  # Final custom .pt only; native Trainer .ckpt is separate.
        cfg.loss = dict(sigreg=OmegaConf.to_container(base.loss.sigreg, resolve=True),
                        delta_weight=0., sparse_weight=0.,
                        ent_weight=1e-4 if run.endswith('_gated') else 0.)
    return cfg


def adapter_scene_manifest(dataset, num_eval):
    import numpy as np
    names = set(dataset.column_names) | set(getattr(dataset, '_schema_names', ()))
    ep = np.asarray(dataset.get_col_data('episode_idx' if 'episode_idx' in names else 'ep_idx'))
    step = np.asarray(dataset.get_col_data('step_idx'))
    maximum = {}
    for e, s in zip(ep, step):
        maximum[int(e)] = max(maximum.get(int(e), 0), int(s))
    valid = np.flatnonzero(step <= np.asarray([maximum[int(e)]-25 for e in ep]))
    if len(valid) < num_eval:
        raise ValueError('Insufficient native valid starts')
    pick = np.sort(np.random.default_rng(42).choice(valid, num_eval, replace=False))
    return dict(dataset=None, episode_lengths=list(map(int, dataset.lengths)),
                scenes=[dict(episode_idx=int(ep[i]), start_step=int(step[i])) for i in pick],
                goal_offset=25, eval_budget=50)


def adapter_pair(before, after):
    import numpy as np
    d = 100*(np.asarray(after, dtype=float)-np.asarray(before, dtype=float))
    rng = np.random.default_rng(0)
    ci = np.percentile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), [2.5, 97.5])
    return dict(change_pp=float(d.mean()), paired_bootstrap95_pp=ci.tolist(),
                lost_successes=int(((np.asarray(before) == 1) & (np.asarray(after) == 0)).sum()),
                gained_successes=int(((np.asarray(before) == 0) & (np.asarray(after) == 1)).sum()))


def run_adapter_t2(root, home, tag, base_policy, epochs, num_eval):
    """Fixed-capacity M1 comparison; growth/frozen-path diagnosis is separate."""
    print('[ADAPTER_START]', flush=True)
    import copy
    import gc
    import shutil
    import numpy as np
    import torch
    import hydra
    import yaml
    from omegaconf import OmegaConf, open_dict
    import stable_worldmodel as swm
    from stable_worldmodel.wm.lewm.module import CausalAdapterPredictor, FixedActionNormalizer

    # The native backbone registers its own 'eval' resolver on first import.
    # Load it before our fallback, otherwise its later Hydra import collides.
    from importlib import import_module
    import_module('stable_pretraining.backbone.utils')
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', lambda s: sum(int(x.strip()) for x in s.split('+')))
    print('[ADAPTER_BACKBONE_READY]', flush=True)
    checkpoint = (home/'checkpoints'/base_policy).resolve()
    if not checkpoint.is_relative_to((home/'checkpoints').resolve()) or not checkpoint.is_file():
        raise FileNotFoundError('Specify an existing local base-policy under checkpoints; no automatic selection')
    base = OmegaConf.load(checkpoint.parent/'config.yaml')
    source_config = json.loads((checkpoint.parent/'config.json').read_text())
    if (base.data.dataset.name != 'pusht_expert_train.lance'
            or not source_config['predictor']['_target_'].endswith('.AdapterPredictor')
            or source_config['predictor'].get('freeze_gate') is not True
            or int(source_config['predictor'].get('num_adapters', 4)) < 2
            or int(base.get('mechanism_version', 1)) != 1
            or base.wm.history_size != 3 or base.wm.num_preds != 1
            or base.data.dataset.frameskip != 5
            or any(base.get(k) for k in ('fixed_action_stats', 'episode_split', 'old_path_diagnostic'))):
        raise ValueError('Need the original uniform multi-adapter PushT T1 checkpoint/config; do not substitute another model')
    arms = ('finetune', 'uniform', 'gated')
    for name in (tag, *(tag+'_'+a for a in arms), *(tag+'_initial_'+a for a in arms),
                 *(tag+'_check_'+a for a in arms)):
        if (root/'results'/name).exists() or (home/'checkpoints'/name).exists():
            raise FileExistsError(f'Use a fresh tag; output already exists: {name}')
    for name in ('pusht_expert_train.lance', 'pusht_t2.lance'):
        if not (home/'datasets'/name/'_versions').is_dir():
            raise FileNotFoundError(f'Local data missing: {name}; no download attempted')
    required = 12*checkpoint.stat().st_size + 512*2**20
    if shutil.disk_usage(home).free < required:
        raise RuntimeError(f'Need at least {required/2**30:.2f} GiB for initial/final checkpoints and logs')
    if not torch.cuda.is_available():
        raise RuntimeError('Run this queue on the AutoDL GPU')
    os.environ['STABLEWM_HOME'] = str(home)
    os.environ.pop('LOCAL_DATASET_DIR', None)
    env = dict(os.environ, PYTHONUNBUFFERED='1', MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa')
    output = root/'results'/tag
    output.mkdir(parents=True)
    receipt = dict(status='preflight', base_policy=base_policy, epochs=epochs, batch=128,
        train_seed=3072, eval_seed=42, num_eval=num_eval, rows=[], comparisons=[],
        scope='Fixed-capacity adapter/routing T1-to-T2 comparison, one training seed, native same-dataset scenes; '
              'not a growth, full-old-path freeze or task-free condition-identification result. '
              'Retain original image/output projections; only routing uses causal prefix pooling. '
              'Train-mode image BatchNorm retains the original cross-frame statistics.',
        common_loss='MSE + original SIGReg; no delta L1/sparsity/anchors/subspace/replay',
        gate_entropy_weight=1e-4, checkpoint_space_budget_bytes=required)
    def save():
        (output/'summary.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    def invoke(command, label):
        records = []
        routes = []
        print('[ADAPTER_STEP_START]', label, flush=True)
        with (output/(label+'.log')).open('w', encoding='utf-8') as log:
            with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1) as proc:
                for line in proc.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
                    if line.startswith('CELL_RESULT '):
                        records.append(json.loads(line[len('CELL_RESULT '):]))
                    if line.startswith('[ADAPTER_ROUTE_PROTOCOL] '):
                        routes.append(json.loads(line[len('[ADAPTER_ROUTE_PROTOCOL] '):]))
                code = proc.wait()
        if code:
            raise RuntimeError(f'{label} exited {code}; see {label}.log')
        if label in ('uniform_train', 'gated_train', 'uniform_check', 'gated_check'):
            uniform = label.startswith('uniform_')
            if (len(routes) != 1 or routes[0].get('gate_version') != 2
                    or routes[0].get('entropy_sign') != 'positive'
                    or routes[0].get('freeze_gate') != uniform):
                raise RuntimeError('Child training did not confirm the standalone corrected routing protocol')
        return records
    def evaluate(policy, stats_path, label, task):
        scene_path = output/(task+'_scenes.json')
        result_path = output/(label+'_'+task+'_scenes.json')
        records = invoke([sys.executable, '-u', str(root/'cl_protocol/eval_task.py'),
            '--sequence', 'A_pusht', '--task', task, '--policy', policy,
            '--num-eval', str(num_eval), '--seed', '42', '--action-stats', str(stats_path),
            '--scene-manifest', str(scene_path), '--scene-result', str(result_path),
            '--extra', '++plan_config.history_len=3'], label+'_'+task)
        if len(records) != 1:
            raise ValueError('Expected one fresh CELL_RESULT')
        record = records[0]
        expected = dict(policy=policy, task=task, dataset=datasets[task], num_eval=num_eval, seed=42,
            darken=1. if task == 'T1' else .5,
            variation_values={} if task == 'T1' else {'goal.position': [100, 100]})
        if any(record.get(k) != v for k,v in expected.items()):
            raise ValueError('Adapter evaluation protocol changed')
        result = json.loads(result_path.read_text())
        if (result['status'] != 'complete' or result['scenes'] != manifests[task]['scenes']
                or len(result['successes']) != num_eval
                or abs(100*sum(result['successes'])/num_eval-record['success_rate']) > 1e-7):
            raise ValueError('Adapter cell scenes or success flags mismatch')
        with Path(record['result_path']).open('rb') as f:
            f.seek(record['result_start_offset'])
            fresh = f.read().decode('utf-8')
        actual_cfg = yaml.safe_load(fresh.split('==== CONFIG ====')[1].split('==== RESULTS ====')[0])
        validate_history_config(actual_cfg, plan_dict, solver_dict, 3)
        normal = record['model_protocol']['normalization']
        if any(normal[k] != stats[k] for k in ('mean', 'scale', 'source')):
            raise ValueError('Evaluation did not use the fixed old action coordinates')
        print('[ADAPTER_CELL_DONE]', json.dumps(dict(arm=label, task=task,
              success_rate=record['success_rate'])), flush=True)
        return result
    save()
    try:
        with hydra.initialize_config_dir(config_dir=str(root/'scripts/plan/config'), version_base=None):
            plan_cfg = hydra.compose(config_name='pusht', overrides=['seed=42', f'eval.num_eval={num_eval}'])
        plan_dict = OmegaConf.to_container(plan_cfg.plan_config, resolve=True)
        solver_dict = OmegaConf.to_container(plan_cfg.solver, resolve=True)
        if (plan_dict['horizon'] != 5 or plan_dict['action_block'] != 5
                or plan_dict['receding_horizon'] != 5 or plan_cfg.eval.goal_offset_steps != 25
                or plan_cfg.eval.eval_budget != 50 or plan_cfg.get('compile', False)
                or plan_cfg.get('bf16', False) or not solver_dict['_target_'].endswith('.CEMSolver')):
            raise ValueError('Native planning settings changed')
        shape = native_policy_history_probe(swm, swm.PlanConfig(**{**plan_dict, 'history_len': 3}),
            {**OmegaConf.to_container(plan_cfg.world, resolve=False), 'num_envs': 1,
             'max_episode_steps': 200, 'image_shape': (224, 224)})
        if shape['solver_input_frames'] != [1,3] or shape['solver_action_history_blocks'] != [0,2]:
            raise ValueError('Install the already validated native policy history repair first')
        datasets = dict(T1='pusht_expert_train.lance', T2='pusht_t2.lance')
        manifests, loaded = {}, {}
        for task, name in datasets.items():
            loaded[task] = swm.data.load_dataset(str(home/'datasets'/name), frameskip=1,
                num_steps=1, keys_to_load=['action'], keys_to_cache=['action'])
            manifests[task] = adapter_scene_manifest(loaded[task], num_eval)
            manifests[task]['dataset'] = name
            (output/(task+'_scenes.json')).write_text(json.dumps(manifests[task], indent=2)+'\n')
        # Recreate the old training coordinate with the native scaler, not a fit on new T2.
        from stable_worldmodel.data.normalization import ZScoreScaler
        scaler = ZScoreScaler().fit(loaded['T1'].get_col_data('action'))
        stats = dict(mean=np.asarray(scaler.mean).reshape(-1).tolist(),
            scale=np.maximum(np.asarray(scaler.std).reshape(-1), 1e-8).tolist(),
            source=f'Native ZScoreScaler on original T1 dataset {datasets["T1"]}; '
                   f'base config {checkpoint.parent/"config.yaml"}', policy=base_policy)
        stats_path = output/'action_stats.json'
        stats_path.write_text(json.dumps(stats, indent=2)+'\n')
        original = swm.wm.utils.load_pretrained(str(checkpoint)).float().cuda().eval()
        if type(original.predictor).__name__ != 'AdapterPredictor' or not original.predictor.freeze_gate:
            raise ValueError('Loaded checkpoint differs from the uniform legacy source configuration')
        original.requires_grad_(False)
        sd = {k:v.detach().cpu() for k,v in original.state_dict().items()}
        generator = torch.Generator().manual_seed(3072)
        pixels = torch.randn(1,3,3,224,224, generator=generator).cuda()
        past = torch.randn(1,2,10, generator=generator).cuda()
        candidates = torch.randn(1,4,5,10, generator=generator).cuda()
        probe = dict(pixels=pixels, action_history=past)
        with torch.inference_mode():
            target = original.rollout(expand_planning_info(probe,4), candidates)['predicted_emb']
        if target.shape[2] != 8:
            raise ValueError('Install the already validated five-future-step LeWM rollout first')
        prepared = []
        for arm in arms:
            model_cfg = copy.deepcopy(source_config)
            pred = model_cfg['predictor']
            if arm == 'finetune':
                pred['_target_'] = pred['_target_'].rsplit('.',1)[0]+'.DiffPredictor'
                for k in ('num_adapters', 'rank', 'gate_hidden', 'gate_tau', 'freeze_gate'):
                    pred.pop(k, None)
            else:
                pred['_target_'] = pred['_target_'].rsplit('.',1)[0]+'.CausalAdapterPredictor'
                pred['freeze_gate'] = arm == 'uniform'
            model = hydra.utils.instantiate(model_cfg).float().cuda().eval()
            model.load_state_dict(adapter_initial_state(sd,arm), strict=True)
            with torch.inference_mode():
                actual = model.rollout(expand_planning_info(probe,4), candidates)['predicted_emb']
            torch.testing.assert_close(actual, target, rtol=1e-4, atol=1e-4)
            record = dict(arm=arm, initial_rollout_max_abs_diff=float((actual-target).abs().max()))
            initial_run = tag+'_initial_'+arm
            swm.wm.utils.save_pretrained(model.cpu(), initial_run, config=model_cfg, filename='weights_initial.pt')
            initial = home/'checkpoints'/initial_run/'weights_initial.pt'
            restored = swm.wm.utils.load_pretrained(str(initial))
            if any(not torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items()):
                raise RuntimeError('Converted checkpoint roundtrip changed its state')
            run = tag+'_'+arm
            cfg = adapter_stage_config(base, model_cfg, run, initial, stats_path, epochs)
            path = output/(arm+'.yaml')
            OmegaConf.save(cfg, path)
            prepared.append((arm,run,path,initial))
            receipt.setdefault('initial_checks', []).append(record)
            print('[ADAPTER_INITIAL_OK]', json.dumps(record), flush=True)
            del model, restored
        receipt['action_normalization'] = stats
        receipt['status'] = 'evaluating_before'
        save()
        del original, loaded, target, actual
        gc.collect()
        torch.cuda.empty_cache()
        before = {task:evaluate(base_policy, stats_path, 'before', task) for task in datasets}
        receipt['before'] = before
        save()
        expected_updates = None
        # Exercise the installed trainer and real dataset before any long arm.
        # These discarded checks use separate names and never initialize the full runs.
        for arm,run,path,initial in prepared:
            check_cfg = OmegaConf.load(path)
            check_run = tag+'_check_'+arm
            with open_dict(check_cfg):
                check_cfg.output_model_name = check_cfg.subdir = check_run
                check_cfg.trainer.max_epochs = 1
                check_cfg.trainer.limit_train_batches = 2
                check_cfg.trainer.limit_val_batches = 1
                check_cfg.save_epoch_interval = 1
            check_path = output/(arm+'_check.yaml')
            OmegaConf.save(check_cfg, check_path)
            receipt['status'] = 'checking_'+arm
            save()
            invoke([sys.executable, '-u', str(root/'scripts/train/lewm.py'),
                '--config-path', str(output), '--config-name', check_path.stem], arm+'_check')
            check_dir = home/'checkpoints'/check_run
            check_trace = json.loads((check_dir/'learning_rate_trace.json').read_text())
            if check_trace['status'] != 'complete' or check_trace['global_step'] != 2:
                raise RuntimeError('Two-update real-data check did not complete')
            checked_state = torch.load(check_dir/'weights_epoch_1.pt', map_location='cpu', weights_only=True)
            initial_state = torch.load(initial, map_location='cpu', weights_only=True)
            if not any(v.is_floating_point() and not torch.equal(v,checked_state[k]) for k,v in initial_state.items()):
                raise RuntimeError('Real-data check did not update the model')
            receipt.setdefault('real_data_checks', []).append(dict(arm=arm, updates=2, changed=True))
            save()
            print('[ADAPTER_CHECK_OK]', arm, flush=True)
        for arm,run,path,initial in prepared:
            receipt['status'] = 'training_'+arm
            save()
            invoke([sys.executable, '-u', str(root/'scripts/train/lewm.py'),
                '--config-path', str(output), '--config-name', path.stem], arm+'_train')
            policy = run+f'/weights_epoch_{epochs}.pt'
            final = home/'checkpoints'/policy
            trace = json.loads((final.parent/'learning_rate_trace.json').read_text())
            if (not final.is_file() or trace['status'] != 'complete' or trace['epochs_completed'] != epochs
                    or trace['global_step'] < 1 or trace['scheduler_interval'] != 'step'
                    or not trace['schedulers'] or any(s['interval'] != 'step' for s in trace['schedulers'])):
                raise RuntimeError('Adapter training budget/scheduler check failed')
            if expected_updates is None:
                expected_updates = trace['global_step']
            if expected_updates != trace['global_step']:
                raise RuntimeError('Adapter arms completed different update counts')
            old_state = torch.load(initial, map_location='cpu', weights_only=True)
            new_state = torch.load(final, map_location='cpu', weights_only=True)
            changed = [k for k,v in old_state.items() if v.is_floating_point() and not torch.equal(v,new_state[k])]
            if not changed or (arm != 'finetune' and not any(k.startswith('predictor.adapter_') for k in changed)):
                raise RuntimeError('Expected predictor parameters did not change')
            if arm == 'gated' and not any(k.startswith('predictor.gate.') for k in changed):
                raise RuntimeError('The learned gate did not update')
            print('[ADAPTER_TRAIN_OK]', json.dumps(dict(arm=arm, global_step=trace['global_step'],
                  changed_tensors=len(changed))), flush=True)
            receipt['status'] = 'evaluating_'+arm
            save()
            behavior = {task:evaluate(policy, stats_path, arm, task) for task in datasets}
            row = dict(arm=arm, policy=policy, global_step=trace['global_step'], behavior=behavior,
                       change_from_before={t:adapter_pair(before[t]['successes'],behavior[t]['successes']) for t in datasets})
            receipt['rows'].append(row)
            save()
            print('[ADAPTER_ARM_DONE]', json.dumps(dict(arm=arm,
                success_rates={t:x['success_rate'] for t,x in behavior.items()})), flush=True)
        control = receipt['rows'][0]
        receipt['comparisons'] = [dict(arm=r['arm'], relative_to='finetune',
            tasks={t:adapter_pair(control['behavior'][t]['successes'],r['behavior'][t]['successes']) for t in datasets})
            for r in receipt['rows'][1:]]
        receipt['status'] = 'complete'
        save()
        print('[ADAPTER_DONE]', json.dumps(dict(summary=str(output/'summary.json'),
            before={t:x['success_rate'] for t,x in before.items()},
            rows=[dict(arm=r['arm'], success_rates={t:b['success_rate'] for t,b in r['behavior'].items()})
                  for r in receipt['rows']], comparisons=receipt['comparisons'])), flush=True)
    except Exception as exc:
        receipt.update(status='failed', error=str(exc))
        save()
        print('[ADAPTER_FAILED]', str(exc), flush=True)
        raise
