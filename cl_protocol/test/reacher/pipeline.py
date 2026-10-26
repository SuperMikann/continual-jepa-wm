"""Fixed-budget screening, T2 adaptation/comparison and T3 queues."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .common import CASES, emit, save, configuration, validate, inspect_collection, paired_behavior, prediction_windows
from .training import adaptation_config, comparison_training_config


def experiment_entrypoint(root):
    """Use an explicitly deployed entrypoint; never move or delete an upload at runtime."""
    target = root.resolve() / 'cl_protocol/test/run_reacher_experiments.py'
    if not target.is_file() or not (target.parent/'reacher/pipeline.py').is_file():
        raise FileNotFoundError('Deploy cl_protocol/test, including reacher/, before running the queue')
    return target


def batch(args):
    import fcntl
    home = Path(os.environ.get('STABLEWM_HOME', '/root/autodl-tmp/.stable_worldmodel'))
    checkpoint = home / 'checkpoints' / args.policy
    dataset = home / 'datasets' / args.dataset
    for path in (checkpoint, args.root / 'scripts/plan/eval_wm.py',
                 args.root / 'cl_protocol/drift_world.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (dataset / '_versions').is_dir():
        raise FileNotFoundError(f'Local Lance dataset missing; no download attempted: {dataset}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = experiment_entrypoint(args.root)
        summary_path = args.out_dir / 'summary.json'
        if summary_path.exists():
            previous = json.loads(summary_path.read_text(encoding='utf-8'))
            if previous['config'] != configuration(args):
                raise ValueError('Use a new output directory when changing the run configuration')
        report = dict(status='running', config=configuration(args), rows=[])
        baseline = None
        try:
            for case in args.cases:
                report['current_case'] = case
                save(summary_path, report)
                result_path = args.out_dir / f'{case}.json'
                if not result_path.exists():
                    emit('CASE_START', dict(case=case, condition=CASES[case]))
                    command = [sys.executable, '-u', str(target), '--root', str(args.root),
                               '--out-dir', str(args.out_dir), '--policy', args.policy,
                               '--dataset', args.dataset, '--num-eval', str(args.num_eval),
                               '--seed', str(args.seed), '--cases', *args.cases, '--case', case]
                    if args.scene_manifest is not None:
                        command += ['--scene-manifest', str(args.scene_manifest)]
                    with (args.out_dir / f'{case}.log').open('a', encoding='utf-8') as log:
                        log.write('\nSTART ' + datetime.now(timezone.utc).isoformat() + '\n')
                        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              text=True, encoding='utf-8', errors='replace', bufsize=1) as p:
                            for line in p.stdout:
                                log.write(line)
                                log.flush()
                                if line.startswith('[DYNAMICS_'):
                                    print(line, end='', flush=True)
                            code = p.wait()
                    if code != 0:
                        raise RuntimeError(f'{case} exited {code}; see {args.out_dir / (case + ".log")}')
                result = json.loads(result_path.read_text(encoding='utf-8'))
                validate(result, args, case)
                scenes = [(r['episode_idx'], r['start_step'], r['initial'])
                          for r in result['per_scene']]
                if baseline is None:
                    baseline = result
                    reference_scenes = scenes
                if scenes != reference_scenes or result['planning'] != baseline['planning']:
                    raise RuntimeError('Scenes, physical initial/goal states, or planning settings differ')
                losses = sum(a['success'] and not b['success'] for a, b in
                             zip(baseline['per_scene'], result['per_scene']))
                gains = sum(not a['success'] and b['success'] for a, b in
                            zip(baseline['per_scene'], result['per_scene']))
                row = dict(case=case, **result['summary'],
                           success_change_pp=result['summary']['success_rate'] - baseline['summary']['success_rate'],
                           lost_baseline_successes=losses, gained_successes=gains)
                report['rows'].append(row)
                save(summary_path, report)
                emit('CHECK_OK', row)
            report.update(status='complete', current_case=None, scenes_equal=True,
                          purpose='Frozen-checkpoint exploratory screening only; no adaptation or forgetting conclusion.')
            save(summary_path, report)
            emit('DONE', dict(summary=str(summary_path), completed=len(report['rows'])))
        except BaseException as exc:
            report.update(status='failed', error=str(exc))
            save(summary_path, report)
            emit('FAILED', str(exc))
            raise


def make_scene_manifest(dataset, count, seed, excluded):
    import lance
    import numpy as np
    ds = lance.dataset(str(dataset))
    table = ds.to_table(columns=['episode_idx', 'step_idx'])
    ep = np.asarray(table.column('episode_idx').to_numpy())
    step = np.asarray(table.column('step_idx').to_numpy())
    unique, inverse = np.unique(ep, return_inverse=True)
    maximum = np.zeros(len(unique), dtype=np.int64)
    np.maximum.at(maximum, inverse, step.astype(np.int64))
    valid = np.flatnonzero(step <= maximum[inverse]-25)
    valid = np.array([i for i in valid if (int(ep[i]), int(step[i])) not in excluded])
    if len(valid) < count:
        raise ValueError('Insufficient starts after excluding the screening scenes')
    chosen = np.sort(np.random.default_rng(seed).choice(valid, count, replace=False))
    return dict(seed=seed, source=str(dataset),
                excluded_screening_starts=[list(x) for x in sorted(excluded)],
                purpose='Fixed confirmation scenes, not the final method test set.',
                scenes=[dict(episode_idx=int(ep[i]), start_step=int(step[i])) for i in chosen])


def adaptation_batch(args):
    import fcntl
    import time
    import yaml
    home = Path(os.environ.get('STABLEWM_HOME', '/root/autodl-tmp/.stable_worldmodel'))
    os.environ.update(STABLEWM_HOME=str(home), MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa')
    sys.path.insert(0, str(args.root/'cl_protocol'))
    checkpoint = home/'checkpoints'/args.policy
    source_config = checkpoint.parent/'config.yaml'
    source_dataset = home/'datasets'/args.dataset
    screening = args.root/'results/reacher_dynamics_screen_20261022/baseline.json'
    for path in (checkpoint, source_config, screening, args.root/'scripts/train/lewm.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (source_dataset/'_versions').is_dir():
        raise FileNotFoundError(source_dataset)
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Use a simple run id without path components')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = experiment_entrypoint(args.root)
        manifest_path = args.out_dir/'scenes.json'
        screen = json.loads(screening.read_text(encoding='utf-8'))
        if (screen['status'] != 'complete' or screen['config']['policy'] != args.policy
                or screen['config']['dataset'] != args.dataset):
            raise ValueError('Use the same teacher and source dataset as the screening run')
        excluded = {(r['episode_idx'], r['start_step']) for r in screen['per_scene']}
        manifest = make_scene_manifest(source_dataset, args.num_eval, args.seed, excluded)
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Existing fixed scenes differ; use a new run directory')
        save(manifest_path, manifest)
        train_dataset = 'dmc/'+args.run_id+'_train.lance'
        pilot_dataset = 'dmc/'+args.run_id+'_capacity.lance'
        cfg = adaptation_config(yaml.safe_load(source_config.read_text(encoding='utf-8')),
                                args.run_id, train_dataset, checkpoint, home/'checkpoints', args.epochs)
        config_path = args.out_dir/'train.yaml'
        if config_path.exists() and yaml.safe_load(config_path.read_text(encoding='utf-8')) != cfg:
            raise ValueError('Training configuration changed; use a new run directory')
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        original_text = source_config.read_text(encoding='utf-8')
        (args.out_dir/'teacher_config.yaml').write_text(original_text, encoding='utf-8')
        run_config = dict(policy=args.policy, dataset=args.dataset, num_eval=args.num_eval,
                          seed=args.seed, epochs=args.epochs, training_episodes=args.train_episodes,
                          run_id=args.run_id)
        summary_path = args.out_dir/'summary.json'
        if summary_path.exists():
            state = json.loads(summary_path.read_text(encoding='utf-8'))
            if state['config'] != run_config:
                raise ValueError('Run configuration differs from the existing output')
        else:
            state = dict(config=run_config, steps=[], collection={}, behavior={})
        state['status'] = 'running'
        save(summary_path, state)

        def command(label, argv):
            state['current'] = label
            save(summary_path, state)
            emit('ADAPT_STEP_START', label)
            log_path = args.out_dir/(label+'.log')
            start = time.monotonic()
            with log_path.open('a', encoding='utf-8') as log:
                with subprocess.Popen([str(v) for v in argv], cwd=args.root,
                                      stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                      errors='replace', bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if '[DYNAMICS_' in line or '[EPOCH_DONE]' in line:
                            print(line, end='', flush=True)
                    code = process.wait()
            if code:
                tail = '\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-25:])
                raise RuntimeError(f'{label} exited {code}; {log_path}\n{tail}')
            state['steps'].append(dict(label=label, seconds=time.monotonic()-start))
            save(summary_path, state)

        def evaluate(label, policy):
            folder = args.out_dir/label
            command(label, [sys.executable, '-u', script, '--root', args.root,
                            '--policy', policy, '--dataset', args.dataset,
                            '--num-eval', args.num_eval, '--seed', args.seed,
                            '--out-dir', folder, '--scene-manifest', manifest_path,
                            '--cases', 'baseline', 'damping_2'])
            report = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            if report['status'] != 'complete' or not report['scenes_equal']:
                raise RuntimeError('Incomplete paired behavioral evaluation')
            state['behavior'][label] = report['rows']
            return [json.loads((folder/(case+'.json')).read_text(encoding='utf-8'))
                    for case in ('baseline', 'damping_2')]

        def collect(label, name, episodes, seed):
            path = home/'datasets'/name
            if not path.exists():
                collect_cfg = args.out_dir/(label+'.json')
                save(collect_cfg, dict(dataset=name, episodes=episodes, seed=seed))
                command(label, [sys.executable, '-u', script, '--mode', 'collect',
                                '--root', args.root, '--config', collect_cfg])
            info = inspect_collection(path, episodes)
            state['collection'][label] = info
            save(summary_path, state)
            emit('DATA_CHECK_OK', info)
            return path

        try:
            before = evaluate('before', args.policy)
            pilot = collect('capacity_pilot', pilot_dataset, 10, 202610220)
            size = sum(p.stat().st_size for p in pilot.rglob('*') if p.is_file())
            need = (0 if (home/'datasets'/train_dataset).exists()
                    else size*args.train_episodes/10*1.35) + 2**30
            free = shutil.disk_usage(home).free
            state['capacity'] = dict(pilot_bytes=size, required_bytes=need, free_bytes=free)
            if free < need:
                raise RuntimeError(f'Insufficient data disk: need {need/2**30:.2f} GiB, free {free/2**30:.2f} GiB')
            emit('CAPACITY_OK', state['capacity'])
            collect('training_data', train_dataset, args.train_episodes, 202610221)
            student = args.run_id+f'/weights_epoch_{args.epochs}.pt'
            student_path = home/'checkpoints'/student
            if not student_path.exists():
                command('training', [sys.executable, '-u', script, '--mode', 'train',
                                     '--root', args.root, '--config', config_path])
            import torch
            old = torch.load(checkpoint, map_location='cpu', weights_only=True)
            new = torch.load(student_path, map_location='cpu', weights_only=True)
            if new['predictor.adapter_A'].shape[0] != 1:
                raise RuntimeError('The ordinary adaptation run unexpectedly grew adapters')
            if not all(torch.isfinite(v).all() for v in new.values() if torch.is_tensor(v) and v.is_floating_point()):
                raise RuntimeError('Non-finite trained weights')
            changed = any(k in old and v.is_floating_point() and v.shape == old[k].shape
                          and not torch.equal(v, old[k]) for k, v in new.items() if torch.is_tensor(v))
            if not changed:
                raise RuntimeError('Checkpoint did not change during adaptation')
            state['training'] = dict(policy=student, adapters=1, changed=True)
            emit('TRAIN_CHECK_OK', state['training'])
            after = evaluate('after', student)
            signatures = [[(r['episode_idx'], r['start_step'], r['initial']) for r in cell['per_scene']]
                          for cell in before+after]
            if any(s != signatures[0] for s in signatures[1:]):
                raise RuntimeError('Before/after conditions do not share physical starts and goals')
            import numpy as np
            state['comparison'] = []
            rng = np.random.default_rng(0)
            for i, case in enumerate(('baseline', 'damping_2')):
                a = np.asarray([r['success'] for r in before[i]['per_scene']], dtype=int)
                b = np.asarray([r['success'] for r in after[i]['per_scene']], dtype=int)
                delta = b-a
                ci = np.percentile(delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)*100,
                                   [2.5, 97.5]).tolist()
                state['comparison'].append(dict(condition=case,
                    before=before[i]['summary']['success_rate'],
                    after=after[i]['summary']['success_rate'],
                    change_pp=float(delta.mean()*100), paired_bootstrap95_pp=ci,
                    lost_successes=int(((a == 1) & (b == 0)).sum()),
                    gained_successes=int(((a == 0) & (b == 1)).sum())))
            state.update(status='complete', current=None, scenes_equal=True,
                         scope='One ordinary continuation of the screened K=1 model; not a retention-method comparison.')
            save(summary_path, state)
            emit('ADAPT_DONE', dict(summary=str(summary_path), comparison=state['comparison']))
        except BaseException as exc:
            state.update(status='failed', error=str(exc))
            save(summary_path, state)
            emit('ADAPT_FAILED', str(exc))
            raise


def comparison_batch(args):
    import fcntl
    import time
    import yaml
    import torch
    home = Path(os.environ.get('STABLEWM_HOME', '/root/autodl-tmp/.stable_worldmodel'))
    os.environ.update(STABLEWM_HOME=str(home), MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa')
    base = args.base_run
    accepted = json.loads((base/'summary.json').read_text(encoding='utf-8'))
    if accepted['status'] != 'complete' or not accepted['scenes_equal']:
        raise RuntimeError('The ordinary adaptation run must finish first')
    if (accepted['config']['num_eval'] != args.num_eval or accepted['config']['seed'] != args.seed
            or accepted['config']['epochs'] != args.epochs or accepted['config']['policy'] != args.policy
            or accepted['config']['dataset'] != args.dataset):
        raise ValueError('Keep the accepted continuation configuration for this comparison')
    base_cfg = yaml.safe_load((base/'train.yaml').read_text(encoding='utf-8'))
    if (base_cfg['model']['predictor']['num_adapters'] != 1
            or not base_cfg['loader']['drop_last'] or not base_cfg['loader']['shuffle']):
        raise ValueError('Expected the accepted single-adapter training configuration')
    manifest = base/'scenes.json'
    checkpoint = home/'checkpoints'/args.policy
    if (Path(base_cfg['warm_start']).resolve() != checkpoint.resolve()
            or base_cfg['trainer']['max_epochs'] != args.epochs):
        raise ValueError('Method controls must share the accepted T1 start and epoch budget')
    train_path = home/'datasets'/base_cfg['data']['dataset']['name']
    for path in (checkpoint, manifest, args.root/'scripts/train/lewm.py',
                 args.root/'scripts/plan/eval_wm.py', args.root/'cl_protocol/drift_world.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (home/'datasets'/args.dataset/'_versions').is_dir():
        raise FileNotFoundError('The original local T1 dataset is required; no download is attempted')
    inspect_collection(train_path, accepted['config']['training_episodes'])
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Use a run id without path components')

    def cells(folder):
        result = [json.loads((folder/(case+'.json')).read_text(encoding='utf-8'))
                  for case in ('baseline', 'damping_2')]
        for cell, case in zip(result, ('baseline', 'damping_2')):
            if (cell['status'] != 'complete' or cell['condition'] != CASES[case]
                    or len(cell['per_scene']) != args.num_eval):
                raise RuntimeError(f'Incomplete behavioral cell: {folder}/{case}')
            if abs(cell['summary']['success_rate']-100*sum(r['success'] for r in cell['per_scene'])/args.num_eval) > 1e-7:
                raise RuntimeError('Per-scene success flags differ from the reported rate')
        return result

    teacher = cells(base/'before')
    finetune = cells(base/'after')
    paired_behavior(teacher, finetune)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = experiment_entrypoint(args.root)
        config = dict(base_run=str(base), policy=args.policy, seed=args.seed, num_eval=args.num_eval,
                      epochs=args.epochs, methods=args.methods, run_id=args.run_id,
                      training_dataset=str(train_path), replay_memory_windows=4096,
                      nominal_replay_fraction=0.25)
        summary_path = args.out_dir/'summary.json'
        state = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else dict(steps=[], methods={})
        if state.get('config', config) != config:
            raise ValueError('Existing comparison configuration differs; use a new run directory')
        state.update(config=config, status='running')
        state.pop('error', None)
        state['rows'] = [dict(method=name, policy=policy,
                              old_success=group[0]['summary']['success_rate'],
                              new_success=group[1]['summary']['success_rate'])
                         for name, policy, group in (('T1', args.policy, teacher),
                             ('finetune', accepted['training']['policy'], finetune))]
        save(summary_path, state)

        def invoke(label, command):
            state['current'] = label
            save(summary_path, state)
            emit('COMPARE_STEP_START', label)
            log_path = args.out_dir/(label+'.log')
            start = time.monotonic()
            with log_path.open('a', encoding='utf-8') as log:
                with subprocess.Popen([str(x) for x in command], cwd=args.root,
                                      stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                      errors='replace', bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if '[DYNAMICS_' in line or '[EPOCH_DONE]' in line:
                            print(line, end='', flush=True)
                    code = process.wait()
            if code:
                tail = '\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-25:])
                raise RuntimeError(f'{label} exited {code}; {log_path}\n{tail}')
            state['steps'].append(dict(label=label, seconds=time.monotonic()-start))
            save(summary_path, state)

        try:
            old = torch.load(checkpoint, map_location='cpu', weights_only=True)
            for method in args.methods:
                folder = args.out_dir/method
                receipt_path = folder/'training_receipt.json'
                run_id = args.run_id+'_'+method
                cfg = comparison_training_config(base_cfg, run_id, method, args.dataset,
                                                 receipt_path, home/'checkpoints')
                config_path = folder/'train.yaml'
                folder.mkdir(parents=True, exist_ok=True)
                if config_path.exists() and yaml.safe_load(config_path.read_text(encoding='utf-8')) != cfg:
                    raise ValueError('Saved method training configuration changed')
                config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
                policy = run_id+f'/weights_epoch_{args.epochs}.pt'
                weights = home/'checkpoints'/policy
                if not weights.exists():
                    if shutil.disk_usage(home).free < 768*2**20:
                        raise RuntimeError('Less than 768 MiB free for checkpoints; comparison stopped without deleting data')
                    invoke(method+'_train', [sys.executable, '-u', script, '--mode', 'train',
                                            '--root', args.root, '--config', config_path])
                receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                if (receipt['status'] != 'complete' or receipt['method'] != method
                        or receipt['epochs'] != args.epochs
                        or receipt['global_step'] != receipt['expected_updates']):
                    raise RuntimeError('Missing or mismatched optimizer-budget receipt')
                if method == 'replay' and not 0.20 < receipt['actual_old_fraction'] < 0.30:
                    raise RuntimeError('The actual replay fraction differs materially from 25%')
                new = torch.load(weights, map_location='cpu', weights_only=True)
                expected_k = 2 if method == 'growth' else 1
                if new['predictor.adapter_A'].shape[0] != expected_k:
                    raise RuntimeError('Unexpected adapter count')
                if not all(torch.isfinite(v).all() for v in new.values() if torch.is_tensor(v) and v.is_floating_point()):
                    raise RuntimeError('Non-finite method checkpoint')
                if not any(k in old and v.is_floating_point() and v.shape == old[k].shape
                           and not torch.equal(v, old[k]) for k, v in new.items() if torch.is_tensor(v)):
                    raise RuntimeError('The method checkpoint did not change')
                if method == 'growth':
                    for key in ('predictor.adapter_A', 'predictor.adapter_B'):
                        if not torch.equal(new[key][:1], old[key]):
                            raise RuntimeError('Old adapter slots changed despite the optimizer guard')
                    expected_scale = old.get('predictor.adapter_scale', torch.tensor(1.0))*2
                    if not torch.allclose(new['predictor.adapter_scale'], expected_scale):
                        raise RuntimeError('Growth did not preserve the initial old-adapter mixture scale')
                emit('METHOD_TRAIN_CHECK_OK', dict(method=method, adapters=expected_k,
                                                  global_step=receipt['global_step'],
                                                  actual_old_fraction=receipt['actual_old_fraction']))
                behavior = folder/'behavior'
                invoke(method+'_eval', [sys.executable, '-u', script, '--root', args.root,
                                       '--policy', policy, '--dataset', args.dataset,
                                       '--num-eval', args.num_eval, '--seed', args.seed,
                                       '--out-dir', behavior, '--scene-manifest', manifest,
                                       '--cases', 'baseline', 'damping_2'])
                result = cells(behavior)
                changes = dict(vs_T1=paired_behavior(teacher, result),
                               vs_finetune=paired_behavior(finetune, result))
                row = dict(method=method, policy=policy,
                           old_success=result[0]['summary']['success_rate'],
                           new_success=result[1]['summary']['success_rate'])
                state['methods'][method] = dict(training=receipt, adapters=expected_k,
                                                old_slots_unchanged=True if method == 'growth' else None, **changes)
                state['rows'].append(row)
                save(summary_path, state)
                emit('METHOD_DONE', row)
            state.update(status='complete', current=None, scenes_equal=True,
                         scope='One-seed matched-update replay and repaired-growth comparison. Parameter and old-memory budgets differ by method.')
            save(summary_path, state)
            emit('COMPARE_DONE', dict(summary=str(summary_path), rows=state['rows']))
        except BaseException as exc:
            state.update(status='failed', error=str(exc))
            save(summary_path, state)
            emit('COMPARE_FAILED', str(exc))
            raise


def stage3_batch(args):
    """Fixed third condition; each method continues its own accepted T2 model."""
    import fcntl
    import time
    import yaml
    import torch
    home = Path(os.environ.get('STABLEWM_HOME', '/root/autodl-tmp/.stable_worldmodel'))
    os.environ.update(STABLEWM_HOME=str(home), MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa')
    base, controls = args.base_run, args.comparison_run
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    accepted, compared = read(base/'summary.json'), read(controls/'summary.json')
    if any(d['status'] != 'complete' or not d['scenes_equal'] for d in (accepted, compared)):
        raise ValueError('Both accepted T2 queues must be complete')
    for key in ('policy', 'dataset', 'num_eval', 'seed', 'epochs'):
        if accepted['config'][key] != getattr(args, key):
            raise ValueError(f'Keep the accepted T2 protocol: {key}')
    if args.train_episodes != accepted['config']['training_episodes'] or args.methods != ['replay', 'growth']:
        raise ValueError('Stage3 uses all three methods and the same new-data budget as T2')
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Use a simple run id without path components')
    base_cfg = yaml.safe_load((base/'train.yaml').read_text(encoding='utf-8'))
    t2_dataset = base_cfg['data']['dataset']['name']
    for name in (args.dataset, t2_dataset):
        if not (home/'datasets'/name/'_versions').is_dir():
            raise FileNotFoundError('Local training dataset missing; no download attempted: '+name)
    inspect_collection(home/'datasets'/t2_dataset, args.train_episodes)
    cases = ('baseline', 'damping_2', 'damping_0p5')
    manifest = base/'scenes.json'
    t2_policies = {r['method']: r['policy'] for r in compared['rows'] if r['method'] != 'T1'}
    if set(t2_policies) != {'finetune', 'replay', 'growth'}:
        raise ValueError('Missing accepted T2 method policies')
    if t2_policies['finetune'] != accepted['training']['policy']:
        raise ValueError('T2 finetune provenance differs')
    sources = {'T1': (args.policy, base/'before')}
    sources.update({f'T2_{method}': (policy, base/'after' if method == 'finetune'
                                   else controls/method/'behavior')
                    for method, policy in t2_policies.items()})
    for policy, _ in sources.values():
        if not (home/'checkpoints'/policy).is_file():
            raise FileNotFoundError(policy)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = experiment_entrypoint(args.root)
        summary_path = args.out_dir/'summary.json'
        train_name = 'dmc/'+args.run_id+'_train.lance'
        config = dict(base_run=str(base), comparison_run=str(controls), policy=args.policy,
                      dataset=args.dataset, num_eval=args.num_eval, seed=args.seed,
                      epochs=args.epochs, training_episodes=args.train_episodes, run_id=args.run_id,
                      conditions=list(cases), t3_damping=0.005, training_seed=base_cfg['seed'],
                      collection_seed=202610231, replay_memory=4096, replay_fraction=0.25,
                      t3_min_success=60, t3_min_gain_pp=20,
                      prediction_episodes=args.pred_episodes, prediction_windows=args.pred_windows,
                      prediction_batch=args.pred_batch)
        state = read(summary_path) if summary_path.exists() else dict(steps=[], cells={}, training={}, prediction={})
        if state.get('config', config) != config:
            raise ValueError('Stage3 configuration changed; use a new output directory')
        state.update(config=config, status='running')
        state.pop('error', None)
        save(summary_path, state)

        def invoke(label, argv):
            state['current'] = label
            save(summary_path, state)
            emit('T3_STEP_START', label)
            log_path = args.out_dir/(label+'.log')
            start = time.monotonic()
            with log_path.open('a', encoding='utf-8') as log:
                with subprocess.Popen([str(v) for v in argv], cwd=args.root, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      encoding='utf-8', errors='replace', bufsize=1) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if '[DYNAMICS_' in line or '[EPOCH_DONE]' in line:
                            print(line, end='', flush=True)
                    code = process.wait()
            if code:
                tail = '\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-20:])
                raise RuntimeError(f'{label} exited {code}; {log_path}\n{tail}')
            state['steps'].append(dict(label=label, seconds=time.monotonic()-start))
            save(summary_path, state)

        def collect(label, name, episodes, seed, case):
            path = home/'datasets'/name
            if not path.exists():
                cfg_path = args.out_dir/(label+'.json')
                save(cfg_path, dict(dataset=name, episodes=episodes, seed=seed, case=case))
                invoke(label, [sys.executable, '-u', script, '--mode', 'collect',
                               '--root', args.root, '--config', cfg_path])
            info = inspect_collection(path, episodes, case)
            emit('DATA_CHECK_OK', dict(case=case, **info))
            return info

        reference = read(base/'before/baseline.json')

        def cell(key, policy, case, existing=None):
            folder = args.out_dir/'behavior'/key
            folder.mkdir(parents=True, exist_ok=True)
            path = existing if existing is not None else folder/(case+'.json')
            if existing is not None and not path.is_file():
                raise FileNotFoundError('Accepted T2 result is missing: '+str(path))
            if not path.is_file():
                invoke(key+'_'+case, [sys.executable, '-u', script, '--root', args.root,
                       '--out-dir', folder, '--policy', policy, '--dataset', args.dataset,
                       '--num-eval', args.num_eval, '--seed', args.seed,
                       '--scene-manifest', manifest, '--cases', *cases, '--case', case])
            result = read(path)
            if (result['status'] != 'complete' or result['case'] != case
                    or result['condition'] != CASES[case]
                    or any(result['config'][k] != v for k, v in dict(
                        policy=policy, dataset=args.dataset, num_eval=args.num_eval, seed=args.seed).items())):
                raise ValueError(f'Mismatched behavioral result: {path}')
            paired_behavior([reference], [result], (case,))
            if len(result['per_scene']) != args.num_eval or abs(
                    result['summary']['success_rate']-100*sum(r['success'] for r in result['per_scene'])/args.num_eval) > 1e-7:
                raise ValueError('Incomplete or inconsistent success counts')
            state['cells'].setdefault(key, {})[case] = dict(path=str(path), policy=policy,
                                                           success_rate=result['summary']['success_rate'])
            save(summary_path, state)
            emit('T3_CELL_OK', dict(model=key, case=case, success_rate=result['summary']['success_rate']))
            return result

        try:
            size = sum(p.stat().st_size for p in (home/'datasets'/t2_dataset).rglob('*') if p.is_file())
            new_sets = [(train_name, args.train_episodes)] + [
                ('dmc/'+args.run_id+'_eval_'+case+'.lance', args.pred_episodes) for case in cases]
            needed = sum(size*n/args.train_episodes*1.35 for name, n in new_sets
                         if not (home/'datasets'/name).exists())+768*2**20
            free = shutil.disk_usage(home).free
            state['capacity'] = dict(reference_dataset=t2_dataset, reference_bytes=size,
                                     required_bytes=needed, free_bytes=free)
            save(summary_path, state)
            if free < needed:
                raise RuntimeError(f'Stage3 needs {needed/2**30:.2f} GiB, free {free/2**30:.2f} GiB; no data deleted')
            emit('CAPACITY_OK', state['capacity'])
            before = {}
            for key, (policy, old_folder) in sources.items():
                before[key] = [cell(key, policy, case, old_folder/(case+'.json') if i < 2 else None)
                               for i, case in enumerate(cases)]
            state['collection'] = collect('collect_T3', train_name, args.train_episodes, 202610231, cases[2])
            t3_results, all_policies = {}, {key: policy for key, (policy, _) in sources.items()}
            for method in ('finetune', 'replay', 'growth'):
                folder = args.out_dir/method
                folder.mkdir(parents=True, exist_ok=True)
                receipt_path = folder/'training_receipt.json'
                method_base = copy.deepcopy(base_cfg)
                method_base['warm_start'] = str(home/'checkpoints'/t2_policies[method])
                method_base['model']['predictor']['num_adapters'] = 2 if method == 'growth' else 1
                start_weights = torch.load(method_base['warm_start'], map_location='cpu', weights_only=True)
                if start_weights['predictor.adapter_A'].shape[0] != method_base['model']['predictor']['num_adapters']:
                    raise ValueError('Accepted T2 checkpoint has an unexpected adapter count')
                del start_weights
                method_base['data']['dataset']['name'] = train_name
                cfg = comparison_training_config(method_base, args.run_id+'_'+method, method,
                                                 args.dataset, receipt_path, home/'checkpoints')
                if method == 'replay':
                    cfg['comparison']['old_datasets'] = [args.dataset, t2_dataset]
                cfg_path = folder/'train.yaml'
                if cfg_path.exists() and yaml.safe_load(cfg_path.read_text(encoding='utf-8')) != cfg:
                    raise ValueError('Saved T3 training configuration differs')
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
                policy = cfg['output_model_name']+f'/weights_epoch_{args.epochs}.pt'
                weights = home/'checkpoints'/policy
                if not weights.is_file():
                    if shutil.disk_usage(home).free < 768*2**20:
                        raise RuntimeError('Less than 768 MiB free for training; no data deleted')
                    invoke(method+'_train', [sys.executable, '-u', script, '--mode', 'train',
                                            '--root', args.root, '--config', cfg_path])
                receipt = read(receipt_path)
                expected = compared['methods']['replay']['training']['expected_updates']
                if (receipt['status'] != 'complete' or receipt['method'] != method
                        or receipt['epochs'] != args.epochs or receipt['batch_size'] != cfg['loader']['batch_size']
                        or receipt['global_step'] != expected or receipt['expected_updates'] != expected
                        or receipt['old_samples']+receipt['new_samples'] != expected*receipt['batch_size']):
                    raise ValueError('T3 training budget receipt does not match T2')
                if method == 'replay':
                    if (receipt.get('memory_windows_by_task') != [2048, 2048]
                            or not 0.20 < receipt['actual_old_fraction'] < 0.30
                            or set(receipt['old_samples_by_task']) != {'0', '1'}
                            or sum(receipt['old_samples_by_task'].values()) != receipt['old_samples']
                            or min(receipt['old_samples_by_task'].values()) <= 0):
                        raise ValueError('T3 replay must include both past conditions within 4096 windows')
                elif receipt['old_samples']:
                    raise ValueError('Unexpected old data in a non-replay method')
                old = torch.load(method_base['warm_start'], map_location='cpu', weights_only=True)
                new = torch.load(weights, map_location='cpu', weights_only=True)
                old_k = old['predictor.adapter_A'].shape[0]
                if new['predictor.adapter_A'].shape[0] != (old_k+1 if method == 'growth' else old_k):
                    raise ValueError('Incorrect T3 adapter count')
                if not all(torch.isfinite(v).all() for v in new.values() if torch.is_tensor(v) and v.is_floating_point()):
                    raise ValueError('Non-finite T3 weights')
                if not any(k in old and v.is_floating_point() and v.shape == old[k].shape
                           and not torch.equal(v, old[k]) for k, v in new.items() if torch.is_tensor(v)):
                    raise ValueError('T3 weights did not change')
                if method == 'growth':
                    if old_k != 2 or any(not torch.equal(new[k][:old_k], old[k]) for k in
                                         ('predictor.adapter_A', 'predictor.adapter_B')):
                        raise ValueError('T3 growth changed one of the two old adapter slots')
                    if not torch.allclose(new['predictor.adapter_scale'], old['predictor.adapter_scale']*1.5):
                        raise ValueError('Incorrect K=2 to K=3 mixture compensation')
                state['training'][method] = dict(receipt=receipt, policy=policy,
                                                warm_start=t2_policies[method],
                                                adapters=int(new['predictor.adapter_A'].shape[0]),
                                                old_slots_unchanged=True if method == 'growth' else None)
                del old, new
                emit('T3_TRAIN_CHECK_OK', dict(method=method, global_step=expected, policy=policy))
                key = 'T3_'+method
                all_policies[key] = policy
                t3_results[method] = [cell(key, policy, case) for case in cases]
                state.setdefault('paired_changes', {})[method] = paired_behavior(
                    before['T2_'+method], t3_results[method], cases)
                if method == 'finetune':
                    after = t3_results[method][2]['summary']['success_rate']
                    gain = after-before['T2_finetune'][2]['summary']['success_rate']
                    passed = after >= config['t3_min_success'] and gain >= config['t3_min_gain_pp']
                    state['qualification'] = dict(passed=passed, after_success=after, gain_pp=gain,
                                                  rule='T3 finetune success >=60% and gain over its T2 model >=20pp; fixed before this run')
                    save(summary_path, state)
                    if not passed:
                        state['status'] = 'qualification_failed'
                        emit('T3_QUALIFICATION_STOP', state['qualification'])
                        raise RuntimeError('T3 qualification failed; results retained, remaining methods not started')
                    emit('T3_QUALIFIED', state['qualification'])
                save(summary_path, state)
            state['paired_vs_finetune'] = {method: paired_behavior(t3_results['finetune'], t3_results[method], cases)
                                          for method in ('replay', 'growth')}
            state['control_matrices'] = {method: {case: [before['T1'][i]['summary']['success_rate'],
                before['T2_'+method][i]['summary']['success_rate'], t3_results[method][i]['summary']['success_rate']]
                for i, case in enumerate(cases)} for method in ('finetune', 'replay', 'growth')}
            state['matrix_layout'] = 'conditions in rows; T1/T2/T3 trained stages in columns'
            state['scenes_equal'] = True
            save(summary_path, state)
            emit('T3_BEHAVIOR_DONE', state['control_matrices'])
            test_names, window_paths = {}, {}
            for i, case in enumerate(cases):
                name = 'dmc/'+args.run_id+'_eval_'+case+'.lance'
                collect('collect_eval_'+case, name, args.pred_episodes, 202610240+i, case)
                path = args.out_dir/('windows_'+case+'.json')
                windows = prediction_windows(home/'datasets'/name, args.pred_windows, 20261025)
                if path.exists() and read(path) != windows:
                    raise ValueError('Fixed prediction windows changed')
                save(path, windows)
                test_names[case], window_paths[case] = name, str(path)
            for key, policy in all_policies.items():
                folder = args.out_dir/'prediction'
                folder.mkdir(exist_ok=True)
                output, cfg_path = folder/(key+'.json'), folder/(key+'_config.json')
                cfg = dict(policy=policy, scaler_dataset=args.dataset, datasets=test_names,
                           windows=window_paths, batch=args.pred_batch, output=str(output))
                if cfg_path.exists() and read(cfg_path) != cfg:
                    raise ValueError('Saved prediction configuration differs')
                save(cfg_path, cfg)
                if not output.exists():
                    invoke('prediction_'+key, [sys.executable, '-u', script, '--mode', 'prediction',
                                              '--root', args.root, '--config', cfg_path])
                result = read(output)
                if (result['status'] != 'complete' or result['config'] != cfg
                        or set(result['results']) != set(cases)
                        or any(r['n_windows'] != args.pred_windows for r in result['results'].values())):
                    raise ValueError('Incomplete prediction matrix row')
                state['prediction'][key] = {case: {k: v for k, v in row.items() if k != 'per_window'}
                                            for case, row in result['results'].items()}
                save(summary_path, state)
            state['prediction_matrices'] = {method: {case: [state['prediction'][key][case]['mse_mean']
                for key in ('T1', 'T2_'+method, 'T3_'+method)] for case in cases}
                for method in ('finetune', 'replay', 'growth')}
            state.update(status='complete', current=None,
                         scope='Single training seed; confirmation controls, separate prediction rollouts. Own-encoder latent MSE includes representation/scale changes; target variance also reported.')
            save(summary_path, state)
            emit('T3_DONE', dict(summary=str(summary_path), control=state['control_matrices']))
        except BaseException as exc:
            if state['status'] != 'qualification_failed':
                state['status'] = 'failed'
            state.update(error=str(exc))
            save(summary_path, state)
            emit('T3_FAILED', dict(status=state['status'], error=str(exc)))
            raise
