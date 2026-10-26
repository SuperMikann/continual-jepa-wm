"""Paired repair check: fresh shared training data, legacy vs repaired growth.

--prepare installs uploaded files, checks prerequisites, runs real GPU growth
validation and a 10-episode capacity pilot. --run performs the finite queue.
--mode fixed waits for a completed paired run, then adds its fixed K=4 control.
No old data/weights are deleted. Test datasets are never used for training.
"""
import argparse
import copy
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import yaml

GIB = 2**30
RUN = 'reacher_repair_pair_20261021'
HOME = Path('/root/autodl-tmp/.stable_worldmodel')
TEACHER = 'm3g_reacher_t1/weights_epoch_30.pt'
ARMS = ('legacy', 'repaired')
FIXED_TEACHER = 'm3_reacher_t1/weights_epoch_30.pt'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def emit(tag, value):
    print(f'[{tag}]', value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), flush=True)


def build_config(source, arm, stage, run_id, dataset, checkpoint_home):
    cfg = copy.deepcopy(source)
    assert cfg['trainer']['max_epochs'] == 10, 'Expected the archived 10-epoch configuration'
    assert cfg['seed'] == 3072, 'Saved training seed differs from 3072'
    assert cfg['model']['predictor']['freeze_gate'] is True, 'Expected uniform routing'
    assert cfg['model']['predictor']['_target_'] == 'stable_worldmodel.wm.lewm.module.AdapterPredictor', 'Unexpected predictor class in saved config'
    assert cfg['optimizer'] == dict(type='AdamW', lr=5e-5, weight_decay=0.001)
    name = f'{run_id}_{arm}_t{stage}'
    previous = TEACHER if stage == 2 else f'{run_id}_{arm}_t2/weights_epoch_10.pt'
    cfg.update(output_model_name=name, subdir=name,
               warm_start=str(checkpoint_home/previous),
               freeze_old_adapters=stage-1, save_epoch_interval=10)
    cfg['trainer']['default_root_dir'] = str(checkpoint_home/name/'trainer')
    cfg['model']['predictor']['num_adapters'] = stage
    cfg['data']['dataset']['name'] = dataset
    if 'wandb' in cfg:
        cfg['wandb']['enabled'] = False
    return cfg


def build_fixed_config(source, stage, run_id, dataset, checkpoint_home):
    """Reuse the archived fixed-capacity setup on the newly collected data."""
    assert source['model']['predictor']['num_adapters'] == 4, 'Expected archived fixed K=4 model'
    cfg = build_config(source, 'fixed', stage, run_id, dataset, checkpoint_home)
    cfg['model']['predictor']['num_adapters'] = 4
    cfg['freeze_old_adapters'] = 0
    previous = FIXED_TEACHER if stage == 2 else f'{run_id}_fixed_t2/weights_epoch_10.pt'
    cfg['warm_start'] = str(checkpoint_home/previous)
    return cfg


def dataset_info(path, expected_episodes=None, expected_variations=None):
    import lance
    import numpy as np
    ds = lance.dataset(str(path))
    names = ds.schema.names
    required = {'pixels', 'action', 'observation', 'qpos', 'qvel', 'episode_idx', 'step_idx'}
    if not required <= set(names):
        raise RuntimeError(f'{path}: missing columns {required-set(names)}')
    table = ds.to_table(columns=['episode_idx', 'step_idx', 'qpos', 'qvel'], filter='step_idx = 0')
    rows = table.to_pylist()
    ids = {int(row['episode_idx']) for row in rows}
    if expected_episodes is not None and len(ids) != expected_episodes:
        raise RuntimeError(f'{path}: {len(ids)} episodes != {expected_episodes}')
    starts = {np.asarray(row['qpos']+row['qvel'], dtype=np.float64).tobytes() for row in rows}
    for key, value in (expected_variations or {}).items():
        candidates = ['variation.'+key, ('variation.'+key).replace('.', '_')]
        column = next((name for name in candidates if name in names), None)
        if column is None:
            raise RuntimeError(f'{path}: no recorded variation column for {key}')
        values = ds.to_table(columns=[column]).column(0).to_pylist()
        if not np.allclose(np.asarray(values).reshape(-1), float(value[0])):
            raise RuntimeError(f'{path}: recorded {key} does not match task')
    return dict(path=str(path), rows=ds.count_rows(), episodes=len(ids)), starts


def worker(args):
    # Both arms receive the same explicit global seed and DataLoader seed.
    import importlib
    import random
    import runpy
    import types
    import numpy as np
    import torch
    import stable_worldmodel
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.arm == 'legacy':
        reference = types.ModuleType('legacy_growth_reference')
        sys.modules[reference.__name__] = reference
        source = (args.config.parent.parent/'reference/module.py').read_text(encoding='utf-8')
        exec(compile(source, '<exported-server-module>', 'exec'), reference.__dict__)
        native = importlib.import_module('stable_worldmodel.wm.lewm.module')
        native.AdapterPredictor = reference.AdapterPredictor
        entry = args.config.parent.parent/'reference/train_lewm.py'
    else:
        entry = args.root/'scripts/train/lewm.py'
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    emit('WORKER', dict(arm=args.arm, seed=seed, source=str(entry)))
    sys.argv = [str(entry), '--config-path', str(args.config.parent),
                '--config-name', args.config.stem, 'hydra.job.chdir=false',
                'hydra.run.dir='+str(args.config.parent.parent/'hydra'/args.config.stem)]
    runpy.run_path(str(entry), run_name='__main__')


class Batch:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve()
        self.out = self.root/'results'/args.run_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.summary = self.out/'summary.json'
        self.state = json.loads(self.summary.read_text()) if self.summary.exists() else {
            'run_id': args.run_id, 'status': 'preparing', 'training': {},
            'behavior': {}, 'prediction': {}, 'steps': []}
        self.env = dict(os.environ, STABLEWM_HOME=str(HOME), MUJOCO_GL='osmesa',
                        PYOPENGL_PLATFORM='osmesa', PYTHONUNBUFFERED='1',
                        LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
        self.env.pop('LOCAL_DATASET_DIR', None)
        os.environ.update({k:self.env[k] for k in ('STABLEWM_HOME','MUJOCO_GL','PYOPENGL_PLATFORM','LD_PRELOAD')})

    def store(self):
        save(self.summary, self.state)

    def command(self, label, argv, cwd=None):
        self.state['current'] = label
        self.store()
        emit('STEP_START', label)
        log = self.out/'logs'/(label+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log.open('w', encoding='utf-8') as f, subprocess.Popen(
                [str(x) for x in argv], cwd=cwd or self.root, env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                f.write(line)
                if any(key in line for key in ('[EPOCH_DONE]', '[FROZEN_OPTIMIZER_BOUND]', 'CELL_RESULT ', '[完成]', '[GROWTH_CHECK_DONE]')):
                    f.flush()
                    emit('PROGRESS', label+' '+line.strip())
            code = process.wait()
        if code:
            tail = log.read_text(encoding='utf-8', errors='replace').splitlines()[-35:]
            raise RuntimeError(f'{label} exited {code}; log={log}\n'+'\n'.join(tail))
        elapsed = time.monotonic()-started
        self.state['steps'].append(dict(label=label, seconds=elapsed, log=str(log)))
        self.store()
        emit('STEP_OK', dict(label=label, seconds=round(elapsed, 1)))
        return log

    def collect(self, task, count, seed):
        self.command('collect_'+task, [sys.executable, self.root/'cl_protocol/collect_task.py',
            '--tasks', self.out/'tasks.yaml', '--sequence', 'B_reacher', '--task', task,
            '--episodes', count, '--seed', seed, '--num-envs', 10])

    def prepare(self):
        checkpoint_home = HOME/'checkpoints'
        for arm in ARMS:
            for stage in (2,3):
                if (checkpoint_home/f'{self.args.run_id}_{arm}_t{stage}').exists():
                    raise RuntimeError('New run directory already exists; preserve it and use another --run-id')
        uploads = self.args.upload
        files = {'module.py': None, 'train_lewm.py': self.root/'scripts/train/lewm.py',
                 'growth_check.py': self.root/'cl_protocol/test/growth_check.py',
                 'eval_task.py': self.root/'cl_protocol/eval_task.py'}
        required = [uploads/name for name in files]
        required += [checkpoint_home/TEACHER, self.root/'cl_protocol/collect_task.py',
                     self.root/'cl_protocol/pred_error.py', self.root/'cl_protocol/tasks.yaml']
        required += [checkpoint_home/f'm3g_reacher_t{s}/config.yaml' for s in (2,3)]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError('MISSING: '+', '.join(missing))
        # Do not replace shared sources while another training/evaluation is active.
        for process in Path('/proc').glob('[0-9]*'):
            if process.name == str(os.getpid()):
                continue
            try:
                cmd = (process/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            except (OSError, PermissionError):
                continue
            if any(key in cmd for key in ('scripts/train/lewm.py', 'cl_protocol/eval_task.py', 'scripts/plan/eval_wm.py')):
                raise RuntimeError('Another training/evaluation is active: '+cmd[:250])
        original = yaml.safe_load((self.root/'cl_protocol/tasks.yaml').read_text(encoding='utf-8'))
        tasks = copy.deepcopy(original)
        seq = tasks['sequences']['B_reacher']
        assert seq['max_episode_steps'] == 200 and seq['collect_policy'] == 'random'
        expected = {'T1':{}, 'T2':{'agent.arm_density':[700.0]},
                    'T3':{'agent.arm_density':[500.0], 'agent.finger_density':[700.0]}}
        all_tasks = {**seq['tasks'], **seq['probe_tasks']}
        for task in ('T1','T2','T3','T2_test','T3_test'):
            ref = task.replace('_test','')
            assert (all_tasks[task].get('variation_values') or {}) == expected[ref]
            assert all_tasks[task].get('darken',1.0) == 1.0
        # Verify preserved evaluation data before any collection/training.
        for task in ('T1','T2_test','T3_test'):
            dataset_info(HOME/'datasets'/all_tasks[task]['dataset'])
        for stage in (2,3):
            seq['tasks'][f'T{stage}']['dataset'] = f'dmc/reacher_t{stage}_{self.args.run_id}.lance'
            if (HOME/'datasets'/seq['tasks'][f'T{stage}']['dataset']).exists():
                raise RuntimeError('New training dataset already exists; choose a new run-id')
        pilot = copy.deepcopy(seq['tasks']['T2'])
        pilot['dataset'] = f'dmc/reacher_capacity_{self.args.run_id}.lance'
        pilot['collect_episodes'] = 10
        seq['probe_tasks']['capacity_pilot'] = pilot
        (self.out/'tasks.yaml').write_text(yaml.safe_dump(tasks, allow_unicode=True, sort_keys=False), encoding='utf-8')
        (self.out/'original_tasks.yaml').write_text(yaml.safe_dump(original, allow_unicode=True, sort_keys=False), encoding='utf-8')
        exports = [uploads/'implementation_audit_server_20261021.json', self.root/'results/implementation_audit_server_20261021.json']
        evidence = next((p for p in exports if p.is_file()), None)
        if evidence is None:
            raise FileNotFoundError('MISSING: implementation_audit_server_20261021.json')
        old = json.loads(evidence.read_text(encoding='utf-8-sig'))['sources']
        ref = self.out/'reference'
        ref.mkdir(exist_ok=True)
        (ref/'module.py').write_text(old['package/wm/lewm/module.py'], encoding='utf-8')
        legacy_train = old['project/scripts/train/lewm.py']
        assert legacy_train.count('epoch_interval=1,') == 1
        legacy_train = legacy_train.replace('epoch_interval=1,', 'epoch_interval=10,')
        legacy_train = legacy_train.replace('        if trainer.is_global_zero:',
            "        if trainer.is_global_zero:\n            print(f'[EPOCH_DONE] {trainer.current_epoch + 1}/{trainer.max_epochs}', flush=True)", 1)
        (ref/'train_lewm.py').write_text(legacy_train, encoding='utf-8')
        pkg = Path(importlib.util.find_spec('stable_worldmodel').origin).parent
        files['module.py'] = pkg/'wm/lewm/module.py'
        backup = self.out/'backup'
        backup.mkdir(exist_ok=True)
        for name, target in files.items():
            if target.exists() and not (backup/name).exists():
                shutil.copy2(target, backup/name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(uploads/name, target)
        target = self.root/'cl_protocol/test'/Path(__file__).name
        if Path(__file__).resolve() != target.resolve():
            shutil.copy2(__file__, target)
        configs = self.out/'configs'
        configs.mkdir(exist_ok=True)
        for stage in (2,3):
            source = yaml.safe_load((checkpoint_home/f'm3g_reacher_t{stage}/config.yaml').read_text())
            for arm in ARMS:
                cfg = build_config(source, arm, stage, self.args.run_id,
                    seq['tasks'][f'T{stage}']['dataset'], checkpoint_home)
                (configs/f'{arm}_t{stage}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        self.command('real_growth_check', [sys.executable, self.root/'cl_protocol/test/growth_check.py',
            '--policy', TEACHER, '--device','cuda', '--out', self.out/'real_growth_check.json'])
        pilot_path = HOME/'datasets'/pilot['dataset']
        if not pilot_path.exists():
            self.collect('capacity_pilot',10,20261021)
        info,_ = dataset_info(pilot_path,10,expected['T2'])
        size = sum(p.stat().st_size for p in pilot_path.rglob('*') if p.is_file())
        free = shutil.disk_usage(HOME).free
        estimated = size*100*1.25
        required_bytes = estimated+2*GIB
        self.state['capacity'] = dict(pilot_bytes=size, pilot=info,
            estimated_two_training_datasets_bytes=estimated,
            reserved_model_and_evaluation_bytes=2*GIB, available_bytes=free)
        if free < required_bytes:
            raise RuntimeError(f'Insufficient capacity: free={free/GIB:.2f} GiB, estimated need={required_bytes/GIB:.2f} GiB; no old files deleted')
        self.state.update(status='ready', created=datetime.now().astimezone().isoformat(),
            design='Fresh shared T2/T3 data; paired legacy/repaired growth; same T1 checkpoint, data, global/data seeds, hyperparameters and fixed evaluation tasks.',
            collection_seeds={'T2':20261021,'T3':20262021}, evaluation_seed=42,
            note='Recollection is not restoration of the original trajectories; compare paired new runs, not causal differences against historical runs.')
        self.store()
        emit('PREFLIGHT_OK', self.state['capacity'])

    def train(self, arm, stage):
        cfgpath = self.out/'configs'/f'{arm}_t{stage}.yaml'
        cfg = yaml.safe_load(cfgpath.read_text())
        log = self.command(f'train_{arm}_t{stage}', [sys.executable, Path(__file__).resolve(),
            '--mode','worker','--root',self.root,'--arm',arm,'--config',cfgpath])
        import torch
        path = HOME/'checkpoints'/cfg['output_model_name']/'weights_epoch_10.pt'
        old = torch.load(cfg['warm_start'],map_location='cpu',weights_only=True)
        new = torch.load(path,map_location='cpu',weights_only=True)
        changes = {}
        for key in ('predictor.adapter_A','predictor.adapter_B'):
            changes[key] = float((new[key][:stage-1]-old[key]).abs().max())
        scale = float(new.get('predictor.adapter_scale',torch.tensor(1.0)))
        expected_scale = float(old.get('predictor.adapter_scale',torch.tensor(1.0)))*stage/(stage-1) if arm=='repaired' else 1.0
        if abs(scale-expected_scale)>1e-6 or new['predictor.adapter_A'].shape[0]!=stage:
            raise RuntimeError('Wrong saved growth scale/slot count')
        if arm == 'repaired':
            if any(changes.values()) or '[FROZEN_OPTIMIZER_BOUND]' not in log.read_text(encoding='utf-8', errors='replace'):
                raise RuntimeError('Strict frozen-slot validation failed after actual training')
        if not new['predictor.adapter_A'][stage-1:].abs().max()>0:
            raise RuntimeError('New adapter did not learn')
        self.state['training'][f'{arm}_t{stage}'] = dict(policy=cfg['output_model_name']+'/weights_epoch_10.pt',
            old_slot_max_abs_changes=changes, adapter_scale=scale, config=str(cfgpath))
        self.store()
        emit('TRAIN_CHECK_OK', self.state['training'][f'{arm}_t{stage}'])
        return cfg['output_model_name']+'/weights_epoch_10.pt'

    def evaluate(self, label, policy, tasks):
        matrix = self.state['behavior'].setdefault(label,{})
        for task in tasks:
            log = self.command(f'eval_{label}_{task}', [sys.executable,self.root/'cl_protocol/eval_task.py',
                '--tasks',self.out/'tasks.yaml','--sequence','B_reacher','--task',task,
                '--policy',policy,'--num-eval',100,'--seed',42,
                '--eval-wm',self.root/'scripts/plan/eval_wm.py'])
            rows = [json.loads(line[len('CELL_RESULT '):]) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('CELL_RESULT ')]
            if len(rows)!=1 or rows[0]['policy']!=policy or rows[0]['task']!=task or rows[0]['num_eval']!=100 or rows[0]['seed']!=42:
                raise RuntimeError('Missing/mismatched fresh evaluation result')
            matrix[task] = rows[0]
            self.store()
            emit('EVAL_CHECK_OK', dict(label=label,task=task,success_rate=rows[0]['success_rate']))

    def predict(self,label,policy,tasks):
        folder = self.out/'prediction'/label
        folder.mkdir(parents=True,exist_ok=True)
        self.command('pred_'+label,[sys.executable,self.root/'cl_protocol/pred_error.py',
            '--tasks-cfg',self.out/'tasks.yaml','--sequence','B_reacher','--policy',policy,
            '--tasks',*tasks,'--windows',1000,'--reserve-last',200,'--batch',64,'--seed',0],folder)
        files = list(folder.glob('pred_error_*.json'))
        if len(files)!=1:
            raise RuntimeError('Expected one prediction result')
        data = json.loads(files[0].read_text())
        if set(data['results'])!=set(tasks) or any(row['n_windows']!=1000 or not math.isfinite(row['mse_mean']) or row['mse_mean']<0 for row in data['results'].values()):
            raise RuntimeError('Incomplete prediction results')
        self.state['prediction'][label]=data
        self.store()
        emit('PRED_CHECK_OK',label)

    def run(self):
        if self.state['status']!='ready':
            raise RuntimeError('Run --mode prepare successfully first; a failed/running batch is not silently resumed')
        self.state['status']='running'
        self.store()
        tasks=yaml.safe_load((self.out/'tasks.yaml').read_text())['sequences']['B_reacher']
        for stage in (2,3):
            task=f'T{stage}'
            self.collect(task,500,self.state['collection_seeds'][task])
            info,starts=dataset_info(HOME/'datasets'/tasks['tasks'][task]['dataset'],500,tasks['tasks'][task]['variation_values'])
            test_info,test_starts=dataset_info(HOME/'datasets'/tasks['probe_tasks'][task+'_test']['dataset'])
            if starts & test_starts:
                raise RuntimeError('Recollected training/test datasets share initial physical states')
            self.state.setdefault('collection',{})[task]=dict(training=info,test=test_info,overlapping_initial_states=0)
            self.store()
            emit('DATA_CHECK_OK',self.state['collection'][task])
        if shutil.disk_usage(HOME).free<2*GIB:
            raise RuntimeError('Less than 2 GiB remains for new weights/evaluation; old files preserved')
        self.evaluate('t1',TEACHER,['T1'])
        if self.state['behavior']['t1']['T1']['success_rate']==0:
            raise RuntimeError('Shared T1 baseline is zero; stop before training')
        for stage in (2,3):
            policies={arm:self.train(arm,stage) for arm in ARMS}
            selected=['T1','T2_test']+(['T3_test'] if stage==3 else [])
            for arm,policy in policies.items():
                label=f'{arm}_t{stage}'
                self.evaluate(label,policy,selected)
                self.predict(label,policy,selected)
        self.predict('t1',TEACHER,['T1','T2_test','T3_test'])
        self.state.update(status='complete',current=None,finished=datetime.now().astimezone().isoformat())
        self.store()
        emit('BATCH_DONE',str(self.summary))


class FixedBatch(Batch):
    def run(self):
        """Wait for the paired queue, then train its same-data K=4 comparator."""
        import fcntl
        if self.state['status'] == 'complete':
            emit('FIXED_BATCH_DONE', str(self.summary))
            return
        parent = self.root/'results'/self.args.after_run_id
        if not (parent/'summary.json').is_file():
            raise FileNotFoundError(f'Missing preceding batch: {parent}')
        self.state.update(status='waiting', after_run_id=self.args.after_run_id)
        self.store()
        emit('QUEUE_WAIT', str(parent/'run.lock'))
        # Hold the preceding lock throughout, so another launch cannot overlap.
        with (parent/'run.lock').open('a') as previous_lock:
            fcntl.flock(previous_lock, fcntl.LOCK_EX)
            paired = json.loads((parent/'summary.json').read_text())
            if paired['status'] != 'complete':
                raise RuntimeError('Preceding batch has not completed successfully; fixed comparison stopped')
            installed = self.root/'cl_protocol/test'/Path(__file__).name
            if Path(__file__).resolve() != installed.resolve():
                shutil.copy2(__file__, installed)
            self.run_fixed(parent, paired)

    def run_fixed(self, parent, paired):
        if self.state['training'] or self.state['behavior'] or self.state['prediction']:
            raise RuntimeError('Existing fixed results preserved; automatic retraining is disabled')
        checkpoint_home = HOME/'checkpoints'
        configs = self.out/'configs'
        configs.mkdir(exist_ok=True)
        tasks = yaml.safe_load((parent/'tasks.yaml').read_text(encoding='utf-8'))
        seq = tasks['sequences']['B_reacher']
        for stage in (2, 3):
            task = f'T{stage}'
            dataset = seq['tasks'][task]['dataset']
            record = paired['collection'][task]
            if record['training']['episodes'] != 500 or record['overlapping_initial_states'] != 0:
                raise RuntimeError('Preceding training/test checks are incomplete')
            if (HOME/'datasets'/dataset).resolve() != Path(record['training']['path']).resolve():
                raise RuntimeError('Task snapshot differs from the preceding training dataset')
            if (checkpoint_home/f'{self.args.run_id}_fixed_t{stage}').exists():
                raise RuntimeError('New fixed checkpoint directory already exists; old weights preserved')
            source = yaml.safe_load((checkpoint_home/f'm3_reacher_t{stage}'/'config.yaml').read_text())
            cfg = build_fixed_config(source, stage, self.args.run_id, dataset, checkpoint_home)
            (configs/f'fixed_t{stage}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
            info, starts = dataset_info(HOME/'datasets'/dataset, 500, seq['tasks'][task]['variation_values'])
            test_info, test_starts = dataset_info(HOME/'datasets'/seq['probe_tasks'][task+'_test']['dataset'])
            if starts & test_starts:
                raise RuntimeError('Training/test initial states overlap')
            self.state.setdefault('collection', {})[task] = dict(training=info, test=test_info, overlapping_initial_states=0)
        if shutil.disk_usage(HOME).free < GIB:
            raise RuntimeError('Less than 1 GiB free for two final checkpoints and evaluation; no files deleted')
        import torch
        initial = torch.load(checkpoint_home/FIXED_TEACHER, map_location='cpu', weights_only=True)
        if initial['predictor.adapter_A'].shape[0] != 4 or float(initial.get('predictor.adapter_scale', torch.tensor(1.0))) != 1.0:
            raise RuntimeError('Unexpected fixed T1 checkpoint capacity/scale')
        del initial
        (self.out/'tasks.yaml').write_text(yaml.safe_dump(tasks, allow_unicode=True, sort_keys=False), encoding='utf-8')
        self.state.update(status='running', design='Archived fixed K=4 chain, same fresh T2/T3 training data and evaluation protocol as paired growth. Different archived T1 initial checkpoint; report change relative to each own T1 baseline, not a single-variable growth ablation.',
                          initial_policy=FIXED_TEACHER, training_seed=3072, evaluation_seed=42)
        self.store()
        emit('FIXED_PREFLIGHT_OK', dict(after_run=self.args.after_run_id, initial_policy=FIXED_TEACHER))
        self.evaluate('fixed_t1', FIXED_TEACHER, ['T1'])
        if self.state['behavior']['fixed_t1']['T1']['success_rate'] == 0:
            raise RuntimeError('Fixed T1 baseline is zero; stop before training')
        for stage in (2, 3):
            cfgpath = configs/f'fixed_t{stage}.yaml'
            cfg = yaml.safe_load(cfgpath.read_text())
            # repaired selects the installed training entry, not growth settings;
            # the fixed config explicitly uses K=4 and freezes zero slots.
            self.command(f'train_fixed_t{stage}', [sys.executable, Path(__file__).resolve(),
                '--mode', 'worker', '--root', self.root, '--arm', 'repaired', '--config', cfgpath])
            policy = cfg['output_model_name']+'/weights_epoch_10.pt'
            state = torch.load(checkpoint_home/policy, map_location='cpu', weights_only=True)
            if state['predictor.adapter_A'].shape[0] != 4 or float(state['predictor.adapter_scale']) != 1.0:
                raise RuntimeError('Fixed K=4 training changed capacity/scale')
            del state
            self.state['training'][f'fixed_t{stage}'] = dict(policy=policy, config=str(cfgpath), num_adapters=4, frozen_slots=0)
            self.store()
            emit('TRAIN_CHECK_OK', self.state['training'][f'fixed_t{stage}'])
            selected = ['T1', 'T2_test']+(['T3_test'] if stage == 3 else [])
            self.evaluate(f'fixed_t{stage}', policy, selected)
            self.predict(f'fixed_t{stage}', policy, selected)
        self.predict('fixed_t1', FIXED_TEACHER, ['T1', 'T2_test', 'T3_test'])
        self.state.update(status='complete', current=None, finished=datetime.now().astimezone().isoformat())
        self.store()
        emit('FIXED_BATCH_DONE', str(self.summary))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('prepare','run','worker','fixed'))
    parser.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/stable-worldmodel'))
    parser.add_argument('--upload',type=Path,default=Path('/root/autodl-tmp'))
    parser.add_argument('--run-id',default=RUN)
    parser.add_argument('--after-run-id',default=RUN)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    if args.mode=='worker':
        worker(args)
        return
    if args.mode == 'fixed' and args.run_id == args.after_run_id:
        parser.error('fixed comparison requires a distinct --run-id')
    emit('BATCH_START', args.mode)
    import fcntl
    batch=(FixedBatch if args.mode == 'fixed' else Batch)(args)
    with (batch.out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            (batch.prepare if args.mode=='prepare' else batch.run)()
        except Exception as error:
            batch.state.update(status='failed',error=str(error))
            batch.store()
            emit('BATCH_FAILED',str(error))
            traceback.print_exc()
            raise SystemExit(1)


if __name__=='__main__':
    main()
