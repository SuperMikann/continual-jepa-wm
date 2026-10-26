"""Collection, importable replay datasets and native training integration."""
import copy
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

from .common import CASES, emit, save, inspect_collection
from .evaluation import apply_condition


def adaptation_config(source, run_id, dataset, checkpoint, checkpoint_home, epochs):
    """Continue the same K=1 architecture and objective, without retention terms."""
    cfg = copy.deepcopy(source)
    if cfg['model']['predictor']['num_adapters'] != 1:
        raise ValueError('The qualification run must start from the screened K=1 model')
    if cfg['optimizer'] != dict(type='AdamW', lr=5e-5, weight_decay=0.001):
        raise ValueError('The teacher optimizer differs from the supplied configuration')
    cfg.update(output_model_name=run_id, subdir=run_id,
               warm_start=str(checkpoint), freeze_old_adapters=0,
               freeze_encoder=False, freeze_dynamics=False,
               save_epoch_interval=epochs, dump_object=False)
    cfg.pop('anchor_path', None)
    for name in ('anchor_weight', 'sub_weight', 'ent_weight'):
        if name in cfg['loss']:
            cfg['loss'][name] = 0.0
    # The native entrypoint supplies enable_checkpointing explicitly.
    cfg['trainer'].pop('enable_checkpointing', None)
    cfg['trainer'].update(max_epochs=epochs, devices=1,
                          default_root_dir=str(checkpoint_home/run_id/'trainer'))
    cfg['data']['dataset']['name'] = dataset
    cfg['wandb']['enabled'] = False
    return cfg


def build_collection_world(base_world, case='damping_2'):
    import numpy as np

    class DampingCollectionWorld(base_world):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for env in self.envs.envs:
                raw = env.unwrapped
                original = raw.apply_runtime_variations

                def runtime(original=original, raw=raw):
                    original()
                    apply_condition(raw, CASES[case])

                raw.apply_runtime_variations = runtime

        def _run_iter(self, *args, on_step=None, **kwargs):
            calls = 0

            def record(world, mask):
                nonlocal calls
                damping = np.asarray([e.unwrapped.env.physics.model.dof_damping.copy()
                                      for e in self.envs.envs])
                gear = np.asarray([e.unwrapped.env.physics.model.actuator_gear[:, 0].copy()
                                   for e in self.envs.envs])
                if not (np.allclose(damping, 0.01*CASES[case]['damping'])
                        and np.allclose(gear, 0.05)):
                    raise RuntimeError('Collection reset lost the requested damping/actuation')
                world.infos['dynamics_damping'] = damping[:, None, :]
                world.infos['dynamics_motor_gear'] = gear[:, None, :]
                calls += 1
                if calls % 200 == 0:
                    if shutil.disk_usage(os.environ['STABLEWM_HOME']).free < 512*2**20:
                        raise RuntimeError('Collection stopped before the data disk fills')
                    emit('COLLECT_PROGRESS', dict(callbacks=calls))
                if on_step:
                    on_step(world, mask)

            yield from super()._run_iter(*args, on_step=record, **kwargs)

    return DampingCollectionWorld


def collection_worker(args):
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    sys.path.insert(0, str(args.root/'cl_protocol'))
    import stable_worldmodel as swm
    from drift_world import DriftWorld
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    path = Path(os.environ['STABLEWM_HOME'])/'datasets'/cfg['dataset']
    if path.exists():
        raise FileExistsError(f'Never append to an existing collection: {path}')
    case = cfg.get('case', 'damping_2')
    cls = build_collection_world(DriftWorld, case)
    world = cls('swm/ReacherDMControl-v0', task='hard', num_envs=10,
                image_shape=(224, 224), max_episode_steps=200)
    world.set_policy(swm.policy.RandomPolicy(seed=cfg['seed']))
    world.collect(path, episodes=cfg['episodes'], seed=cfg['seed'], progress=False)
    info = inspect_collection(path, cfg['episodes'], case)
    emit('COLLECT_DONE', info)


class ReplayWindows:
    """Keep the original epoch length; replace about 1/4 of windows by replay.

    Only indices are kept in memory. Both sources use the new-data transform,
    as in ordinary continuation. DataLoader worker RNGs draw windows anew.
    """
    def __init__(self, new, old, memory):
        if not len(new) or not memory:
            raise ValueError('Replay needs nonempty new data and old memory')
        self.new, self.old, self.memory = new, old, memory

    def __len__(self):
        return len(self.new)

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        import torch
        output = [None]*len(indices)
        for replay, dataset in ((False, self.new), (True, self.old)):
            positions = [i for i, index in enumerate(indices) if (int(index) % 4 == 0) == replay]
            if not positions:
                continue
            size = len(self.memory) if replay else len(self.new)
            selected = torch.randint(size, (len(positions),)).tolist()
            if replay:
                selected = [self.memory[i] for i in selected]
            fetch = getattr(dataset, '__getitems__', None)
            values = fetch(selected) if fetch is not None else [dataset[i] for i in selected]
            if not isinstance(values, (list, tuple)) or len(values) != len(positions):
                raise TypeError('Expected batched dataset access to return one dict per window')
            for position, value in zip(positions, values):
                item = dict(value, _replay_source=int(replay))
                item.setdefault('_replay_task', 0 if replay else -1)
                output[position] = item
        return output


class OldMemoryWindows:
    """Batched access to a fixed, balanced memory of multiple past tasks."""
    def __init__(self, datasets, memories):
        self.datasets, self.memories = datasets, memories
        self.lookup = [(task, index) for task, memory in enumerate(memories)
                       for index in memory]

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        output = [None]*len(indices)
        for task, dataset in enumerate(self.datasets):
            positions = [i for i, index in enumerate(indices) if self.lookup[index][0] == task]
            selected = [self.lookup[indices[i]][1] for i in positions]
            if not positions:
                continue
            fetch = getattr(dataset, '__getitems__', None)
            values = fetch(selected) if fetch else [dataset[i] for i in selected]
            if len(values) != len(positions):
                raise RuntimeError('Incomplete old-memory batch')
            for position, value in zip(positions, values):
                output[position] = dict(value, _replay_task=task)
        return output


def install_comparison_training(cfg, entry):
    """Use native training unchanged except for its train subset and a receipt.

    The old-data transform is fitted on the same new data as the completed
    continuation. Validation stays on its original new-data split. Adding a
    callback through Trainer.__init__ preserves the Trainer class identity.
    """
    import lightning as pl
    from lightning.pytorch.callbacks import Callback
    from omegaconf import OmegaConf
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch

    protocol = cfg['comparison']
    method = protocol['method']
    replay_type = ReplayWindows if method == 'replay' else None
    original_split = spt.data.random_split
    original_init = pl.Trainer.__init__
    metadata = dict(method=method, epochs=cfg['trainer']['max_epochs'],
                    batch_size=cfg['loader']['batch_size'], memory_windows=0)
    if method == 'growth':
        import hydra
        predictor = hydra.utils.get_class(cfg['model']['predictor']['_target_'])
        if ('FreezeOldAdaptersCallback' not in entry.read_text(encoding='utf-8')
                or not callable(getattr(predictor, 'prepare_growth_state_dict', None))
                or not callable(getattr(predictor, 'bind_frozen_optimizer', None))):
            raise RuntimeError('Install the previously repaired growth/frozen-optimizer implementation first')

    def split(dataset, *args, **kwargs):
        train, val = original_split(dataset, *args, **kwargs)
        metadata.update(new_train_windows=len(train), validation_windows=len(val),
                        batches_per_epoch=len(train)//cfg['loader']['batch_size'])
        metadata['expected_updates'] = metadata['batches_per_epoch']*metadata['epochs']
        if method == 'replay':
            dc = OmegaConf.to_container(OmegaConf.create(cfg).data.dataset, resolve=True)
            dc.pop('name')
            names = protocol.get('old_datasets', [protocol['old_dataset']])
            sources, memories = [], []
            for name in names:
                old = swm.data.load_dataset(name, transform=dataset.transform, **dc)
                old_train, _ = original_split(old, lengths=kwargs['lengths'],
                                             generator=torch.Generator().manual_seed(cfg['seed']))
                memory = torch.randperm(len(old_train), generator=torch.Generator().manual_seed(20261024))
                memories.append(memory[:min(4096//len(names), len(old_train))].tolist())
                sources.append(old_train)
            if len(names) == 1:
                old_train, memory = sources[0], memories[0]
                record = dict(dataset=names[0], old_split_seed=cfg['seed'],
                              selection_seed=20261024, indices_in_old_train_split=memory)
            else:
                import importlib
                pool_type = importlib.import_module(replay_type.__module__).OldMemoryWindows
                old_train = pool_type(sources, memories)
                memory = list(range(len(old_train)))
                record = dict(datasets=names, old_split_seed=cfg['seed'], selection_seed=20261024,
                              indices_in_old_train_splits=memories)
            memory_path = Path(protocol['receipt']).parent/'replay_indices.json'
            if memory_path.exists() and json.loads(memory_path.read_text(encoding='utf-8')) != record:
                raise RuntimeError('Replay memory differs from the saved run')
            save(memory_path, record)
            metadata.update(memory_windows=len(memory), old_train_windows=sum(len(s) for s in sources),
                            memory_index_file=str(memory_path), nominal_old_fraction=0.25)
            if len(names) > 1:
                metadata.update(old_datasets=names, memory_windows_by_task=[len(x) for x in memories])
            train = replay_type(train, old_train, memory)
        emit('BUDGET', metadata)
        return train, val

    class TrainingReceipt(Callback):
        def __init__(self):
            self.old_samples = self.new_samples = 0
            self.old_samples_by_task = {}

        def state_dict(self):
            return dict(old_samples=self.old_samples, new_samples=self.new_samples,
                        old_samples_by_task=self.old_samples_by_task)

        def load_state_dict(self, state_dict):
            self.old_samples = state_dict['old_samples']
            self.new_samples = state_dict['new_samples']
            self.old_samples_by_task = state_dict.get('old_samples_by_task', {})

        def on_train_start(self, trainer, pl_module):
            if int(trainer.num_training_batches) != metadata['batches_per_epoch']:
                raise RuntimeError('Native training batch count changed from the continuation budget')

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            marker = batch.pop('_replay_source', None)
            task = batch.pop('_replay_task', None)
            old = 0 if marker is None else int(marker.sum().item())
            self.old_samples += old
            self.new_samples += int(batch['pixels'].shape[0])-old
            if task is not None:
                for index in range(len(protocol.get('old_datasets', [protocol['old_dataset']]))):
                    key = str(index)
                    self.old_samples_by_task[key] = self.old_samples_by_task.get(key, 0)+int((task == index).sum().item())

        def on_fit_end(self, trainer, pl_module):
            if trainer.global_step != metadata['expected_updates']:
                raise RuntimeError('Optimizer update count differs from the continuation budget')
            if method == 'replay' and not self.old_samples:
                raise RuntimeError('Replay never supplied an old-data window')
            total = self.old_samples+self.new_samples
            receipt = dict(metadata, status='complete', global_step=int(trainer.global_step),
                           old_samples=self.old_samples, new_samples=self.new_samples,
                           old_samples_by_task=self.old_samples_by_task,
                           actual_old_fraction=self.old_samples/total,
                           parameters_total=sum(p.numel() for p in pl_module.model.parameters()),
                           parameters_requires_grad=sum(p.numel() for p in pl_module.model.parameters()
                                                        if p.requires_grad))
            save(Path(protocol['receipt']), receipt)
            emit('TRAINING_RECEIPT', receipt)

    def init(trainer, *args, **kwargs):
        kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), TrainingReceipt()]
        original_init(trainer, *args, **kwargs)

    spt.data.random_split = split
    pl.Trainer.__init__ = init

    def restore():
        spt.data.random_split = original_split
        pl.Trainer.__init__ = original_init
    return restore


def training_worker(args):
    import random
    import yaml
    import numpy as np
    import torch
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    os.environ.pop('LOCAL_DATASET_DIR', None)
    os.chdir(args.root)
    entry = args.root/'scripts/train/lewm.py'
    sys.argv = [str(entry), '--config-path', str(args.config.parent),
                '--config-name', args.config.stem, 'hydra.job.chdir=false',
                'hydra.run.dir='+str(args.config.parent/'hydra')]
    emit('TRAIN_START', dict(epochs=cfg['trainer']['max_epochs'], seed=cfg['seed'],
                            checkpoint=cfg['warm_start'], adapters=cfg['model']['predictor']['num_adapters']))
    restore = install_comparison_training(cfg, entry) if cfg.get('comparison') else None
    try:
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        if restore is not None:
            restore()


def comparison_training_config(base, run_id, method, old_dataset, receipt, checkpoint_home):
    cfg = copy.deepcopy(base)
    cfg.update(output_model_name=run_id, subdir=run_id)
    cfg['trainer']['default_root_dir'] = str(checkpoint_home/run_id/'trainer')
    old_k = base['model']['predictor']['num_adapters']
    cfg['model']['predictor']['num_adapters'] = old_k+1 if method == 'growth' else old_k
    cfg['freeze_old_adapters'] = old_k if method == 'growth' else 0
    cfg['comparison'] = dict(method=method, old_dataset=old_dataset, receipt=str(receipt))
    return cfg
