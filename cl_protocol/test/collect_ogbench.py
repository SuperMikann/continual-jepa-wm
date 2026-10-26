"""Collect baseline Cube or PointMaze data locally using the installed SWM.

No model training. Reuse this entry for the pilot and full collection.
"""

import argparse
from functools import partial
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('MUJOCO_GL', 'glfw')

import gymnasium as gym
import h5py
import hdf5plugin
import mujoco
import numpy as np
from PIL import Image
import torch
import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset, HDF5Writer
from stable_worldmodel.policy import BasePolicy
from stable_worldmodel.wrapper import MapKeysWrapper


class PhysicsInfo(gym.Wrapper):
    """Save simulator state, including reset frames, for later replay."""

    def _record(self, info):
        raw = self.unwrapped
        model = raw._model if hasattr(raw, '_model') else raw.model
        data = raw._data if hasattr(raw, '_data') else raw.data
        info['qpos'] = data.qpos.copy()
        info['qvel'] = data.qvel.copy()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
        mujoco.mj_getState(model, data, state, spec)
        info['physics_state'] = state
        if hasattr(raw, 'cur_goal_xy'):
            info['collection_goal_xy'] = np.asarray(raw.cur_goal_xy).copy()
        info['collection_seed'] = self.episode_seed
        return info

    def reset(self, **kwargs):
        self.episode_seed = kwargs.get('seed')
        if self.episode_seed is not None:
            # The pinned SWM Cube.reset accepts seed but drops it before super.
            # Seed its native generator and variation spaces explicitly here.
            gym.Env.reset(self.unwrapped, seed=self.episode_seed)
            if hasattr(self.unwrapped, 'variation_space'):
                self.unwrapped.variation_space.seed(self.episode_seed)
            self.action_space.seed(self.episode_seed)
        observation, info = self.env.reset(**kwargs)
        # Goals for model training are future dataset frames. Native PointMaze
        # supplies a reset-only goal image, which must not become a short column.
        info.pop('goal', None)
        info.pop('goal_rendered', None)
        return observation, self._record(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, self._record(info)


class PointMazePolicy(BasePolicy):
    """Follow OGBench's native BFS subgoals with noisy bounded actions.

    This is a data collection controller, not a learned policy or a claim to
    reproduce a particular paper's data. Resample navigation targets on arrival.
    """

    def __init__(self, seed, noise=0.2):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.targets_reached = 0

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict, **kwargs):
        actions = []
        for wrapped in self.env.envs:
            env = wrapped.unwrapped
            xy = env.get_xy()
            if np.linalg.norm(xy - env.cur_goal_xy) < 0.6:
                free = np.argwhere(env.maze_map == 0)
                cell = free[self.rng.integers(len(free))]
                env.set_goal(goal_xy=env.ij_to_xy(cell))
                self.targets_reached += 1
            goal = env.cur_goal_xy
            if env.xy_to_ij(xy) == env.xy_to_ij(goal):
                subgoal = goal
            else:
                subgoal, _ = env.get_oracle_subgoal(xy, goal)
            direction = np.asarray(subgoal) - xy
            # Native PointEnv adds 0.2 * action to position before collision.
            action = direction / max(float(np.linalg.norm(direction)), 0.2)
            action += self.rng.normal(0.0, self.noise, 2)
            actions.append(np.clip(action, -1, 1).astype(np.float32))
        return np.stack(actions)


class CompactWriter(HDF5Writer):
    """Flatten slash-delimited names for the native HDF5 reader."""

    def __init__(self, *args, compression='gzip', gzip_level=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = time.perf_counter()
        self.key_mapping = {}
        self.compression = compression
        self.gzip_level = gzip_level

    def _init_schema(self, sample):
        super()._init_schema(sample)
        for key, dataset in list(self._f.items()):
            if dataset.dtype == np.uint8 and dataset.ndim == 4:
                shape, dtype = dataset.shape, dataset.dtype
                assert shape[0] == 0
                del self._f[key]
                options = ({'compression_opts': self.gzip_level}
                           if self.compression == 'gzip' else {})
                self._f.create_dataset(key, shape=shape,
                    maxshape=(None, *shape[1:]), dtype=dtype,
                    chunks=(1, *shape[1:]), compression=self.compression,
                    **options)

    def write_episode(self, episode):
        mapping = {key: key.replace('/', '_') for key in episode}
        assert len(set(mapping.values())) == len(mapping), 'Column name collision'
        self.key_mapping.update(mapping)
        super().write_episode({mapping[k]: v for k, v in episode.items()})
        if self._ep_written == 1 or self._ep_written % 100 == 0:
            self._f.flush()
            print('[COLLECT_PROGRESS]', json.dumps({
                'episodes': self._ep_written, 'frames': self._global_ptr,
                'seconds': round(time.perf_counter() - self.started, 1),
            }), flush=True)


def check_replay(world, path):
    raw = world.envs.envs[0].unwrapped
    model = raw._model if hasattr(raw, '_model') else raw.model
    data = raw._data if hasattr(raw, '_data') else raw.data
    with h5py.File(path, 'r') as f:
        initial_state = f['physics_state'][0]
        actions = f['action'][:10]
        positions = f['qpos'][:11]
        velocities = f['qvel'][:11]
        image = f['pixels'][0]
    mujoco.mj_setState(model, data, initial_state,
                       mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    rendered = raw.render()
    pixel_mae = float(np.abs(rendered.astype(float) - image.astype(float)).mean())
    errors = []
    for i, action in enumerate(actions, 1):
        raw.step(action)
        errors.append(max(float(np.abs(data.qpos - positions[i]).max()),
                          float(np.abs(data.qvel - velocities[i]).max())))
    error = max(errors)
    assert error < 1e-4, f'Replay state mismatch: {error}'
    assert pixel_mae < 1.0, f'Replay image mismatch: {pixel_mae}'
    return {'steps': len(actions), 'max_state_error': error,
            'initial_pixel_mae': pixel_mae}


def repack_for_upload(source, output_dir):
    """Keep every frame/column; only change lossless storage compression."""
    source = source.resolve()
    source_report = source.parent / 'collection_summary.json'
    report = json.loads(source_report.read_text(encoding='utf-8'))
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source.name
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    started = time.perf_counter()
    with h5py.File(source, 'r') as f, h5py.File(destination, 'x', libver='latest') as g:
        assert len(f['ep_len']) == report['episodes'], 'Source not complete'
        for key, ds in f.items():
            if ds.dtype == np.uint8 and ds.ndim == 4:
                target = g.create_dataset(key, shape=ds.shape,
                    maxshape=(None, *ds.shape[1:]), dtype=ds.dtype,
                    chunks=(32, *ds.shape[1:]), **hdf5plugin.Zstd(clevel=3))
                for lo in range(0, len(ds), 32):
                    values = ds[lo:lo+32]
                    target[lo:lo+32] = values
                    assert np.array_equal(values, target[lo:lo+32])
                    if lo == 0 or lo % 100000 == 0:
                        print('[REPACK_PROGRESS]', key, lo, '/', len(ds), flush=True)
            else:
                f.copy(key, g)
                assert np.array_equal(ds[:], g[key][:], equal_nan=ds.dtype.kind in 'fc')
        episodes, frames = len(g['ep_len']), int(g['ep_len'][:].sum())
        size = g['pixels'].shape[1]
    reader = HDF5Dataset(path=destination, frameskip=5, num_steps=4,
                        keys_to_load=['pixels', 'action', 'qpos', 'qvel'])
    try:
        assert tuple(reader[0]['pixels'].shape) == (4, 3, size, size)
    finally:
        if reader.h5_file is not None:
            reader.h5_file.close()
    for name in ('episode_splits.json', 'preview.png'):
        if (source.parent / name).exists():
            shutil.copy2(source.parent / name, output_dir / name)
    shutil.copy2(source_report, output_dir / 'source_collection_summary.json')
    result = {'status': 'lossless repack checked; server reader compatibility pending',
        'source': str(source), 'file': str(destination), 'episodes': episodes,
        'frames': frames, 'image_size': size, 'source_bytes': source.stat().st_size,
        'file_bytes': destination.stat().st_size, 'compression': 'HDF5 Zstd level 3, 32-frame chunks',
        'all_columns_equal': True, 'native_reader_passed': True,
        'under_40GB_upload_budget': destination.stat().st_size < 40_000_000_000,
        'seconds': time.perf_counter() - started,
        'server_dependency': 'hdf5plugin imported before opening HDF5'}
    (output_dir / 'collection_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[REPACK_DONE]', json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=('cube', 'pointmaze'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=3072)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--compression', choices=('lzf', 'gzip'), default='gzip')
    parser.add_argument('--gzip-level', type=int, choices=range(1, 10), default=6)
    parser.add_argument('--image-size', type=int, default=None,
                        help='Default: Cube 224, PointMaze 64 (native pixel interface).')
    parser.add_argument('--repack-input', type=Path,
                        help='Repack an already completed collection for upload; do not collect.')
    args = parser.parse_args()
    if args.repack_input is not None:
        repack_for_upload(args.repack_input, args.out.resolve())
        return
    if args.image_size is None:
        args.image_size = 224 if args.env == 'cube' else 64
    if args.episodes < 1 or args.max_steps < 26 or args.threads < 1:
        parser.error('episodes/threads must be positive; max-steps must be >= 26')
    if args.image_size < 32:
        parser.error('image-size must be at least 32')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'{args.env}_baseline.h5'
    if path.exists() or (args.out / 'collection_summary.json').exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    print('[COLLECT_START]', json.dumps({**vars(args), 'out': str(args.out)}), flush=True)
    common = dict(num_envs=1, image_shape=(args.image_size, args.image_size),
                  max_episode_steps=args.max_steps, goal_conditioned=False,
                  pre_wrappers=[PhysicsInfo])
    if args.env == 'cube':
        from stable_worldmodel.envs.ogbench import ExpertPolicy
        environment_id = 'swm/OGBCube-v0'
        world = swm.World(environment_id, **common, env_type='single',
            ob_type='states', multiview=False,
            width=args.image_size, height=args.image_size,
            visualize_info=False, terminate_at_goal=False, mode='data_collection')
        policy = ExpertPolicy(seed=args.seed, action_noise=0.1)
        policy_description = 'SWM native Cube markov oracle; action_noise=0.1'
    else:
        import ogbench  # Registers the native baseline environment.
        environment_id = 'pointmaze-medium-v0'
        # The pinned SWM visual-variation wrapper produced black images locally;
        # use native OGBench for this fixed baseline, retaining SWM collection IO.
        world = swm.World(environment_id, **common, ob_type='pixels',
            width=args.image_size, height=args.image_size,
            terminate_at_goal=False, add_pixels=False,
            extra_wrappers=[partial(MapKeysWrapper,
                                    key_map={'observation': 'pixels'})])
        policy = PointMazePolicy(args.seed)
        policy_description = 'OGBench native BFS subgoals; direction actions; Gaussian noise std=0.2'
    world.set_policy(policy)
    writer = CompactWriter(path, mode='error', compression=args.compression,
                           gzip_level=args.gzip_level)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            world.collect(writer=writer, episodes=args.episodes, seed=args.seed,
                          progress=False)
        elapsed = time.perf_counter() - started
        replay = check_replay(world, path)
    finally:
        world.close()
    with h5py.File(path, 'r') as f:
        lengths = f['ep_len'][:]
        offsets = f['ep_offset'][:]
        assert len(lengths) == args.episodes
        assert np.all(lengths == args.max_steps + 1)
        assert np.array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])])
        assert f['pixels'].shape == (int(lengths.sum()), args.image_size, args.image_size, 3)
        assert f['pixels'].dtype == np.uint8
        for offset, length in zip(offsets, lengths):
            actions = f['action'][offset:offset+length-1]
            assert np.isfinite(actions).all() and (np.abs(actions) <= 1.00001).all()
        for key in ('qpos', 'qvel', 'physics_state'):
            assert f[key].shape[0] == int(lengths.sum())
            assert np.isfinite(f[key][:]).all()
        pixels = f['pixels'][0]
        assert float(pixels.std()) > 5, 'Blank rendering'
        positions = f['qpos'][:]
        movement = float(np.max(np.ptp(positions, axis=0)))
        assert movement > 0.01, 'No state movement'
        cube_movement = None
        if args.env == 'cube':
            cube_movement = float(np.max(np.ptp(f['privileged_block_0_pos'][:], axis=0)))
            assert cube_movement > 0.02, 'Cube oracle did not move the object'
        columns = list(f.keys())
    reader = HDF5Dataset(path=path, frameskip=5, num_steps=4,
                        keys_to_load=['pixels', 'action', 'qpos', 'qvel'])
    sample = reader[0]
    assert sample['pixels'].shape == (4, 3, args.image_size, args.image_size)
    assert torch.isfinite(sample['action']).all()
    if reader.h5_file is not None:
        reader.h5_file.close()
    Image.fromarray(pixels).save(args.out / 'preview.png')
    permutation = np.random.default_rng(0).permutation(args.episodes)
    n_train = int(args.episodes * 0.9)
    n_validation = int(args.episodes * 0.05)
    splits = {k: sorted(map(int, v)) for k, v in (
        ('train', permutation[:n_train]),
        ('validation', permutation[n_train:n_train+n_validation]),
        ('test', permutation[n_train+n_validation:]))}
    (args.out/'episode_splits.json').write_text(json.dumps({
        'split_seed': 0, 'unit': 'complete episode', 'episode_indices': splits,
        'counts': {k: len(v) for k,v in splits.items()},
    }, indent=2), encoding='utf-8')
    report = {
        'status': 'local collection checked; server compatibility and model training pending',
        'config': {**vars(args), 'out': str(args.out)},
        'condition': 'baseline only; no continual-learning drift conditions collected',
        'environment_id': environment_id,
        'source_revision': '21446f1ede6d5284e981bd7b47f432b994e6d812',
        'versions': {k: importlib.metadata.version(k) for k in
                     ('stable-worldmodel', 'ogbench', 'mujoco', 'dm-control', 'torch')},
        'policy': policy_description,
        'image': f'single view {args.image_size}x{args.image_size} RGB, lossless {args.compression}',
        'episodes': int(len(lengths)), 'frames': int(lengths.sum()),
        'collection_seconds': elapsed, 'file_bytes': path.stat().st_size,
        'frames_per_second': float(lengths.sum()/elapsed),
        'qpos_max_range': movement, 'cube_position_max_range': cube_movement,
        'replay': replay,
        'navigation_targets_reached': getattr(policy, 'targets_reached', None),
        'training_sample_shapes': {k: list(v.shape) for k,v in sample.items()},
        'column_renames': writer.key_mapping, 'columns': columns,
    }
    (args.out/'collection_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[COLLECT_CHECK_OK]', json.dumps(report, ensure_ascii=False), flush=True)
    print('[COLLECT_DONE]', args.out/'collection_summary.json', flush=True)


if __name__ == '__main__':
    main()
