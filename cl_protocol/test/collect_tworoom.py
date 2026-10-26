"""CPU collection using the official Two-Room environment and HDF5 writer.

Default: 10 episodes at each of three speeds. Reuse this entry for later batches;
split complete episodes before constructing training/validation/test windows.
Use nonoverlapping episode seeds when collecting additional held-out batches.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image
import torch
import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset, HDF5Writer
from stable_worldmodel.envs.two_room import ExpertPolicy, TwoRoomEnv


class CompactWriter(HDF5Writer):
    """Keep the native schema/timing; compress RGB losslessly for transfer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._report_start = time.perf_counter()

    def write_episode(self, ep_data):
        super().write_episode(ep_data)
        if self._ep_written == 1 or self._ep_written % 100 == 0:
            self._f.flush()
            print('[COLLECT_PROGRESS]', json.dumps({
                'file': str(self.path), 'episodes': self._ep_written,
                'frames': self._global_ptr,
                'seconds': round(time.perf_counter() - self._report_start, 1),
            }), flush=True)

    def _init_schema(self, sample_ep):
        super()._init_schema(sample_ep)
        pixels = self._f['pixels']
        shape, dtype = pixels.shape, pixels.dtype
        # The native schema has just been created: no frames exist yet.
        assert shape[0] == 0
        del self._f['pixels']
        self._f.create_dataset('pixels', shape=shape, maxshape=(None, *shape[1:]),
                               dtype=dtype, chunks=(1, *shape[1:]), compression='lzf')


def collect(args, speed):
    tag = f"speed_{speed:g}".replace('.', 'p')
    path = args.out / f'{tag}.h5'
    report_path = args.out / f'{tag}.json'
    if path.exists() or report_path.exists():
        raise FileExistsError(f'Refusing to overwrite: {path}')
    options = {
        'variation': ['agent.position', 'target.position', 'agent.speed'],
        'variation_values': {'agent.speed': np.array([speed], np.float32)},
    }
    world = swm.World('swm/TwoRoom-v1', num_envs=1, image_shape=(224, 224),
                      max_episode_steps=args.max_steps)
    policy = ExpertPolicy(action_noise=args.action_noise, seed=args.seed)
    # Upstream 21446f1 accepts this argument but omits assigning the attribute.
    # Disable action repetition explicitly; use the same setting at every speed.
    policy.action_repeat_prob = 0.0
    world.set_policy(policy)
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            world.collect(writer=CompactWriter(path, mode='error'),
                          episodes=args.episodes, seed=args.seed, options=options,
                          progress=False)
    finally:
        world.close()
    elapsed = time.perf_counter() - start

    with h5py.File(path, 'r') as f:
        lengths = f['ep_len'][:]
        offsets = f['ep_offset'][:]
        assert len(lengths) == args.episodes
        assert np.array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])])
        assert f['pixels'].shape[1:] == (224, 224, 3)
        assert f['pixels'].dtype == np.uint8
        assert f['action'].shape[1:] == (2,)
        assert np.allclose(f['variation.agent.speed'][:], speed)
        for offset, length in zip(offsets, lengths):
            actions = f['action'][offset:offset + length - 1]
            assert np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
        eligible = np.flatnonzero(lengths >= 26)
        if len(eligible) == 0:
            raise ValueError('No episode supports a 25-step goal window.')
        first = int(eligible[0])
        offset, length = int(offsets[first]), int(lengths[first])
        states = f['state'][offset:offset + length]
        actions = f['action'][offset:offset + length - 1]
        first_pixels = f['pixels'][offset]
        success = [bool(f['terminated'][int(o + n - 1)])
                   for o, n in zip(offsets, lengths)]
        columns = list(f.keys())

    # Verify the writer's action shift against the official simulator, before
    # interpreting these files as x[t], a[t] -> x[t+1] training trajectories.
    raw = TwoRoomEnv()
    try:
        raw.reset(seed=args.seed, options=options)
        raw._set_state(states[0])
        initial_pixel_mae = float(np.abs(raw.render().astype(float)
                                        - first_pixels.astype(float)).mean())
        max_state_error = 0.0
        for step, action in enumerate(actions[:10], 1):
            _, _, _, _, info = raw.step(action)
            max_state_error = max(max_state_error,
                                  float(np.max(np.abs(info['state'] - states[step]))))
        assert max_state_error < 1e-4, max_state_error
        assert initial_pixel_mae < 1e-6, initial_pixel_mae
    finally:
        raw.close()
    reader = HDF5Dataset(path=path, frameskip=5, num_steps=4,
                         keys_to_load=['pixels', 'action', 'proprio'])
    sample = reader[0]
    assert sample['pixels'].shape == (4, 3, 224, 224)
    assert torch.isfinite(sample['action']).all()
    if reader.h5_file is not None:
        reader.h5_file.close()
    Image.fromarray(first_pixels).save(args.out / f'{tag}.png')
    report = {
        'status': 'local collection checked; server reader and T1 planning pending',
        'config': {**vars(args), 'out': str(args.out), 'speed': speed},
        'environment': 'swm/TwoRoom-v1',
        'stable_worldmodel_version': importlib.metadata.version('stable-worldmodel'),
        'torch_version': torch.__version__,
        'episodes': int(len(lengths)), 'frames': int(lengths.sum()),
        'episodes_with_25_step_goal': int(len(eligible)),
        'episode_lengths_min_mean_max': [int(lengths.min()), float(lengths.mean()),
                                         int(lengths.max())],
        'collection_seconds': elapsed,
        'frames_per_second': float(lengths.sum() / elapsed),
        'file_bytes': path.stat().st_size,
        'pixel_compression': 'lossless HDF5 lzf',
        'noisy_collection_success_fraction': float(np.mean(success)),
        'replay_max_state_error': max_state_error,
        'replay_initial_pixel_mae': initial_pixel_mae,
        'training_sample_shapes': {k: list(v.shape) for k, v in sample.items()},
        'columns': columns,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print('[COLLECT_CHECK_OK]', json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--speeds', type=float, nargs='+', default=[5.0, 3.5, 7.0])
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--max-steps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=3072)
    ap.add_argument('--action-noise', type=float, default=2.0,
                    help='Official collector uses 2.0; unchanged across conditions.')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--source-revision', default='21446f1ede6d5284e981bd7b47f432b994e6d812',
                    help='Installed upstream revision, recorded for provenance.')
    args = ap.parse_args()
    if args.episodes < 1 or args.max_steps < 26 or args.threads < 1:
        ap.error('episodes/threads must be positive and max-steps >= 26')
    if len(set(args.speeds)) != len(args.speeds):
        ap.error('speeds must be distinct')
    torch.set_num_threads(args.threads)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    print('[COLLECT_START]', json.dumps({**vars(args), 'out': str(args.out)}), flush=True)
    rows = [collect(args, speed) for speed in args.speeds]
    summary = args.out / 'collection_summary.json'
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[COLLECT_DONE]', summary, flush=True)


if __name__ == '__main__':
    main()
