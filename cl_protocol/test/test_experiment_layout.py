"""CPU regression checks for deployment, planning time alignment and replay workers."""
from collections import deque
import multiprocessing
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from implementation.install import install
from implementation.planning_history import prepare_planning_history, future_action_rollout
from reacher.training import ReplayWindows, OldMemoryWindows


class Numbers:
    def __init__(self, offset):
        self.offset = offset

    def __len__(self):
        return 20

    def __getitem__(self, index):
        return {'value': self.offset + index}


def replay_child(dataset, connection):
    try:
        connection.send(dataset.__getitems__(list(range(8))))
    finally:
        connection.close()


class PlanningTests(unittest.TestCase):
    def test_executed_blocks_and_episode_flush(self):
        policy = SimpleNamespace(cfg=SimpleNamespace(history_len=3, action_block=2),
                                 env=SimpleNamespace(num_envs=1), _action_buffer=[deque()])
        for step in range(5):
            info = {'pixels': torch.tensor([[[float(step)]]]),
                    'action': torch.tensor([[[float(step)]]])}
            result = prepare_planning_history(policy, info)
        self.assertEqual(result['pixels'].flatten().tolist(), [0., 2., 4.])
        self.assertEqual(result['action_history'].tolist(), [[[1., 2.], [3., 4.]]])
        reset = prepare_planning_history(policy, {'pixels': torch.tensor([[[99.]]]),
                                                  'action': torch.tensor([[[0.]]])}, [True])
        self.assertEqual(reset['pixels'].flatten().tolist(), [99.])
        self.assertNotIn('action_history', reset)

    def test_future_actions_are_executed_once(self):
        class Integrator:
            predictor = SimpleNamespace(num_frames=3)
            action_encoder = torch.nn.Identity()

            def encode(self, info):
                return dict(info, emb=info['pixels'])

            def predict(self, emb, act_emb):
                return emb + act_emb

        model = Integrator()
        actions = torch.arange(1., 6.).reshape(1, 1, 5, 1)
        for context in ([30.], [0., 10., 30.]):
            h = len(context)
            info = {'pixels': torch.tensor(context).reshape(1, 1, h, 1)}
            if h == 3:
                info['action_history'] = torch.tensor([10., 20.]).reshape(1, 1, 2, 1)
            result = future_action_rollout(model, info, actions)['predicted_emb']
            self.assertEqual(result.shape[2], h + 5)
            self.assertEqual(result[0, 0, -5:, 0].tolist(), [31., 33., 36., 40., 45.])
        with self.assertRaises(ValueError):
            future_action_rollout(model, {'pixels': torch.zeros(1, 1, 3, 1)}, actions)


class DeploymentTests(unittest.TestCase):
    def checkout(self, root):
        files = {
            'stable_worldmodel/wm/lewm/module.py': '# old module\n',
            'scripts/train/lewm.py': '# old trainer\n',
            'stable_worldmodel/policy.py': '''class WorldModelPolicy:
    def set_env(self, env):
        self.env = env
    def get_action(self, info_dict):
        info_dict = self._prepare_info(info_dict)
        return info_dict
''',
            'stable_worldmodel/wm/lewm/lewm.py': '''class LeWM:
    def rollout(self, info, action_sequence, history_size=None):
        H, T = 1, 5
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        n_steps = T - H
        for t in range(n_steps + 1):
            pass
        return info
''',
        }
        for relative, text in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
        return {name: (root / name).read_bytes() for name in files}

    def test_explicit_install_backup_and_second_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = self.checkout(root)
            checked = install(root)
            self.assertEqual(len(checked['changes']), 4)
            self.assertFalse((root / '.code_backups').exists())
            self.assertTrue(all((root / p).read_bytes() == data for p, data in before.items()))
            applied = install(root, apply=True)
            backup = Path(applied['backup'])
            self.assertTrue(all((backup / p).read_bytes() == data for p, data in before.items()))
            self.assertEqual(install(root, apply=True)['changes'], [])

    def test_unknown_source_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.checkout(root)
            path = root / 'stable_worldmodel/wm/lewm/lewm.py'
            path.write_text('class LeWM:\n    def rollout(self): return None\n', encoding='utf-8')
            before = {p: p.read_bytes() for p in root.rglob('*.py')}
            with self.assertRaises(ValueError):
                install(root, apply=True)
            self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
            self.assertFalse((root / '.code_backups').exists())


class ReplayWorkerTests(unittest.TestCase):
    def test_dataset_survives_spawn_and_keeps_replay_fraction(self):
        memory = OldMemoryWindows([Numbers(100), Numbers(200)], [[0, 1], [2, 3]])
        dataset = ReplayWindows(Numbers(0), memory, list(range(len(memory))))
        method = 'forkserver' if 'forkserver' in multiprocessing.get_all_start_methods() else 'spawn'
        context = multiprocessing.get_context(method)
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=replay_child, args=(dataset, send))
        process.start()
        send.close()
        try:
            self.assertTrue(receive.poll(45), 'Replay worker failed to send a batch')
            rows = receive.recv()
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        finally:
            receive.close()
            if process.is_alive():
                process.terminate()
                process.join()
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row['_replay_source'] for row in rows), 2)
        self.assertTrue(all(row['value'] >= 100 for row in rows if row['_replay_source']))


if __name__ == '__main__':
    unittest.main()
