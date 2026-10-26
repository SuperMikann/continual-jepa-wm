"""漂移检测回归检查：CPU 合成数据，不加载真实权重、不产生实验成绩。

运行：python -m unittest discover -s cl_protocol/test -p test_drift_trigger.py -v
"""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drift_trigger as drift
import pred_error as pred


class TinyDataset:
    column_names = ['action', 'pixels']
    _schema_names = ['episode_idx', 'step_idx']

    def __init__(self, action=3.0, length=6):
        self.action = action
        self.length = length
        self.bad_chunks = False

    def get_col_data(self, key):
        return {
            'episode_idx': np.repeat([0, 1], self.length),
            'step_idx': np.tile(np.arange(self.length), 2),
            'action': np.full((2 * self.length, 1), self.action),
        }[key]

    def load_chunk(self, episodes, starts, ends):
        if self.bad_chunks:
            return []
        result = []
        for start, end in zip(starts, ends):
            size = int(end - start)
            action = torch.full((size, 1), self.action)
            action[-1] = float('nan')  # 未使用的末帧动作不能污染预测
            result.append({'pixels': torch.zeros(size, 1), 'action': action})
        return result


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.predictor = SimpleNamespace(num_frames=2)
        self.seen = []
        self.nan_output = False

    def encode(self, info):
        result = {'emb': info['pixels'] * self.weight}
        if 'action' in info:
            result['act_emb'] = info['action']
            self.seen.append(info['action'].clone())
        return result

    def predict(self, emb, action):
        return emb + action * (float('nan') if self.nan_output else self.weight)


class ManyEpisodes(TinyDataset):
    def __init__(self, action=3., episodes=12):
        super().__init__(action=action)
        self.ids = np.arange(10,10+episodes*10,10)

    def get_col_data(self,key):
        if key=='episode_idx':
            return np.repeat(self.ids,self.length)
        if key=='step_idx':
            return np.tile(np.arange(self.length),len(self.ids))
        return np.full((len(self.ids)*self.length,1),self.action)

    def load_chunk(self,episodes,starts,ends):
        assert all(0 <= int(e) < len(self.ids) for e in episodes), 'load_chunk expects episode ordinals'
        return super().load_chunk(episodes,starts,ends)


class DriftTests(unittest.TestCase):
    def setUp(self):
        # 临时文件跟随项目所在磁盘，避免向系统盘写测试产物。
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(windows=5, batch=2, stride=1, seed=42)
        self.scaler = pred.FixedActionScaler([1.0], [2.0])
        self.model = TinyModel().eval().requires_grad_(False)

    def error(self, dataset, scaler=None):
        with patch.object(pred, 'img_transform', return_value=lambda x: x):
            return pred.task_pred_error(
                self.model, dataset, [0, 1], {0: dataset.length, 1: dataset.length},
                self.args, 'cpu', action_scaler=scaler or self.scaler,
            )

    def test_common_scale_keeps_distribution_shift(self):
        # 同一模型及同一统计：原动作 3 -> 1，变化后 5 -> 2。
        # 新条件单独去均值会把这种变化消掉。
        self.assertEqual(self.error(TinyDataset(3.0)), (1.0, 0.0, 5))
        self.assertEqual(self.error(TinyDataset(5.0)), (4.0, 0.0, 5))
        self.assertEqual(drift.drift_decision(1, 4, 1.15), (4, True))
        np.testing.assert_array_equal(self.scaler.mean, [1.0])

    def test_action_shape_and_nonfinite_rejected(self):
        for actions in (np.zeros((3, 2)), np.array([[np.nan]]), np.array([[np.inf]])):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                self.scaler.transform(actions)

    def test_invalid_stats_rejected(self):
        for mean, scale in (([], []), ([0], [0]), ([0], [-1]), ([0], [np.inf]),
                            ([np.nan], [1]), ([0, 1], [1])):
            with self.subTest(mean=mean, scale=scale), self.assertRaises(ValueError):
                pred.FixedActionScaler(mean, scale)

    def test_decision_boundaries(self):
        self.assertEqual(drift.drift_decision(1, 1.15, 1.15), (1.15, False))
        self.assertEqual(drift.drift_decision(1, 0, 1.15), (0, False))
        for a, b, t in ((0, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 0),
                        (np.nan, 1, 1), (1, np.inf, 1), (1, 1, np.nan),
                        (1e-320, 1e300, 1)):
            with self.subTest(values=(a, b, t)), self.assertRaises(ValueError):
                drift.drift_decision(a, b, t)

    def test_empty_and_short_windows_fail(self):
        with self.assertRaises(ValueError):
            self.error(TinyDataset(length=2))
        empty = TinyDataset()
        empty.bad_chunks = True
        with self.assertRaises(ValueError):
            self.error(empty)

    def test_nonfinite_model_output_fails(self):
        self.model.nan_output = True
        with self.assertRaises(ValueError):
            self.error(TinyDataset())

    def test_sampling_counts_and_repeatability(self):
        one = self.error(TinyDataset())
        two = self.error(TinyDataset())
        self.assertEqual(one, two)
        self.assertEqual(one[-1], 5)  # 包含最后一个非满 batch

    def test_last_episodes_not_claimed_independent(self):
        ids, lengths, metadata = drift.select_episodes(TinyDataset(), 'ref', 'ref', 1)
        self.assertEqual(ids.tolist(), [1])
        self.assertEqual(lengths, {1: 6})
        self.assertEqual(metadata['independence'], 'unverified')

    def manifest(self):
        return {'ref': {'dataset': 'ref', 'evaluation_episodes': [1],
                        'training_episodes': [0], 'source': 'synthetic training inventory'}}

    def test_manifest_disjoint_and_overlap(self):
        manifest = self.manifest()
        ids, _, meta = drift.select_episodes(TinyDataset(), 'ref', 'ref', 200, manifest)
        self.assertEqual(ids.tolist(), [1])
        self.assertFalse(meta['training_manifest_completeness_verified'])
        manifest['ref']['training_episodes'] = [0, 1]
        with self.assertRaisesRegex(ValueError, '重叠'):
            drift.select_episodes(TinyDataset(), 'ref', 'ref', 200, manifest)

    def test_invalid_manifest(self):
        for key, value in (('dataset', 'wrong'), ('evaluation_episodes', [3]),
                           ('evaluation_episodes', [1, 1]), ('source', '')):
            manifest = self.manifest()
            manifest['ref'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                drift.select_episodes(TinyDataset(), 'ref', 'ref', 200, manifest)

    def test_no_implicit_legacy_and_invalid_cli(self):
        base = ['--sequence', 'test', '--policy', 'p', '--ref-task', 'T1', '--probe-task', 'T2']
        for extra in ([], ['--legacy', '--windows', '0'], ['--legacy', '--tau', 'nan'],
                      ['--legacy', '--screen-repeats', '1'], ['--legacy', '--seed', '-1']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                drift.parse_args(base + extra)
        self.assertTrue(drift.parse_args(base + ['--legacy']).legacy)

    def test_output_separates_protocols(self):
        path = self.root / 'history.json'
        path.write_text('[{"s": 1.2}]', encoding='utf-8')
        old = path.read_bytes()
        with self.assertRaises(ValueError):
            drift.save_record(path, {'protocol': 'fixed_action_v2', 's': 1.2})
        self.assertEqual(path.read_bytes(), old)

    def test_full_entrypoint_shared_scaler_and_no_model_change(self):
        stats = self.root / 'stats.json'
        stats.write_text(json.dumps({'policy': 'p', 'mean': [1], 'scale': [2],
                                     'source': 'synthetic deployment stats'}), encoding='utf-8')
        with self.assertRaises(ValueError):
            drift.load_action_stats(stats, 'different_checkpoint')
        config = self.root / 'tasks.yaml'
        config.write_text('sequences:\n  test:\n    tasks:\n      T1: {dataset: ref}\n      T2: {dataset: probe}\n', encoding='utf-8')
        out = self.root / 'result.json'
        datasets = {'ref': TinyDataset(3.0), 'probe': TinyDataset(5.0)}
        native = SimpleNamespace(
            data=SimpleNamespace(load_dataset=lambda name, **kw: datasets[name]),
            wm=SimpleNamespace(utils=SimpleNamespace(load_pretrained=lambda policy: self.model)),
        )
        before = copy.deepcopy(self.model.state_dict())
        with patch.dict(sys.modules, {'stable_worldmodel': native}), \
                patch.object(pred, 'img_transform', return_value=lambda x: x), \
                patch.object(torch.cuda, 'is_available', return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            drift.main(['--tasks-cfg', str(config), '--sequence', 'test', '--policy', 'p',
                        '--ref-task', 'T1', '--probe-task', 'T2', '--stride', '1',
                        '--windows', '5', '--batch', '2', '--action-stats', str(stats),
                        '--out', str(out)])
        result = json.loads(out.read_text(encoding='utf-8'))[0]
        self.assertEqual(result['s'], 4.0)
        self.assertTrue(result['decision_only'])
        self.assertEqual(result['n_ref'], 5)
        self.assertEqual(result['action_normalization']['scale'], [2.0])
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(self.model.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def screen_fixture(self):
        home = self.root / 'home'
        for name in ('ref', 'probe'):
            (home/'datasets'/name).mkdir(parents=True)
        (home/'checkpoints').mkdir()
        (home/'checkpoints/p').write_text('synthetic checkpoint marker')
        stats = self.root/'screen_stats.json'
        stats.write_text(json.dumps(dict(policy='p', mean=[1], scale=[2], source='synthetic fixture')))
        config = self.root/'screen_tasks.yaml'
        config.write_text('sequences:\n  test:\n    tasks:\n      T1: {dataset: ref}\n      T2: {dataset: probe}\n')
        datasets = dict(ref=TinyDataset(3.), probe=TinyDataset(5.))
        loads = []
        def load_dataset(name, **kw):
            loads.append((name, kw))
            return datasets[Path(name).name]
        native = SimpleNamespace(
            data=SimpleNamespace(utils=SimpleNamespace(get_cache_dir=lambda sub_folder:home/sub_folder),
                                 load_dataset=load_dataset),
            wm=SimpleNamespace(utils=SimpleNamespace(load_pretrained=lambda path:self.model)))
        out = self.root/'screen.json'
        command = ['--tasks-cfg', str(config), '--sequence', 'test', '--policy', 'p',
                   '--ref-task', 'T1', '--probe-task', 'T2', '--action-stats', str(stats),
                   '--screen-repeats', '3', '--windows', '5', '--batch', '2', '--stride', '1',
                   '--out', str(out)]
        return native, datasets, loads, out, command

    def test_screen_distinct_control_samples_and_single_dataset_load(self):
        native, datasets, loads, out, command = self.screen_fixture()
        seeds, progresses = [], []
        actual = drift.task_pred_error
        def track(model, dataset, episodes, lengths, args, device, **kwargs):
            seeds.append(args.seed)
            callback = kwargs['progress_callback']
            kwargs['progress_callback'] = lambda n,total:(progresses.append(n), callback(n,total))
            return actual(model,dataset,episodes,lengths,args,device,**kwargs)
        before = copy.deepcopy(self.model.state_dict())
        with patch.dict(sys.modules, {'stable_worldmodel':native}), \
                patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(pred, 'img_transform', return_value=lambda x:x), \
                patch.object(drift, 'task_pred_error', side_effect=track), \
                contextlib.redirect_stdout(io.StringIO()):
            result = drift.main(command)
        self.assertEqual(seeds, [0,1,1,2,3,3,4,5,5])
        self.assertEqual(len(loads), 2)
        self.assertTrue(all(x[1]['keys_to_cache'] == ['action'] for x in loads))
        self.assertEqual(progresses, [2,4,5]*9)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['model_unchanged'])
        self.assertFalse(result['threshold_validated'])
        self.assertEqual(result['groups']['same_condition']['triggered_count'], 0)
        self.assertEqual(result['groups']['changed_condition']['triggered_count'], 3)
        self.assertEqual(len(result['rows']), 6)
        self.assertEqual(json.loads(out.read_text(encoding='utf-8'))['status'], 'complete')
        self.assertTrue(all(r['ref_seed'] != r['probe_seed'] for r in result['rows']))
        self.assertTrue(all(m['independence']=='unverified' for m in result['data_selection'].values()))
        for name,value in self.model.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        saved = out.read_bytes()
        with patch.dict(sys.modules, {'stable_worldmodel':native}), self.assertRaises(FileExistsError):
            drift.main(command)
        self.assertEqual(out.read_bytes(), saved)

    def test_screen_nonfinite_error_preserves_failed_record(self):
        native, datasets, loads, out, command = self.screen_fixture()
        self.model.nan_output = True
        with patch.dict(sys.modules, {'stable_worldmodel':native}), \
                patch.object(torch.cuda,'is_available',return_value=False), \
                patch.object(pred,'img_transform',return_value=lambda x:x), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            drift.main(command)
        self.assertEqual(json.loads(out.read_text(encoding='utf-8'))['status'], 'failed')

    def test_screen_excludes_short_episodes(self):
        native, datasets, loads, out, command = self.screen_fixture()
        original_select = drift.select_episodes
        def select(*args, **kwargs):
            episodes,lengths,meta = original_select(*args, **kwargs)
            lengths[0] = 2  # span=2: only episode 1 is eligible.
            return episodes,lengths,meta
        with patch.dict(sys.modules, {'stable_worldmodel':native}), \
                patch.object(torch.cuda,'is_available',return_value=False), \
                patch.object(pred,'img_transform',return_value=lambda x:x), \
                patch.object(drift,'select_episodes',side_effect=select), \
                contextlib.redirect_stdout(io.StringIO()):
            result = drift.main(command)
        for item in result['data_selection'].values():
            self.assertEqual(item['episode_ids'],[1])
            self.assertEqual(item['excluded_short_episode_ids'],[0])

    def test_calibration_quantile_and_invalid_inputs(self):
        scores = np.arange(1,41,dtype=float)
        result = drift.calibrated_threshold(scores,.05)
        self.assertEqual(result['rank'],39)
        self.assertEqual(result['tau'],39.)
        self.assertEqual(drift.calibrated_threshold([.1,.2,.3,.4],.25)['tau'],1.)
        for values,alpha in (([],.05),([1,np.nan],.5),([-1,2],.5),([1,2],.05),([1,2],0)):
            with self.subTest(values=values,alpha=alpha),self.assertRaises(ValueError):
                drift.calibrated_threshold(values,alpha)

    def test_calibration_pools_disjoint_and_prior_episodes_excluded(self):
        data = ManyEpisodes()
        pools,lengths = drift.calibration_pools(data,dict(reference=2,calibration=2,validation=2),2,42,[10,20])
        ids = [item for pool in pools.values() for item in pool['episode_ids']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(set(ids).isdisjoint([10,20]))
        self.assertTrue(all(length==6 for length in lengths.values()))
        with self.assertRaises(ValueError):
            drift.calibration_pools(data,dict(reference=20),2,42)

    def calibration_fixture(self):
        native, _, loads, out, command = self.screen_fixture()
        datasets = dict(ref=ManyEpisodes(3.),probe=ManyEpisodes(5.))
        def load_dataset(name,**kwargs):
            loads.append((name,kwargs))
            return datasets[Path(name).name]
        native.data.load_dataset = load_dataset
        i = command.index('--screen-repeats')
        del command[i:i+2]
        prior = self.root/'prior.json'
        prior.write_text(json.dumps(dict(status='complete',protocol='fixed_action_v2_screen',policy='p',
            data_selection={role:dict(dataset=name,episode_ids=[10,20]) for role,name in (('ref','ref'),('probe','probe'))},
            rows=[dict(s=999999)])))  # Prior scores must not set the new threshold.
        command += ['--calibrate','--calibration-batches','4','--validation-batches','3',
                    '--reference-windows','5','--pool-episodes','2','--target-fpr','.25',
                    '--exclude-screen-record',str(prior)]
        return native,datasets,loads,out,command

    def test_calibration_full_entrypoint_freezes_before_validation(self):
        native,datasets,loads,out,command = self.calibration_fixture()
        order = []
        actual = drift.task_pred_error
        def track(model,dataset,*args,**kwargs):
            if dataset is datasets['probe']:
                saved = json.loads(out.read_text(encoding='utf-8'))
                self.assertTrue(saved['threshold_calibrated'])
                self.assertEqual(saved['calibration']['tau'],1.)
                order.append('probe')
            else:
                order.append('ref')
            return actual(model,dataset,*args,**kwargs)
        with patch.dict(sys.modules,{'stable_worldmodel':native}), \
                patch.object(torch.cuda,'is_available',return_value=False), \
                patch.object(pred,'img_transform',return_value=lambda x:x), \
                patch.object(drift,'task_pred_error',side_effect=track), \
                contextlib.redirect_stdout(io.StringIO()):
            result = drift.main(command)
        self.assertEqual(order,['ref']*5+['ref','probe']*3)
        self.assertEqual(result['status'],'complete')
        self.assertTrue(result['model_unchanged'])
        self.assertTrue(result['threshold_calibrated'])
        self.assertFalse(result['threshold_validated'])
        self.assertEqual(result['calibration']['tau'],1.)
        self.assertEqual(result['excluded_episode_counts'],dict(ref=2,probe=2))
        ref_ids = [e for name,pool in result['episode_pools'].items() if name!='changed_validation'
                   for e in pool['episode_ids']]
        self.assertEqual(len(ref_ids),len(set(ref_ids)))
        self.assertTrue(set(ref_ids).isdisjoint([10,20]))
        self.assertEqual(result['groups']['same_condition']['triggered_count'],0)
        self.assertEqual(result['groups']['changed_condition']['triggered_count'],3)
        self.assertEqual(len(loads),2)
        saved = out.read_bytes()
        with patch.dict(sys.modules,{'stable_worldmodel':native}),self.assertRaises(FileExistsError):
            drift.main(command)
        self.assertEqual(out.read_bytes(),saved)

    def test_changed_validation_cannot_choose_threshold(self):
        native,datasets,loads,out,command = self.calibration_fixture()
        def run():
            with patch.dict(sys.modules,{'stable_worldmodel':native}), \
                    patch.object(torch.cuda,'is_available',return_value=False), \
                    patch.object(pred,'img_transform',return_value=lambda x:x), \
                    contextlib.redirect_stdout(io.StringIO()):
                return drift.main(command)
        first = run()
        datasets['probe'].action = 300.
        command[command.index('--out')+1] = str(self.root/'second_calibration.json')
        second = run()
        self.assertEqual(first['calibration'],second['calibration'])
        self.assertNotEqual(first['groups']['changed_condition']['score_min'],second['groups']['changed_condition']['score_min'])

    def test_calibration_invalid_cli_and_failure_record(self):
        native,datasets,loads,out,command = self.calibration_fixture()
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            drift.parse_args(command+['--screen-repeats','1'])
        without_calibration = [arg for arg in command if arg!='--calibrate']
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            drift.parse_args(without_calibration+['--previous-calibration',str(out)])
        bad = list(command)
        bad[bad.index('--target-fpr')+1] = '.001'
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            drift.parse_args(bad)
        self.model.nan_output=True
        with patch.dict(sys.modules,{'stable_worldmodel':native}), \
                patch.object(torch.cuda,'is_available',return_value=False), \
                patch.object(pred,'img_transform',return_value=lambda x:x), \
                contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):
            drift.main(command)
        self.assertEqual(json.loads(out.read_text(encoding='utf-8'))['status'],'failed')

    def test_512_windows_fresh_normal_pools_and_explicit_changed_pool_reuse(self):
        native,datasets,loads,out,command = self.calibration_fixture()
        datasets['ref'] = ManyEpisodes(3.,episodes=20)
        with patch.dict(sys.modules,{'stable_worldmodel':native}), \
                patch.object(torch.cuda,'is_available',return_value=False), \
                patch.object(pred,'img_transform',return_value=lambda x:x), \
                contextlib.redirect_stdout(io.StringIO()):
            first = drift.main(command)
            saved = out.read_bytes()
            second_out = self.root/'larger_windows.json'
            second_cmd = list(command)
            for flag,value in (('--out',str(second_out)),('--windows','512'),
                               ('--reference-windows','512'),('--batch','32')):
                second_cmd[second_cmd.index(flag)+1] = value
            second_cmd += ['--previous-calibration',str(out),'--seed','123']
            second = drift.main(second_cmd)
        self.assertEqual(out.read_bytes(),saved)
        self.assertEqual(second['status'],'complete')
        self.assertTrue(second['normal_pools_disjoint_from_previous'])
        self.assertTrue(second['changed_validation_reused'])
        self.assertTrue(second['model_unchanged'])
        self.assertEqual(second['changed_pool_seed'],first['seed']+1)
        self.assertEqual(second['reference']['n'],512)
        self.assertTrue(all(row['n']==512 for row in second['calibration_rows']+second['validation_rows']))
        before_ids = {e for name,pool in first['episode_pools'].items() if name!='changed_validation'
                      for e in pool['episode_ids']}
        after_ids = [e for name,pool in second['episode_pools'].items() if name!='changed_validation'
                     for e in pool['episode_ids']]
        self.assertTrue(before_ids.isdisjoint(after_ids))
        self.assertEqual(len(after_ids),len(set(after_ids)))
        self.assertEqual(second['excluded_episode_counts']['ref'],8)
        self.assertEqual(second['episode_pools']['changed_validation'],first['episode_pools']['changed_validation'])
        self.assertEqual(second['groups']['same_condition']['count'],3)
        self.assertEqual(second['groups']['changed_condition']['count'],3)
        self.assertEqual(len(loads),4)

    def test_previous_calibration_mismatch_stops_without_inference(self):
        native,datasets,loads,out,command = self.calibration_fixture()
        datasets['ref'] = ManyEpisodes(3.,episodes=20)
        with patch.dict(sys.modules,{'stable_worldmodel':native}), \
                patch.object(torch.cuda,'is_available',return_value=False), \
                patch.object(pred,'img_transform',return_value=lambda x:x), \
                contextlib.redirect_stdout(io.StringIO()):
            first = drift.main(command)
            alternatives = []
            for key,value in (('policy','another'),('stride',2),('status','failed')):
                item = copy.deepcopy(first)
                item[key] = value
                alternatives.append(item)
            changed = copy.deepcopy(first)
            available = set(datasets['probe'].ids.tolist())-{10,20}
            old = set(first['episode_pools']['changed_validation']['episode_ids'])
            changed['episode_pools']['changed_validation']['episode_ids'] = sorted(available-old)[:2]
            alternatives.append(changed)
            for i,item in enumerate(alternatives):
                with self.subTest(i=i):
                    previous = self.root/f'bad_previous_{i}.json'
                    previous.write_text(json.dumps(item),encoding='utf-8')
                    new_out = self.root/f'failed_{i}.json'
                    new_command = list(command)
                    new_command[new_command.index('--out')+1] = str(new_out)
                    new_command += ['--previous-calibration',str(previous),'--seed','123']
                    with patch.object(drift,'task_pred_error') as measure,self.assertRaises(ValueError):
                        drift.main(new_command)
                    measure.assert_not_called()
                    self.assertEqual(json.loads(new_out.read_text(encoding='utf-8'))['status'],'failed')


if __name__ == '__main__':
    unittest.main()
