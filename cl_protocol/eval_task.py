"""多任务评估脚本：把 tasks.yaml 的漂移注入官方 eval_wm.py，支持单格与矩阵模式。

原理（不修改官方脚本的一行代码）：
    在运行官方 scripts/plan/eval_wm.py 之前，把 swm.World 替换成
    DriftWorld（见 drift_world.py），让每一次 reset 自动带上任务的
    variation_values / 暗光变换。官方脚本的模型加载、CEM 规划、
    callables、结果落盘全部原样复用 —— 84% 的评估管线一字不动。

用法：
    # 单格评估（T1 种子在 T2 上的表现 = zero-shot 标定）
    python eval_task.py --sequence B_reacher --task T2 \
        --policy lewm_reacher/weights_epoch_30.pt --num-eval 50

    # 矩阵模式（遗忘曲线数据：多个检查点 × 多个任务）
    python eval_task.py --sequence B_reacher --matrix \
        --policies lewm_reacher/weights_epoch_30.pt lewm_reacher_t2/weights_epoch_10.pt

验证：先跑 --sequence B_reacher --task T1 --policy lewm_reacher/weights_epoch_30.pt
     应复现 ~84%（与 dmc_results.txt 里已有结果一致），一致即说明注入机制无副作用。

注意：在仓库根目录运行（或 --eval-wm 指定 eval_wm.py 路径）；
     DMC 环境变量同采集（PYOPENGL_PLATFORM=osmesa / LD_PRELOAD / STABLEWM_HOME），
     服务器上的 eval_wm.py 应已把 MUJOCO_GL 改为 osmesa。
"""

import argparse
import functools
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

import stable_worldmodel as swm

from drift_world import DriftWorld, make_darken


class ManifestWorld(DriftWorld):
    """Use an explicit held-out scene list, bypassing native random scene selection."""
    def __init__(self, *args, scene_manifest, scene_result, **kwargs):
        super().__init__(*args, **kwargs)
        self.scene_manifest = json.loads(Path(scene_manifest).read_text())
        self.scene_result = Path(scene_result)

    def evaluate(self, *args, **kwargs):
        manifest = self.scene_manifest
        dataset = kwargs['dataset']
        if list(map(int, dataset.lengths)) != manifest['episode_lengths']:
            raise ValueError('Evaluation dataset does not match scene manifest')
        if (kwargs['goal_offset'] != manifest['goal_offset'] or
                kwargs['eval_budget'] != manifest['eval_budget']):
            raise ValueError('Evaluation horizon/budget changed')
        scenes = manifest['scenes']
        if len(scenes) != self.num_envs:
            raise ValueError('Scene count differs from environment count')
        kwargs['episodes_idx'] = [s['episode_idx'] for s in scenes]
        kwargs['start_steps'] = [s['start_step'] for s in scenes]
        result = super().evaluate(*args, **kwargs)
        successes = [bool(x) for x in result['episode_successes']]
        if len(successes) != len(scenes):
            raise ValueError('Missing per-scene results')
        receipt = dict(status='complete', scenes=scenes, successes=successes,
                       success_rate=100 * sum(successes) / len(successes),
                       goal_offset=manifest['goal_offset'], eval_budget=manifest['eval_budget'])
        self.scene_result.write_text(json.dumps(receipt, indent=2) + '\n')
        return result


def load_seq(tasks_path, seq_name):
    with open(tasks_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['sequences'][seq_name]


def results_file_for(seq, policy):
    """复现官方脚本的结果落盘路径：<checkpoints>/<policy>/../<results_file>"""
    ckpt_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))
    return ckpt_dir / policy / '..' / seq['results_file']


def read_fresh_success_rate(path, offset, args, dataset):
    """Validate the result appended by this invocation, never an older score."""
    path = Path(path).resolve()
    with path.open('rb') as f:
        f.seek(offset)
        text = f.read().decode('utf-8')
    blocks = text.split('==== CONFIG ====')
    if len(blocks) != 2 or '==== RESULTS ====' not in blocks[1]:
        raise RuntimeError('本次评估未写入唯一完整的新结果，拒绝读取历史成绩')
    config_text, result = blocks[1].split('==== RESULTS ====', 1)
    cfg = yaml.safe_load(config_text)
    canonical = lambda p: str(p).replace('\\', '/').split('/checkpoints/')[-1]
    actual = (canonical(cfg['policy']), cfg['eval']['dataset_name'],
              cfg['eval']['num_eval'], cfg['seed'])
    expected = (canonical(args.policy), dataset, args.num_eval, args.seed)
    if actual != expected:
        raise RuntimeError(f'新结果配置不符: {actual!r} != {expected!r}')
    hits = re.findall(r"'success_rate':\s*([\d.eE+-]+)", result)
    flags = re.search(r"'episode_successes':\s*array\(\[([\s\S]*?)\]", result)
    if len(hits) != 1 or flags is None:
        raise RuntimeError('新结果缺少唯一成功率或逐回合成功记录')
    success = float(hits[0])
    episodes = re.findall(r'\b(?:True|False)\b', flags.group(1))
    if (len(episodes) != args.num_eval or not 0 <= success <= 100 or
            abs(success - 100 * episodes.count('True') / args.num_eval) > 1e-7):
        raise RuntimeError('新结果回合数或成功率与逐回合记录不一致')
    return success


def install_model_protocol(args):
    """给完整旧通路诊断显式选条件，并锁定模型实际使用的动作坐标。

    在原生 WorldModelPolicy 构造时替换 process['action']；起点/目标不变。
    返回恢复函数与运行收据，避免在当前进程遗留全局替换。
    """
    native_load = swm.wm.utils.load_pretrained
    native_policy = swm.policy.WorldModelPolicy
    condition = getattr(args, 'condition_branch', None)
    stats_path = getattr(args, 'action_stats', None)
    state = {'condition': condition, 'normalization': None}
    fixed = None
    if stats_path:
        from stable_worldmodel.wm.lewm.module import FixedActionNormalizer
        data = json.loads(Path(stats_path).read_text(encoding='utf-8'))
        if not data.get('source') or not data.get('policy'):
            raise ValueError('Action statistics require source and policy provenance')
        fixed = FixedActionNormalizer(data['mean'], data['scale'])
        state['normalization'] = data

    def load_model(*a, **kw):
        nonlocal fixed
        model = native_load(*a, **kw)
        if hasattr(model, 'set_condition') and hasattr(model, 'action_normalizer'):
            if condition is None:
                raise ValueError('Known-condition diagnostic requires --condition-branch old/new')
            if fixed is not None:
                raise ValueError('Diagnostic checkpoint already contains fixed action statistics')
            model.set_condition(condition)
            fixed = model.action_normalizer
            state['normalization'] = {
                'source': model.normalization_source,
                'mean': fixed.mean.tolist(), 'scale': fixed.scale.tolist(),
                'embedded_in_checkpoint': True,
            }
        elif condition is not None:
            raise ValueError('condition-branch only applies to the complete-old-path diagnostic')
        return model

    def make_policy(*a, **kw):
        if fixed is not None:
            kw['process'] = {**kw.get('process', {}), 'action': fixed}
        print('EVAL_MODEL_PROTOCOL ' + json.dumps(state, ensure_ascii=False), flush=True)
        return native_policy(*a, **kw)

    def restore():
        swm.wm.utils.load_pretrained = native_load
        swm.policy.WorldModelPolicy = native_policy

    swm.wm.utils.load_pretrained = load_model
    swm.policy.WorldModelPolicy = make_policy
    return restore, state


def run_cell(args, task_name):
    """单格评估：monkey-patch 后在当前进程内执行官方 eval_wm.py。"""
    import runpy

    seq = load_seq(args.tasks, args.sequence)
    # 链任务 + 探针档位（probe_tasks）合并查找
    all_tasks = {**seq['tasks'], **seq.get('probe_tasks', {})}
    if task_name not in all_tasks:
        sys.exit(f'[拒绝] 未知任务 {task_name}，可用: {sorted(all_tasks)}')
    task = all_tasks[task_name]
    vv = task.get('variation_values') or {}
    darken = task.get('darken', 1.0)

    print(f'[评估] {args.sequence}/{task_name} <- {args.policy}')
    print(f'[漂移] variation_values={vv}, darken={darken}')

    # —— 核心：替换 swm.World，官方脚本随后创建的 World 全部带漂移 ——
    scene_manifest = getattr(args, 'scene_manifest', None)
    scene_result = getattr(args, 'scene_result', None)
    if bool(scene_manifest) != bool(scene_result):
        raise ValueError('Provide both scene-manifest and scene-result')
    swm.World = functools.partial(
        ManifestWorld if scene_manifest else DriftWorld,
        **(dict(scene_manifest=scene_manifest, scene_result=scene_result) if scene_manifest else {}),
        variation_values=vv,
        image_transform=make_darken(darken),
    )

    # 数据集覆盖同时改变初始状态与目标图像。未固定action-stats时还改变
    # 动作归一化，不能将其成绩变化直接归因于目标图像这个单一因素。
    eval_dataset = args.eval_dataset or task['dataset']
    if scene_manifest and json.loads(Path(scene_manifest).read_text())['dataset'] != eval_dataset:
        raise ValueError('Scene manifest dataset differs from evaluation dataset')
    if eval_dataset != task['dataset']:
        print(f'[审计] 数据集覆盖: {task["dataset"]} -> {eval_dataset}（漂移不变）')

    path = results_file_for(seq, args.policy).resolve()
    result_offset = path.stat().st_size if path.exists() else 0
    sys.argv = [
        'eval_wm.py',
        '--config-name', seq['eval_config'],
        f'policy={args.policy}',
        f'eval.dataset_name={eval_dataset}',
        f'eval.num_eval={args.num_eval}',
        f'seed={args.seed}',
        *(args.extra or []),
    ]
    restore, model_protocol = install_model_protocol(args)
    try:
        runpy.run_path(str(Path(args.eval_wm).resolve()), run_name='__main__')
    finally:
        restore()

    sr = read_fresh_success_rate(path, result_offset, args, eval_dataset)
    record = {
        'task': task_name, 'policy': args.policy, 'success_rate': sr,
        'dataset': eval_dataset, 'num_eval': args.num_eval, 'seed': args.seed,
        'variation_values': vv, 'darken': darken,
        'result_path': str(path), 'result_start_offset': result_offset,
        'model_protocol': model_protocol,
    }
    print(f'CELL_RESULT {json.dumps(record)}', flush=True)
    return sr


def run_matrix(args):
    """矩阵模式：每个 (policy × task) 格用子进程跑（hydra 不可重入，必须隔离进程）。"""
    seq = load_seq(args.tasks, args.sequence)
    task_names = args.tasks_sel or list(seq['tasks'].keys())
    matrix = {}

    for policy in args.policies:
        matrix[policy] = {}
        for task_name in task_names:
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                '--sequence', args.sequence,
                '--task', task_name,
                '--policy', policy,
                '--num-eval', str(args.num_eval),
                '--seed', str(args.seed),
                '--tasks', args.tasks,
                '--eval-wm', args.eval_wm,
            ]
            for kv in (args.extra or []):
                cmd += ['--extra', kv]
            if args.eval_dataset is not None:
                cmd += ['--eval-dataset', args.eval_dataset]
            if args.condition_branch is not None:
                cmd += ['--condition-branch', args.condition_branch]
            if args.action_stats is not None:
                cmd += ['--action-stats', args.action_stats]
            print(f'\n===== 评估格: {policy} @ {task_name} =====')
            records = []
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace',
                                  bufsize=1) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    if line.startswith('CELL_RESULT '):
                        try:
                            records.append(json.loads(line[len('CELL_RESULT '):]))
                        except json.JSONDecodeError:
                            records.append({})
                ret = process.wait()
            if ret != 0:
                print(f'[警告] 该格子进程退出码 {ret}，记为 None')
                matrix[policy][task_name] = None
                continue
            record = records[0] if len(records) == 1 else {}
            if (record.get('task') == task_name and record.get('policy') == policy
                    and record.get('num_eval') == args.num_eval
                    and record.get('seed') == args.seed
                    and isinstance(record.get('success_rate'), (int, float))
                    and 0 <= record['success_rate'] <= 100):
                sr = record['success_rate']
            else:
                print('[警告] 子进程缺少唯一且匹配本格的CELL_RESULT，记为 None')
                sr = None
            matrix[policy][task_name] = sr

    suffix = f'_{args.condition_branch}' if args.condition_branch else ''
    out = Path(f'forgetting_matrix_{args.sequence}{suffix}.json').resolve()
    out.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding='utf-8')

    # 打印表格（行=评估任务，列=检查点）
    policies = list(matrix.keys())
    header = ['task'] + [Path(p).parent.name for p in policies]
    rows = []
    for t in task_names:
        rows.append([t] + [
            f'{matrix[p][t]:.1f}' if matrix[p][t] is not None else 'FAIL'
            for p in policies
        ])
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda r: '  '.join(str(x).ljust(w) for x, w in zip(r, widths))
    print('\n===== 遗忘矩阵（success_rate %） =====')
    print(line(header))
    for r in rows:
        print(line(r))
    print(f'\n[完成] 矩阵已保存: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=str(Path(__file__).parent / 'tasks.yaml'))
    ap.add_argument('--sequence', required=True, choices=None)  # 序列名以 tasks.yaml 为准（A_pusht/B_reacher/C_walker/D_cheetah）
    ap.add_argument('--task', help='单格模式：评估哪个任务（T1/T2/T3 或探针档位）')
    ap.add_argument('--policy', help='单格模式：检查点（checkpoints/ 下的相对路径）')
    ap.add_argument('--matrix', action='store_true', help='矩阵模式')
    ap.add_argument('--policies', nargs='+', help='矩阵模式：检查点列表')
    ap.add_argument('--tasks-sel', nargs='+',
                    help='矩阵模式：只评这些任务（默认全部链任务，可选探针档位）')
    ap.add_argument('--num-eval', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-wm', default='scripts/plan/eval_wm.py',
                    help='官方评估脚本路径（默认假设在仓库根目录运行）')
    ap.add_argument('--extra', action='append',
                    help='追加给 eval_wm.py 的 hydra 覆盖，可多次：--extra bf16=true')
    ap.add_argument('--eval-dataset', default=None,
                    help='审计用：覆盖评估数据集（初始状态/goal 来源），环境漂移不变')
    ap.add_argument('--condition-branch', choices=('old', 'new'),
                    help='完整旧通路诊断专用：显式给定条件，不是自动识别')
    ap.add_argument('--action-stats', help='为普通模型固定动作坐标的JSON；诊断模型使用权重内统计')
    ap.add_argument('--scene-manifest', help='单格：固定留出轨迹起点清单')
    ap.add_argument('--scene-result', help='单格：保存清单及逐场景成功标记')
    args = ap.parse_args()

    if args.matrix:
        if args.scene_manifest or args.scene_result:
            ap.error('Explicit scene manifests currently require single-cell mode')
        if not args.policies:
            sys.exit('[拒绝] 矩阵模式需要 --policies')
        run_matrix(args)
    else:
        if not args.task or not args.policy:
            sys.exit('[拒绝] 单格模式需要 --task 和 --policy')
        run_cell(args, args.task)


if __name__ == '__main__':
    main()
