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


def load_seq(tasks_path, seq_name):
    with open(tasks_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['sequences'][seq_name]


def results_file_for(seq, policy):
    """复现官方脚本的结果落盘路径：<checkpoints>/<policy>/../<results_file>"""
    ckpt_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))
    return ckpt_dir / policy / '..' / seq['results_file']


def read_last_success_rate(path):
    """从结果文件读最后一次评估的 success_rate（文件是 append 模式）。"""
    text = Path(path).resolve().read_text(encoding='utf-8')
    # metrics dict 里含 numpy 数组（episode_successes/seeds），不能直接 literal_eval，
    # 只抽 success_rate 标量
    hits = re.findall(r"'success_rate': ([\d.eE+-]+)", text)
    if not hits:
        raise RuntimeError(f'结果文件里没有 success_rate: {path}')
    return float(hits[-1])


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
    swm.World = functools.partial(
        DriftWorld,
        variation_values=vv,
        image_transform=make_darken(darken),
    )

    sys.argv = [
        'eval_wm.py',
        '--config-name', seq['eval_config'],
        f'policy={args.policy}',
        f'eval.dataset_name={task["dataset"]}',
        f'eval.num_eval={args.num_eval}',
        f'seed={args.seed}',
        *(args.extra or []),
    ]
    runpy.run_path(str(Path(args.eval_wm).resolve()), run_name='__main__')

    path = results_file_for(seq, args.policy)
    sr = read_last_success_rate(path)
    print(f'CELL_RESULT {json.dumps({"task": task_name, "policy": args.policy, "success_rate": sr})}')
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
            print(f'\n===== 评估格: {policy} @ {task_name} =====')
            ret = subprocess.run(cmd).returncode
            if ret != 0:
                print(f'[警告] 该格子进程退出码 {ret}，记为 None')
                matrix[policy][task_name] = None
                continue
            try:
                sr = read_last_success_rate(results_file_for(seq, policy))
            except Exception as e:
                print(f'[警告] 结果解析失败: {e}')
                sr = None
            matrix[policy][task_name] = sr

    out = Path(f'forgetting_matrix_{args.sequence}.json').resolve()
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
    ap.add_argument('--sequence', required=True, choices=['A_pusht', 'B_reacher'])
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
    args = ap.parse_args()

    if args.matrix:
        if not args.policies:
            sys.exit('[拒绝] 矩阵模式需要 --policies')
        run_matrix(args)
    else:
        if not args.task or not args.policy:
            sys.exit('[拒绝] 单格模式需要 --task 和 --policy')
        run_cell(args, args.task)


if __name__ == '__main__':
    main()
