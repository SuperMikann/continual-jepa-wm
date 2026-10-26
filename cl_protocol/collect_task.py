"""按 tasks.yaml 采集指定任务的漂移数据。

用法：
    # 正式采集（铁律：输出路径必须不存在，脚本会强制检查）
    python collect_task.py --sequence B_reacher --task T2
    python collect_task.py --sequence A_pusht --task T3 --episodes 500 --seed 7

    # 采集前探针检查：只 reset 几次并保存画面，确认漂移真的生效
    python collect_task.py --sequence A_pusht --task T2 --check

DMC 环境（Reacher）需要的渲染环境变量（参照已验证的采集管线）：
    STABLEWM_HOME=/root/autodl-tmp/.stable_worldmodel \
    PYOPENGL_PLATFORM=osmesa \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    python collect_task.py --sequence B_reacher --task T2
"""

import argparse
import os
import sys
from pathlib import Path

# DMC 环境渲染后端：必须在 import stable_worldmodel 之前设置。
# 与服务器上已验证的 collect_reacher.py 保持一致（headless 无显示器只能走 osmesa）。
os.environ.setdefault('MUJOCO_GL', 'osmesa')

import numpy as np
from PIL import Image
import yaml

import stable_worldmodel as swm

from drift_world import make_darken, to_space_value


def load_task(tasks_path, seq_name, task_name):
    with open(tasks_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    seq = cfg['sequences'][seq_name]
    # 链任务 + 探针档位（probe_tasks，批注 2 剂量-响应用）合并查找
    tasks = {**seq['tasks'], **seq.get('probe_tasks', {})}
    if task_name not in tasks:
        sys.exit(f'[拒绝] 未知任务 {task_name}，可用: {sorted(tasks)}')
    return seq, tasks[task_name]


class SACPolicy:
    """加载 SAC 专家策略（expert_policy.zip + vec_normalize.pkl）用于数据采集。

    观测为环境原生状态向量（如 walker 24 维），先经 VecNormalize 归一化
    （与 SAC 训练时的观测处理完全一致），再确定性预测动作。
    """

    def __init__(self, model_path, vecnorm_path):
        from stable_baselines3 import SAC
        import pickle
        self.model = SAC.load(model_path)
        with open(vecnorm_path, 'rb') as f:
            self.vecnorm = pickle.load(f)
        self.env = None
        self.type = 'sac'

    def set_env(self, envs):
        """World.set_policy 时注册环境句柄（与 BasePolicy 接口对齐）。"""
        self.env = envs

    def get_action(self, obs, **kwargs):
        # World 采集循环传入的是 infos 字典（含 pixels/qpos/observation 等），
        # SAC 训练输入是其中的 observation 状态向量；直接传入向量也兼容。
        # infos 中的形状为 (num_envs, 1, d)，需压扁成 (num_envs, d)
        if isinstance(obs, dict):
            obs = obs['observation']
        obs = np.asarray(obs).reshape(len(obs), -1)
        o = self.vecnorm.normalize_obs(obs)
        action, _ = self.model.predict(o, deterministic=True)
        return action


def make_policy(kind, seed, args=None):
    if kind == 'random':
        return swm.policy.RandomPolicy(seed=seed)
    if kind == 'weak':
        # PushT 官方采集脚本同款弱策略（collect_pusht_fov.py / collect_weak_pusht.py）
        from stable_worldmodel.envs.pusht import WeakPolicy
        return WeakPolicy(dist_constraint=100, seed=seed)
    if kind == 'sac':
        assert args is not None and args.sac_policy and args.vecnorm, \
            'SAC 采集需要 --sac-policy=<expert_policy.zip> --vecnorm=<vec_normalize.pkl>'
        return SACPolicy(args.sac_policy, args.vecnorm)
    raise ValueError(f'未知采集策略: {kind}')


def build_options(task):
    """构造 reset options：设置漂移值 + 把漂移键的当前值记录进数据集。

    options['variation'] 里的键会以 'variation.<key>' 列写入 lance，
    之后可以随时审计数据集确实是在漂移条件下采的。
    """
    vv = task.get('variation_values') or {}
    if not vv:
        return None
    # env 侧按空间 dtype 的 ndarray 使用 variation 值（如 .value.tolist()），
    # 且 contains() 对 ndarray 做安全转换检查，必须按空间 dtype 转换
    vv = {k: to_space_value(k, v) for k, v in vv.items()}
    return {
        'variation_values': vv,
        'variation': sorted(vv.keys()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=str(Path(__file__).parent / 'tasks.yaml'))
    ap.add_argument('--sequence', required=True, choices=None)  # 序列名以 tasks.yaml 为准（A_pusht/B_reacher/C_walker/D_cheetah）
    ap.add_argument('--task', required=True,
                    help='任务名（链任务 T1/T2/T3 或 probe_tasks 里的探针档位）')
    ap.add_argument('--episodes', type=int, default=None, help='覆盖 yaml 里的 collect_episodes')
    ap.add_argument('--seed', type=int, default=3072)
    ap.add_argument('--num-envs', type=int, default=10)
    ap.add_argument('--check', action='store_true', help='只保存探针画面，不采集')
    ap.add_argument('--collect-policy', default=None,
                    help='覆盖 yaml 的 collect_policy（random/weak/sac）')
    ap.add_argument('--sac-policy', default=None,
                    help='SAC 专家策略路径（expert_policy.zip，配合 --collect-policy sac）')
    ap.add_argument('--vecnorm', default=None,
                    help='VecNormalize 统计量路径（vec_normalize.pkl）')
    args = ap.parse_args()

    seq, task = load_task(args.tasks, args.sequence, args.task)
    print(f"[任务] {args.sequence}/{args.task}: {task['desc']}")

    episodes = args.episodes or task.get('collect_episodes')
    if not args.check and episodes is None:
        sys.exit(f'[拒绝] {args.task} 没有 collect_episodes（T1 用官方/已有数据，无需采集）')

    image_transform = make_darken(task.get('darken', 1.0))
    options = build_options(task)
    print(f'[漂移] variation_values={task.get("variation_values") or {}}, '
          f'darken={task.get("darken", 1.0)}')

    world = swm.World(
        seq['env'],
        num_envs=args.num_envs,
        image_shape=(224, 224),
        max_episode_steps=seq['max_episode_steps'],
        image_transform=image_transform,
    )
    policy_kind = args.collect_policy or seq['collect_policy']
    world.set_policy(make_policy(policy_kind, args.seed, args))

    if args.check:
        # 探针：reset 后保存 8 个并行环境的首帧，人眼确认漂移生效
        world.reset(seed=args.seed, options=options)
        frames = [np.asarray(world.infos['pixels'][i, 0]) for i in range(min(8, args.num_envs))]
        grid = Image.new('RGB', (224 * 4, 224 * 2))
        for idx, fr in enumerate(frames):
            r, c = divmod(idx, 4)
            grid.paste(Image.fromarray(fr), (c * 224, r * 224))
        out = Path(f'check_{args.sequence}_{args.task}.png').resolve()
        grid.save(out)
        for key in sorted((task.get('variation_values') or {})):
            col = f'variation.{key}'
            if col in world.infos:
                # infos 里 variation 值可能是数组 (num_envs,1,...) 也可能是按环境的列表
                print(f'[探针] {col} = {np.asarray(world.infos[col][0]).squeeze()}')
        print(f'[探针] 画面已保存: {out}，确认无误后去掉 --check 正式采集')
        return

    out_path = (
        Path(swm.data.utils.get_cache_dir(sub_folder='datasets')) / task['dataset']
    )
    # 铁律：正式采集前输出路径必须为空（append 模式触发过 NaN 检查崩溃）
    if out_path.exists():
        sys.exit(f'[拒绝] 输出路径已存在: {out_path}\n'
                 f'       请先确认旧数据已备份并删除，再重新运行。')

    print(f'[采集] {episodes} episodes -> {out_path}')
    world.collect(
        out_path,
        episodes=episodes,
        seed=args.seed,
        options=options,
    )
    print(f'[完成] {args.sequence}/{args.task} 采集结束: {out_path}')


if __name__ == '__main__':
    main()
