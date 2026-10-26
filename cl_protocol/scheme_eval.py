"""scheme_eval.py — 历史方案评价误差 E_S + 排序一致性（Kendall τ）实验
====================================================================

重要：此旧入口的时间轴未对齐。模型看到数据集 t、t+5、t+10 的图像，
而模拟器从 t 执行全部随机候选动作；后两张历史图像不属于这些候选轨迹。
因此历史 τ/E_S 不能作为世界模型预测排序准确性的证据。保留此入口只为
复现旧结果；新的时间对齐诊断请用 cl_protocol/test/aligned_scheme_eval.py。

要回答的问题：表示层的漂移（pred_error）到底怎么传导成行为退化？
规划器选方案靠的是模型对候选动作序列的**打分排名**，不是平均预测误差。
本脚本把「模型排名」和「真实排名」直接对上：

  1. 模型打分：100% 复用官方 CEMSolver 第一次迭代的总体口径——
     randn(S, horizon, frameskip*A) × 1.0 + 0（标准化动作空间），
     再经官方 ShootingCostEvaluator + GoalMSE 打分（规划器本尊）。
     候选 S 个动作块 × frameskip 个物理步 = horizon×frameskip 步，
     恰等于 goal_offset，即模型 rollout 终点帧与真实执行终点帧对齐到数据
     集同一行（start+goal_offset 处那一帧）。
  2. 真实执行：复用官方 world.evaluate 的数据集驱动模式——每场景开 S 个
     并行环境，从数据集的 qpos/qvel 精确恢复起始状态（variation 漂移 +
     darken 照常注入），每个环境跑一条候选的原始动作，末帧即该候选的
     真实落点。
  3. 指标（每场景各算一次，再跨场景汇总）：
     - τ_current：模型排名 vs 真实排名（当前模型编码器口径，端到端）
     - τ_t1：     模型排名 vs 真实排名（T1 链首模型编码器口径）
     - hit@1：    模型选的最优候选是否也是真实最优（argmin 一致率）
     - E_S：      模型分与真实分的平均绝对差（仅同编码器空间可比，
                  即 E_S 用 current 口径；t1 口径只报 τ）

局限（诚实声明）：
  - τ 测的是**初始总体**的排名。CEM 正式跑 30 轮迭代，最终动作是精英
    分布均值；初始总体排名崩了迭代救不回来，但排名好不保证迭代后仍好。
    τ 是诊断量，不是规划成功率的充分条件。
  - 场景取自各任务数据集留出 episode（与 pred_error 同口径：最后
    --holdout 条；注意其中 T1 的最后若干条参与过训练，绝对值偏乐观，
    但同场景内排名不受影响）。
  - 仅支持有 qpos/qvel 状态恢复的 DMC 序列（B_reacher / C_walker /
    D_cheetah）；PushT（A_pusht）没有物理状态恢复机制，拒绝运行。

用法（服务器，仓库根目录）：
    # 0) 探针自检：打印数据集列名/动作维度，跑 1 场景 × 4 候选全链路
    python cl_protocol/scheme_eval.py --check \
      --sequence D_cheetah --task T3 \
      --policy m6_cheetah_t3/weights_epoch_10.pt

    # 1) 单格正式跑（链尾模型在 T3 任务场景上的方案评价质量）
    python cl_protocol/scheme_eval.py \
      --sequence D_cheetah --task T3 \
      --policy m6_cheetah_t3/weights_epoch_10.pt \
      --t1-policy m6_cheetah_t1/weights_epoch_30.pt \
      --num-scenes 20 --num-candidates 24

    # 2) 矩阵：shell 循环若干 (policy × task) 格，各自出一个 json

环境变量（DMC headless 渲染，同 collect_task 踩坑结论）：
    STABLEWM_HOME=/root/autodl-tmp/.stable_worldmodel \
    PYOPENGL_PLATFORM=osmesa \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    python cl_protocol/scheme_eval.py ...

产物：仓库根目录 scheme_eval_<sequence>_<policy名>_<task>.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# DMC headless 渲染后端：必须在 import stable_worldmodel 之前设置
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

from collections import deque

import numpy as np
import torch
import yaml
from sklearn import preprocessing
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import Compose

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator

from drift_world import DriftWorld, make_darken

# 有物理状态恢复机制（set_state/qpos/qvel）的序列才支持
DMC_SEQUENCES = ('B_reacher', 'C_walker', 'D_cheetah')

# 与官方 scripts/plan/config/reacher.yaml 对齐的规划参数
DEFAULT_HORIZON = 5        # 规划步数
DEFAULT_FRAMESKIP = 5      # 每个规划步的物理步数（action_block）
DEFAULT_GOAL_OFFSET = 25   # 目标帧偏移 = horizon × frameskip


def img_transform():
    """与 pred_error.py / 官方评估管线完全一致的图像变换。"""
    return Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


def episode_col(dataset):
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def kendall_tau(x, y):
    """Kendall τ（含 ties 的标准公式，n≤几百够用）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - tx) * (n0 - ty))
    if denom == 0:
        return float('nan')
    return (conc - disc) / denom


class BakedPolicy(swm.policy.BasePolicy):
    """把预生成的候选动作序列逐环境步进式回放的策略。

    每个环境一条候选；队列空的动作补零（环境一般已无剩余预算，
    或已被终止冻结）。 achieved 帧在循环结束后从 world.infos 读。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = 'baked'
        self._queues = None

    def set_actions(self, actions):
        """actions: list of (T, A) numpy 数组，每个环境一条。"""
        self._queues = [deque(a.tolist()) for a in actions]

    def get_action(self, info_dict, **kwargs):
        n = self.env.num_envs
        a_dim = self.env.single_action_space.shape[0]
        action = np.zeros((n, a_dim), dtype=np.float32)
        for i in range(n):
            if self._queues[i]:
                action[i] = self._queues[i].popleft()
        return action


def build_callables(dataset, columns):
    """按官方 reacher.yaml 的口径构造状态恢复 callables。

    qpos/qvel 必须有；goal_qpos 只影响环境的成功判定（本脚本不用），
    缺列就自动去掉 set_target_qpos。
    """
    if 'qpos' not in columns or 'qvel' not in columns:
        sys.exit('[拒绝] 数据集缺少 qpos/qvel 列（set_state 两个参数都必须有），'
                 f'当前列: {sorted(columns)}')
    args = {'qpos': {'value': 'qpos'}, 'qvel': {'value': 'qvel'}}
    specs = [{'method': 'set_state', 'args': args}]
    if 'goal_qpos' in columns:
        specs.append({
            'method': 'set_target_qpos',
            'args': {'target_qpos': {'value': 'goal_qpos'}},
        })
    return specs


def encode_frames(model, frames_u8, device):
    """把 (N,H,W,C) uint8 帧编码成 (N,D) 潜变量（与评估管线同变换）。"""
    tf = img_transform()
    batch = torch.stack([
        tf(np.asarray(f, dtype=np.uint8)) for f in frames_u8
    ]).to(device)                                    # (N,C,H,W)
    with torch.no_grad():
        info = model.encode({'pixels': batch.unsqueeze(1)})  # (N,1,C,H,W)
    return info['emb'][:, 0]                         # (N,D)


def true_cost(model, achieved_u8, goal_u8, device):
    """真实方案成本，与官方 GoalMSE 同定义：末帧 embedding 差方和。"""
    a = encode_frames(model, achieved_u8, device)
    g = encode_frames(model, goal_u8[None], device)[0]
    return ((a - g) ** 2).sum(dim=-1).cpu().numpy()  # (N,)


def main():
    print('[LEGACY_MISALIGNED] Historical frames are not aligned with candidate replay; '
          'use cl_protocol/test/aligned_scheme_eval.py for model-fidelity claims.',
          flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=str(Path(__file__).parent / 'tasks.yaml'))
    ap.add_argument('--sequence', required=True)
    ap.add_argument('--task', required=True,
                    help='任务名（链任务 T1/T2/T3 或 probe_tasks 里的探针档位）')
    ap.add_argument('--policy', required=True,
                    help='待评测检查点（checkpoints/ 下相对路径）')
    ap.add_argument('--t1-policy', default=None,
                    help='T1 口径编码器所用链首检查点（可选；不给则不出 τ_t1）')
    ap.add_argument('--num-scenes', type=int, default=20, help='评估场景数')
    ap.add_argument('--num-candidates', type=int, default=24, help='每场景候选数')
    ap.add_argument('--horizon', type=int, default=DEFAULT_HORIZON)
    ap.add_argument('--frameskip', type=int, default=DEFAULT_FRAMESKIP)
    ap.add_argument('--goal-offset', type=int, default=DEFAULT_GOAL_OFFSET)
    ap.add_argument('--holdout', type=int, default=200,
                    help='每任务取最后多少条 episode 作场景池（与 pred_error 同口径）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--check', action='store_true',
                    help='探针模式：打印数据集列名/维度，1 场景 × 4 候选全链路试跑')
    args = ap.parse_args()

    if args.check:
        args.num_scenes = 1
        args.num_candidates = 4

    with open(args.tasks, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if args.sequence not in cfg['sequences']:
        sys.exit(f'[拒绝] 未知序列 {args.sequence}，可用: {sorted(cfg["sequences"])}')
    seq = cfg['sequences'][args.sequence]
    if args.sequence not in DMC_SEQUENCES:
        sys.exit(f'[拒绝] {args.sequence} 没有 qpos/qvel 状态恢复机制，'
                 f'本脚本仅支持 {DMC_SEQUENCES}')
    all_tasks = {**seq['tasks'], **seq.get('probe_tasks', {})}
    if args.task not in all_tasks:
        sys.exit(f'[拒绝] 未知任务 {args.task}，可用: {sorted(all_tasks)}')
    task = all_tasks[args.task]

    plan_steps = args.horizon * args.frameskip
    assert plan_steps == args.goal_offset, (
        f'horizon×frameskip={plan_steps} 与 goal_offset={args.goal_offset} 不一致，'
        '候选执行长度与目标帧偏移必须对齐'
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[配置] {args.sequence}/{args.task} <- {args.policy}')
    print(f'[规划] horizon={args.horizon} frameskip={args.frameskip} '
          f'候选执行={plan_steps} 物理步 = goal_offset')

    # ---- 数据集 ----
    dataset = swm.data.load_dataset(
        task['dataset'], keys_to_cache=['pixels', 'action']
    )
    columns = set(dataset.column_names) | set(getattr(dataset, '_schema_names', ()))
    col = episode_col(dataset)
    ep_arr = dataset.get_col_data(col)
    step_arr = dataset.get_col_data('step_idx')
    uniq_eps = np.unique(ep_arr)
    ep_len = {e: int(step_arr[ep_arr == e].max()) + 1 for e in uniq_eps}

    # 候选动作维度：数据集 action 列的 A × frameskip
    all_actions = dataset.get_col_data('action')
    a_dim = int(np.asarray(all_actions).shape[1])
    cand_dim = a_dim * args.frameskip
    print(f'[数据] {task["dataset"]} 动作维度 A={a_dim} 候选块维度={cand_dim}')

    # 动作标准化：与官方评估一致，StandardScaler 拟合整个数据集 action 列
    scaler = preprocessing.StandardScaler()
    scaler.fit(all_actions[~np.isnan(all_actions).any(axis=1)])

    # ---- 模型 ----
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    t1_model = None
    if args.t1_policy:
        t1_model = swm.wm.utils.load_pretrained(args.t1_policy)
        t1_model = t1_model.to(device).eval()
        t1_model.requires_grad_(False)
        print(f'[模型] T1 口径编码器 <- {args.t1_policy}')
    evaluator = ShootingCostEvaluator(model, GoalMSE())

    # ---- 场景采样（固定种子：跨 policy 跑时场景与候选完全一致） ----
    rng = np.random.default_rng(args.seed)
    hold_eps = uniq_eps[-args.holdout:]
    valid = []  # (ep, start)
    for e in hold_eps:
        max_start = ep_len[e] - args.goal_offset - 1
        starts = step_arr[(ep_arr == e) & (step_arr <= max_start)]
        for s in np.unique(starts):
            valid.append((int(e), int(s)))
    if len(valid) < args.num_scenes:
        sys.exit(f'[拒绝] 有效场景只有 {len(valid)} 个，不够 {args.num_scenes}')
    scenes = [valid[i] for i in rng.choice(len(valid), args.num_scenes, replace=False)]
    print(f'[场景] 留出池 {len(hold_eps)} episodes / {len(valid)} 个有效起点，'
          f'抽 {args.num_scenes} 个（seed={args.seed}）')

    # ---- 候选（CEM 第一次迭代总体：randn × 1.0 + 0，标准化空间） ----
    gen = torch.Generator().manual_seed(args.seed)
    candidates = torch.randn(
        1, args.num_candidates, args.horizon, cand_dim, generator=gen
    )  # (1,S,horizon,D) 所有场景共用 → 跨场景可配对比较

    # ---- 真实执行世界（每场景 S 个并行环境，各跑一条候选） ----
    world = DriftWorld(
        seq['env'],
        num_envs=args.num_candidates,
        image_shape=(224, 224),
        max_episode_steps=seq['max_episode_steps'],
        image_transform=make_darken(task.get('darken', 1.0)),
        variation_values=task.get('variation_values') or {},
    )
    baked = BakedPolicy()
    world.set_policy(baked)

    if args.check:
        print(f'[检查] 数据集列: {sorted(columns)}')
        env0 = world.envs.envs[0].unwrapped
        print(f'[检查] 环境方法: set_state={hasattr(env0, "set_state")} '
              f'set_target_qpos={hasattr(env0, "set_target_qpos")}')
        if not hasattr(env0, 'set_state'):
            sys.exit('[拒绝] 环境没有 set_state，无法恢复数据集状态')

    callables = build_callables(dataset, columns)

    tf = img_transform()
    per_scene = []
    for idx, (ep, start) in enumerate(scenes):
        # 一次 load_chunk 取 [start, start+goal_offset] 全部行
        chunk = dataset.load_chunk(
            np.array([ep]), np.array([start]),
            np.array([start + args.goal_offset + 1]),
        )[0]
        pixels = chunk['pixels']                       # (L,C,H,W) torch
        acts = chunk['action'].numpy() if hasattr(chunk['action'], 'numpy') \
            else np.asarray(chunk['action'])           # (L,A)
        fs = args.frameskip
        hist_idx = [i * fs for i in range(3)]          # t, t+5, t+10
        window = torch.stack([tf(pixels[i]) for i in hist_idx])   # (3,C,H,W)
        goal_frame = tf(pixels[args.goal_offset])                # (C,H,W)
        # 历史动作块（供 default_goal_encode 的 pop('action') 契约；rollout 会用候选前 3 块覆盖）
        hist_act = acts[:3 * fs]
        hist_chunks = np.stack([
            scaler.transform(hist_act[i * fs:(i + 1) * fs]).reshape(-1)
            for i in range(3)
        ])                                              # (3, cand_dim)

        # ---- 模型打分（官方 ShootingCostEvaluator 口径） ----
        S = args.num_candidates
        info_dict = {
            'pixels': window[None, None].expand(1, S, 3, *window.shape[1:]),
            'goal': goal_frame[None, None, None].expand(1, S, 1, *goal_frame.shape),
            'action': torch.from_numpy(hist_chunks)[None, None]
                              .expand(1, S, 3, cand_dim).clone(),
        }
        with torch.no_grad():
            model_cost = evaluator.get_cost(
                {k: v.to(device) for k, v in info_dict.items()},
                candidates.to(device),
            )[0].cpu().numpy()                          # (S,)

        # ---- 真实执行：候选 → 原始动作 → 环境 ----
        cand_flat = candidates[0].reshape(S, plan_steps, a_dim)   # (S,T,A)
        raw = scaler.inverse_transform(
            cand_flat.reshape(S * plan_steps, a_dim)
        ).reshape(S, plan_steps, a_dim)
        baked.set_actions([raw[i] for i in range(S)])
        world.evaluate(
            dataset=dataset,
            episodes_idx=[ep] * S,
            start_steps=[start] * S,
            goal_offset=args.goal_offset,
            eval_budget=plan_steps,
            callables=callables,
        )
        achieved = world.infos['pixels'][:, -1]          # (S,H,W,C) uint8
        goal_np = np.asarray(
            pixels[args.goal_offset].permute(1, 2, 0).detach().cpu().numpy()
        )
        if np.issubdtype(goal_np.dtype, np.floating):
            goal_np = np.clip(goal_np * 255.0, 0, 255)
        goal_u8 = goal_np.astype(np.uint8)             # (H,W,C)
        if achieved.ndim == 5:                          # 防御：(S,1,H,W,C) 形态
            achieved = achieved[:, -1]

        true_cur = true_cost(model, achieved, goal_u8, device)
        rec = {
            'scene': idx, 'ep': ep, 'start': start,
            'tau_cur': kendall_tau(model_cost, true_cur),
            'hit_cur': float(np.argmin(model_cost) == np.argmin(true_cur)),
            'Es_cur': float(np.mean(np.abs(model_cost - true_cur))),
            'model_cost_mean': float(model_cost.mean()),
            'true_cost_mean': float(true_cur.mean()),
        }
        if t1_model is not None:
            true_t1 = true_cost(t1_model, achieved, goal_u8, device)
            rec['tau_t1'] = kendall_tau(model_cost, true_t1)
            rec['hit_t1'] = float(np.argmin(model_cost) == np.argmin(true_t1))

        per_scene.append(rec)
        msg = (f"[场景 {idx + 1}/{args.num_scenes}] ep={ep} start={start} "
               f"τ_current={rec['tau_cur']:+.3f} hit={rec['hit_cur']:.0f} "
               f"E_S={rec['Es_cur']:.4f}")
        if t1_model is not None:
            msg += f" τ_t1={rec['tau_t1']:+.3f} hit_t1={rec['hit_t1']:.0f}"
        print(msg)

    world.close()

    def agg(key):
        vals = np.array([r[key] for r in per_scene if key in r and r[key] == r[key]])
        return {'mean': float(vals.mean()), 'std': float(vals.std())} if len(vals) else None

    summary = {
        'tau_current': agg('tau_cur'),
        'hit_current': agg('hit_cur'),
        'Es_current': agg('Es_cur'),
    }
    if t1_model is not None:
        summary['tau_t1'] = agg('tau_t1')
        summary['hit_t1'] = agg('hit_t1')

    out = {
        'sequence': args.sequence, 'task': args.task,
        'policy': args.policy, 't1_policy': args.t1_policy,
        'config': {
            'num_scenes': args.num_scenes,
            'num_candidates': args.num_candidates,
            'horizon': args.horizon, 'frameskip': args.frameskip,
            'goal_offset': args.goal_offset, 'holdout': args.holdout,
            'seed': args.seed,
        },
        'summary': summary, 'per_scene': per_scene,
    }
    safe = args.policy.replace('/', '_').replace('.pt', '')
    out_path = Path(f'scheme_eval_{args.sequence}_{safe}_{args.task}.json').resolve()
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n===== 汇总（跨场景 mean ± std） =====')
    for k, v in summary.items():
        if v:
            print(f'  {k:14s} {v["mean"]:+.4f} ± {v["std"]:.4f}')
    print(f'\n[完成] 已保存: {out_path}')


if __name__ == '__main__':
    main()
