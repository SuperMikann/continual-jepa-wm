"""指定轨迹上的潜空间预测误差。

默认取数据集最后 N 条轨迹采样；这不保证与训练集独立。
用世界模型做一步潜空间预测，计算当前模型表示空间内的 MSE：
    frames  = [t, t+5, t+10]（历史 3 帧，步长 5 = 训练 frameskip / 评估 action_block）
    actions = 每帧槽对应 5 步动作拼接成的 10 维块（训练管线：action.reshape(num_steps, frameskip*A)）
    pred    = model.predict(encode(frames), action_chunks)[:, -1]
    target  = encode(frame t+15)
    error   = (pred - target).pow(2).mean(-1)

用途：
    比较任务/检查点的预测误差；需另核动作预处理、数据独立性和表示尺度。
    不能仅凭跨检查点的潜空间 MSE 判定控制能力遗忘。

用法：
    python pred_error.py --sequence B_reacher \
        --policy lewm_reacher/weights_epoch_30.pt
    python pred_error.py --sequence A_pusht \
        --policy pusht_t3/weights_epoch_10.pt --tasks T1 T2 --windows 2000

历史 CLI 保留每任务分别拟合 StandardScaler 的方式，供复现已有数字。
drift_trigger.py 新版显式传入同一个动作变换，避免新旧条件各自换输入坐标。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml


class FixedActionScaler:
    """使用已记录的有效 mean/scale；不在检测数据上重新拟合。"""

    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        if (not self.mean.size or self.mean.shape != self.scale.shape
                or not np.isfinite(self.mean).all()
                or not np.isfinite(self.scale).all() or (self.scale <= 0).any()):
            raise ValueError('动作 mean/scale 必须同维、有限，且 scale > 0')

    def transform(self, actions):
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[-1] != self.mean.size:
            raise ValueError('动作维度与固定归一化统计不一致')
        if not np.isfinite(actions).all():
            raise ValueError('预测窗口内存在非有限动作，拒绝计算漂移分数')
        return (actions - self.mean) / self.scale


def episode_col(dataset):
    """与官方 eval_wm.py 相同的 episode 索引列兼容处理。"""
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def img_transform():
    """与官方 eval_wm.py 的 img_transform 完全一致。"""
    from torchvision.transforms import v2 as transforms
    import stable_pretraining as spt

    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


@torch.no_grad()
def task_pred_error(model, dataset, held_eps, ep_len, args, device, *, action_scaler=None,
                    progress_callback=None):
    """在指定 episode 上采样，返回 (mean, std, n)。不自行认定训练独立性。

    action_scaler=None 保留历史逐任务拟合；漂移检测应显式传入同一变换。
    """
    hist = getattr(model.predictor, 'num_frames', 3)  # 历史帧数（LeWM 默认 3）
    stride = args.stride
    span = hist * stride  # 目标帧相对窗口起点的偏移

    if min(args.windows, args.batch, stride, hist) <= 0:
        raise ValueError('windows/batch/stride/history 必须为正整数')
    if not len(held_eps) or any(ep_len[e] <= span for e in held_eps):
        raise ValueError('指定轨迹为空或不足以构造完整预测窗口')

    scaler = action_scaler
    if scaler is None:
        from sklearn import preprocessing

        # 仅供历史复现；不能据此称新旧条件使用了同一动作坐标。
        col = episode_col(dataset)
        all_actions = dataset.get_col_data('action')
        all_ep = dataset.get_col_data(col)
        held_actions = all_actions[np.isin(all_ep, held_eps)]
        scaler = preprocessing.StandardScaler()
        scaler.fit(held_actions[~np.isnan(held_actions).any(axis=1)])

    tf = img_transform()
    rng = np.random.default_rng(args.seed)
    errors = []

    while len(errors) < args.windows:
        # 采一批窗口：随机 episode + 随机起点
        bsz = min(args.batch, args.windows - len(errors))
        ep_pick = rng.choice(held_eps, size=bsz)
        starts = np.array([
            rng.integers(0, ep_len[e] - span) for e in ep_pick
        ])
        chunks = dataset.load_chunk(ep_pick, starts, starts + span + 1)

        pixels, actions, targets = [], [], []
        for ep in chunks:
            pix = ep['pixels']   # (L, C, H, W) torch
            act = ep['action']   # (L, A)
            if isinstance(act, torch.Tensor):
                act = act.cpu().numpy()
            if len(pix) != span + 1 or len(act) < span:
                raise ValueError('数据加载返回了不完整的预测窗口')
            if action_scaler is None:
                act = np.nan_to_num(act, nan=0.0)  # 保留历史缺失动作处理
                act = scaler.transform(act).astype(np.float32)
            else:
                # 只消耗目标帧之前的动作；末帧可能是没有后继动作的终止帧。
                act = scaler.transform(act[:span]).astype(np.float32)
            # 动作块：帧槽 i 的动作 = 步 [i*stride, (i+1)*stride) 的拼接
            # （训练管线 dataset.py: action.reshape(num_steps, frameskip*A)，
            #   action_encoder 输入维度 = frameskip * action_dim = 10）
            a_chunks = act[:span].reshape(hist, -1)    # (hist, stride*A)
            hist_idx = [i * stride for i in range(hist)]
            frames = torch.stack([tf(pix[i]) for i in hist_idx])
            target = tf(pix[span])
            pixels.append(frames)
            actions.append(torch.from_numpy(a_chunks))
            targets.append(target)
        if len(pixels) != bsz:
            raise ValueError('数据加载返回的窗口数与请求不一致')

        info = {
            'pixels': torch.stack(pixels).to(device),           # (B, hist, C, H, W)
            'action': torch.stack(actions).to(device),          # (B, hist, A)
        }
        info = model.encode(info)
        pred = model.predict(info['emb'], info['act_emb'])[:, -1]   # (B, D)

        tgt_info = {'pixels': torch.stack(targets).unsqueeze(1).to(device)}
        tgt = model.encode(tgt_info)['emb'][:, 0]                   # (B, D)

        mse = (pred - tgt).pow(2).mean(dim=-1)                      # (B,)，与训练 pred_loss 同尺度
        if mse.shape != (bsz,) or not torch.isfinite(mse).all():
            raise ValueError('模型预测误差无效或数量不符，拒绝输出检测结果')
        errors.extend(mse.cpu().numpy().tolist())
        if progress_callback is not None:
            progress_callback(len(errors), args.windows)

    errors = np.array(errors)
    return float(errors.mean()), float(errors.std()), int(errors.size)


def main():
    import stable_worldmodel as swm

    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks-cfg', default=str(Path(__file__).parent / 'tasks.yaml'))
    ap.add_argument('--sequence', required=True, choices=None)  # 序列名以 tasks.yaml 为准（A_pusht/B_reacher/C_walker/D_cheetah）
    ap.add_argument('--policy', required=True,
                    help='检查点（checkpoints/ 下的相对路径，如 lewm_reacher/weights_epoch_30.pt）')
    ap.add_argument('--tasks', nargs='+',
                    help='只算这些任务（默认全部链任务，可选探针档位）')
    ap.add_argument('--reserve-last', type=int, default=200,
                    help='取每个数据集最后多少条检测（不保证与训练独立，默认 200）')
    ap.add_argument('--windows', type=int, default=1000, help='每任务采样窗口数')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--stride', type=int, default=5, help='帧步长（=训练 frameskip）')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(args.tasks_cfg, encoding='utf-8') as f:
        seq = yaml.safe_load(f)['sequences'][args.sequence]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    print(f'[模型] {args.policy} 已加载（{device}）')

    # 链任务 + 探针档位（probe_tasks，批注 2 剂量-响应用）合并查找；
    # 默认仍只迭代链任务
    all_tasks = {**seq['tasks'], **seq.get('probe_tasks', {})}
    task_names = args.tasks or list(seq['tasks'].keys())
    results = {}
    for task_name in task_names:
        if task_name not in all_tasks:
            raise KeyError(f'未知任务 {task_name}，可用: {sorted(all_tasks)}')
        task = all_tasks[task_name]
        dataset = swm.data.load_dataset(
            task['dataset'], keys_to_cache=['pixels', 'action']
        )
        col = episode_col(dataset)
        eps = np.unique(dataset.get_col_data(col))
        held_eps = eps[-args.reserve_last:]
        step_idx = dataset.get_col_data('step_idx')
        ep_arr = dataset.get_col_data(col)
        ep_len = {e: int(step_idx[ep_arr == e].max()) + 1 for e in held_eps}

        mean, std, n = task_pred_error(model, dataset, held_eps, ep_len, args, device)
        results[task_name] = {'mse_mean': mean, 'mse_std': std, 'n_windows': n}
        print(f'[{args.sequence}/{task_name}] 预测误差 MSE = {mean:.4f} ± {std:.4f} (n={n})')

    safe = args.policy.replace('/', '_').replace('.pt', '')
    out = Path(f'pred_error_{args.sequence}_{safe}.json').resolve()
    out.write_text(json.dumps({
        'policy': args.policy, 'sequence': args.sequence,
        'reserve_last': args.reserve_last, 'stride': args.stride,
        'data_independence': 'unverified',
        'action_normalization': 'legacy_per_task_fit',
        'results': results,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[完成] 已保存: {out}')


if __name__ == '__main__':
    main()
