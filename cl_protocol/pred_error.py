"""留出轨迹潜空间预测误差（每任务遗忘度量，思路来自 arXiv 2507.09177）。

对每个任务：取数据集最后 N 条轨迹作为留出集，采样观测窗口，
用世界模型做一步潜空间预测，计算 MSE（与训练 pred_loss 完全同约定）：
    frames  = [t, t+5, t+10]（历史 3 帧，步长 5 = 训练 frameskip / 评估 action_block）
    actions = 每帧槽对应 5 步动作拼接成的 10 维块（训练管线：action.reshape(num_steps, frameskip*A)）
    pred    = model.predict(encode(frames), action_chunks)[:, -1]
    target  = encode(frame t+15)
    error   = (pred - target).pow(2).mean(-1)

用途：
    同一检查点在不同任务的留出集上 → 哪任务忘了多少；
    不同检查点在同一任务上 → 顺序微调后的遗忘曲线（与 eval_task.py 的成功率矩阵互补）。

用法：
    python pred_error.py --sequence B_reacher \
        --policy lewm_reacher/weights_epoch_30.pt
    python pred_error.py --sequence A_pusht \
        --policy pusht_t3/weights_epoch_10.pt --tasks T1 T2 --windows 2000

预处理与官方评估管线完全一致（ImageNet 归一化 + 动作 StandardScaler），
保证误差数值与训练/规划处于同一分布。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm


def episode_col(dataset):
    """与官方 eval_wm.py 相同的 episode 索引列兼容处理。"""
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def img_transform():
    """与官方 eval_wm.py 的 img_transform 完全一致。"""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


@torch.no_grad()
def task_pred_error(model, dataset, held_eps, ep_len, args, device):
    """在指定留出 episode 上采样窗口，返回 (mean, std, n)。"""
    hist = getattr(model.predictor, 'num_frames', 3)  # 历史帧数（LeWM 默认 3）
    stride = args.stride
    span = hist * stride  # 目标帧相对窗口起点的偏移

    # 动作归一化：与官方评估一致，StandardScaler 拟合在当前评估数据集上
    col = episode_col(dataset)
    all_actions = dataset.get_col_data('action')
    all_ep = dataset.get_col_data(col)
    held_mask = np.isin(all_ep, held_eps)
    scaler = preprocessing.StandardScaler()
    held_actions = all_actions[held_mask]
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
                act = act.numpy()
            act = np.nan_to_num(act, nan=0.0)          # 与训练 lejepa_forward 一致
            act = scaler.transform(act).astype(np.float32)
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
        if not pixels:
            continue

        info = {
            'pixels': torch.stack(pixels).to(device),           # (B, hist, C, H, W)
            'action': torch.stack(actions).to(device),          # (B, hist, A)
        }
        info = model.encode(info)
        pred = model.predict(info['emb'], info['act_emb'])[:, -1]   # (B, D)

        tgt_info = {'pixels': torch.stack(targets).unsqueeze(1).to(device)}
        tgt = model.encode(tgt_info)['emb'][:, 0]                   # (B, D)

        mse = (pred - tgt).pow(2).mean(dim=-1)                      # (B,)，与训练 pred_loss 同尺度
        errors.extend(mse.cpu().numpy().tolist())

    errors = np.array(errors)
    return float(errors.mean()), float(errors.std()), int(errors.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks-cfg', default=str(Path(__file__).parent / 'tasks.yaml'))
    ap.add_argument('--sequence', required=True, choices=['A_pusht', 'B_reacher'])
    ap.add_argument('--policy', required=True,
                    help='检查点（checkpoints/ 下的相对路径，如 lewm_reacher/weights_epoch_30.pt）')
    ap.add_argument('--tasks', nargs='+', choices=['T1', 'T2', 'T3'],
                    help='只算这些任务（默认全部）')
    ap.add_argument('--reserve-last', type=int, default=200,
                    help='取每个数据集最后多少条做留出集（默认 200）')
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

    task_names = args.tasks or list(seq['tasks'].keys())
    results = {}
    for task_name in task_names:
        task = seq['tasks'][task_name]
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
        'results': results,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[完成] 已保存: {out}')


if __name__ == '__main__':
    main()
