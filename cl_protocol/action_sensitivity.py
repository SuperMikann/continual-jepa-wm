"""动作敏感性探针（idea 指标 7，Delta-JEPA Fig 6 样式）。

512 条固定历史（从留出集采样），历史动作保持不变，只替换最后一个
动作块（frameskip×act_dim 维），测量预测位移对动作的敏感程度：

  separation = ‖ pred(z_hist, [a_hist, a']) − pred(z_hist, [a_hist, a0]) ‖₂

其中 a0 = 零动作块，a' = 从数据集动作分布中采样的替代动作块。
敏感性好 → 不同动作产生的预测位移分得开；挤在一起 → 动作不敏感
（Delta-JEPA 论文里 LeWM 的病征）。

输出：JSON（分离度统计）+ PNG（位移向量的 2D PCA 散点，按动作着色；
理想图 = 不同颜色簇分离，病征图 = 全挤原点）。

用法：
    python action_sensitivity.py --sequence B_reacher \
        --policy m3_reacher_t1/weights_epoch_30.pt
    python action_sensitivity.py --sequence A_pusht \
        --policy m2_t1/weights_epoch_30.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn import preprocessing
from sklearn.decomposition import PCA

import stable_worldmodel as swm

from pred_error import episode_col, img_transform


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks-cfg', default=str(Path(__file__).parent / 'tasks.yaml'))
    ap.add_argument('--sequence', required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--task', default='T1')
    ap.add_argument('--histories', type=int, default=512)
    ap.add_argument('--actions-per-hist', type=int, default=8,
                    help='每条历史采多少个替代动作块')
    ap.add_argument('--reserve-last', type=int, default=200)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(args.tasks_cfg, encoding='utf-8') as f:
        seq = yaml.safe_load(f)['sequences'][args.sequence]
    task = {**seq['tasks'], **seq.get('probe_tasks', {})}[args.task]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    print(f'[模型] {args.policy} 已加载（{device}）')

    dataset = swm.data.load_dataset(
        task['dataset'], keys_to_cache=['pixels', 'action']
    )
    col = episode_col(dataset)
    all_actions = dataset.get_col_data('action')
    all_ep = dataset.get_col_data(col)
    eps = np.unique(all_ep)
    held_eps = eps[-args.reserve_last:]
    held_mask = np.isin(all_ep, held_eps)
    step_idx = dataset.get_col_data('step_idx')

    # 与 pred_error 同约定的动作归一化
    held_actions = all_actions[held_mask]
    scaler = preprocessing.StandardScaler()
    scaler.fit(held_actions[~np.isnan(held_actions).any(axis=1)])

    hist = getattr(model.predictor, 'num_frames', 3)
    stride = args.stride
    span = hist * stride
    ep_len = {e: int(step_idx[all_ep == e].max()) + 1 for e in held_eps}
    tf = img_transform()
    rng = np.random.default_rng(args.seed)

    seps = []          # 每个 (历史, 替代动作) 的分离度
    disps = []         # 位移向量（PCA 画图用）
    act_labels = []    # 哪个替代动作
    hist_ids = []      # 哪条历史（v4 逐历史面板用）

    n_done = 0
    while n_done < args.histories:
        bsz = min(64, args.histories - n_done)
        ep_pick = rng.choice(held_eps, size=bsz)
        starts = np.array([rng.integers(0, ep_len[e] - span) for e in ep_pick])
        chunks = dataset.load_chunk(ep_pick, starts, starts + span + 1)

        pixels, actions = [], []
        for ep in chunks:
            pix = ep['pixels']
            act = ep['action']
            if isinstance(act, torch.Tensor):
                act = act.numpy()
            act = np.nan_to_num(act, nan=0.0)
            act = scaler.transform(act).astype(np.float32)
            a_chunks = act[:span].reshape(hist, -1)     # (hist, stride*A)
            hist_idx = [i * stride for i in range(hist)]
            frames = torch.stack([tf(pix[i]) for i in hist_idx])
            pixels.append(frames)
            actions.append(torch.from_numpy(a_chunks))
        if not pixels:
            continue

        info = {
            'pixels': torch.stack(pixels).to(device),
            'action': torch.stack(actions).to(device),
        }
        info = model.encode(info)
        emb, act_emb = info['emb'], info['act_emb']     # (B, hist, d / da)

        # 基线：历史动作原样
        base_pred = model.predict(emb, act_emb)[:, -1]  # (B, d)

        # 同一批历史 × 零动作末块
        zero = torch.zeros_like(act_emb[:, -1:])
        pred_zero = model.predict(
            emb, torch.cat([act_emb[:, :-1], zero], dim=1)
        )[:, -1]

        # 替代动作末块：v2——固定 K 个动作方向（从首批历史取 K 个真实末块，
        # 之后所有历史共用同一组），颜色跨历史可比，簇才分得开
        if n_done == 0:
            fixed_alts = act_emb[: args.actions_per_hist, -1:].clone()
        for k in range(args.actions_per_hist):
            alt = fixed_alts[k : k + 1].expand(act_emb.size(0), -1, -1)
            pred_alt = model.predict(
                emb, torch.cat([act_emb[:, :-1], alt], dim=1)
            )[:, -1]
            d = (pred_alt - pred_zero)                  # (B, d) 位移差
            seps.extend(d.norm(dim=-1).cpu().numpy().tolist())
            disps.append(d.cpu().numpy())
            act_labels.append(np.full(d.size(0), k))
            hist_ids.append(np.arange(n_done, n_done + d.size(0)))
        n_done += bsz
        if n_done % 128 < 64:
            print(f'[进度] {n_done}/{args.histories}')

    seps = np.array(seps)
    disps = np.concatenate(disps)
    act_labels = np.concatenate(act_labels)
    hist_ids = np.concatenate(hist_ids)

    # 参照尺度：零动作预测位移本身的范数
    ref = (pred_zero - emb[:, -1]).norm(dim=-1).mean().item()

    print(f'[结果] 分离度 mean={seps.mean():.4f} ± {seps.std():.4f} '
          f'（n={seps.size}；零动作位移范数参照 {ref:.4f}）')

    # 图 v4：固定历史逐面板（忠实 Fig 6 语义"固定历史、只改最后动作"）——
    # 跨历史平均会抵消方向（动作效应依赖历史状态），所以按历史分开画：
    # 每条历史一个子图，8 个动作各一条位移箭头。局部线性 → 箭头扇形分开
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    proj = PCA(n_components=2).fit_transform(disps)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    show_hists = [int(h) for h in np.unique(hist_ids)[:4]]
    colors = plt.cm.tab10.colors
    for ax, h in zip(axes.flat, show_hists):
        vs = []
        for k in range(args.actions_per_hist):
            m = (hist_ids == h) & (act_labels == k)
            if not m.any():
                continue
            v = proj[m].mean(axis=0)  # 该历史该动作只有 1 个点
            vs.append(v)
            ax.annotate('', xy=v, xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', lw=2,
                                        color=colors[k % 10]))
            ax.text(v[0], v[1], f'a{k}', fontsize=10, color=colors[k % 10])
        # annotate 不更新坐标范围，必须显式设（含原点 + 10% 边距）
        if vs:
            vs = np.array(vs)
            lo = np.minimum(vs.min(axis=0), 0)
            hi = np.maximum(vs.max(axis=0), 0)
            span = np.maximum(hi - lo, 1e-6)
            ax.set_xlim(lo[0] - 0.1 * span[0], hi[0] + 0.1 * span[0])
            ax.set_ylim(lo[1] - 0.1 * span[1], hi[1] + 0.1 * span[1])
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        ax.set_title(f'history #{h}')
    fig.suptitle(f'action sensitivity (per-history): {args.policy} @ '
                 f'{args.sequence}/{args.task}\n'
                 f'separation={seps.mean():.4f} (ref {ref:.4f})')
    fig.tight_layout()
    safe = args.policy.replace('/', '_').replace('.pt', '')
    png = Path(f'action_sens_{args.sequence}_{safe}.png').resolve()
    fig.savefig(png, dpi=150)

    out = Path(f'action_sens_{args.sequence}_{safe}.json').resolve()
    out.write_text(json.dumps({
        'policy': args.policy, 'sequence': args.sequence, 'task': args.task,
        'histories': int(args.histories),
        'actions_per_hist': args.actions_per_hist,
        'separation_mean': float(seps.mean()),
        'separation_std': float(seps.std()),
        'separation_p25': float(np.percentile(seps, 25)),
        'separation_p75': float(np.percentile(seps, 75)),
        'zero_action_disp_ref': float(ref),
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[完成] {png} / {out}')


if __name__ == '__main__':
    main()
