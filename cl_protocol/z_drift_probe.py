"""z 漂移直接度量探针（表示层通道归因证据）。

对同一批 T1 留出帧，分别用链首（T1 训完）与链尾（T3 训完）检查点的
编码器做纯前向 encode，直接比较 latent 表示漂移程度：

  1. 逐帧 cosine 相似度 cos(z_first, z_last) —— 直觉可读，
     1.0 = 完全没漂，越低漂越狠
  2. 线性 CKA —— 对旋转/缩放不敏感的整体几何对齐度（0~1）

用途：把「T1 表示漂移残余通道 = 编码器」从排除法推断升级为直接证据；
若 z 漂得厉害 → 冻编码器先验成立；若 z 基本没漂而 pred 仍劣化 →
漂移载体另有其人（预测器/投影层），冻编码器开局前即可证伪。

用法：
    python z_drift_probe.py --sequence B_reacher \
        --policy-a m3_reacher_t1/weights_epoch_30.pt \
        --policy-b m3_reacher_t3/weights_epoch_10.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

import stable_worldmodel as swm

from pred_error import episode_col, img_transform


def linear_cka(X, Y):
    """线性 CKA（Kornblith et al. 2019）。X, Y: (N, d)，已减均值。"""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X @ Y.T).pow(2).sum()
    norm = (X @ X.T).pow(2).sum().sqrt() * (Y @ Y.T).pow(2).sum().sqrt()
    return (hsic / norm).item()


@torch.no_grad()
def encode_frames(model, dataset, frames_idx, tf, args, device):
    """按行号取帧、编码，返回 (N, d) numpy。"""
    from torchvision.io import decode_image

    def to_tensor(frame):
        # lance 底层按压缩字节存像素（load_chunk 才解码），直接取列需先解码
        if isinstance(frame, (bytes, bytearray, np.bytes_)):
            return decode_image(
                torch.frombuffer(bytearray(frame), dtype=torch.uint8)
            )
        return frame

    pixels = dataset.get_col_data('pixels')
    zs = []
    for i in range(0, len(frames_idx), args.batch):
        batch_idx = frames_idx[i:i + args.batch]
        imgs = torch.stack([tf(to_tensor(pixels[j])) for j in batch_idx])
        info = {'pixels': imgs.unsqueeze(1).to(device)}  # (B, 1, C, H, W)
        z = model.encode(info)['emb'][:, 0]
        zs.append(z.float().cpu())
    return torch.cat(zs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks-cfg', default=str(Path(__file__).parent / 'tasks.yaml'))
    ap.add_argument('--sequence', required=True, choices=['A_pusht', 'B_reacher'])
    ap.add_argument('--policy-a', required=True, help='链首检查点（如 T1 训完）')
    ap.add_argument('--policy-b', required=True, help='链尾检查点（如 T3 训完）')
    ap.add_argument('--task', default='T1', choices=['T1', 'T2', 'T3'],
                    help='在哪个任务的留出集上测（默认 T1）')
    ap.add_argument('--reserve-last', type=int, default=200)
    ap.add_argument('--frames', type=int, default=2000, help='采样帧数')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(args.tasks_cfg, encoding='utf-8') as f:
        seq = yaml.safe_load(f)['sequences'][args.sequence]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = swm.data.load_dataset(
        seq['tasks'][args.task]['dataset'], keys_to_cache=['pixels']
    )
    col = episode_col(dataset)
    ep_arr = dataset.get_col_data(col)
    held_eps = np.unique(ep_arr)[-args.reserve_last:]
    held_rows = np.where(np.isin(ep_arr, held_eps))[0]
    rng = np.random.default_rng(args.seed)
    frames_idx = np.sort(rng.choice(held_rows, size=min(args.frames, held_rows.size), replace=False))
    tf = img_transform()

    embs = {}
    for tag, policy in (('A', args.policy_a), ('B', args.policy_b)):
        model = swm.wm.utils.load_pretrained(policy)
        model = model.to(device).eval()
        model.requires_grad_(False)
        embs[tag] = encode_frames(model, dataset, frames_idx, tf, args, device)
        print(f'[模型 {tag}] {policy} 已编码 {embs[tag].shape[0]} 帧')
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None

    za, zb = torch.from_numpy(embs['A']), torch.from_numpy(embs['B'])
    cos = torch.nn.functional.cosine_similarity(za, zb, dim=-1)
    cka = linear_cka(za, zb)

    print(f'[z_drift] {args.sequence}/{args.task} 留出集，n={za.shape[0]}')
    print(f'[z_drift] 逐帧 cosine: mean={cos.mean():.4f} std={cos.std():.4f} '
          f'min={cos.min():.4f} p5={cos.quantile(0.05):.4f}')
    print(f'[z_drift] 线性 CKA = {cka:.4f}')
    print(f'ZDRIFT_RESULT {{"seq": "{args.sequence}", "task": "{args.task}", '
          f'"A": "{args.policy_a}", "B": "{args.policy_b}", '
          f'"cos_mean": {cos.mean():.4f}, "cos_p5": {cos.quantile(0.05):.4f}, '
          f'"cka": {cka:.4f}, "n": {za.shape[0]}}}')

    out = Path(f'zdrift_{args.sequence}_{args.task}.json').resolve()
    import json
    history = []
    if out.exists():
        history = json.loads(out.read_text(encoding='utf-8'))
    history.append({
        'policy_a': args.policy_a, 'policy_b': args.policy_b,
        'sequence': args.sequence, 'task': args.task,
        'cos_mean': float(cos.mean()), 'cos_std': float(cos.std()),
        'cos_min': float(cos.min()), 'cos_p5': float(cos.quantile(0.05)),
        'cka': cka, 'n': int(za.shape[0]),
    })
    out.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[完成] 已保存: {out}')


if __name__ == '__main__':
    main()
