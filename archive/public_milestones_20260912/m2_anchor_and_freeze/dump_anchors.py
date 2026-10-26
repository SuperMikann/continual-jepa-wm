"""dump_anchors.py — 任务训练结束后导出潜在锚点库（方法里程碑 2，idea 式 15）。

用法（在仓库根目录）：
  python scripts/train/dump_anchors.py \
    data.dataset.name=<任务数据集.lance> \
    +policy=<刚训完的 weights_epoch_30.pt 绝对路径> \
    +out=<锚点输出绝对路径.pt> +num_anchors=4096

产物：{'ctx': (M, history, d), 'act': (M, history, da), 'pred': (M, n_preds, d)}
  全部冻结在导出模型的表征空间里（含其 encoder / action encoder 的漂移状态）。
  下一任务训练时以 L_anchor = |predict(ctx, act) - pred|_1 做 L1 蒸馏，
  约束 dynamics 函数在旧任务区域的形状（target 端无梯度）。
"""
import os

import hydra
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import OmegaConf
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer as get_column_normalizer


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    assert cfg.get('policy'), '必须指定 +policy=<权重路径>'
    assert cfg.get('out'), '必须指定 +out=<输出路径.pt>'
    num_anchors = int(cfg.get('num_anchors', 4096))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #########################
    ## dataset（与 lewm.py 同款加载，但不做 train/val 切分）
    #########################
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(f'Loading dataset "{dataset_name}"')
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith('pixels'):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=4,
        drop_last=False, pin_memory=True,
    )

    #########################
    ## model（从权重 + 同目录 config.json 复原，含 Diff/Adapter 头）
    #########################
    model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to(device).eval()

    h = cfg.wm.history_size

    ctxs, acts, preds, nexts = [], [], [], []
    collected = 0
    # v2：bf16 autocast 对齐训练精度——锚点目标与训练时前向同精度，
    # 消除自蒸馏地板偏差（v1 fp32 导出导致 ~0.04-0.1 系统性偏差）
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for batch in loader:
            batch['action'] = torch.nan_to_num(batch['action'], 0.0)
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            out = model.encode(batch)
            emb = out['emb']          # (B, T, d)
            act_emb = out['act_emb']  # (B, T, da)
            ctx = emb[:, :h]
            act = act_emb[:, :h]
            nxt = emb[:, cfg.wm.num_preds:]  # (B, n_preds, d) 真实下一 latent
            pred = model.predict(ctx, act)   # (B, n_preds, d) 旧模型预测（分析用）
            ctxs.append(ctx.cpu())
            acts.append(act.cpu())
            preds.append(pred.cpu())
            nexts.append(nxt.cpu())
            collected += ctx.size(0)
            if collected >= num_anchors:
                break

    anchors = {
        'ctx': torch.cat(ctxs)[:num_anchors],
        'act': torch.cat(acts)[:num_anchors],
        # v3：锚点目标 = 真实下一 latent（idea 式 13 的 sg(z⁺−z)，
        # 无偏动力学回放）；'pred' 保留仅作分析对照
        'next': torch.cat(nexts)[:num_anchors],
        'pred': torch.cat(preds)[:num_anchors],
    }
    torch.save(anchors, cfg.out)
    print(
        f"[anchor] 已导出 {anchors['ctx'].shape[0]} 条锚点 -> {cfg.out} "
        f"(ctx {tuple(anchors['ctx'].shape)}, pred {tuple(anchors['pred'].shape)})"
    )


if __name__ == '__main__':
    run()
