"""bn_recal.py — BN running stats 重校准探针（不改任何权重，只刷新统计量）。

背景：projector/pred_proj 含 BN，其 running_mean/var 是 buffer，
随新任务微调漂向新数据分布；而评估/规划在 eval 模式用 running stats
→ 旧任务的预测被系统性偏移（疑似 T1 在所有干预臂中均无法保持的元凶）。

做法：加载训练后的模型，置 train 模式（BN 前向时更新 running stats），
在指定数据集上空跑 N 个 batch（torch.no_grad，无优化器、权重不动），
另存为新检查点用于重评估。

用法（在仓库根目录）：
  python cl_protocol/bn_recal.py \
    data.dataset.name=pusht_t2_replay.lance \
    +policy=<训练后的 weights_epoch_30.pt 绝对路径> \
    +num_batches=200
产物：<policy 同级目录>/../<原文件夹名>_bnrecal/weights_epoch_30.pt + config.json
"""
import os
import shutil
from pathlib import Path

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


@hydra.main(version_base=None, config_path='../scripts/train/config', config_name='lewm')
def run(cfg):
    assert cfg.get('policy'), '必须指定 +policy=<权重路径>'
    num_batches = int(cfg.get('num_batches', 200))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = Path(cfg.policy)

    # dataset（与 lewm.py 同款加载）
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    print(f'Loading dataset "{dataset_name}"')
    dataset = swm.data.load_dataset(dataset_name, transform=None, **dataset_cfg)
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
        dataset, batch_size=256, shuffle=True, num_workers=4, drop_last=True
    )

    model = swm.wm.utils.load_pretrained(str(policy))
    model = model.to(device)

    # 重校准：train 模式让 BN 更新 running stats，no_grad 保证权重不动
    # 2026-09-01 修复①：包装器源码确认 encode() 过 projector(BN)、predict() 过
    # pred_proj(BN)——两个 BN 层分别死在两条路径上，必须两条前向都跑才刷得全
    # 修复②：predict 只取前 history_size 帧（与训练 lejepa_forward 完全一致），
    # 否则全长序列撞 pos_embedding 形状（4 vs 3）
    hist = getattr(model.predictor, 'num_frames', 3)
    model.train()
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch['action'] = torch.nan_to_num(batch['action'], 0.0)
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            info = model.encode(batch)
            _ = model.predict(
                info['emb'][:, :hist], info['act_emb'][:, :hist]
            )
            n += 1
            if n >= num_batches:
                break

    out_dir = policy.parent.parent / f'{policy.parent.name}_bnrecal'
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / policy.name)
    shutil.copy(policy.parent / 'config.json', out_dir / 'config.json')
    print(f'[BN重校准] {n} 个 batch 刷新完毕 -> {out_dir / policy.name}')
    print('[BN重校准] 权重未动，仅 running stats 更新')


if __name__ == '__main__':
    run()
