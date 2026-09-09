import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class FreezeEncoderCallback(Callback):
    """冻结编码器的配套件：强制编码器永远处于 eval 模式。

    requires_grad=False 只冻权重；BN 的 running_mean/var 是 buffer，
    train 模式下每个 forward 都会随新任务数据更新（评估时用的正是
    running stats）→ "冻结"的编码器有效函数照样漂移。
    本回调在每个训练 epoch 开始（及验证后返回训练）时把编码器重新
    钉回 eval 模式：前向恒用 running stats、buffer 不再更新，
    latent 坐标系才真正固定。
    """

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.model.encoder.eval()

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.model.encoder.eval()


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    # [子空间正则] L_sub（idea 式 10/11；Sub-JEPA 原版语义：J 个冻结随机
    # 正交子空间投影 P_j 作用在编码器嵌入 z 上，约束子空间内均值→0、
    # 协方差→I，比全空间高斯约束柔和。定位：M3 归因「T1 漂移残余通道 =
    # 编码器」的正面对决实验。用法：+loss.sub_weight=0.1）
    # 投影矩阵固定种子生成、首次使用时懒建——不进 state_dict，
    # 不改变检查点格式，warm_start 兼容性和之前完全一致
    w_sub = cfg.loss.get('sub_weight', 0.0)
    if w_sub > 0:
        sub_P = getattr(self, '_sub_P', None)
        J = int(cfg.loss.get('sub_num', 16))   # Sub-JEPA PushT 配置
        ds = int(cfg.loss.get('sub_dim', 12))  # 192/16
        d = emb.size(-1)
        if sub_P is None or sub_P.size(-1) != d:
            g = torch.Generator().manual_seed(42)
            Q, _ = torch.linalg.qr(
                torch.randn(d, J * ds, generator=g)
            )  # (d, J*ds) 列正交
            sub_P = Q.T.reshape(J, ds, d)  # (J, ds, d)，行正交
            self._sub_P = sub_P
        sub_P = self._sub_P.to(emb.device)
        z = emb.reshape(-1, d).float()                      # (N, d)
        u = torch.einsum('nd,jsd->njs', z, sub_P)           # (N, J, ds)
        mu = u.mean(dim=0)                                  # (J, ds)
        uc = u - mu
        cov = torch.einsum('njs,njt->jst', uc, uc) / max(1, u.size(0) - 1)
        eye = torch.eye(ds, device=emb.device)
        output['sub_loss'] = (
            mu.pow(2).sum(-1) + (cov - eye).pow(2).sum((-2, -1))
        ).mean()
        output['loss'] = output['loss'] + w_sub * output['sub_loss']

    # [差分动力学] DiffPredictor 附加损失（idea 式 17/18；原版 Predictor 无此属性，自动跳过）
    pred_module = getattr(self.model, 'predictor', None)
    last_delta = getattr(pred_module, 'last_delta', None)
    if last_delta is not None:
        w_delta = cfg.loss.get('delta_weight', 0.0)
        w_sparse = cfg.loss.get('sparse_weight', 0.0)
        if w_delta > 0:
            # 与 L_pred 同约定：端到端、无 stop-gradient；
            # 注意 m⊙Δ 在 pred_proj 之前的空间，这里假设 pred_proj 近似恒等
            # （两端同为 BN 归一化的 192 维嵌入空间），消融会检验该假设
            true_delta = tgt_emb - ctx_emb
            output['delta_loss'] = (
                pred_module.last_mask * last_delta - true_delta
            ).abs().mean()
            output['loss'] = output['loss'] + w_delta * output['delta_loss']
        if w_sparse > 0:
            output['sparse_loss'] = pred_module.last_mask.mean()
            output['loss'] = output['loss'] + w_sparse * output['sparse_loss']

    # [适配器门控] AdapterPredictor 附加门控熵损失（无 last_pi 的预测器自动跳过）
    last_pi = getattr(pred_module, 'last_pi', None)
    if last_pi is not None:
        w_ent = cfg.loss.get('ent_weight', 0.0)
        if w_ent > 0:
            # 负熵 Σ π·logπ：压低 H(π)，鼓励门控做决策，
            # 防止塌缩成均匀分布（均匀分布 = 适配器库形同虚设）
            output['ent_loss'] = (
                last_pi * (last_pi + 1e-8).log()
            ).sum(-1).mean()
            output['loss'] = output['loss'] + w_ent * output['ent_loss']

    # [锚点记忆] L_anchor：旧任务锚点上的 L1 约束（idea 式 13；无锚点库自动跳过）
    # v3：目标 = 真实下一 latent z⁺（无偏动力学回放），优先读 'next' 键；
    # 旧版自蒸馏锚点（'pred' 键）仅兼容兜底。配合 freeze_encoder 使用：
    # 编码器冻结后 latent 坐标系固定，z⁺ 目标长期有效
    anchor = getattr(self, '_anchor_bank', None)
    if anchor is not None:
        w_anchor = cfg.loss.get('anchor_weight', 0.0)
        if w_anchor > 0:
            z_a, a_a, p_a = anchor
            if z_a.device != emb.device:  # 首次使用时懒搬到 GPU（一次性）
                z_a = z_a.to(emb.device)
                a_a = a_a.to(emb.device)
                p_a = p_a.to(emb.device)
                self._anchor_bank = (z_a, a_a, p_a)
            n_anchor = min(128, z_a.size(0))
            idx = torch.randint(0, z_a.size(0), (n_anchor,), device=emb.device)
            pred_a = self.model.predict(z_a[idx], a_a[idx])
            output['anchor_loss'] = (pred_a - p_a[idx]).abs().mean()
            output['loss'] = output['loss'] + w_anchor * output['anchor_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    # [热启动] 顺序微调：从上一任务检查点初始化（用法：+warm_start=<绝对路径.pt>）
    # strict=False：允许原版 LeWM 权重热启动差分版（新增头随机初始化），
    # 缺失/多余参数会打印出来，注意检查是否符合预期
    if cfg.get('warm_start'):
        sd = torch.load(cfg.warm_start, map_location='cpu')
        # [漂移触发增长] 适配器槽数扩展的兼容加载（M3）：
        # strict=False 只容忍缺/多键，不容忍形状不匹配——旧检查点 K_old 槽
        # 切片拷进新 K 张量的前 K_old 槽（新槽保留模型初始化值：
        # A=0 / B=randn*0.01，与 grow_adapter 语义一致）；
        # gate 末层形状不符直接丢弃（门控冻结时恒用均匀 π，无影响）
        pred = getattr(world_model, 'predictor', None)
        for key in ('predictor.adapter_A', 'predictor.adapter_B'):
            if key in sd and pred is not None:
                cur = getattr(pred, key.split('.')[-1])
                if sd[key].shape != cur.shape:
                    k_old = sd[key].shape[0]
                    grown = cur.detach().clone()
                    grown[:k_old] = sd[key]
                    sd[key] = grown
                    print(
                        f'[grow] {key}: 槽数 {k_old} → {cur.shape[0]}，'
                        f'旧槽已切片拷入，新槽保留初始化'
                    )
        for key in ('predictor.gate.2.weight', 'predictor.gate.2.bias'):
            if key in sd and pred is not None:
                cur = dict(pred.gate.named_parameters())[
                    '2.' + key.rsplit('.', 1)[-1]
                ]
                if sd[key].shape != cur.shape:
                    sd.pop(key)
                    print(f'[grow] {key}: 形状不符已丢弃（门控冻结，无影响）')
        missing, unexpected = world_model.load_state_dict(sd, strict=False)
        print(f'[warm_start] loaded from {cfg.warm_start}')
        if missing:
            print(f'[warm_start] 随机初始化的新参数: {missing}')
        if unexpected:
            print(f'[warm_start] 被忽略的多余参数: {unexpected}')

    # [旧适配器冻结] 漂移触发增长后冻结旧槽（用法：+freeze_old_adapters=N，
    # N = 触发前已有的槽数；梯度 hook 置零旧槽，主干与其余模块照常训练）
    n_freeze_old = int(cfg.get('freeze_old_adapters', 0))
    if n_freeze_old > 0:
        world_model.predictor.freeze_old_adapters(n_freeze_old)
        print(
            f'[freeze_old_adapters] 已冻结前 {n_freeze_old} 个适配器槽；'
            f'当前总槽数 {world_model.predictor.num_adapters}'
        )

    # [锚点记忆] 加载旧任务锚点库（用法：+anchor_path=<dump_anchors.py 产物.pt>，
    # 配合 +loss.anchor_weight=0.1；不指定则锚点损失自动跳过）
    anchor_bank = None
    if cfg.get('anchor_path'):
        anchor_bank = torch.load(cfg.anchor_path, map_location='cpu')
        print(
            f"[anchor] 已加载锚点库 {cfg.anchor_path}: "
            f"{anchor_bank['ctx'].shape[0]} 条"
        )

    # [编码器冻结] 顺序微调时冻结视觉编码器（用法：+freeze_encoder=true）
    # 依据：任务漂移在动力学/目标层，视觉表征跨任务共享；
    # vanilla 链 T1 pred_error 0.03→0.65 的漂移是编码器继续训练自致，
    # 冻结后 latent 坐标系固定，锚点蒸馏才有意义
    if cfg.get('freeze_encoder', False):
        n_frozen = 0
        for name, p in world_model.named_parameters():
            if name.startswith('encoder'):
                p.requires_grad = False
                n_frozen += p.numel()
        # 关键：requires_grad 冻不住 BN running stats（是 buffer 不是 parameter），
        # 且 Lightning 每 epoch 重调 model.train() 会覆盖回调里设的 eval()。
        # 防弹做法：先 eval()，再把实例的 train 方法锁死成空操作——
        # 无论谁调 train()，编码器都永远用 running stats、永不更新 buffer
        world_model.encoder.eval()
        _enc = world_model.encoder
        _enc.train = lambda mode=True: _enc
        print(f'[freeze_encoder] 已冻结编码器参数 {n_frozen / 1e6:.1f}M + 锁死 eval 模式')

    # [主干动力学冻结] T2 只学适配器（用法：+freeze_dynamics=true）
    # 对应 idea「适配器吸收任务漂移」：transformer / delta_head / mask_head / gate
    # 全部冻结，梯度只能进 adapter_A / adapter_B（配合 freeze_gate 门控锁均匀），
    # T1 的动力学主干物理上不可能被改写 → 检验遗忘主通道是否在预测器主干。
    # 编码器保持可训练（冻结臂已证明 T2 的 darken 漂移需要视觉可塑性）。
    if cfg.get('freeze_dynamics', False):
        n_frozen = 0
        for name, p in world_model.named_parameters():
            if name.startswith((
                'predictor.transformer', 'predictor.delta_head',
                'predictor.mask_head', 'predictor.gate',
                'predictor.pos_embedding',
            )):
                p.requires_grad = False
                n_frozen += p.numel()
        trainable = [
            n for n, p in world_model.predictor.named_parameters()
            if p.requires_grad
        ]
        print(
            f'[freeze_dynamics] 已冻结主干动力学 {n_frozen / 1e6:.2f}M；'
            f'预测器仍可训练参数: {trainable}'
        )

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )
    if anchor_bank is not None:
        # v3 锚点含真实下一 latent（'next'），旧版只有自蒸馏目标（'pred'）
        anchor_target = anchor_bank.get('next', anchor_bank['pred'])
        world_model._anchor_bank = (
            anchor_bank['ctx'], anchor_bank['act'], anchor_target
        )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    save_ckpt_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[save_ckpt_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == '__main__':
    run()
