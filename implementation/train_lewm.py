import os
import json
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


def configure_mechanism(cfg):
    """V2 is opt-in and saved in model config; legacy runs keep legacy semantics."""
    version = int(cfg.get('mechanism_version', 1))
    if version not in (1, 2):
        raise ValueError('mechanism_version must be 1 (legacy) or 2')
    if version == 1:
        return
    if cfg.get('anchor_path'):
        raise ValueError('Legacy cached latent anchors do not fix the full input path; '
                         'do not reuse them in the repaired component comparison')
    target = cfg.model.predictor._target_
    prefix, name = target.rsplit('.', 1)
    if name not in ('Predictor', 'DiffPredictor', 'AdapterPredictor',
                    'PredictorV2', 'DiffPredictorV2', 'AdapterPredictorV2'):
        raise ValueError('Unsupported predictor in mechanism V2')
    with open_dict(cfg):
        cfg.model.predictor._target_ = prefix + '.' + (name if name.endswith('V2') else name + 'V2')
        # 两个变化同样用于绝对预测对照：逐样本归一化，移除残差之后的投影。
        cfg.model.pred_proj = {'_target_': 'torch.nn.Identity'}
        if cfg.model.get('projector'):
            if cfg.model.projector._target_.rsplit('.', 1)[-1] != 'MLP':
                raise ValueError('V2 expects the configured per-frame MLP projector')
            cfg.model.projector.norm_fn = {'_target_': 'torch.nn.LayerNorm', '_partial_': True}


def scheduler_interval(cfg):
    """Explicit single-factor override; historical version defaults stay intact."""
    default = 'step' if int(cfg.get('mechanism_version', 1)) == 2 else 'epoch'
    interval = cfg.get('scheduler_interval', default)
    if interval not in ('epoch', 'step'):
        raise ValueError('scheduler_interval must be epoch or step')
    return interval


def seed_initialization(cfg):
    return bool(cfg.get('seed_initialization', int(cfg.get('mechanism_version', 1)) == 2))


class LearningRateTrace(Callback):
    """Record actual optimizer LR before updates, independently of scheduler names."""

    def __init__(self, run_dir, interval, seed):
        self.path = Path(run_dir) / 'learning_rate_trace.json'
        self.interval = interval
        self.seed = int(seed)
        self.rows = []
        self.schedulers = []
        self.epochs_completed = 0
        self.last_row = None

    def save(self, trainer, status):
        data = dict(status=status, scheduler_interval=self.interval, seed=self.seed,
                    global_step=int(trainer.global_step), epochs_completed=self.epochs_completed,
                    schedulers=self.schedulers, records=self.rows, last_batch=self.last_row)
        self.path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

    def on_train_start(self, trainer, pl_module):
        import inspect
        for item in trainer.lr_scheduler_configs:
            cls = type(item.scheduler)
            entry = dict(type=cls.__module__ + '.' + cls.__name__, interval=item.interval,
                         frequency=item.frequency)
            try:
                entry['source'] = inspect.getsource(cls)
            except (OSError, TypeError):
                entry['source'] = None
            self.schedulers.append(entry)
        if not self.schedulers or any(s['interval'] != self.interval or s['frequency'] != 1
                                      for s in self.schedulers):
            raise RuntimeError('Installed scheduler wiring differs from requested interval/frequency')
        print('[LR_PROTOCOL]', json.dumps(dict(interval=self.interval, seed=self.seed,
              schedulers=[{k: v for k, v in s.items() if k != 'source'} for s in self.schedulers])), flush=True)
        self.save(trainer, 'running')

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.last_row = dict(step=int(trainer.global_step), epoch=int(trainer.current_epoch) + 1,
                            lr=[float(g['lr']) for opt in trainer.optimizers for g in opt.param_groups])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict):
            for key in ('loss', 'pred_loss', 'sigreg_loss', 'embedding_std', 'target_variance'):
                value = outputs.get(key)
                if torch.is_tensor(value) and value.numel() == 1:
                    self.last_row[key] = float(value.detach().float())
        if self.last_row['step'] % 100 == 0 or batch_idx == 0:
            self.rows.append(dict(self.last_row))
            print('[LR_TRACE]', json.dumps(self.last_row), flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        self.epochs_completed = int(trainer.current_epoch) + 1
        self.save(trainer, 'running')

    def on_train_end(self, trainer, pl_module):
        self.save(trainer, 'complete')


def validate_mechanism(model, cfg):
    version = int(cfg.get('mechanism_version', 1))
    if int(getattr(model.predictor, 'mechanism_version', 1)) != version:
        raise ValueError('Predictor version and training loss protocol disagree')
    if version != 2:
        return
    if not isinstance(model.pred_proj, torch.nn.Identity):
        raise ValueError('V2 residual output must not pass through a further projection')
    if int(getattr(model.predictor, 'mechanism_version', 0)) != 2:
        raise ValueError('V2 predictor implementation was not installed')
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('V2 training rejects BatchNorm: flattened batch/time mixes future frames')


def gate_entropy_loss(pi, version):
    if version == 1:
        return (pi * (pi + 1e-8).log()).sum(-1).mean()  # 原式保留供复现
    return -(pi * pi.clamp_min(1e-8).log()).sum(-1).mean()


def read_fixed_action_stats(cfg):
    path = cfg.get('fixed_action_stats')
    if not path:
        if cfg.get('old_path_diagnostic', False):
            raise ValueError('Full-old-path diagnostic requires fixed_action_stats')
        return None
    from stable_worldmodel.wm.lewm.module import FixedActionNormalizer
    stats = json.loads(Path(path).read_text(encoding='utf-8'))
    if not stats.get('source') or not stats.get('policy'):
        raise ValueError('Action statistics require actual policy and source provenance')
    canonical = lambda p: str(p).replace('\\', '/').split('/checkpoints/')[-1]
    if cfg.get('old_path_diagnostic', False) and canonical(stats['policy']) != canonical(cfg.warm_start):
        raise ValueError('Diagnostic action statistics must belong to the warm-start model')
    scaler = FixedActionNormalizer(stats['mean'], stats['scale'])
    with open_dict(cfg):
        cfg.fixed_action_normalization = stats
    return scaler


def wrap_old_path_diagnostic(model, cfg, old_slots):
    """新诊断结构；从严格匹配的旧检查点增长一槽，不自动识别条件。"""
    from stable_worldmodel.wm.lewm.module import FrozenOldPathWorldModel
    stats = cfg.fixed_action_normalization
    base_config = OmegaConf.to_container(cfg.model, resolve=True)
    wrapped = FrozenOldPathWorldModel(
        model, old_slots, stats['mean'], stats['scale'], stats['source']
    )
    with open_dict(cfg):
        cfg.model = {
            '_target_': 'stable_worldmodel.wm.lewm.module.FrozenOldPathWorldModel',
            'base': base_config, 'old_slots': old_slots,
            'action_mean': list(stats['mean']), 'action_scale': list(stats['scale']),
            'normalization_source': stats['source'],
        }
    return wrapped


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
    这仅固定encoder；后续projector/action_encoder仍可改变表示，不能称完整坐标系冻结。
    """

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.model.encoder.eval()

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.model.encoder.eval()


def load_warm_start(model, state_dict):
    """Use one expansion path for training and zero-step validation."""
    predictor = getattr(model, 'predictor', None)
    source_version = int(state_dict.get('predictor.mechanism_version', 1))
    target_version = int(getattr(predictor, 'mechanism_version', 1))
    if source_version != target_version:
        raise ValueError('Refuse implicit warm-start conversion between legacy and V2 mechanisms')
    if predictor is not None and hasattr(predictor, 'prepare_growth_state_dict'):
        state_dict = predictor.prepare_growth_state_dict(state_dict, 'predictor.')
    return model.load_state_dict(state_dict, strict=False)


class FreezeOldAdaptersCallback(Callback):
    """Install optimizer guards after Lightning creates its real optimizer."""

    def __init__(self):
        self.handles = []

    def on_fit_start(self, trainer, pl_module):
        predictor = pl_module.model.predictor
        if not getattr(predictor, 'n_frozen_adapters', 0):
            return
        if len(trainer.optimizers) != 1:
            raise RuntimeError('Frozen adapter training expects one model optimizer')
        self.handles = predictor.bind_frozen_optimizer(trainer.optimizers[0])
        print('[FROZEN_OPTIMIZER_BOUND] old slots protected after optimizer steps',
              flush=True)

    def on_fit_end(self, trainer, pl_module):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def episode_subsets(dataset, manifest_path, dataset_name):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['dataset'] != dataset_name:
        raise ValueError('Split manifest dataset mismatch')
    groups = [set(manifest[k]) for k in ('train_episodes', 'validation_episodes', 'test_episodes')]
    if any(not g for g in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Empty or overlapping episode groups')
    if set.union(*groups) != set(range(len(dataset.lengths))):
        raise ValueError('Episode partition does not match dataset')
    if list(map(int, dataset.lengths)) != manifest['episode_lengths']:
        raise ValueError('Dataset episode lengths changed')
    indices = [[i for i, (ep, _) in enumerate(dataset.clip_indices) if int(ep) in g]
               for g in groups[:2]]
    if any(not x for x in indices):
        raise ValueError('Empty training/validation windows')
    print('[EPISODE_SPLIT_OK]', json.dumps(dict(train_windows=len(indices[0]),
          validation_windows=len(indices[1]), test_episodes=len(groups[2]))), flush=True)
    return tuple(torch.utils.data.Subset(dataset, x) for x in indices)


class RealDataSmokeCallback(Callback):
    """Bounded real-data plumbing check; never reports task performance."""

    def __init__(self, steps, run_dir, run_name, model_cfg, comparison=False):
        self.steps, self.run_dir = steps, run_dir
        self.run_name, self.model_cfg = run_name, model_cfg
        self.batches = self.backwards = 0
        self.losses = []
        self.comparison = comparison
        self.metric_rows = []
        self.validation_rows = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if getattr(trainer, 'sanity_checking', False):
            return
        row = {k: float(v.detach().cpu()) for k, v in outputs.items()
               if k in ('pred_loss', 'sigreg_loss', 'embedding_std', 'target_variance')}
        row.update(step=int(trainer.global_step), batch_size=int(batch['pixels'].shape[0]))
        self.validation_rows.append(row)

    def on_train_start(self, trainer, pl_module):
        self.initial = {k: v.detach().cpu().clone()
                        for k, v in pl_module.model.named_parameters()}

    def on_after_backward(self, trainer, pl_module):
        grads = [p.grad for p in pl_module.model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Smoke check: missing/nonfinite gradients')
        if not any(torch.count_nonzero(g).item() for g in grads):
            raise RuntimeError('Smoke check: all gradients zero')
        self.backwards += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        value = float(loss.detach().cpu())
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError('Smoke check: nonfinite loss')
        self.losses.append(value)
        self.batches += 1
        if not self.comparison or self.batches == 1 or self.batches % 100 == 0:
            metrics = {k: float(v.detach().cpu()) for k, v in outputs.items()
                       if k in ('pred_loss', 'sigreg_loss', 'delta_loss', 'sparse_loss',
                                'embedding_std', 'target_variance')}
            metrics.update(step=self.batches, loss=value,
                           lr=float(trainer.optimizers[0].param_groups[0]['lr'])
                           if getattr(trainer, 'optimizers', None) else None)
            self.metric_rows.append(metrics)
            label = 'COMPARE_STEP' if self.comparison else 'REAL_SMOKE_STEP'
            print(f'[{label}] {self.batches}/{self.steps} ' + json.dumps(metrics), flush=True)

    def on_train_end(self, trainer, pl_module):
        if (trainer.global_step != self.steps or self.batches != self.steps
                or self.backwards != self.steps):
            raise RuntimeError('Smoke check: update/batch/backward count mismatch')
        model = pl_module.model
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Smoke check: nonfinite saved state')
        changed = [k for k, v in model.named_parameters()
                   if not torch.equal(v.detach().cpu(), self.initial[k])]
        if not changed:
            raise RuntimeError('Smoke check: no parameter updated')
        filename = 'weights_compare.pt' if self.comparison else 'weights_smoke.pt'
        save_pretrained(model, run_name=self.run_name, config=self.model_cfg,
                        filename=filename)
        saved_cfg = OmegaConf.create(json.loads((self.run_dir / 'config.json').read_text()))
        reloaded = hydra.utils.instantiate(saved_cfg).cpu()
        saved_state = torch.load(self.run_dir / filename, map_location='cpu', weights_only=True)
        reloaded.load_state_dict(saved_state, strict=True)
        if not all(torch.equal(v, reloaded.state_dict()[k]) for k, v in state.items()):
            raise RuntimeError('Smoke check: save/load roundtrip mismatch')
        result = dict(status='passed', global_step=trainer.global_step,
                      batches=self.batches, backwards=self.backwards, losses=self.losses,
                      changed_parameter_tensors=len(changed), checkpoint_roundtrip=True,
                      metrics=self.metric_rows, validation=self.validation_rows,
                      scope='Fixed-budget single-seed training; behavior measured separately'
                      if self.comparison else 'Real-data training plumbing only; no task performance conclusion')
        receipt = 'comparison_train.json' if self.comparison else 'real_smoke_check.json'
        (self.run_dir / receipt).write_text(
            json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print('[COMPARE_TRAIN_OK]' if self.comparison else '[REAL_SMOKE_OK]',
              json.dumps({k: v for k, v in result.items() if k not in ('losses', 'metrics')}), flush=True)


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
            print(f'[EPOCH_DONE] {trainer.current_epoch + 1}/{trainer.max_epochs}', flush=True)
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
    if int(cfg.get('mechanism_version', 1)) == 2 or cfg.get('record_lr_trace', False):
        output['embedding_std'] = emb.detach().float().flatten(0, 1).std(0, unbiased=False).mean()
        output['target_variance'] = tgt_emb.detach().float().flatten(0, 1).var(0, unbiased=False).mean()

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
            # V2 在最终输出空间监督差分；不假设外层非线性投影近似恒等。
            # V1 分支仅复现历史目标，不能作为修正差分公式的实验。
            true_delta = tgt_emb - ctx_emb
            estimated_delta = (pred_emb - ctx_emb if int(cfg.get('mechanism_version', 1)) == 2
                               else pred_module.last_mask * last_delta)
            output['delta_loss'] = (
                estimated_delta - true_delta
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
            # V2 最小化正熵以鼓励明确选择；V1 保留相反符号供历史复现。
            gate_version = int(getattr(pred_module, 'gate_protocol_version',
                                       cfg.get('mechanism_version', 1)))
            output['ent_loss'] = gate_entropy_loss(last_pi, gate_version)
            output['loss'] = output['loss'] + w_ent * output['ent_loss']

    # [锚点记忆] L_anchor：旧任务锚点上的 L1 约束（idea 式 13；无锚点库自动跳过）
    # v3：目标 = 真实下一 latent z⁺（无偏动力学回放），优先读 'next' 键；
    # 旧版自蒸馏锚点（'pred' 键）仅兼容兜底。配合 freeze_encoder 使用：
    # 仅冻encoder不能固定projector/action_encoder；旧锚点只约束缓存后的局部路径。
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
    configure_mechanism(cfg)
    interval = scheduler_interval(cfg)
    if seed_initialization(cfg):
        pl.seed_everything(int(cfg.seed), workers=True)
    smoke_steps = int(cfg.get('smoke_steps', 0))
    comparison_steps = int(cfg.get('comparison_steps', 0))
    if smoke_steps and comparison_steps:
        raise ValueError('Choose smoke_steps or comparison_steps')
    bounded_steps = smoke_steps or comparison_steps
    if bounded_steps:
        if bounded_steps < 1 or int(cfg.get('mechanism_version', 1)) != 2 or cfg.get('warm_start'):
            raise ValueError('Smoke run requires V2, positive steps, and fresh initialization')
        with open_dict(cfg):
            cfg.trainer.max_epochs = bounded_steps if comparison_steps else 1
            cfg.trainer.max_steps = bounded_steps
            cfg.trainer.limit_train_batches = bounded_steps
            cfg.trainer.limit_val_batches = 2
            cfg.trainer.accumulate_grad_batches = 1
            cfg.trainer.devices = 1
        if comparison_steps and (not cfg.get('episode_split') or not cfg.get('fixed_action_stats')):
            raise ValueError('Comparison requires episode split and fixed action coordinates')
        smoke_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.output_model_name)
        if smoke_dir.exists():
            raise ValueError('Use a fresh smoke run name; existing output will not be overwritten')
    print(f'[MECHANISM_PROTOCOL] version={int(cfg.get("mechanism_version", 1))}; '
          f'old_path_diagnostic={bool(cfg.get("old_path_diagnostic", False))}', flush=True)
    fixed_scaler = read_fixed_action_stats(cfg)
    diagnostic = bool(cfg.get('old_path_diagnostic', False))
    if diagnostic and (not cfg.get('warm_start') or any(cfg.get(k, False) for k in
            ('freeze_encoder', 'freeze_dynamics', 'freeze_old_adapters', 'anchor_path'))):
        raise ValueError('old_path_diagnostic requires warm_start and replaces partial-freeze/anchor flags')
    if diagnostic or int(cfg.get('mechanism_version', 1)) == 2:
        previous = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.output_model_name, 'config.json')
        if previous.is_file():
            saved = json.loads(previous.read_text(encoding='utf-8'))
            expected_wrapper = saved.get('_target_', '').endswith('.FrozenOldPathWorldModel')
            expected_v2 = saved.get('predictor', {}).get('_target_', '').endswith('V2')
            if expected_wrapper != diagnostic or (not diagnostic and not expected_v2):
                raise ValueError('Use a new output_model_name; refuse to overwrite a different model protocol')
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

            normalizer = (spt.data.transforms.WrapTorchTransform(fixed_scaler, col, col)
                          if col == 'action' and fixed_scaler is not None
                          else get_column_normalizer(dataset, col, col))
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    if cfg.get('episode_split'):
        train_set, val_set = episode_subsets(dataset, cfg.episode_split, dataset_name)
    else:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen)

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
    validate_mechanism(world_model, cfg)

    # [热启动] 顺序微调：从上一任务检查点初始化（用法：+warm_start=<绝对路径.pt>）
    # strict=False：允许原版 LeWM 权重热启动差分版（新增头随机初始化），
    # 缺失/多余参数会打印出来，注意检查是否符合预期
    if cfg.get('warm_start'):
        sd = torch.load(cfg.warm_start, map_location='cpu')
        # 诊断先严格恢复旧槽数，之后仅增长一槽；不能在加载时先隐式扩容。
        missing, unexpected = (world_model.load_state_dict(sd, strict=True) if diagnostic
                               else load_warm_start(world_model, sd))
        print(f'[warm_start] loaded from {cfg.warm_start}')
        if missing:
            print(f'[warm_start] 随机初始化的新参数: {missing}')
        if unexpected:
            print(f'[warm_start] 被忽略的多余参数: {unexpected}')
        if (diagnostic or int(cfg.get('mechanism_version', 1)) == 2) and (missing or unexpected):
            raise ValueError('Repaired/diagnostic warm start requires a matching checkpoint architecture')

    if diagnostic:
        if not hasattr(world_model.predictor, 'grow_adapter') or not world_model.predictor.freeze_gate:
            raise ValueError('Full-old-path diagnostic starts from a uniform AdapterPredictor checkpoint')
        old_slots = world_model.predictor.num_adapters
        world_model.predictor.grow_adapter()
        with open_dict(cfg):
            cfg.model.predictor.num_adapters = world_model.predictor.num_adapters
        world_model = wrap_old_path_diagnostic(world_model, cfg, old_slots)
        print('[OLD_PATH_DIAGNOSTIC] known condition only; all shared paths eval/frozen; '
              f'old_slots={old_slots}, trainable=last adapter A/B; fixed normalization; '
              f'new_scalar_parameters={2 * world_model.predictor.output_dim * world_model.predictor.rank}', flush=True)

    if hasattr(world_model.predictor, 'gate_protocol_version'):
        gate_version = int(world_model.predictor.gate_protocol_version)
        if gate_version != 2:
            raise ValueError('Unsupported standalone gate protocol')
        print('[ADAPTER_ROUTE_PROTOCOL]', json.dumps(dict(gate_version=gate_version,
              entropy_sign='positive', freeze_gate=world_model.predictor.freeze_gate,
              image_projection_unchanged=True)), flush=True)

    # [旧适配器冻结] 漂移触发增长后冻结旧槽（用法：+freeze_old_adapters=N，
    # N = 触发前已有槽数；梯度屏蔽 + 优化器步后恢复旧槽，主干仍可训练）
    n_freeze_old = int(cfg.get('freeze_old_adapters', 0))
    if n_freeze_old > 0:
        world_model.predictor.freeze_old_adapters(n_freeze_old)
        print(
            f'[freeze_old_adapters] 仅冻结前 {n_freeze_old} 个适配器槽，非完整旧功能；'
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
    # 这只冻结encoder；projector/action_encoder仍能变，不等于固定完整输入坐标。
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

    total_steps = bounded_steps or cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': interval,
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
        anchor_target = anchor_bank['next'] if 'next' in anchor_bank else anchor_bank['pred']
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
        epoch_interval=int(cfg.get('save_epoch_interval', 1)),
    )

    callbacks = [FreezeOldAdaptersCallback(), save_ckpt_callback]
    if cfg.get('record_lr_trace', False):
        callbacks.append(LearningRateTrace(run_dir, interval, cfg.seed))
    if bounded_steps:
        callbacks.append(RealDataSmokeCallback(bounded_steps, run_dir, cfg.output_model_name,
                                              cfg.model, comparison=bool(comparison_steps)))
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
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
