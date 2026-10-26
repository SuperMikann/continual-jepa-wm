import copy

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (
            rearrange(t, 'b t (h d) -> b h t d', h=self.heads) for t in qkv
        )
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=causal
        )
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.smoothed_dim = smoothed_dim
        self.emb_dim = emb_dim
        self.mlp_scale = mlp_scale
        self.patch_embed = nn.Conv1d(
            input_dim, smoothed_dim, kernel_size=1, stride=1
        )
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class Predictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, input_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class DiffPredictor(nn.Module):
    """差分动力学预测器（方法里程碑 0，对应 idea 文档式 2/8/9）。

    与 Predictor 接口完全一致：forward(x, c) -> (B, T, d)，
    此类保留历史实现；原 LeWM.predict 还会执行 pred_proj，不能把内部
    残差式等同于完整模型输出。修正版使用 DiffPredictorV2 + Identity pred_proj。

    差别：transformer 输出 h 经 delta_head 得到 latent 差分 Δ，
    经 mask_head 得到维度级更新掩码 m = sigmoid(·)，
    最终返回 x + m ⊙ Δ，而不是直接回归绝对下一 latent。

    初始化约定：
      - delta_head 零初始化 → 开局预测 = 恒等（z'≈z，持续性先验），
        仅指此预测器内部；完整模型是否恒等取决于外层 pred_proj；
      - mask_head 偏置 +2 → m≈0.88 起步（接近全维更新），
        稀疏化交给 L_sparse 在训练中学会。

    self.last_delta / self.last_mask 暴露给训练脚本计算 L_Δ / L_sparse。
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, input_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            self.output_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )
        d = self.output_dim
        self.delta_head = nn.Linear(d, d)
        self.mask_head = nn.Linear(d, d)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.constant_(self.mask_head.bias, 2.0)

    def forward(self, x, c):
        """
        x: (B, T, d) 历史 latent
        c: (B, T, act_dim) 动作条件
        """
        T = x.size(1)
        h = x + self.pos_embedding[:, :T]
        h = self.dropout(h)
        h = self.transformer(h, c)
        delta = self.delta_head(h)
        m = torch.sigmoid(self.mask_head(h))
        self.last_delta = delta
        self.last_mask = m
        return x + m * delta


class AdapterPredictor(DiffPredictor):
    """低秩适配器库 + 上下文门控（方法里程碑 1，对应 idea 文档式 3/4）。

    继承 DiffPredictor，在 Δ 路径上叠加 K 个低秩适配器：
        h      = transformer(x + pos, c)              # 主干不动
        Δ_base = delta_head(h)                        # 里程碑 0 已有
        Δ_adp  = Σ_k π_k · A_k (B_k h)                # 低秩适配器库加权和
        返回 x + m ⊙ (Δ_base + Δ_adp)

    门控：π = softmax(g_ψ(c̄) / tau)，c̄ = 输入序列的均值池化表征。
    门控用上下文表征而非任务 ID —— 为里程碑 3 的漂移触发增长
    预留接口（新任务 = 门控置信度低 = 触发新适配器）。

    初始化约定（与 DiffPredictor 同哲学：改造不扰动已有学习动态）：
      - adapter_A 全零 → 开局适配器输出恒为 0，严格退化为 DiffPredictor；
      - adapter_B 小随机（std=0.01）提供对称性破缺。

    self.last_pi 暴露给训练脚本计算门控熵损失 L_ent 与日志；
    last_delta / last_mask 语义不变（last_delta 含适配器项）。
    注意：B_k 实际形状为 (r, d)，以 h（已含动作条件）为输入，
    是对 idea 文档 [z;a] 拼接输入的简化——h 经 ConditionalBlock
    已融合动作信息，显式拼接在消融中再检验。
    """

    def __init__(
        self,
        *args,
        num_adapters=4,
        rank=16,
        gate_hidden=64,
        gate_tau=1.0,
        freeze_gate=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        d = self.output_dim
        self.num_adapters = num_adapters
        self.rank = rank
        self.gate_tau = gate_tau
        # Legacy checkpoints use scale=1. Growth preserves old effective weights
        # by compensating for the new uniform mixture denominator.
        self.register_buffer('adapter_scale', torch.tensor(1.0))
        # 隔离消融开关：冻结门控为均匀分布 π=1/K（不学习、无梯度），
        # 用于把"适配器本身"与"门控决策"的影响切开
        self.freeze_gate = freeze_gate
        self.adapter_A = nn.Parameter(torch.zeros(num_adapters, d, rank))
        self.adapter_B = nn.Parameter(
            torch.randn(num_adapters, rank, d) * 0.01
        )
        self.gate = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, num_adapters),
        )

    def forward(self, x, c):
        """
        x: (B, T, d) 历史 latent
        c: (B, T, act_dim) 动作条件
        """
        T = x.size(1)
        h = x + self.pos_embedding[:, :T]
        h = self.dropout(h)
        h = self.transformer(h, c)
        delta = self.delta_head(h)
        # 适配器库：u = B_k h ∈ (B,T,K,r)，v = A_k u ∈ (B,T,K,d)
        u = torch.einsum('btd,krd->btkr', h, self.adapter_B)
        v = torch.einsum('btkr,kdr->btkd', u, self.adapter_A)
        # 门控：上下文均值池化 → MLP → softmax（freeze_gate 时锁死均匀）
        if self.freeze_gate:
            pi = x.new_full((x.size(0), self.num_adapters),
                            1.0 / self.num_adapters)  # (B, K)，无梯度
        else:
            ctx = x.mean(dim=1)  # (B, d)
            pi = torch.softmax(self.gate(ctx) / self.gate_tau, dim=-1)  # (B, K)
        # 仅完整旧通路诊断包装器设置此开关；默认不改变旧检查点计算。
        if getattr(self, '_diagnostic_condition', None) == 'old':
            pi = pi.clone()
            pi[:, self._diagnostic_old_slots:] = 0
        delta = delta + self.adapter_scale * torch.einsum('btkd,bk->btd', v, pi)
        m = torch.sigmoid(self.mask_head(h))
        self.last_delta = delta
        self.last_mask = m
        self.last_pi = pi
        return x + m * delta

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        # Old fixed/grown checkpoints must retain their original inference.
        # Only an explicit expansion below changes this scale.
        state_dict.setdefault(prefix + 'adapter_scale',
                              self.adapter_scale.new_tensor(1.0))
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys,
                                     error_msgs)

    def prepare_growth_state_dict(self, state_dict, prefix=''):
        """Prepare warm-start expansion without changing learned old slots.

        Used by the actual training loader and the zero-step validation.
        Uniform routing is required for function-preserving expansion.
        Same-size loading preserves legacy behavior (scale defaults to one).
        """
        sd = copy.copy(state_dict)
        akey, bkey = prefix + 'adapter_A', prefix + 'adapter_B'
        if akey not in sd and bkey not in sd:
            return sd
        if akey not in sd or bkey not in sd:
            raise ValueError('Checkpoint must contain both adapter_A and adapter_B')
        old_k = sd[akey].shape[0]
        new_k = self.num_adapters
        if (sd[bkey].shape[0] != old_k or
                sd[akey].shape[1:] != self.adapter_A.shape[1:] or
                sd[bkey].shape[1:] != self.adapter_B.shape[1:]):
            raise ValueError('Adapter checkpoint shapes differ beyond slot count')
        if old_k == new_k:
            return sd
        if not 0 < old_k < new_k:
            raise ValueError('Warm-start can expand adapter slots, not truncate them')
        if not self.freeze_gate:
            raise ValueError('Function-preserving growth requires freeze_gate=True')
        for key, parameter in ((akey, self.adapter_A), (bkey, self.adapter_B)):
            expanded = parameter.detach().to(device=sd[key].device,
                                              dtype=sd[key].dtype).clone()
            expanded[:old_k] = sd[key]
            if key == akey:
                expanded[old_k:] = 0
            sd[key] = expanded
        scale_key = prefix + 'adapter_scale'
        old_scale = sd.get(scale_key, sd[akey].new_tensor(1.0))
        sd[scale_key] = old_scale * (new_k / old_k)
        # Gate values are unused under uniform routing; retain its old rows.
        for suffix in ('weight', 'bias'):
            key = prefix + 'gate.2.' + suffix
            if key not in sd:
                continue
            parameter = getattr(self.gate[-1], suffix)
            if sd[key].shape[0] != old_k or sd[key].shape[1:] != parameter.shape[1:]:
                raise ValueError('Checkpoint gate shape is incompatible')
            expanded = torch.zeros_like(parameter, device=sd[key].device,
                                        dtype=sd[key].dtype)
            expanded[:old_k] = sd[key]
            sd[key] = expanded
        return sd

    @torch.no_grad()
    def grow_adapter(self):
        """Expand before creating the optimizer; preserve the old function.

        New A is zero. The saved mixture scale compensates K -> K+1.
        Learned gates need a separate routing design and are rejected here.
        """
        if not self.freeze_gate:
            raise ValueError('Function-preserving growth requires freeze_gate=True')
        if getattr(self, 'n_frozen_adapters', 0):
            raise RuntimeError('Grow before freezing slots and binding the optimizer')
        K, d, r = self.adapter_A.shape
        dev, dt = self.adapter_A.device, self.adapter_A.dtype
        new_A = torch.zeros(K + 1, d, r, device=dev, dtype=dt)
        new_A[:K] = self.adapter_A
        new_B = torch.randn(K + 1, r, d, device=dev, dtype=dt) * 0.01
        new_B[:K] = self.adapter_B
        self.adapter_A = nn.Parameter(new_A)
        self.adapter_B = nn.Parameter(new_B)
        old_last = self.gate[-1]
        new_last = nn.Linear(old_last.in_features, K + 1, device=dev, dtype=dt)
        new_last.weight[:K] = old_last.weight
        new_last.bias[:K] = old_last.bias
        nn.init.zeros_(new_last.weight[K:])
        nn.init.zeros_(new_last.bias[K:])
        self.gate[-1] = new_last
        self.adapter_scale.mul_((K + 1) / K)
        self.num_adapters = K + 1

    def freeze_old_adapters(self, n_old):
        """Mask old gradients; bind_frozen_optimizer also guards optimizer steps."""
        if not 0 <= n_old < self.num_adapters:
            raise ValueError('Keep at least one trainable adapter slot')
        for handle in getattr(self, '_freeze_gradient_handles', []):
            handle.remove()
        self._freeze_gradient_handles = []
        self.n_frozen_adapters = n_old
        if n_old:
            def zero_old_slots(gradient):
                gradient = gradient.clone()
                gradient[:n_old] = 0
                return gradient
            self._freeze_gradient_handles = [
                parameter.register_hook(zero_old_slots)
                for parameter in (self.adapter_A, self.adapter_B)
            ]

    def bind_frozen_optimizer(self, optimizer):
        """Restore old slices after each step, including AdamW decay/momentum.

        Snapshots are taken immediately before each step, after device moves
        and checkpoint restoration. The training callback owns hook cleanup.
        """
        n_old = getattr(self, 'n_frozen_adapters', 0)
        if not n_old:
            return []
        parameters = (self.adapter_A, self.adapter_B)
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        if not all(id(p) in optimized for p in parameters):
            raise ValueError('Optimizer must contain the current adapter parameters')
        snapshots = []

        def before_step(opt, args, kwargs):
            snapshots[:] = [p[:n_old].detach().clone() for p in parameters]

        @torch.no_grad()
        def after_step(opt, args, kwargs):
            if len(snapshots) != len(parameters):
                raise RuntimeError('Frozen-slot snapshot missing before optimizer step')
            for parameter, snapshot in zip(parameters, snapshots):
                parameter[:n_old].copy_(snapshot)
            snapshots.clear()

        return [optimizer.register_step_pre_hook(before_step),
                optimizer.register_step_post_hook(after_step)]


class _Version2:
    """明确版本边界：新结构不能静默加载旧检查点，即使 strict=False。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mechanism_version', torch.tensor(2))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        version = state_dict.get(prefix + 'mechanism_version')
        if version is None or version.numel() != 1 or int(version) != 2:
            error_msgs.append('Version-2 predictors require a version-2 checkpoint; '
                              'legacy weights need a separate explicit migration experiment')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class PredictorV2(_Version2, Predictor):
    """绝对预测对照；配合同样的逐样本投影归一化和 Identity pred_proj。"""


class DiffPredictorV2(_Version2, DiffPredictor):
    """修正差分版本；训练入口强制外层 pred_proj=Identity。"""


class AdapterPredictorV2(_Version2, AdapterPredictor):
    """逐时刻因果门控；每个位置只汇总本位置及之前的历史。"""

    def forward(self, x, c):
        T = x.size(1)
        h = self.transformer(self.dropout(x + self.pos_embedding[:, :T]), c)
        delta = self.delta_head(h)
        u = torch.einsum('btd,krd->btkr', h, self.adapter_B)
        v = torch.einsum('btkr,kdr->btkd', u, self.adapter_A)
        if self.freeze_gate:
            pi = x.new_full((x.size(0), T, self.num_adapters), 1 / self.num_adapters)
        else:
            count = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)[None, :, None]
            context = x.cumsum(dim=1) / count
            pi = torch.softmax(self.gate(context) / self.gate_tau, dim=-1)
        if getattr(self, '_diagnostic_condition', None) == 'old':
            pi = pi.clone()
            pi[..., self._diagnostic_old_slots:] = 0
        delta = delta + self.adapter_scale * torch.einsum('btkd,btk->btd', v, pi)
        self.last_delta = delta
        self.last_mask = torch.sigmoid(self.mask_head(h))
        self.last_pi = pi
        return x + self.last_mask * delta


class CausalAdapterPredictor(AdapterPredictor):
    """Only repair routing; retain the checkpoint's image/output projections.

    This does not claim that a surrounding train-mode BatchNorm is causal.
    The explicit marker requires a separately recorded checkpoint conversion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('gate_protocol_version', torch.tensor(2))

    def forward(self, x, c):
        return AdapterPredictorV2.forward(self, x, c)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        version = state_dict.get(prefix + 'gate_protocol_version')
        if version is None or version.numel() != 1 or int(version) != 2:
            error_msgs.append('Causal adapter routing requires its explicit converted checkpoint')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class FixedActionNormalizer:
    """可序列化的固定动作坐标；用于训练变换和规划的 transform/inverse_transform。"""

    def __init__(self, mean, scale):
        self.mean = torch.as_tensor(mean, dtype=torch.float64).reshape(-1).clone()
        self.scale = torch.as_tensor(scale, dtype=torch.float64).reshape(-1).clone()
        if (not self.mean.numel() or self.mean.shape != self.scale.shape
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all()
                or not (self.scale > 0).all()):
            raise ValueError('Fixed action mean/scale must be finite, same-sized, scale > 0')

    def _apply(self, value, inverse):
        tensor = torch.is_tensor(value)
        x = value if tensor else torch.as_tensor(value)
        if x.shape[-1] != self.mean.numel():
            raise ValueError('Action dimension differs from fixed normalization')
        dtype = x.dtype if x.is_floating_point() else torch.float32
        x = x.to(dtype)
        mean, scale = self.mean.to(x.device, dtype), self.scale.to(x.device, dtype)
        result = x * scale + mean if inverse else (x - mean) / scale
        return result if tensor else result.cpu().numpy()

    def transform(self, value):
        return self._apply(value, False)

    def inverse_transform(self, value):
        return self._apply(value, True)

    def __call__(self, value):
        return self.transform(value)


class FrozenOldPathWorldModel(nn.Module):
    """已知条件诊断，不是自动路由方法。

    base 必须已扩容且为均匀门控。共享路径全部冻结并始终 eval。
    old 路径屏蔽新增槽而保留增长后的系数；new 路径在旧函数上加新槽。
    训练只更新新增 A/B；优化器仍需绑定旧切片保护（见训练回调）。
    原始动作由外部管线用 action_normalizer 固定处理，禁止新任务重拟合。
    """

    def __init__(self, base, old_slots, action_mean, action_scale, normalization_source):
        super().__init__()
        if not isinstance(base.predictor, AdapterPredictor) or not base.predictor.freeze_gate:
            raise ValueError('Frozen-old-path diagnostic requires a uniform AdapterPredictor')
        if not 0 < old_slots < base.predictor.num_adapters:
            raise ValueError('Diagnostic requires old slots and at least one new slot')
        if not normalization_source:
            raise ValueError('The fixed action normalization source must be recorded')
        self.base = base
        self.old_slots = int(old_slots)
        self.normalization_source = normalization_source
        scaler = FixedActionNormalizer(action_mean, action_scale)
        self.register_buffer('action_mean', scaler.mean)
        self.register_buffer('action_scale', scaler.scale)
        self.base.requires_grad_(False)
        self.predictor.adapter_A.requires_grad_(True)
        self.predictor.adapter_B.requires_grad_(True)
        self.predictor.freeze_old_adapters(self.old_slots)
        self.set_condition('new')
        self.base.eval()

    @property
    def predictor(self):
        return self.base.predictor

    @property
    def action_normalizer(self):
        return FixedActionNormalizer(self.action_mean.cpu(), self.action_scale.cpu())

    def _apply(self, fn, recurse=True):
        # model.to(bfloat16)不应把已固定的动作坐标也量化为bfloat16。
        mean, scale = self.action_mean.clone(), self.action_scale.clone()
        super()._apply(fn, recurse=recurse)
        self.action_mean = mean.to(device=self.action_mean.device)
        self.action_scale = scale.to(device=self.action_scale.device)
        return self

    def set_condition(self, condition):
        if condition not in ('old', 'new'):
            raise ValueError('Condition must be explicitly old or new')
        self.condition = condition
        self.predictor._diagnostic_old_slots = self.old_slots
        self.predictor._diagnostic_condition = condition

    def train(self, mode=True):
        self.training = mode
        self.base.eval()  # 包括所有BN统计、dropout，且不受外层train()影响
        return self

    def encode(self, info):
        return self.base.encode(info)

    def predict(self, emb, act_emb):
        return self.base.predict(emb, act_emb)

    def rollout(self, info, actions, history_size=None):
        return self.base.rollout(info, actions, history_size=history_size)
