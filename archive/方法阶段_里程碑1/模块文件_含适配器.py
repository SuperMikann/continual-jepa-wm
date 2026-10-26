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
    LeWM.predict / rollout / CEM 规划 / 评估管线全部零改动。

    差别：transformer 输出 h 经 delta_head 得到 latent 差分 Δ，
    经 mask_head 得到维度级更新掩码 m = sigmoid(·)，
    最终返回 x + m ⊙ Δ，而不是直接回归绝对下一 latent。

    初始化约定：
      - delta_head 零初始化 → 开局预测 = 恒等（z'≈z，持续性先验），
        与 AdaLN-zero 同款做法，保证残差参数化不伤害早期训练；
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
        delta = delta + torch.einsum('btkd,bk->btd', v, pi)
        m = torch.sigmoid(self.mask_head(h))
        self.last_delta = delta
        self.last_mask = m
        self.last_pi = pi
        return x + m * delta
