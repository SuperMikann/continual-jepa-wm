"""M3 增长/冻结机制冒烟测试（服务器上跑，~1 分钟）。

验证三条不变量（全部通过才允许启动 T1 重训）：
1. grow_adapter() 后前向输出与增长前完全一致（新槽 A=0 → 零扰动）
2. freeze_old_adapters(1) 后旧槽梯度恒为 0、新槽梯度非零
3. 切片兼容加载：K=1 检查点权重能正确装进 K=2 模型的旧槽

用法：python smoke_growth.py
"""

import torch

from stable_worldmodel.wm.lewm.module import AdapterPredictor

torch.manual_seed(0)

KW = dict(num_frames=3, depth=2, heads=2, mlp_dim=64,
          input_dim=32, hidden_dim=32, dim_head=16,
          num_adapters=1, rank=4, gate_hidden=8, freeze_gate=True)

x = torch.randn(2, 3, 32)
c = torch.randn(2, 3, 32)

# --- 检查 1：增长前后输出一致 ---
m1 = AdapterPredictor(**KW).eval()
with torch.no_grad():
    y_before = m1(x, c)
m1.grow_adapter()
assert m1.num_adapters == 2
assert m1.adapter_A.shape == (2, 32, 4)
assert m1.adapter_B.shape == (2, 4, 32)
assert m1.gate[-1].out_features == 2
with torch.no_grad():
    y_after = m1(x, c)
assert torch.equal(y_before, y_after), '增长扰动了前向输出！'
print('[1/3] grow_adapter 零扰动 ✓（增长前后输出逐位一致）')

# --- 检查 2：旧槽冻结（有 GPU 就在 GPU 上跑——hook 若在 CPU 建掩码、
# 模型后被搬上 GPU，backward 会设备不匹配；CPU-only 冒烟拦不住这类 bug） ---
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m1 = m1.to(dev)
xg, cg = x.to(dev), c.to(dev)
m1.train()
m1.freeze_old_adapters(1)
loss = m1(xg, cg).sum()
loss.backward()
gA, gB = m1.adapter_A.grad, m1.adapter_B.grad
assert gA[0].abs().sum() == 0 and gB[0].abs().sum() == 0, '旧槽梯度未置零！'
# 新槽可训的判据看 A：∂(A·B·h)/∂A = B·h ≠ 0；
# B 的梯度 ∂/∂B ∝ A，而 A 零初始化 → 开局 grad_B 恒为 0 是**设计行为**
# （与 M1 同款零初始化哲学：A 先动起来，B 随后才有梯度）
assert gA[1].abs().sum() > 0, '新槽 A 梯度为零，不可训！'
print(f'[2/3] freeze_old_adapters ✓（{dev}；旧槽梯度=0；新槽 A 可训，B 开局无梯度属设计行为）')

# --- 检查 3：切片兼容加载（模拟 train_lewm.py 的 warm_start 逻辑） ---
src = AdapterPredictor(**KW)  # K=1 检查点
ref_A = src.adapter_A.detach().clone()
sd = dict(src.state_dict())
m2 = AdapterPredictor(**{**KW, 'num_adapters': 2})
for key in ('adapter_A', 'adapter_B'):
    cur = getattr(m2, key)
    if sd[key].shape != cur.shape:
        k_old = sd[key].shape[0]
        grown = cur.detach().clone()
        grown[:k_old] = sd[key]
        sd[key] = grown
for key in ('gate.2.weight', 'gate.2.bias'):  # 形状不符直接丢弃
    if sd[key].shape != dict(m2.named_parameters())[key].shape:
        sd.pop(key)
m2.load_state_dict(sd, strict=False)
assert torch.equal(m2.adapter_A[0], ref_A[0]), '旧槽数值未保留！'
assert m2.adapter_A[1].abs().sum() == 0, '新槽应为零初始化！'
print('[3/3] 切片兼容加载 ✓（K=1 → K=2，旧槽保留、新槽为零）')

print('SMOKE_OK：可以启动 T1 重训')
