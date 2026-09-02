# Continual JEPA World Model（持续学习 JEPA 世界模型）

> Continual learning on JEPA-style latent world models (LeWorldModel, built on the
> [stable-worldmodel](https://github.com/StableWorldmodel/stable-worldmodel) platform):
> one model learns a sequence of drifting tasks, and we measure / mitigate catastrophic
> forgetting of planning ability on earlier tasks.

项目代码仓库。在 JEPA 类潜在世界模型（LeWorldModel）上做**持续学习**：一个模型依次学习多个发生受控漂移的任务，要求新任务学得动（适应性）、旧任务规划能力不退（保持性）。

## 任务序列

两条受控漂移序列（唯一定义见 `cl_protocol/tasks.yaml`）：

| | T1 | T2 | T3 |
|---|---|---|---|
| A：PushT（视觉/目标漂移） | 默认（官方专家数据） | 目标点→[100,100] + 暗光×0.5 | 目标点→[100,400] + 方块 scale 55 + 变红 |
| B：Reacher（动力学漂移） | 密度 1000（随机数据 2000 条） | 手臂密度 700 | 手臂密度 500 + 手指密度 700 |

双指标评估：规划成功率矩阵（CEM，n=100/格）+ 留出轨迹潜空间预测误差（pred_error）。

## 仓库结构

```
cl_protocol/            持续学习协议代码（数据采集 / 漂移注入评估 / 预测误差探针）
milestones/             方法阶段各里程碑的模型与训练脚本（逐里程碑演进）
  m0_diff_predictor/      差分动力学头 DiffPredictor（残差参数化 ẑ = z + m ⊙ Δẑ + 稀疏约束）
  m1_adapter_predictor/   低秩适配器库 + 门控 AdapterPredictor（K=4，rank=16）
  m2_anchor_and_freeze/   潜在锚点记忆、数据回放合并、编码器/主干冻结、BN 重校准
tools/                  AutoDL 环境部署脚本、训练曲线绘图、PushT 渲染工具
results/baseline/       LeWM 基线复现的训练曲线与成功率对比图
```

## 依赖与部署

- 平台：[stable-worldmodel](https://github.com/StableWorldmodel/stable-worldmodel)（pip 0.1.1 + 仓库 main 快照），所有实验在其 `scripts/train/lewm.py` / `scripts/plan/eval_wm.py` 入口上打补丁实现
- 环境：`tools/autodl_setup.sh`（AutoDL 单卡实例，含 MuJoCo 无头渲染的 OSMesa 配置）
- `milestones/` 下的 `module.py` / `lewm_model.py` 用于覆盖平台安装目录中对应的 `stable_worldmodel/wm/lewm/` 文件；`train_lewm*.py` 覆盖 `scripts/train/lewm.py`。各里程碑文件为递进关系，最新版见 `m2_anchor_and_freeze/`

## 各里程碑要点

- **M0 差分动力学头**：残差参数化 + 逐维掩码 + 稀疏约束，与原版 Predictor 接口完全一致，CLI 切换（`model.predictor._target_=...DiffPredictor`）。剂量扫描定案 λ_Δ=0.01、λ_s=0.001（行为与表征双无损）
- **M1 适配器库 + 门控**：Δ 路径上叠加 K=4 低秩适配器（零初始化，开局严格退化为 DiffPredictor）；门控默认冻结（`+model.predictor.freeze_gate=true`），留待漂移触发机制激活
- **M2 锚点 / 冻结 / 校准工具链**：`dump_anchors.py`（导出潜在锚点库）、`merge_replay.py`（多任务数据等比合并回放）、`bn_recal.py`（训后 BN 统计量重校准，消除投影层统计漂移对旧任务评估的污染）；训练脚本最终版 `train_lewm.py` 集成 `+warm_start` / `+anchor_path` / `+freeze_encoder` / `+freeze_dynamics` 全部开关

## 协议代码用法

见 `cl_protocol/README.md`（任务定义、采集、漂移评估、预测误差三件套的使用说明与平台注意事项）。

## 许可

代码以 MIT 协议发布（见 `LICENSE`）。上游平台 stable-worldmodel 的版权归其原作者所有。
