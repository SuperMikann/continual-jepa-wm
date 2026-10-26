# Continual JEPA World Models

面向顺序条件变化的 JEPA 世界模型研究代码。基于
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) 的
LeWM、原生 GoalMSE/CEM 规划与环境包装，研究差分预测、适配器与条件门控、
旧功能保护和漂移检测。方法的实现状态与实验有效性分别评价。

本目录于 2026-10-26 整理。**当前代码从 `implementation/` 开始阅读**；
`archive/` 保存历史实现。目录整理没有修改模型参数计算、损失或训练预算。

## 代码入口

| 路径 | 职责 |
|---|---|
| `implementation/module.py` | 普通/差分预测器、适配器、因果门控、增长与冻结、完整旧通路诊断包装 |
| `implementation/train_lewm.py` | 模型配置、训练损失、热启动、冻结回调和检查点保存 |
| `implementation/planning_history.py` | 已执行动作与历史帧的对齐、严格未来动作 rollout；包含显式安装所用的源码转换 |
| `implementation/install.py` | 检查安装目标、备份原文件、显式安装上述代码 |
| `cl_protocol/tasks.yaml` | 历史条件序列；阻尼条件另在 Reacher 实验模块中明确列出 |
| `cl_protocol/eval_task.py` | 原生规划评估、固定动作统计、指定场景与本次新增结果解析 |
| `cl_protocol/pred_error.py` | 轨迹窗口预测误差测量 |
| `cl_protocol/drift_trigger.py` | 固定动作坐标下的离线漂移评分、校准与验证 |
| `cl_protocol/test/run_pusht_experiments.py` | PushT 命令入口；实现分为 `pusht/training.py`、`diagnostics.py`、`adapters.py` |
| `cl_protocol/test/run_reacher_experiments.py` | Reacher 命令入口；实现分为 `reacher/common.py`、`evaluation.py`、`training.py`、`pipeline.py` |
| `cl_protocol/test/growth_check.py` | 扩容、AdamW 冻结、门控、版本兼容及旧通路保持的 CPU 检查 |
| `cl_protocol/test/test_*.py` | 漂移、安装、历史动作时序和多进程回放的回归测试 |
| `archive/` | 早期里程碑源码，供追溯和代码讨论 |

## 原 idea 对应关系

| 组件 | 实现位置与范围 |
|---|---|
| 差分动力学与更新掩码 | `module.py` 的 `DiffPredictor` / `DiffPredictorV2`；损失在 `lejepa_forward` |
| 低秩适配器库与上下文门控 | `AdapterPredictor` / `CausalAdapterPredictor`；固定均匀门控和可学习门控是不同对照 |
| 子空间高斯正则 | 历史实验已开展，但当前留存源码未定位到该项完整实现；不能把普通 SIGReg 当作此组件 |
| latent 锚点 | 训练入口保留历史可选加载/损失；生成工具在 `archive/方法阶段_里程碑2/` |
| 漂移触发与增长 | 检测在 `drift_trigger.py`，增长/冻结在预测器和训练入口；当前检测不会自动启动增长训练 |

保留的 legacy/V2 类用于区分实验协议和加载旧检查点。`mechanism_version=2`
包含图像投影归一化等结构变化，**不是通用的“自动使用最佳版本”开关**。
最新固定容量适配器实验使用独立的因果门控版本，不能与 V2 结构变化混同。
`FrozenOldPathWorldModel` 使用已知条件选择分支，属于机制诊断。

## 依赖与部署

这是现有 SWM 工程的覆盖实现，需要已有的 `stable_worldmodel/`、
`scripts/train/config/`、`scripts/plan/` 及该工程使用的训练/环境依赖。
安装工具不会升级依赖或下载数据。历史 AutoDL 环境的完整依赖锁尚未取得；
不能把本机另一份较新 SWM checkout 的版本当作历史服务器版本。

将本项目的 `implementation/` 与完整 `cl_protocol/` 放在目标 SWM 工程中，
在没有训练任务使用该工程时执行：

```bash
cd /root/autodl-tmp/stable-worldmodel
python implementation/install.py --root /root/autodl-tmp/stable-worldmodel
# 上一条只检查。核对输出后，以下命令备份并安装：
python implementation/install.py --root /root/autodl-tmp/stable-worldmodel --apply
```

安装映射为 `module.py` → `stable_worldmodel/wm/lewm/module.py`、
`train_lewm.py` → `scripts/train/lewm.py`；规划时序修复只接受已核对的旧源码布局
或相同修复版本。其他布局会停止，避免覆盖不同版本的平台实现。
原文件和实际包版本保存在目标工程 `.code_backups/<时间>/`。
实验程序只核对已安装版本，不再改写平台源码。

## 检查与运行

CPU 回归检查（第二条需在具备平台源码的 SWM 工程内运行）：

```bash
python -m unittest discover -s cl_protocol/test -p 'test_*.py' -v
python cl_protocol/test/growth_check.py \
  --module-file implementation/module.py \
  --train-file implementation/train_lewm.py \
  --world-model-file stable_worldmodel/wm/lewm/lewm.py \
  --eval-file cl_protocol/eval_task.py --out results/growth_check.json
```

查看实验选项不会训练：

```bash
python cl_protocol/test/run_pusht_experiments.py --help
python cl_protocol/test/run_reacher_experiments.py --help
```

训练继续复用 `scripts/train/lewm.py` 的 Hydra 配置；评估使用
`cl_protocol/eval_task.py`。组件、数据、统计和预算由实验配置明确指定。
协议示例见 [cl_protocol/README.md](cl_protocol/README.md)。长任务使用 `screen`，
日志用 `tail -F` 查看。

旧的 `run_pusht_v2_smoke.py`、`reacher_dynamics_screen.py` 保留为简短兼容入口，
参数和模式保持。以后修改对应实现模块；更新服务器时同步入口及其模块目录，
不再只上传原来的单个大文件。

## 分享与复现材料

教师代码审阅副本只包含本说明、实现、协议、测试和历史源码。
数据、权重、实验结果/日志、私人实验记录和老师批注不包含在代码副本中。
具体实验结果另行提供，并与其实际运行的代码版本、配置和预处理记录对应。
上游代码归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
