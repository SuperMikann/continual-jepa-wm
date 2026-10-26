# 持续学习实验协议

模型与安装说明见 [项目 README](../README.md)。当前模型源码统一放在
`implementation/`；这里负责条件定义、采集、测量与实验编排。
整理前的完整操作说明已内部保留在 `archive/protocol_notes_before_cleanup.md`。

## 核心协议

| 文件 | 作用 |
|---|---|
| `tasks.yaml` | PushT、Reacher 密度及其他历史序列的任务定义 |
| `drift_world.py` | reset 时注入条件变化、图像暗化 |
| `collect_task.py` | 按任务采集，保存实际变化参数 |
| `eval_task.py` | 复用原生 eval_wm；单格、矩阵、固定统计与指定场景 |
| `pred_error.py` | 轨迹窗口潜空间预测误差 |
| `drift_trigger.py` | 离线误差比、校准和验证，不自动训练 |

## 运行环境

```bash
cd /root/autodl-tmp/stable-worldmodel
export STABLEWM_HOME=/root/autodl-tmp/.stable_worldmodel
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

检查点、数据和统计文件必须实际存在。动作统计 JSON 包含对应检查点名
`policy`、来源 `source`、逐维 `mean` 和正的标准化分母 `scale`，来自模型实际预处理。

## PushT

入口为 `test/run_pusht_experiments.py`：

- `test/pusht/training.py`：训练配置、普通/差分、调度与结构对照。
- `test/pusht/diagnostics.py`：一步/多步、候选排序和真实历史的评估检查。
- `test/pusht/adapters.py`：同底座的普通微调、均匀适配器、修正门控三组对照。

```bash
python cl_protocol/test/run_pusht_experiments.py --help
# 以下为长训练示例；需已安装代码、实际T1检查点及两套数据：
python -u cl_protocol/test/run_pusht_experiments.py \
  --mode adapter-t2 --base-policy m2_t1/weights_epoch_30.pt \
  --epochs 30 --num-eval 100 --tag NEW_RUN_NAME
```

三组对照固定原 T1 动作统计，门控单独启用因果版本及正熵损失；不默认改变图像
投影或开启差分/锚点附加损失。该项测固定容量适配器和门控，初始化/两更新短检
通过后才正式训练。完整旧通路冻结和自动条件识别分别验证。
旧入口 `test/run_pusht_v2_smoke.py` 的模式和参数继续有效。

## Reacher 阻尼与其他动力学条件

入口为 `test/run_reacher_experiments.py`：

- `test/reacher/common.py`：固定条件参数、配置、结果校验及配对比较。
- `test/reacher/evaluation.py`：物理参数注入、原生闭环与预测评估。
- `test/reacher/training.py`：采集、回放数据类和训练接入。
- `test/reacher/pipeline.py`：固定预算筛查、T2适应/对照和T3队列。

```bash
python cl_protocol/test/run_reacher_experiments.py --help
```

模式为 `screen/adapt/compare/stage3/collect/train/prediction`，预算和阶段停止条件
延续原配置。回放数据类位于可导入模块，支持多进程数据加载。
worker 对原生入口的内存包装仍用于注入条件及记录预算，退出时恢复，不改写平台
源码。旧入口 `test/reacher_dynamics_screen.py` 继续有效。

## 评估与检测

```bash
python cl_protocol/eval_task.py --help
python cl_protocol/drift_trigger.py --help
```

评估比较同一批起点、物理目标、规划预算和动作统计。
`--eval-dataset` 会改变起点及目标来源，不能单独解释为只换目标画面。
`pred_error.py` 默认保留历史逐任务标准化；跨条件检测使用
`drift_trigger.py --action-stats ACTUAL_STATS.json` 固定动作坐标。
`threshold_calibrated` 表示执行了校准，不保证验证误报率达标。

`scheme_eval.py` 与 `walker_stand_probe.py` 是保留的早期诊断入口。
物理候选时序对齐使用 `test/aligned_scheme_eval.py`；Walker/Cheetah 原生回报
使用 `test/eval_locomotion.py`。这些测量与闭环持续学习方法对照分别解释。

## 检查和进度

```bash
python -m unittest discover -s cl_protocol/test -p 'test_*.py' -v
tail -n 30 -F logs/ACTUAL_RUN_NAME.log
```

输出目录已存在时使用新的实验名称，保留历史回执。
`test/growth_check.py` 检查模型机制和兼容性；数据采集工具和早期规划保持诊断
保留原入口，未纳入自动执行队列。完整旧通路诊断的 old/new 条件选择须显式指定。
