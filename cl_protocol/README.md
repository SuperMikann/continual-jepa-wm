# 持续学习实验协议 · 三件套使用说明

按定稿任务序列（2026-08-04）实现。**所有接口均对照 stable_worldmodel 源码核实**（pip 0.1.1 wheel + 仓库 main 快照），不是猜的。

## 文件清单

| 文件 | 作用 |
|---|---|
| `tasks.yaml` | 任务定义唯一事实来源（序列 A=PushT 目标/视觉漂移，序列 B=Reacher 动力学漂移） |
| `drift_world.py` | 漂移注入核心：DriftWorld（每次 reset 自动带 variation_values）+ 暗光变换 |
| `collect_task.py` | 按任务采集漂移数据（Reacher=RandomPolicy，PushT=WeakPolicy） |
| `eval_task.py` | 把漂移注入官方 eval_wm.py（不改它一行代码），单格/矩阵两种模式 |
| `pred_error.py` | 留出轨迹潜空间预测误差（成功率之外的第二个遗忘度量） |

把这 6 个文件一起上传到服务器仓库根目录旁的同一文件夹（如 `~/cl_protocol/`）。

## 前置条件

```bash
# DMC（Reacher）必备环境变量，采集/评估通用
export STABLEWM_HOME=/root/autodl-tmp/.stable_worldmodel
export MUJOCO_GL=osmesa          # collect_task.py 已内置此行，这里再设一层保险
export PYOPENGL_PLATFORM=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

cd /root/autodl-tmp/stable-worldmodel   # eval_task.py 默认在此找 scripts/plan/eval_wm.py
```

## 第 0 步：先验证注入机制无副作用（重要）

```bash
python ~/cl_protocol/eval_task.py --sequence B_reacher --task T1 \
    --policy lewm_reacher/weights_epoch_30.pt --num-eval 50
```

期望复现 **~84%**（与已有 `dmc_results.txt` 一致）。一致 → DriftWorld 对默认任务是透明的，后续所有漂移评估可信。不一致 → 停下来排查，先别往下跑。

## 第 1 步：采集漂移数据（T2/T3）

```bash
# 先探针：只保存 8 帧画面，人眼确认漂移生效（目标位置/暗光/方块颜色/密度）
python ~/cl_protocol/collect_task.py --sequence B_reacher --task T2 --check
python ~/cl_protocol/collect_task.py --sequence A_pusht --task T2 --check
python ~/cl_protocol/collect_task.py --sequence A_pusht --task T3 --check

# 确认后正式采集（铁律已内置：输出路径存在即拒绝运行）
python ~/cl_protocol/collect_task.py --sequence B_reacher --task T2
python ~/cl_protocol/collect_task.py --sequence B_reacher --task T3
python ~/cl_protocol/collect_task.py --sequence A_pusht --task T2
python ~/cl_protocol/collect_task.py --sequence A_pusht --task T3
```

每个任务 500 条（`tasks.yaml` 里 `collect_episodes` 可调）。采集时漂移键的实际值会以 `variation.<key>` 列写进数据集，随时可审计。

## 第 2 步：zero-shot 标定（免费基线，先跑这个）

T1 种子直接在各漂移任务上评估，不训练：

```bash
python ~/cl_protocol/eval_task.py --sequence B_reacher --matrix \
    --policies lewm_reacher/weights_epoch_30.pt
# PushT 同理（T1 种子是 pusht_seed_epoch_30.pt）
```

这给出遗忘矩阵的第一行，也是"漂移强度是否合适"的判据：**T2/T3 成功率应掉到 ≤70% 左右但不归零**。全绿说明漂移太弱（改 tasks.yaml 加大强度），全零说明太强。

## 第 3 步：顺序微调 + 遗忘矩阵

每个任务用已有命令模板训练（注意 `output_model_name` 区分、`data.dataset.name` 带 `.lance` 后缀）：

```bash
python scripts/train/lewm.py data=dmc \
    data.dataset.name=dmc/reacher_t2.lance \
    output_model_name=lewm_reacher_t2 trainer.max_epochs=10
```

每训完一个任务，把它加进矩阵：

```bash
python ~/cl_protocol/eval_task.py --sequence B_reacher --matrix \
    --policies lewm_reacher/weights_epoch_30.pt lewm_reacher_t2/weights_epoch_10.pt
```

产物：`forgetting_matrix_<sequence>.json` + 终端表格。每格同时写进各检查点目录的 `dmc_results.txt` / `pusht_results.txt`（官方格式）。

## 第 4 步：预测误差遗忘度量

```bash
python ~/cl_protocol/pred_error.py --sequence B_reacher \
    --policy lewm_reacher/weights_epoch_30.pt
python ~/cl_protocol/pred_error.py --sequence B_reacher \
    --policy lewm_reacher_t2/weights_epoch_10.pt
```

产物：`pred_error_<sequence>_<policy>.json`。与成功率矩阵互补：成功率看"还能不能完成任务"，预测误差看"世界模型本身忘了多少"。

## 注意事项（都是定稿时确认过的决策）

1. **T3 手臂密度 500 恰在 variation 空间下限**。平台用 `contains()` 做边界检查（含端点），理论上可行；万一报错就把 `tasks.yaml` 里改成 `[505.0]`。
2. **PushT T2/T3 用 WeakPolicy 采集，T1 是官方专家数据**——策略分布不同是个已知混淆项。缓解：评估协议对全序列统一（数据集定义的起点/目标 + CEM 规划），策略只影响数据覆盖度；论文里在实验设置一节如实写明即可。
3. **T1 的留出 200 条参与过训练**，预测误差绝对值偏乐观；跨任务/跨检查点的差值仍然有效。正式跑建议每任务多采 200 条专用测试集。
4. **暗光漂移的实现**：PushT 没有光照 variation 因子，用 `World(image_transform=...)` 在渲染像素上缩放亮度实现（对齐 AdaJEPA 的 dark lighting）。采集时变换生效 → 数据集里就是暗图 → 评估时目标图与环境图同为暗图，分布一致。
5. **评估为何必须走 DriftWorld**：平台的 variation_space 每次 reset 都先恢复默认值，官方 eval_wm.py 的 reset 不传 options——不打这个补丁，所谓"T2 评估"实际上是在默认任务上评的。这是整套协议里最关键的一个坑。
6. 评估结果文件是 append 模式，eval_task.py 总是读**最后一次**运行的结果，重复跑不会串。
