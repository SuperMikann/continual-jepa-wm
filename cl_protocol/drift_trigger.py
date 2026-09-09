"""漂移触发判定（idea 式 14 的任务边界版，M3 核心组件）。

在任务边界用当前模型对「上一任务」和「新任务」的留出集各算一次潜空间
预测误差（完全复用 pred_error.py 的窗口机制，同分布同尺度），
漂移分数 = 新任务误差 / 参考任务误差，超过阈值 τ 判定触发：

    s = err(probe_task) / err(ref_task) ，s > τ → TRIGGER（增长适配器）

与 idea 式 14 的偏离（论文中如实说明）：
  - 门控冻结时 H(π) = ln K 为常数，熵项不起作用 → β = 0；
  - 触发时机为任务边界离线判定（批量任务序列协议），
    训练中在线窗口触发留作后续扩展。

τ 预注册 = 1.15。标定依据（Reacher，m3_reacher_t1 zero-shot）：
T2/T1 = 0.00265/0.00216 ≈ 1.23（触发），T3/T1 ≈ 1.69（触发）；
任务内采样噪声远小于 15%。PushT 漂移为 45× 量级，用于对侧 sanity 检查。

用法：
    python drift_trigger.py --sequence B_reacher \
        --policy m3g_reacher_t1/weights_epoch_30.pt \
        --ref-task T1 --probe-task T2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import stable_worldmodel as swm

from pred_error import task_pred_error, episode_col


def task_error(model, task_cfg, args, device):
    """加载任务数据集并在留出集上算 pred_error，返回 (mean, std, n)。"""
    dataset = swm.data.load_dataset(
        task_cfg['dataset'], keys_to_cache=['pixels', 'action']
    )
    col = episode_col(dataset)
    eps = np.unique(dataset.get_col_data(col))
    held_eps = eps[-args.reserve_last:]
    step_idx = dataset.get_col_data('step_idx')
    ep_arr = dataset.get_col_data(col)
    ep_len = {e: int(step_idx[ep_arr == e].max()) + 1 for e in held_eps}
    return task_pred_error(model, dataset, held_eps, ep_len, args, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks-cfg', default=str(Path(__file__).parent / 'tasks.yaml'))
    ap.add_argument('--sequence', required=True, choices=['A_pusht', 'B_reacher'])
    ap.add_argument('--policy', required=True,
                    help='检查点（checkpoints/ 下的相对路径）')
    ap.add_argument('--ref-task', required=True, choices=['T1', 'T2', 'T3'],
                    help='参考任务（通常为刚训完的任务）')
    ap.add_argument('--probe-task', required=True, choices=['T1', 'T2', 'T3'],
                    help='探测任务（通常为下一个待学任务）')
    ap.add_argument('--tau', type=float, default=1.15, help='触发阈值（预注册 1.15）')
    ap.add_argument('--reserve-last', type=int, default=200)
    ap.add_argument('--windows', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(args.tasks_cfg, encoding='utf-8') as f:
        seq = yaml.safe_load(f)['sequences'][args.sequence]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    print(f'[模型] {args.policy} 已加载（{device}）')

    err_ref, std_ref, n_ref = task_error(
        model, seq['tasks'][args.ref_task], args, device
    )
    err_probe, std_probe, n_probe = task_error(
        model, seq['tasks'][args.probe_task], args, device
    )
    s = err_probe / err_ref
    triggered = s > args.tau

    print(f'[drift] {args.ref_task} 误差 = {err_ref:.4f} ± {std_ref:.4f} (n={n_ref})')
    print(f'[drift] {args.probe_task} 误差 = {err_probe:.4f} ± {std_probe:.4f} (n={n_probe})')
    print(f'[drift] 漂移分数 s = {s:.3f}（τ = {args.tau}）')
    print(f'DRIFT_RESULT {{"policy": "{args.policy}", "ref": "{args.ref_task}", '
          f'"probe": "{args.probe_task}", "s": {s:.4f}, "tau": {args.tau}, '
          f'"triggered": {str(triggered).lower()}}}')
    print(f'[判定] {"*** TRIGGER：增长适配器 ***" if triggered else "NO-TRIGGER：保持现状"}')

    safe = args.policy.replace('/', '_').replace('.pt', '')
    out = Path(f'drift_trigger_{args.sequence}_{safe}.json').resolve()
    record = {
        'policy': args.policy, 'sequence': args.sequence,
        'ref_task': args.ref_task, 'probe_task': args.probe_task,
        'err_ref': err_ref, 'err_probe': err_probe,
        's': s, 'tau': args.tau, 'triggered': triggered,
        'reserve_last': args.reserve_last, 'windows': args.windows,
    }
    # 追加模式：同一检查点的多次判定累积成列表
    history = []
    if out.exists():
        history = json.loads(out.read_text(encoding='utf-8'))
        if not isinstance(history, list):
            history = [history]
    history.append(record)
    out.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[完成] 已保存: {out}')


if __name__ == '__main__':
    main()
