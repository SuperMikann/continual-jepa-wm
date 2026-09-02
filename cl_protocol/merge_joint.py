"""联合训练基线：合并 T1+T2+T3 为等曝光混合数据集 pusht_joint.lance。

用法：
    export STABLEWM_HOME=/root/autodl-tmp/.stable_worldmodel
    python cl_protocol/merge_joint.py --sequence A_pusht
    python cl_protocol/merge_joint.py --sequence B_reacher

输出：<STABLEWM_HOME>/datasets/{pusht_joint|reacher_joint}.lance

设计：
  - T1 官方专家数据取前 ~10 万帧（= T2/T3 的 500 eps × 200 步），
    保证三个任务在联合训练中的梯度曝光量相等（按整 episode 切，不截断）
  - 列取三数据集 schema 交集（variation_* 只存在于 T2/T3，自动剔除）
  - episode_idx 重编号，保持 episode-contiguous（LanceDataset 硬性要求）
  - 像素列是 JPEG blob，全程 pyarrow 流式复制，无解码/重编码损失
"""
import argparse
import sys

import lancedb
import numpy as np
import pyarrow as pa
import stable_worldmodel as swm

SEQUENCES = {
    'A_pusht': [
        ('pusht_expert_train.lance', 100_000),  # (名字, 帧数上限) None = 全取
        ('pusht_t2.lance', None),
        ('pusht_t3.lance', None),
    ],
    'B_reacher': [
        ('dmc/reacher_random.lance', 100_000),
        ('dmc/reacher_t2.lance', None),
        ('dmc/reacher_t3.lance', None),
    ],
}
IDX = ('episode_idx', 'step_idx')


def open_table(datasets_dir, name):
    p = datasets_dir / name
    db = lancedb.connect(str(p.parent))
    return db.open_table(p.stem)


def episode_cut(table, frame_cap):
    """返回 (要保留的最大 episode_idx, 保留帧数, 总 episodes, 总帧数)。"""
    col = table.to_lance().to_table(columns=['episode_idx']).column('episode_idx').to_numpy()
    eps, counts = np.unique(col, return_counts=True)
    total_eps, total_frames = len(eps), int(counts.sum())
    if frame_cap is None:
        return int(eps[-1]), total_frames, total_eps, total_frames
    cum = np.cumsum(counts)
    n_take = int(np.searchsorted(cum, frame_cap, side='left')) + 1
    n_take = min(n_take, total_eps)
    return int(eps[n_take - 1]), int(cum[n_take - 1]), total_eps, total_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequence', required=True, choices=list(SEQUENCES))
    args = ap.parse_args()
    sources = SEQUENCES[args.sequence]
    out_name = 'pusht_joint' if args.sequence == 'A_pusht' else 'dmc/reacher_joint'

    datasets_dir = swm.data.utils.get_cache_dir(sub_folder='datasets')
    out_dir = datasets_dir / f'{out_name}.lance'
    if out_dir.exists():
        sys.exit(f'[拒绝] 输出路径已存在: {out_dir}\n铁律：不追加不覆盖，先确认再手动删除。')

    tables = [(n, cap, open_table(datasets_dir, n)) for n, cap in sources]

    # --- schema 交集（顺序以 T1 为准） ---
    schemas = [t.schema for _, _, t in tables]
    common = [f.name for f in schemas[0]
              if f.name in schemas[1].names and f.name in schemas[2].names]
    for col in (*IDX, 'pixels', 'action'):
        assert col in common, f'关键列 {col} 不在三数据集交集中: {common}'
    dropped = sorted({n for s in schemas for n in s.names if n not in common})
    print(f'[合并] 保留列: {common}')
    print(f'[合并] 剔除列: {dropped}')
    out_schema = pa.schema([schemas[0].field(c) for c in common])

    # --- 逐源统计与截断 ---
    plans = []
    for name, cap, t in tables:
        max_ep, take_frames, total_eps, total_frames = episode_cut(t, cap)
        plans.append((name, t, max_ep, take_frames))
        print(f'[合并] {name}: 取 episode 0..{max_ep}（{take_frames} 帧）'
              f' / 共 {total_eps} eps {total_frames} 帧')

    ep_i = common.index('episode_idx')

    def batch_iter():
        offset = 0
        for name, t, max_ep, take_frames in plans:
            scanner = t.to_lance().scanner(
                columns=common, filter=f'episode_idx <= {max_ep}')
            n_eps = 0
            for batch in scanner.to_batches():
                arrays = list(batch.columns)
                ep = arrays[ep_i].to_numpy(zero_copy_only=False)
                n_eps = max(n_eps, int(ep.max()) + 1)
                arrays[ep_i] = pa.array(ep + offset, type=pa.int32())
                yield pa.record_batch(arrays, schema=out_schema)
            print(f'[合并] {name}: {n_eps} episodes 写入（offset={offset}）')
            offset += n_eps

    # LanceDB 表名不允许含 '/'：斜杠是目录层级，库连到父目录、表名只取文件名
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(out_dir.parent))
    reader = pa.RecordBatchReader.from_batches(out_schema, batch_iter())
    db.create_table(out_dir.stem, data=reader, schema=out_schema)

    # --- 验收 ---
    out = open_table(datasets_dir, f'{out_name}.lance')
    col = out.to_lance().to_table(columns=['episode_idx']).column('episode_idx').to_numpy()
    n_eps, n_rows = len(np.unique(col)), len(col)
    print(f'[完成] {out_dir}: {n_eps} episodes, {n_rows} 帧')
    print(f'[预估] 训练步数/epoch ≈ {n_rows // 156}（参照 T2：10 万帧 ≈ 639 步）')


if __name__ == '__main__':
    main()
