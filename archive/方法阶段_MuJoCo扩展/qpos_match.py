"""qpos_match 任务类（MuJoCo 扩展，批注 3）：给 Walker/Cheetah 补上
"到达目标关节位形即成功"的评估任务，模板 = custom_tasks/reacher.py 的
ReacherQPosMatchTask。

目标 qpos 由外部通过 env 包装器的 set_target_qpos() 设置；未设置前永不
提前终止。奖励语义与原任务无关（仅用于评估终止判定）。

服务器落位：stable_worldmodel/envs/dmcontrol/custom_tasks/qpos_match.py
"""

import numpy as np

from stable_worldmodel.envs.dmcontrol.custom_tasks.walker import (
    CustomPlanarWalker,
)
from stable_worldmodel.envs.dmcontrol.custom_tasks.cheetah import (
    CustomCheetah,
)

_DEFAULT_QPOS_THRESHOLD = 0.05


class _QPosMatchMixin:
    """共享逻辑：target_qpos 未设置则不终止；逐关节 |Δ| 全小于阈值即成功。"""

    def _init_qpos_match(self, qpos_threshold=_DEFAULT_QPOS_THRESHOLD):
        self.target_qpos = None
        self.qpos_threshold = qpos_threshold

    def get_termination(self, physics):
        if self.target_qpos is None:
            return None
        # 跳过全局平移维（rootx）：locomotion 任务下绝对站位不可精确复现，
        # 计入它会使「全维误差<阈值」构造性不可达（2026-09-23 walker 行为全 0 的根因）
        names = physics.named.data.qpos.axes.row.names
        idx = [i for i, n in enumerate(names) if n != 'rootx']
        diff = np.abs(physics.data.qpos[idx] - self.target_qpos[idx])
        if np.all(diff < self.qpos_threshold):
            return 0.0
        return None


class WalkerQPosMatchTask(_QPosMatchMixin, CustomPlanarWalker):
    """Walker 的 qpos_match 任务（评估用；reward 走 walk 原语义）。"""

    def __init__(self, random=None, qpos_threshold=_DEFAULT_QPOS_THRESHOLD):
        super().__init__(goal='walk', move_speed=1, random=random)
        self._init_qpos_match(qpos_threshold)


class CheetahQPosMatchTask(_QPosMatchMixin, CustomCheetah):
    """Cheetah 的 qpos_match 任务（评估用；reward 走 run 原语义）。"""

    def __init__(self, random=None, qpos_threshold=_DEFAULT_QPOS_THRESHOLD):
        super().__init__(goal='run', move_speed=10, random=random)
        self._init_qpos_match(qpos_threshold)
