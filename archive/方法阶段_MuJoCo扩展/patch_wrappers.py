"""MuJoCo 扩展：给 walker/cheetah 包装器打 qpos_match 补丁（自检+幂等）。

在服务器仓库根目录运行：
    python patch_wrappers.py
补丁对象：
    stable_worldmodel/envs/dmcontrol/walker.py
    stable_worldmodel/envs/dmcontrol/cheetah.py
补丁内容（照 reacher.py 的既有模式）：
    1. _TASKS 增加 'qpos_match'
    2. compile_model 增加 qpos_match 分支（用 QPosMatch 任务类）
    3. 新增 set_target_qpos / reset 透传 / _is_terminated
幂等：已打过的文件直接跳过。锚点缺失或不唯一会断言失败并打印文件名。
"""

import shutil
import sys
from pathlib import Path

ROOT = Path('stable_worldmodel/envs/dmcontrol')

METHODS_WALKER = '''
    def set_target_qpos(self, target_qpos):
        """Set the target qpos for the qpos_match task."""
        assert self._task == 'qpos_match', (
            "set_target_qpos() is only valid for the 'qpos_match' task."
        )
        self.env.task.target_qpos = np.asarray(target_qpos, dtype=np.float64)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._task == 'qpos_match' and options is not None:
            target_qpos = options.get('target_qpos')
            if target_qpos is not None:
                self.env.task.target_qpos = np.asarray(
                    target_qpos, dtype=np.float64
                )
        return obs, info

    def _is_terminated(self, step) -> bool:
        if self._task != 'qpos_match':
            return False
        return (
            step.last()
            and self.env.task.get_termination(self.env.physics) is not None
        )

'''

PATCHES = {
    'walker.py': [
        ("from stable_worldmodel.envs.dmcontrol.custom_tasks.walker import (\n    CustomPlanarWalker,\n)",
         "from stable_worldmodel.envs.dmcontrol.custom_tasks.walker import (\n    CustomPlanarWalker,\n)\nfrom stable_worldmodel.envs.dmcontrol.custom_tasks.qpos_match import (\n    WalkerQPosMatchTask,\n)"),
        ("    'flip',\n)",
         "    'flip',\n    'qpos_match',\n)"),
        ("    'flip': 0,\n}",
         "    'flip': 0,\n    'qpos_match': 0,\n}"),
        ("        task = CustomPlanarWalker(\n            goal=self._task, move_speed=self._move_speed, random=seed\n        )",
         "        if self._task == 'qpos_match':\n            task = WalkerQPosMatchTask(random=seed)\n        else:\n            task = CustomPlanarWalker(\n                goal=self._task, move_speed=self._move_speed, random=seed\n            )"),
        ("    def modify_mjcf_model(self, mjcf_model):",
         METHODS_WALKER + "    def modify_mjcf_model(self, mjcf_model):"),
    ],
    'cheetah.py': [
        ("from stable_worldmodel.envs.dmcontrol.custom_tasks.cheetah import (\n    CustomCheetah,\n    Physics,\n)",
         "from stable_worldmodel.envs.dmcontrol.custom_tasks.cheetah import (\n    CustomCheetah,\n    Physics,\n)\nfrom stable_worldmodel.envs.dmcontrol.custom_tasks.qpos_match import (\n    CheetahQPosMatchTask,\n)"),
        ("    'flip-backward',\n)",
         "    'flip-backward',\n    'qpos_match',\n)"),
        ("    'flip-backward': 0,\n}",
         "    'flip-backward': 0,\n    'qpos_match': 0,\n}"),
        ("        task = CustomCheetah(\n            goal=self._task, move_speed=self._move_speed, random=seed\n        )",
         "        if self._task == 'qpos_match':\n            task = CheetahQPosMatchTask(random=seed)\n        else:\n            task = CustomCheetah(\n                goal=self._task, move_speed=self._move_speed, random=seed\n            )"),
        ("    def modify_mjcf_model(self, mjcf_model):",
         METHODS_WALKER + "    def modify_mjcf_model(self, mjcf_model):"),
    ],
}


def main():
    for fname, patches in PATCHES.items():
        path = ROOT / fname
        if not path.exists():
            sys.exit(f'[失败] 文件不存在: {path}（请在仓库根目录运行）')
        s = path.read_text(encoding='utf-8')
        if 'qpos_match' in s:
            print(f'[跳过] {fname} 已打过补丁')
            continue
        shutil.copy(path, path.with_suffix('.py.bak_qpos'))
        for old, new in patches:
            if s.count(old) != 1:
                sys.exit(f'[失败] {fname} 锚点缺失或不唯一: {old[:60]!r} '
                         f'（出现 {s.count(old)} 次）——源文件可能已漂移，'
                         f'把该文件发回分析')
            s = s.replace(old, new)
        path.write_text(s, encoding='utf-8')
        print(f'[完成] {fname} 已打补丁（备份 {fname}.bak_qpos）')
    print('PATCH_OK')


if __name__ == '__main__':
    main()
