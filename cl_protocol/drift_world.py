"""漂移注入工具：DriftWorld + make_darken。

为什么需要这个模块（平台机制，已对照源码核实）：
  环境的 variation_space 在每次 reset 时都会先恢复默认值
  （spaces.reset_variation_space → variation_space.reset()），
  只有 reset 的 options 里带了 variation_values 才会重新设置漂移。
  官方评估脚本 scripts/plan/eval_wm.py 内部 reset 不传 options，
  直接在它的环境上评估 = 在默认任务上评估，漂移根本没生效。

  DriftWorld 在 World 层劫持 reset，把任务的 variation_values
  注入每一次 reset（包括数据集驱动评估内部的那次），
  保证环境物理/渲染在整个评估期间都处于漂移状态。

  暗光漂移不走 variation（PushT 没有光照因子），
  用 World 自带的 image_transform 钩子实现（MegaWrapper 内部的
  AddPixelsWrapper 会对每帧渲染像素应用该变换）。
"""

import numpy as np

import stable_worldmodel as swm


def make_darken(factor):
    """生成亮度缩放变换。factor=0.5 即亮度减半（暗光）。

    返回 None 表示不做变换（factor >= 1.0）。
    输入输出均为 uint8 HWC 图像（在 resize 之前应用）。
    """
    if factor is None or float(factor) >= 1.0:
        return None

    factor = float(factor)

    def _darken(img):
        arr = np.asarray(img, dtype=np.float32) * factor
        return np.clip(arr, 0, 255).astype(np.uint8)

    return _darken




def to_space_value(key, v):
    """把配置里的 variation 值转成空间期望的 ndarray dtype。

    平台 set_value 只用 contains() 校验、不做转换存储，而 env 侧按
    空间 dtype 的 ndarray 使用（如 .tolist()）；且 gym 的 contains 对
    已是 ndarray 的输入做安全转换检查（int64->float32 会被拒）。
    规则：整数的 *color 键按 RGB uint8，其余数值统一 float32。
    """
    arr = np.asarray(v)
    if arr.dtype.kind == 'f':
        return arr.astype(np.float32)
    if arr.dtype.kind == 'i':
        if key.endswith('color'):
            return arr.astype(np.uint8)
        return arr.astype(np.float32)
    return arr


class DriftWorld(swm.World):
    """每次 reset 自动注入 variation_values 的 World 子类。

    用法（eval_task.py 通过 monkey-patch 替换官方脚本里的 swm.World）：

        swm.World = functools.partial(
            DriftWorld,
            variation_values={'agent.arm_density': [700.0]},
            image_transform=make_darken(0.5),
        )
        # 之后官方 eval_wm.py 里的 swm.World(**cfg.world, ...) 全部带漂移
    """

    def __init__(self, *args, variation_values=None, **kwargs):
        # env 侧按 ndarray 使用 variation 值（如 pusht env.py 的 .value.tolist()），
        # YAML/配置里读出的 list 必须统一转数组
        self._drift_vv = {
            k: to_space_value(k, v)
            for k, v in dict(variation_values or {}).items()
        }
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        if self._drift_vv:
            options = dict(options or {})
            # 显式传入的 options 优先（例如调用方自己指定了 variation_values）
            merged = {**self._drift_vv, **options.get('variation_values', {})}
            options['variation_values'] = merged
        return super().reset(seed=seed, options=options)
