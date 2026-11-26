"""旋转核：只依赖逻辑坐标与模态频率参数，与物理下标无关。

相同 (逻辑坐标, 模态参数, 布局) 必然得到相同相位，因此预填充、增量解码、
分块重算三条路径以及单独请求与拼接请求的结果逐位一致。
"""

from __future__ import annotations

import math
from functools import lru_cache

from .config import ModalitySpec, RopeLayout
from .positions import Coord


@lru_cache(maxsize=None)
def _inv_freq(theta: float, pairs: int) -> tuple[float, ...]:
    dim = 2 * pairs
    return tuple(theta ** (-2.0 * j / dim) for j in range(pairs))


def rotate_vector(vector, coord: Coord, spec: ModalitySpec, layout: RopeLayout) -> list[float]:
    """对单个头维向量施加旋转。

    旋转角 = 逻辑坐标 * freq_scale * theta**(-2j/d)。向量按 layout.sections
    切成 (t, h, w) 三段，每段内部按前后半配对旋转。
    """
    if len(vector) != layout.head_dim:
        raise ValueError(f"向量维度 {len(vector)} 与 head_dim {layout.head_dim} 不一致")
    out = list(vector)
    axes = (coord.t, coord.h, coord.w)
    base = 0
    for axis, pairs in enumerate(layout.sections):
        if pairs == 0:
            continue
        inv_freq = _inv_freq(spec.theta, pairs)
        scale = axes[axis] * spec.freq_scale
        for j in range(pairs):
            angle = scale * inv_freq[j]
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            i1 = base + j
            i2 = i1 + pairs
            x1 = vector[i1]
            x2 = vector[i2]
            out[i1] = x1 * cos_a - x2 * sin_a
            out[i2] = x2 * cos_a + x1 * sin_a
        base += 2 * pairs
    return out


def rotate_block(vectors, coords, spec: ModalitySpec, layout: RopeLayout) -> list[list[float]]:
    """对一段同模态的向量块施加旋转。"""
    vectors = list(vectors)
    coords = list(coords)
    if len(vectors) != len(coords):
        raise ValueError(f"向量数 {len(vectors)} 与坐标数 {len(coords)} 不一致")
    return [rotate_vector(vec, coord, spec, layout) for vec, coord in zip(vectors, coords)]
