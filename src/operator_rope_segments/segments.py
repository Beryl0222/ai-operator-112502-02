"""段落描述与校验。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import MODE_GRID2D, FrequencyRules
from .errors import SegmentError


@dataclass(frozen=True)
class Segment:
    """一个逻辑段落。

    position_start 是该段的局部位置起点，与它在 batch 中的物理位置无关；
    截断、重排、恢复都不得改变既有 token 的逻辑坐标。
    """

    segment_id: str
    kind: str
    position_start: int
    length: int | None = None
    grid: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.grid is not None:
            object.__setattr__(self, "grid", tuple(self.grid))

    def token_count(self, spec) -> int:
        """段内 token 数：一维段取 length，二维网格段取 grid 面积。"""
        if spec.mode == MODE_GRID2D:
            if self.grid is None:
                raise SegmentError(f"段 {self.segment_id} 缺少 grid")
            return self.grid[0] * self.grid[1]
        if self.length is None:
            raise SegmentError(f"段 {self.segment_id} 缺少 length")
        return self.length

    def truncated(self, keep: int) -> Segment:
        """返回只保留前 keep 个 token 的新段落（仅一维段）。

        逻辑坐标由 position_start 与局部偏移决定，截断不影响保留 token 的坐标。
        """
        if self.length is None:
            raise SegmentError(f"段 {self.segment_id} 是二维网格段，不支持按长度截断")
        if not 0 < keep <= self.length:
            raise SegmentError(f"段 {self.segment_id} 非法截断长度 {keep}")
        return replace(self, length=keep)


def validate_segments(segments, packed_order, rules: FrequencyRules) -> dict:
    """校验段落集合与打包顺序，返回 {segment_id: Segment}。

    packed_order 必须是段落 id 的一个排列：物理打包顺序只决定物理下标，
    不参与逻辑位置编号。
    """
    segs = list(segments)
    if not segs:
        raise SegmentError("段落集合不能为空")
    by_id: dict[str, Segment] = {}
    for seg in segs:
        if not seg.segment_id:
            raise SegmentError("segment_id 不能为空")
        if seg.segment_id in by_id:
            raise SegmentError(f"重复的 segment_id: {seg.segment_id}")
        if not isinstance(seg.position_start, int) or seg.position_start < 0:
            raise SegmentError(f"段 {seg.segment_id} 的 position_start 必须为非负整数")
        if seg.kind not in rules.modalities:
            raise SegmentError(f"段 {seg.segment_id} 的模态 {seg.kind!r} 未在规则版本中定义")
        spec = rules.spec_for(seg.kind)
        if spec.mode == MODE_GRID2D:
            if seg.grid is None:
                raise SegmentError(f"段 {seg.segment_id} 是二维网格段，缺少 grid")
            if seg.length is not None:
                raise SegmentError(f"段 {seg.segment_id} 是二维网格段，不允许同时给出 length")
            if len(seg.grid) != 2 or any(not isinstance(d, int) or d <= 0 for d in seg.grid):
                raise SegmentError(f"段 {seg.segment_id} 的 grid 必须为两个正整数")
        else:
            if seg.length is None or not isinstance(seg.length, int) or seg.length <= 0:
                raise SegmentError(f"段 {seg.segment_id} 是一维段，length 必须为正整数")
            if seg.grid is not None:
                raise SegmentError(f"段 {seg.segment_id} 是一维段，不允许给出 grid")
        by_id[seg.segment_id] = seg
    order = list(packed_order)
    if sorted(order) != sorted(by_id):
        raise SegmentError("packed_order 必须是段落 id 的一个排列（不多不少、不重复）")
    return by_id
