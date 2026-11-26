"""逻辑位置分配与可追溯的位置映射。

逻辑坐标只由段落自身描述（position_start、length/grid）与模态模式决定，
与段落在 batch 中的物理排列无关；物理下标仅是拼接后的偏移。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import MODE_GRID2D, FrequencyRules, ModalitySpec
from .segments import Segment, validate_segments


@dataclass(frozen=True)
class Coord:
    """单个 token 的三轴逻辑坐标 (t, h, w)。"""

    t: int
    h: int
    w: int


def coords_for_segment(segment: Segment, spec: ModalitySpec) -> list[Coord]:
    """生成段内每个 token 的逻辑坐标（网格段按行主序），与物理打包位置无关。

    - 一维段（文本/检索）：(s+i, s+i, s+i)，s 为 position_start；
    - 二维网格段（图像）：(s, s+r, s+c)，时间轴恒定，行列来自网格坐标。
    """
    base = segment.position_start
    if spec.mode == MODE_GRID2D:
        if segment.grid is None:
            raise SegmentError(f"段 {segment.segment_id} 缺少 grid")
        rows, cols = segment.grid
        return [Coord(base, base + r, base + c) for r in range(rows) for c in range(cols)]
    if segment.length is None:
        raise SegmentError(f"段 {segment.segment_id} 缺少 length")
    return [Coord(base + i, base + i, base + i) for i in range(segment.length)]


@dataclass(frozen=True)
class PositionRecord:
    """单个 token 的可追溯位置记录。

    physical_index 只是拼接后的物理偏移；segment_id + local_offset + coord
    才是注意力应当依据的逻辑身份。
    """

    physical_index: int
    segment_id: str
    local_offset: int
    coord: Coord


@dataclass(frozen=True)
class PositionMap:
    """全序列的位置映射，按物理下标排序。"""

    records: tuple[PositionRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    def segment_spans(self) -> dict[str, tuple[int, int]]:
        """每个段占据的物理区间 [start, end)。"""
        spans: dict[str, tuple[int, int]] = {}
        for rec in self.records:
            if rec.segment_id in spans:
                start, end = spans[rec.segment_id]
                spans[rec.segment_id] = (min(start, rec.physical_index), max(end, rec.physical_index + 1))
            else:
                spans[rec.segment_id] = (rec.physical_index, rec.physical_index + 1)
        return spans

    def segment_ids(self) -> list[str]:
        """每个物理位置所属的段 id。"""
        return [rec.segment_id for rec in self.records]

    def coords(self) -> list[Coord]:
        """每个物理位置的逻辑坐标。"""
        return [rec.coord for rec in self.records]

    def same_segment_mask(self) -> list[list[bool]]:
        """n×n 布尔矩阵：mask[i][j] 为真表示两个物理位置属于同一段。

        供注意力区分“同一段的位置”与“仅拼接后物理相邻的位置”。
        """
        ids = self.segment_ids()
        return [[a == b for b in ids] for a in ids]

    def record(self, segment_id: str, local_offset: int) -> PositionRecord:
        for rec in self.records:
            if rec.segment_id == segment_id and rec.local_offset == local_offset:
                return rec
        raise KeyError((segment_id, local_offset))

    def records_for(self, segment_id: str) -> tuple[PositionRecord, ...]:
        return tuple(rec for rec in self.records if rec.segment_id == segment_id)


def build_position_map(segments, packed_order, rules: FrequencyRules) -> PositionMap:
    """按打包顺序分配物理下标；逻辑坐标只来自各段自身的局部起点。"""
    by_id = validate_segments(segments, packed_order, rules)
    records: list[PositionRecord] = []
    index = 0
    for segment_id in packed_order:
        seg = by_id[segment_id]
        spec = rules.spec_for(seg.kind)
        for offset, coord in enumerate(coords_for_segment(seg, spec)):
            records.append(PositionRecord(index, segment_id, offset, coord))
            index += 1
    return PositionMap(tuple(records))
