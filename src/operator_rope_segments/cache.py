"""段级前缀缓存：只在规则版本一致、坐标元数据一致、覆盖范围完整时复用。

缓存按段索引，与打包顺序无关；任何校验失败都返回结构化原因，
绝不把旧结果混入新请求。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .config import ModalitySpec, OperatorConfig, RopeLayout
from .segments import Segment


class MissReason(enum.Enum):
    """缓存不可复用的结构化原因。"""

    ABSENT = "absent"  # 缓存中没有该段
    VERSION_MISMATCH = "version_mismatch"  # 位置规则版本不一致
    METADATA_MISMATCH = "metadata_mismatch"  # 坐标元数据（起点/网格/频率参数/布局）不一致
    INCOMPLETE_COVERAGE = "incomplete_coverage"  # 覆盖范围不完整


@dataclass(frozen=True)
class SegmentMetadata:
    """决定旋转相位的全部坐标元数据；任何一项变化都禁止复用。"""

    kind: str
    mode: str
    position_start: int
    grid: tuple[int, int] | None
    theta: float
    freq_scale: float

    @classmethod
    def of(cls, segment: Segment, spec: ModalitySpec) -> "SegmentMetadata":
        return cls(
            kind=segment.kind,
            mode=spec.mode,
            position_start=segment.position_start,
            grid=segment.grid,
            theta=spec.theta,
            freq_scale=spec.freq_scale,
        )

    def diff_fields(self, other: "SegmentMetadata") -> list[str]:
        return [
            name
            for name in ("kind", "mode", "position_start", "grid", "theta", "freq_scale")
            if getattr(self, name) != getattr(other, name)
        ]


@dataclass
class CacheEntry:
    """单段前缀缓存条目。覆盖范围恒为局部偏移 [0, covered_end)。"""

    segment_id: str
    rules_version: str
    layout: RopeLayout
    metadata: SegmentMetadata
    covered_end: int
    keys: list  # 旋转后的键，keys[i] 对应段内局部偏移 i


@dataclass(frozen=True)
class CacheLookup:
    """一次复用检查的结果；未命中时携带原因与可读细节。"""

    hit: bool
    reason: MissReason | None
    detail: str
    entry: CacheEntry | None = None


@dataclass(frozen=True)
class CacheStoreRecord:
    """一次写入/扩展的记录，便于调用方审计。"""

    segment_id: str
    covered_end: int
    stored: bool
    detail: str


class CacheReuseError(RuntimeError):
    """缓存不可复用时抛出，携带全部失败的 CacheLookup，绝不混入旧结果。"""

    def __init__(self, message: str, lookups=()):
        super().__init__(message)
        self.lookups = tuple(lookups)


class PrefixCache:
    """段级旋转键前缀缓存。"""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    def __contains__(self, segment_id: str) -> bool:
        return segment_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, segment_id: str) -> CacheEntry | None:
        return self._entries.get(segment_id)

    def invalidate(self, segment_id: str) -> bool:
        """显式作废一段缓存；版本或元数据变更后应先作废再写入。"""
        return self._entries.pop(segment_id, None) is not None

    def clear(self) -> None:
        self._entries.clear()

    def lookup(
        self,
        segment: Segment,
        spec: ModalitySpec,
        config: OperatorConfig,
        needed_end: int,
    ) -> CacheLookup:
        """检查局部偏移 [0, needed_end) 是否可复用；未命中时给出结构化原因。"""
        entry = self._entries.get(segment.segment_id)
        if entry is None:
            return CacheLookup(False, MissReason.ABSENT, f"段 {segment.segment_id} 不在缓存中")
        if entry.rules_version != config.version:
            return CacheLookup(
                False,
                MissReason.VERSION_MISMATCH,
                f"段 {segment.segment_id} 位置规则版本不一致: 缓存={entry.rules_version}, 当前={config.version}",
                entry,
            )
        if entry.layout != config.layout:
            return CacheLookup(
                False,
                MissReason.METADATA_MISMATCH,
                f"段 {segment.segment_id} 旋转布局不一致: 缓存={entry.layout}, 当前={config.layout}",
                entry,
            )
        current = SegmentMetadata.of(segment, spec)
        if entry.metadata != current:
            fields = ", ".join(entry.metadata.diff_fields(current))
            return CacheLookup(
                False,
                MissReason.METADATA_MISMATCH,
                f"段 {segment.segment_id} 坐标元数据不一致字段: {fields}",
                entry,
            )
        if needed_end > entry.covered_end:
            return CacheLookup(
                False,
                MissReason.INCOMPLETE_COVERAGE,
                f"段 {segment.segment_id} 缓存覆盖 [0,{entry.covered_end})，需要 [0,{needed_end})",
                entry,
            )
        return CacheLookup(True, None, "命中", entry)

    def store(
        self,
        segment: Segment,
        spec: ModalitySpec,
        config: OperatorConfig,
        covered_end: int,
        keys,
    ) -> CacheStoreRecord:
        """写入新条目。已有一致条目时幂等跳过；不一致条目拒绝覆盖，须先 invalidate。"""
        keys = [list(k) for k in keys]
        if covered_end <= 0 or covered_end != len(keys):
            raise CacheReuseError(f"段 {segment.segment_id} 存储覆盖范围与键数量不一致")
        entry = self._entries.get(segment.segment_id)
        if entry is not None:
            identical = (
                entry.rules_version == config.version
                and entry.layout == config.layout
                and entry.metadata == SegmentMetadata.of(segment, spec)
                and entry.covered_end == covered_end
                and entry.keys == keys
            )
            if identical:
                return CacheStoreRecord(segment.segment_id, covered_end, False, "条目已存在且一致，幂等跳过")
            raise CacheReuseError(
                f"段 {segment.segment_id} 已存在不一致的缓存条目，拒绝覆盖混用；请先 invalidate"
            )
        self._entries[segment.segment_id] = CacheEntry(
            segment_id=segment.segment_id,
            rules_version=config.version,
            layout=config.layout,
            metadata=SegmentMetadata.of(segment, spec),
            covered_end=covered_end,
            keys=keys,
        )
        return CacheStoreRecord(segment.segment_id, covered_end, True, "新建缓存条目")

    def extend(
        self,
        segment: Segment,
        spec: ModalitySpec,
        config: OperatorConfig,
        start: int,
        keys,
    ) -> CacheStoreRecord:
        """把局部偏移 [start, start+len(keys)) 的旋转键并入既有前缀条目。"""
        keys = [list(k) for k in keys]
        lookup = self.lookup(segment, spec, config, needed_end=start)
        if not lookup.hit:
            raise CacheReuseError(f"段 {segment.segment_id} 扩展前校验失败: {lookup.detail}", lookups=[lookup])
        entry = lookup.entry
        end = start + len(keys)
        if end <= entry.covered_end:
            return CacheStoreRecord(segment.segment_id, entry.covered_end, False, "扩展范围已完全被覆盖")
        entry.keys[start:end] = keys
        entry.covered_end = end
        return CacheStoreRecord(segment.segment_id, end, True, f"覆盖扩展至 [0,{end})")
