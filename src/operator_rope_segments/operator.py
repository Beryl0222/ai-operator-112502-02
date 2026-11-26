"""段落感知的旋转位置编码算子。

三条计算路径（预填充 / 增量解码 / 分块重算）共享同一份位置规划：
旋转相位只取决于逻辑坐标与模态频率参数，与物理打包顺序无关，
因此对同一组段落必然给出一致的相位与可追溯的位置映射。
"""

from __future__ import annotations

from dataclasses import dataclass

from .cache import CacheReuseError, PrefixCache
from .config import MODE_GRID2D, OperatorConfig
from .errors import SegmentError
from .positions import PositionMap, build_position_map
from .rope import rotate_block, rotate_vector
from .segments import validate_segments


@dataclass(frozen=True)
class RotationOutput:
    """预填充 / 分块重算的输出。q、k 按物理打包顺序排列。"""

    q: list
    k: list
    position_map: PositionMap
    stores: tuple = ()


@dataclass(frozen=True)
class DecodeResult:
    """增量解码的输出。

    full_k 由校验通过的缓存前缀与本次新计算的键按打包顺序拼接而成；
    前缀的查询向量在解码路径中不存在，如需全序列查询请对照 prefill 的对应切片。
    """

    q_new: list
    k_new: list
    full_k: list
    position_map: PositionMap
    lookups: tuple = ()
    stores: tuple = ()

    @property
    def new_token_count(self) -> int:
        return len(self.k_new)


class RopeSegmentOperator:
    """根据段落描述、缓存前缀与每段局部位置生成旋转后的查询与键。"""

    def __init__(self, config: OperatorConfig):
        self.config = config

    def plan(self, segments, packed_order) -> PositionMap:
        """位置规划：三条路径共用。物理下标按打包顺序分配，逻辑坐标与打包无关。"""
        return build_position_map(segments, packed_order, self.config.rules)

    # ---- 路径一：预填充 ----

    def prefill(self, segments, packed_order, q, k, cache: PrefixCache | None = None) -> RotationOutput:
        """整批旋转全部段落的查询与键；可选地把旋转键写入缓存。"""
        by_id = validate_segments(segments, packed_order, self.config.rules)
        pmap = build_position_map(segments, packed_order, self.config.rules)
        q = self._check_vectors(q, len(pmap), "q")
        k = self._check_vectors(k, len(pmap), "k")
        rot_q = self._rotate_packed(by_id, packed_order, pmap, q)
        rot_k = self._rotate_packed(by_id, packed_order, pmap, k)
        stores = self._store_all(cache, by_id, packed_order, pmap, rot_k) if cache is not None else []
        return RotationOutput(q=rot_q, k=rot_k, position_map=pmap, stores=tuple(stores))

    # ---- 路径二：增量解码 ----

    def decode_incremental(
        self,
        segments,
        packed_order,
        q_new,
        k_new,
        new_offsets: dict,
        cache: PrefixCache | None,
    ) -> DecodeResult:
        """复用缓存前缀，只旋转新增 token。

        new_offsets: {segment_id: (start, end)}，段内局部偏移的新增范围；
        end 必须等于段当前 token 数（仅支持末尾追加），未列出的段视为完全复用缓存。
        任何一段复用校验失败都整体拒绝并报告原因，绝不混入旧结果。
        """
        if cache is None:
            raise SegmentError("增量解码需要前缀缓存；无缓存时请使用 prefill")
        by_id = validate_segments(segments, packed_order, self.config.rules)
        pmap = build_position_map(segments, packed_order, self.config.rules)
        unknown = set(new_offsets) - set(by_id)
        if unknown:
            raise SegmentError(f"new_offsets 包含未知段: {sorted(unknown)}")
        plan: dict[str, tuple[int, int]] = {}
        for sid, seg in by_id.items():
            spec = self.config.rules.spec_for(seg.kind)
            total = seg.token_count(spec)
            start, end = new_offsets.get(sid, (total, total))
            if not (0 <= start <= end <= total):
                raise SegmentError(f"段 {sid} 的新增范围 ({start}, {end}) 超出 [0, {total}]")
            if end != total:
                raise SegmentError(f"段 {sid} 仅支持末尾追加：新增范围须结束于 {total}")
            if spec.mode == MODE_GRID2D and 0 < start < end:
                raise SegmentError(f"段 {sid} 是二维网格段，只能整体新增或整体复用，不支持中段追加")
            plan[sid] = (start, end)
        total_new = sum(end - start for start, end in plan.values())
        q_new = self._check_vectors(q_new, total_new, "q_new")
        k_new = self._check_vectors(k_new, total_new, "k_new")

        # 先校验全部复用请求：任何一段不可复用就整体拒绝。
        lookups: dict[str, object] = {}
        rejections = []
        for sid in packed_order:
            start, _ = plan[sid]
            if start > 0:
                seg = by_id[sid]
                spec = self.config.rules.spec_for(seg.kind)
                lookup = cache.lookup(seg, spec, self.config, needed_end=start)
                lookups[sid] = lookup
                if not lookup.hit:
                    rejections.append(lookup)
        if rejections:
            raise CacheReuseError(
                "缓存校验失败，拒绝复用: " + "；".join(item.detail for item in rejections),
                lookups=rejections,
            )

        q_rot: list = []
        k_rot: list = []
        full_k: list = []
        stores = []
        cursor = 0
        for sid in packed_order:
            seg = by_id[sid]
            spec = self.config.rules.spec_for(seg.kind)
            start, end = plan[sid]
            coords = [rec.coord for rec in pmap.records_for(sid)]
            take = end - start
            block_q = q_new[cursor:cursor + take]
            block_k = k_new[cursor:cursor + take]
            cursor += take
            rot_q = rotate_block(block_q, coords[start:end], spec, self.config.layout)
            rot_k = rotate_block(block_k, coords[start:end], spec, self.config.layout)
            q_rot.extend(rot_q)
            k_rot.extend(rot_k)
            prefix = [list(v) for v in lookups[sid].entry.keys[:start]] if start > 0 else []
            full_k.extend(prefix + rot_k)
            if take:
                if start == 0:
                    stores.append(cache.store(seg, spec, self.config, end, rot_k))
                else:
                    stores.append(cache.extend(seg, spec, self.config, start, rot_k))
        return DecodeResult(
            q_new=q_rot,
            k_new=k_rot,
            full_k=full_k,
            position_map=pmap,
            lookups=tuple(lookups.values()),
            stores=tuple(stores),
        )

    # ---- 路径三：分块重算 ----

    def chunked_recompute(
        self,
        segments,
        packed_order,
        q,
        k,
        chunk_size: int,
        cache: PrefixCache | None = None,
    ) -> RotationOutput:
        """按固定块长沿物理轴分块重算。块长不影响结果：相位与物理下标无关。"""
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise SegmentError("chunk_size 必须为正整数")
        by_id = validate_segments(segments, packed_order, self.config.rules)
        pmap = build_position_map(segments, packed_order, self.config.rules)
        q = self._check_vectors(q, len(pmap), "q")
        k = self._check_vectors(k, len(pmap), "k")
        specs = {sid: self.config.rules.spec_for(seg.kind) for sid, seg in by_id.items()}
        rot_q: list = []
        rot_k: list = []
        for lo in range(0, len(pmap), chunk_size):
            for rec in pmap.records[lo:lo + chunk_size]:
                spec = specs[rec.segment_id]
                rot_q.append(rotate_vector(q[rec.physical_index], rec.coord, spec, self.config.layout))
                rot_k.append(rotate_vector(k[rec.physical_index], rec.coord, spec, self.config.layout))
        stores = self._store_all(cache, by_id, packed_order, pmap, rot_k) if cache is not None else []
        return RotationOutput(q=rot_q, k=rot_k, position_map=pmap, stores=tuple(stores))

    # ---- 内部工具 ----

    def _rotate_packed(self, by_id, packed_order, pmap: PositionMap, vectors) -> list:
        out: list = []
        for sid in packed_order:
            seg = by_id[sid]
            spec = self.config.rules.spec_for(seg.kind)
            records = pmap.records_for(sid)
            block = [vectors[rec.physical_index] for rec in records]
            coords = [rec.coord for rec in records]
            out.extend(rotate_block(block, coords, spec, self.config.layout))
        return out

    def _store_all(self, cache, by_id, packed_order, pmap: PositionMap, rot_k) -> list:
        stores = []
        for sid in packed_order:
            seg = by_id[sid]
            spec = self.config.rules.spec_for(seg.kind)
            records = pmap.records_for(sid)
            block = [rot_k[rec.physical_index] for rec in records]
            stores.append(cache.store(seg, spec, self.config, len(block), block))
        return stores

    def _check_vectors(self, vectors, expected: int, name: str) -> list:
        vectors = [list(v) for v in vectors]
        if len(vectors) != expected:
            raise SegmentError(f"{name} 的向量数 {len(vectors)} 与 token 总数 {expected} 不一致")
        dim = self.config.layout.head_dim
        for vec in vectors:
            if len(vec) != dim:
                raise SegmentError(f"{name} 的向量维度 {len(vec)} 与 head_dim {dim} 不一致")
        return vectors
