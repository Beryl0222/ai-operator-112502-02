import json
import unittest
from pathlib import Path

from operator_rope_segments import (
    CacheReuseError,
    ConfigError,
    Coord,
    FrequencyRules,
    MissReason,
    ModalitySpec,
    OperatorConfig,
    PrefixCache,
    RopeLayout,
    RopeSegmentOperator,
    Segment,
    SegmentError,
    builtin_rules,
    rotate_vector,
)

HEAD_DIM = 8
SECTIONS = (2, 1, 1)  # t 轴 2 对，h/w 各 1 对


def make_config(version="v3"):
    return OperatorConfig(rules=builtin_rules(version), layout=RopeLayout(head_dim=HEAD_DIM, sections=SECTIONS))


def make_operator(version="v3"):
    return RopeSegmentOperator(make_config(version))


def make_vectors(count, seed=0):
    return [[((i * 7 + j * 3 + seed) % 13) * 0.25 - 1.5 for j in range(HEAD_DIM)] for i in range(count)]


def fixture_segments():
    base = Path(__file__).parents[1]
    payload = json.loads((base / "data" / "input.json").read_text(encoding="utf-8"))
    segments = [
        Segment(
            segment_id=raw["segment_id"],
            kind=raw["kind"],
            position_start=raw["position_start"],
            length=raw.get("length"),
            grid=tuple(raw["grid"]) if raw.get("grid") is not None else None,
        )
        for raw in payload["segments"]
    ]
    return segments, payload["packed_order"]


class PositionPlanTest(unittest.TestCase):
    """位置规划：逻辑坐标与物理打包解耦，可追溯。"""

    def setUp(self):
        self.op = make_operator()
        self.segments, self.order = fixture_segments()

    def test_fixture_logical_coords(self):
        pmap = self.op.plan(self.segments, self.order)
        self.assertEqual(len(pmap), 9)
        self.assertEqual(
            [r.coord for r in pmap.records_for("text-0")],
            [Coord(10, 10, 10), Coord(11, 11, 11), Coord(12, 12, 12)],
        )
        self.assertEqual(
            [r.coord for r in pmap.records_for("image-0")],
            [Coord(0, 0, 0), Coord(0, 0, 1), Coord(0, 1, 0), Coord(0, 1, 1)],
        )
        self.assertEqual(
            [r.coord for r in pmap.records_for("retrieval-0")],
            [Coord(40, 40, 40), Coord(41, 41, 41)],
        )

    def test_physical_spans_and_segment_mask(self):
        pmap = self.op.plan(self.segments, self.order)
        self.assertEqual(
            pmap.segment_spans(),
            {"text-0": (0, 3), "image-0": (3, 7), "retrieval-0": (7, 9)},
        )
        mask = pmap.same_segment_mask()
        self.assertTrue(mask[0][2])   # 同属 text-0
        self.assertFalse(mask[0][3])  # text-0 与 image-0 仅物理相邻
        self.assertTrue(mask[3][6])   # 同属 image-0
        self.assertFalse(mask[3][7])  # image-0 与 retrieval-0 仅物理相邻
        self.assertTrue(mask[7][8])   # 同属 retrieval-0
        rec = pmap.record("image-0", 3)
        self.assertEqual((rec.physical_index, rec.coord), (6, Coord(0, 1, 1)))

    def test_reorder_keeps_logical_coords(self):
        pmap = self.op.plan(self.segments, self.order)
        reordered = list(reversed(self.order))
        pmap2 = self.op.plan(self.segments, reordered)
        for sid in ("text-0", "image-0", "retrieval-0"):
            self.assertEqual(
                [r.coord for r in pmap.records_for(sid)],
                [r.coord for r in pmap2.records_for(sid)],
            )
        self.assertEqual(pmap2.segment_spans()["retrieval-0"], (0, 2))
        self.assertEqual(pmap2.segment_spans()["image-0"], (2, 6))
        self.assertEqual(pmap2.segment_spans()["text-0"], (6, 9))

    def test_truncate_and_restore_keep_coords(self):
        text5 = Segment("t", "text", 10, length=5)
        img = Segment("i", "image", 0, grid=(2, 2))
        order = ["t", "i"]
        full = self.op.plan([text5, img], order)
        kept = self.op.plan([text5.truncated(3), img], order)
        self.assertEqual(
            [r.coord for r in kept.records_for("t")],
            [r.coord for r in full.records_for("t")][:3],
        )
        self.assertEqual(
            [r.coord for r in kept.records_for("i")],
            [r.coord for r in full.records_for("i")],
        )
        restored = self.op.plan([text5, img], order)
        self.assertEqual(restored, full)


class RotationTest(unittest.TestCase):
    """旋转核：模态频率缩放与二维网格坐标。"""

    def setUp(self):
        self.config = make_config()

    def test_origin_coord_is_identity(self):
        spec = self.config.rules.spec_for("text")
        vec = make_vectors(1)[0]
        self.assertEqual(rotate_vector(vec, Coord(0, 0, 0), spec, self.config.layout), vec)

    def test_modality_freq_scaling_differs(self):
        vec = make_vectors(1)[0]
        coord = Coord(4, 4, 4)
        out_text = rotate_vector(vec, coord, self.config.rules.spec_for("text"), self.config.layout)
        out_retr = rotate_vector(vec, coord, self.config.rules.spec_for("retrieval"), self.config.layout)
        self.assertNotEqual(out_text, out_retr)

    def test_grid_axes_are_independent(self):
        spec = self.config.rules.spec_for("image")
        vec = make_vectors(1)[0]
        by_w = rotate_vector(vec, Coord(0, 0, 1), spec, self.config.layout)
        by_h = rotate_vector(vec, Coord(0, 1, 0), spec, self.config.layout)
        # t 轴坐标同为 0：t 段（前 4 个分量）保持一致
        self.assertEqual(by_w[:4], by_h[:4])
        self.assertNotEqual(by_w, by_h)
        # w 段（最后两个分量）只在 w 坐标非零时旋转
        self.assertNotEqual(by_w[6:], vec[6:])
        self.assertEqual(by_h[6:], vec[6:])


class PathConsistencyTest(unittest.TestCase):
    """预填充 / 增量解码 / 分块重算三条路径对同一组段落结果一致。"""

    def setUp(self):
        self.op = make_operator()
        self.segments, self.order = fixture_segments()
        self.q = make_vectors(9)
        self.k = make_vectors(9, seed=5)

    def test_chunked_recompute_matches_prefill(self):
        full = self.op.prefill(self.segments, self.order, self.q, self.k)
        for chunk_size in (1, 2, 3, 4, 9, 100):
            out = self.op.chunked_recompute(self.segments, self.order, self.q, self.k, chunk_size)
            self.assertEqual(out.q, full.q)
            self.assertEqual(out.k, full.k)
            self.assertEqual(out.position_map, full.position_map)

    def test_decode_with_cached_prefix_matches_prefill(self):
        # 服务场景：已缓存的文本前缀接到新图像/检索片段前面
        full = self.op.prefill(self.segments, self.order, self.q, self.k)
        cache = PrefixCache()
        text = self.segments[0]
        self.op.prefill([text], ["text-0"], self.q[:3], self.k[:3], cache=cache)
        result = self.op.decode_incremental(
            self.segments,
            self.order,
            q_new=self.q[3:],
            k_new=self.k[3:],
            new_offsets={"image-0": (0, 4), "retrieval-0": (0, 2)},
            cache=cache,
        )
        self.assertEqual(result.full_k, full.k)
        self.assertEqual(result.q_new, full.q[3:])
        self.assertEqual(result.position_map, full.position_map)
        self.assertEqual(len(result.lookups), 1)
        self.assertTrue(result.lookups[0].hit)

    def test_standalone_and_packed_image_phases_match(self):
        # 回归：同一图像段在单独请求与拼接请求中的旋转相位必须一致
        image = self.segments[1]
        solo = self.op.prefill([image], ["image-0"], self.q[3:7], self.k[3:7])
        packed = self.op.prefill(self.segments, self.order, self.q, self.k)
        self.assertEqual(solo.q, packed.q[3:7])
        self.assertEqual(solo.k, packed.k[3:7])

    def test_incremental_growth_matches_prefill(self):
        cache = PrefixCache()
        q = make_vectors(5)
        k = make_vectors(5, seed=3)
        seg3 = Segment("text-0", "text", 10, length=3)
        self.op.prefill([seg3], ["text-0"], q[:3], k[:3], cache=cache)
        for step in (4, 5):
            seg = Segment("text-0", "text", 10, length=step)
            result = self.op.decode_incremental(
                [seg],
                ["text-0"],
                q_new=q[step - 1:step],
                k_new=k[step - 1:step],
                new_offsets={"text-0": (step - 1, step)},
                cache=cache,
            )
            ref = self.op.prefill([seg], ["text-0"], q[:step], k[:step])
            self.assertEqual(result.full_k, ref.k)
            self.assertEqual(result.q_new, ref.q[step - 1:step])
            last = result.position_map.record("text-0", step - 1)
            self.assertEqual(last.coord, Coord(10 + step - 1, 10 + step - 1, 10 + step - 1))

    def test_cache_is_order_independent(self):
        # 段级缓存与打包顺序无关：重排后复用缓存仍与整体预填充一致
        cache = PrefixCache()
        self.op.prefill(self.segments[:2], ["text-0", "image-0"], self.q[:7], self.k[:7], cache=cache)
        new_order = ["retrieval-0", "image-0", "text-0"]
        result = self.op.decode_incremental(
            self.segments,
            new_order,
            q_new=self.q[7:9],
            k_new=self.k[7:9],
            new_offsets={"retrieval-0": (0, 2)},
            cache=cache,
        )
        q_re = self.q[7:9] + self.q[3:7] + self.q[0:3]
        k_re = self.k[7:9] + self.k[3:7] + self.k[0:3]
        ref = self.op.prefill(self.segments, new_order, q_re, k_re)
        self.assertEqual(result.full_k, ref.k)
        self.assertEqual(result.q_new, ref.q[0:2])  # retrieval-0 位于新顺序最前


class CacheContractTest(unittest.TestCase):
    """缓存复用契约：版本/元数据/覆盖范围校验与结构化拒绝原因。"""

    def setUp(self):
        self.op = make_operator()
        self.segments, self.order = fixture_segments()
        self.q = make_vectors(9)
        self.k = make_vectors(9, seed=5)
        self.cache = PrefixCache()
        self.op.prefill([self.segments[0]], ["text-0"], self.q[:3], self.k[:3], cache=self.cache)

    def test_version_mismatch_reported(self):
        other = make_operator("v2")
        with self.assertRaises(CacheReuseError) as ctx:
            other.decode_incremental([self.segments[0]], ["text-0"], [], [], {}, self.cache)
        lookup = ctx.exception.lookups[0]
        self.assertEqual(lookup.reason, MissReason.VERSION_MISMATCH)
        self.assertIn("v3", lookup.detail)
        self.assertIn("v2", lookup.detail)
        # 缓存未被污染：用原版本仍可命中
        ok = self.cache.lookup(
            self.segments[0], self.op.config.rules.spec_for("text"), self.op.config, 3
        )
        self.assertTrue(ok.hit)

    def test_position_start_mismatch_reported(self):
        moved = Segment("text-0", "text", 11, length=3)
        with self.assertRaises(CacheReuseError) as ctx:
            self.op.decode_incremental([moved], ["text-0"], [], [], {}, self.cache)
        lookup = ctx.exception.lookups[0]
        self.assertEqual(lookup.reason, MissReason.METADATA_MISMATCH)
        self.assertIn("position_start", lookup.detail)

    def test_grid_mismatch_reported(self):
        self.op.prefill([self.segments[1]], ["image-0"], self.q[3:7], self.k[3:7], cache=self.cache)
        grown = Segment("image-0", "image", 0, grid=(3, 2))
        with self.assertRaises(CacheReuseError) as ctx:
            self.op.decode_incremental([grown], ["image-0"], [], [], {}, self.cache)
        lookup = ctx.exception.lookups[0]
        self.assertEqual(lookup.reason, MissReason.METADATA_MISMATCH)
        self.assertIn("grid", lookup.detail)

    def test_frequency_param_mismatch_reported(self):
        # 版本号相同但频率参数不同，同样按元数据不一致拒绝
        mods = dict(builtin_rules("v3").modalities)
        mods["text"] = ModalitySpec("text", "linear", 999.0, 1.0)
        altered = RopeSegmentOperator(OperatorConfig(FrequencyRules("v3", mods), make_config().layout))
        with self.assertRaises(CacheReuseError) as ctx:
            altered.decode_incremental([self.segments[0]], ["text-0"], [], [], {}, self.cache)
        lookup = ctx.exception.lookups[0]
        self.assertEqual(lookup.reason, MissReason.METADATA_MISMATCH)
        self.assertIn("theta", lookup.detail)

    def test_incomplete_coverage_reported(self):
        longer = Segment("text-0", "text", 10, length=5)
        with self.assertRaises(CacheReuseError) as ctx:
            self.op.decode_incremental([longer], ["text-0"], [], [], {}, self.cache)
        lookup = ctx.exception.lookups[0]
        self.assertEqual(lookup.reason, MissReason.INCOMPLETE_COVERAGE)
        self.assertIn("[0,3)", lookup.detail)
        self.assertIn("[0,5)", lookup.detail)

    def test_absent_reported(self):
        image = self.segments[1]
        lookup = self.cache.lookup(image, self.op.config.rules.spec_for("image"), self.op.config, 4)
        self.assertFalse(lookup.hit)
        self.assertEqual(lookup.reason, MissReason.ABSENT)

    def test_store_idempotent_and_conflict_requires_invalidate(self):
        text = self.segments[0]
        out = self.op.prefill([text], ["text-0"], self.q[:3], self.k[:3], cache=self.cache)
        self.assertFalse(out.stores[0].stored)  # 重复写入幂等跳过
        moved = Segment("text-0", "text", 11, length=3)
        with self.assertRaises(CacheReuseError):
            self.op.prefill([moved], ["text-0"], self.q[:3], self.k[:3], cache=self.cache)
        self.assertTrue(self.cache.invalidate("text-0"))
        out = self.op.prefill([moved], ["text-0"], self.q[:3], self.k[:3], cache=self.cache)
        self.assertTrue(out.stores[0].stored)

    def test_truncated_view_reuses_cached_prefix(self):
        cache = PrefixCache()
        text5 = Segment("text-0", "text", 10, length=5)
        q5, k5 = make_vectors(5), make_vectors(5, seed=3)
        self.op.prefill([text5], ["text-0"], q5, k5, cache=cache)
        text3 = text5.truncated(3)
        result = self.op.decode_incremental([text3], ["text-0"], [], [], {}, cache)
        ref = self.op.prefill([text3], ["text-0"], q5[:3], k5[:3])
        self.assertEqual(result.full_k, ref.k)

    def test_restore_after_truncation_extends_cache(self):
        cache = PrefixCache()
        text5 = Segment("text-0", "text", 10, length=5)
        text3 = text5.truncated(3)
        q5, k5 = make_vectors(5), make_vectors(5, seed=3)
        self.op.prefill([text3], ["text-0"], q5[:3], k5[:3], cache=cache)
        result = self.op.decode_incremental(
            [text5], ["text-0"], q5[3:], k5[3:], {"text-0": (3, 5)}, cache
        )
        ref = self.op.prefill([text5], ["text-0"], q5, k5)
        self.assertEqual(result.full_k, ref.k)
        self.assertEqual(cache.get("text-0").covered_end, 5)


class ValidationTest(unittest.TestCase):
    """非法输入一律显式拒绝。"""

    def setUp(self):
        self.op = make_operator()

    def test_layout_validation(self):
        with self.assertRaises(ConfigError):
            RopeLayout(head_dim=7, sections=(2, 1, 1))
        with self.assertRaises(ConfigError):
            RopeLayout(head_dim=8, sections=(2, 2, 1))
        with self.assertRaises(ConfigError):
            RopeLayout(head_dim=8, sections=(0, 0, 0))

    def test_unknown_builtin_version(self):
        with self.assertRaises(ConfigError):
            builtin_rules("v9")

    def test_segment_validation(self):
        with self.assertRaises(SegmentError):
            self.op.plan([Segment("i", "image", 0)], ["i"])
        with self.assertRaises(SegmentError):
            self.op.plan([Segment("t", "text", 0)], ["t"])
        with self.assertRaises(SegmentError):
            self.op.plan([Segment("i", "image", 0, length=4, grid=(2, 2))], ["i"])
        with self.assertRaises(SegmentError):
            self.op.plan([Segment("a", "audio", 0, length=1)], ["a"])
        with self.assertRaises(SegmentError):
            self.op.plan([Segment("t", "text", -1, length=1)], ["t"])

    def test_packing_validation(self):
        t1 = Segment("t1", "text", 0, length=2)
        t1dup = Segment("t1", "text", 5, length=2)
        t2 = Segment("t2", "text", 0, length=2)
        with self.assertRaises(SegmentError):
            self.op.plan([t1, t1dup], ["t1", "t1"])
        with self.assertRaises(SegmentError):
            self.op.plan([t1, t2], ["t1"])
        with self.assertRaises(SegmentError):
            self.op.plan([t1], ["t1", "t2"])

    def test_vector_validation(self):
        t1 = Segment("t1", "text", 0, length=2)
        with self.assertRaises(SegmentError):
            self.op.prefill([t1], ["t1"], make_vectors(1), make_vectors(2))
        bad_dim = [[0.0] * 4 for _ in range(2)]
        with self.assertRaises(SegmentError):
            self.op.prefill([t1], ["t1"], bad_dim, bad_dim)
        with self.assertRaises(SegmentError):
            self.op.chunked_recompute([t1], ["t1"], make_vectors(2), make_vectors(2), 0)

    def test_decode_validation(self):
        t3 = Segment("t", "text", 0, length=3)
        with self.assertRaises(SegmentError):
            self.op.decode_incremental([t3], ["t"], [], [], {}, None)
        cache = PrefixCache()
        with self.assertRaises(SegmentError):
            self.op.decode_incremental([t3], ["t"], make_vectors(1), make_vectors(1), {"t": (1, 2)}, cache)
        with self.assertRaises(SegmentError):
            self.op.decode_incremental([t3], ["t"], [], [], {"ghost": (0, 1)}, cache)
        img = Segment("i", "image", 0, grid=(2, 2))
        with self.assertRaises(SegmentError):
            self.op.decode_incremental([img], ["i"], make_vectors(3), make_vectors(3), {"i": (1, 4)}, cache)


if __name__ == "__main__":
    unittest.main()
