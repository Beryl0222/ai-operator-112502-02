import json
from pathlib import Path
import unittest


class FixtureContractTest(unittest.TestCase):
    def test_rope_fixture_preserves_logical_segment_positions(self):
        base = Path(__file__).parents[1]
        payload = json.loads((base / "data/input.json").read_text(encoding="utf-8"))
        segments = payload["segments"]
        self.assertEqual(set(payload["packed_order"]), {segment["segment_id"] for segment in segments})
        self.assertEqual({segment["kind"] for segment in segments}, {"text", "image", "retrieval"})
        self.assertNotEqual(len({segment["position_start"] for segment in segments}), 1)

if __name__ == "__main__":
    unittest.main()
