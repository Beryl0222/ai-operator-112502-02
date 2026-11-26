"""算子资料包命名空间。

段落感知的旋转位置编码算子：逻辑位置只由段落自身描述与模态频率参数决定，
缓存按规则版本与坐标元数据校验，预填充 / 增量解码 / 分块重算三条路径相位一致。
"""

from .cache import (
    CacheEntry,
    CacheLookup,
    CacheReuseError,
    CacheStoreRecord,
    MissReason,
    PrefixCache,
    SegmentMetadata,
)
from .config import (
    FrequencyRules,
    ModalitySpec,
    OperatorConfig,
    RopeLayout,
    builtin_rules,
)
from .errors import ConfigError, SegmentError
from .operator import DecodeResult, RopeSegmentOperator, RotationOutput
from .positions import (
    Coord,
    PositionMap,
    PositionRecord,
    build_position_map,
    coords_for_segment,
)
from .rope import rotate_block, rotate_vector
from .segments import Segment, validate_segments

__all__ = [
    "CacheEntry",
    "CacheLookup",
    "CacheReuseError",
    "CacheStoreRecord",
    "ConfigError",
    "Coord",
    "DecodeResult",
    "FrequencyRules",
    "MissReason",
    "ModalitySpec",
    "OperatorConfig",
    "PositionMap",
    "PositionRecord",
    "PrefixCache",
    "RopeLayout",
    "RopeSegmentOperator",
    "RotationOutput",
    "Segment",
    "SegmentError",
    "SegmentMetadata",
    "build_position_map",
    "builtin_rules",
    "coords_for_segment",
    "rotate_block",
    "rotate_vector",
    "validate_segments",
]
