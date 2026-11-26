"""频率规则与旋转布局配置。

位置规则按版本管理：同一版本号必须对应完全相同的模态参数。缓存条目记录
版本号与模态参数快照，跨版本或参数不一致时一律拒绝复用。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ConfigError

MODE_LINEAR = "linear"
MODE_GRID2D = "grid2d"


@dataclass(frozen=True)
class ModalitySpec:
    """单一模态的频率参数。

    - mode: ``linear``（文本/检索，一维时间轴）或 ``grid2d``（图像，二维网格）。
    - theta: 旋转基底。
    - freq_scale: 坐标缩放因子；旋转角 = 逻辑坐标 * freq_scale * theta**(-2j/d)。
    """

    kind: str
    mode: str
    theta: float
    freq_scale: float

    def __post_init__(self) -> None:
        if not self.kind:
            raise ConfigError("模态 kind 不能为空")
        if self.mode not in (MODE_LINEAR, MODE_GRID2D):
            raise ConfigError(f"未知模态模式: {self.mode!r}")
        if not self.theta > 0:
            raise ConfigError("theta 必须为正数")
        if not self.freq_scale > 0:
            raise ConfigError("freq_scale 必须为正数")


@dataclass(frozen=True)
class RopeLayout:
    """头维度在三个逻辑轴 (t, h, w) 上的切分方式。

    sections 为每个轴的旋转对数，三者之和必须等于 head_dim // 2。
    """

    head_dim: int
    sections: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ConfigError("head_dim 必须为正偶数")
        if len(self.sections) != 3 or any(not isinstance(s, int) or s < 0 for s in self.sections):
            raise ConfigError("sections 必须为三个非负整数")
        if sum(self.sections) != self.head_dim // 2:
            raise ConfigError("sections 之和必须等于 head_dim // 2")
        if not any(self.sections):
            raise ConfigError("sections 不能全为 0")


@dataclass(frozen=True)
class FrequencyRules:
    """一个规则版本的全部模态频率参数。"""

    version: str
    modalities: dict[str, ModalitySpec]

    def __post_init__(self) -> None:
        if not self.version:
            raise ConfigError("version 不能为空")
        if not self.modalities:
            raise ConfigError("modalities 不能为空")
        for kind, spec in self.modalities.items():
            if spec.kind != kind:
                raise ConfigError(f"模态键 {kind!r} 与 spec.kind {spec.kind!r} 不一致")

    def spec_for(self, kind: str) -> ModalitySpec:
        try:
            return self.modalities[kind]
        except KeyError:
            raise ConfigError(f"规则版本 {self.version} 未定义模态 {kind!r}") from None


@dataclass(frozen=True)
class OperatorConfig:
    """算子完整配置：频率规则版本 + 头维布局。"""

    rules: FrequencyRules
    layout: RopeLayout

    @property
    def version(self) -> str:
        return self.rules.version


def _v2_modalities() -> dict[str, ModalitySpec]:
    return {
        "text": ModalitySpec("text", MODE_LINEAR, theta=5_000.0, freq_scale=1.0),
        "image": ModalitySpec("image", MODE_GRID2D, theta=5_000.0, freq_scale=1.0),
        "retrieval": ModalitySpec("retrieval", MODE_LINEAR, theta=10_000.0, freq_scale=1.0),
    }


def _v3_modalities() -> dict[str, ModalitySpec]:
    return {
        "text": ModalitySpec("text", MODE_LINEAR, theta=10_000.0, freq_scale=1.0),
        "image": ModalitySpec("image", MODE_GRID2D, theta=10_000.0, freq_scale=1.0),
        "retrieval": ModalitySpec("retrieval", MODE_LINEAR, theta=20_000.0, freq_scale=0.5),
    }


_BUILTIN = {"v2": _v2_modalities, "v3": _v3_modalities}


def builtin_rules(version: str) -> FrequencyRules:
    """返回内置规则版本；未知版本抛出 ConfigError。"""
    try:
        factory = _BUILTIN[version]
    except KeyError:
        raise ConfigError(f"未知频率规则版本 {version!r}，可用版本: {sorted(_BUILTIN)}") from None
    return FrequencyRules(version=version, modalities=factory())
