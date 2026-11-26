"""算子异常类型。"""


class ConfigError(ValueError):
    """频率规则或旋转布局配置非法。"""


class SegmentError(ValueError):
    """段落描述、打包顺序或输入向量非法。"""
