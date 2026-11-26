# 变长 RoPE 位置编码与跨模态段落拼接

这是一个 Python 算子工程的资料基线，包含与 跨模态段落的位置、网格和版本样例。 运行约定和合成输入。资料文件只描述输入对象、版本和边界，不包含算子实现。

## 环境

项目运行和资料检查仅依赖 Python 标准库。运行测试与打包验证前安装锁定的测试工具：

```bash
python -m pip install -e .
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m build --wheel --no-isolation
```

## 资料

`data/input.json` 是可重复使用的本地合成输入，`data/schema.json` 记录字段含义和允许的状态。包目录保留公开入口的命名空间；调用方可以在此基础上接入自己的批处理或推理流程。

## 算子

包内提供段落感知的旋转位置编码算子 `RopeSegmentOperator`，旋转相位只取决于逻辑坐标与模态频率参数，与张量物理下标无关：

- `plan(segments, packed_order)`：位置规划。物理下标按打包顺序分配；逻辑坐标只由段落自身的 `position_start`、`length`/`grid` 与模态模式决定。返回的 `PositionMap` 记录每个物理位置的 `(segment_id, local_offset, coord)`，并提供 `same_segment_mask()` 让注意力区分“同一段”与“仅拼接后物理相邻”。
- `prefill(...)`：预填充，整批旋转查询与键，可选写入缓存。
- `decode_incremental(...)`：增量解码。复用缓存前缀前逐段校验：规则版本、旋转布局、坐标元数据（起点/网格/频率参数）必须一致且覆盖完整，否则抛出 `CacheReuseError` 并附带结构化原因（`MissReason`），绝不混入旧结果。
- `chunked_recompute(...)`：分块重算。任意块长与预填充结果逐位一致，用于校验与缓存重建。

三条路径共享同一位置规划，对同一组段落给出一致的相位与位置映射；段落截断、重排、恢复都不会改变既有 token 的逻辑坐标。模态参数（`theta`、`freq_scale`、一维/二维网格）按 `frequency_version` 管理，内置 `v2`、`v3` 两个版本，也可用 `FrequencyRules` 自定义。

```python
from operator_rope_segments import (
    OperatorConfig, PrefixCache, RopeLayout, RopeSegmentOperator, Segment, builtin_rules,
)

config = OperatorConfig(rules=builtin_rules("v3"), layout=RopeLayout(head_dim=8, sections=(2, 1, 1)))
op = RopeSegmentOperator(config)
cache = PrefixCache()

text = Segment("text-0", "text", position_start=10, length=3)
image = Segment("image-0", "image", position_start=0, grid=(2, 2))

# 预填充文本前缀并写入缓存
op.prefill([text], ["text-0"], q_text, k_text, cache=cache)
# 增量解码：缓存前缀 + 新图像段，相位与一次性预填充完全一致
result = op.decode_incremental(
    [text, image], ["text-0", "image-0"],
    q_new=q_image, k_new=k_image,
    new_offsets={"image-0": (0, 4)}, cache=cache,
)
```
