"""Bound native conversion workspace during an owned synchronous loading step."""

from contextlib import contextmanager

# The pinned native FP8 helper uses a bounded conversion chunk. A smaller row
# chunk bounds intermediates without changing native scales or arithmetic.
# The unavoidable source and destination tensors remain a complete native unit.
NATIVE_FP8_CHUNK_ELEMS = 16 * 1024 * 1024
FP8_CHUNK_ELEMS = 1024 * 1024


@contextmanager
def native_workspace_budget(config):
    """Scope the native chunk budget; never hold it across an engine yield.

    The supported runtime has one synchronous loading owner in EngineCore. This
    is not safe for concurrent loaders and does not enable them. Refuse a nested
    or conflicting override rather than changing somebody else's budget. No
    allocator, tensor ownership, quantization kernel or installed file changes.
    """
    if config.model_config.quantization != "online":
        yield
        return
    from vllm.model_executor.layers.quantization.online import fp8
    original = fp8._QUANT_CHUNK_ELEMS
    if original != NATIVE_FP8_CHUNK_ELEMS:
        raise RuntimeError("Unexpected native FP8 workspace budget; concurrent or conflicting override")
    fp8._QUANT_CHUNK_ELEMS = FP8_CHUNK_ELEMS
    try:
        yield
    finally:
        changed = fp8._QUANT_CHUNK_ELEMS != FP8_CHUNK_ELEMS
        fp8._QUANT_CHUNK_ELEMS = original
        if changed:
            raise RuntimeError("Native FP8 workspace budget changed during owned loading")
