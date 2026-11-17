"""Private, bounded candidate inference using the native attention contracts.

No serving runner, scheduler, request, block table or live KV is passed here.
All global contexts close before yielding back to the engine thread.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from time import monotonic

PREFILL_TOKENS = 4
MAX_KV_BYTES = 32 * 1024**2
DEADLINE_SECONDS = 30


@dataclass(frozen=True)
class CanaryPlan:
    name: str
    layer: object
    spec: object
    group: object
    scratch_bytes: int


def require_finite(tensor, stage):
    import torch
    if not isinstance(tensor, torch.Tensor) or not tensor.numel():
        raise ValueError(f"Candidate canary {stage}: missing tensor output")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"Candidate canary {stage}: non-finite output")


def plan_canary(config, allow_bound=False):
    """Check the native validation adapter before any component publication."""
    import torch
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes
    context = config.compilation_config.static_forward_context
    scratch_bytes = 0
    plans = []
    for name, layer in context.items():
        if type(layer) is MoERunner:
            continue  # expert ownership and checkpoint coverage checked separately
        if type(layer) is not Attention:
            raise ValueError("Missing canary state adapter: " + type(layer).__name__)
        backend = layer.get_attn_backend()
        if backend not in (FlashAttentionBackend, TritonAttentionBackend):
            raise ValueError("Candidate canary requires reviewed Flash/Triton attention")
        if (layer.kv_cache.numel() and not allow_bound) or layer.kv_sharing_target_layer_name:
            raise ValueError("Candidate canary requires unbound, unshared KV")
        spec = layer.get_kv_cache_spec(config)
        if not isinstance(spec, AttentionSpec) or spec.dtype is not torch.bfloat16:
            raise ValueError("Candidate canary requires BF16 attention KV")
        group = AttentionGroup(backend, [name], spec, 0)
        sizing = KVCacheConfig(2, [], [KVCacheGroupSpec([name], spec)])
        block_size = prepare_kernel_block_sizes(sizing, [[group]])[0]
        spec = spec.copy_with_new_block_size(block_size)
        if block_size < PREFILL_TOKENS + 1:
            raise ValueError("Candidate canary KV block is too small")
        group.kv_cache_spec = spec
        size = 2 * spec.page_size_bytes  # block zero remains unused
        scratch_bytes += size
        plans.append(CanaryPlan(name, layer, spec, group, size))
    if not plans or scratch_bytes > MAX_KV_BYTES:
        raise ValueError("Candidate canary exceeds the private KV budget")
    return plans


def canary_slices(config, model, allow_bound=False):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig, KVCacheGroupSpec, KVCacheTensor,
    )
    from vllm.v1.worker.utils import allocate_kv_cache

    started = monotonic()
    device = torch.device(config.device_config.device)
    context = config.compilation_config.static_forward_context
    saved, caches, groups = {}, {}, []
    scratch_bytes = 0

    @contextmanager
    def scope():
        with (torch.inference_mode(), set_current_vllm_config(config),
              set_default_torch_dtype(config.model_config.dtype), device):
            yield

    def check_deadline():
        if monotonic() - started > DEADLINE_SECONDS:
            raise TimeoutError("Candidate canary cooperative deadline exceeded")

    def restore_bindings():
        for name, original in saved.items():
            context[name].bind_kv_cache(original)

    @contextmanager
    def forward_scope():
        with scope():
            try:
                if allow_bound:
                    for name, cache in caches.items():
                        context[name].bind_kv_cache(cache)
                yield
            finally:
                if allow_bound:
                    restore_bindings()

    try:
        with scope():
            plans = plan_canary(config, allow_bound)
            scratch_bytes = sum(plan.scratch_bytes for plan in plans)
            for plan in plans:
                name = plan.name
                layer = plan.layer
                spec = plan.spec
                group = plan.group
                size = plan.scratch_bytes
                private = KVCacheConfig(2, [KVCacheTensor(
                    size, [name], size, spec.page_size_bytes,
                )], [KVCacheGroupSpec([name], spec)])
                cache = allocate_kv_cache(private, device,
                                         config.cache_config.get_resolved_kv_cache_layout(),
                                         [spec.block_size])[name]
                saved[name] = layer.kv_cache
                caches[name] = cache
                layer.bind_kv_cache(cache)
                group.create_metadata_builders(config, device, spec.block_size)
                groups.append((name, spec.block_size, group.get_metadata_builder()))
            torch.cuda.synchronize()
            if allow_bound:
                restore_bindings()
        check_deadline()
        yield {"kind": "canary_allocate", "scratch_kv_bytes": scratch_bytes}

        vocab_size = config.model_config.get_vocab_size()
        if vocab_size < 1 or config.model_config.max_model_len < PREFILL_TOKENS + 1:
            raise ValueError("Candidate canary needs a vocabulary and five context tokens")
        next_token = 0
        for stage, count, offset in (("prefill", PREFILL_TOKENS, 0),
                                     ("decode", 1, PREFILL_TOKENS)):
            check_deadline()
            with forward_scope():
                positions = torch.arange(offset, offset + count, device=device)
                ids = (torch.arange(count, device=device) % vocab_size if stage == "prefill"
                       else torch.tensor([next_token], dtype=torch.long, device=device))
                starts_cpu = torch.tensor([0, count], dtype=torch.int32, device="cpu")
                lengths_cpu = torch.tensor([offset + count], dtype=torch.int32, device="cpu")
                metadata, slots = {}, {}
                for name, block_size, builder in groups:
                    slot = positions + block_size
                    common = CommonAttentionMetadata(
                        query_start_loc=starts_cpu.to(device), query_start_loc_cpu=starts_cpu,
                        seq_lens=lengths_cpu.to(device), seq_lens_cpu_upper_bound=lengths_cpu,
                        num_reqs=1, num_actual_tokens=count, max_query_len=count,
                        max_seq_len=offset + count,
                        block_table_tensor=torch.tensor([[1]], dtype=torch.int32, device=device),
                        slot_mapping=slot, positions=positions,
                        is_prefilling=torch.tensor([stage == "prefill"], device=device),
                    )
                    metadata[name] = builder.build(0, common)
                    slots[name] = slot
                with set_forward_context(metadata, config, num_tokens=count,
                                         slot_mapping=slots):
                    hidden = model(input_ids=ids, positions=positions)
                    require_finite(hidden, stage + " hidden states")
                    logits = model.compute_logits(hidden[-1:])
                    require_finite(logits, stage + " logits")
                    next_token = int(logits.argmax(dim=-1).item())
                    if not 0 <= next_token < vocab_size:
                        raise ValueError("Candidate canary selected a token outside the vocabulary")
                torch.cuda.synchronize()
                del hidden, logits, metadata, slots, common, slot, ids, positions
            check_deadline()
            yield {"kind": "canary_" + stage, "tokens": count}
    finally:
        # Unbind only candidate-owned caches, including partial allocation failure.
        # Device-health synchronization belongs to the caller's recovery path.
        restore_bindings()
        caches.clear()
        groups.clear()
    yield {"kind": "canary_complete", "ready": True,
           "validation": {"prefill_tokens": PREFILL_TOKENS, "decode_tokens": 1,
                          "scratch_kv_bytes": scratch_bytes,
                          "elapsed_seconds": monotonic() - started}}
