"""Cooperative native loading, with global contexts closed before every yield."""

from time import monotonic_ns
from .models import model_adapter


def initialize_online_layers(model):
    from vllm.model_executor.model_loader.reload.layerwise import initialize_online_processing
    # LinearBase registers bias after quant_method.create_weights(). A checkpoint
    # can present a bias before its first weight, so wrap late parameters once
    # construction is complete and BEFORE consuming any tensors. Otherwise the
    # first bias bypasses native load accounting and the layer stays unfinished.
    for layer in model.modules():
        if getattr(getattr(layer, "quant_method", None), "uses_meta_device", False):
            initialize_online_processing(layer)


def canonical_loaded_names(model, loaded):
    """Native MoERunner delegates loading but omits its routed child prefix."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    canonical = set(loaded)
    for prefix, layer in model.named_modules():
        if type(layer) is MoERunner:
            for name, _ in layer.routed_experts.named_parameters(recurse=False):
                reported = f"{prefix}.{name}"
                if reported in loaded:
                    canonical.add(f"{prefix}.routed_experts.{name}")
    return canonical


def load_slices(config, holder, slice_ms=2.0):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.base_loader import _has_online_quant
    from vllm.model_executor.model_loader.reload import finalize_layerwise_processing
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.utils.torch_utils import set_default_torch_dtype

    loader = get_model_loader(config.load_config)
    mc = config.model_config
    adapter = model_adapter(mc.hf_config)
    device = torch.device(config.device_config.device)
    with torch.inference_mode(), set_current_vllm_config(config), set_default_torch_dtype(mc.dtype):
        with device:
            model = loader.create_model(config, mc)
            holder["model"] = model
        adapter.check_model(model)
        initialize_online_layers(model)
        # Retain the native top-level loader across slices for tied aliases;
        # model-specific preprocessing happens before each delegated load.
        weights_loader = AutoWeightsLoader(model)
        weights_loader._loaded_params_are_complete = False
        original_names = {name for name, _ in model.named_parameters()}
        if mc.quantization == "bitsandbytes":
            loader._verify_model_compatibility(model, mc)
            loader._initialize_loader_state(model, mc)
            weights, quant_states = loader._get_quantized_weights_iterator(mc.model, mc.revision)
        else:
            loader._init_ep_weight_filter(mc)
            weights = loader.get_all_weights(mc, model)
            quant_states = None
        holder["iterator"] = weights
        torch.cuda.synchronize()
    yield {"kind": "construct", "tensors": 0}

    loaded = set()
    exhausted = False
    while not exhausted:
        count = 0
        start = monotonic_ns()
        with torch.inference_mode(), set_current_vllm_config(config), set_default_torch_dtype(mc.dtype):
            while count < 4:
                try:
                    item = next(weights)
                except StopIteration:
                    exhausted = True
                    break
                loaded.update(adapter.load_items(weights_loader, model, [item]))
                del item
                count += 1
                if monotonic_ns() - start >= slice_ms * 1_000_000:
                    break
            torch.cuda.synchronize()
        if count:
            yield {"kind": "load", "tensors": count}

    with torch.inference_mode(), set_current_vllm_config(config), set_default_torch_dtype(mc.dtype):
        weights_loader._loaded_params_are_complete = True
        weights_loader._check_skipped_aliases(loaded)
        missing = original_names - canonical_loaded_names(model, loaded)
        # Attention scale sentinels need no checkpoint entries for BF16 KV.
        missing = {n for n in missing if not n.endswith(("._k_scale", "._v_scale", "._q_scale", "._prob_scale"))}
        if missing:
            raise ValueError(f"Incomplete candidate weights: {sorted(missing)}")
        if quant_states is not None:
            experts = loader._fuse_moe_quant_states(model, quant_states)
            stacked = loader._stack_quantization_states(model, quant_states)
            loader._bind_quant_states_to_params(model, {**experts, **stacked})
        if _has_online_quant(model):
            # Reviewed models use no padded quantized linears. A layer waiting for
            # fused shards is incomplete; do not quantize uninitialized storage.
            from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
            for name, layer in model.named_modules():
                if getattr(getattr(layer, "quant_method", None), "uses_meta_device", False):
                    info = get_layerwise_info(layer)
                    if info.can_load():
                        raise ValueError(f"Incomplete fused quantization layer: {name}")
            finalize_layerwise_processing(model, mc)
        process_weights_after_loading(model, mc, device)
        model.eval()
        torch.cuda.synchronize()
    yield {"kind": "finalize", "tensors": 0, "ready": True}
