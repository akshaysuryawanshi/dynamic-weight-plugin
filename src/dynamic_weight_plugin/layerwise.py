"""Native loading and retirement shared by component replacement units."""

from weakref import ref


def read_unit(entries):
    """Yield one tensor at a time while each safetensors shard is open."""
    from safetensors import safe_open

    for filename in dict.fromkeys(entry[0] for entry in entries):
        with safe_open(filename, framework="pt", device="cpu") as stream:
            for path, name, _, _ in entries:
                if path == filename:
                    yield name, stream.get_tensor(name)


def target_bytes(entries, target):
    """Conservative storage estimate for the admitted component recipes."""
    result = 0
    for _, name, count, shape in entries:
        linear = len(shape) == 2 and name.endswith(".weight")
        if target == "bf16" or not linear:
            result += count * 2
        elif target in {"fp8", "fp8_per_channel"}:
            result += count + shape[0] * 4 + 1024
        else:
            result += (count + 1) // 2 + ((count + 63) // 64) * 4 + 2048
    return result


def retire_block(block, retained):
    """Break tensor and loader cycles without collecting the serving heap."""
    from vllm.model_executor.model_loader.reload.layerwise import (
        LAYERWISE_INFO,
        LOADING_LAYERS,
    )

    retired = []
    for module in list(block.modules()):
        if id(module) in retained:
            continue
        retired.append(ref(module))
        info = LAYERWISE_INFO.pop(module, None)
        if info is not None:
            info.reset()
            info.restore_metadata = ({}, {})
        LOADING_LAYERS.discard(module)
        module._parameters.clear()
        module._buffers.clear()
        module.__dict__.pop("_forward_method", None)
        module._modules.clear()
    return retired


def load_native_unit(cfg, view, unit, entries):
    import torch
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.reload.layerwise import (
        record_metadata_for_reloading,
    )
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    from .incremental import initialize_online_layers

    with torch.device(cfg.device_config.device):
        record_metadata_for_reloading(unit)
        initialize_online_layers(unit)

    if cfg.model_config.quantization == "bitsandbytes":
        loader = get_model_loader(cfg.load_config)

        def iterator(files, safetensors):
            for name, tensor in read_unit(entries):
                yield name, loader.weight_mapper(name), tensor

        loader._hf_weight_iter = iterator
        loader.load_weights(view, cfg.model_config)
    else:
        names = {name for name, _ in view.named_parameters()}
        loaded = view.load_weights(read_unit(entries))
        missing = names - loaded
        missing = {
            name
            for name in missing
            if not name.endswith(("._k_scale", "._v_scale", "._q_scale", "._prob_scale"))
        }
        if missing:
            raise ValueError(f"Incomplete replacement unit: {sorted(missing)}")

        from vllm.model_executor.model_loader.reload.layerwise import (
            get_layerwise_info,
        )

        for layer in unit.modules():
            quant_method = getattr(layer, "quant_method", None)
            if (
                getattr(quant_method, "uses_meta_device", False)
                and get_layerwise_info(layer).can_load()
            ):
                raise ValueError("Incomplete fused quantization layer")

    process_weights_after_loading(
        unit, cfg.model_config, torch.device(cfg.device_config.device)
    )
    if any(parameter.is_meta for parameter in unit.parameters()):
        raise ValueError("Unmaterialized replacement unit")
    unit.eval()
