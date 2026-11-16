"""Destructive replacement of one native decoder block at a time.

No inference may run inside a slice. Dirty mode permits numerical generation
mixing only at the synchronized boundary after a complete block is installed.
"""

from copy import copy
from importlib import import_module
from math import prod
from pathlib import Path
from time import monotonic_ns
from weakref import ref

FORMATS = {"bf16", "fp8", "fp8_per_channel", "nf4"}
BLOCKS = {"qwen3": "Qwen3DecoderLayer", "qwen2": "Qwen2DecoderLayer",
          "llama": "LlamaDecoderLayer"}


def checkpoint_plan(source, count):
    """Index tensor metadata only; never materialize a full checkpoint."""
    from safetensors import safe_open
    groups = [[] for _ in range(count)]
    seen = set()
    allowed = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for path in sorted(Path(source).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in seen:
                    raise ValueError("Duplicate checkpoint tensor: " + name)
                seen.add(name)
                if name in allowed:
                    continue
                parts = name.split(".")
                if len(parts) < 4 or parts[:2] != ["model", "layers"] or not parts[2].isdigit():
                    raise ValueError("Unreviewed layerwise checkpoint tensor: " + name)
                index = int(parts[2])
                if index >= count:
                    raise ValueError("Checkpoint layer outside model")
                tensor = stream.get_slice(name)
                if tensor.get_dtype() not in {"F32", "F16", "BF16"}:
                    raise ValueError("Layerwise source must contain floating weights")
                shape = tuple(tensor.get_shape())
                size = prod(shape)
                groups[index].append((str(path), name, size, shape))
    if not all(groups):
        raise ValueError("Checkpoint is missing an entire decoder block")
    return groups


def read_block(entries):
    from safetensors import safe_open
    # A block may span shards. One tensor view is handed to the native loader
    # at a time; views remain valid while the loader consumes them.
    for filename in dict.fromkeys(entry[0] for entry in entries):
        with safe_open(filename, framework="pt", device="cpu") as stream:
            for path, name, _, _ in entries:
                if path == filename:
                    yield name, stream.get_tensor(name)


def target_bytes(entries, target):
    """Conservative storage estimate for the reviewed dense linear recipes."""
    result = 0
    for _, name, count, shape in entries:
        linear = len(shape) == 2 and name.endswith(".weight")
        if target == "bf16" or not linear:
            result += count * 2
        elif target in {"fp8", "fp8_per_channel"}:
            result += count + shape[0] * 4 + 1024
        else:
            # Native BNB stores packed nibbles and dequantized block absmax.
            result += (count + 1) // 2 + ((count + 63) // 64) * 4 + 2048
    return result


def block_view(model, index, block):
    """Native model loading methods with only this block exposed as children."""
    from torch import nn
    outer, inner = copy(model), copy(model.model)
    outer._modules = {"model": inner}
    inner._modules = {"layers": nn.ModuleDict({str(index): block})}
    return outer


def retire_block(block, retained):
    """Break tensor/loader cycles without collecting the whole Python heap.

    RoPE modules can be cached and shared. Never mutate a surviving module.
    Cross-block parameter aliases are refused by preflight.
    """
    from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO, LOADING_LAYERS
    retired = []
    for module in list(block.modules()):
        if id(module) in retained:
            continue
        retired.append(ref(module))
        info = LAYERWISE_INFO.pop(module, None)
        if info is not None:
            # Saved meta tensors can own loaders that close over this same
            # info object. Removing the weak-key entry alone leaves that cycle.
            info.reset()
            info.restore_metadata = ({}, {})
        LOADING_LAYERS.discard(module)
        module._parameters.clear()
        module._buffers.clear()
        # Native CustomOp stores its own bound forward method. Remove that
        # self-cycle only after the module is unreachable by serving code.
        module.__dict__.pop("_forward_method", None)
        module._modules.clear()
    return retired


def preflight(backend, target, dirty=False, linear=False):
    import torch
    model = backend.runner.model
    kind = backend.model_adapter.model_type
    if kind not in BLOCKS or target not in FORMATS:
        raise ValueError("Layerwise replacement requires reviewed Qwen3/Qwen2/Llama dense blocks and BF16/FP8/NF4")
    block_cls = getattr(import_module("vllm.model_executor.models." + kind), BLOCKS[kind])
    layers = model.model.layers
    if len(layers) != backend.canonical_model_config.hf_config.num_hidden_layers or any(type(b) is not block_cls for b in layers):
        raise ValueError("Exact native decoder blocks required")
    if dirty and backend.engine.vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("Dirty mode requires --no-enable-prefix-caching at startup")
    if dirty and not getattr(backend.runner, "_dynamic_quantization_activation_peak", None):
        raise ValueError("Dirty mode requires captured serving activation headroom")
    # Permit tied embeddings outside the replacement units, but no parameter
    # storage may be owned by both a block and another part of the model.
    owners = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        owner = ".".join(name.split(".")[:3]) if name.startswith("model.layers.") else "outside"
        key = (str(param.device), param.untyped_storage().data_ptr())
        prior = owners.setdefault(key, owner)
        if prior != owner:
            raise ValueError("Cross-block parameter aliases are unsupported")
    for name, module in model.named_modules():
        if name.startswith("model.layers."):
            continue
        method = getattr(module, "quant_method", None)
        if method is not None and type(method).__name__ != "UnquantizedEmbeddingMethod":
            raise ValueError("Quantized weights outside decoder blocks are unsupported")
    backend._check_source()
    groups = checkpoint_plan(backend.source, len(layers))
    units = None
    if linear:
        from .linearwise import plan_units
        units = plan_units(model, groups)
    from .control import MAX_PREPARATION_SLICES
    if dirty and len(units if linear else groups) + (4 if linear else 1) > MAX_PREPARATION_SLICES:
        raise ValueError("Replacement exceeds the bounded preparation slice count")
    largest_block = max(sum(n * 4 for _, _, n, _ in group) for group in groups)
    largest = max(sum(n * 4 for _, _, n, _ in unit["entries"]) for unit in units) if linear else largest_block
    target_upper = sum(target_bytes(group, target) for group in groups)
    resident = sum(p.numel() * p.element_size() for b in layers for p in b.parameters())
    growth = max(0, target_upper - resident)
    reserve = (backend.runner._dynamic_quantization_activation_peak + 256 * 1024**2) if dirty else 0
    free, _ = torch.cuda.mem_get_info()
    reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    # Estimated destination growth plus one FP32 replacement unit and
    # native workspaces. This is admission, not an allocator-enforced hard cap.
    required = growth + largest * 2 + 128 * 1024**2
    available = free - reserve if dirty else free + reusable
    if available < required:
        raise ValueError(f"Insufficient layerwise headroom: need {required}, available {available}")
    backend.memory_admission = {"mode": "dirty" if dirty else "drain", "strategy": "native_linears" if linear else "decoder_blocks", "required_bytes": required,
                                "available_bytes": available, "largest_source_block_bytes": largest_block,
                                "largest_source_unit_bytes": largest,
                                "estimated_target_block_bytes": target_upper,
                                "serving_reserve_bytes": reserve}
    return units if linear else groups


def prepare_block(backend, cfg, index, entries):
    kind = backend.model_adapter.model_type
    cls = getattr(import_module("vllm.model_executor.models." + kind), BLOCKS[kind])
    kwargs = {"prefix": f"model.layers.{index}"}
    if kind == "llama":
        kwargs["vllm_config"] = cfg
    else:
        kwargs.update(config=cfg.model_config.hf_config, cache_config=cfg.cache_config,
                      quant_config=cfg.quant_config)
    block = cls(**kwargs)
    backend.holder["block"] = block
    view = block_view(backend.runner.model, index, block)
    load_native_unit(cfg, view, block, entries)
    return block


def load_native_unit(cfg, view, unit, entries):
    import torch
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading
    from .incremental import initialize_online_layers
    with torch.device(cfg.device_config.device):
        record_metadata_for_reloading(unit)
        initialize_online_layers(unit)
    if cfg.model_config.quantization == "bitsandbytes":
        loader = get_model_loader(cfg.load_config)
        # Instance-local iterator only. Quantization, fusion, packing and
        # auxiliary state binding all remain the native BitsAndBytes path.
        def iterator(files, safetensors):
            for name, tensor in read_block(entries):
                yield name, loader.weight_mapper(name), tensor
        loader._hf_weight_iter = iterator
        loader.load_weights(view, cfg.model_config)
    else:
        names = {name for name, _ in view.named_parameters()}
        loaded = view.load_weights(read_block(entries))
        missing = names - loaded
        missing = {n for n in missing if not n.endswith(("._k_scale", "._v_scale", "._q_scale", "._prob_scale"))}
        if missing:
            raise ValueError(f"Incomplete replacement unit: {sorted(missing)}")
        from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
        for layer in unit.modules():
            if getattr(getattr(layer, "quant_method", None), "uses_meta_device", False) and get_layerwise_info(layer).can_load():
                raise ValueError("Incomplete fused quantization layer")
    process_weights_after_loading(unit, cfg.model_config, torch.device(cfg.device_config.device))
    if any(p.is_meta for p in unit.parameters()):
        raise ValueError("Unmaterialized replacement unit")
    unit.eval()


def replace_slices(backend, cfg, groups, dirty=False):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.model_executor.models.interfaces import SupportsQuant
    model = backend.runner.model
    if cfg.quant_config is not None:
        configure_quant_config(cfg.quant_config, type(model))
        # Same SupportsQuant hook used by native top-level construction; the
        # temporary view owns only the new configuration, not new tensors.
        if isinstance(model, SupportsQuant):
            mapping_view = copy(model)
            mapping_view.quant_config = cfg.quant_config
            mapping_view._maybe_apply_model_mapping()
            del mapping_view
    for index, entries in enumerate(groups):
        started = monotonic_ns()
        backend._check_source()
        with torch.inference_mode(), set_current_vllm_config(cfg), set_default_torch_dtype(cfg.model_config.dtype), torch.device(cfg.device_config.device):
            block = prepare_block(backend, cfg, index, entries)
            torch.cuda.synchronize()
            backend._check_source()
            prefix = f"model.layers.{index}."
            names = [name for name in cfg.compilation_config.static_forward_context if name.startswith(prefix)]
            live = backend.runner.compilation_config.static_forward_context
            old_names = [name for name in live if name.startswith(prefix)]
            if len(names) != 1 or set(names) != set(old_names):
                raise ValueError("Replacement block changes attention registry")
            for name in names:
                old_attn, new_attn = live[name], cfg.compilation_config.static_forward_context[name]
                if (type(old_attn) is not type(new_attn) or old_attn.get_attn_backend() is not new_attn.get_attn_backend()
                        or old_attn.get_kv_cache_spec(backend.runner.vllm_config) != new_attn.get_kv_cache_spec(cfg)):
                    raise ValueError("Replacement block changes KV/attention contract")
            # From this point onward rollback is prohibited, including if the
            # assignment or retirement itself throws.
            backend.destructive_started = True
            old = model.model.layers[index]
            model.model.layers[index] = block
            for name in names:
                new_attn = cfg.compilation_config.static_forward_context[name]
                new_attn.bind_kv_cache(live[name].kv_cache)
                live[name] = new_attn
            del old_attn, new_attn
            retained = {id(m) for m in model.modules()}
            retired = retire_block(old, retained)
            del old
            # A new ownership pattern must fail closed instead of silently
            # retaining old generations or requiring a whole-heap GC scan.
            if any(item() is not None for item in retired):
                raise RuntimeError("Retired decoder block still has module owners")
            backend.holder.pop("block")
            torch.cuda.synchronize()
        yield {"kind": "replace_block", "index": index, "elapsed_ns": monotonic_ns() - started,
               "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}
    if hasattr(model, "quant_config"):
        model.quant_config = cfg.quant_config
    if hasattr(model.model, "quant_config"):
        model.model.quant_config = cfg.quant_config
    backend.holder["model"] = model
    backend.validate_candidate(cfg, model)
    from .canary import canary_slices
    # In dirty mode the canary borrows bindings only within this uninterrupted
    # slice and restores every binding before another inference step runs.
    for result in canary_slices(cfg, model, allow_bound=True):
        if result.get("ready"):
            backend.holder["validation"] = result["validation"]
    yield {"kind": "layerwise_complete", "ready": True}
