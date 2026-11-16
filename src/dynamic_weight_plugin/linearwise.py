"""Native projection replacement without constructing a replacement decoder."""

from copy import copy
from time import monotonic_ns

from .layerwise import load_native_unit, retire_block
from .workspace import native_workspace_budget

PROJECTIONS = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
FORMATS = {"bf16", "fp8", "fp8_per_channel", "nf4"}


def unit_view(model, path, replacement):
    """Preserve native load_weights methods, exposing only the target leaf.

    Every ancestor gets separate registries. A shallow copy with its original
    parameter registry could otherwise expose serving tensors to native loading.
    """
    # A recursive nested function captures itself and `replacement`, keeping
    # the candidate alive in a closure cycle after this view has been dropped.
    # Build the ancestor chain iteratively so retirement needs no cyclic GC.
    result = view = copy(model)
    source = model
    parts = path.split(".")
    for index, name in enumerate(parts):
        view._parameters = {}
        view._buffers = {}
        if index == len(parts)-1:
            view._modules = {name: replacement}
        else:
            source = source._modules[name]
            child = copy(source)
            view._modules = {name: child}
            view = child
    return result


def constructor_args(old, quant_config):
    from vllm.model_executor.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
    args = dict(bias=old.has_bias, skip_bias_add=old.skip_bias_add,
                params_dtype=old.params_dtype, quant_config=quant_config,
                prefix=old.prefix, return_bias=old.return_bias, disable_tp=old.disable_tp)
    if type(old) is QKVParallelLinear:
        args.update(hidden_size=old.hidden_size, head_size=old.head_size,
                    total_num_heads=old.total_num_heads, total_num_kv_heads=old.total_num_kv_heads,
                    v_head_size=old.v_head_size)
    elif type(old) is MergedColumnParallelLinear:
        args.update(input_size=old.input_size, output_sizes=list(old.output_sizes), gather_output=old.gather_output)
    elif type(old) is RowParallelLinear:
        args.update(input_size=old.input_size, output_size=old.output_size,
                    input_is_parallel=old.input_is_parallel, reduce_results=old.reduce_results)
    else:
        raise ValueError("Unreviewed native linear class")
    return args


def plan_units(model, groups):
    """Validate all source shards and ownership before any destructive work."""
    from vllm.model_executor.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.layernorm import RMSNorm
    classes = (QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear, RowParallelLinear)
    shards = (("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"), ("down_proj",))
    units = []
    for index, entries in enumerate(groups):
        source = {entry[1]: entry for entry in entries}
        for suffix, cls, pieces in zip(PROJECTIONS, classes, shards):
            path = f"model.layers.{index}.{suffix}"
            module = model.get_submodule(path)
            if (type(module) is not cls or module.prefix != path or module.tp_size != 1
                    or list(module.children())):
                raise ValueError("Unreviewed projection layout: " + path)
            sizes = module.output_sizes if cls is not RowParallelLinear else [module.output_size]
            if len(sizes) != len(pieces) or sum(sizes) != module.output_size:
                raise ValueError("Unreviewed fused projection partitions: " + path)
            selected = []
            for piece, rows in zip(pieces, sizes):
                name = path.rsplit(".", 1)[0] + "." + piece
                for kind, shape in (("weight", (rows, module.input_size)), ("bias", (rows,))):
                    if kind == "bias" and not module.has_bias:
                        continue
                    key = name + "." + kind
                    entry = source.pop(key, None)
                    if entry is None or entry[3] != shape:
                        raise ValueError("Missing or incompatible projection tensor: " + key)
                    selected.append(entry)
            units.append({"path": path, "entries": selected})
        # These canonical norms never change precision. All other tensors must
        # be accounted for; do not silently skip an unfamiliar checkpoint field.
        for suffix in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            path = f"model.layers.{index}.{suffix}"
            norm = model.get_submodule(path)
            entry = source.pop(path + ".weight", None)
            if type(norm) is not RMSNorm or entry is None or entry[3] != tuple(norm.weight.shape):
                raise ValueError("Unreviewed retained norm: " + path)
        if source:
            raise ValueError("Unreviewed projection source tensors: " + str(sorted(source)))
    paths = {unit["path"] for unit in units}
    owners = {}
    for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        if tensor.is_meta or tensor.numel() == 0:
            continue
        prefix = name.rsplit(".", 1)[0]
        owner = prefix if prefix in paths else "retained"
        key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        if owners.setdefault(key, owner) != owner:
            raise ValueError("Cross-projection tensor storage alias: " + name)
    return units


def prepare_linear(backend, cfg, unit):
    import torch
    old = backend.runner.model.get_submodule(unit["path"])
    if 'kind' in unit:
        from .components import constructor_args as component_args
        policy = backend.native_quantization_policy
        quant_config = cfg.quant_config if policy.get(unit["path"], True) else None
        args = component_args(old, quant_config)
    else:
        args = constructor_args(old, cfg.quant_config)
    # Native startup scopes the default device to construction, then leaves it
    # before loading/quantization. Compiled quantizers must see the same scopes.
    with torch.device(cfg.device_config.device):
        new = type(old)(**args)
    backend.holder["unit"] = new
    view = unit_view(backend.runner.model, unit["path"], new)
    if 'kind' in unit:
        from .component_policy import bind_policy
        bind_policy(view, cfg.quant_config, backend.native_quantization_policy)
    with native_workspace_budget(cfg):
        load_native_unit(cfg, view, new, unit["entries"])
    return new


def replace_slices(backend, cfg, units, dirty=False):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from .canary import canary_slices
    model = backend.runner.model
    if cfg.quant_config is not None:
        configure_quant_config(cfg.quant_config, type(model))
    # Reviewed online FP8/NF4 recipes do not change BF16 attention or KV state.
    # Keep the live objects and addresses; never construct spare attention/KV.
    live = backend.runner.compilation_config.static_forward_context
    cfg.compilation_config.static_forward_context = dict(live)
    cfg.compilation_config.static_all_moe_layers = list(backend.runner.compilation_config.static_all_moe_layers)
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    for name, attention in live.items():
        if type(attention) is MoERunner:
            if not any(u['path'] == name + '.routed_experts' and u.get('kind') == 'experts' for u in units):
                raise ValueError('MoE runner has no complete replacement component')
            continue
        if (getattr(attention, "quant_method", None) is not None
                or (cfg.quant_config is not None and cfg.quant_config.get_quant_method(attention, name) is not None)
                or attention.get_kv_cache_spec(cfg) != attention.get_kv_cache_spec(backend.runner.vllm_config)):
            raise ValueError("Projection replacement would change attention/KV")
    for index, unit in enumerate(units):
        started = monotonic_ns()
        backend._check_source()
        with torch.inference_mode(), set_current_vllm_config(cfg), set_default_torch_dtype(cfg.model_config.dtype):
            new = prepare_linear(backend, cfg, unit)
            torch.cuda.synchronize()
            backend._check_source()
            # No forward may run between publication and synchronized retirement.
            backend.destructive_started = True
            from .components import publish_component
            old = publish_component(model, unit, new)
            retired = retire_block(old, set())
            del old
            if any(item() is not None for item in retired):
                raise RuntimeError("Retired projection still has module owners")
            backend.holder.pop("unit")
            torch.cuda.synchronize()
        yield {"kind": "replace_experts" if unit.get('kind') == 'experts' else "replace_linear", "index": index, "path": unit["path"],
               "elapsed_ns": monotonic_ns() - started,
               "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}
    if hasattr(backend, 'native_quantization_policy'):
        from .component_policy import bind_policy
        bind_policy(model, cfg.quant_config, backend.native_quantization_policy)
    else:
        for module in model.modules():
            if hasattr(module, 'quant_config'):
                module.quant_config = cfg.quant_config
    for attention in live.values():
        attention.quant_config = cfg.quant_config
    backend.holder["model"] = model
    backend.validate_candidate(cfg, model)
    for result in canary_slices(cfg, model, allow_bound=True):
        if result.get("ready"):
            backend.holder["validation"] = result["validation"]
        # The canary restores serving bindings and closes global scopes before
        # every yield, retaining only its private scratch KV between stages.
        yield result
