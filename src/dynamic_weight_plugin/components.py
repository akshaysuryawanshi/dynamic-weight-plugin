"""Plan replacement from native modules and native loading, without GPU weights.

The planning view contains only meta tensors. Source tensors are also meta:
native name mapping, transposes and fused shard loaders execute normally, but
no checkpoint values are read and no target representation is cached.
"""

from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from math import prod
from pathlib import Path


def linear_types():
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
        ReplicatedLinear, RowParallelLinear,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    return (ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
            ReplicatedLinear, RowParallelLinear, GateLinear)


def checkpoint_entries(source):
    from safetensors import safe_open
    entries, seen = [], set()
    for path in sorted(Path(source).glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as stream:
            for name in stream.keys():
                if name in seen:
                    raise ValueError('Duplicate checkpoint tensor: ' + name)
                seen.add(name)
                tensor = stream.get_slice(name)
                if tensor.get_dtype() not in {'F32', 'F16', 'BF16'}:
                    raise ValueError('Component source requires floating weights: ' + name)
                shape = tuple(tensor.get_shape())
                entries.append((str(path), name, prod(shape), shape))
    if not entries:
        raise ValueError('Empty canonical checkpoint')
    return entries


def constructor_args(old, quant_config):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    if type(old) is GateLinear:
        import torch
        if old.params_dtype is not torch.bfloat16 or old.out_dtype is not None:
            raise ValueError('Unreviewed router gate dtype')
        return dict(input_size=old.input_size, output_size=old.output_size,
                    bias=old.has_bias, params_dtype=old.params_dtype,
                    quant_config=quant_config, prefix=old.prefix,
                    skip_bias_add=old.skip_bias_add, return_bias=old.return_bias,
                    out_dtype=old.out_dtype)
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    if type(old) is RoutedExperts:
        from .expert_components import constructor_args as expert_args
        return expert_args(old, quant_config)
    from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
    from .linearwise import constructor_args as fused_args
    if type(old) not in (ColumnParallelLinear, ReplicatedLinear):
        return fused_args(old, quant_config)
    args = dict(input_size=old.input_size, output_size=old.output_size,
                bias=old.has_bias, skip_bias_add=old.skip_bias_add,
                params_dtype=old.params_dtype, quant_config=quant_config,
                prefix=old.prefix, return_bias=old.return_bias, disable_tp=old.disable_tp)
    if type(old) is ColumnParallelLinear:
        args['gather_output'] = old.gather_output
    return args


def registered_aliases(model, paths):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    identities = {id(model.get_submodule(path)): path for path in paths}
    aliases = {}
    for name, module in model.named_modules(remove_duplicate=False):
        canonical = identities.get(id(module))
        if canonical is not None and name != canonical:
            parent, _, leaf = name.rpartition('.')
            runner = model.get_submodule(parent)
            if (type(module) is not GateLinear or type(runner) is not MoERunner
                    or leaf != 'gate' or runner.shared_experts is not None
                    or runner.shared_expert_gate is not None):
                raise ValueError('Shared replacement module: ' + name)
            aliases[name] = canonical
    return aliases


def validate_owners(model, paths):
    """Reject aliases across replaceable units, including unregistered owners."""
    import torch
    from types import MethodType
    owners, modules = {}, {}
    aliases = registered_aliases(model, paths)
    def owner_of(path):
        for alias, canonical in aliases.items():
            if path == alias or path.startswith(alias + '.'):
                path = canonical + path[len(alias):]
                break
        return next((p for p in paths if path == p or path.startswith(p + '.')), 'retained')
    for path, module in model.named_modules(remove_duplicate=False):
        owner = owner_of(path)
        if owner != 'retained':
            if id(module) in modules and not any(path == a or path.startswith(a+'.') for a in aliases):
                raise ValueError('Shared replacement module: ' + path)
            modules[id(module)] = owner
    owned_tensors = list(model.named_parameters(remove_duplicate=False))
    owned_tensors += list(model.named_buffers(remove_duplicate=False))
    for name, tensor in owned_tensors:
        if tensor.is_meta or tensor.numel() == 0:
            continue
        owner = name.rsplit('.', 1)[0]
        owner = owner_of(owner)
        key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        if owners.setdefault(key, owner) != owner:
            raise ValueError('Cross-component tensor storage alias: ' + name)
    # Extra Python references to replacement modules cannot survive retirement.
    # Registered _modules are handled above; ordinary containers must be reviewed.
    def check(value, origin, owner, seen):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            if id(value) in modules:
                raise ValueError('Unregistered replacement owner: ' + origin)
        elif isinstance(value, torch.Tensor) and not value.is_meta and value.numel():
            storage_owner = owners.get((str(value.device), value.untyped_storage().data_ptr()))
            if storage_owner not in (None, 'retained', owner):
                raise ValueError('Unregistered replacement storage alias: ' + origin)
        elif isinstance(value, MethodType):
            if id(value.__self__) in modules and modules[id(value.__self__)] != owner:
                raise ValueError('Unregistered replacement method owner: ' + origin)
        elif isinstance(value, (tuple, list, dict)):
            for item in (value.values() if isinstance(value, dict) else value):
                check(item, origin, owner, seen)
    for name, module in model.named_modules():
        for key, value in vars(module).items():
            if key not in {'_modules', '_parameters', '_buffers'}:
                check(value, name + '.' + key, owner_of(name), set())


def publish_component(model, unit, new):
    """Engine-boundary publication of one unit and its reviewed aliases."""
    old = model.get_submodule(unit['path'])
    owners = []
    for path in [unit['path'], *unit.get('aliases', ())]:
        parent_path, _, name = path.rpartition('.')
        parent = model.get_submodule(parent_path)
        if parent._modules[name] is not old:
            raise RuntimeError('Replacement alias owner changed')
        owners.append((parent, name))
    for parent, name in owners:
        parent._modules[name] = new
    if unit.get('kind') == 'experts':
        owners[0][0].moe_config = new.moe_config
    return old


def plan_units(model, entries):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    class MetaOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if any(isinstance(t, torch.Tensor) and not t.is_meta
                   for t in tree_leaves((args, kwargs))):
                raise ValueError('Planning loader attempted to access real tensor storage')
            if 'device' in kwargs and kwargs['device'] is not None and torch.device(kwargs['device']).type != 'meta':
                raise ValueError('Planning loader attempted a real tensor allocation')
            return func(*args, **kwargs)
    with MetaOnly(), torch.device('meta'):
        return _plan_units(model, entries)


@dataclass
class _LoadTrace:
    current_source: str | None = None
    destinations: dict = field(default_factory=lambda: defaultdict(set))
    loaded_elements: dict = field(default_factory=lambda: defaultdict(int))
    events: set = field(default_factory=set)


def _component_paths(model):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase

    supported = linear_types()
    paths = {}
    for name, module in model.named_modules():
        if type(module) in supported:
            if getattr(module, 'tp_size', 1) != 1 or list(module.children()):
                raise ValueError('Unsupported native component ownership: ' + name)
            paths[name] = module
        elif type(module) is RoutedExperts:
            from .expert_components import validate_experts

            validate_experts(model, name, module)
            paths[name] = module
        elif isinstance(module, LinearBase):
            raise ValueError(
                'Unreviewed native linear component: ' + type(module).__name__
            )
        elif getattr(module, 'quant_method', None) is not None:
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                VocabParallelEmbedding,
            )

            if not isinstance(module, VocabParallelEmbedding):
                raise ValueError(
                    'Unreviewed quantized component: ' + type(module).__name__
                )
    if not paths:
        raise ValueError('No replaceable native components')
    validate_owners(model, paths)
    return paths


def _empty_clone(module):
    clone = copy(module)
    clone._parameters = {}
    clone._buffers = {}
    clone._modules = {}
    clone._non_persistent_buffers_set = set(module._non_persistent_buffers_set)
    return clone


def _populate_planning_view(model, view, paths):
    """Clone retained structure and construct replaceable meta components."""
    import torch

    probes = {}
    queue = [(model, view, '')]
    while queue:
        old, new, prefix = queue.pop()
        if hasattr(new, 'quant_config'):
            new.quant_config = None
        for kind in ('_parameters', '_buffers'):
            for name, tensor in getattr(old, kind).items():
                if tensor is None:
                    continue
                value = torch.empty(
                    tuple(tensor.shape), dtype=tensor.dtype, device='meta'
                )
                if kind == '_parameters':
                    value = torch.nn.Parameter(value, requires_grad=False)
                getattr(new, kind)[name] = value
        for name, child in old._modules.items():
            if child is None:
                new._modules[name] = None
                continue
            path = prefix + '.' + name if prefix else name
            if path in paths:
                with torch.device('meta'):
                    probe = type(child)(**constructor_args(child, None))
                new._modules[name] = probe
                probes[path] = probe
            else:
                clone = _empty_clone(child)
                new._modules[name] = clone
                queue.append((child, clone, path))
    return probes


def _plan_units(model, entries):
    import torch
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.model_loader.reload.meta import CopyCounter
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from .layerwise import retire_block

    paths = _component_paths(model)
    # Clone each registered path separately: meta aliases are deliberately not
    # tied, allowing per-source tracing without a skipped-alias completeness
    # check intended for an entire checkpoint. Live aliases were checked above.
    view = _empty_clone(model)
    trace = _LoadTrace()
    try:
        probes = _populate_planning_view(model, view, paths)

        for name, param in view.named_parameters():
            path = name.rsplit('.', 1)[0]
            native = getattr(param, 'weight_loader', default_weight_loader)
            if path not in probes:
                # Retained tensors are only mapped, never reloaded by ALTER.
                def retained(param, data, *args, _name=name, **kwargs):
                    trace.destinations[trace.current_source].add(('retained', _name))
                param.weight_loader = retained
                continue

            def recorded(
                param,
                data=None,
                *args,
                _name=name,
                _path=path,
                _native=native,
                **kwargs,
            ):
                keyword_data = data is None and 'loaded_weight' in kwargs
                if keyword_data:
                    data = kwargs['loaded_weight']
                shard = getattr(data, 'shard_id', args[0] if args else None)
                key = (_name, repr(shard), repr(kwargs))
                if key in trace.events:
                    raise ValueError('Duplicate native destination shard: ' + _name)
                trace.events.add(key)
                with CopyCounter() as counter:
                    result = (
                        _native(param, *args, **kwargs)
                        if keyword_data
                        else _native(param, data, *args, **kwargs)
                    )
                if counter.copied_numel <= 0:
                    raise ValueError('Native component loader did not copy: ' + _name)
                trace.loaded_elements[_name] += counter.copied_numel
                trace.destinations[trace.current_source].add((_path, _name))
                return result

            param.weight_loader = recorded
        for entry in entries:
            trace.current_source = entry[1]
            source = torch.empty(entry[3], dtype=torch.bfloat16, device='meta')
            view.load_weights(iter([(entry[1], source)]))
            if not trace.destinations[entry[1]]:
                # Native loaders discard derived rotary tables; no other
                # ignored checkpoint fields are accepted without review.
                if not entry[1].endswith(
                    (
                        'rotary_emb.inv_freq',
                        'rotary_emb.cos_cached',
                        'rotary_emb.sin_cached',
                    )
                ):
                    raise ValueError('Unaccounted canonical tensor: ' + entry[1])
        for path, probe in probes.items():
            for name, param in probe.named_parameters():
                key = path + '.' + name
                if trace.loaded_elements[key] != param.numel():
                    raise ValueError('Incomplete native component: ' + key)
        result = [
            {
                'path': path,
                'entries': [],
                'kind': 'experts' if type(probe) is RoutedExperts else 'linear',
                'logical_bytes': sum(p.numel() * 2 for p in probe.parameters()),
            }
            for path, probe in probes.items()
        ]
        by_path = {unit['path']: unit for unit in result}
        aliases = registered_aliases(model, paths)
        for unit in result:
            unit['aliases'] = [
                alias for alias, path in aliases.items() if path == unit['path']
            ]
        for entry in entries:
            owners = {path for path, _ in trace.destinations[entry[1]]}
            if len(owners) > 1:
                raise ValueError('Canonical tensor spans replacement owners: ' + entry[1])
            if owners and next(iter(owners)) != 'retained':
                by_path[next(iter(owners))]['entries'].append(entry)
        return result
    finally:
        # Remove native bound-loader cycles, including the recording closures.
        retire_block(view, set())


def preflight(backend, target, dirty=False):
    import torch
    from .control import MAX_PREPARATION_SLICES
    from .layerwise import target_bytes

    backend._check_source()
    if not hasattr(backend, 'component_plan'):
        from vllm.config import set_current_vllm_config

        with torch.inference_mode(), set_current_vllm_config(backend.runner.vllm_config):
            backend.component_plan = plan_units(
                backend.runner.model, checkpoint_entries(backend.source)
            )
    units = backend.component_plan
    validate_owners(backend.runner.model, {unit['path'] for unit in units})
    if dirty and backend.engine.vllm_config.cache_config.enable_prefix_caching:
        raise ValueError('Dirty mode requires disabled prefix caching')
    activation = getattr(backend.runner, '_dynamic_quantization_activation_peak', None)
    if dirty and activation is None:
        raise ValueError('Dirty mode requires captured serving activation headroom')
    if len(units) + 4 > MAX_PREPARATION_SLICES:
        raise ValueError('Replacement exceeds bounded slice count')
    largest = max(sum(e[2]*4 for e in unit['entries']) for unit in units)
    # The bound is deliberately conservative for newly introduced formats.
    estimated_sizes = (
        target_bytes(unit['entries'], target)
        if target in {'bf16', 'fp8', 'fp8_per_channel', 'nf4'}
        else unit['logical_bytes']
        for unit in units
    )
    target_upper = sum(estimated_sizes)
    resident = sum(
        parameter.numel() * parameter.element_size()
        for unit in units
        for parameter in backend.runner.model.get_submodule(
            unit['path']
        ).parameters()
    )
    reserve = activation + 256 * 1024**2 if dirty else 0
    free, _ = torch.cuda.mem_get_info()
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    # Ada's native MXFP8 Torch path simultaneously retains BF16 input, two
    # FP32 element arrays, FP8 output and block-scale vectors. Reserve 12
    # bytes/element (3 * the FP32 source bound), plus the fixed margin below.
    # The generic 8-byte allowance underestimates this path at realistic widths.
    workspace = largest * (3 if target == 'mxfp8' else 2)
    required = max(0, target_upper - resident) + workspace + 128 * 1024**2
    available = free - reserve if dirty else free + cached
    if available < required:
        raise ValueError(
            f'Insufficient component headroom: need {required}, available {available}'
        )
    backend.serving_reserve_bytes = reserve
    return units
