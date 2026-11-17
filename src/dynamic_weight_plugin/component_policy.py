"""Derive per-component quantization intent from the native model constructor.

BF16 startup cannot tell an intentionally unquantized router from a router that
accepts the model's (currently None) quantization config. A private meta model
answers that question without reading weights or constructing another GPU model.
"""
from copy import deepcopy
import sys
from unittest.mock import patch


def inspect_policy(live, prototype, quant_config, units):
    paths = {u['path'] for u in units}
    for path in paths:
        if type(prototype.get_submodule(path)) is not type(live.get_submodule(path)):
            raise ValueError('Native target changes component class: '+path)
    policy = {}
    for name, module in prototype.named_modules():
        if hasattr(module, 'quant_config'):
            value = module.quant_config
            if value is not None and value is not quant_config:
                raise ValueError('Native target uses a distinct component quantization config: '+name)
            policy[name] = value is not None
        replaced = any(name == p or name.startswith(p+'.') for p in paths)
        if not replaced:
            old = live.get_submodule(name)
            if type(getattr(module,'quant_method',None)) is not type(getattr(old,'quant_method',None)):
                raise ValueError('Native target would change a retained quantization method: '+name)
    return policy


def initialize_constructor_runtime(cfg):
    """Initialize verified optional code before the meta-only constructor."""
    if cfg.model_config.quantization == 'bitsandbytes':
        from .compat import verify_bitsandbytes_runtime
        verify_bitsandbytes_runtime()
        if 'bitsandbytes' not in sys.modules:
            raise RuntimeError('Verified bitsandbytes runtime was not initialized')


def native_policy(backend, cfg, units):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.model_executor.layers import rotary_embedding
    from .layerwise import retire_block
    # bitsandbytes creates process-global CPU NF4/FP4 lookup tables on first import.
    # Import the already fingerprinted dependency before MetaOnly so that the
    # constructor guard continues to reject every real allocation it observes.
    initialize_constructor_runtime(cfg)
    class MetaOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if any(isinstance(t,torch.Tensor) and not t.is_meta for t in tree_leaves((args,kwargs))):
                raise ValueError('Native constructor touched real tensor storage')
            if kwargs.get('device') is not None and torch.device(kwargs['device']).type != 'meta':
                raise ValueError('Native constructor requested real tensor allocation')
            return func(*args, **kwargs)
    probe_cfg = deepcopy(cfg)  # candidate config owns no live forward modules
    prototype = None
    retained = {id(m) for m in backend.runner.model.modules()}
    # RoPE's native cache is process-global and omits device from its key. Keep
    # this engine-thread-only metadata construction out of the serving cache.
    try:
        with (torch.inference_mode(), MetaOnly(), torch.device('meta'),
              set_default_torch_dtype(cfg.model_config.dtype),
              patch.object(rotary_embedding, '_ROPE_DICT', {})):
            prototype = initialize_model(probe_cfg, model_class=type(backend.runner.model))
            if any(not p.is_meta for p in prototype.parameters()):
                raise ValueError('Native constructor retained real parameters')
            return inspect_policy(backend.runner.model,prototype,probe_cfg.quant_config,units)
    finally:
        if prototype is not None:
            retire_block(prototype, retained)
        probe_cfg.compilation_config.static_forward_context.clear()
        probe_cfg.compilation_config.static_all_moe_layers.clear()


def bind_policy(model, quant_config, policy):
    for name, module in model.named_modules():
        if name in policy:
            module.quant_config = quant_config if policy[name] else None
