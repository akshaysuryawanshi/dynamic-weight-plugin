"""Shared recipe admission; native vLLM still owns exact layer/kernel selection.

The metadata probe is deliberately not a numerical qualification or a promise
that every tensor shape will fit. Native construction and canaries remain gates.
No kernel, device capability, backend preference or weight is overridden here.
"""

from contextlib import nullcontext
import os

from .representations import REPRESENTATIONS


def native_kernel_probe(target):
    """Run native selection on a representative dense shape without weights.

    Return the probe selection separately from actual serving kernel evidence.
    MoE banks have an additional native oracle during component construction.
    """
    import torch
    from vllm.model_executor.kernels.linear import (
        init_fp8_linear_kernel, init_mxfp4_linear_kernel, init_mxfp8_linear_kernel,
    )
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod, Fp8PtpcOnlineLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym, kFp8Static128BlockSym, kFp8StaticTensorSym, kMxfp4Dynamic,
    )
    shape = (256, 256)
    if target == 'mxfp4':
        kernel = init_mxfp4_linear_kernel(activation_quant_key=kMxfp4Dynamic)
    elif target == 'mxfp8':
        kernel = init_mxfp8_linear_kernel(weight_shape=shape)
    else:
        keys = {
            'fp8': (Fp8PerTensorOnlineLinearMethod.activation_quant_key, kFp8StaticTensorSym),
            'fp8_per_channel': (Fp8PtpcOnlineLinearMethod.activation_quant_key,
                                Fp8PtpcOnlineLinearMethod.weight_quant_key),
            'fp8_per_block': (kFp8Dynamic128Sym, kFp8Static128BlockSym),
        }
        activation, weight = keys[target]
        kernel = init_fp8_linear_kernel(
            activation_quant_key=activation, weight_quant_key=weight,
            input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=shape)
        # Match the native online PTPC semantic guard, not a named SM89 kernel.
        from vllm.model_executor.kernels.linear.scaled_mm import MarlinFP8ScaledMMLinearKernel
        if target == 'fp8_per_channel' and isinstance(kernel, MarlinFP8ScaledMMLinearKernel):
            raise ValueError('per-channel FP8 requires per-token FP8 activations; native Marlin is W8A16')
    return {'kernel': type(kernel).__name__, 'weight_shape': list(shape),
            'scope': 'dense_metadata_probe_not_serving_evidence'}


def inspect_device():
    """Bind native selection to the active CUDA device, not a synthetic SM."""
    import torch
    from vllm.platforms import current_platform
    if not current_platform.is_cuda():
        raise ValueError('Native CUDA platform required')
    index = torch.cuda.current_device()
    cc = tuple(torch.cuda.get_device_capability(index))
    native = current_platform.get_device_capability(index)
    if native is None or tuple(native) != cc:
        raise ValueError('Torch and native CUDA device capabilities disagree')
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('Native BF16 arithmetic required')
    return cc


def validate_representation_hardware(target, config=None, device=None):
    """Shared hardware/dependency gate for startup, ALTER and status."""
    from .compat import runtime_profile, verify_bitsandbytes_runtime

    profile = runtime_profile()
    if target not in REPRESENTATIONS:
        raise ValueError('Unknown representation: ' + target)
    if target not in profile.get('representations', REPRESENTATIONS):
        raise ValueError(f"vLLM {profile.get('version')} profile does not qualify representation {target}")
    import torch
    with torch.cuda.device(device) if device is not None else nullcontext():
        cc = inspect_device()
        result = {'compute_capability': list(cc)}
        if target == 'nf4':
            if 'bitsandbytes' not in os.environ.get('VLLM_PLUGINS', '').split(','):
                raise ValueError('NF4 requires the bitsandbytes general plugin to be enabled')
            verify_bitsandbytes_runtime()
            result['native_probe'] = {'scope': 'pinned_bitsandbytes_dependency_check_not_kernel_execution'}
        elif target in {
            'fp8', 'fp8_per_channel', 'fp8_per_block', 'mxfp4', 'mxfp8'
        }:
            from vllm.config import set_current_vllm_config
            with set_current_vllm_config(config) if config is not None else nullcontext():
                result['native_probe'] = native_kernel_probe(target)
        return result


def representation_decision(target, adapter=None, config=None, device=None):
    """One explainable availability decision, including model/runtime constraints."""
    from .compat import validate_runtime_representation
    decision = {'name': target, 'available': False, 'reason': None}
    try:
        if target not in REPRESENTATIONS:
            raise ValueError('Unknown representation: ' + target)
        if adapter is not None and target not in adapter.formats:
            raise ValueError(f'{target} is not reviewed for {adapter.model_type}')
        validate_runtime_representation(target)
        decision.update(validate_representation_hardware(target, config, device))
        decision['available'] = True
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        decision['reason'] = f'{type(exc).__name__}: {exc}'[:1600]
    return decision


def require_representation(target, adapter, config=None, device=None):
    decision = representation_decision(target, adapter, config, device)
    if not decision['available']:
        raise ValueError(decision['reason'])
    return decision
