"""Opt-in vLLM plugin using the native EngineCore utility transport."""

import os
from functools import wraps

from .control import (
    ControlError, Controller, DEFAULT_DRAIN_TIMEOUT_MS,
    DEFAULT_PREPARATION_TIMEOUT_MS,
)


def controller(engine):
    if not hasattr(engine, "_dynamic_weight"):
        from .adapter import EngineBackend
        engine._dynamic_weight = Controller(EngineBackend(engine))
    return engine._dynamic_weight


def status(self):
    owned = controller(self)
    return {**owned.status(), "observation": owned.backend.observe_progress(),
            "capabilities": owned.backend.capabilities()}


def alter(self, target, mode="drain", drain_timeout_ms=DEFAULT_DRAIN_TIMEOUT_MS,
          preparation_timeout_ms=DEFAULT_PREPARATION_TIMEOUT_MS):
    try:
        return {"operation": controller(self).alter(
            target, mode, drain_timeout_ms, preparation_timeout_ms)}
    except ControlError as error:
        return {"control_error": error.as_dict()}


def guard(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        def check_fence():
            failure = getattr(self, "_dynamic_weight_fatal", None)
            if failure:
                raise RuntimeError("Weight publication uncertain: " + failure)
        owned = getattr(self, "_dynamic_weight", None)
        if owned is not None:
            owned.poll_deadline()
        check_fence()
        result = function(self, *args, **kwargs)
        if function.__name__ == "_process_engine_step":
            owned = getattr(self, "_dynamic_weight", None)
            if owned is not None:
                owned.advance(result)
        check_fence()
        return result
    return wrapped


def capture_activation_peak(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        import torch
        before = torch.cuda.memory_allocated()
        result = function(self, *args, **kwargs)
        torch.cuda.synchronize()
        peak = max(0, torch.cuda.max_memory_allocated() - before)
        self._dynamic_quantization_activation_peak = max(
            peak, getattr(self, "_dynamic_quantization_activation_peak", 0))
        return result
    return wrapped


def verified_load(function):
    """Bind one complete native load to a stable SHA256 source snapshot."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if ((args and args[0]) or kwargs.get("load_dummy_weights", False)
                or kwargs.get("model") is not None
                or hasattr(self, "_dynamic_weight_source")):
            raise ValueError("Dynamic weight requires one complete native checkpoint load")
        from .source import CheckpointSnapshot
        allow_writable = os.environ.get("DYNAMIC_WEIGHT_ALLOW_WRITABLE_SOURCE", "0") == "1"
        source = CheckpointSnapshot(
            self.vllm_config.model_config.model, require_protected=not allow_writable)
        result = function(self, *args, **kwargs)
        source.check_content()
        self._dynamic_weight_source = source
        return result
    return wrapped


def register():
    if "dynamic_weight" not in os.environ.get("VLLM_PLUGINS", "").split(","):
        return
    install()


def install():
    from .compat import verify_runtime
    verify_runtime()
    if os.environ.get("VLLM_SERVER_DEV_MODE", "0") != "0":
        raise RuntimeError("Dynamic weight refuses development RPC routes")
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "0") != "0":
        raise RuntimeError("Dynamic weight refuses insecure callable serialization")
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(EngineCore, "_dynamic_weight_installed", False):
        return
    utilities = (("dynamic_weight_status", status), ("dynamic_weight_alter", alter))
    for name, _ in utilities:
        if hasattr(EngineCore, name):
            raise RuntimeError(f"Refusing to replace existing {name}")
    for name, function in utilities:
        setattr(EngineCore, name, function)
    for name in ("_handle_client_request", "_process_engine_step"):
        setattr(EngineCoreProc, name, guard(getattr(EngineCoreProc, name)))
    GPUModelRunner.profile_run = capture_activation_peak(GPUModelRunner.profile_run)
    GPUModelRunner.load_model = verified_load(GPUModelRunner.load_model)
    EngineCore._dynamic_weight_installed = True
