"""Private, version-pinned single-GPU adapter. Calls run on the engine thread."""

from copy import copy, deepcopy
from pathlib import Path
import os

from .control import ControlError
from .capabilities import (
    representation_decision,
    require_representation,
    validate_representation_hardware,
)
from .models import model_adapter
from .representations import (
    REPRESENTATIONS,
    initial_representation,
    validate_initial_load_format,
)


def collect_retired():
    import gc
    frozen = gc.get_freeze_count() > 0
    gc.unfreeze()
    gc.collect()
    if frozen:
        gc.freeze()


def validate_config(config):
    parallel = config.parallel_config
    model = config.model_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    reasons = []

    if os.environ.get("VLLM_SERVER_DEV_MODE", "0") != "0":
        reasons.append("development RPC routes must be disabled")
    if parallel.distributed_executor_backend != "uni":
        reasons.append("--distributed-executor-backend uni required")
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            reasons.append(name + " must equal 1")
    if getattr(parallel, "enable_fault_tolerance", False) or getattr(
        parallel, "use_ubatching", False
    ):
        reasons.append("fault tolerance and ubatching unsupported")
    if config.use_v2_model_runner or scheduler.async_scheduling:
        reasons.append("V1 runner with --no-async-scheduling required")
    if model.trust_remote_code:
        reasons.append("remote model/tokenizer code must be disabled")
    if getattr(model, "enable_prompt_embeds", False):
        reasons.append("prompt embeddings must be disabled on the reviewed runtime")
    if getattr(model, "is_multimodal_model", False):
        reasons.append("multimodal serving is outside the reviewed safety contract")
    if getattr(
        getattr(config, "observability_config", None),
        "otlp_traces_endpoint",
        None,
    ):
        reasons.append("OTLP tracing is outside the reviewed safety contract")
    if not model.enforce_eager or config.compilation_config.cudagraph_mode.name != "NONE":
        reasons.append("--enforce-eager required")
    if getattr(config.compilation_config.mode, "value", config.compilation_config.mode) != 0:
        reasons.append("compilation must be disabled")
    if str(model.dtype) != "torch.bfloat16":
        reasons.append("BF16 model configuration required")

    try:
        adapter = model_adapter(model.hf_config)
        initial = validate_initial_load_format(model, config.load_config)
        if config.device_config.device_type == "cuda":
            require_representation(initial, adapter, config)
        else:
            from .compat import validate_runtime_representation

            validate_runtime_representation(initial)
            if initial not in adapter.formats:
                raise ValueError(f"{initial} is not reviewed for {adapter.model_type}")
    except ValueError as exc:
        reasons.append(str(exc))
    if getattr(model.hf_config, "quantization_config", None):
        reasons.append("canonical checkpoint must be unquantized")
    if not Path(model.model).is_dir():
        reasons.append("an immutable local checkpoint directory is required")
    if model.enable_sleep_mode or getattr(model, "enable_cumem_allocator", False):
        reasons.append("sleep/CuMem allocator unsupported")
    if cache.cache_dtype not in ("auto", "bfloat16"):
        reasons.append("BF16 KV required")
    if scheduler.scheduler_cls is not None or parallel.worker_extension_cls:
        reasons.append("custom schedulers/worker extensions unsupported")
    for name in (
        "speculative_config",
        "lora_config",
        "kv_transfer_config",
        "ec_transfer_config",
        "weight_transfer_config",
    ):
        if getattr(config, name, None) is not None:
            reasons.append(name + " unsupported")
    if getattr(getattr(config, "aux_output_config", None), "enabled", False):
        reasons.append("auxiliary output connectors unsupported")
    offload = getattr(config, "offload_config", None)
    if offload and (
        offload.uva.cpu_offload_gb or offload.prefetch.offload_group_size
    ):
        reasons.append("weight offload unsupported")
    if config.device_config.device_type != "cuda":
        reasons.append("CUDA required")
    if reasons:
        raise ValueError(
            "Unsupported dynamic quantization configuration: " + "; ".join(reasons)
        )


class EngineBackend:
    def __init__(self, engine):
        from vllm.v1.engine.core import EngineCoreProc
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        if not isinstance(engine, EngineCoreProc):
            raise ValueError("Multiprocess EngineCoreProc required for progress and fencing")
        validate_config(engine.vllm_config)
        if type(engine.model_executor) is not UniProcExecutor:
            raise ValueError("Exact UniProcExecutor required")
        self.engine = engine
        self.worker = engine.model_executor.driver_worker.worker
        self.runner = self.worker.model_runner
        self.verified_source = getattr(self.runner, "_dynamic_weight_source", None)
        if self.verified_source is None:
            raise ValueError("Native model load has no verified canonical source snapshot")
        self.verified_source.check_content()
        quant = engine.vllm_config.model_config.quantization
        if quant == "bitsandbytes":
            from .compat import verify_bitsandbytes_runtime
            verify_bitsandbytes_runtime()
        self.current_format = initial_representation(engine.vllm_config.model_config)
        self.model_adapter = model_adapter(engine.vllm_config.model_config.hf_config)
        self.model_adapter.check_model(self.runner.model, self.runner.model_config)
        self.canonical_model_config = deepcopy(engine.vllm_config.model_config)
        self.source = str(self.verified_source.root)
        self.preparation = None
        self.holder = {}
        self.saved_input_block = None
        self.serving_reserve_bytes = 0
        self.destructive_started = False
        self.replacement_mode = None
        self.pause_lease = None
        from .lifecycle import DeadlineWakeup
        from vllm.v1.engine import EngineCoreRequestType
        self.deadline_wakeup = DeadlineWakeup(engine.input_queue, (EngineCoreRequestType.WAKEUP, None))

    def _check_source(self):
        self.verified_source.check()

    def representation_capability(self, target):
        """Static admission and request-policy plans; excludes transient engine state."""
        from .policy import loading_plan
        runner = getattr(self, "runner", None)
        decision = representation_decision(
            target,
            getattr(self, "model_adapter", None),
            getattr(runner, "vllm_config", None),
            getattr(runner, "device", None),
        )
        decision["loading_plans"] = {}
        decision["unavailable_modes"] = {}
        if decision["available"]:
            current = initial_representation(self.runner.model_config)
            for mode in ("drain", "dirty"):
                try:
                    plan = loading_plan(current, target, mode)
                    if (
                        mode == "dirty"
                        and self.runner.vllm_config.cache_config.enable_prefix_caching
                    ):
                        raise ControlError(
                            "invalid_request",
                            "Dirty mode requires --no-enable-prefix-caching at startup",
                        )
                    decision["loading_plans"][mode] = plan
                except ControlError as exc:
                    decision["unavailable_modes"][mode] = str(exc)
        return decision

    def validate(self, target, mode):
        decision = self.representation_capability(target)
        if not decision['available']:
            raise ControlError("invalid_request", decision["reason"])
        if mode not in decision["loading_plans"]:
            reason = decision["unavailable_modes"].get(
                mode, "mode must be drain or dirty; read capabilities"
            )
            raise ControlError("invalid_request", reason)
        if self.engine.model_executor.is_sleeping or self.engine.is_scheduler_paused():
            raise ControlError("operation_conflict", "Engine is already paused or sleeping")
        return decision["loading_plans"][mode]

    def pause(self, callback):
        from .lifecycle import PauseLease
        if self.pause_lease is not None:
            raise RuntimeError("A native pause lease is already owned")
        lease = self.pause_lease = PauseLease(self.engine, callback)
        lease.start()

    def arm_deadline(self, deadline_ns):
        self.deadline_wakeup.arm(deadline_ns)

    def clear_deadline(self):
        self.deadline_wakeup.clear()

    def release_pause(self):
        if self.pause_lease is not None:
            self.pause_lease.close()
            self.pause_lease = None

    def assert_drained(self):
        engine = self.engine
        scheduler = engine.scheduler
        runner = self.runner
        if not engine.is_scheduler_paused() or scheduler.running or engine.batch_queue:
            raise RuntimeError("Engine has not drained")
        if scheduler.has_finished_requests() or scheduler.deferred_frees:
            raise RuntimeError("Worker cleanup has not completed")
        blocks = scheduler.kv_cache_manager.block_pool.blocks
        if any(block.ref_cnt for block in blocks if not block.is_null):
            raise RuntimeError("KV references remain")
        if any(request.num_computed_tokens for request in scheduler.requests.values()):
            raise RuntimeError("Queued requests retain computed KV")
        if any(
            getattr(request, "num_preemptions", 0) or request.num_output_tokens
            for request in scheduler.requests.values()
        ):
            raise RuntimeError("preempted_request_holds_old_generation_output")
        if runner.requests or runner.input_batch.num_reqs:
            raise RuntimeError("Worker request state remains")

    def candidate_config(self, target):
        import torch
        from vllm.config import VllmConfig
        from vllm.config.quantization import _ONLINE_SHORTHANDS

        self._check_source()
        torch.cuda.synchronize()
        cfg = copy(self.runner.vllm_config)
        # Derive every candidate from canonical startup metadata. In particular,
        # restoration to BF16 must not inherit the current quantization recipe.
        cfg.model_config = deepcopy(self.canonical_model_config)
        cfg.load_config = deepcopy(cfg.load_config)
        cfg.cache_config = deepcopy(cfg.cache_config)
        # Never deepcopy live CUDA modules through the compilation registry.
        comp = cfg.compilation_config
        cfg.compilation_config = deepcopy(comp, {id(comp.static_forward_context): {}})
        cfg.compilation_config.static_all_moe_layers = []
        cfg.model_config.model = self.source
        rep = REPRESENTATIONS[target]
        cfg.model_config.quantization = rep.quantization
        cfg.model_config.quantization_config = (
            deepcopy(_ONLINE_SHORTHANDS[rep.shorthand])
            if rep.shorthand
            else None
        )
        cfg.load_config.load_format = (
            "bitsandbytes" if target == "nf4" else "safetensors"
        )
        cfg.quant_config = VllmConfig._get_quantization_config(
            cfg.model_config, cfg.load_config
        )
        return cfg

    def begin_replacement(self, target, dirty=False):
        from .canary import plan_canary
        from .component_policy import native_policy
        from .components import preflight
        from .linearwise import replace_slices

        self.destructive_started = False
        self.replacement_mode = "dirty" if dirty else "drain"
        cfg = self.candidate_config(target)
        plan_canary(self.runner.vllm_config, allow_bound=True)
        components = preflight(self, target, dirty)
        self.native_quantization_policy = native_policy(self, cfg, components)
        self._begin_holder(cfg)
        self.preparation = replace_slices(self, cfg, components, dirty)
        self.saved_input_block = self.engine.process_input_queue_block
        self.engine.process_input_queue_block = False

    def prepare_components(self, target, check_deadline):
        self.begin_replacement(target)
        while True:
            check_deadline()
            result = self.advance_replacement()
            check_deadline()
            if result.get("ready"):
                return self.take_candidate()

    def advance_replacement(self):
        result = next(self.preparation)
        if self.replacement_mode == "dirty":
            import torch

            if (
                torch.cuda.mem_get_info()[0]
                < self.serving_reserve_bytes
            ):
                raise ValueError("Insufficient serving headroom after replacement unit")
        return result

    def _begin_holder(self, cfg):
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
        self.holder = {"config": cfg, "metadata_before": {id(layer) for layer in LAYERWISE_INFO}}

    def take_candidate(self):
        cfg, model = self.holder["config"], self.holder["model"]
        self.validate_candidate(cfg, model)
        self.preparation.close()
        self.preparation = None
        return cfg, model

    def validate_candidate(self, cfg, model):
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        self._check_source()
        self.model_adapter.check_model(model, cfg.model_config)
        old_context = self.runner.compilation_config.static_forward_context
        context = cfg.compilation_config.static_forward_context
        if context.keys() != old_context.keys():
            raise ValueError("Candidate forward layer names differ")
        if cfg.compilation_config.static_all_moe_layers != self.runner.compilation_config.static_all_moe_layers:
            raise ValueError("Candidate MoE layer order differs")
        for name, layer in context.items():
            old = old_context[name]
            if type(layer) is not type(old):
                raise ValueError(f"Incompatible candidate layer type at {name}")
            if isinstance(layer, Attention):
                if (layer.get_attn_backend() is not old.get_attn_backend()
                        or layer.get_kv_cache_spec(cfg) != old.get_kv_cache_spec(self.runner.vllm_config)):
                    raise ValueError(f"Incompatible candidate attention at {name}")
            elif type(layer) is MoERunner:
                if layer.layer_name != old.layer_name:
                    raise ValueError(f"Incompatible MoE layer name at {name}")
            else:
                raise ValueError(f"Unreviewed forward registry entry at {name}")
        if any(p.is_meta for p in model.parameters()):
            raise ValueError("Candidate retains unmaterialized weights")

    def precommit(self, candidate):
        import torch
        if self.replacement_mode != "dirty":
            self.assert_drained()
        cfg, model = candidate
        if (self.holder.get("model") is not model or self.holder.get("config") is not cfg
                or not self.holder.get("validation")):
            raise ValueError("Candidate does not have owned successful validation")
        torch.cuda.synchronize()
        if self.replacement_mode != "dirty" and not self.engine.reset_prefix_cache():
            raise RuntimeError("Prefix cache reset refused")

    def commit(self, candidate):
        import torch
        from vllm.model_executor.layers.attention import Attention

        candidate_config, candidate_model = candidate
        engine = self.engine
        runner = self.runner
        candidate_context = candidate_config.compilation_config.static_forward_context
        for name, layer in candidate_context.items():
            if isinstance(layer, Attention):
                live_layer = runner.compilation_config.static_forward_context[name]
                layer.bind_kv_cache(live_layer.kv_cache)

        # Keep pre-cutover bindings and config references alive until every
        # publication assignment and synchronization succeeds.
        publication_references = (
            runner.model,
            runner.compilation_config.static_forward_context,
            runner.vllm_config.quant_config,
            runner.vllm_config.model_config,
        )
        live_config = runner.vllm_config
        runner.model = candidate_model
        live_config.quant_config = candidate_config.quant_config
        live_config.model_config = candidate_config.model_config
        live_config.load_config = candidate_config.load_config
        runner.load_config = candidate_config.load_config
        # Preserve the config object referenced by runner metadata builders.
        runner.compilation_config.static_forward_context = candidate_context
        runner.compilation_config.static_all_moe_layers = (
            candidate_config.compilation_config.static_all_moe_layers
        )
        runner.model_config = candidate_config.model_config
        self.worker.model_config = candidate_config.model_config
        engine.vllm_config.model_config = candidate_config.model_config
        engine.vllm_config.quant_config = candidate_config.quant_config
        engine.vllm_config.load_config = candidate_config.load_config
        torch.cuda.synchronize()
        self.holder = {}
        del publication_references
        self.destructive_started = False
        self.replacement_mode = None

    def recover_preparation(self):
        import torch
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO, LOADING_LAYERS
        if self.destructive_started:
            raise RuntimeError("Serving weights have been replaced; rollback is unavailable")
        # A failed native quantizer may have queued work on another stream.
        # Establish device completion before dropping any private GPU owners.
        torch.cuda.synchronize()
        if self.preparation is not None:
            self.preparation.close()
            self.preparation = None
        from .layerwise import retire_block
        retained = {id(module) for module in self.runner.model.modules()}
        retired = []
        private = self.holder.pop("unit", None)
        if private is not None:
            if id(private) in retained:
                raise RuntimeError("Recovery candidate is already serving")
            retired.extend(retire_block(private, retained))
        private = None
        before = self.holder.get("metadata_before")
        if before is not None:
            for layer in list(LAYERWISE_INFO):
                if id(layer) not in before:
                    info = LAYERWISE_INFO.pop(layer, None)
                    if info is not None:
                        info.reset()
                        info.restore_metadata = ({}, {})
                    LOADING_LAYERS.discard(layer)
        # Do not keep the final private layer alive across collection/refreeze.
        layer = info = None
        self.holder = {}
        torch.cuda.synchronize()
        collect_retired()
        if any(item() is not None for item in retired):
            raise RuntimeError("Private replacement unit still has module owners")
        self.replacement_mode = None

    def resume(self):
        from vllm.v1.engine import EngineCoreRequestType
        self.clear_deadline()
        self.release_pause()
        if self.saved_input_block is not None:
            self.engine.process_input_queue_block = self.saved_input_block
            self.saved_input_block = None
        self.engine.resume_scheduler()
        self.engine.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    def fence(self, error):
        from vllm.v1.engine import EngineCoreRequestType
        self.engine._dynamic_weight_fatal = error
        self.engine.process_input_queue_block = False
        self.engine.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
        # The fence is visible even if bounded cleanup itself fails.
        try:
            self.clear_deadline()
        finally:
            self.release_pause()

    def observe_progress(self):
        # No model traversal, synchronization, checkpoint reads or receipt lists.
        # These allocator counters are lifetime/current values, not ALTER peaks.
        import torch

        source = getattr(self, "verified_source", None)
        process_id = os.getpid()
        return {
            "pid": process_id,
            "worker_pid": process_id,
            "source_identity": source.content_identity if source else None,
            "source_protected": source.protected if source else None,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "replacement_mode": self.replacement_mode,
            "destructive_started": self.destructive_started,
            "running": len(self.engine.scheduler.running),
            "waiting": len(self.engine.scheduler.waiting),
            "worker_requests": len(self.runner.requests),
        }

    def capabilities(self):
        import torch
        from . import __version__
        from .compat import runtime_profile

        profile = runtime_profile()
        cc = torch.cuda.get_device_capability()
        decisions = [self.representation_capability(name) for name in REPRESENTATIONS]
        available = []
        for decision in decisions:
            if not decision["available"]:
                continue
            name = decision["name"]
            available.append(
                {
                    **REPRESENTATIONS[name].descriptor(),
                    "modes": list(decision["loading_plans"]),
                    "requires": (
                        "vllm-bnb-plugin 0.0.3 + bitsandbytes 0.50.2"
                        if name == "nf4"
                        else None
                    ),
                    **{key: value for key, value in decision.items() if key != "name"},
                }
            )
        return {
            "plugin_version": __version__,
            "adapter": f"vllm-{profile['version']}-v1-uni",
            "vllm_version": profile["version"],
            "model_type": self.model_adapter.model_type,
            "architecture": self.model_adapter.architecture,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(cc),
            "model_config_dtype": "bfloat16",
            "kv_dtype": "bfloat16",
            "activation_arithmetic": "selected by each native quantization recipe",
            "representations": available,
            "unavailable_representations": [
                decision for decision in decisions if not decision["available"]
            ],
            "admission_scope": (
                "recipe candidate; exact-shape construction, memory admission, "
                "and canaries remain required"
            ),
            "plugin_startup_source": "canonical_unquantized_checkpoint",
            "source_content_identity": self.verified_source.content_identity,
            "source_protected": self.verified_source.protected,
            "active_source_content_verification": (
                "sha256_inventory_plus_stat_identity"
            ),
        }
