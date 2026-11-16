"""Private, version-pinned single-GPU adapter. Calls run on the engine thread."""

from copy import copy, deepcopy
from pathlib import Path
import os

from .control import ControlError
from .capabilities import (require_representation, representation_decision,
                           validate_representation_hardware)
from .models import model_adapter
from .representations import REPRESENTATIONS, initial_representation


def forget_reload_metadata(model):
    from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO, LOADING_LAYERS
    # Online loader closures in saved meta tensors can retain their layer even
    # though the registry uses weak keys. Retired models will never reload again.
    for layer in model.modules():
        LAYERWISE_INFO.pop(layer, None)
        LOADING_LAYERS.discard(layer)


def collect_retired():
    import gc
    frozen = gc.get_freeze_count() > 0
    gc.unfreeze()
    gc.collect()
    if frozen:
        gc.freeze()


def validate_config(config):
    p, m, s, c = (config.parallel_config, config.model_config,
                  config.scheduler_config, config.cache_config)
    reasons = []
    if os.environ.get("VLLM_SERVER_DEV_MODE", "0") != "0":
        reasons.append("development RPC routes must be disabled")
    if p.distributed_executor_backend != "uni":
        reasons.append("--distributed-executor-backend uni required")
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(p, name, 1) != 1:
            reasons.append(name + " must equal 1")
    if getattr(p, "enable_fault_tolerance", False) or getattr(p, "use_ubatching", False):
        reasons.append("fault tolerance and ubatching unsupported")
    if config.use_v2_model_runner or s.async_scheduling:
        reasons.append("V1 runner with --no-async-scheduling required")
    if m.trust_remote_code:
        reasons.append("remote model/tokenizer code must be disabled")
    if getattr(m, "enable_prompt_embeds", False):
        # Pinned vLLM 0.30.0 is affected by CVE-2026-105922 for prompt
        # embeddings plus penalties. This plugin qualifies token inputs only.
        reasons.append("prompt embeddings must be disabled on the reviewed runtime")
    if getattr(m, "is_multimodal_model", False):
        reasons.append("multimodal serving is outside the reviewed safety contract")
    if getattr(getattr(config, "observability_config", None), "otlp_traces_endpoint", None):
        # The reviewed opentelemetry-api has GHSA-xcq7-p3pf-p88h. Optional
        # tracing needs a patched, separately qualified dependency profile.
        reasons.append("OTLP tracing is outside the reviewed safety contract")
    if not m.enforce_eager or config.compilation_config.cudagraph_mode.name != "NONE":
        reasons.append("--enforce-eager required")
    if getattr(config.compilation_config.mode, "value", config.compilation_config.mode) != 0:
        reasons.append("compilation must be disabled")
    if str(m.dtype) != "torch.bfloat16":
        reasons.append("BF16 model configuration required")
    try:
        adapter = model_adapter(m.hf_config)
        initial = initial_representation(m)
        if config.device_config.device_type == "cuda":
            require_representation(initial, adapter, config)
        else:
            from .compat import validate_runtime_model
            validate_runtime_model(adapter.model_type, initial)
            if initial not in adapter.formats:
                raise ValueError(f"{initial} is not reviewed for {adapter.model_type}")
    except ValueError as exc:
        reasons.append(str(exc))
    if getattr(m.hf_config, "quantization_config", None):
        reasons.append("canonical checkpoint must be unquantized")
    if os.environ.get("DYNAMIC_QUANTIZATION_ARTIFACT_MANIFEST"):
        reasons.append("prepared artifact adapters are not part of this package")
    if not Path(m.model).is_dir():
        reasons.append("an immutable local checkpoint directory is required")
    if config.load_config.load_format not in ("auto", "safetensors", "bitsandbytes"):
        reasons.append("default safetensors loader required initially")
    if m.enable_sleep_mode or getattr(m, "enable_cumem_allocator", False):
        reasons.append("sleep/CuMem allocator unsupported")
    if c.cache_dtype not in ("auto", "bfloat16"):
        reasons.append("BF16 KV required")
    if s.scheduler_cls is not None or p.worker_extension_cls:
        reasons.append("custom schedulers/worker extensions unsupported")
    for name in ("speculative_config", "lora_config", "kv_transfer_config", "ec_transfer_config", "weight_transfer_config"):
        if getattr(config, name, None) is not None:
            reasons.append(name + " unsupported")
    if getattr(getattr(config, "aux_output_config", None), "enabled", False):
        reasons.append("auxiliary output connectors unsupported")
    offload = getattr(config, "offload_config", None)
    if offload and (offload.uva.cpu_offload_gb or offload.prefetch.offload_group_size):
        reasons.append("weight offload unsupported")
    if config.device_config.device_type != "cuda":
        reasons.append("CUDA required")
    if reasons:
        raise ValueError("Unsupported dynamic quantization configuration: " + "; ".join(reasons))


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
        self.source_bytes = self.verified_source.source_bytes
        self.source_identity = self.verified_source.identity
        self.artifacts = None
        self.preparation = None
        self.holder = {}
        self.saved_input_block = None
        self.memory_admission = None
        self.destructive_started = False
        self.replacement_mode = None
        self.replacement_metrics = []
        self.pause_lease = None
        from .lifecycle import DeadlineWakeup
        from vllm.v1.engine import EngineCoreRequestType
        self.deadline_wakeup = DeadlineWakeup(engine.input_queue, (EngineCoreRequestType.WAKEUP, None))

    def _source_identity(self):
        result = {}
        for path in Path(self.source).iterdir():
            if path.suffix in {".json", ".safetensors"}:
                info = path.stat()
                result[str(path)] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                                     info.st_ctime_ns, info.st_mode, info.st_uid, info.st_gid)
        return result

    def _check_source(self):
        self.verified_source.check()
        if self._source_identity() != {
            str(Path(self.source) / name): identity
            for name, identity in self.source_identity.items()
            if Path(name).suffix in {".json", ".safetensors"}
        }:
            raise ValueError("Canonical checkpoint stat identity changed")

    def representation_capability(self, target):
        """Static admission and request-policy plans; excludes transient engine state."""
        from .policy import loading_plan
        runner = getattr(self, 'runner', None)
        decision = representation_decision(
            target, getattr(self, 'model_adapter', None),
            getattr(runner, 'vllm_config', None),
            getattr(self, 'artifacts', None) is not None, getattr(runner, 'device', None))
        decision['loading_plans'] = {}
        decision['unavailable_modes'] = {}
        if decision['available']:
            current = initial_representation(self.runner.model_config)
            for mode in ('drain', 'dirty'):
                try:
                    plan = loading_plan(getattr(self.model_adapter, 'model_type', None), current, target, mode)
                    if mode == 'dirty' and self.runner.vllm_config.cache_config.enable_prefix_caching:
                        raise ControlError('invalid_request', 'Dirty mode requires --no-enable-prefix-caching at startup')
                    decision['loading_plans'][mode] = plan
                except ControlError as exc:
                    decision['unavailable_modes'][mode] = str(exc)
        return decision

    def validate(self, target, mode):
        decision = self.representation_capability(target)
        if not decision['available']:
            raise ControlError('invalid_request', decision['reason'])
        if mode not in decision['loading_plans']:
            raise ControlError('invalid_request', decision['unavailable_modes'].get(mode, 'mode must be drain or dirty; read capabilities'))
        if self.engine.model_executor.is_sleeping or self.engine.is_scheduler_paused():
            raise ControlError("operation_conflict", "Engine is already paused or sleeping")
        return decision['loading_plans'][mode]

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
        e, s, r = self.engine, self.engine.scheduler, self.runner
        if not e.is_scheduler_paused() or s.running or e.batch_queue:
            raise RuntimeError("Engine has not drained")
        if s.has_finished_requests() or s.deferred_frees:
            raise RuntimeError("Worker cleanup has not completed")
        if any(b.ref_cnt for b in s.kv_cache_manager.block_pool.blocks if not b.is_null):
            raise RuntimeError("KV references remain")
        if any(q.num_computed_tokens for q in s.requests.values()):
            raise RuntimeError("Queued requests retain computed KV")
        if any(getattr(q, "num_preemptions", 0) or q.num_output_tokens
               for q in s.requests.values()):
            raise RuntimeError("preempted_request_holds_old_generation_output")
        if r.requests or r.input_batch.num_reqs:
            raise RuntimeError("Worker request state remains")

    def candidate_config(self, target, online=False, replacement=False):
        import torch
        from vllm.config import VllmConfig
        from vllm.config.quantization import _ONLINE_SHORTHANDS
        self._check_source()
        torch.cuda.synchronize()
        # Retain old weights for rollback. Online loading cannot spend cached
        # activation blocks: serving will need them between preparation slices.
        free, _ = torch.cuda.mem_get_info()
        reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        required = self.source_bytes + 512 * 1024**2
        activation = getattr(self.runner, "_dynamic_quantization_activation_peak", None)
        if online and activation is None:
            raise ValueError("Online preparation requires captured startup activation profiling")
        reserve = (activation + 256 * 1024**2) if online else 0
        available = free - reserve if online else free + reusable
        self.memory_admission = {"online": online, "driver_free_bytes": free,
                                 "cached_free_bytes": reusable, "required_bytes": required,
                                 "serving_reserve_bytes": reserve, "available_bytes": available}
        if not replacement and available < required:
            raise ValueError(f"Insufficient shadow headroom: need {required}, available {available}")
        cfg = copy(self.runner.vllm_config)
        # Always start from canonical model metadata, including after loading a
        # derived checkpoint. Never carry compressed-tensors metadata into BF16.
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
        cfg.model_config.quantization_config = deepcopy(_ONLINE_SHORTHANDS[rep.shorthand]) if rep.shorthand else None
        cfg.load_config.load_format = "bitsandbytes" if target == "nf4" else "safetensors"
        cfg.quant_config = VllmConfig._get_quantization_config(cfg.model_config, cfg.load_config)
        return cfg

    def begin_replacement(self, target, dirty=False):
        from .layerwise import preflight, replace_slices
        from .policy import loading_plan
        plan = loading_plan(self.model_adapter.model_type, initial_representation(self.runner.model_config), target,
                            "dirty" if dirty else "drain")
        linear = plan["strategy"] in {"native_linears", "native_components"}
        if linear:
            from .linearwise import replace_slices
        self.destructive_started = False
        self.replacement_mode = "dirty" if dirty else "drain"
        self.replacement_metrics = []
        cfg = self.candidate_config(target, replacement=True)
        if plan['strategy'] == 'native_components':
            from .canary import plan_canary
            plan_canary(self.runner.vllm_config, allow_bound=True)
            from .components import preflight as component_preflight
            groups = component_preflight(self, target, dirty)
            from .component_policy import native_policy
            self.native_quantization_policy = native_policy(self, cfg, groups)
        else:
            groups = preflight(self, target, dirty, linear=linear)
        self._begin_holder(cfg)
        self.preparation = replace_slices(self, cfg, groups, dirty)
        self.saved_input_block = self.engine.process_input_queue_block
        self.engine.process_input_queue_block = False

    def prepare_layerwise(self, target, check_deadline):
        self.begin_replacement(target)
        while True:
            check_deadline()
            result = self.advance_replacement()
            check_deadline()
            if result.get("ready"):
                return self.take_candidate()

    def prepare_unload(self, target):
        from .layerwise import preflight
        cfg = self.candidate_config(target, replacement=True)
        preflight(self, target)
        self.memory_admission["mode"] = "unload"
        self.replacement_mode = "unload"
        self.replacement_metrics = []
        # No old model reference is retained for rollback. Existing Attention
        # objects keep the serving KV bindings until the new model is committed.
        self.destructive_started = True
        old = self.runner.model
        self.runner.model = None
        forget_reload_metadata(old)
        del old
        collect_retired()
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return self.prepare(target, cfg=cfg)

    def advance_replacement(self):
        result = next(self.preparation)
        self.replacement_metrics.append(result)
        if self.replacement_mode == "dirty":
            import torch
            if torch.cuda.mem_get_info()[0] < self.memory_admission["serving_reserve_bytes"]:
                raise ValueError("Insufficient serving headroom after replacement unit")
        return result

    def _begin_holder(self, cfg):
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
        self.holder = {"config": cfg, "metadata_before": {id(layer) for layer in LAYERWISE_INFO}}

    def prepare(self, target, cfg=None):
        import torch
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader import get_model_loader
        cfg = cfg if cfg is not None else self.candidate_config(target)
        self._begin_holder(cfg)
        loader = get_model_loader(cfg.load_config)
        create = loader.create_model
        def capture(*args, **kwargs):
            model = create(*args, **kwargs)
            self.holder["model"] = model
            return model
        loader.create_model = capture
        with torch.inference_mode(), set_current_vllm_config(cfg):
            model = loader.load_model(cfg, cfg.model_config)
            torch.cuda.synchronize()
        self.validate_candidate(cfg, model)
        from .canary import canary_slices
        for result in canary_slices(cfg, model):
            if result.get("ready"):
                self.holder["validation"] = result["validation"]
        return cfg, model

    def begin_prepare(self, target):
        cfg = self.candidate_config(target, online=True)
        extra = cfg.load_config.model_loader_extra_config
        if extra.get("enable_multithread_load") or cfg.load_config.safetensors_load_strategy in {"eager", "prefetch"}:
            raise ValueError("Online slices require serial lazy safetensors loading")
        self._begin_holder(cfg)
        self.preparation = self._candidate_slices(cfg)
        self.saved_input_block = self.engine.process_input_queue_block
        self.engine.process_input_queue_block = False

    def _candidate_slices(self, cfg):
        from .incremental import load_slices
        from .canary import canary_slices
        for result in load_slices(cfg, self.holder):
            yield {**result, "ready": False}
        self.validate_candidate(cfg, self.holder["model"])
        for result in canary_slices(cfg, self.holder["model"]):
            if result.get("ready"):
                self.holder["validation"] = result["validation"]
            yield result

    def advance_prepare(self):
        import torch
        if self.preparation is None:
            raise RuntimeError("No owned preparation")
        result = next(self.preparation)
        free, _ = torch.cuda.mem_get_info()
        if free < self.memory_admission["serving_reserve_bytes"]:
            raise ValueError("Insufficient serving activation reserve after preparation slice")
        return result

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
        cfg, model = candidate
        e, r = self.engine, self.runner
        for name, layer in cfg.compilation_config.static_forward_context.items():
            if isinstance(layer, Attention):
                layer.bind_kv_cache(r.compilation_config.static_forward_context[name].kv_cache)
        # Keep references until every publication and observation succeeds.
        old = (r.model, r.compilation_config.static_forward_context,
               r.vllm_config.quant_config, r.vllm_config.model_config)
        live = r.vllm_config
        r.model = model
        live.quant_config = cfg.quant_config
        live.model_config = cfg.model_config
        live.load_config = cfg.load_config
        r.load_config = cfg.load_config
        # Preserve the config object referenced by runner metadata builders.
        r.compilation_config.static_forward_context = cfg.compilation_config.static_forward_context
        r.compilation_config.static_all_moe_layers = cfg.compilation_config.static_all_moe_layers
        r.model_config = cfg.model_config
        self.worker.model_config = cfg.model_config
        e.vllm_config.model_config = cfg.model_config
        e.vllm_config.quant_config = cfg.quant_config
        e.vllm_config.load_config = cfg.load_config
        torch.cuda.synchronize()
        validation = self.holder["validation"]
        self.holder = {}
        replacement = self.replacement_mode is not None
        if not replacement:
            forget_reload_metadata(old[0])
        del old
        # vLLM model wrappers/weight loaders contain Python reference cycles.
        # Reclaim retired generations here, as part of measured cutover, rather
        # than waiting for an unrelated future Python allocation to trigger GC.
        # EngineCore freezes the startup heap to avoid serving-time GC scans.
        # Ordinary collect() cannot see retired objects in that generation.
        if self.replacement_mode not in {"drain", "dirty"}:
            collect_retired()
        self.destructive_started = False
        self.replacement_mode = None
        return {**self.observe(), "candidate_validation": validation}

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
        iterator = self.holder.get("iterator")
        if iterator is not None:
            iterator.close()
        from .layerwise import retire_block
        retained = {id(module) for module in self.runner.model.modules()}
        retired = []
        for key in ("unit", "block"):
            private = self.holder.pop(key, None)
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
        layer = info = iterator = None
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
        source = getattr(self, 'verified_source', None)
        return {"pid": os.getpid(), "worker_pid": os.getpid(),
                "source_identity": source.content_identity if source else None,
                "source_protected": source.protected if source else None,
                "torch_allocated_bytes": torch.cuda.memory_allocated(),
                "torch_reserved_bytes": torch.cuda.memory_reserved(),
                "replacement_mode": self.replacement_mode,
                "destructive_started": self.destructive_started,
                "running": len(self.engine.scheduler.running),
                "waiting": len(self.engine.scheduler.waiting),
                "worker_requests": len(self.runner.requests)}

    def observe(self):
        import torch
        from collections import Counter
        r = self.runner
        params = list(r.model.parameters())
        storage = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in params}
        return {"pid": os.getpid(), "worker_pid": os.getpid(),
                "parameter_bytes": sum(storage.values()),
                "parameter_dtypes": dict(Counter(str(p.dtype) for p in params)),
                "torch_allocated_bytes": torch.cuda.memory_allocated(),
                "torch_reserved_bytes": torch.cuda.memory_reserved(),
                # Lifetime high-water marks; not per-operation or NVML usage.
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "source": self.source, "source_bytes": self.source_bytes,
                "loaded_checkpoint": r.model_config.model,
                "memory_admission": self.memory_admission,
                "replacement_mode": self.replacement_mode,
                "destructive_started": self.destructive_started,
                "replacement_blocks": self.replacement_metrics,
                "running": len(self.engine.scheduler.running),
                "waiting": len(self.engine.scheduler.waiting),
                "worker_requests": len(r.requests)}

    def capabilities(self):
        import torch
        from . import __version__
        from .representations import native_online_inventory
        from .compat import runtime_profile
        profile = runtime_profile()
        cc = torch.cuda.get_device_capability()
        decisions = [self.representation_capability(name) for name in REPRESENTATIONS]
        available = [dict(REPRESENTATIONS[d['name']].descriptor(),
                          native_startup=True,
                          modes=list(d['loading_plans']), plugin_modes=list(d['loading_plans']),
                          requires="vllm-bnb-plugin 0.0.3 + bitsandbytes 0.50.2" if d['name'] == 'nf4' else None,
                          **{k:v for k,v in d.items() if k != 'name'})
                     for d in decisions if d['available']]
        return {"plugin_version": __version__, "adapter": f"vllm-{profile['version']}-v1-uni", "vllm_version": profile["version"], "model_type": self.model_adapter.model_type,
                "architecture": self.model_adapter.architecture,
                "gpu": torch.cuda.get_device_name(), "compute_capability": list(cc),
                "model_config_dtype": "bfloat16", "kv_dtype": "bfloat16",
                "activation_arithmetic": "selected by each native quantization recipe",
                "representations": available,
                "unavailable_representations": [d for d in decisions if not d['available']],
                "admission_scope": "recipe candidate; native exact-shape construction, memory admission and canaries still required",
                "plugin_startup_source": "canonical_unquantized_checkpoint",
                "native_online_inventory": (native_online_inventory()
                                            if profile['version'] == '0.32.0' else []),
                "qualification": "experimental; native exact-shape construction and operation canaries remain required",
                "source_content_identity": self.verified_source.content_identity,
                "source_protected": self.verified_source.protected,
                "active_source_content_verification": "sha256_inventory_plus_stat_identity"}
