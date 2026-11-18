import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import textwrap
import unittest
from unittest.mock import patch
import sys

from dynamic_weight_plugin.component_policy import (
    initialize_constructor_runtime,
    inspect_policy,
    native_policy,
)
from dynamic_weight_plugin.adapter import configuration_issues, validate_config
from dynamic_weight_plugin.compat import runtime_profile
from dynamic_weight_plugin.config_issues import ConfigIssueCode
from dynamic_weight_plugin.control import ControlError
from dynamic_weight_plugin.models import _verify_selected_class_source
from dynamic_weight_plugin.policy import loading_plan
from dynamic_weight_plugin.representations import (
    REPRESENTATIONS,
    validate_initial_load_format,
)
from dynamic_weight_plugin.source import CheckpointSnapshot


class Component:
    def __init__(self, quant_config=None, quant_method=None):
        self.quant_config = quant_config
        self.quant_method = quant_method


class Model:
    def __init__(self, modules):
        self.modules = modules

    def named_modules(self):
        return self.modules.items()

    def get_submodule(self, name):
        return self.modules[name]


class NativeContractTests(unittest.TestCase):
    def test_configuration_findings_keep_identity_order_and_messages(self):
        parallel = SimpleNamespace(
            distributed_executor_backend="ray",
            tensor_parallel_size=2,
            pipeline_parallel_size=3,
            data_parallel_size=4,
            decode_context_parallel_size=5,
            prefill_context_parallel_size=6,
            enable_fault_tolerance=True,
            use_ubatching=False,
            worker_extension_cls="custom.Worker",
        )
        model = SimpleNamespace(
            trust_remote_code=True,
            enable_prompt_embeds=True,
            is_multimodal_model=True,
            enforce_eager=False,
            dtype="torch.float16",
            hf_config=SimpleNamespace(quantization_config={"quant_method": "old"}),
            model="/path/that/does/not/exist",
            enable_sleep_mode=True,
            enable_cumem_allocator=True,
        )
        config = SimpleNamespace(
            parallel_config=parallel,
            model_config=model,
            scheduler_config=SimpleNamespace(
                async_scheduling=True, scheduler_cls="custom.Scheduler"
            ),
            cache_config=SimpleNamespace(cache_dtype="fp8"),
            compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(name="FULL"),
                mode=SimpleNamespace(value=1),
            ),
            observability_config=SimpleNamespace(
                otlp_traces_endpoint="http://traces"
            ),
            load_config=SimpleNamespace(load_format="auto"),
            device_config=SimpleNamespace(device_type="cpu"),
            use_v2_model_runner=True,
            speculative_config=object(),
            lora_config=object(),
            kv_transfer_config=object(),
            ec_transfer_config=object(),
            weight_transfer_config=object(),
            aux_output_config=SimpleNamespace(enabled=True),
            offload_config=SimpleNamespace(
                uva=SimpleNamespace(cpu_offload_gb=1),
                prefetch=SimpleNamespace(offload_group_size=0),
            ),
        )
        expected_messages = [
            "development RPC routes must be disabled",
            "--distributed-executor-backend uni required",
            "tensor_parallel_size must equal 1",
            "pipeline_parallel_size must equal 1",
            "data_parallel_size must equal 1",
            "decode_context_parallel_size must equal 1",
            "prefill_context_parallel_size must equal 1",
            "fault tolerance and ubatching unsupported",
            "V1 runner with --no-async-scheduling required",
            "remote model/tokenizer code must be disabled",
            "prompt embeddings must be disabled on the reviewed runtime",
            "multimodal serving is outside the reviewed safety contract",
            "OTLP tracing is outside the reviewed safety contract",
            "--enforce-eager required",
            "compilation must be disabled",
            "BF16 model configuration required",
            "representation rejected",
            "canonical checkpoint must be unquantized",
            "an immutable local checkpoint directory is required",
            "sleep/CuMem allocator unsupported",
            "BF16 KV required",
            "custom schedulers/worker extensions unsupported",
            "speculative_config unsupported",
            "lora_config unsupported",
            "kv_transfer_config unsupported",
            "ec_transfer_config unsupported",
            "weight_transfer_config unsupported",
            "auxiliary output connectors unsupported",
            "weight offload unsupported",
            "CUDA required",
        ]
        with (
            patch.dict("os.environ", {"VLLM_SERVER_DEV_MODE": "1"}),
            patch(
                "dynamic_weight_plugin.adapter.model_adapter",
                side_effect=ValueError("representation rejected"),
            ),
        ):
            issues = configuration_issues(config)
            self.assertEqual([issue.message for issue in issues], expected_messages)
            with self.assertRaisesRegex(
                ValueError,
                "^Unsupported dynamic quantization configuration: "
                + "; ".join(expected_messages)
                + "$",
            ):
                validate_config(config)

        self.assertEqual(issues[1].code, ConfigIssueCode.EXECUTOR)
        self.assertEqual((issues[1].actual, issues[1].expected), ("ray", "uni"))
        parallel_issue = issues[2]
        self.assertEqual(parallel_issue.code, ConfigIssueCode.PARALLEL_SIZE)
        self.assertEqual(
            (parallel_issue.field, parallel_issue.actual, parallel_issue.expected),
            ("tensor_parallel_size", 2, 1),
        )
        self.assertEqual(issues[15].code, ConfigIssueCode.MODEL_DTYPE)
        self.assertEqual(issues[20].code, ConfigIssueCode.KV_DTYPE)
        self.assertEqual(issues[-1].code, ConfigIssueCode.DEVICE)

        with (
            patch.dict("os.environ", {"VLLM_SERVER_DEV_MODE": "0"}),
            patch(
                "dynamic_weight_plugin.adapter.model_adapter",
                side_effect=OSError("native import failed"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "native import failed"):
                configuration_issues(config)

    def test_profiles_admit_the_same_seven_component_formats(self):
        expected = {
            "bf16",
            "fp8",
            "fp8_per_channel",
            "fp8_per_block",
            "mxfp8",
            "mxfp4",
            "nf4",
        }
        for version in ("0.32.0", "0.33.0"):
            profile = runtime_profile(version)
            self.assertEqual(set(profile["representations"]), expected)
            self.assertEqual(set(profile["component_formats"]), expected)
        self.assertEqual(set(REPRESENTATIONS), expected)

    def test_block_fp8_uses_native_component_methods(self):
        from vllm.config.quantization import _ONLINE_SHORTHANDS
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig,
        )

        config = OnlineQuantizationConfig(_ONLINE_SHORTHANDS["fp8_per_block"])
        linear = config.resolve_quant_method_cls(LinearBase, "component")
        experts = config.resolve_quant_method_cls(RoutedExperts, "component")
        self.assertEqual(linear[-1].__name__, "Fp8PerBlockOnlineLinearMethod")
        self.assertEqual(experts[-1].__name__, "Fp8PerBlockOnlineMoEMethod")
        plan = loading_plan("bf16", "fp8_per_block", "drain")
        self.assertEqual(plan["strategy"], "native_components")
        self.assertEqual(plan["scheduling"], "uninterrupted")

    def test_mx_formats_use_native_methods_and_mode_contracts(self):
        from vllm.config.quantization import _ONLINE_SHORTHANDS
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig,
        )

        expected = {
            "mxfp8": ("Mxfp8OnlineLinearMethod", "Mxfp8OnlineMoEMethod"),
            "mxfp4": ("Mxfp4OnlineLinearMethod", "Mxfp4OnlineMoEMethod"),
        }
        for name, (linear_name, expert_name) in expected.items():
            config = OnlineQuantizationConfig(_ONLINE_SHORTHANDS[name])
            linear = config.resolve_quant_method_cls(LinearBase, "component")
            experts = config.resolve_quant_method_cls(RoutedExperts, "component")
            self.assertEqual(linear[-1].__name__, linear_name)
            self.assertEqual(experts[-1].__name__, expert_name)
            self.assertEqual(
                loading_plan("bf16", name, "drain")["strategy"],
                "native_components",
            )

        dirty = loading_plan("bf16", "mxfp8", "dirty")
        self.assertEqual(dirty["scheduling"], "between_steps")
        for current, target in (("bf16", "mxfp4"), ("mxfp4", "bf16")):
            with self.assertRaisesRegex(ControlError, "mxfp4 only supports drain mode"):
                loading_plan(current, target, "dirty")

    def test_initial_representation_matches_native_loader_dispatch(self):
        bf16 = SimpleNamespace(quantization=None)
        nf4 = SimpleNamespace(quantization="bitsandbytes")
        self.assertEqual(
            validate_initial_load_format(
                bf16, SimpleNamespace(load_format="safetensors")
            ),
            "bf16",
        )
        self.assertEqual(
            validate_initial_load_format(
                nf4, SimpleNamespace(load_format="bitsandbytes")
            ),
            "nf4",
        )
        for model_config, load_format in (
            (bf16, "bitsandbytes"),
            (nf4, "safetensors"),
        ):
            with self.assertRaisesRegex(ValueError, "used exactly"):
                validate_initial_load_format(
                    model_config, SimpleNamespace(load_format=load_format)
                )

    def test_registry_selected_model_source_must_be_fingerprinted(self):
        from vllm.entrypoints.serve.middleware.authenticate import (
            AuthenticationMiddleware,
        )
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

        _verify_selected_class_source(Qwen3ForCausalLM)
        with self.assertRaisesRegex(ValueError, "not fingerprinted"):
            _verify_selected_class_source(AuthenticationMiddleware)

    def test_canonical_bitsandbytes_loader_quantizes_nf4(self):
        from vllm_bnb_plugin.bitsandbytes_loader import BitsAndBytesModelLoader

        loader = object.__new__(BitsAndBytesModelLoader)
        loader.pre_quant = False
        loader.load_8bit = False
        model = SimpleNamespace(load_weights=lambda weights: None, packed_modules_mapping={})
        model_config = SimpleNamespace(hf_config=SimpleNamespace())
        loader._verify_model_compatibility(model, model_config)
        self.assertFalse(loader.pre_quant)

        source = textwrap.dedent(
            inspect.getsource(BitsAndBytesModelLoader._unquantized_generator)
        )
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "quantize_4bit"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["quant_type"]), "nf4")
        self.assertIs(ast.literal_eval(keywords["compress_statistics"]), True)
        self.assertEqual(
            loading_plan("bf16", "nf4", "drain")["strategy"],
            "native_components",
        )

    def test_native_constructor_preserves_unquantized_router(self):
        target = object()
        live = Model(
            {"": Component(), "router": Component(), "experts": Component()}
        )
        prototype = Model(
            {
                "": Component(target),
                "router": Component(None),
                "experts": Component(target),
            }
        )
        policy = inspect_policy(
            live, prototype, target, [{"path": "router"}, {"path": "experts"}]
        )
        self.assertFalse(policy["router"])
        self.assertTrue(policy["experts"])

    def test_optional_runtime_initializes_before_meta_only_constructor(self):
        import torch

        live = torch.nn.Module()
        prototype = torch.nn.Module()
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(
                dtype=torch.bfloat16,
                quantization="bitsandbytes",
            ),
            quant_config=object(),
            compilation_config=SimpleNamespace(
                static_forward_context={},
                static_all_moe_layers=[],
            ),
        )
        backend = SimpleNamespace(runner=SimpleNamespace(model=live))

        calls = []
        with (
            patch(
                "dynamic_weight_plugin.compat.verify_bitsandbytes_runtime",
                side_effect=lambda: calls.append("verify"),
            ),
            patch.dict(sys.modules, {"bitsandbytes": object()}),
        ):
            initialize_constructor_runtime(cfg)
        self.assertEqual(calls, ["verify"])

        def initialize(_cfg):
            torch.tensor([1.0], device="cpu")

        with (
            patch(
                "dynamic_weight_plugin.component_policy.initialize_constructor_runtime",
                side_effect=initialize,
            ) as initialized,
            patch(
                "vllm.model_executor.model_loader.utils.initialize_model",
                return_value=prototype,
            ),
        ):
            self.assertEqual(native_policy(backend, cfg, []), {})
        initialized.assert_called_once()

    def test_checkpoint_snapshot_guards_the_complete_inventory(self):
        def create_snapshot(root):
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "tokenizer.json").write_text("{}")
            return CheckpointSnapshot(root, require_protected=False)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot = create_snapshot(root)
            self.assertEqual(
                set(snapshot.digests),
                {"config.json", "model.safetensors", "tokenizer.json"},
            )
            snapshot.check_content()
            (root / "added.json").write_text("{}")
            with self.assertRaises(ValueError):
                snapshot.check()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot = create_snapshot(root)
            (root / "tokenizer.json").unlink()
            with self.assertRaises(ValueError):
                snapshot.check()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot = create_snapshot(root)
            weights = root / "model.safetensors"
            # Keep the unlinked inode allocated so recreation cannot reuse it.
            with weights.open("rb"):
                weights.unlink()
                weights.write_bytes(b"weights")
                with self.assertRaises(ValueError):
                    snapshot.check()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            snapshot = create_snapshot(root)
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                snapshot.check_content()

    def test_checkpoint_snapshot_rejects_symlinked_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "real"
            root.mkdir()
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"weights")
            link = base / "linked"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                CheckpointSnapshot(link, require_protected=False)


if __name__ == "__main__":
    unittest.main()
