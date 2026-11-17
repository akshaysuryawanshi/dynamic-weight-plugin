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
from dynamic_weight_plugin.compat import runtime_profile
from dynamic_weight_plugin.control import ControlError
from dynamic_weight_plugin.policy import loading_plan
from dynamic_weight_plugin.representations import REPRESENTATIONS
from dynamic_weight_plugin.source import CheckpointSnapshot


class Component:
    def __init__(self, quant_config=None, quant_method=None):
        self.quant_config = quant_config
        self.quant_method = quant_method


class Model:
    def __init__(self, modules): self.modules = modules
    def named_modules(self): return self.modules.items()
    def get_submodule(self, name): return self.modules[name]


class NativeContractTests(unittest.TestCase):
    def test_version_profiles_admit_only_declared_core_formats(self):
        expected32 = {"bf16", "fp8", "fp8_per_channel", "fp8_per_block",
                      "mxfp8", "mxfp4", "nf4"}
        expected33 = {"bf16", "fp8", "fp8_per_channel", "fp8_per_block",
                      "mxfp8", "mxfp4", "nf4"}
        self.assertEqual(set(runtime_profile("0.32.0")["representations"]), expected32)
        self.assertEqual(set(runtime_profile("0.33.0")["representations"]), expected33)
        self.assertEqual(set(REPRESENTATIONS), expected32)
        self.assertNotIn("awq_int4", REPRESENTATIONS)
        self.assertNotIn("int2", REPRESENTATIONS)

    def test_block_fp8_uses_native_online_linear_and_moe_methods(self):
        from vllm.config.quantization import _ONLINE_SHORTHANDS
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig,
        )

        config = OnlineQuantizationConfig(_ONLINE_SHORTHANDS["fp8_per_block"])
        linear = config.resolve_quant_method_cls(LinearBase, "component")
        experts = config.resolve_quant_method_cls(RoutedExperts, "component")
        self.assertEqual(linear[-1].__name__, "Fp8PerBlockOnlineLinearMethod")
        self.assertEqual(experts[-1].__name__, "Fp8PerBlockOnlineMoEMethod")
        plan = loading_plan("qwen3", "bf16", "fp8_per_block", "drain")
        self.assertEqual(plan["strategy"], "native_components")
        self.assertEqual(plan["scheduling"], "uninterrupted")

    def test_mx_formats_use_native_online_methods_and_loading_plans(self):
        from vllm.config.quantization import _ONLINE_SHORTHANDS
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
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
            plan = loading_plan("qwen3", "bf16", name, "drain")
            self.assertEqual(plan["strategy"], "native_components")
        dirty = loading_plan("qwen3", "bf16", "mxfp8", "dirty")
        self.assertEqual(dirty["scheduling"], "between_steps")

    def test_representation_modes_reject_dirty_mxfp4_before_loading(self):
        with self.assertRaisesRegex(ControlError, "mxfp4 only supports drain mode"):
            loading_plan("qwen3", "bf16", "mxfp4", "dirty")
        with self.assertRaisesRegex(ControlError, "mxfp4 only supports drain mode"):
            loading_plan("qwen3", "mxfp4", "bf16", "dirty")

    def test_canonical_bitsandbytes_loader_quantizes_nf4(self):
        from vllm_bnb_plugin.bitsandbytes_loader import BitsAndBytesModelLoader

        loader = object.__new__(BitsAndBytesModelLoader)
        loader.pre_quant = False
        loader.load_8bit = False
        model = SimpleNamespace(load_weights=lambda weights: None,
                                packed_modules_mapping={})
        model_config = SimpleNamespace(hf_config=SimpleNamespace())
        loader._verify_model_compatibility(model, model_config)
        self.assertFalse(loader.pre_quant)

        source = textwrap.dedent(
            inspect.getsource(BitsAndBytesModelLoader._unquantized_generator))
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "quantize_4bit"]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["quant_type"]), "nf4")
        self.assertIs(ast.literal_eval(keywords["compress_statistics"]), True)
        plan = loading_plan("qwen3", "bf16", "nf4", "drain")
        self.assertEqual(plan["strategy"], "native_components")

    def test_native_constructor_intent_preserves_unquantized_router(self):
        target = object()
        live = Model({"": Component(), "router": Component(), "experts": Component()})
        prototype = Model({"": Component(target), "router": Component(None),
                           "experts": Component(target)})
        policy = inspect_policy(live, prototype, target,
                                [{"path": "router"}, {"path": "experts"}])
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
        with (patch("dynamic_weight_plugin.compat.verify_bitsandbytes_runtime",
                    side_effect=lambda: calls.append("verify")),
              patch.dict(sys.modules, {"bitsandbytes": object()})):
            initialize_constructor_runtime(cfg)
        self.assertEqual(calls, ["verify"])

        def initialize(_cfg):
            # This reproduces the real CPU NF4 table allocation. It is allowed
            # only because optional runtime initialization precedes MetaOnly.
            torch.tensor([1.0], device="cpu")

        with (patch("dynamic_weight_plugin.component_policy.initialize_constructor_runtime",
                    side_effect=initialize) as initialized,
              patch("vllm.model_executor.model_loader.utils.initialize_model",
                    return_value=prototype)):
            self.assertEqual(native_policy(backend, cfg, []), {})
        initialized.assert_called_once()

    def test_checkpoint_snapshot_hashes_complete_inventory_and_detects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "tokenizer.json").write_text("{}")
            snapshot = CheckpointSnapshot(root, require_protected=False)
            self.assertEqual(set(snapshot.digests),
                             {"config.json", "model.safetensors", "tokenizer.json"})
            snapshot.check_content()
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "content"):
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
