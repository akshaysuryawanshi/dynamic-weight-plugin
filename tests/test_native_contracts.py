import json
from pathlib import Path
import tempfile
import unittest

from dynamic_weight_plugin.component_policy import inspect_policy
from dynamic_weight_plugin.compat import runtime_profile
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
        expected33 = {"bf16", "fp8", "fp8_per_channel"}
        self.assertEqual(set(runtime_profile("0.32.0")["representations"]), expected32)
        self.assertEqual(set(runtime_profile("0.33.0")["representations"]), expected33)
        self.assertEqual(set(REPRESENTATIONS), expected32)
        self.assertNotIn("awq_int4", REPRESENTATIONS)
        self.assertNotIn("int2", REPRESENTATIONS)

    def test_native_constructor_intent_preserves_unquantized_router(self):
        target = object()
        live = Model({"": Component(), "router": Component(), "experts": Component()})
        prototype = Model({"": Component(target), "router": Component(None),
                           "experts": Component(target)})
        policy = inspect_policy(live, prototype, target,
                                [{"path": "router"}, {"path": "experts"}])
        self.assertFalse(policy["router"])
        self.assertTrue(policy["experts"])

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
