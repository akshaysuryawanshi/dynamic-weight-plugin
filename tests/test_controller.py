import unittest
from unittest.mock import patch

from dynamic_weight_plugin.control import ControlError, Controller


class Backend:
    current_format = "bf16"

    def __init__(self):
        self.callback = None
        self.resume_count = 0
        self.recover_count = 0
        self.fenced = None
        self.destructive_started = False
        self.fail_precommit = False
        self.fail_commit = False
        self.slices = []

    def validate(self, target, mode):
        if target not in {"bf16", "fp8"}:
            raise ControlError("invalid_request", "unsupported")
        return {"strategy": "native_components", "scheduling": "between_steps",
                "consistency": "mixed" if mode == "dirty" else "drained"}

    def arm_deadline(self, deadline): self.deadline = deadline
    def clear_deadline(self): self.deadline = None
    def pause(self, callback): self.callback = callback
    def assert_drained(self): return None
    def prepare_layerwise(self, target, check): return target
    def begin_replacement(self, target, dirty=False): self.slices = [{"ready": True}]
    def advance_replacement(self): return self.slices.pop(0)
    def take_candidate(self): return "fp8"

    def precommit(self, candidate):
        if self.fail_precommit:
            raise RuntimeError("pre-publication")

    def commit(self, candidate):
        if self.fail_commit:
            raise RuntimeError("post-publication")
        return {"format": candidate}

    def recover_preparation(self):
        if self.destructive_started:
            raise RuntimeError("rollback unavailable")
        self.recover_count += 1

    def resume(self): self.resume_count += 1
    def fence(self, error): self.fenced = error


class ControllerTests(unittest.TestCase):
    def test_drain_owns_one_mutation_and_retains_only_latest(self):
        backend = Backend()
        controller = Controller(backend)
        first = controller.alter("fp8")
        self.assertEqual(first["state"], "draining")
        with self.assertRaisesRegex(ControlError, "Another alteration"):
            controller.alter("bf16")
        backend.callback()
        status = controller.status()
        self.assertEqual(status["format"], "fp8")
        self.assertEqual(status["operation"]["state"], "completed")
        self.assertIsNone(status["active"])
        second = controller.alter("fp8")
        self.assertEqual(second["state"], "unchanged")
        self.assertEqual(controller.status()["operation"]["operation_id"],
                         second["operation_id"])

    def test_dirty_advances_between_steps_and_records_mixed_generation(self):
        backend = Backend()
        controller = Controller(backend)
        operation = controller.alter("fp8", mode="dirty")
        self.assertEqual(operation["state"], "preparing")
        controller.advance(model_executed=True)
        final = controller.status()["operation"]
        self.assertEqual(final["state"], "completed")
        self.assertEqual(final["preparing_inference_steps"], 1)
        self.assertIn("mixed_weight_generations", final["consistency"])

    def test_deadline_before_publication_rolls_back_and_resumes(self):
        backend = Backend()
        controller = Controller(backend)
        controller.alter("fp8")
        controller.active["deadline_ns"] = 0
        controller.poll_deadline()
        self.assertEqual(controller.status()["operation"]["state"], "timed_out")
        self.assertEqual(backend.recover_count, 1)
        self.assertEqual(backend.resume_count, 1)
        self.assertIsNone(backend.fenced)

    def test_prepublication_failure_rolls_back_but_commit_failure_fences(self):
        backend = Backend()
        backend.fail_precommit = True
        controller = Controller(backend)
        controller.alter("fp8")
        backend.callback()
        self.assertEqual(controller.status()["operation"]["state"], "rolled_back")
        self.assertFalse(controller.status()["restart_required"])

        backend = Backend()
        backend.fail_commit = True
        controller = Controller(backend)
        controller.alter("fp8")
        backend.callback()
        self.assertEqual(controller.status()["operation"]["state"], "failed_unsafe")
        self.assertTrue(controller.status()["restart_required"])
        self.assertIn("post-publication", backend.fenced)


if __name__ == "__main__":
    unittest.main()
