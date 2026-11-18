from concurrent.futures import Future
from functools import partial
from queue import Empty, Queue
from time import monotonic_ns
import unittest

from dynamic_weight_plugin.control import ControlError, Controller
from dynamic_weight_plugin.lifecycle import DeadlineWakeup, PauseLease


class Backend:
    current_format = "bf16"

    def __init__(self):
        self.callback = None
        self.resume_count = 0
        self.recover_count = 0
        self.fenced = None
        self.destructive_started = False
        self.publication_count = 0
        self.fail_precommit = False
        self.fail_commit = False
        self.fail_after_publication = False
        self.timeout_after_publication = False
        self.controller = None
        self.slices = []

    def validate(self, target, mode):
        if target not in {"bf16", "fp8"}:
            raise ControlError("invalid_request", "unsupported")
        return {
            "strategy": "native_components",
            "scheduling": "between_steps",
            "consistency": "mixed" if mode == "dirty" else "drained",
        }

    def arm_deadline(self, deadline):
        self.deadline = deadline

    def clear_deadline(self):
        self.deadline = None

    def pause(self, callback):
        self.callback = callback

    def assert_drained(self):
        return None

    def prepare_components(self, target, check):
        if self.fail_after_publication or self.timeout_after_publication:
            self._publish_first_component()
            check()
        return target

    def begin_replacement(self, target, dirty=False):
        self.slices = [{"ready": True}]

    def advance_replacement(self):
        if self.fail_after_publication or self.timeout_after_publication:
            self._publish_first_component()
            return {"ready": False}
        return self.slices.pop(0)

    def _publish_first_component(self):
        self.publication_count += 1
        self.destructive_started = True
        if self.fail_after_publication:
            raise RuntimeError("failure after first component publication")
        self.controller.active.deadline_ns = 0

    def take_candidate(self):
        return "fp8"

    def precommit(self, candidate):
        if self.fail_precommit:
            raise RuntimeError("pre-publication")

    def commit(self, candidate):
        if self.fail_commit:
            raise RuntimeError("post-publication")

    def recover_preparation(self):
        if self.destructive_started:
            raise RuntimeError("rollback unavailable")
        self.recover_count += 1

    def resume(self):
        self.resume_count += 1

    def fence(self, error):
        self.fenced = error


class ControllerTests(unittest.TestCase):
    def test_drain_owns_one_mutation_and_retains_only_latest(self):
        backend = Backend()
        controller = Controller(backend)
        first = controller.alter("fp8")
        self.assertEqual(first["state"], "draining")
        self.assertIs(controller.active, controller.latest)
        self.assertNotIn("finished_ns", first)
        self.assertNotIn("error", first)
        first["state"] = "detached"
        self.assertEqual(controller.status()["active"]["state"], "draining")
        with self.assertRaisesRegex(ControlError, "Another alteration"):
            controller.alter("bf16")
        backend.callback()
        status = controller.status()
        self.assertEqual(status["format"], "fp8")
        self.assertEqual(status["operation"]["state"], "completed")
        self.assertIsNone(status["active"])
        second = controller.alter("fp8")
        self.assertEqual(second["state"], "unchanged")
        self.assertEqual(
            controller.status()["operation"]["operation_id"],
            second["operation_id"],
        )

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
        controller.active.deadline_ns = 0
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
        self.assertEqual(
            controller.status()["operation"]["state"], "failed_unsafe"
        )
        self.assertTrue(controller.status()["restart_required"])
        self.assertIn("post-publication", backend.fenced)

    def test_error_or_timeout_after_publication_always_fences(self):
        for mode in ("drain", "dirty"):
            for failure in ("error", "timeout"):
                with self.subTest(mode=mode, failure=failure):
                    backend = Backend()
                    controller = Controller(backend)
                    backend.controller = controller
                    if failure == "error":
                        backend.fail_after_publication = True
                    else:
                        backend.timeout_after_publication = True

                    controller.alter("fp8", mode=mode)
                    if mode == "drain":
                        backend.callback()
                    else:
                        controller.advance(model_executed=True)

                    status = controller.status()
                    self.assertEqual(
                        status["operation"]["state"], "failed_unsafe"
                    )
                    self.assertTrue(status["restart_required"])
                    self.assertIsNone(status["active"])
                    self.assertIsNotNone(backend.fenced)
                    self.assertEqual(backend.publication_count, 1)
                    self.assertEqual(backend.recover_count, 0)
                    self.assertEqual(backend.resume_count, 0)
                    with self.assertRaisesRegex(
                        ControlError, "Engine requires restart"
                    ):
                        controller.alter("bf16")


class NativeLifecycleTests(unittest.TestCase):
    def test_pause_lease_removes_only_owned_callback_and_cancels_future(self):
        foreign = object()

        class Engine:
            def __init__(self):
                self._idle_state_callbacks = [foreign]
                self.future = Future()

            def pause_scheduler(self, **kwargs):
                owned = partial(lambda *, future: None, future=self.future)
                self._idle_state_callbacks.append(owned)
                return self.future

        engine = Engine()
        lease = PauseLease(engine, lambda error=None: None)
        lease.start()
        self.assertEqual(engine._idle_state_callbacks[0], foreign)
        self.assertEqual(len(engine._idle_state_callbacks), 2)
        lease.close()
        self.assertEqual(engine._idle_state_callbacks, [foreign])
        self.assertTrue(engine.future.cancelled())

    def test_pause_lease_handles_immediate_and_future_completion(self):
        immediate_calls = []

        class ImmediateEngine:
            _idle_state_callbacks = []

            def pause_scheduler(self, **kwargs):
                return None

        lease = PauseLease(
            ImmediateEngine(), lambda error=None: immediate_calls.append(error)
        )
        lease.start()
        self.assertEqual(immediate_calls, [None])
        self.assertFalse(lease.active)

        completed_calls = []

        class FutureEngine:
            def __init__(self):
                self._idle_state_callbacks = []
                self.future = Future()

            def pause_scheduler(self, **kwargs):
                self._idle_state_callbacks.append(
                    partial(lambda *, future: None, future=self.future)
                )
                return self.future

        engine = FutureEngine()
        lease = PauseLease(
            engine, lambda error=None: completed_calls.append(error)
        )
        lease.start()
        engine.future.set_result(None)
        self.assertEqual(completed_calls, [None])
        self.assertEqual(engine._idle_state_callbacks, [])

    def test_deadline_wakeup_fires_and_clear_owns_timer(self):
        queue = Queue()
        wakeup = ("WAKEUP", None)
        deadline = DeadlineWakeup(queue, wakeup)
        deadline.arm(monotonic_ns())
        self.assertEqual(queue.get(timeout=1), wakeup)
        deadline.clear()
        self.assertIsNone(deadline.timer)

        deadline.arm(monotonic_ns() + 5_000_000_000)
        deadline.clear()
        self.assertIsNone(deadline.timer)
        with self.assertRaises(Empty):
            queue.get_nowait()


if __name__ == "__main__":
    unittest.main()
