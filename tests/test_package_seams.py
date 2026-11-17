import asyncio
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from fastapi import FastAPI
import httpx

import dynamic_weight_plugin
from dynamic_weight_plugin import compat
from dynamic_weight_plugin.adapter import EngineBackend
from dynamic_weight_plugin.compat import verify_runtime
from dynamic_weight_plugin.endpoints import (
    DynamicWeightEndpoints,
    _UtilityTasks,
    _call_utility,
)
from dynamic_weight_plugin.plugin import guard, verified_load
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler


class Core:
    def __init__(self):
        self.calls = []

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        if method == "dynamic_weight_status":
            return {
                "protocol": 1,
                "controller_id": "controller",
                "fatal": None,
                "restart_required": False,
                "format": "bf16",
                "operation": None,
            }
        if args[:2] == ("mxfp4", "dirty"):
            return {
                "control_error": {
                    "code": "invalid_request",
                    "message": "mxfp4 only supports drain mode",
                }
            }
        return {
            "operation": {
                "operation_id": "operation",
                "state": "completed",
                "target": args[0],
                "mode": args[1],
            }
        }


class Client:
    def __init__(self, core=None):
        self.engine_core = core or Core()

    async def check_health(self):
        return None


class PackageSeamTests(unittest.TestCase):
    def test_successful_bitsandbytes_identity_is_cached_after_import(self):
        profile = {
            "bitsandbytes": {"version": "0.50.2", "files": {}},
            "bitsandbytes_plugin": {"version": "0.0.3", "files": {}},
        }
        compat.verify_bitsandbytes_runtime.cache_clear()
        try:
            with (
                patch.object(compat, "runtime_profile", return_value=profile),
                patch.object(compat, "_verify_distribution") as verify,
                patch.object(compat, "import_module") as runtime_import,
            ):
                compat.verify_bitsandbytes_runtime()
                compat.verify_bitsandbytes_runtime()
            self.assertEqual(verify.call_count, 2)
            runtime_import.assert_called_once_with("bitsandbytes")
        finally:
            compat.verify_bitsandbytes_runtime.cache_clear()

    def test_failed_bitsandbytes_identity_is_retried(self):
        profile = {
            "bitsandbytes": {"version": "0.50.2", "files": {}},
            "bitsandbytes_plugin": {"version": "0.0.3", "files": {}},
        }
        compat.verify_bitsandbytes_runtime.cache_clear()
        try:
            with (
                patch.object(compat, "runtime_profile", return_value=profile),
                patch.object(
                    compat,
                    "_verify_distribution",
                    side_effect=[RuntimeError("changed"), None, None],
                ) as verify,
                patch.object(compat, "import_module") as runtime_import,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    compat.verify_bitsandbytes_runtime()
                compat.verify_bitsandbytes_runtime()
            self.assertEqual(verify.call_count, 3)
            runtime_import.assert_called_once_with("bitsandbytes")
        finally:
            compat.verify_bitsandbytes_runtime.cache_clear()

    def test_distribution_identity_ignores_installer_state_but_binds_payload(self):
        class Distribution:
            version = "1.0"

            def __init__(self, root):
                self.root = root

            def locate_file(self, name):
                return self.root / name

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "package/runtime.py"
            library = root / "package/libcuda.so"
            installer = root / "package-1.0.dist-info/INSTALLER"
            for path in (runtime, library, installer):
                path.parent.mkdir(parents=True, exist_ok=True)
            runtime.write_bytes(b"runtime")
            library.write_bytes(b"native")
            installer.write_text("pip\n")
            expected = {
                "version": "1.0",
                "files": {
                    "package/runtime.py": hashlib.sha256(b"runtime").hexdigest(),
                    "package/libcuda.so": hashlib.sha256(b"native").hexdigest(),
                },
            }
            with patch.object(
                compat, "distribution", return_value=Distribution(root)
            ):
                compat._verify_distribution("package", expected)
                runtime.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "runtime.py"):
                    compat._verify_distribution("package", expected)
                runtime.write_bytes(b"runtime")
                library.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "libcuda.so"):
                    compat._verify_distribution("package", expected)

    def test_entry_points_and_current_runtime_boundaries(self):
        project = Path(__file__).parents[1]
        data = tomllib.loads((project / "pyproject.toml").read_text())
        self.assertEqual(dynamic_weight_plugin.__version__, "0.1.0a1")
        self.assertEqual(
            data["project"]["entry-points"]["vllm.general_plugins"][
                "dynamic_weight"
            ],
            "dynamic_weight_plugin.plugin:register",
        )
        self.assertEqual(
            data["project"]["entry-points"]["vllm.endpoint_plugins"][
                "dynamic_weight_api"
            ],
            "dynamic_weight_plugin.endpoints:DynamicWeightEndpoints",
        )
        removed = {
            "vllm/entrypoints/serve/middleware/authenticate.py",
            "vllm/entrypoints/serve/middleware/register.py",
            "vllm/entrypoints/serve/exception_handling/handlers/http.py",
        }
        required = {
            "vllm/entrypoints/launchers/app.py",
            "vllm/plugins/endpoint_plugins/interface.py",
            "vllm/v1/engine/core.py",
            "vllm/v1/engine/core_client.py",
        }
        for version in ("0.32.0", "0.33.0"):
            files = compat.runtime_profile(version)["files"]
            self.assertTrue(required <= files.keys())
            self.assertTrue(removed.isdisjoint(files))
        verify_runtime()

    def test_backend_fence_trips_plugin_guard(self):
        class Wakeup:
            def put_nowait(self, value):
                self.value = value

        backend = object.__new__(EngineBackend)
        backend.engine = SimpleNamespace(
            input_queue=Wakeup(), process_input_queue_block=True
        )
        backend.deadline_wakeup = SimpleNamespace(clear=lambda: None)
        backend.pause_lease = None
        EngineBackend.fence(backend, "uncertain publication")
        with self.assertRaisesRegex(RuntimeError, "publication uncertain"):
            guard(lambda self: None)(backend.engine)

    def test_verified_load_binds_model_source_before_native_load(self):
        events = []

        class Snapshot:
            def __init__(self, path, require_protected=True):
                events.append("snapshot")

            def check_content(self):
                events.append("content")

        runner = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(quantization=None, model="/model"),
                load_config=SimpleNamespace(load_format="auto"),
            )
        )

        def native_load(self):
            events.append("native")
            return "model"

        wrapped = verified_load(native_load)
        with (
            patch(
                "dynamic_weight_plugin.models.verify_model_config_source",
                side_effect=lambda config: events.append("verify"),
            ),
            patch("dynamic_weight_plugin.source.CheckpointSnapshot", Snapshot),
        ):
            self.assertEqual(wrapped(runner), "model")
        self.assertLess(events.index("verify"), events.index("native"))

        runner = SimpleNamespace(vllm_config=runner.vllm_config)
        events.clear()
        with (
            patch(
                "dynamic_weight_plugin.models.verify_model_config_source",
                side_effect=ValueError("unreviewed model source"),
            ),
            patch("dynamic_weight_plugin.source.CheckpointSnapshot", Snapshot),
        ):
            with self.assertRaisesRegex(ValueError, "unreviewed model source"):
                wrapped(runner)
        self.assertEqual(events, [])


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_auth_startup_status_and_mutation(self):
        endpoint = DynamicWeightEndpoints()
        engine = Client()
        state = SimpleNamespace()
        with patch.dict(
            os.environ,
            {"VLLM_PLUGINS": "dynamic_weight,dynamic_weight_api"},
            clear=False,
        ):
            await endpoint.init_state(
                engine, state, SimpleNamespace(api_server_count=1)
            )
        self.assertIs(state.dynamic_weight_client, engine)

        app = FastAPI()
        app.state.args = SimpleNamespace(log_error_stack=False)
        app.state.dynamic_weight_client = engine
        endpoint.attach_router(app)
        init_exception_handler(app)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            status = await client.get("/v1/dynamic-weight/status")
            self.assertEqual(status.status_code, 200)
            altered = await client.post(
                "/v1/dynamic-weight/alter",
                json={"target": "fp8", "mode": "drain"},
            )
            self.assertEqual(altered.status_code, 200)
            self.assertEqual(altered.json()["operation"]["state"], "completed")

    async def test_validation_and_control_errors_are_route_local(self):
        app = FastAPI()
        app.state.args = SimpleNamespace(log_error_stack=False)
        engine = Client()
        app.state.dynamic_weight_client = engine
        DynamicWeightEndpoints().attach_router(app)
        init_exception_handler(app)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            invalid = await client.post(
                "/v1/dynamic-weight/alter",
                json={"target": "fp8", "unexpected": True},
            )
            self.assertEqual(invalid.status_code, 422)
            self.assertEqual(
                invalid.json()["detail"][0]["loc"], ["body", "unexpected"]
            )
            self.assertEqual(engine.engine_core.calls, [])

            rejected = await client.post(
                "/v1/dynamic-weight/alter",
                json={"target": "mxfp4", "mode": "dirty"},
            )
            self.assertEqual(rejected.status_code, 422)
            self.assertEqual(
                rejected.json()["detail"],
                {
                    "code": "invalid_request",
                    "message": "mxfp4 only supports drain mode",
                },
            )

    async def test_response_cancellation_leaves_utility_call_owned(self):
        class SlowCore:
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.completed = False

            async def call_utility_async(self, method, *args):
                self.started.set()
                await self.release.wait()
                self.completed = True
                return {"operation": {"state": "completed"}}

        core = SlowCore()
        engine = Client(core)
        registry = _UtilityTasks()
        waiter = asyncio.create_task(
            _call_utility(
                registry,
                engine,
                "dynamic_weight_alter",
                "fp8",
                "drain",
                1,
                1,
            )
        )
        await core.started.wait()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(len(registry.tasks), 1)
        core.release.set()
        for _ in range(10):
            if not registry.tasks:
                break
            await asyncio.sleep(0)
        self.assertTrue(core.completed)
        self.assertFalse(registry.unavailable)
        self.assertEqual(registry.tasks, set())

    def test_guard_checks_fence_before_and_after_engine_work(self):
        class Engine:
            pass

        engine = Engine()
        engine._dynamic_weight_fatal = "uncertain"
        called = []
        with self.assertRaisesRegex(RuntimeError, "publication uncertain"):
            guard(lambda self: called.append(True))(engine)
        self.assertEqual(called, [])

        engine = Engine()

        def fence_during_step(self):
            self._dynamic_weight_fatal = "commit failure"

        with self.assertRaisesRegex(RuntimeError, "publication uncertain"):
            guard(fence_during_step)(engine)


if __name__ == "__main__":
    unittest.main()
