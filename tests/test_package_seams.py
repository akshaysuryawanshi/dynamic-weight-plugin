import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tomllib
import unittest
from unittest.mock import patch

from fastapi import FastAPI
import httpx

import dynamic_weight_plugin
from dynamic_weight_plugin.compat import verify_runtime
from dynamic_weight_plugin import compat
from dynamic_weight_plugin.adapter import EngineBackend
from dynamic_weight_plugin.endpoints import DynamicWeightEndpoints
from dynamic_weight_plugin.plugin import guard
from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler


class Core:
    def __init__(self): self.calls = []

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        if method == "dynamic_weight_status":
            return {"protocol": 1, "controller_id": "controller", "fatal": None,
                    "restart_required": False, "format": "bf16", "operation": None}
        if args[:2] == ("mxfp4", "dirty"):
            return {"control_error": {
                "code": "invalid_request", "message": "mxfp4 only supports drain mode"}}
        return {"operation": {"operation_id": "operation", "state": "completed",
                              "target": args[0], "mode": args[1]}}


class Client:
    def __init__(self): self.engine_core = Core()
    async def check_health(self): return None


class PackageSeamTests(unittest.TestCase):
    def test_successful_bitsandbytes_identity_is_verified_once(self):
        profile = {
            "bitsandbytes": {"version": "0.50.2", "files": {}},
            "bitsandbytes_plugin": {"version": "0.0.3", "files": {}},
        }
        compat.verify_bitsandbytes_runtime.cache_clear()
        try:
            with (patch.object(compat, "runtime_profile", return_value=profile),
                  patch.object(compat, "_verify_distribution") as verify,
                  patch.object(compat, "import_module") as runtime_import):
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
            with (patch.object(compat, "runtime_profile", return_value=profile),
                  patch.object(compat, "_verify_distribution",
                               side_effect=[RuntimeError("changed"), None, None]) as verify,
                  patch.object(compat, "import_module") as runtime_import):
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    compat.verify_bitsandbytes_runtime()
                compat.verify_bitsandbytes_runtime()
            self.assertEqual(verify.call_count, 3)
            runtime_import.assert_called_once_with("bitsandbytes")
        finally:
            compat.verify_bitsandbytes_runtime.cache_clear()

    def test_entry_points_and_exact_runtime_fingerprints(self):
        project = Path(__file__).parents[1]
        data = tomllib.loads((project / "pyproject.toml").read_text())
        self.assertEqual(dynamic_weight_plugin.__version__, "0.1.0a1")
        self.assertEqual(
            data["project"]["entry-points"]["vllm.general_plugins"]["dynamic_weight"],
            "dynamic_weight_plugin.plugin:register")
        self.assertEqual(
            data["project"]["entry-points"]["vllm.endpoint_plugins"]["dynamic_weight_api"],
            "dynamic_weight_plugin.endpoints:DynamicWeightEndpoints")
        from dynamic_weight_plugin.compat import runtime_profile
        authentication_boundary = {
            "vllm/entrypoints/serve/middleware/authenticate.py",
            "vllm/entrypoints/serve/middleware/register.py",
            "vllm/entrypoints/launchers/app.py",
        }
        exception_boundary = {
            "vllm/entrypoints/serve/exception_handling/register.py",
            "vllm/entrypoints/serve/exception_handling/handlers/http.py",
            "vllm/entrypoints/serve/exception_handling/utils.py",
            "vllm/entrypoints/serve/engine/protocol.py",
        }
        for version in ("0.32.0", "0.33.0"):
            self.assertLessEqual(authentication_boundary,
                                 runtime_profile(version)["files"].keys())
            self.assertLessEqual(exception_boundary,
                                 runtime_profile(version)["files"].keys())
        verify_runtime()

    def test_backend_fence_trips_plugin_guard(self):
        class Wakeup:
            def put_nowait(self, value): self.value = value
        backend = object.__new__(EngineBackend)
        backend.engine = SimpleNamespace(input_queue=Wakeup(), process_input_queue_block=True)
        backend.deadline_wakeup = SimpleNamespace(clear=lambda: None)
        backend.pause_lease = None
        EngineBackend.fence(backend, "uncertain publication")
        with self.assertRaisesRegex(RuntimeError, "publication uncertain"):
            guard(lambda self: None)(backend.engine)


class EndpointStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_bearer_auth_runs_before_body_or_utility(self):
        app = FastAPI()
        engine = Client()
        app.state.dynamic_weight_client = engine
        DynamicWeightEndpoints().attach_router(app)
        app.state.args = SimpleNamespace(log_error_stack=False)
        init_exception_handler(app)
        transport = httpx.ASGITransport(app=AuthenticationMiddleware(app, ["secret"]))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            self.assertEqual((await client.get("/v1/dynamic-weight/status")).status_code, 401)
            self.assertEqual((await client.post(
                "/v1/dynamic-weight/alter", headers={"Authorization": "Bearer wrong"},
                json={"target": "fp8"})).status_code, 401)
            self.assertEqual(engine.engine_core.calls, [])

            invalid = await client.post(
                "/v1/dynamic-weight/alter",
                headers={"Authorization": "Bearer secret"},
                json={"target": "fp8", "unexpected": True})
            self.assertEqual(invalid.status_code, 400)
            self.assertEqual(engine.engine_core.calls, [])

            oversized = await client.post(
                "/v1/dynamic-weight/alter",
                headers={"Authorization": "Bearer secret",
                         "Content-Type": "application/json"},
                content=b" " * 9000)
            self.assertEqual(oversized.status_code, 413)
            self.assertEqual(engine.engine_core.calls, [])

            response = await client.post(
                "/v1/dynamic-weight/alter",
                headers={"Authorization": "Bearer secret"},
                json={"target": "fp8", "mode": "drain"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["operation"]["state"], "completed")
            self.assertEqual(engine.engine_core.calls[0][0], "dynamic_weight_alter")

            rejected = await client.post(
                "/v1/dynamic-weight/alter",
                headers={"Authorization": "Bearer secret"},
                json={"target": "mxfp4", "mode": "dirty"})
            self.assertEqual(rejected.status_code, 422)
            detail = json.loads(rejected.json()["error"]["message"])
            self.assertEqual(detail, {
                "code": "invalid_request",
                "message": "mxfp4 only supports drain mode",
            })

    def test_guard_raises_before_and_after_engine_work_when_fenced(self):
        class Engine: pass
        engine = Engine()
        engine._dynamic_weight_fatal = "uncertain"
        called = []
        wrapped = guard(lambda self: called.append(True))
        with self.assertRaisesRegex(RuntimeError, "publication uncertain"):
            wrapped(engine)
        self.assertEqual(called, [])

        engine = Engine()
        def fence_during_step(self):
            self._dynamic_weight_fatal = "commit failure"
        with self.assertRaisesRegex(RuntimeError, "publication uncertain"):
            guard(fence_during_step)(engine)


    async def test_startup_requires_native_api_key_and_loopback(self):
        endpoint = DynamicWeightEndpoints()
        engine = Client()
        state = SimpleNamespace()
        base = dict(api_server_count=1, host="127.0.0.1", api_key=None)
        with patch.dict(os.environ, {
            "VLLM_PLUGINS": "dynamic_weight,dynamic_weight_api",
            "VLLM_API_KEY": "",
        }, clear=False):
            with self.assertRaisesRegex(RuntimeError, "api-key"):
                await endpoint.init_state(engine, state, SimpleNamespace(**base))
        with patch.dict(os.environ, {
            "VLLM_PLUGINS": "dynamic_weight,dynamic_weight_api",
            "VLLM_API_KEY": "secret",
        }, clear=False):
            args = SimpleNamespace(**{**base, "host": "0.0.0.0"})
            with self.assertRaisesRegex(RuntimeError, "loopback"):
                await endpoint.init_state(engine, state, args)
            await endpoint.init_state(engine, state, SimpleNamespace(**base))
        self.assertIs(state.dynamic_weight_client, engine)


if __name__ == "__main__":
    unittest.main()
