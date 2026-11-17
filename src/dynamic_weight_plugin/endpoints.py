"""Bearer-authenticated HTTP control through vLLM's native /v1 middleware."""

import asyncio
import json
import os
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from .control import (
    DEFAULT_DRAIN_TIMEOUT_MS, DEFAULT_PREPARATION_TIMEOUT_MS,
    MAX_TIMEOUT_MS, PROTOCOL_VERSION,
)

_STARTUP_TIMEOUT_SECONDS = 30.0
_UTILITY_LIMIT = 4
_MAX_BODY_BYTES = 8192
_BODY_TIMEOUT_SECONDS = 5.0
_ERROR_STATUS = {"invalid_request": 422, "operation_conflict": 409,
                 "unavailable": 503}
_OPERATION_STATUS = {"draining": 202, "preparing": 202, "committing": 202,
                     "rolled_back": 409, "timed_out": 409, "failed_unsafe": 503}


class _BoundedRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def dispatch(request):
            if request.method == "POST":
                length = request.headers.get("content-length")
                if length is not None:
                    try:
                        if int(length) > _MAX_BODY_BYTES:
                            raise HTTPException(413, "Request body too large")
                    except ValueError:
                        raise HTTPException(400, "Invalid Content-Length") from None

                async def read_bounded():
                    body = bytearray()
                    async for chunk in request.stream():
                        body.extend(chunk)
                        if len(body) > _MAX_BODY_BYTES:
                            raise HTTPException(413, "Request body too large")
                    request._body = bytes(body)

                await asyncio.wait_for(read_bounded(), timeout=_BODY_TIMEOUT_SECONDS)
            return await handler(request)

        return dispatch


class _UtilityTasks:
    """Own native utility calls after HTTP cancellation or response loss."""

    def __init__(self):
        self.tasks = set()
        self.unavailable = False

    def _done(self, task):
        try:
            if task.cancelled() or task.exception() is not None:
                self.unavailable = True
        except BaseException:
            self.unavailable = True
        finally:
            self.tasks.discard(task)

    async def call(self, engine, method, *args):
        if self.unavailable:
            raise HTTPException(503, "Dynamic-weight utility transport failed; restart API")
        if len(self.tasks) >= _UTILITY_LIMIT:
            raise HTTPException(429, "Too many outstanding dynamic-weight utility calls")

        async def invoke():
            result = await engine.engine_core.call_utility_async(method, *args)
            _validate_result(result, method)
            return result

        task = asyncio.create_task(invoke())
        self.tasks.add(task)
        task.add_done_callback(self._done)
        return await asyncio.shield(task)


def _registry(state):
    registry = getattr(state, "dynamic_weight_utility_tasks", None)
    if registry is None:
        registry = state.dynamic_weight_utility_tasks = _UtilityTasks()
    return registry


def _validate_result(result, method):
    if not isinstance(result, dict):
        raise ValueError("Engine utility result must be a dictionary")
    if "control_error" in result:
        error = result["control_error"]
        if (not isinstance(error, dict) or set(error) != {"code", "message"}
                or error.get("code") not in _ERROR_STATUS
                or not isinstance(error.get("message"), str)
                or len(error["message"]) > 4096):
            raise ValueError("Malformed engine control error")
    elif method == "dynamic_weight_alter":
        operation = result.get("operation")
        states = {"draining", "preparing", "committing", "completed", "unchanged",
                  "rolled_back", "timed_out", "failed_unsafe"}
        if not isinstance(operation, dict) or operation.get("state") not in states:
            raise ValueError("Malformed engine alter result")


async def _call(registry, engine, method, *args):
    if registry.unavailable:
        raise HTTPException(503, "Dynamic-weight utility transport failed; restart API")
    try:
        await engine.check_health()
        result = await registry.call(engine, method, *args)
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as error:
        registry.unavailable = True
        raise HTTPException(
            503, "Engine utility response was lost; inspect status before retrying"
        ) from error
    if "control_error" in result:
        error = result["control_error"]
        detail = json.dumps(error, separators=(",", ":"), ensure_ascii=True)
        raise HTTPException(_ERROR_STATUS[error["code"]], detail)
    return result


class AlterBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target: Literal[
        "bf16", "fp8", "fp8_per_channel", "fp8_per_block", "mxfp8", "mxfp4", "nf4"
    ]
    mode: Literal["drain", "dirty"] = "drain"
    drain_timeout_ms: int = Field(default=DEFAULT_DRAIN_TIMEOUT_MS, ge=1, le=MAX_TIMEOUT_MS)
    preparation_timeout_ms: int = Field(
        default=DEFAULT_PREPARATION_TIMEOUT_MS, ge=1, le=MAX_TIMEOUT_MS)


class DynamicWeightEndpoints:
    name = "dynamic_weight_api"
    required_tasks = ("generate",)

    def attach_router(self, app):
        router = APIRouter(prefix="/v1/dynamic-weight", route_class=_BoundedRoute)
        registry = _registry(app.state)

        @router.get("/status")
        async def status(request: Request):
            return await _call(registry, request.app.state.dynamic_weight_client,
                               "dynamic_weight_status")

        @router.post("/alter")
        async def alter(body: AlterBody, request: Request):
            result = await _call(
                registry, request.app.state.dynamic_weight_client, "dynamic_weight_alter",
                body.target, body.mode, body.drain_timeout_ms, body.preparation_timeout_ms)
            return JSONResponse(
                result, status_code=_OPERATION_STATUS.get(result["operation"]["state"], 200))

        app.include_router(router)

    async def init_state(self, engine_client, state, args):
        from .compat import verify_runtime
        verify_runtime()
        allowed = {name.strip() for name in os.environ.get("VLLM_PLUGINS", "").split(",")}
        if not {"dynamic_weight", "dynamic_weight_api"} <= allowed:
            raise RuntimeError("Enable both dynamic_weight and dynamic_weight_api plugins")
        keys = getattr(args, "api_key", None) or [os.environ.get("VLLM_API_KEY")]
        if (type(keys) is not list or not keys
                or any(type(key) is not str or not key for key in keys)):
            raise RuntimeError("Configure vLLM --api-key or VLLM_API_KEY")
        if getattr(args, "host", None) not in ("127.0.0.1", "::1", "localhost"):
            raise RuntimeError("Dynamic-weight control requires an explicit loopback --host")
        if not hasattr(getattr(engine_client, "engine_core", None), "call_utility_async"):
            raise RuntimeError("Requires the asynchronous multiprocess EngineClient")
        if getattr(args, "api_server_count", 1) not in (None, 1):
            raise RuntimeError("One API server process is supported")
        initial = await asyncio.wait_for(
            engine_client.engine_core.call_utility_async("dynamic_weight_status"),
            timeout=_STARTUP_TIMEOUT_SECONDS)
        if (initial.get("protocol") != PROTOCOL_VERSION or initial.get("fatal")
                or initial.get("restart_required") is not False
                or not isinstance(initial.get("controller_id"), str)):
            raise RuntimeError("Dynamic-weight controller did not initialize safely")
        state.dynamic_weight_client = engine_client
