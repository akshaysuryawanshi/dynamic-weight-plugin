"""Minimal HTTP control surface for a trusted vLLM deployment."""

import asyncio
import os
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from .control import (
    DEFAULT_DRAIN_TIMEOUT_MS,
    DEFAULT_PREPARATION_TIMEOUT_MS,
    MAX_TIMEOUT_MS,
    PROTOCOL_VERSION,
)

_STARTUP_TIMEOUT_SECONDS = 30.0
_UTILITY_LIMIT = 4
_CONTROL_ERROR_STATUS = {
    "invalid_request": 422,
    "operation_conflict": 409,
    "unavailable": 503,
}
_OPERATION_STATUS = {
    "draining": 202,
    "preparing": 202,
    "committing": 202,
    "rolled_back": 409,
    "timed_out": 409,
    "failed_unsafe": 503,
}


class _DynamicWeightRoute(APIRoute):
    """Keep plugin errors independent of vLLM's global OpenAI handlers."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def dispatch(request):
            try:
                return await handler(request)
            except HTTPException as exc:
                return JSONResponse(
                    {"detail": exc.detail},
                    status_code=exc.status_code,
                    headers=exc.headers,
                )
            except RequestValidationError as exc:
                errors = [
                    {
                        key: value
                        for key, value in error.items()
                        if key in {"type", "loc", "msg"}
                    }
                    for error in exc.errors()
                ]
                return JSONResponse({"detail": errors}, status_code=422)

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
            raise HTTPException(
                503, "Dynamic-weight utility transport failed; restart the API process"
            )
        if len(self.tasks) >= _UTILITY_LIMIT:
            raise HTTPException(429, "Too many outstanding dynamic-weight utility calls")

        async def invoke():
            result = await engine.engine_core.call_utility_async(method, *args)
            _validate_result(result, method)
            return result

        # A disconnected waiter must not cancel a mutation already accepted by
        # the engine. The registry retains and observes the native call.
        task = asyncio.create_task(invoke())
        self.tasks.add(task)
        task.add_done_callback(self._done)
        return await asyncio.shield(task)


def _utility_tasks(state):
    registry = getattr(state, "dynamic_weight_utility_tasks", None)
    if registry is None:
        registry = state.dynamic_weight_utility_tasks = _UtilityTasks()
    return registry


def _client(request):
    engine = getattr(request.app.state, "dynamic_weight_client", None)
    if engine is None:
        raise HTTPException(503, "Dynamic-weight engine is not initialized")
    return engine


def _validate_result(result, method):
    if not isinstance(result, dict):
        raise ValueError("Engine utility result must be a dictionary")
    if "control_error" in result:
        error = result["control_error"]
        if (
            not isinstance(error, dict)
            or set(error) != {"code", "message"}
            or error.get("code") not in _CONTROL_ERROR_STATUS
            or not isinstance(error.get("message"), str)
            or len(error["message"]) > 4096
        ):
            raise ValueError("Malformed engine control error")
    elif method == "dynamic_weight_alter":
        operation = result.get("operation")
        states = {
            "draining",
            "preparing",
            "committing",
            "completed",
            "unchanged",
            "rolled_back",
            "timed_out",
            "failed_unsafe",
        }
        if not isinstance(operation, dict) or operation.get("state") not in states:
            raise ValueError("Malformed engine alter result")


def _validate_initial_status(initial):
    _validate_result(initial, "dynamic_weight_status")
    if (
        initial.get("protocol") != PROTOCOL_VERSION
        or initial.get("fatal") is not None
        or initial.get("restart_required") is not False
        or not isinstance(initial.get("controller_id"), str)
    ):
        raise RuntimeError("Dynamic-weight controller did not initialize safely")


async def _call_utility(registry, engine, method, *args):
    if registry.unavailable:
        raise HTTPException(
            503, "Dynamic-weight utility transport failed; restart the API process"
        )
    try:
        await engine.check_health()
    except Exception as exc:
        registry.unavailable = True
        raise HTTPException(503, "Engine is unavailable") from exc
    try:
        result = await registry.call(engine, method, *args)
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        registry.unavailable = True
        raise HTTPException(
            503,
            "Engine utility response was lost; restart the API process before retrying",
        ) from exc
    if "control_error" in result:
        error = result["control_error"]
        raise HTTPException(_CONTROL_ERROR_STATUS[error["code"]], error)
    return result


class AlterBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target: Literal[
        "bf16",
        "fp8",
        "fp8_per_channel",
        "fp8_per_block",
        "mxfp8",
        "mxfp4",
        "nf4",
    ]
    mode: Literal["drain", "dirty"] = "drain"
    drain_timeout_ms: int = Field(
        default=DEFAULT_DRAIN_TIMEOUT_MS, ge=1, le=MAX_TIMEOUT_MS
    )
    preparation_timeout_ms: int = Field(
        default=DEFAULT_PREPARATION_TIMEOUT_MS, ge=1, le=MAX_TIMEOUT_MS
    )


class DynamicWeightEndpoints:
    name = "dynamic_weight_api"
    required_tasks = ("generate",)

    def attach_router(self, app):
        router = APIRouter(
            prefix="/v1/dynamic-weight", route_class=_DynamicWeightRoute
        )
        registry = _utility_tasks(app.state)

        @router.get("/status")
        async def status(request: Request):
            return await _call_utility(
                registry, _client(request), "dynamic_weight_status"
            )

        @router.post("/alter")
        async def alter(body: AlterBody, request: Request):
            result = await _call_utility(
                registry,
                _client(request),
                "dynamic_weight_alter",
                body.target,
                body.mode,
                body.drain_timeout_ms,
                body.preparation_timeout_ms,
            )
            state = result["operation"]["state"]
            return JSONResponse(result, status_code=_OPERATION_STATUS.get(state, 200))

        app.include_router(router)

    async def init_state(self, engine_client, state, args):
        from .compat import verify_runtime

        verify_runtime()
        allowed = {
            name.strip()
            for name in os.environ.get("VLLM_PLUGINS", "").split(",")
        }
        if not {"dynamic_weight", "dynamic_weight_api"} <= allowed:
            raise RuntimeError(
                "Enable both dynamic_weight and dynamic_weight_api plugins"
            )
        if not hasattr(
            getattr(engine_client, "engine_core", None), "call_utility_async"
        ):
            raise RuntimeError("Requires the asynchronous multiprocess EngineClient")
        if getattr(args, "api_server_count", 1) not in (None, 1):
            raise RuntimeError("One API server process is supported")

        state.dynamic_weight_client = None
        initial = await asyncio.wait_for(
            engine_client.engine_core.call_utility_async("dynamic_weight_status"),
            timeout=_STARTUP_TIMEOUT_SECONDS,
        )
        _validate_initial_status(initial)
        state.dynamic_weight_client = engine_client
