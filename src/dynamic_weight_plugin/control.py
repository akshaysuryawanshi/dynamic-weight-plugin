"""Engine-thread-owned weight transition state, independent of torch and HTTP."""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from time import monotonic_ns
from uuid import uuid4

PROTOCOL_VERSION = 1
MAX_PREPARATION_SLICES = 1024
DEFAULT_DRAIN_TIMEOUT_MS = 30_000
DEFAULT_PREPARATION_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 3_600_000


class _Absent(Enum):
    VALUE = "absent"


_OMITTED = _Absent.VALUE


class ControlError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)

    def as_dict(self):
        return {"code": self.code, "message": str(self)}


class OperationDeadline(TimeoutError):
    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"{stage} deadline exceeded at an engine safe boundary")


def release_exception_frames(error):
    """Drop loader tracebacks so failures do not retain candidate GPU tensors."""
    pending, visited = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        current.__traceback__ = None
        current.__cause__ = current.__context__ = None


@dataclass
class Operation:
    operation_id: str
    target: str
    mode: str
    state: str
    drain_timeout_ms: int
    preparation_timeout_ms: int
    started_ns: int
    strategy: str | _Absent = _OMITTED
    scheduling: str | _Absent = _OMITTED
    consistency: str | _Absent = _OMITTED
    rollback_available: bool | _Absent = _OMITTED
    finished_ns: int | _Absent = _OMITTED
    generation: int | _Absent = _OMITTED
    error: str | _Absent = _OMITTED
    error_code: str | _Absent = _OMITTED
    deadline_stage: str | _Absent = _OMITTED
    deadline_ns: int | _Absent = _OMITTED
    preparing_inference_steps: int | _Absent = _OMITTED
    drained_ns: int | _Absent = _OMITTED
    slice_count: int = 0
    phase_count: int = 0

    @classmethod
    def start(
        cls,
        target,
        mode,
        drain_timeout_ms,
        preparation_timeout_ms,
        plan,
    ):
        planned = {
            name: plan[name]
            for name in ("strategy", "scheduling", "consistency", "rollback_available")
            if name in plan
        }
        return cls(
            operation_id=str(uuid4()),
            target=target,
            mode=mode,
            state="draining",
            drain_timeout_ms=drain_timeout_ms,
            preparation_timeout_ms=preparation_timeout_ms,
            started_ns=monotonic_ns(),
            **planned,
        )


_SUMMARY_FIELDS = (
    "operation_id",
    "target",
    "mode",
    "state",
    "started_ns",
    "finished_ns",
    "generation",
    "strategy",
    "scheduling",
    "consistency",
    "error",
    "error_code",
    "deadline_stage",
    "deadline_ns",
    "drain_timeout_ms",
    "preparation_timeout_ms",
    "preparing_inference_steps",
    "slice_count",
    "phase_count",
)


def operation_summary(operation):
    if operation is None:
        return None
    return deepcopy(
        {
            name: value
            for name in _SUMMARY_FIELDS
            if (value := getattr(operation, name)) is not _OMITTED
        }
    )


class Controller:
    """Serialize one volatile mutation and retain only its latest summary."""

    def __init__(self, backend):
        self.backend = backend
        self.controller_id = str(uuid4())
        self.generation = 0
        self.current_format = backend.current_format
        self.active = None
        self.latest = None
        self.fatal = None

    def status(self):
        return {
            "protocol": PROTOCOL_VERSION,
            "controller_id": self.controller_id,
            "generation": self.generation,
            "format": self.current_format,
            "restart_required": self.fatal is not None,
            "fatal": self.fatal,
            "active": operation_summary(self.active),
            "operation": operation_summary(self.latest),
        }

    def alter(
        self,
        target,
        mode="drain",
        drain_timeout_ms=DEFAULT_DRAIN_TIMEOUT_MS,
        preparation_timeout_ms=DEFAULT_PREPARATION_TIMEOUT_MS,
    ):
        if self.fatal:
            raise ControlError("unavailable", "Engine requires restart")
        if self.active is not None:
            raise ControlError("operation_conflict", "Another alteration is in progress")
        if type(target) is not str or type(mode) is not str:
            raise ControlError("invalid_request", "target and mode must be strings")
        if mode not in {"drain", "dirty"}:
            raise ControlError("invalid_request", "mode must be drain or dirty")
        for value in (drain_timeout_ms, preparation_timeout_ms):
            if type(value) is not int or not 1 <= value <= MAX_TIMEOUT_MS:
                raise ControlError(
                    "invalid_request", f"timeouts must be integers in 1..{MAX_TIMEOUT_MS} ms"
                )
        plan = self.backend.validate(target, mode)
        operation = Operation.start(
            target,
            mode,
            drain_timeout_ms,
            preparation_timeout_ms,
            plan,
        )
        self.latest = operation
        if target == self.current_format:
            operation.state = "unchanged"
            operation.generation = self.generation
            operation.finished_ns = monotonic_ns()
            return operation_summary(operation)
        self.active = operation
        try:
            if mode == "dirty":
                operation.state = "preparing"
                operation.preparing_inference_steps = 0
                operation.consistency = "mixed_weight_generations_with_existing_kv"
                self._arm_deadline(operation, "preparation", preparation_timeout_ms)
                self.backend.begin_replacement(target, dirty=True)
                self._check_deadline(operation)
            else:
                self._arm_deadline(operation, "drain", drain_timeout_ms)
                self.backend.pause(lambda error=None: self._drained(operation, error))
        except Exception as error:
            self._fail(operation, error)
        return operation_summary(operation)

    def _arm_deadline(self, operation, stage, timeout_ms):
        deadline = monotonic_ns() + timeout_ms * 1_000_000
        operation.deadline_stage = stage
        operation.deadline_ns = deadline
        self.backend.arm_deadline(deadline)

    @staticmethod
    def _check_deadline(operation):
        if monotonic_ns() >= operation.deadline_ns:
            raise OperationDeadline(operation.deadline_stage)

    def poll_deadline(self):
        operation = self.active
        if operation is not None and operation.state in {"draining", "preparing"}:
            try:
                self._check_deadline(operation)
            except OperationDeadline as error:
                self._fail(operation, error)

    def advance(self, model_executed=False):
        self.poll_deadline()
        operation = self.active
        if not operation or operation.mode != "dirty" or operation.state != "preparing":
            return
        try:
            operation.preparing_inference_steps += bool(model_executed)
            result = self.backend.advance_replacement()
            operation.slice_count += 1
            self._check_deadline(operation)
            if operation.slice_count > MAX_PREPARATION_SLICES:
                raise RuntimeError("Preparation exceeded the bounded slice count")
            if result.get("ready"):
                operation.phase_count += 1
                self._drained(operation)
        except Exception as error:
            self._fail(operation, error)

    def _drained(self, operation, error=None):
        if self.active is not operation:
            return
        if error is not None:
            self._fail(operation, error)
            return
        candidate = None
        try:
            self._check_deadline(operation)
            if operation.mode == "dirty":
                candidate = self.backend.take_candidate()
            else:
                self.backend.assert_drained()
                operation.drained_ns = monotonic_ns()
                operation.state = "preparing"
                self._arm_deadline(
                    operation, "preparation", operation.preparation_timeout_ms
                )
                candidate = self.backend.prepare_components(
                    operation.target, lambda: self._check_deadline(operation)
                )
                operation.phase_count += 1
            self._check_deadline(operation)
            self.backend.precommit(candidate)
            self._check_deadline(operation)
            self.backend.clear_deadline()
        except Exception as failure:
            candidate = None
            self._fail(operation, failure)
            return
        try:
            operation.state = "committing"
            self.backend.commit(candidate)
            operation.phase_count += 1
            self.current_format = operation.target
            self.generation += 1
            self.backend.resume()
            operation.state = "completed"
            operation.generation = self.generation
            operation.finished_ns = monotonic_ns()
            self.active = None
        except Exception as failure:
            self._fence(operation, failure)

    def _fail(self, operation, error):
        if self.active is not operation:
            return
        message = str(error)[:2048]
        timed_out = isinstance(error, OperationDeadline)
        if timed_out:
            operation.error_code = "deadline_exceeded"
            operation.deadline_stage = error.stage
        release_exception_frames(error)
        try:
            self.backend.recover_preparation()
            self.backend.resume()
        except Exception as recovery:
            self._fence(operation, RuntimeError(f"{message}; recovery: {recovery}"))
            return
        operation.state = "timed_out" if timed_out else "rolled_back"
        operation.error = message
        operation.finished_ns = monotonic_ns()
        self.active = None

    def _fence(self, operation, error):
        self.fatal = str(error)[:2048]
        operation.state = "failed_unsafe"
        operation.error = self.fatal
        operation.finished_ns = monotonic_ns()
        self.active = None
        self.backend.fence(self.fatal)
