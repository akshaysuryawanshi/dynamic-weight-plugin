"""Structured configuration findings and their existing presentation."""

from dataclasses import dataclass
from enum import StrEnum


class ConfigIssueCode(StrEnum):
    DEVELOPMENT_ROUTES = "development_routes"
    EXECUTOR = "executor"
    PARALLEL_SIZE = "parallel_size"
    FAULT_TOLERANCE_OR_UBATCHING = "fault_tolerance_or_ubatching"
    RUNNER_OR_SCHEDULING = "runner_or_scheduling"
    REMOTE_CODE = "remote_code"
    PROMPT_EMBEDS = "prompt_embeds"
    MULTIMODAL = "multimodal"
    OTLP_TRACING = "otlp_tracing"
    EAGER_EXECUTION = "eager_execution"
    COMPILATION = "compilation"
    MODEL_DTYPE = "model_dtype"
    REPRESENTATION = "representation"
    QUANTIZED_CHECKPOINT = "quantized_checkpoint"
    CHECKPOINT_DIRECTORY = "checkpoint_directory"
    ALLOCATOR_POOLS = "allocator_pools"
    KV_DTYPE = "kv_dtype"
    CUSTOM_RUNTIME = "custom_runtime"
    UNSUPPORTED_FEATURE = "unsupported_feature"
    AUX_OUTPUT = "aux_output"
    WEIGHT_OFFLOAD = "weight_offload"
    DEVICE = "device"


@dataclass(frozen=True)
class ConfigIssue:
    code: ConfigIssueCode
    field: str = ""
    actual: object = None
    expected: object = None
    detail: str = ""

    @property
    def message(self):
        return _MESSAGES[self.code].format(field=self.field, detail=self.detail)


_MESSAGES = {
    ConfigIssueCode.DEVELOPMENT_ROUTES: "development RPC routes must be disabled",
    ConfigIssueCode.EXECUTOR: "--distributed-executor-backend uni required",
    ConfigIssueCode.PARALLEL_SIZE: "{field} must equal 1",
    ConfigIssueCode.FAULT_TOLERANCE_OR_UBATCHING: (
        "fault tolerance and ubatching unsupported"
    ),
    ConfigIssueCode.RUNNER_OR_SCHEDULING: (
        "V1 runner with --no-async-scheduling required"
    ),
    ConfigIssueCode.REMOTE_CODE: "remote model/tokenizer code must be disabled",
    ConfigIssueCode.PROMPT_EMBEDS: (
        "prompt embeddings must be disabled on the reviewed runtime"
    ),
    ConfigIssueCode.MULTIMODAL: (
        "multimodal serving is outside the reviewed safety contract"
    ),
    ConfigIssueCode.OTLP_TRACING: (
        "OTLP tracing is outside the reviewed safety contract"
    ),
    ConfigIssueCode.EAGER_EXECUTION: "--enforce-eager required",
    ConfigIssueCode.COMPILATION: "compilation must be disabled",
    ConfigIssueCode.MODEL_DTYPE: "BF16 model configuration required",
    ConfigIssueCode.REPRESENTATION: "{detail}",
    ConfigIssueCode.QUANTIZED_CHECKPOINT: "canonical checkpoint must be unquantized",
    ConfigIssueCode.CHECKPOINT_DIRECTORY: (
        "an immutable local checkpoint directory is required"
    ),
    ConfigIssueCode.ALLOCATOR_POOLS: "sleep/CuMem allocator unsupported",
    ConfigIssueCode.KV_DTYPE: "BF16 KV required",
    ConfigIssueCode.CUSTOM_RUNTIME: "custom schedulers/worker extensions unsupported",
    ConfigIssueCode.UNSUPPORTED_FEATURE: "{field} unsupported",
    ConfigIssueCode.AUX_OUTPUT: "auxiliary output connectors unsupported",
    ConfigIssueCode.WEIGHT_OFFLOAD: "weight offload unsupported",
    ConfigIssueCode.DEVICE: "CUDA required",
}
