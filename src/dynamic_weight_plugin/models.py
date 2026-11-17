"""Bind the selected native model implementation to the reviewed runtime."""

from dataclasses import dataclass
import hashlib
import inspect
from importlib.metadata import distribution
from pathlib import Path


def _verify_selected_class_source(model_class):
    """Require the registry-selected implementation to have a pinned source file."""
    from .compat import runtime_profile

    source_name = inspect.getsourcefile(model_class)
    if source_name is None:
        raise ValueError("Native model implementation has no reviewable source file")

    distribution_root = Path(distribution("vllm").locate_file("vllm")).resolve()
    source = Path(source_name).resolve(strict=True)
    try:
        relative = source.relative_to(distribution_root)
    except ValueError as exc:
        raise ValueError(
            "Native model implementation is outside the installed vLLM package"
        ) from exc

    manifest_name = (Path("vllm") / relative).as_posix()
    expected = runtime_profile().get("files", {}).get(manifest_name)
    if expected is None:
        raise ValueError(
            f"Native model implementation is not fingerprinted: {manifest_name}"
        )
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Native model implementation differs from its fingerprint: {manifest_name}"
        )


@dataclass(frozen=True)
class ModelAdapter:
    model_type: str
    architecture: str
    formats: tuple[str, ...]

    def check_model(self, model, model_config):
        selected_class = verify_model_config_source(model_config)
        if type(model) is not selected_class:
            raise ValueError("Expected the model implementation selected by the native registry")


def verify_model_config_source(model_config):
    """Bind native registry selection before the first model construction."""
    from vllm.model_executor.model_loader.utils import get_model_architecture

    selected_class, _ = get_model_architecture(model_config)
    _verify_selected_class_source(selected_class)
    return selected_class


def model_adapter(hf_config):
    from .compat import runtime_profile

    architectures = tuple(getattr(hf_config, "architectures", None) or ())
    if not architectures:
        raise ValueError("Native model registry architecture is required")
    profile = runtime_profile()
    return ModelAdapter(
        model_type=getattr(hf_config, "model_type", "unknown"),
        architecture=",".join(architectures),
        formats=tuple(profile["component_formats"]),
    )
