"""Reviewed native model-loading contracts, independent of representation."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ModelAdapter:
    model_type: str
    architecture: str
    module: str
    remap_method: str | None = None
    serving: bool = True
    mapper: bool = False
    prefix_transform: str | None = None
    formats: tuple[str, ...] = ()

    native_architectures: tuple[str, ...] = ()

    def check_model(self, model, model_config=None):
        if model_config is not None:
            from vllm.model_executor.model_loader.utils import get_model_architecture
            cls, _ = get_model_architecture(model_config)
            if type(model) is not cls:
                raise ValueError('Expected the model implementation selected by the native registry')
            return
        raise ValueError('Native registry validation requires the full model configuration')

    def map_weight(self, model, item):
        # Run inside the same scoped config/dtype context as the native loader.
        return getattr(model, self.remap_method)(*item) if self.remap_method else item

    def load_items(self, loader, model, items):
        if self.prefix_transform:
            transform = getattr(import_module("vllm.model_executor.models." + self.module), self.prefix_transform)
            items = transform(items)
        items = (self.map_weight(model, item) for item in items)
        mapper = model.hf_to_vllm_mapper if self.mapper else None
        return loader.load_weights(items, mapper=mapper)


def model_adapter(hf_config):
    from .compat import runtime_profile
    profile = runtime_profile()
    architectures = tuple(getattr(hf_config, 'architectures', None) or ())
    if not architectures:
        raise ValueError('Native model registry architecture is required')
    return ModelAdapter(
        getattr(hf_config, 'model_type', 'unknown'), ','.join(architectures), '',
        formats=tuple(profile['component_formats']), native_architectures=architectures,
    )
