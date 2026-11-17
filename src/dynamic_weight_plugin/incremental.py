"""Initialize native online layers before their first checkpoint tensor."""


def initialize_online_layers(model):
    from vllm.model_executor.model_loader.reload.layerwise import (
        initialize_online_processing,
    )

    # LinearBase registers bias after quant_method.create_weights(). A bias can
    # precede its first weight in the checkpoint, so wrap late parameters once
    # construction is complete and before any tensor is consumed.
    for layer in model.modules():
        quant_method = getattr(layer, "quant_method", None)
        if getattr(quant_method, "uses_meta_device", False):
            initialize_online_processing(layer)
