"""Explicit representations; compute and KV dtype remain BF16."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Representation:
    name: str
    quantization: str | None
    shorthand: str | None
    weight_type: str
    scale: str | None
    activation: str
    algorithm: str
    source_kind: str = "canonical"
    group_size: int | None = None
    zero_point: bool = False
    exclusions: tuple[str, ...] = ("embedding", "lm_head")
    modes: tuple[str, ...] = ("drain", "dirty")

    def descriptor(self):
        return asdict(self)


REPRESENTATIONS = {
    "fp8_per_block": Representation("fp8_per_block", "online", "fp8_per_block", "float8_e4m3fn",
                                     "per_128x128_block", "native_kernel_policy", "native_online"),
    "mxfp8": Representation("mxfp8", "online", "mxfp8", "float8_e4m3fn",
                            "per_block_e8m0", "native_kernel_policy", "native_online", group_size=32),
    "mxfp4": Representation("mxfp4", "online", "mxfp4", "fp4_e2m1", "per_block_e8m0",
                            "native_kernel_policy", "native_online", group_size=32, modes=("drain",)),
    "bf16": Representation("bf16", None, None, "bfloat16", None, "bfloat16", "cast"),
    "fp8": Representation("fp8", "online", "fp8_per_tensor", "float8_e4m3fn",
                          "per_tensor", "native_kernel_policy", "native_online"),
    "fp8_per_channel": Representation("fp8_per_channel", "online", "fp8_per_channel",
                                      "float8_e4m3fn", "per_output_channel",
                                      "dynamic_fp8_per_token", "native_online"),
    "nf4": Representation("nf4", "bitsandbytes", None, "nf4", "per_block",
                          "bfloat16", "bitsandbytes", group_size=64),
}


def initial_representation(model_config):
    if model_config.quantization is None:
        return "bf16"
    if model_config.quantization == "bitsandbytes":
        return "nf4"
    from vllm.config.quantization import _ONLINE_SHORTHANDS
    for rep in REPRESENTATIONS.values():
        if rep.shorthand and model_config.quantization in {"online", rep.shorthand}:
            if model_config.quantization_config == _ONLINE_SHORTHANDS[rep.shorthand]:
                return rep.name
    raise ValueError("Initial quantization specification is not a reviewed representation")


def validate_initial_load_format(model_config, load_config):
    """Bind the inferred initial representation to native loader dispatch."""
    initial = initial_representation(model_config)
    load_format = load_config.load_format
    if (initial == "nf4") != (load_format == "bitsandbytes"):
        raise ValueError(
            "bitsandbytes load format must be used exactly when the initial "
            "representation is NF4"
        )
    if initial != "nf4" and load_format not in ("auto", "safetensors"):
        raise ValueError("default safetensors loader required initially")
    return initial
