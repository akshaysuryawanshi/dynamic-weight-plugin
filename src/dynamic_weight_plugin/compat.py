"""Refuse unreviewed private adapter implementations."""

import hashlib
from functools import lru_cache
from importlib.metadata import distribution, version
import json
from pathlib import Path
import sys

@lru_cache(maxsize=3)
def runtime_profile(vllm_version=None):
    installed = vllm_version or version("vllm")
    path = Path(__file__).with_name(f"compatibility-{installed}.json")
    if not path.is_file():
        raise RuntimeError(f"Unsupported vLLM version {installed}")
    return json.loads(path.read_text())


def validate_runtime_model(model_type, target):
    profile = runtime_profile()
    if profile.get("admission_contract") != "native_components" and profile.get("model_types") and model_type not in profile["model_types"]:
        raise ValueError(f"vLLM {profile['version']} profile does not qualify model type {model_type}")
    if profile.get("representations") and target not in profile["representations"]:
        raise ValueError(f"vLLM {profile['version']} profile does not qualify representation {target}")
    reviewed = (None if profile.get('admission_contract') == 'native_components'
                else profile.get('model_formats', {}).get(model_type))
    if reviewed is not None and target not in reviewed:
        raise ValueError(f'vLLM {profile["version"]} profile does not qualify {model_type}/{target}')


def verify_runtime():
    expected = runtime_profile()
    if list(sys.version_info[:2]) != expected.get("python_minor", [3, 12]):
        raise RuntimeError(f"Expected reviewed Python {expected.get('python_minor', [3, 12])}")
    _verify_distribution("vllm", expected)
    for name, reviewed in expected.get("dependencies", {}).items():
        if version(name) != reviewed:
            raise RuntimeError(f"Expected reviewed {name} {reviewed}, got {version(name)}")


def verify_bitsandbytes_runtime():
    expected = runtime_profile()
    if version("bitsandbytes") != expected["bitsandbytes_version"]:
        raise RuntimeError(f"Expected bitsandbytes {expected['bitsandbytes_version']}")
    _verify_distribution("vllm-bnb-plugin", expected["bitsandbytes_plugin"])


def _verify_distribution(name, expected):
    dist = distribution(name)
    if dist.version != expected["version"]:
        raise RuntimeError(f"Expected {name} {expected['version']}, got {dist.version}")
    for name, digest in expected["files"].items():
        actual = hashlib.sha256(Path(dist.locate_file(name)).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"Unreviewed adapter dependency: {name}")


def verify_lowbit_runtime():
    expected = runtime_profile()
    if "humming" not in expected:
        raise RuntimeError("Low-bit experiments are unavailable on this runtime profile")
    _verify_distribution("humming-kernels", expected["humming"])
