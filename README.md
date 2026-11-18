# Dynamic Weight Plugin

Change a running single-GPU vLLM engine between native weight representations.
Drain mode finishes active requests before replacement. Dirty mode replaces one
complete component between engine steps, so an active request can span weight
generations and retain KV from an older generation.

## Install

Use a separate Python 3.11 environment for each supported vLLM release:

```bash
cd /path/to/dynamic-weight-plugin
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[runtime032]'
```

Use `'.[runtime033]'` for vLLM 0.33.0.

## Start

The source must be an absolute, canonical path to an unquantized safetensors
checkpoint. The selected native model class must have a source fingerprint in
the matching compatibility profile.

```bash
export VLLM_PLUGINS='dynamic_weight,dynamic_weight_api'
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /srv/models/qwen3-0.6b \
  --served-model-name qwen3-weight \
  --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni \
  --no-async-scheduling --no-enable-prefix-caching \
  --no-enable-cumem-allocator --dtype bfloat16 \
  --enforce-eager --compilation-config '{"mode": 0}'
```

The plugin adds no authentication. These control routes are intended for a
trusted operator interface; native vLLM or deployment-level access controls can
be configured separately.

By default, the checkpoint and its ancestors must be protected from writes by
the serving user and other users. The plugin hashes the complete checkpoint
inventory before native loading, rehashes it afterward, and checks the same file
identities around every replacement. Local development can set
`DYNAMIC_WEIGHT_ALLOW_WRITABLE_SOURCE=1`; hashing remains active, but a process
that can rewrite and restore the source is outside that mode's protection.

For NF4, add `bitsandbytes` to `VLLM_PLUGINS`. The 0.33 profile fingerprints the
reviewed bitsandbytes runtime payload and plugin files; the 0.32 profile checks
the bitsandbytes version and fingerprints the plugin files. The native in-flight
loader quantizes the canonical BF16 source with NF4 and compressed statistics.

Some 0.33 CUDA installations select a Humming kernel that compiles through
NVRTC. If `libnvrtc-builtins.so.13.0` is not found, add that environment's
existing `nvidia/cu13/lib` directory to `LD_LIBRARY_PATH` before starting vLLM.

## Use

```bash
curl -sS http://127.0.0.1:8000/v1/dynamic-weight/status

curl -sS http://127.0.0.1:8000/v1/dynamic-weight/alter \
  -H 'Content-Type: application/json' \
  -d '{"target":"fp8","mode":"drain"}'

curl -sS http://127.0.0.1:8000/v1/dynamic-weight/alter \
  -H 'Content-Type: application/json' \
  -d '{"target":"bf16","mode":"dirty","preparation_timeout_ms":120000}'
```

Poll status until `operation.state` is terminal before another alteration. Only
the latest volatile operation is retained. Drain and preparation deadlines are
checked at engine-safe boundaries and do not interrupt a native CUDA call.

## Compatibility

| vLLM | PyTorch | Transformers | Admitted representations |
| --- | --- | --- | --- |
| 0.32.0 | 2.13.0 | 5.19.0 | BF16, FP8 tensor/channel/block, MXFP8, MXFP4 (drain only), NF4 |
| 0.33.0 | 2.14.0 | 5.20.2 | BF16, FP8 tensor/channel/block, MXFP8, MXFP4 (drain only), NF4 |

Both profiles use Python 3.11 and `cuda-bindings==13.4.4`. Admission still
depends on native kernel selection on the active GPU, exact component
construction, memory admission, and a numerical canary. Model configuration and
KV remain BF16; activation arithmetic follows the native recipe.
Startup verifies the exact dependency versions and native interface
fingerprints in the selected profile and rejects differing builds.

The current source passes 30 focused CPU tests on both exact runtime profiles.
Fresh Python 3.11 installations of both complete runtime extras passed dependency
and native fingerprint checks. This structural revision names component-planning
and canary resource phases and uses typed internal records for operations and
replacement preparation; dictionaries remain at the API boundary. The final
November 17 build was also checked on both exact releases with eager,
batch-invariant Qwen3-0.6B on one Ada SM89 GPU and the trusted-local
writable-source setting. Unauthenticated routes worked; invalid-schema and MXFP4
dirty requests returned 422 without changing state; a BF16-to-FP8 drain preserved
an active 384-token response; BF16 dirty restoration interleaved 116 inference
steps; and a restored 32-token output matched the BF16 baseline prefix. Native
checks that predate this structural revision ran on November 16 with the same model and GPU.
Per-tensor FP8 was exercised on both vLLM releases. On 0.33, block FP8 and MXFP8
matched three fresh-native 32-token settled outputs after drain and dirty
switches; MXFP4 Marlin W4A16 matched in drain mode; and
bitsandbytes NF4 matched a fresh native NF4 target. BF16 restorations matched the
BF16 baseline. These checks do not establish throughput, broad model or hardware
coverage, MoE qualification, or equivalence for mixed-generation dirty requests.

## Mechanism and limits

The plugin asks the native constructor which linear and MoE components use the
target quantizer, including routers that intentionally stay unquantized. Before
replacement starts, it checks aliases, checkpoint coverage, conversion
workspace, and activation reserve. Each complete component is loaded from the
original BF16 checkpoint, source-checked, published, and checked for retired
owners before the next engine boundary. After all components are replaced, a
numerical canary gates the final configuration commit. Returning to BF16 reloads
the original tensors; it does not upcast quantized weights.

This is an experimental single-worker, single-GPU V1 plugin. It rejects CUDA
graphs, async scheduling, distributed execution, sleep mode, weight offload,
LoRA, speculative decoding, multimodal serving, custom workers and schedulers,
and remote model code. Dirty mode requires prefix caching to be disabled and can
change the semantics of an active response. Failures before publication attempt
bounded cleanup and resume; failures after publication fence the engine and
require restart.

```bash
python -m unittest discover -s tests -v
```

## License

[MIT](LICENSE)
