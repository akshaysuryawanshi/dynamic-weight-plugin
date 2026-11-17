# Dynamic Weight Plugin

Change a running single-GPU vLLM engine between native weight representations.

- **Drain:** finish active requests, replace weights, then resume queued work.
- **Dirty:** replace one complete native component between engine steps. Requests
  may span weight generations and retain KV computed by an older generation.

## Admitted compatibility profiles

| vLLM | PyTorch | Transformers | Representations |
| --- | --- | --- | --- |
| 0.32.0 | 2.13.0 | 5.19.0 | BF16, FP8 tensor/channel/block, MXFP8, MXFP4 (drain only), NF4 |
| 0.33.0 | 2.14.0 | 5.20.2 | BF16, FP8 tensor/channel/block, MXFP8, MXFP4 (drain only), NF4 |

Both profiles use Python 3.11 and `cuda-bindings==13.4.4`. These rows describe
admission gates, not validation of every recipe. The current source passed 20
focused CPU tests on each exact profile. Startup verifies the exact vLLM version,
relevant dependency versions, and fingerprints of the native interfaces used by
the plugin. Availability is still decided on the active GPU by native kernel
selection, exact component construction, memory admission, and a numerical
canary. The model configuration and KV cache stay BF16; activation arithmetic
follows the selected native quantization recipe.

An earlier scoped extraction smoke on each runtime used Qwen3-0.6B, an RTX 3000
Ada 8 GiB GPU, and the trusted-local writable-source setting described below.
BF16 to per-tensor FP8 drain preserved an active 384-token request; FP8 to BF16
dirty interleaved 116 inference steps; the settled BF16 32-token output matched
the baseline prefix; and missing or wrong bearer tokens returned 401. This smoke
predates the current optional-format and error-boundary changes.

Additional bounded vLLM 0.33 checks used batch-invariant, eager, single-GPU
execution on the same model and Ada SM89 GPU. Three greedy 32-token prompts were
compared with fresh native targets. Block FP8 and MXFP8 matched settled tokens
and log probabilities after drain and dirty switches and after BF16 restoration.
MXFP4 selected native Marlin W4A16 and matched after drain and BF16 restoration;
dirty entry and exit returned 422 without changing status. NF4 settled outputs
after drain and dirty switches matched a fresh native bitsandbytes target whose
pinned in-flight loader hardcodes NF4; BF16 restorations matched the BF16
baseline. Dirty
requests themselves can span generations and are not equivalence claims. The
tested MXFP8 path was native Marlin W8A16. These checks do not establish quality,
throughput, W8A8, W4A4, MoE, other models, or other hardware.

## Install

```bash
cd /path/to/the/prepared/dynamic-weight-plugin-checkout
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[runtime032]'
```

Use `'.[runtime033]'` in a separate environment for vLLM 0.33.0.

## Start vLLM

Use an absolute, canonical local path to an unquantized safetensors checkpoint.
By default the checkpoint and its ancestors must be protected from writes by the
serving user and other users. The plugin hashes its complete checkpoint inventory
before native loading, rehashes it afterward, and checks the same file identities
around every replacement.

```bash
export VLLM_API_KEY='replace-with-a-strong-random-token'
export VLLM_PLUGINS='dynamic_weight,dynamic_weight_api'
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /srv/models/qwen3-0.6b \
  --served-model-name qwen3-weight \
  --host 127.0.0.1 --port 8000 --api-key "$VLLM_API_KEY" \
  --distributed-executor-backend uni \
  --no-async-scheduling --no-enable-prefix-caching \
  --no-enable-cumem-allocator --dtype bfloat16 \
  --enforce-eager --compilation-config '{"mode": 0}'
```

For NF4 on either runtime, also include `bitsandbytes` in `VLLM_PLUGINS`. A local
development copy can opt out of protected-path enforcement with
`DYNAMIC_WEIGHT_ALLOW_WRITABLE_SOURCE=1`; content is still hashed and stat-checked,
but a process able to rewrite and restore that source is outside this mode's
protection.

On 0.33, NF4 admission fingerprints the complete reviewed bitsandbytes
distribution and plugin payload. On 0.32, it checks the bitsandbytes version and
fingerprints the reviewed plugin files. Both initialize bitsandbytes before the
unchanged meta-only constructor guard. A canonical checkpoint must not declare
prequantized metadata: the native in-flight loader supplies `quant_type='nf4'`
with compressed statistics. The fresh native reference uses vLLM's
`bitsandbytes` quantization and load format.

The native Humming kernel selected by some 0.33 CUDA installations compiles
through NVRTC. If it cannot find `libnvrtc-builtins.so.13.0`, add that Python
environment's existing `nvidia/cu13/lib` directory to `LD_LIBRARY_PATH` before
starting vLLM. This does not require a different kernel or downloaded library.

## Status and alter

The endpoints use vLLM's native bearer-token middleware and require a configured
API key plus an explicit loopback host.

```bash
curl -sS http://127.0.0.1:8000/v1/dynamic-weight/status \
  -H "Authorization: Bearer $VLLM_API_KEY"

curl -sS http://127.0.0.1:8000/v1/dynamic-weight/alter \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"target":"fp8","mode":"drain"}'

curl -sS http://127.0.0.1:8000/v1/dynamic-weight/alter \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"target":"bf16","mode":"dirty","preparation_timeout_ms":120000}'
```

Poll status until `operation.state` is terminal before another alteration. Only
the latest volatile operation is retained. Drain and preparation timeouts are
checked at engine-safe boundaries and do not interrupt a native CUDA call.

## Mechanism and limits

The plugin asks the native model constructor which components intentionally use
the target quantizer, including unquantized routers, then reloads complete linear
or MoE components from the canonical checkpoint. Native aliases, checkpoint
coverage, conversion workspaces, activation reserve, numerical validation, and
retired ownership are checked before publication. Dirty mode requires prefix
caching to be disabled because existing KV may outlive a weight generation, and
it may change the semantics of a response already being generated. Returning to
BF16 reloads the original BF16 checkpoint tensors; it does not upcast quantized
weights.

This is an experimental single-worker, single-GPU V1 plugin. It rejects CUDA
graphs, async scheduling, distributed execution, sleep mode, weight offload,
LoRA, speculative decoding, multimodal serving, custom workers/schedulers, and
remote model code. A failure before publication attempts bounded cleanup and
resume. Once serving weights have changed, recovery is not promised: any failure
fences the engine and requires restart.

```bash
python -m unittest discover -s tests -v
```

## License

[MIT](LICENSE).
