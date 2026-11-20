# Dynamic Weight Plugin

Change a running vLLM engine's weight precision without reloading the whole model.

- **Drain:** finish active requests, replace weights, then resume queued work.
- **Dirty:** replace one complete component between engine steps. Active requests
  can span weight generations and retain KV generated with earlier weights.

## Compatibility

| vLLM | PyTorch | Transformers | Installation extra |
| --- | --- | --- | --- |
| 0.32.0 | 2.13.0 | 5.19.0 | `runtime032` |
| 0.33.0 | 2.14.0 | 5.20.2 | `runtime033` |

Requires Linux, Python 3.11 and one NVIDIA CUDA GPU. Both profiles use
`cuda-bindings==13.4.4` and admit BF16, FP8 tensor/channel/block, MXFP8, MXFP4 and
NF4, subject to model, kernel and memory checks. MXFP4 requires drain mode in
both directions. KV stays BF16; activation arithmetic follows the native recipe.
Startup checks exact dependencies and fingerprints of supported vLLM internals.

## Install

```bash
git clone https://github.com/akshaysuryawanshi/dynamic-weight-plugin.git
cd dynamic-weight-plugin
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[runtime032]'
```

Use `'.[runtime033]'` in a separate environment for vLLM 0.33.0.

## Start vLLM

Use an absolute, canonical path to a local, unquantized safetensors checkpoint.

```bash
export VLLM_PLUGINS='dynamic_weight,dynamic_weight_api'
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_BATCH_INVARIANT=1

vllm serve /srv/models/qwen3-0.6b \
  --served-model-name qwen3-weight \
  --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni \
  --no-async-scheduling --no-enable-prefix-caching \
  --no-enable-cumem-allocator --dtype bfloat16 \
  --max-model-len 1024 --max-num-seqs 4 \
  --kv-cache-memory-bytes 268435456 \
  --enforce-eager --compilation-config '{"mode": 0}'
```

The checkpoint and its ancestors must be protected from writes by the serving
user and other users. The plugin hashes the checkpoint before and after initial
loading and checks file identities around replacements. Trusted local development
can set `DYNAMIC_WEIGHT_ALLOW_WRITABLE_SOURCE=1`; hashing remains enabled, but
cannot protect against a process that rewrites and restores the source.

For NF4, add `bitsandbytes` to `VLLM_PLUGINS`. If a 0.33 Humming kernel cannot
find `libnvrtc-builtins.so.13.0`, add the environment's existing `nvidia/cu13/lib`
directory to `LD_LIBRARY_PATH`.

## Change precision

```bash
curl -sS http://127.0.0.1:8000/v1/dynamic-weight/status

curl -sS http://127.0.0.1:8000/v1/dynamic-weight/alter \
  -H 'Content-Type: application/json' \
  -d '{"target":"fp8","mode":"drain"}'

# After completion, restore BF16 using dirty mode.
curl -sS http://127.0.0.1:8000/v1/dynamic-weight/alter \
  -H 'Content-Type: application/json' \
  -d '{"target":"bf16","mode":"dirty","preparation_timeout_ms":120000}'
```

Poll status for `operation.state: completed` or `unchanged` before another change.
Only one operation runs at a time. Deadlines are checked between engine steps;
they cannot interrupt a CUDA call. A failure with `restart_required: true`
requires restarting the engine. A lost engine utility response disables both
routes until the API process is restarted.

## Mechanism and limits

In-process hooks coordinate native component loading and replacement between
engine steps; installed vLLM source files stay unchanged. Each replacement uses
the original checkpoint, with source, ownership and memory checks. A numerical
canary gates the final configuration commit. Returning to BF16 reloads original
tensors rather than upcasting quantized weights.

This is an experimental single-worker V1 plugin. CUDA graphs, async scheduling,
distributed execution, sleep mode, offload, LoRA, speculative decoding, multimodal
serving, custom workers/schedulers and remote model code are unsupported.
Dirty mode requires prefix caching disabled. Failure after replacement begins
fences serving and requires restart; there is no complete old model for rollback.

The control routes have no plugin authentication or authorization. Keep them on
a trusted operator interface. Only the latest operation is retained in memory;
there is no durable journal or automatic retry.

## Tests

```bash
python -m unittest discover -s tests -v
```

GPU checks cover BF16/FP8 drain and dirty restoration with Qwen3-0.6B on both
runtimes. Qwen3-32B/H100 drain completed under load; its dirty reverse returned
HTTP 409 and remains unqualified. These checks do not establish broad model,
hardware, output-quality or mixed-generation equivalence guarantees.

## License

[MIT](LICENSE).
