# Fedpara — MoE expert execution, selective safetensors, quantized bases, paging and custom room code

This is an **experimental extension** to the previous room-ID and elastic P2P code. Do not mistake the modules for an already-complete heterogeneous, fault-tolerant, asynchronous **LoHA + remote-Expert** end-to-end trainer. The existing `train-live` command retains its uniform-layout elastic patch semantics. **It does not automatically distribute different experts among nodes.** The new `ExpertRuntime` is an independently usable, real-WebRTC execution primitive; the `expert-demo-p2p` command exercises remote expert forward/backward, but not a convergent training optimizer.

## Quick checks

```bash
python -m pip install -r requirements-patches.txt
python -m patchlib.cli expert-smoke
python -m patchlib.cli gui
./scripts/check_nim.sh
# or: nim c -d:release --threads:on --mm:orc -r src/main.nim
```

This release locally passed **65 Python tests**; Nim cannot be compiled in the delivery environment, so `scripts/check_nim.sh` is provided to surface your first actual compile error. Obvious ambiguous `=%*` and `=$` expressions were replaced with spaced Nim expressions; optional `rtcNative` library-name declarations and Mummy `newServer` signature compatibility branches were updated. This is not a guarantee of all Nim syntax/semantic issues being fixed.

## A. Real WebRTC expert execution demonstration

Start Mummy, then run this on TWO terminals with the SAME room-ID generated from `room-create`. Both peers must enroll during the same time window:

```bash
python -m patchlib.cli expert-demo-p2p \
  --room-id 'YOUR_24_TO_64_CHAR_RANDOM_ROOM' \
  --ws ws://127.0.0.1:8080/ws --journal ./alice-expert-demo --quant4 --kind loha --rank 8
```

Use a different `--journal` path on the other machine, and WSS for remote signaling. The runtime:

1. Creates private signing identities automatically; the Mummy server is never handed the model's hidden activations.
2. Freezes a weighted-rendezvous mapping across members discovered during the enrollment window.
3. Preserves a model's original top-k gate, replaces non-owner `ModuleList`/Switch-`ModuleDict` expert objects with `RemoteExpert` autograd proxies.
4. Executes forward and backward on the owner over an authenticated aiortc DataChannel. Owner gradients are accumulated on the owner and can be optimized there.
5. Refuses a mismatched generation, malformed tensors, replayed request IDs, disconnected owner or overlarge payloads; it **does not** silently fall back to an unverified stale replica.

**Important:** This sample computes forward/backward only. It is NOT a synchronized global optimizer, patch rebasing protocol, or safe dynamic resharding algorithm. A failed remote expert mid-step must abort that step. To train a model across changing expert placements requires a checkpointed/acknowledged expert owner lease, optimizer-state transfer, and reconciliation with the global elastic journal.

The class-based API is `patchlib.expert_runtime.ExpertRuntime`, `patchlib.expert_parallel.ExpertOwner`, `patchlib.expert_channel.ExpertDataChannelBridge`.

## B. Router support

`expert-discover --model ...` detects conventional unpacked MoE blocks with:

- an `nn.ModuleList` of experts and an `nn.Linear` top-k `gate` (historical Mixtral and Qwen2MoE-style layouts),
- an unpacked DeepSeek-style custom gate containing a 2D `weight` tensor, where the model's existing `forward` computes the top-k indices,
- a Switch-style `nn.ModuleDict` containing `expert_0`, `expert_1`, etc. and a `router.classifier` linear gate.

Runtime dispatch uses the original router's own implementation and substitutes only calls `expert(x) -> Tensor`. Actual architecture variants and versions must be tested individually. Newer Transformers releases with **fused/packed** expert kernels, grouped GEMM, complicated expert arguments or auxiliary routing outputs are **NOT** handled by this generic proxy. Unsupported cases must remain explicit; arbitrary model support cannot be guaranteed by inspecting `nn.Module` alone.

```bash
python -m patchlib.cli expert-discover --model YOUR_PACKAGED_HF_MODEL > layers.json
python -m patchlib.cli expert-checkpoint-inspect --checkpoint /path/to/safetensors-folder
```

For an immutable room epoch generate owner placement with key-derived peer IDs and coarse integer capacities:

`capacity.json`: `{"pYOUR_32_HEX_PEER_ID": 2, "pANOTHER_32_HEX_PEER_ID": 1}` (IDs must be actual values)

```bash
python -m patchlib.cli expert-plan --room-id YOUR_ROOM_ID \
  --layers layers.json --capacity capacity.json --replicas 2 \
  --generation checkpoint-0001 --output expert-plan.json
```

The full placement SHA-256 (not just the membership) MUST match among all participating peers. Mid-round joins/departures must not remap experts without safely transferring weights and optimizer state. There is currently no automatic migration consensus.

## C. Safetensors first-load filtering (unpacked experts)

`patchlib.expert_safetensors.iter_selected_tensors` iterates a local `.safetensors` or HF `model.safetensors.index.json` WITHOUT reading excluded individual experts into RAM. `load_into_meta_model` assigns included tensors to a meta-initialized PyTorch model one tensor at a time. Typical order:

1. Instantiate a trusted packaged architecture with `torch.device('meta')` or a trusted empty-weights factory, without creating full weights.
2. Replace nonlocal `ModuleList` experts with placeholder modules BEFORE loading.
3. Call `load_into_meta_model(model, local_safetensors_dir, selected_owned_experts, device='cpu')` and verify there are no unresolved meta parameters.
4. Launch the `ExpertRuntime` for the fixed plan; it will replace placeholders with remote expert proxies.
5. Explicitly validate tied embedding/LM head parameters and known model-specific buffers. A model that requires custom post-load hooks may need an architecture-specific loader.

Per-expert `.experts.N.` parameter names are recognized. **Packed tensors with multiple experts in one tensor are not selectively unpacked.** No pickle `.bin`, arbitrary Python or auto-downloading happens in this loader.

## D. 4-bit frozen expert bases + LoRA/LoHA

```python
from patchlib.expert_memory import quantize_expert_linears
for local_expert in my_local_experts:
    quantize_expert_linears(local_expert, rank=32, kind='loha')
```

`QuantizedLoHaLinear` packs frozen Linear matrices using groupwise signed 4-bit quantization with fp16 group scales; it trains fp32 LoHA factor tensors. 4-bit weights are dequantized transiently during forward (this saves *storage*, not necessarily peak compute memory). This implementation is **symmetric INT4**, not NF4 and not bitsandbytes. Bias remains frozen. It is not yet a compatible target with the previous `UniversalAdapter`'s *global* `.fpatch` aggregation protocol: expert-only updates need their own checkpoint generation and owner-specific patch merge.

For mainstream Hugging Face models that work with bitsandbytes/QLoRA, prefer their native `BitsAndBytesConfig` and PEFT integration when possible; package support, accelerator and training format vary by architecture.

## E. CPU/NVMe expert offload

`patchlib.expert_memory.ExpertPager` provides explicitly acquired/leased experts:

```python
from patchlib.expert_memory import ExpertPager
pager = ExpertPager({'block.0.expert.7': my_expert}, Path('expert-cold-store'))
pager.page_out('block.0.expert.7', 'nvme')
with pager.lease('block.0.expert.7', device='cpu') as expert:
    output = expert(input_tensor)
# call page_out after backward AND optimizer.step, not during an autograd graph
```

CPU paging uses `module.to('cpu')`. NVMe paging stores safetensors and replaces parameters with meta placeholders, so the **old optimizer references become invalid**. Recreate the optimizer after page-in, or persist and restore optimizer state separately. Never call `page_out` while an autograd graph is active or from a concurrent RPC request. There is currently no automatic LRU eviction or optimizer-state paging policy.

## F. Room creation with custom Python model code (PyTorch, Transformers available)

WebView desktop GUI: `python -m patchlib.cli gui` (after installing `requirements-gui.txt`). Choose **Python model .py/.zip upload** under the room settings, max 256KiB. The GUI stores it locally as `./.fedpara-codes/<sha256>.py` or `.zip` with restricted permissions and automatically copies its path to the manifest form. The **room manifest distributes its SHA-256 and optional HTTPS source reference only**; server does not store or run user Python. Other peers must explicitly acquire, verify and approve the exact source from a trusted channel before executing. This release does not automatically exchange source code over P2P.

CLI alternative:

```bash
python -m patchlib.cli code-register --file ./my_model.py
python -m patchlib.cli room-manifest-create --model custom/my_model \
  --revision custom-v1 --dataset private-dataset \
  --code-file .fedpara-codes/SHA256_FROM_PREVIOUS_COMMAND.py \
  --output room-manifest.json
```

Model source should define `build_model() -> torch.nn.Module`. `import torch`, `from transformers import ...` and other dependencies preinstalled in `sandbox/Dockerfile.offline` are usable. Build the image WITH internet access beforehand, run untrusted source without internet using `sandbox/run_untrusted.sh`. Example:

```bash
docker build -f sandbox/Dockerfile.offline -t fedpara-untrusted-cpu .
mkdir -p train-batches output
sandbox/run_untrusted.sh "$PWD/.fedpara-codes" "$PWD/train-batches" "$PWD/output" \
  train --model custom/my_model --model-id custom/my_model \
  --factory-file "/model/ACTUAL_SHA256.py" \
  --batches-dir /data --peer-id alice --kind loha --rank 32 \
  --steps 10 --output /out/alice.fpatch
```

The image includes torch, Transformers, safetensors, accelerate, huggingface_hub; the network stays disabled at runtime, and the script receives only deliberately mounted input/model/output paths. Container isolation is meaningful but not a mathematically perfect sandbox. Review untrusted code and do not mount secrets. **Network-enabled `train-live` deliberately refuses arbitrary Python, including custom uploaded source**; do not bypass this by setting the runtime flag manually.

## G. Known major limitations

- True end-to-end **Elastic global LoHA + heterogeneous distributed Expert parallel training** is not integrated. Remote forward/backward calls are usable and demonstrated; expert-owner optimizer synchronization, consistent rebalancing on departures, owner checkpoint transfer and consensus are outstanding.
- No general automatic plugin capable of transforming **every** Hugging Face model into a safe PyTorch MoE. Fused kernels/unique routers require model-specific adapters.
- 4-bit bases and ExpertPager are separate opt-in components. They are not automatically selected by the existing adaptive policy and cannot be silently mixed into existing uniform LoHA `.fpatch` streams.
- No 100% guarantee of model correctness or reproducibility with dropout/stochastic MoE routing across uncoordinated peers. Guard generation IDs, tensor shapes and checkpoint epochs.
- Nim compiler and `aiortc` were unavailable when building this ZIP, so the real network demo and Mummy build remain **UNVERIFIED**. See `scripts/check_nim.sh`; attach the compiler's *first actual error line* to request a precise follow-up fix.
