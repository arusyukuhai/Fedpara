# Fedpara Universal LoRA/LoHA patches — Mummy + WebRTC

**Update:** Custom Hugging Face Python models, isolated custom factories and P2P full model weights are now supported via `README_CUSTOM_STATE.md`. The restrictions below describe the original, safe-by-default packaged-model mode.


**Status:** A runnable reference implementation and tested PyTorch adapter/patch serializer. It builds on the included Nim/Mummy WebRTC signaling/STUN/TURN project. **Real WebRTC peer-to-peer negotiation and Hugging Face Hub downloads could not be run in this build environment** (the Nim compiler, aiortc, and transformers are not installed here). Python CPU unit/smoke tests passed (`pytest`). Please test against your target architecture before deployment.

## What it does

- Inspects the **actual floating-point `nn.Parameter` tensors** registered on a PyTorch model. It doesn't target a fixed list of Linear names, and includes `Embedding`, tied `lm_head`, Conv1d/2d/3d kernels, normalization parameters, experts, and arbitrary user-defined shapes whenever they are normal PyTorch Parameters.
- For **order-2 or greater tensors where every axis is >= rank**, creates LoRA-style **CP rank-r** factors. For 2-D it is ordinary `A @ B.T`. For 3-D and above it is `sum_r a[:,r] ⊗ b[:,r] ⊗ ...`. This is a defined N-D extension, **not** automatically equivalent to PEFT's LoRA for every architecture.
- In LoHA mode, the update is `(CP_r(A)) ⊙ (CP_r(B)) * alpha/rank`; for matrices this is two element-wise-multiplied low-rank matrices. Initialize the second branch to zero to preserve the initial model output while retaining nonzero gradients.
- For **any tensor with an axis smaller than the chosen rank**, or for a scalar/vector, attaches a **full-size trainable delta**. No such parameter is silently excluded just because it is 1-D or a small 4-D tensor.
- Preserves weight tying by discovering identical parameter objects and registering one shared adapter for their aliases. The original pretrained tensor stays frozen; optimizers touch only adapter factors or dense deltas.
- Produces a `.fpatch` archive: explicit JSON manifest + **safetensors** factor payload (no pickle). Validates model identifier/revision/layout, factor shapes, finite values, peer ID, and version counter before applying.
- **Aggregates reconstructed update tensors**, not raw factor coordinates. With K remote peers it computes `W + (Δ_local + Σ Δ_peer)/(K+1)`. The newest snapshot **replaces** that peer's previous snapshot rather than accumulating duplicates. This is exact for the represented tensors (subject to wire dtype), but it is **not full FedAvg training** and does not reconcile optimizer state.
- Transfers patches over **DTLS-protected, ordered WebRTC DataChannel**, using 16 KiB chunks, backpressure, length checks and SHA-256. Mummy forwards only SDP/ICE. TURN relays ciphertext through coturn when required. Browser exchange demo is at `/patches`.

### Supported and unsupported models

- The **patching mechanism itself** takes *any ordinary PyTorch `nn.Module`*, including modules inside Hugging Face packaged architectures, provided their trainable weights are discoverable floating-point `nn.Parameter`s and their forward pass supports PyTorch autograd and parametrized tensors.
- The CLI loads **packaged, built-in Hugging Face `transformers` AutoModel variants**, including causal LM, MLM, seq2seq LM, classification, image, audio, vision and other task heads via `--auto-class AutoModelFor...`. It **does not** download/execute repositories' custom Python (`trust_remote_code=False`). Repositories with only remote-code implementations will be rejected.
- Generic task-specific **training** requires a model that exposes a scalar differentiable `loss` from `model(**batch)`. For arbitrary modalities, use named batch tensors in `.safetensors` files (`--batches-dir`). Built-in LM / seq2seq / classification text collators are also provided. There is **no principled way to infer supervision, preprocessing, or losses from arbitrary weights**.
- GGUF-only, ONNX-only, TensorFlow-only, Flax-only, MLX-only, compressed/opaque quantized weights, fused kernels bypassing parameter access, certain tied-weight implementations, meta/sharded FSDP/DeepSpeed/offloaded parameters, and non-PyTorch model runtimes are **not automatically supported**. Non-floating parameters are skipped.
- The adapter **materializes the reconstructed full dense delta in forward**. This is not yet optimized for gigantic embeddings, convolution kernels, >100B models, mobile-device training, or massive MoE layers. A dense fallback (axis < rank) can itself be enormous. For those models you need specialized kernels, sharding and a memory/bandwidth-aware policy.
- The Mummy signaling server provides **two clients per room**. `max_peers` allows additive snapshots from other transfer sessions, but a genuine multi-peer gossip overlay, authenticated peer identities, Sybil defenses, Byzantine-robust aggregation, privacy accounting and fully decentralized discovery are **not included**.

## Install

```sh
cd webrtc_mummy
nimble install -y mummy
nim c -d:release --threads:on --mm:orc -r src/main.nim
```

Use Python 3.10+ (3.11/3.12 recommended). For the training/patch agent:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-patches.txt
python -m pytest -q
```

First start the Nim Mummy server in a **separate terminal**, then run the peer agents. The server defaults to `127.0.0.1:8080`; each peer needs a copy of the *same* pretrained checkpoint or a byte-identical base model. For WAN connections configure coturn, HTTPS/WSS and `APP_TOKEN` as described in `README.md`. Keep `APP_TOKEN`, TURN_SECRET and any private training data out of patches.

## One-step test: train a local text model

Create `mytext.txt` with several lines of UTF-8 text. Pick a small pretrained causal LM:

```sh
python -m patchlib.cli train \
  --model sshleifer/tiny-gpt2 --hf-task causal-lm \
  --kind loha --rank 8 --steps 40 --batch-size 2 \
  --text-file mytext.txt --peer-id alice --output alice.fpatch

python -m patchlib.cli inspect alice.fpatch
```

Run the same training command on computer B with `--peer-id bob --output bob.fpatch`. Prefer a pinned `--revision <40-character-commit-sha>` for actual reproducible P2P training. Identical architecture shapes are not enough: the **original model parameters must also match**.

In two shells / computers, use the **same** randomly generated room ID (16–64 ASCII letters/numbers/`_`/`-`):

```sh
# Peer A
python -m patchlib.cli exchange --ws ws://127.0.0.1:8080/ws \
  --room abcdefgh12345678 --send alice.fpatch --recv-dir alice_inbox

# Peer B
python -m patchlib.cli exchange --ws ws://127.0.0.1:8080/ws \
  --room abcdefgh12345678 --send bob.fpatch --recv-dir bob_inbox
```

Each peer writes a `.fpatch` received from the other. On an HTTPS/WSS public server, use a `wss://` endpoint and `--token`. The **browser test** is `http://127.0.0.1:8080/patches`, also interoperable with the Python patch protocol when connected as the other peer.

To materialize a standalone HF model that has the **arithmetic average of Alice and Bob's weight updates**:

```sh
python -m patchlib.cli merge \
  --model sshleifer/tiny-gpt2 --hf-task causal-lm \
  --kind loha --rank 8 \
  --local-patch alice.fpatch --remote-patch alice_inbox/bob_40.fpatch \
  --output-dir merged-model
```

`merge` removes parametrizations, preserving tied original weights. **Do not average LoHA or LoRA factor tensors elementwise**: that is not generally equal to averaging their effective weight updates. The merged checkpoint can be large: it is an ordinary full-size pretrained model, not a small patch.

## Live P2P training

Launch both processes with the same model revision, task, kind, rank, alpha, and room, but different peer IDs and training data. Each publishes complete factor snapshots every 10 steps and replaces the latest patch received from the other node:

```sh
python -m patchlib.cli train-p2p \
  --model sshleifer/tiny-gpt2 --kind loha --rank 8 \
  --peer-id alice --text-file alice.txt --steps 100 \
  --sync-steps 10 --room abcdefgh12345678 \
  --ws ws://127.0.0.1:8080/ws \
  --output alice.fpatch --merged-output alice_merged
```

The command waits for the DataChannel to connect, trains with a local AdamW optimizer, receives patches and applies them between steps, and optionally saves a merged model. It does **not** merge AdamW optimizer states, provide gradient privacy, or guarantee convergence under data heterogeneity, dropped peers or asynchronous convergence. If running in parallel, choose moderate sync intervals, a stable shared base revision, a secure network, and compare validation loss.

## Generic training inputs for non-text models

Supply one or more **safetensors** files in `--batches-dir`. Each file is an already collated **batch**; the tensor keys must match `model(**kwargs)`. Example for a packaged image-classification model:

```
train-batches/
  batch000.safetensors    # pixel_values [B,C,H,W] float32 and labels [B] int64
  batch001.safetensors
```

```sh
python -m patchlib.cli train \
  --model google/vit-base-patch16-224 --hf-task image-classification \
  --batches-dir train-batches --peer-id vision-node \
  --kind lora --rank 8 --steps 50 --output vision.fpatch
```

For multimodal models, pack all required named tensor inputs into each file. If the model does not produce a scalar `loss`, the CLI correctly raises an error rather than silently inventing a training objective. You can still instantiate `UniversalAdapter(any_nn_module, ...)` from Python inside a trusted packaged training application and use `export`, `import_remote`, `load_local`, `merge_into_base` yourself.

## P2P protocol and security

- Signaling is existing `join` / `offer` / `answer` / `candidate` messages, handled by `src/main.nim`. Patch bytes never go through Mummy's WebSocket router.
- The initiating Python client creates an ordered reliable DataChannel named `fedpara-patch`.
- Peer sends JSON `{v:1,type:'start',id,size,sha256}`, binary chunks <=16384 bytes, then JSON `{v:1,type:'end',id}`. SCTP buffering is throttled with `bufferedAmount`; max packet bytes default to 512 MiB per snapshot.
- `.fpatch` = ASCII magic `FPAT1` + big-endian uint32 JSON header length + strict JSON manifest + serialized `safetensors` payload. `safetensors` avoids executable pickle. SHA-256 is a corruption check **not sender authentication**.
- The receiver enforces matching model ID/revision/layout/rank/mode/alpha and factor shapes, rejects NaN and infinity, rejects replayed monotonically numbered snapshots **per active peer**, and caps the number of retained remote peers.
- **Authentication is not solved by the peer_id string, WebRTC encryption, or SHA-256.** `APP_TOKEN` is a shared signaling-room secret, not a per-peer identity. Any authorized/compromised peer can send poisoned gradients or malicious model updates. For production use per-peer public-key signatures, authorization, transport quotas, reputation, and strict parameter/gradient anomaly checks.
- Python aiortc exchanges complete SDP with embedded ICE candidates (non-trickled *outbound*); incoming browser Trickle ICE candidates are accepted. Browser↔browser signaling supports Trickle ICE via Mummy. This difference only affects negotiation timing, not the patch payload format.

## Files

- `patchlib/core.py`: arbitrary tensor LoRA/LoHA, tied weights, snapshot archive, exact update-overlay aggregation, merging.
- `patchlib/p2p.py`: Python aiortc + Mummy signaling + reliable binary DataChannel transport.
- `patchlib/cli.py`: `inspect`, `train`, `train-p2p`, `exchange`, `merge` commands.
- `web/patches.html`: browser patch sender/receiver UI (`/patches`).
- `src/main.nim`: the original signaling/TURN/native WebRTC service, with the additional `/patches` static route.
- `tests/`: CPU unit tests and a full CLI smoke test using a fake *packaged* pretrained model, requiring no network downloads.

**Implementation note:** Rank-r CP factors save parameters (`O(r Σ d_i)`) only when sufficiently smaller than `Π d_i`; the computation still materializes an `O(Π d_i)` update tensor per weight use. Massive networks will need lazy/on-the-fly fused factor kernels, sparse index-aware Embedding updates, checkpointing, optimizer sharding, and possibly factor-only inference kernels.
