# Kagami v32 — learned-checkpoint EVAL / secured Transformers model definitions

**Build date:** 2026-10-25. This is a source distribution; the runtime has NOT been tested on a real HF GPU installation in this environment. The new v32 changes are backward compatible with existing v31 rooms **only when the old room's immutable manifest is kept unchanged**. A new code/architecture-mode model requires v32 on all participating peers.

## Launch

```bash
python -m pip install -r requirements-patches.txt -r requirements-gui.txt -r requirements-eval.txt
python kagami.py
```

Install a compatible PyTorch build for your GPU independently. Some dependency extras depend on accelerator and OS: `bitsandbytes` for FP4/NF4/INT8; `torchao>=0.15` for torchao INT4; `accelerate` for large-model dispatch and disk offloading. Transformers >=4.48 is required; for the new TorchAo AOBaseConfig interface, use a current compatible Transformers release. Hugging Face authentication, repo permissions, and tokenizer availability are still required when applicable.

## What NEW MODEL creates

**1. Copy existing model + weights:** A pinned Hugging Face model/revision (`trust_remote_code=False`) is the initial model, with its pretrained tensors. This is the `weights` mode. It does NOT fork the model's Hub repository or automatically publish weights to Hugging Face.

**2. Copy architecture only:** Resolve a Hugging Face `AutoConfig` from a pinned model/revision, then `AutoModel...from_config` initializes a new model with the room's shared seed. No pretrained weights are copied. Use a compatible tokenizer from the same repository. This is `architecture` mode.

**3. Define structure in Python:** Edit or upload a `.py` file containing one `build_config()` function that returns a *supported built-in Transformers Config* constructed only with literal arguments. The source is parsed with Python AST; **it is not executed** by the GUI, the signaling server, or remote peers. The validated JSON config, code SHA-256, shared seed, tokenizer repo/revision and dataset descriptors are pinned to the model manifest. Every peer constructs the same built-in `AutoModelFor*` class locally. This is `code` mode.

Example:

```python
from transformers import GPT2Config

def build_config():
    return GPT2Config(
        vocab_size=50257,
        n_positions=512,
        n_embd=256,
        n_layer=6,
        n_head=8,
    )
```

`build_config()` accepts no arguments and a single return. `import transformers` or explicit built-in Config imports are allowed. No arbitrary imports, loops, side effects, expressions, custom classes, remote code factories, `eval` or `exec` are supported. This intentionally **does not implement fully arbitrary nn.Module / custom Transformers architecture source on peer machines**; claiming that would require a genuine audited untrusted-code isolation architecture. Any offered `.py` content is fingerprinted but peers only use the data-only config. Network participants can join using the pinned manifest **without obtaining or running the original Python file**. The model's tokenizer can be a *different* HF repository/revision, enabling new local model IDs before publication to the Hub.

This approach supports the subset of model architectures offered by installed Transformers' built-in AutoModel mappings and their selected task. A config may be valid syntactically but unsupported by a particular task, tokenizer or library version; Kagami reports those errors instead of silently running another model. Peer environments still need compatible Transformers implementations for identical initialization.

**Security caveat:** fingerprinting Python source only protects identity, not execution safety. The GUI has **no third-party CDN JavaScript** and uses a restrictive Content-Security-Policy. It still uses pywebview's local bridge and its required `unsafe-eval`; do not embed untrusted HTML or navigation inside that privileged view. Do not enable `trust_remote_code` on untrusted Hub repositories.

## Actual inference from learned weights

1. Start training from the TRAINING screen; allow local and Gossip updates to accumulate.
2. In EVAL choose **学習済みモデル** (`trained`), click **現在の重みを保存** while the worker is online. This sends a nonblocking request; completion appears as `EVAL SNAPSHOT COMPLETE` in TRAINING logs. Pending GPU operations and snapshot export may take time.
3. Press **推論**. Kagami locates the newest completed local model checkpoint under `.fedpara-gui/rooms/<room>/checkpoints/<profile>/...shards.json`, reads the matching `models/<profile>/objects/` shards, validates the model/revision and each SHA-256 digest, reconstructs exactly the snapshot's *effective weights including LoHa/LoRA contributions*, then calls Transformers `model.generate`. It never silently falls back to the initial HF weights.
4. Only by explicitly selecting **初期重み** (`base`) does EVAL call `AutoModel...from_pretrained` for original unmodified HF weights. New randomly initialized/code-designed models cannot select this unsupported mode.

Output contains provenance including `checkpoint_seq`, tensor count, scope, precision and generated token count. The snapshot is a snapshot of **the current local peer's view** of converged Gossip updates, not a guaranteed globally-consistent state across asynchronously running peers. At any instant two peers can have different learned weights.

## Memory-adaptive inference and quantization

For small unquantized checkpoints that conservatively fit available RAM, Kagami installs verified learned state directly into a CPU model and optionally moves it to GPU. For large or quantized checkpoints, it converts content-addressed source shards to standard Hugging Face sharded safetensors with **bounded ~1MiB read/write buffers**, a tensor-level SHA-256 check, and an index, without holding the whole state in CPU RAM. Transformers + Accelerate then uses `device_map="auto"` and a temporary offload directory. Conversion runs inside an isolated temporary directory; the model remains open while generation finishes.

FP32 / FP16 / BF16 are available where the device supports them. Optional INT8 and FP4 / NF4 use bitsandbytes, and INT4 can use the torchao `Int4WeightOnlyConfig` backend. Quantization requires suitable hardware, kernels and compatible library versions. Conversion may need an **additional amount of free temporary disk at least comparable to the full unquantized model**, and Accelerator disk offload can require more. A 300B model, unusual MoE router, custom fused operation or sparse memory setup is **not guaranteed to execute** merely by selecting these modes. CPU-only 4-bit backends differ from GPU implementations.

For older checkpoints, a compatible Transformers architecture and matching state-dict names/shapes are required. Missing or extra weight keys, corrupted objects, or unsupported dtypes cause explicit rejection rather than silently restoring random/default values.

## Network / large-swarm improvements

- Code-designed model config and tokenizer provenance are part of the immutable manifest; new joining peers don't fetch code. A mismatching local room cannot silently overwrite an existing room definition.
- Signed historic participant keys are persisted in an incrementally maintained SQLite database (`.trusted.sqlite3`) instead of rewriting a growing JSON registry on every join. Old JSON keys migrate automatically on startup.
- The NETWORK SVG caps the number of visualized neighbors rather than placing thousands of SVG nodes into the GUI; it is explicitly a one-hop view of locally observed connections.
- Inference logs and training logs are read using bounded tail operations, not unbounded full-log reads. Latest checkpoint discovery retains only a bounded set of candidate manifests in memory.
- Training pause/resume keeps verified WebRTC edges and relays running. Network disconnect and closing the GUI are separate actions.

The distributed engine retains prior v29/v30 offload, loss/accuracy telemetry and model-shard checkpoint behaviors. **Actual scaling with thousands of real-world WebRTC peers, real multi-GPU offload, huge MoE models and arbitrary source-defined architectures has NOT been established by these tests.**

## Verification performed in this build

- New v32 unit/integration tests verify three creation modes; local + remote architecture initialization manifest identity; refusal of executable Python constructs; malicious manifest/tokenizer rejection; learned effective tensor restoration with `torch.nn.Module`; corrupt SHA-256 shard rejection; safetensors binary reconstruction using `safetensors.safe_open`; snapshot control; inference output that differs between trained and base scopes.
- Targeted Python suite: 38 passed. Broader suite reached 59% with zero observed failures but was stopped by the execution time limit; **not all tests passed/ran**.
- HTML/JS behavior was rendered in Chromium at 1024×660 and 1366×768 across six tabs and three model creation modes; no document scrolling or browser JS errors were observed.
- The environment lacked `transformers` and `accelerate` and outbound installation was unavailable. The PyTorch and safetensors operations were genuinely exercised, while HF integration was checked through mocks and API-level compatibility review. End-to-end real pretrained HF model training/inference and thousand-peer runs remain unverified.

Existing v31 notes are retained for background, but **the EVAL description in `README_KAGAMI_V31.md` is superseded by this v32 document**.
