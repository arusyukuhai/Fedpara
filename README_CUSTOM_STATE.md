# Fedpara — custom Hugging Face code + P2P full model parameters

This is an extension to `README_PATCHES.md` and the existing Nim/Mummy signaling server.

**Test status:** 14 CPU Python tests pass, including 4-D layers, tied Embedding/LM
Head, bfloat16/scalars/buffers, safe-archive checks, Ed25519 signatures, a
custom `build_model()` factory and full-state transfer framing. `aiortc`, Nim,
Transformers, Docker, an actual Hugging Face repository and public Internet
WebRTC are **not installed/tested in this environment**. No real connection
across two machines has been verified. This is a reference implementation.

## New capabilities

- **Packaged HF `AutoModel`**: unchanged (no repo Python execution by default).
- **HF with custom Python**: explicit `--trust-remote-code` opt-in, pinned full
  40-character Hub commit, pre-downloaded local repo and offline isolated
  runtime. Repository `modeling_*.py`, custom configs/tokenizers can run
  *only in the offline Docker sandbox under the provided launcher*.
- **Your own Python module**: `--factory-file /model/my_model.py` with a
  `build_model() -> torch.nn.Module` factory, also requiring the offline
  isolated runtime. Its factory source SHA-256 becomes the model revision.
  Custom imports, dependencies, datasets and initial random seeds must still
  be identical across peers; the one-file hash doesn't cover dependencies.
- **Full-weight `.fstate`**: full current model `state_dict`, including
  trainable/frozen weights, Embedding, LM Head, Conv, and persistent buffers.
  With `--local-patch`/`--remote-patch` it serializes the **effective weight**
  instead of adapter internals. Uses a versioned, uncompressed ZIP with a JSON
  manifest and **raw tensor bytes**, never pickle/Python objects.
- **Stream full weights over WebRTC**: `state-exchange` sends 16 KiB chunks via
  the same reliable ordered WebRTC DataChannel (`fedpara-patch`). It performs
  SHA-256 over the entire transfer and stores to a temporary on-disk file,
  then atomically promotes a structurally valid archive. Adjustable cap:
  `--max-state-gb 32` (GiB; increase for big models).
- **Optional Ed25519 signed transfers**: pin the other peer's public key to
  validate the outgoing archive hash/size before receiving it. Applies to
  both `exchange` (patch) and `state-exchange` (full model), and `train-p2p`.
- **Bootstrap from a full-state file**: `--state-base original.fstate` first
  verifies and applies tensors, then binds *new* LoRA/LoHA patches to a
  `fstate-sha256:<digest>` base revision. This avoids accidentally combining
  an old-base patch with a new full-state checkpoint.

**Important scope:** A full-state archive contains tensor values, names,
shapes and types, **not** the model architecture, model source code,
configuration, tokenizer, training dataset, optimizer state or quantizer state.
To import it, the receiving peer must independently have a compatible
`nn.Module` constructor and version-pinned dependencies. Automatic execution
of code received over P2P is **never allowed**. Other frameworks and runtimes
(GGUF, ONNX, native Flax/TensorFlow/MLX) need separate adapters.

## Installation

```bash
pip install -r requirements-patches.txt
python -m pytest -q
# Nim signaling server in another shell
nimble install -y mummy
nim c -d:release --threads:on --mm:orc -r src/main.nim
```

### A. Normal pretrained model: export and P2P-transfer full current weights

```bash
# Use a fixed commit and the same model identifier on both machines.
python -m patchlib.cli state-export \
  --model sshleifer/tiny-gpt2 --hf-task causal-lm \
  --revision <40-hex-commit> --peer-id alice --seq 1 \
  --output alice.fstate

# For an *adapted* model: one option is to train and simultaneously snapshot
python -m patchlib.cli train \
  --model sshleifer/tiny-gpt2 --revision <40-hex-commit> \
  --peer-id alice --kind loha --rank 8 --batches-dir train-batches \
  --steps 20 --output alice.fpatch --full-state-output alice.fstate

python -m patchlib.cli state-inspect alice.fstate
```

Alternatively, turn a previously trained adapter into a whole-state archive:

```bash
python -m patchlib.cli state-export \
  --model sshleifer/tiny-gpt2 --revision <40-hex-commit> \
  --kind loha --rank 8 --peer-id alice --seq 2 \
  --local-patch alice.fpatch --output alice.fstate
```

Generate sender identities *once per peer* (keep `*.key.pem` private):

```bash
python -m patchlib.cli keygen --private alice.key.pem --public alice.pub.pem
python -m patchlib.cli keygen --private bob.key.pem --public bob.pub.pem
```

Distribute public keys **through a trusted separate channel**. Both clients
must use the same randomly generated 16–64 character room ID, point to the
same Mummy signaling URL and have their own `.fstate` or be receive-only.

```bash
# Alice (shell/computer A; Alice trusts Bob's public key)
python -m patchlib.cli state-exchange \
  --ws wss://example.net/ws --room yourLongRandomRoomId1234 \
  --send alice.fstate --recv-dir ./received \
  --signing-key alice.key.pem --peer-public-key bob.pub.pem

# Bob (shell/computer B; Bob trusts Alice's public key)
python -m patchlib.cli state-exchange \
  --ws wss://example.net/ws --room yourLongRandomRoomId1234 \
  --send bob.fstate --recv-dir ./received \
  --signing-key bob.key.pem --peer-public-key alice.pub.pem
```

Receiving without sending is also supported: omit `--send`, but provide
`--peer-public-key` if you require sender authentication. The binary
`.fstate` never travels through the Mummy HTTP server; only SDP/ICE signals do.
Coturn can relay encrypted DataChannel traffic if direct ICE is unavailable.

Validate, then import a received archive into a *fresh* compatible model:

```bash
python -m patchlib.cli state-inspect received/alice_1_....fstate
python -m patchlib.cli state-import \
  --model sshleifer/tiny-gpt2 --revision <40-hex-commit> \
  --state received/alice_1_....fstate --output-dir ./restored
```

If training LoRA/LoHA on top of the **exact same received new base**, both
peers should specify `--state-base` pointing to the identical archive (or
cryptographically identical file) when running `train`, `merge`, or
`train-p2p` with a packaged model. New patches automatically encode its
`fstate-sha256` identity and won't match the original Hub revision.

## B. Custom Hugging Face remote code (isolated, offline)

The HF team's guidance is to review custom Python and use a pinned revision;
`trust_remote_code=True` executes arbitrary repository Python. Do not run it
in the server process or unrestricted alongside credentials. The optional
sandbox here is an **isolation layer**, not a proof of complete security.

Download **without executing Python** on your ordinary workstation:

```bash
python -m patchlib.cli fetch-custom \
  --repo organization/custom-model \
  --revision <40-hex-commit> --output-dir ./downloaded-custom
```

`fetch-custom` excludes common pickle checkpoint extensions; it downloads
source files that you must still review. It does not download dependencies
beyond those already present in the image. Ensure the model uses safetensors.

Build the CPU-only image from this project's root:

```bash
docker build -f sandbox/Dockerfile.offline -t fedpara-untrusted-cpu .
mkdir -p ./private-batches ./isolated-output
```

Run the **entire training/import/export workload** in an offline, read-only,
no-capabilities, non-root Docker container, with only the model repo and
explicit dataset mounted read-only, plus a *dedicated output directory*:

```bash
sandbox/run_untrusted.sh \
  ./downloaded-custom ./private-batches ./isolated-output \
  train --model /model --model-id organization/custom-model \
  --revision <40-hex-commit> --trust-remote-code \
  --hf-task causal-lm --batches-dir /data --peer-id alice \
  --kind loha --rank 8 --steps 20 \
  --output /out/alice.fpatch --full-state-output /out/alice.fstate
```

After this command finishes, use the **ordinary** `exchange` or
`state-exchange` CLI *outside* the sandbox. **Do not give the sandbox access
to the WebRTC/HTTP server's credentials, Docker socket, host filesystem,
GPU devices or network.** The wrapper intentionally disallows `train-p2p`
for untrusted code because it would combine code execution with live network
access. In a distributed training loop, alternate an offline sandboxed
training round and a separate P2P exchange round.

### C. Arbitrary local Python factory

Put your model in `downloaded-custom/my_model.py`:

```python
import torch
from torch import nn

def build_model() -> nn.Module:
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 64)
            self.conv = nn.Conv1d(64, 64, 3)
        def forward(self, **batch):
            ...  # Return a scalar differentiable `loss` in dict or output.loss
    return MyModel()
```

Run `sandbox/run_untrusted.sh` as above but replace `--trust-remote-code`
with `--factory-file /model/my_model.py`; include `--model /model` and
`--model-id some-stable-model-id`. Supply collated `.safetensors` tensors in
`--batches-dir`; generic loss/data preprocessing cannot be inferred.
To restore the model, use the **same** factory file in the sandbox and
`state-import`. Since plain `nn.Module` has no HF `save_pretrained`, the
import command emits `state.safetensors` and needs your factory code to
reconstruct the module next time.

## Security and accuracy caveats

1. The sandbox's Python environment variables are **guards only**, not a
   sandbox by themselves. Isolation is enforced by the Docker configuration.
   If the container engine or kernel is compromised, isolation can fail. A
   dedicated unprivileged machine/VM is preferable for adversarial models.
2. `--peer-public-key` must be pinned out-of-band; trusting a key carried in
   the P2P message would offer no authentication. SHA-256 alone checks only
   corruption. Signatures bind **each transferred archive**, not a long-term
   user account or a revocation system. Signatures do not prevent poisoned
   updates from an *authorized* malicious peer.
3. Received .fstate archive names, shapes, dtypes, lengths, total size and
   per-tensor SHA-256 are validated before changing the recipient's model.
   Floating/complex tensors must be finite. Import still requires adequate
   memory/disk and can partially update if the OS fails after validation;
   load into a fresh process. The wire SHA-256 includes all bytes.
4. `.fstate` snapshots include buffers and floating/int/complex tensors, but
   not arbitrary `get_extra_state()` Python objects, sparse/meta/quantized
   parameters, or native non-PyTorch model formats. Float8 isn't supported in
   the current raw tensor reader. Tensor axes smaller than rank still get a
   full dense trainable delta as described in `README_PATCHES.md`.
5. Large models: raw weights plus snapshots require substantial disk,
   bandwidth, and possibly CPU/GPU memory. Some 300B+ models will exceed
   `--max-state-gb 32`; increase explicitly, and use a sufficiently large
   storage volume. The new `README_SHARDS.md` implementation adds content-addressed storage and resumable two-peer shard transfer. It does not yet provide erasure coding or a multi-peer shard gossip mesh.
6. **Data-only snapshots are not enough to reproduce training** without a
   pinned model implementation, preprocessing, compatible optimizer state,
   dataset version, model configuration, and tensor semantics. A file of the
   same shapes may still represent a functionally incompatible model.
7. The Nim signaling server is still a **two-peer-per-room** signaling design;
   this is not yet a multi-node gossip mesh. For Internet use: HTTPS/WSS,
   coturn, high-entropy room IDs, APP_TOKEN and a server origin policy.
