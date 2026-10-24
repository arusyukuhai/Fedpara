# Fedpara automatic genesis + CPU/MPS/CUDA placement repair

## What changed

- The HTML/pywebview GUI **accepts an empty Initial state field**. Clicking Start creates a room-specific `.fedpara-gui/rooms/<room-id>/genesis.fstate` from a trusted packaged Hugging Face model when absent, and then starts the existing Elastic loop. Generation happens in its child Python process: the WebView does not block. The GUI itself opens no listening port.
- The new `train-live --auto-state` CLI flag provides the same initialization. You can also supply `--state-base /path/to/my-base.fstate` with `--auto-state` to generate that file if missing. Existing valid checkpoints are reused, never silently overwritten.
- The new exporter runs on CPU to avoid allocating the complete initial model in accelerator memory. It serializes a data-only `.fstate` with the previous strict validation and size cap.
- The LoRA/LoHA factors are now constructed on their corresponding original parameter device (CPU, MPS or CUDA). A placement check runs before training. Input batches are moved to the input embedding's device (or the first available parameter device) rather than an unverified CLI string. `--device auto` now chooses CUDA, MPS or CPU by local availability.
- Federated genesis identity now hashes tensor **content metadata** (individual tensor hashes, names, dtype and shape, with model/revision), not the ZIP archive bytes. Thus file timestamps and the `peer_id` entry no longer make two identical initial models incompatible. Tensor content mismatch still produces different IDs. Received checkpoints are always fully verified before applying.

## From the desktop GUI

1. Install `requirements-patches.txt` and `requirements-gui.txt`.
2. Run `python -m patchlib.cli gui`.
3. Choose a packaged Hugging Face model, room ID and dataset collection. Save the room settings.
4. Leave **Initial state** empty and select **Start**. Logs show `bootstrap: downloading / loading packaged HF model` and `bootstrap: saved ...` before local training starts. Startup can take time for large models.
5. For future runs, the cached genesis is verified and reused. The UI remains interactive while the worker downloads.

## CLI

```bash
python -m patchlib.cli train-live \
  --room-id YOUR_RANDOM_ROOM_ID \
  --model sshleifer/tiny-gpt2 \
  --room-manifest ./room.json \
  --auto-state --device auto \
  --kind loha --rank 32 --rounds 0
```

`--auto-state` without `--state-base` stores the snapshot under `.fedpara-gui/rooms/<room-id>/genesis.fstate`. This is **local** state; it is not copied through the central Mummy server. It will initially download weights directly from the Hugging Face Hub. If the machine cannot reach the Hub, explicit manual snapshot provisioning remains necessary.

## Multi-peer initial agreement and security

- All participants need **identical initial tensor contents, model revision, adapter rank/layout, and room manifest**. The Elastic protocol checks the same genesis and refuses to merge incompatible updates. A mutable `main` revision is not a suitable long-term commitment: select an immutable model commit hash for reproducible shared genesis. If you change `main` after peers begin training, do **not** reuse an old journal with different weights.
- Automatic import does **not** execute a custom Python factory or `trust_remote_code`; custom model code still requires explicit review and offline sandboxing. A room containing a custom code fingerprint cannot use this automatic importer.
- A corrupt existing checkpoint is rejected, not overwritten. If a journal already references an older ZIP-byte-based genesis ID, the new semantic identity will not match it. Back up previous logs and initialize a new epoch deliberately; don't silently merge incompatible histories.
- **Not yet included:** automatic peer-to-peer retrieval of the initial model shards. Existing P2P shard transfer commands remain available. The generator covers public Hugging Face models and cached files; a P2P bootstrap stream for private models still requires an authenticated, pinned manifest protocol and a seed peer able to serve it.
- **Not yet tested:** real macOS MPS and NVIDIA CUDA execution, model-sharded `device_map` layouts, and full multi-peer negotiation. CPU tests and mock checkpoint export pass. GPU offload and custom Expert Parallelism may require further device-specific code.

## Diagnosing a remaining mixed-device error

If you still see `Expected all tensors to be on the same device`, include the **entire Python traceback** and the model ID. Some customized MoE, `device_map` sharding, or offload kernels move tensors within their own forward pass; that cannot be determined solely from the generic error string.
