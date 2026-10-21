# Fedpara sharded P2P model states

This extends the existing Nim/Mummy signaling + WebRTC DataChannel implementation. `patchlib.shards` handles content-addressed tensor shards; `patchlib.shardp2p` transfers **only missing shards**. LoRA/LoHA `.fpatch` remains unchanged. Shards cover *every serializable PyTorch parameter and persistent buffer*, including Embedding, LM Head, 3D/4D convolutions, and non-floating buffers. It is compatible with HF `AutoModel*` and isolated `--factory-file` loaders.

## What was implemented

- Tensor-aligned chunking (default **4 MiB**, configurable 1–64 MiB in the CLI). Changing one tensor does not shift chunk boundaries for others.
- Content-addressed object files at `STORE/objects/ab/<64-character-SHA256>.shard`. Identical chunks are saved once, even across different models/snapshots (local disk deduplication).
- A small, data-only `.fmanifest` containing model identifier, revision, peer ID, sequence, tensor names, shapes, dtypes, tensor hashes, and ordered shard hashes.
- Direct `nn.Module` -> shard export, including *effective* LoRA/LoHA weights, avoiding the full `.fstate` intermediate.
- Existing `.fstate` -> shards conversion, and shards -> fully verified `.fstate` reconstruction (usable with `state-import` / `--state-base`).
- WebRTC DataChannel manifest announcement, bounded 4-shard request window, 16 KiB wire frames, sender backpressure, streaming-to-disk reception, per-shard SHA-256 verification, and **resumability across reconnects**: complete cached shards survive disconnect and are not retransmitted.
- Optional Ed25519 sender authentication: a signed manifest commits to all tensor/shard hashes; *pin* the public key independently. A receiver with `--peer-public-key` rejects unsigned/invalid manifest announcements.
- Model identity/revision checks for two-way transfers. When receiving without a local manifest, choose trusted sender credentials explicitly.

This is **transport sharding**, not tensor-parallel compute, training aggregation, fault-tolerant erasure coding, or distributed storage replication. A node still requires enough space to store the shards it requests. It is currently **two peers per Mummy room**, not multi-source swarm downloading.

## Setup

```bash
python -m pip install -r requirements-patches.txt
# In a separate shell, start Nim/Mummy signaling as documented in README.md:
nim c -d:release --threads:on --mm:orc -r src/main.nim
```

Internet access requires TLS/WSS, TURN if necessary, and firewall configuration. `aiortc` performs the DataChannel DTLS/SCTP transport; the Mummy server carries SDP/ICE only.

## Option 1: Export directly into shards (recommended for large models)

```bash
python -m patchlib.cli shard-export \
  --model sshleifer/tiny-gpt2 --revision main \
  --peer-id alice --seq 1 \
  --store ./shard-store --output ./alice.fmanifest \
  --shard-mb 4 --max-state-gb 32
```

To export effective adapted weights, append `--local-patch alice.fpatch` and/or one or more `--remote-patch bob.fpatch`, using the same model ID/revision, rank and kind used to train them. For example, add `--kind loha --rank 32`.

For `train` / `train-p2p`, you can also append `--shard-state-manifest alice.fmanifest --shard-store ./shard-store --shard-mb 4`. This creates a sharded full effective state after training, without a separate `.fstate`. (Remote/custom model code still needs the **offline sandbox** as documented in `README_CUSTOM_STATE.md`.)

## Option 2: Split a pre-existing full snapshot

```bash
python -m patchlib.cli shard-create \
  --state ./alice.fstate --store ./shard-store \
  --output ./alice.fmanifest --shard-mb 4
```

Inspect/verify cache:

```bash
python -m patchlib.cli shard-inspect \
  --manifest ./alice.fmanifest --store ./shard-store --verify
```

## P2P transfer (one-way)

Use the same random, high-entropy `--room` on both peers. **Start receiver and sender concurrently.** Optionally pin the sender's public key; the key can be generated using the existing `keygen` command and copied to the receiver over an independently trusted channel.

**Peer A, source**:

```bash
python -m patchlib.cli shard-exchange \
  --ws ws://127.0.0.1:8080/ws --room this_is_a_long_random_room_123 \
  --send ./alice.fmanifest --store ./shard-store \
  --signing-key ./alice.key.pem --send-only
```

**Peer B, receiver**:

```bash
python -m patchlib.cli shard-exchange \
  --ws ws://127.0.0.1:8080/ws --room this_is_a_long_random_room_123 \
  --store ./shard-store --recv-dir ./manifests \
  --peer-public-key ./alice.pub.pem \
  --assemble-output ./received.fstate
```

`--assemble-output` is optional. With the full manifest and cache stored, run `shard-assemble` at any later time.

After an interrupted connection, **repeat the same two commands** with the same receiver `--store`. Only absent or invalid shard hashes are requested. Incomplete `.part` files are discarded, while complete hash-verified shard files remain. This resumes at *shard granularity*, not the byte offset within an interrupted shard.

## P2P transfer (both ends send their model snapshot)

Both ends run `shard-exchange --send their-own.fmanifest --store their-own-store --recv-dir ...` (without `--send-only`). Each manifest should have the same `model_id` and `revision`, different `peer_id` and appropriate `seq`. The two peers can exchange different full-state versions of the same model without implicitly overwriting either running model. This is transport, **not parameter averaging**.

## Reconstruct & load

```bash
python -m patchlib.cli shard-assemble \
  --manifest ./manifests/<received-manifest-hash>.fmanifest \
  --store ./shard-store --output ./restored.fstate

python -m patchlib.cli state-import \
  --model sshleifer/tiny-gpt2 --revision main \
  --state ./restored.fstate --output-dir ./restored-model
```

`state-import` validates the complete archive before applying tensors. **Use an unused model instance/process**; an OS/device error during copying can still leave a live model partially changed.

## Security and limitations

- No incoming Python code is executed in shard transfer, inspection, or reconstruction. Model code (including `trust_remote_code` or local factory code) must still run inside the separately documented offline container; that sandbox is an isolation precaution, not a formal guarantee.
- Received manifests are structurally and size validated; paths are derived exclusively from lowercase 64-byte hex hashes. Each chunk and tensor is SHA-256 verified. The maximum snapshot size defaults to 32 GiB, configurable with `--max-state-gb`; storage exhaustion is still possible if the receiver approves very large snapshots or retains many generations.
- **Ed25519 sender authentication is optional but strongly recommended.** Without `--peer-public-key`, SHA-256 checks integrity relative to an *untrusted* manifest, not origin or correctness. A public key received over the same connection is not a trusted key. Transport authenticates the manifest in transit; saved manifest files themselves do not carry durable detached signatures. A local filesystem attacker can replace saved manifests; restrict local directory access.
- Signed manifests prevent substitution by unauthenticated senders, but not replay of an older correctly signed manifest, malicious gradients from a legitimately authorized peer, or semantic incompatibility of unrelated custom model code. Track `peer_id` and monotonically increasing `seq` in your application if you need anti-rollback.
- For a model whose *every weight changes*, a new snapshot may replace most or all shard hashes: **this does not magically compress model updates**. Keep sending `.fpatch` for efficient LoRA/LoHA updates; shard snapshots are for baseline synchronization, checkpoint dissemination and loss recovery.
- To download shards of a 300B+ model, explicitly increase `--max-state-gb` on both ends. This implementation retains content-addressed objects after use; manually delete old stores/objects only after ensuring no manifest you need refers to them. There is no automated garbage collection or multi-peer provider selection yet.
- A `.fmanifest` does not contain a tokenizer, HF configuration, source code or optimizer state. Quantized/meta/sparse/nonstandard tensor representations remain subject to the existing `.fstate` restrictions.

## Local validation

`PYTHONPATH=. pytest -q` verifies conventional transfers plus shard round trips, direct adapted-weight exports, corrupt/missing/malicious shard rejection, per-shard resume, signed manifest rejection, and simulated bidirectional DataChannel messaging. The transport implementation has not been verified across two physical machines with real `aiortc`/TURN and the Nim server in this environment.
