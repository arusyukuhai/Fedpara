# Fedpara elastic streaming training (default)

This repository provides model-agnostic LoRA/LoHA updates over PyTorch `nn.Module` tensors, a Mummy signaling server and aiortc DataChannels. **`train-live` now defaults to `--mode elastic`.** The fixed-membership, round-synchronous FedAvg remains available under `--mode synchronous --rounds N`.

## How it works

1. Every peer first installs **exactly the same, verified initial `.fstate`** (via the existing `shard-exchange` and `shard-assemble` workflow). It must also use the same model id, adapter kind, rank and alpha.
2. While training, the local adapter learns in minibatches. After `--local-steps`, or just before applying a new remote event, the local optimizer is stepped. The learned LoRA/LoHA factors are serialized as one **immutable, Ed25519-signed event**, with a strictly increasing `(origin, seq)` ID and the base/genesis digest.
3. Each event specifies its original denominator: `1 + count(connected, verified, recently responsive and synchronized neighbors)` at emission time. The receiver reconstructs the **delta tensor** and merges `W <- W + delta / event.denominator`. Factor matrices themselves are never averaged.
4. All changed factors and AdamW moments are cleared before training continues on the new base. Receiving multiple updates from the same sender never changes the base more than once.
5. Events are committed to a local SQLite WAL journal **before** changing the model. On crash/restart, install the **original genesis `.fstate`**, then replay the durable events. Keep the same journal path and immutable genesis.
6. Every pairwise WebRTC DataChannel regularly exchanges signed control frames and a *paged inventory*. Receivers ask for missing `(origin, seq)` events. Origin-signed envelopes allow another peer to forward an event even if the original author is permanently offline.
7. Dynamic membership uses a user-controlled, hot-reloaded roster JSON. A lost connection is retried with exponential backoff (up to ~60 seconds). After `--peer-timeout`, it does **not** contribute to the current denominator; training **never waits for an absent peer's FedAvg vote**. A late joiner catches up by requesting the old signed event history from surviving peers.

**Accuracy caveat:** Elastic is an **asynchronous, additive federated optimization algorithm**, *not* exactly round-synchronous FedAvg. Its averaging denominator reflects a local liveness view recorded in each event. During partitions, peers can have different bases and take different gradients. After receiving the same events, replicas converge *approximately* in floating-point arithmetic, but identical weights/strict FedAvg convergence are not guaranteed.

## Installation

```bash
pip install -r requirements-patches.txt
nimble install -y mummy
nim c -d:release --threads:on --mm:orc -r src/main.nim
```

Start the Mummy signaling server first. Production remote connections should use WSS with a valid TLS certificate and a TURN deployment if NAT requires it. The Nim server supports **two peers per room**, so configure one unique pairwise room per peer pair.

## Public-key setup

Generate long-lived Ed25519 identities for each peer:

```bash
python -m patchlib.cli keygen --private alice.key --public alice.pub
python -m patchlib.cli keygen --private bob.key --public bob.pub
```

Exchange **trusted public keys out of band**. Never send private keys. The user-authored roster should specify all potential direct peers; it is not populated from untrusted network messages.

On Alice's machine, create `roster-alice.json`:

```json
{
  "bob": {
    "room": "alice_bob_room_2026",
    "public_key": "/trusted/keys/bob.pub"
  }
}
```

On Bob's machine, create `roster-bob.json`:

```json
{
  "alice": {
    "room": "alice_bob_room_2026",
    "public_key": "/trusted/keys/alice.pub"
  }
}
```

Both use the same pairwise room. Add peers by modifying these files **on both endpoints**; delete a roster entry to stop connecting to a departed peer. A permanently departed peer's public key is retained in the local `.trusted.json` sidecar so that its previously signed updates can still be authenticated during gossip. This is a trust-history record; protect it along with the local SQLite journal.

## Start live training

```bash
python -m patchlib.cli train-live \
  --model sshleifer/tiny-gpt2 \
  --model-id my-stable-model \
  --revision <PINNED_HF_COMMIT> \
  --state-base /model/genesis.fstate \
  --text-file train.txt \
  --kind loha --rank 32 \
  --peer-id alice \
  --signing-key alice.key \
  --self-public-key alice.pub \
  --roster-file roster-alice.json \
  --ws wss://YOUR_SERVER/ws \
  --journal ./alice-events.sqlite3 \
  --local-steps 16 \
  --peer-timeout 18 \
  --heartbeat-interval 4 \
  --rounds 0
```

Bob uses the **same immutable initial model**, `--model-id`, revision, rank and alpha, with `--peer-id bob`, `--signing-key bob.key`, `--self-public-key bob.pub`, `--roster-file roster-bob.json`, its own dataset and its own SQLite journal.

`--rounds 0` means keep training until interrupted; a positive integer ends after that many locally emitted updates. The default is elastic, so `--mode elastic` may be omitted. `--peers` is *not supported* in elastic: the denominator is measured and stored per event, never silently guessed from a fixed roster. Use `--max-delta-rms` and `--max-patch-mb` to bound untrusted updates.

## Permanent disconnects / late joining

* **Network failure:** `--peer-timeout` expires peer liveness; the link is retried independently and offline peers are dropped from the *current* denominator. If every other peer is offline, local updates use denominator 1. There is no global quorum that can freeze learning indefinitely.
* **Disconnected for days:** events stay in durable local journals. A returning peer on the **same original genesis** fetches missing history from any online peer that has it. A late newcomer can do the same from genesis. Unknown/rotated signing keys and divergent genesis are rejected.
* **Any survivor missing a signed event:** if that event is held by a different surviving peer it can be forwarded. If an event disappears from **all** surviving journals, it cannot be reconstructed. A verified complete state checkpoint is then needed.
* **Disk usage:** journals are intentionally never silently discarded, even for long-gone peers. The default hard cap is `--max-journal-gb 20`. When it fills, the system raises a clear error rather than continuing with corrupted/incomplete replay. Change the cap, provision storage or plan a **coordinated checkpoint/epoch transition**.
* **Checkpoints:** `--checkpoint-every 100` writes `.fmanifest`/shard objects plus a signed `.checkpoint.json` receipt to `--checkpoint-dir`. The receipt binds the genesis, manifest digest and all locally observed signed event IDs. It does **not** automatically prune old history or force all peers onto the same model. Before discarding old event history, all active peers must agree on the checkpoint event set and **change the genesis**; stale clients must explicitly resync from the new checkpoint.
* **Partitions lasting forever:** physically disconnected networks cannot be kept mutually consistent without eventually exchanging data. They continue locally and are reconciled if communication is restored; no solution can guarantee global synchronous FedAvg without a quorum/consensus service.

Do **not** open `--unsafe-unsigned` on the internet. Custom Python factories and `trust_remote_code` must run offline inside the sandbox; the networked live trainer will refuse to execute such code.

## Testing and verification

```bash
PYTHONPATH=. python -m pytest -q
python -m patchlib.cli train-live --help
```

Tests cover 2D and higher-order LoRA/LoHA tensors, tied weights, missing peers, delayed joining, signature checks, restart replay, duplicate/equivocating events, hard log limits and mocked bidirectional DataChannels. **Real inter-host WebRTC with Nim/coturn, internet partitions lasting days and multi-GPU/HF model combinations have not been tested here.**
