# Kagami v43 — Generation recovery and safe-GC hardening

**Status: incremental, fail-closed implementation. NOT the requested complete zero-pause online epoch rollover.**

This package extends v42. It does **not** silently delete logs from currently running trainers, nor pretend certificates alone can rebase a LoHa optimizer. **Production auto-rollover remains disabled.** Use with backups and test rooms.

## What changed

1. `patchlib/live_generation_transport.py` + `elasticnet.py`:
   - Direct peers advertise a certified manifest they actually hold (`generation-have`) under their already authenticated WebRTC control envelope. A late peer requests it from any verified full holder, not only committee members.
   - A peer serving a certified manifest verifies its SHA-256, tensor manifest semantics and every referenced tensor shard. The receiver also verifies the downloaded tensor shards and hash. Advertising does not confer authority; the quorum certificate does.
   - No availability claim is sent by a generic bridge until a real snapshot availability callback exists (default fail-closed).
   - Certificate chain/head repair remains bounded and resilient to disconnect/backpressure.
2. `patchlib/generation_committee_v43.py`:
   - Committee public Ed25519 keys can be pinned in `--generation-v1-committee` JSON, so a verifier does not need every voter to be online just to learn its public key.
   - Backward-compatible simple peer ID arrays remain supported but require authenticated learning of keys.
3. `patchlib/epoch_runtime_v43.py`:
   - Persistent SQLite state machine for staging committed epoch cuts and indexing **all post-cut events** without copying their multi-MiB payloads. Verifies checkpoint manifest, its semantic digest and every shard before acknowledging a new active model.
   - Actual optimizer/model installation **must happen in the trainer**, before the trainer writes its durable `generation_certified_*` new journal metadata. A simulated UI flag cannot install anything.
   - Polling GC deletes only an **exclusively unlocked old** journal, after a complete certified frontier check and an installed replacement. **Any post-cut events block destructive GC** until a mathematically verified replay/carry format exists. Remote data can never issue a delete command.
   - Two-phase durable GC intent and `.retired.json` tombstone prevent crash-partial deletion from resurrecting the old genesis.
4. `scripts/kagami_epoch_housekeeper_v43.py`:
   - Optional supervised `--once` or periodic crash-recoverable housekeeping process, with full pinned committee public keys. Rechecks certificates and performs only eligible local GC. It does not start/stop the trainer or install tensor data.
5. New tests cover certified snapshot availability, malicious manifest hints, disconnected member keys, old-writer file locks, uncovered suffix preservation, corrupt manifest refusal, restart safety and power-loss during GC. Broad regression subset: 93 tests passing in this build environment.

## Pinned committee example

The file supplied as `--generation-v1-committee` may now contain:

```json
{
  "keys": {
    "<peer-id-derived-from-ed25519-pubkey-A>": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----\n",
    "<peer-id-derived-from-ed25519-pubkey-B>": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----\n",
    "<peer-id-derived-from-ed25519-pubkey-C>": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----\n"
  }
}
```

Use real derived peer IDs, not labels. Never put private keys in this file.

## Housekeeper

Start as a separate process only after configuring a pre-existing, signed committee ledger and the replacement trainer. **It will not retire anything until the new trainer has acknowledged a verified checkpoint and the old trainer has released its lock.** Example (replace every placeholder):

```bash
PYTHONPATH=. python scripts/kagami_epoch_housekeeper_v43.py \
 --ledger /path/to/old.sqlite3.generation.sqlite3 \
 --root /path/to/generation-state \
 --room EXISTING_ROOM_ID --genesis 64_HEX_GENESIS \
 --committee /path/to/pinned-committee.json \
 --old-journal /path/to/old.sqlite3 --once
```

Without a prepared and acknowledged new journal, GC is intentionally a no-op.

## Explicitly still missing for true full-scale autonomous rollover

* **v1 proposal scaling**: inline `cutoffs` has a hard 256-origin limit; a Merkle-paged frontier and proof validation must replace it before thousands of unique update origins can propose an epoch.
* **Automatic certification**: v42's snapshot/cut validation voting callbacks remain fail-closed; currently no live trainer issues an atomic snapshot/cut receipt and replicates it to a voting quorum.
* **Worker hot switch**: the current `ContinualElasticEngine`, adapter base, LoHa anchors, optimizer moments, local event sequence, journal pointer, anti-entropy cursors, and connected RTC workers are not atomically switched when a cert arrives.
* **Post-cut carry**: it is not enough to preserve original signed bytes. Additive FedSum post-cut deltas must be verified and deduplicated against the certified frontier, committed to the new epoch's durable replay format, and then served to peers. Current v43 holds these updates rather than discarding them or unsafely rebasing them.
* **New checkpoint activation on offline return**: certificate and verified checkpoint transport exist, but the old live trainer does not yet install that base and reset its optimizer automatically. A stale peer cannot resume training safely on a new epoch without that handoff.
* **Network-level tests**: no production thousands-of-peers WebRTC/GPU chaos test has been run. A majority committee is crash-fault tolerant under assumptions, not Byzantine consensus.

Do not delete `.fplog`, `.sqlite3`, `.fedpara-gui`, signing identities, or checkpoint shards manually. Keep backups of all existing v42/v39 data.
