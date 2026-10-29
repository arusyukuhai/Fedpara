# Fedpara v18 — Sparse event push + paginated authenticated anti-entropy

## What changed

Fedpara formerly queued every newly signed low-rank event for **all** connected
neighbors (up to eight). Every relay repeated that same fanout, multiplying
unsolicited copies. `train-live` now defaults to `--push-fanout 2`:

- Select a deterministic clockwise successor in the bounded-v2 ring and a
  hash-selected alternate/shortcut from the currently connected, eligible
  neighbors. Exclude the source of a relayed event and its original author.
- Queue at most two *unsolicited* copies per hop; if the queue is full, leave
  the signed event in the **durable journal**. No event is dropped from durable
  storage because a fanout queue is full.
- Keep signed `frontier`, `need`, journal-backed resend, occasional full
  inventory audits, Ed25519 origin verification, and exact-once application.
  Missing events still propagate even when the initial push did not reach a
  peer or an RTC link was offline.
- Instead of transmitting all per-origin maxima in one 16-KiB control packet,
  send **48-origin pages** every 0.3s to each authenticated neighbor. With
  4,096 origins, each link cycles through its frontier in roughly 26 seconds,
  assuming sends succeed and the roster is stable. Failed signed sends never
  advance the cursor. v17-sized received pages remain accepted.
- Full digest inventory audits are retained for conflict detection and
  recovery, but their minimum page interval defaults to 2.5s instead of 0.4s.
- GUI exposes **Immediate Gossip targets** (1–8, default 2) and local
  transmitted payload MiB (PUSH vs NEED/repair, excluding SCTP headers).
  Network counters refer to currently active link generations.
- GUI and CLI expected-peer count validation now accepts 2–4,096 (the
  number is still informational, **not** a synchronization barrier).

## Invariants and limitations

**This is a redundancy reduction, not mathematical compression of all updates.**
Exact replicated FedSum needs every replica to eventually receive each
original signed delta (or a genuinely equivalent, separately proven aggregate).
Sparse push reduces duplicate *immediate sends* but does not make that
information-theoretic requirement disappear. Clients continue storing signed
updates and may request additional copies during churn. The journal hard cap,
network partitions, poisoning, and very large training event throughput remain
separate operational limits; 4,096 physical peers were not load-tested.

The signed event envelope, model layout, `.fplog` archive, and global additive
merge semantics are unchanged from v17; no room recreation is required for a
v17-to-v18 client-only rollout. Mixed v17 and v18 clients can still interpret
all received signed frames; network load reduction is largest when all clients
use v18. Back up `.sqlite3` and `.sqlite3.fplog` before updating.

## CLI

```bash
python -m patchlib.cli train-live --help
# ... --push-fanout {1..8}, default 2
```

To return to previous eager fanout, set `--push-fanout 8`. The GUI offers
that same parameter in the P2P training section.

## Reproducible tests and measurements

```bash
python -m pytest -q tests/test_sparse_gossip_v18.py
python -m pytest -q tests/test_frontier_gossip_repair.py
```

In an **in-memory routing simulation** with a stable 256-peer degree-8 ring,
for a single update:

| Strategy | Immediate attempted event transmissions |
|---|---:|
| v17 broadcast all eligible neighbors | 1,793 |
| v18 fanout=2 (with deterministic ring successor) | 512 |

That is **71.4% fewer attempted unsolicited event transmissions** in this
specific model. This is *not* a measured bandwidth or steps/s increase on
aiortc/WebRTC and does not include extra NEED replies under link failures.
Tests also cover 4,096-origin signed frontier paging and dropped pushes
recovered via simulated NEED repair.
