# Kagami v78 — Root-anchored, certified-epoch distributed Critic continuity

## Goal and implementation

v77's Critic gossip was bound to the *active* Actor genesis and explicitly rejected
Actor certified epoch migration. v78 adds a **new signed Critic protocol, KVC78**, that
binds the Critic value-head updates to the immutable **room-root Actor genesis**,
while signing the exact Actor-epoch genesis that produced each PPO value update.
This makes the Critic value head and its signed updates survive *certified* Actor
checkpoint generations without relabelling/re-signing any old Critic events.

The v78 independent Critic event archive is `<original-actor-journal>.critic-v78.sqlite3`.
It survives epoch switches. `--distributed-critic` / GUI Critic training: P2P
activates v78 **on newly created rooms**. Existing v77 Critic journal archives
and room policies are NOT auto-migrated: their signatures, root identities and
optimizer semantics differ, so attempting to reuse an old v77 Critic journal
raises an explicit error. Local-only Critic remains available.

### What changes

- v78 events are signed with a fresh domain, KVC78 magic, immutable root Critic
  identity, source Actor-epoch genesis, and the existing Actor journal-tip marker.
  Every receiver checks the signed metadata, verified root identity, shape, and
  finite values. A source epoch not present in the receiver's verified local
  Actor certificate ancestry is rejected **before durable append**.
- On startup from an existing certified checkpoint pointer, the trusted local
  epoch quorum ledger is opened and its full 1..N ancestry verified. These are
  the only epoch identifiers the restored Critic will accept. Absent, truncated,
  or forked certificates prevent restored Critic updates from being replayed.
- On a live certified Actor epoch switch, Critic keeps the same signed lane and
  trained head. After the existing Actor certified switch/carry fence, the
  Critic registers the new certified Actor epoch, invalidates cached PPO
  trajectories and Critic AdamW moments, and rebinds PPO to the new Actor
  adapter. Actor's optimizer/journal still follow their existing switch path.
- During old Actor epoch shutdown, buffered KVC78 Critic frames are processed
  as Critic updates, not misrouted into Actor's signed event pipeline.
- Critic event ingestion preflights the **entire proposed next value head**
  (including target-precision overflow) before committing the durable event;
  a corrupt or malicious update cannot poison the replay log with a rejected
  nonfinite head.
- The shared objective protocol and authenticated Hello capability are
  upgraded to v78, separating the new stream from old v77 rooms/peers.

### Important mathematical interpretation

The root-pinned Critic head is a *continuous* value approximation for a fixed
room and reward policy across Actor generation checkpoints. The gossip is
**asynchronous additive parameter merging** of local AdamW increments; it is
not synchronous averaged gradients and not an exact mathematical off-policy
correction. The source Actor-epoch stamp gives verifiable provenance *relative
to the local certified chain*, **not** a guarantee that all remote Actor
trajectories have equivalent value targets. Individual source `actor_tip`
markers are signed by their originating peer but are **not** independently
certified as globally complete actor rollouts.

Future/uncertified epoch updates are rejected; they can be re-requested after
both peers have joined the certified epoch through v78 frontier/need. Peers on
different active Actor generations cannot directly exchange normal update
frames over the old Actor channel; they resume Critic gossip when the verified
Actor transport is rekeyed to a common generation.

No V-trace, Retrace, policy-conditioned value network, off-policy importance
correction for Critic, full HF Critic-backbone LoRA distribution, or dual-Actor/
Critic atomic database transaction is claimed. Stable PPO optimisation over
stale Actor versions remains a research limitation.

### Safety/compatibility constraints

* Existing v77 rooms: **do not** open with v78 and do not rename archives.
  Start a new room or implement an explicit consensus-approved migration.
* Certified committee and intact Actor carry checkpoint are prerequisites to
  cross-generation continuation. A raw network epoch ID is never sufficient.
* Missing or corrupt signed Critic event journals, mismatched head layouts,
  unknown Actor generations, divergent baseline hashes, and nonfinite updates
  fail closed. Existing v76 Actor replay/crash recovery remains in use.
* The v78 continuity procedure has not been demonstrated with genuine WAN
  partitions, simultaneous multiple live generation switches, large HF GPU
  models, power failures, or thousands of real peers.

### Files

- `patchlib/distributed_critic_v78.py`: domain-separated root Critic event lane
- `patchlib/critic_epochs_v78.py`: verified quorum ancestry admission
- `patchlib/elasticnet.py`: startup, WebRTC v78, restart and certified handoff
- `patchlib/mixed_learning_v65.py`, `patchlib/cli.py` and GUI: v78 policy and commands
- `tests/test_critic_epochs_v78.py`: multi-epoch/gap/tamper/overflow cases
- `VALIDATION_V78.md`: exact test scope
