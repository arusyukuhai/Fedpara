# Fedpara Swarm crash/rejoin recovery fix

## Observed failure

The initial 8-node join finished (`all-joined snapshots=7`), peer 6 was
SIGKILLed and restarted, but the harness timed out at `peer-rejoined`.
The console summary alone cannot tell whether the restarted process failed
P2P authentication, failed to restore local state, or was stuck behind the
cold-join gate. See `peer-06/run-02.log` for the actual failure.

## Audited root cause in code

`train_elastic` recreated `initial_complete=False` on **every** process launch,
including when SQLite had an `initial_ready=1` marker written only after a
verified snapshot or an authoritative room genesis transition. The restored
`BootstrapState` was correct, but the generic `initial_join_decision` still
waited for **every** selected neighbor's continuously changing signed event
inventory. During busy 8-peer training this may never quiesce, so JOIN READY
could be withheld even after successful local replay.

The simulator also previously used a stale cumulative Hello count to describe
peer health and only looked for `JOIN READY`, without distinguishing a
recovered member from a true newcomer.

## Changes

* `patchlib/join_guard.py`: explicit verified-member rejoin decision; fresh
  joiners are never allowed to use this path.
* `patchlib/elasticnet.py`: restored members with durable `initial_ready` can
  resume with their replayed journal and local working factors while async
  anti-entropy handles missing remote updates. Prints `REJOIN READY:`.
  Fresh joiners still require a verified full-model snapshot.
* `patchlib/swarm_sim.py`: peer 6 is considered rejoined only when this run has
  emitted `REJOIN READY` and **a new** `SIGNED HELLO OK`; the previous run's
  authentication evidence is not reused. On early process exit or timeout,
  saves `peer-NN/rejoin-diagnostic.json` with the recent log lines and phase.
* `tests/test_swarm_rejoin_regression.py`: recovery, fresh join isolation,
  simulator stdout monitor, and diagnostics regression coverage.

## Usage

Apply the patch, then run an **entirely new** simulator output directory:

```sh
python -m patchlib.cli swarm-sim --peers 8 --faults 2 \
  --seed 20261022 --ws ws://127.0.0.1:8080/ws \
  --output ./swarm-runs/recovery-001
```

Optional: add `--embedded-signaling` instead of `--ws` to separate the
signaling-server implementation from the model-synchronization behavior.
This still uses actual aiortc WebRTC peer connections when aiortc is installed.

If `peer-rejoined` still fails, inspect:

```sh
cat ./swarm-runs/recovery-001/peer-06/rejoin-diagnostic.json
rg -n 'Traceback|ERROR|JOIN WAIT|REJOIN READY|SIGNED HELLO|room presence|recovery' \
  ./swarm-runs/recovery-001/peer-06/run-02.log
```

The peer chosen for SIGKILL may differ when the seed or fault count changes.
The path from the latest `fault-injected peer=N` event identifies it.

## Limitations

This is **not** proof that 8-way convergence works. The harness must still
finish its gossip-settlement and final full-state numeric comparison. Running
out of remote signed updates, model inconsistencies, broken ICE connectivity
and model-commit bugs are distinct possible causes of failure. A returning
member's local model can temporarily lag during anti-entropy, as is normal
for this experimental asynchronous additive algorithm; this does not imply
true synchronous FedAvg convergence.

Tested locally: Python unit/integration regression suite on CPU. aiortc and
transformers are unavailable in the build environment, so the full 8-process
WebRTC learning test has **not** been executed here.
