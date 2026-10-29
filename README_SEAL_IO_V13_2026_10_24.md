# Fedpara v13: Local SEAL breakdown and opt-in WAL throughput mode

This version keeps the v12 FedSum/periodic ReLoRA and mixed LoHa/LoRA
wire format unchanged. It addresses an observed two-window live-monitor example:
`LOCAL SEAL ~2301ms`, `WORKING SAVE ~67ms`. The remote `JOURNAL WRITE` tile
measures **inbound**, not local-origin commits, and cannot diagnose the local
seal's slowest step.

## Local-only telemetry

Five new fields now split every local seal:

* `LOCAL LOCK`: wait for the inbound/local commit mutex
* `LOCAL EXPORT`: copy factors, serialize to safetensors, sign envelope
* `LOCAL VERIFY`: verify/decode and preflight the signed local delta
* `LOCAL JOURNAL`: append signed wire to SQLite, including the transaction commit
* `LOCAL APPLY`: materialize and apply verified local model delta

`LOCAL SEAL` also includes minor control/dispatch overhead, so the component
sum may differ slightly. These are *latest operation*, not averages or percentiles.
All metrics remain isolated to the local GUI process.

## SQLite durability (explicit selection)

The GUI's "署名更新の保存モード" setting and `train-live --journal-sync` option
support two modes:

- **full** (default). SQLite WAL `synchronous=FULL`. Attempts to durably fsync
  each published/received event transaction; durability also depends on the
  operating system, filesystem and hardware.
- **normal** (opt-in). SQLite WAL `synchronous=NORMAL`. Avoids a per-event
  fsync, normally improving throughput on slow synchronized disks. Safe against
  a normal **process** crash under SQLite's WAL guarantees, but a sudden OS
  crash/power loss may lose some most recently acknowledged events. If a lost
  event had already been shared with other peers, anti-entropy may restore it;
  *this is not guaranteed*. Do not choose `normal` for irreplaceable updates.

The signed event **content/protocol is unchanged**. WAL checkpointing remains
outside the foreground update transaction. This is an I/O experiment; there is
no promised speed multiplier and CPU/GPU contention may remain.

Example CLI: `python -m patchlib.cli train-live ... --journal-sync normal`

**To compare fairly:** keep the same model/seed/dataset and two peers, record
~100 local updates with `full` vs `normal`, and compare median steps/s plus
LOCAL JOURNAL / LOCAL EXPORT / LOCAL VERIFY / LOCAL APPLY times. Stop both
processes before upgrading. Back up existing journals. A full cold join may
have a separate snapshot cost not measured by these seal fields.

## Verification

* `330 passed, 3 skipped` across three isolated pytest partitions.
* JavaScript syntax: `node --check web/desktop_webview.js`
* Added `tests/test_local_seal_io_v13.py` for the durability mode, local
  publication stage timing, and GUI-to-CLI option forwarding.
* Existing room/genesis/signature protocol unchanged; v12 room compatibility
  is expected, though mixing trainers with different I/O timing can still
  affect instantaneous learning curves.
