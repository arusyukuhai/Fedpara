# Kagami v39 — durable journal / restart recovery fixes

This release is based on v38. It does **not** delete or rewrite old rooms, model shards, checkpoints, journals, identities, or telemetry.

## What was failing

1. **`durable update log is full`:** v38's default GUI launcher sent `--max-journal-gb 20`, and `EventJournal` rejected new local or relayed signed updates when the archive reached that cumulative capacity. This was an artificial *storage byte cap*, not the publication/step limit.
2. **Reopening old models:** The GUI created a random unsaved draft at every launch and kept its separate activated-model variable empty, even when it highlighted an existing model card. Pressing Start was rejected. Clicking an existing model card now enables a safe subsequent Start. A successfully selected model is also remembered for next launch.
3. **Different unpack directories:** Before v39, the default data root was `Path.cwd()`. Starting a newly unzipped version from another folder could make all prior rooms appear missing even though they remained on disk. v39 pins the data root outside the distribution folder and can detect *one* unambiguous neighboring legacy installation.
4. **Settings silently changing after restart:** v38 forgot the genesis-affecting training options (streaming-base, custom `.fstate` path, ReLoRA interval and optimizer). A pinned per-profile launch manifest is now reused rather than resetting to GUI defaults; an incompatible/corrupt saved pin is never silently overwritten.
5. **Wrong dormant identity:** When multiple GUI profiles existed, the first alphabetical dormant profile could be reclaimed even if another held far more signed history. Dormant profiles now prioritize the one with the most existing archive data.

## Changes

- GUI launches with `--max-journal-gb 0` (no artificial per-journal cap). CLI default also becomes 0. If a positive cap is set **explicitly by an advanced CLI user**, the safe rejection behavior stays: no silent pruning of events needed by offline peers.
- Old SQLite + `.fplog` records remain in their original format. `MAX(seq)` survives restart and signed histories are replayed without modifying their event IDs.
- Shared peer rank and immutable model descriptor are still respected; random new model IDs are not minted during Start.
- `~/.config/kagami/data-root.txt` (Unix) remembers the original *directory*, not copies of giant model files. A user may explicitly set `KAGAMI_DATA_DIR` to their old work directory. Never delete that directory simply because the new ZIP opens successfully.
- Every chosen local model is saved as `.fedpara-gui/last-room.txt`; next launch restores it. The immutable manifest is verified again before a new trainer is spawned.
- Long event replay now prints `KAGAMI RESUME replaying ...` progress every five seconds.
- Read-only diagnostic: `python scripts/report_kagami_resume.py [OLD_DATA_ROOT]` prints rooms, journals, history sizes, and saved resume policy presence; **no model data is changed**.

## Running after an update

Launch normally:

```sh
python kagami.py
```

If the model catalog is empty because the earlier app was run from another folder, launch with the old **working directory** as the data root:

```sh
KAGAMI_DATA_DIR='/absolute/path/to/previous/kagami-working-directory' python kagami.py
```

`KAGAMI_DATA_DIR` must be the directory **containing** `.fedpara-gui`, NOT `.fedpara-gui` itself. On Windows PowerShell, set `$env:KAGAMI_DATA_DIR='C:\path\to\previous\directory'` before starting.

On a first v39 launch after v38, explicitly choose the old model card in MODELS if it has not already been restored, then select/start it. Reusing a model never creates a new room ID. A v38 journal may lack the new `.launch.json`: v39 preserves the journal and infers only a limited, unambiguous legacy streaming case; an old custom external genesis path or changed ReLoRA/optimizer settings still requires an exact matching configuration. Do not erase the journal if the worker reports an immutable-genesis mismatch. Read the `.gui.log` next to the journal to diagnose it.

## Storage guarantees and honest limits

**Unlimited by software ≠ unlimited physical disk.** Signed individual events must remain available for P2P reconciliation, including offline peers. This patch deliberately does not delete/archive-away signed updates at an arbitrary threshold. When physical storage is exhausted, new events cannot be made durable and learning cannot safely continue. Changing `20` to `0` prevents the *premature* artificial-cap shutdown; it does not solve finite storage or O(history length) startup replay.

Distributed signed checkpoint/epoch transition with safe P2P garbage collection is separate future protocol work, **not** implemented by this patch. Actual remote multi-peer rejoin and very large journal performance remain unverified here.

## Validation

- New tests: no fixed cap while preserving old signed entries, explicit small-cap safe rejection, dormant-profile reuse, pinned optimizer/reinit/streamed-base recovery after controller restart, persistent data-root resolution.
- Existing v38 duplicate-room checks, v37 inference and GUI checks, durability, P2P elastic, and previous GUI/auto-plan tests run separately. Older tests asserting Courier/Chicago are obsolete because the requested font was changed in v36.
