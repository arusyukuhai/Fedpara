# Fedpara v29: unlimited GUI training, persistent weights and large-room hardening

## Summary

- **No fixed training/publication-count cutoff from the desktop GUI.** The previous GUI's `200` is removed; the trainer always receives `--rounds 0`. CLI callers can still choose a finite round count. The room-size adaptive **steps per published update** (default 4, capped by `--max-sync-steps 64`) is unrelated to a total training cutoff and is unchanged.
- **Graceful stop exports effective trained model weights as sharded, content-addressed data.** Each session generates a unique immutable manifest under `.fedpara-gui/rooms/<room>/checkpoints/<local-peer-profile>/effective-*.shards.json`. Corresponding tensor blocks are stored under `.fedpara-gui/rooms/<room>/models/<profile>/objects/...`. Snapshots are committed by an atomic manifest rename, so an interrupted export will not produce a valid-looking partial manifest. Existing complete manifests remain in place.
- **Previous immutable genesis, signed SQLite journal, working adapter/optimizer snapshot, and cryptographic peer profile remain on disk.** On next `train-live` start, the **original immutable genesis + original journal are replayed**. Never replace the genesis `.fstate` with a derived trained checkpoint while retaining the same journal: that would double-apply signed updates and change room identity. The newly exported weights are a separate artifact for loading/archival/export, not an automatic compacted training genesis.
- Model-export calls are moved off the asyncio event loop, so they do not block asynchronous RTC housekeeping during the stop-save procedure. POSIX trainer subprocesses have a distinct process session to avoid receiving a terminal/window hangup during lengthy exports. Closing a GUI sends a graceful stop; if exporting takes longer than the UI cleanup timeout, the child may continue independently until the checkpoint is complete. **Check for `CHECKPOINT SAVE COMPLETE` in the GUI logs and inspect the manifest before treating it as a completed backup.** Do not kill the worker or delete the directory while saving.
- GUI has **Open checkpoint location**, **Open trained model shard location**, and **Open genesis location** buttons. These expose only validated room-owned paths and launch the native file manager without shell interpolation. The current run's manifest path is also shown.
- The desktop UI exposes the **signed journal disk budget** in GiB (default 20 GiB, selectable 1–4096 GiB). The count limit is gone, but disk space and archive safety budgets are physical and security constraints; no infinite-duration or infinite-capacity guarantee is made.
- For huge model exports, desktop passes `--max-state-gb 2048` and `--shard-mb 16`, avoiding the previous 32 GiB export quota and reducing shard count. This is a *configured upper bound*, **not** a guarantee that a 2 TiB model can be trained on the local device.
- Archived peer-public-key registry now supports up to **65,536 historical identities**, instead of rejecting journals with more than 257 after high-churn rooms; this is independent of the 4096 currently concurrent member room limit.
- Replay and long-running training no longer retain an unbounded in-memory `set` of all `(origin, sequence)` IDs. A bounded 8192-entry LRU hot cache plus indexed SQLite journal provides durable duplicate detection. Replay temporarily bypasses SQLite membership lookup because journal entries exist before their weights have been applied to the freshly loaded base.

## Usage

From `fedpara/`:

```bash
python -m patchlib.cli gui
```

Open or create a room, press **学習開始（無期限）**, and use **停止・モデルを保存** for a graceful stop. The UI shows checkpoint locations and a completed-save indicator after the worker exits successfully. The names `checkpoints/` (manifests) and `models/` (content-addressed weight shards) must **both** be preserved.

For verification of an existing save:

```bash
python -m patchlib.cli shard-inspect \
  --manifest /path/to/effective-....shards.json \
  --store /path/to/models/peer-... \
  --max-state-gb 2048 --verify
```

The format is **not** a single Hugging Face `model.safetensors`. To reconstruct a `.fstate` (requires extra disk space), use `shard-assemble` with the same manifest, store and `--max-state-gb 2048`.

## Testing / remaining constraints

Run `PYTHONPATH=. pytest -q tests/test_unlimited_durable_gui_v29.py` and the existing `test_webview_gui.py`, `test_elastic.py`, `test_continual_elastic.py`, `test_snapshot_boundary_paging_v26.py`, `test_shard_runtime_v27.py`, `test_offload_v28.py` suites. New regressions cover unlimited CLI flags, distinct checkpoint filenames, bridge path whitelisting and 100,000 event IDs with bounded cache.

**Not solved:** Training an arbitrary 300B+ model on a phone still requires a full out-of-core model-loader, disk-safe merge and model-specific MoE kernels, as detailed in `README_DISK_EXPERT_OFFLOAD_V28.md`. The live trainer currently rejects experimental `--disk-linear-dir`/`--disk-optimizer-dir` configurations. Very large signed LoHa events can still hit `--max-patch-mb` and available RAM; gossip transfer of a single signed event has not been redesigned as a chunkwise streamed transaction. Replay still reconstructs historical weights by applying every signed event, and can be slow after millions of updates; a verified journal-watermarked compacted genesis requires protocol changes. Journal budgets and 4096 concurrent membership also remain intentionally bounded, not silently removed.
