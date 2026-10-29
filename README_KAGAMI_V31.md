# Kagami v31 · Distributed learning desktop (2026-10-25)

Kagami is the new **application name and GUI** of the existing Fedpara distributed-training runtime. The internal `patchlib` package and `fedpara` wire protocol/storage directories intentionally retain their historic identifiers; changing them without a protocol migration would break existing peer identities, signed journals and saved models.

## Launch

1. Use Python with installed compatible PyTorch, Transformers, safetensors and existing project backend dependencies (see the legacy README for their setup).
2. Install the desktop bridge: `python -m pip install -r requirements-gui.txt`.
3. Launch: **`python kagami.py`**, or `python -m patchlib.cli gui`. A local OS WebView desktop session is required. No separate GUI HTTP server is opened.
4. The GUI has **Japanese and English** language controls. Its typography uses OS-available Japanese font fallbacks (`Hiragino Kaku Gothic ProN`, `Noto Sans CJK JP`, `Noto Sans JP`, `Yu Gothic`, `Meiryo`); fonts are **not bundled**. Installing Noto Sans CJK JP on your machine improves Japanese glyph consistency.
5. Design: six fixed-height tabs with no document/page scroll, even when using the app offline. Compact panes such as long logs may have *internal* scrolling. Narrower than the 960px minimum window width is not a supported desktop layout.

## Six tabs

- **MODELS:** A model is an immutable identifier binding a Hugging Face model/revision, task and default dataset collection to a distributed training identity. Displays local model descriptors and public signaling directory entries. Only **verified** hardware/disk impossibility is marked *Unavailable / 学習不可*; others show *Needs test / 要動作確認* or *Unverified / 未検証*. Detection is heuristic and does not establish model support or real trainability, particularly for exotic model types. Compatibility checks are rate-limited to visible cards.
- **TRAINING:** Start/stop optimization, live local training loss, step rate, and local top-1 token/label accuracy where supported; neither value should be confused with validation-set metrics. GPU cache fraction and v30 safetensors streaming can be configured. Underlying training preserves existing optimizer/adapter checkpoint design and may not yet support every HF architecture.
- **DATASETS:** Search the public Hugging Face Hub, select multiple repositories (up to 12 per model), configure source split/revision/config/column mappings and sampling weights, and attach to a model draft. Hugging Face connectivity is required to search.
- **EVAL:** Launch a separate base-model causal-LM inference process with `fp32`, `fp16`, `bf16`, `int8`, bitsandbytes `fp4`/`nf4`, or torchao `int4`. Optional dependencies and GPU support vary by option. **Important: This EVAL does not yet apply Kagami's accumulated distributed updates or checkpoint merges.** Loading large base models here may itself require substantial RAM/VRAM; EVAL is not guaranteed to use v30 layer streaming.
- **NETWORK:** Actual directly connected local WebRTC peers and counters from the training engine; no fabricated remote-to-remote edges. Topology is a **local 1-hop view**, not a global swarm graph.
- **NEW MODEL:** Create a new distributed model with Hugging Face architecture/revision metadata, multiple default datasets, training identity and a built-in `.py` editor. Edited Python can be registered by content SHA-256, but **it is not executed inside the networked training worker**. Arbitrary remote/P2P Python execution is deliberately disabled for security; safe execution of custom architectures requires separately reviewed packaging or an offline isolated runtime.

## Always-visible bottom bar

When training is active, a bottom bar shows the model, last-known loss, worker activity and a **pause/resume** button from every tab. The bar is a status indicator; activity animation is not an exact percent-complete measurement. The `TRAINING` tab also has a **Disconnect** button for a complete network exit.

**Pause vs. disconnect:** Pause creates the worker's `--swarm-pause-file` sentinel. At its next training-loop checkpoint, it seals any pending local update, *stops local optimizer steps* and continues authenticated WebRTC, signed update Gossip, inventory/repair and snapshot serving. Resume clears that sentinel without restarting the worker. This minimizes restart latency but is not an instantaneous GPU-kernel preemption; a currently running long forward/backward step must finish. Explicit Disconnect requests graceful shutdown and checkpoint persistence. Closing the entire WebView also disconnects the subprocess in this release; **relay-only operation requires keeping Kagami open**.

## Retained v29/v30 storage and compatibility

Per-model manifests and signed journals remain under `.fedpara-gui/rooms/<id>/...` for backwards compatibility. The sidebar has file manager buttons for checkpoints, streamed model weights, and genesis. v29's unlimited local publication (rounds=0), journal durability, and sharded final weights are retained. v30's opt-in memory-adaptive safetensors streaming and GPU cache remain; unsupported model operators and device kernels still exist.

For updates, never overwrite a currently used immutable room manifest with a different model or dataset. Create a fresh model identity instead.

## Verification and known limitations

The v31 Python bridge, opt-in worker contract and fixed-layout HTML/JS have local unit tests. The GUI was rendered in headless Chromium at 1366×768 and 1024×660 without page scrolling. Large-scale real hardware tests for arbitrary MoE/300B models and 1000+ real WebRTC peers were **not conducted** in this build. Some limits (large model kernels, source-specific dataset loading, quantization library support, and cross-peer execution of a user-defined Python factory) need further work. This is a desktop application and protocol refresh, not proof of universal model support.

For v30 memory details, consult `README_STREAMING_MASSIVE_MODELS_V30.md`.
