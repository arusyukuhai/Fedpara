# Kagami v33 — Usable forms and automatic training defaults

## Changes in v33

- **Scrolling:** The application shell and training dock remain fixed in the window. Real datasets, selected dataset drafts, NEW MODEL settings, the Python editor, training controls, logs and model directory panes can scroll independently. The code editor has a usable minimum height, selection/caret, clipboard and Tab indentation.
- **Online peer count:** Removed the misleading Expected peers field. The legacy CLI field did not govern membership or aggregation. Actual room membership is obtained from authenticated discovery when available, with a recent verified-neighbor fallback. The monitor displays membership count separately from directly observed WebRTC neighbor channels.
- **Context length:** Derived from built-in Transformers config (local cache, stored design or lightweight HF metadata) and the detected free RAM/VRAM, within a conservative 8192-token application ceiling. If metadata is unavailable, the conservative fallback ceiling is 1024 tokens. This is a heuristic, **not** a measured OOM guarantee; architecture-specific memory peaks vary widely.
- **Local steps per publication:** Initial cadence is derived from an approximate LoRA/LoHa update size, not a GUI text input. At runtime, the existing adaptive controller re-evaluates the cadence using observed room membership and device tiers. The current actual value is taken from training telemetry and displayed. Runtime auto-tuner cadence is bounded by `--max-sync-steps` (64) for automatic tuning.
- **Hugging Face datasets:** Search result selection now inspects actual repository config/split metadata via Hugging Face Hub's non-executing parquet inventory. For one known configuration it adds directly; for multiple configurations or missing metadata it fills the form, then asks the user to select/verify fields before adding. The selected draft list now scrolls; duplicate selections are indicated rather than silently counted as successful additions. New models receive all selected draft sources. The immutable manifest is still checked by the backend.
- **Python editor:** Editable even before selecting code mode. Focusing it opts in to Python Config design mode. Text editing, Japanese input, pasting and Tab indentation are enabled, with inner scrolling and a visible caret. The existing restricted `build_config()` parser remains: **arbitrary Python still does not run on other peers**.
- **Dataset constraints:** The immutable manifest intentionally allows up to 12 sources per model. Source column mappings must match the task, and private/gated datasets still require explicit access credentials on individual peers; no credentials are propagated through gossip.

## What was tested

- Python: 88 focused tests passed, including v32 model creation and learned-checkpoint regressions, legacy GUI, adaptive controller and elastic networking tests.
- Chromium: 1024 × 660 and 1366 × 768: all six screens show with no document-level scrolling; dataset draft persists across tabs; code textarea accepts input and automatically switches to code mode; no JavaScript runtime errors.
- Mocked Hugging Face Hub: search -> inspect -> add; one-config and multi-config cases -> NEW MODEL propagation.

## Limitations

- A real Hugging Face Hub connection, an uncached production model, GPU out-of-memory recovery and thousands of live peers were not tested in this environment.
- Context-size planning is only an estimate before runtime. The final supported context and memory behavior depend on the model's actual forward/backward kernels, attention scaling, tokenizer and hardware.
- No silent change of a signed room's immutable model/dataset definition: create a new model if you want to change source datasets.
- The base Kagami v32 architecture/security limitations still apply, including the fact that arbitrary `nn.Module` Python is not sent for execution on other peers.

## Launch

```bash
python kagami.py
# or
python -m patchlib.cli gui
```

For actual HF dataset integration, install the packages in the existing requirements files as appropriate for the machine.
