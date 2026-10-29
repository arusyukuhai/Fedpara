# Kagami v70 — purpose-aware Hugging Face discovery & GUI refresh

This release builds on v69. It does **not** change actor/critic gradient equations, the PT/SFT/PPO input schema, or the signed P2P objective policy.

## Changes

- **Purpose-aware Hub discovery** (`patchlib/hub_discovery_v70.py`): metadata-only candidate discovery for `pt`, `sft`, `ppo`, `reward`, `critic`. Initial **recommended** results use role-specific seeded queries and Hub server filters where trustworthy (PT language-modeling tag; Reward/Critic text classification), plus transparent name/tag ranking. `broader` is opt-in. Bounded fetches (up to 3 × 20 metadata records), deduped, highest-ranked 24 returned. No model weights or dataset rows are fetched; no remote Hub Python is executed.
- **Unified selection dialog**: opens immediately with use-specific results *without requiring a keyword*. Search, popularity, gated access notice, reason/uncertainty per result, and fallback to broad Hub discovery. Role-specific headings and guidance.
- **Selection with immutable pinning**: SFT/PPO and PT use existing dataset-info + SHA pin logic and prefill split/config only when unambiguous; Reward/Critic use existing model-info + SHA pin. Never claim a candidate actually loads or has trained scalar head. Column mapping and model head must still be validated.
- **Reduced settings clutter**: folds config/split/column mapping into an optional details section, automatically expanded for ambiguous variants. Adds a concise objective selection summary. Preserves v69 local JSONL settings and previous mixed-learning manifests.
- **PT datasets screen**: PT recommendations are the default when entering DATASETS; optional full role-aware browser available.
- **Accessibility**: keyboard closing and focus return, basic focus loop, live status, keyboard search, and loading/error/empty states.

## What discovery can and cannot prove

Hub naming, tags, and pipeline metadata are *hints*. A dataset named `sft` might have no `prompt`/`response` columns. A text-classification model called `value` may lack a trained scalar critic head. Candidate selection is **not** compatibility verification. Existing runtime validation remains the authoritative check.

The Hub can be unavailable, gated, private, or fail to provide config/split info. These cases are reported; users can enter IDs and fields manually. We do not auto-execute `trust_remote_code`.

## Testing

See `VALIDATION_V70.md`. Run `python -m pytest -q tests/test_hub_discovery_v70.py` and `python tests/manual_browser_v70.py` (requires Playwright and Chromium; optional). Existing GUI, mixed training, critic and reward regression tests were also exercised.

## Remaining limitations

- No live Hugging Face Hub network test with production credentials in this environment; discovery tests use deterministic mock results. The real browser scenario uses simulated bridge responses, not a desktop OS WebView.
- Broader/full model architecture and datatypes support remain subject to v69 runtime limits.
- Column auto-detection from actual dataset rows is not implemented: explicit mapping remains available.
- All-suite pytest run did not complete; focused regressions passed (details in validation file).
