# Kagami v38 — accidental duplicate model-room fix

This version patches v37 without erasing any of your existing models, training histories, signing keys, or checkpoints.

## Root cause

The v37 GUI filled a **fresh random room ID** each time it started, and `startOrResume()` automatically ran `saveModel()` when the ID wasn't already in the local catalog. Pressing Start without explicitly selecting an existing model therefore created another room under an identical model ID/dataset configuration. The UI catalog already merged equal `room_id`s, but a new random room ID is a different training namespace.

## Changes

- TRAINING Start/Resume **no longer calls `saveModel()`**. Select an existing entry in MODELS with **Select this model**, or explicitly create one through NEW MODEL before starting. This applies to the bottom quick-start button too.
- Selecting a public model imports its **existing, server-pinned network ID and manifest** locally before enabling TRAINING. Joining does not create a replacement room.
- NEW MODEL creation checks for an **identical immutable manifest** under any other local room ID. A new accidental duplicate is rejected before writing any manifest. This check and file commit are serialized across GUI windows with a catalog-level filesystem lock.
- For intentionally independent runs with the same configuration, the checkbox **Create an independent experiment with identical settings** allows it. These runs remain separate networks, with separate histories and journal keys.
- MODELS displays the short network ID and notes when the same upstream HF model+revision is used by several distinct IDs. Same upstream HF repo does **not** imply that their learned weights or history can be merged.
- Existing duplicates are **retained** (never deleted or silently merged).

## Reviewing past accidental duplicates

From the Kagami working directory, run:

```bash
python scripts/report_duplicate_model_rooms.py
```

This prints only *exactly identical immutable manifests* found under distinct room IDs. It is read-only. If you started training in both rooms, inspect their histories/checkpoints before deciding which to keep. Do not simply delete a folder: old rooms may have unique updates or checkpoints.

## Verification

Tests `tests/test_kagami_v38_room_duplicates.py` cover accidental-duplicate rejection, explicit independent creation, distinct datasets, idempotent saving, corrupt manifests, concurrent GUI windows and joining an existing public model. `node tests/kagami_v38_no_implicit_room.js` verifies that pressing TRAINING Start with an auto-generated unsaved draft does not create or start a room.

## Scope and limitations

No existing data is migrated/deleted. Public signaling directories are still keyed by room ID, and deliberately created distinct rooms are distinct model networks, even when they share the same HF model. Real Internet signaling was not rerun for this UI/catalog change. The changes do not claim to fix separate inference errors.
