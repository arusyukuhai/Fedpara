# Fedpara — Hugging Face dataset collections + native Python GUI

This release corrects the old meaning of `dataset`: **a room now holds a collection of actionable Hugging Face dataset references**, not a text summary. Each peer independently obtains examples from the Hub using `datasets>=4`, without sending dataset examples, access tokens, or model parameters through Mummy.

## 1. Native desktop GUI (no HTTP server)

Install Python with Tkinter (macOS: a Python build with Tcl/Tk; Linux: `python3-tk` if not bundled) and:

```sh
python -m pip install -r requirements-patches.txt
python -m patchlib.cli gui
```

The GUI is a native Tkinter window. **No 8765 port** is opened and nothing is bound to an HTTP interface. It has: room ID, model/revision/task, base checkpoint, signaling URL, dataset collection add/remove/import, custom-code hash registration, save/load room, start/stop, and live logs. `Join / load room datasets` imports a published room manifest by room ID. That does not download the initial model checkpoint automatically. Native UI functionality requires a desktop display.

Dataset columns are entered as e.g. `text=text` or `input=question,target=answer` or `text=review,label=sentiment` depending on the selected task. Sources may specify independent weight/config/split/revision. Private/gated Hub datasets require the **client's own** Hugging Face authentication and dataset license acceptance (not shared in room metadata).

Custom Python model code is *never* executed by entering or joining a room. A local source file may be SHA-256-pinned in the descriptor, and execution still requires explicit use of the separate offline sandbox. The network-enabled trainer rejects arbitrary custom Python factories, as before.

## 2. Command-line usage without GUI

Example `hf-sources.json`, an **array of real Hub dataset references**:

```json
[
  {
    "repo": "roneneldan/TinyStories",
    "revision": "main",
    "config": null,
    "split": "train",
    "columns": {"text": "text"},
    "weight": 3.0
  },
  {
    "repo": "Salesforce/wikitext",
    "revision": "main",
    "config": "wikitext-2-raw-v1",
    "split": "train",
    "columns": {"text": "text"},
    "weight": 1.0
  }
]
```

Create a v2 room definition:

```sh
python -m patchlib.cli room-manifest-create \
  --model sshleifer/tiny-gpt2 --revision main --architecture causal-lm \
  --hf-datasets-file hf-sources.json --dataset-seed 42 \
  --output room-manifest.json
```

Then start a creator node (requires a valid matching base model checkpoint):

```sh
python -m patchlib.cli train-live \
  --room-id YOUR_RANDOM_ROOM_ID --room-manifest room-manifest.json \
  --model sshleifer/tiny-gpt2 --revision main --hf-task causal-lm \
  --state-base genesis.fstate --kind loha --rank 32 \
  --journal creator.sqlite3 --rounds 0
```

Another node can inspect the *actual registered collection*, without a local JSON:

```sh
python -m patchlib.cli room-info --room-id YOUR_RANDOM_ROOM_ID \
  --ws ws://127.0.0.1:8080/ws --output joined-room.json
```

Or start training with `--room-id` and **no `--text-file` or `--room-manifest`**; the trainer fetches the room's registered collection and pins it under its journal path. It still needs the model identifier, matching initial `.fstate` and task specified on the command line:

```sh
python -m patchlib.cli train-live \
  --room-id YOUR_RANDOM_ROOM_ID --model sshleifer/tiny-gpt2 \
  --revision main --hf-task causal-lm \
  --state-base genesis.fstate --kind loha --rank 32 \
  --journal joiner.sqlite3 --rounds 0
```

Use a **different journal path per device**. Room manifests and datasets do not replace synchronized model weights; all nodes must use the same genesis hash and compatible LoHA settings. Remote deployments require `wss://`.

## 3. Runtime behavior and security

- A room manifest is immutable during an active room's lifetime, maximum 4 KiB. Supports up to 12 dataset sources.
- Only Hub dataset references, config, split, revision, a strict column mapping, mixing weight and seed cross signaling. The server neither accesses Hugging Face nor trains models.
- Actual data downloads/streams directly on the client through Hugging Face `datasets>=4`, which does not support legacy remote dataset loading scripts. Training examples can be public, or private if the user has separately authenticated to Hugging Face on that device. Never put tokens in JSON manifests.
- Current collators support `causal-lm` / `masked-lm` text; `seq2seq-lm` input/target; `sequence-classification` text/integer label. For image, audio, nested chat format, and arbitrary model inputs, supply **pretokenized .safetensors batches** or extend the **trusted local** preprocessing layer. A room reference alone does not imply a universal task collator.
- The sampling method uses a bounded shuffled streaming buffer and weighted picks. It is *not* exact non-overlapping partitioning across peers. Different peers receive different seeds but can overlap; the weights are approximate over time, not exact epoch proportions.
- `main` branch revisions can change. For reproducibility **pin every model and dataset to a commit SHA**.
- An invited user should verify the manifest fingerprint with the creator through a trusted channel before executing training. Room IDs are bearer secrets; an untrusted signaling service could otherwise substitute metadata.
- Room metadata is held by the signaling server only while the room exists. If that server dies, **established** DataChannels survive but a new disconnected peer cannot retrieve metadata until a signaling endpoint recovers, or a copy of the room JSON is shared directly.
- This is not a CDN; data consumption obeys Hugging Face usage limits and licensing terms. Dataset access may require internet connectivity or a prepopulated local HF cache.

## 4. Notes for updating existing installations

Version 1 manifests (display-only `dataset`) remain valid for legacy local data (`--text-file`, `--jsonl`, `--batches-dir`). Version 2 manifests require the updated Mummy server from this ZIP. Older servers will reject `version: 2`.

If the previous GUI occupied TCP port 8765, shut down its old process separately. The new desktop window never binds a TCP listening socket.

Tests: `python -m pytest -q` (mock HF streaming tests avoid downloading large public datasets). Nim: `./scripts/check_nim.sh` (Nim compiler required; not available during package creation). Real 2-device network / private-dataset / display tests require local environment.
