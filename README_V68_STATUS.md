# Kagami v68 — Hugging Face SFT/PPO dataset selection

This revision extends v67's GUI from local-only SFT/PPO JSONL to independently selected, pinned Hugging Face Hub datasets. PT's immutable room dataset remains unchanged. The Hugging Face PPO Reward Model and independent Critic added earlier still work.

## Desktop GUI

1. Open the selected room's **TRAINING > PT + SFT + PPO** pane, enable concurrent mode.
2. Under **SFT data source**, select **Hugging Face Hub**. Enter a search phrase or `organization/dataset` and click **Search Hub**. Select the result; use **Pin + inspect** to resolve the current 40-character commit and discover available configurations and splits when the Hub provides metadata. Set `config`, `split`, `prompt column`, `response column` to match the dataset.
3. Under **PPO data source**, select **Hugging Face Hub** and choose a DIFFERENT dataset if desired. Set `prompt`/`reference` mappings. If using a frozen external HF Reward Model, only prompt is required. Otherwise reference text must exist.
4. Select your existing independent **Reward Model** and **Value Critic** settings. Start the worker.
5. Alternatively select **Local JSONL** for either SFT or PPO. The local source choices and HF dataset revisions are private per-peer; the shared objective/policy and scorer settings remain immutable.

For each Hub source, the dataset is streamed via `datasets.load_dataset(repo, name=config, split=split, revision=<40 hex SHA>, streaming=True)`. Data is not fully downloaded and rewritten into a giant JSONL. When the finite iterator is exhausted it starts a new pass. The streaming loader must support the selected Hub repository; legacy script-only datasets, unsupported custom code, gated repositories without credentials, network outages and bad column mappings raise explicit errors.

The `prompt` and `response` or `reference` fields can be simple column names or dotted paths such as `instruction.text`. Strings are accepted. Lists of `{role,content}` chat turns in a selected field are converted to a plain role-labelled text representation (not an automatic tokenizer-specific chat template). Preference datasets with `chosen/rejected` pairs are *not* automatically turned into PPO trajectories. Column mappings must reflect the intended task.

## CLI example

Add the following to the existing `train-elastic --mode elastic` command (including your room/model/network options):

```bash
--sft-hf-repo myorg/sft-data \
--sft-hf-revision 0123456789abcdef0123456789abcdef01234567 \
--sft-hf-config default --sft-hf-split train \
--sft-hf-prompt-column instruction --sft-hf-response-column output \
--ppo-hf-repo myorg/rollout-prompts \
--ppo-hf-revision 0123456789abcdef0123456789abcdef01234567 \
--ppo-hf-split train --ppo-hf-prompt-column prompt \
--ppo-hf-reference-column reference \
--pt-weight 1.0 --sft-weight 0.5 --ppo-weight 0.1
```

For each objective the Hub source and local JSONL are mutually exclusive. A real commit SHA rather than a mutable branch or tag is required. The example commit is a placeholder; pin your actual dataset in the GUI. `huggingface_hub` and `datasets` dependencies are in `requirements-patches.txt`.

## Tests

- Full suite after implementation: **814 passed, 3 skipped, 0 failed**.
- Final directly affected subset after documentation and minor cleanup: **37 passed**.
- New `tests/test_hf_mixed_datasets_v68.py`: config/split/revision argument forwarding, incremental row mapping, iterator restart, chat-text fields, bad/missing columns, no-reference PPO with an external reward, real small-tensor SFT/PPO backward path (fake streamed data), Hub metadata-only pin method, GUI->real CLI command, immutable per-room shared policy, private per-peer dataset choice, v67 local config migration.
- `node --check web/desktop_webview.js` succeeded.
- ZIP CRC is checked during release packaging.

## Known limits

- Actual Hub downloads, real model/GPU execution, gated-repository login and live OS click-through were not verified in this environment. Dataset loading is tested by a mocked iterable factory and trainer-integration tests.
- The GUI currently asks the user to name columns; it does **not** sample remote rows or automatically infer schema. The **Pin + inspect** action uses Hub metadata and sometimes cannot list variants.
- For the highest safety/correctness, the loader does not execute arbitrary Hub dataset scripts. Some repositories will not work unless they provide a supported data format.
- Hugging Face streaming iterators may block on network I/O, so a remote outage can delay the local worker. There is no disk-backed resumable queue for SFT/PPO data yet; a restarted worker will start from the dataset's beginning.
- Each peer may choose local data while agreeing on *objective weights and reward-model scoring*. Dataset IDs/paths are not pinned in the shared P2P genesis; use a dataset provenance policy separately if all peers must consume exactly the same dataset.
