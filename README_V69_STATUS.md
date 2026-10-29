# Kagami v69 — Hugging Face PPO Value Critic + Hub-first Training UI

## Implemented

- TRAINING UI defaults SFT/PPO data source pickers to Hugging Face Hub. Existing JSONL input remains as an explicitly selected advanced option, to preserve prior user data and compatibility. Hub dataset search, revision pinning, config/split and column mapping remain available.
- PPO VALUE CRITIC now supports `Hugging Face · pretrained Value Critic` alongside the existing local MLP/Linear. Select a Hugging Face repo, pin its 40-hex commit SHA, choose scalar sequence or token classification, select CPU/CUDA/MPS and maximum critic context.
- Critic is NOT the Reward Model. HF Reward Models remain frozen scorers for completed answers; HF Critic provides **per-decision state values** for Generalized Advantage Estimation (GAE) and PPO value loss.
- An HF sequence classifier supplies one scalar per *actor state prefix*; HF token classifier supplies the last-token scalar for each prefix. The Critic uses its own tokenizer and a suffix-preserving truncation policy, even when its vocabulary differs from the Actor's. A separate forward pass for each prefix makes this significantly slower than the original 4-feature value critic.
- Hub Critic loads with `trust_remote_code=False`, `use_safetensors=True`, an immutable revision, mandatory loading diagnostics, and `num_labels=1`; missing/mismatched initialized weights and ambiguous head structures are rejected. Requires one identifiable learned `nn.Linear(..., 1)` scalar head named score/classifier/v_head/value_head/etc. A repository ID alone does *not* imply compatibility.
- The pre-existing backbone is frozen and kept in eval mode. Only the scalar value head is trained by local AdamW and NEVER enters signed P2P Actor updates. The frozen base is reloaded from immutable Hub revision after restart; only the small trainable head + optimizer moments are atomically saved as data-only safetensors, validated against room, repo, revision, head and optimizer layout. Storage limit: 128 MiB checkpoint payload. No untrusted remote Python/pickle.
- The shared room objective policy includes the HF Critic type/repo/revision/value-head semantic configuration so misconfigured peers cannot share the same training history. Local device and private storage path do not enter policy. Existing v68 PT/SFT/PPO rooms without HF Critic retain their old protocol.
- `--critic-kind hf --critic-hf-repo Org/model --critic-hf-revision <40-hex SHA> --critic-hf-loader sequence_classification --critic-hf-device cpu --critic-hf-max-length 512` CLI support; GUI fully wired through controller -> CLI -> PPO -> Critic checkpoint.

## Not yet verified / deliberate limitations

- A random Hugging Face LM checkpoint, reward-only model, TRL-specific value-head wrapper or opaque quantized layer is not automatically a trained scalar Value Critic. Compatible built-in Transformers scalar Sequence/Token Classification checkpoints can be selected, but actual Hub models have **not** been downloaded or trained in this sandbox (Transformers package and model checkpoints unavailable).
- Critic backbone is FROZEN, only the scalar head is optimized, to limit memory use and checkpoint size. Full Critic LM fine-tuning/LoRA/quantized value head adapters are not implemented.
- Prefix-conditioned scoring requires up to one Critic forward per PPO token; a long-context HF Critic is expensive and can run out of VRAM/RAM. Choose the smaller compatible model and a separate device if possible.
- Critic state checkpoints still do not have a cryptographically coupled transaction with the actor's signed Journal tip. Network churn and 300B training / 4,096 WAN devices have not been verified.
- The model's value predictions do not judge answer quality; RLHF reward scoring is still done by the separately chosen Reward Model. Self-Critic reward grading is not introduced.
- Changing pinned Critic model or its architecture requires a new mixed-training room. HF Critic device can differ across peers.

## Quick start

1. Create/select a causal LM room, choose SFT and/or PPO datasets from Hugging Face Hub, use **Pin + inspect** for their immutable SHA; adjust split and column mapping only if needed.
2. TRAINING > PT + SFT + PPO > VALUE CRITIC > Architecture: **Hugging Face · pretrained Value Critic**. Search or enter a compatible *trained scalar value checkpoint*, click **Pin current commit**, choose scalar sequence/token classification and device.
3. Choose PPO's independent Reward Model under PPO Reward. Start the worker. The training log reports any incompatibility instead of using untrained/incorrect values.

See `VALIDATION_V69.md` for what was actually tested.
