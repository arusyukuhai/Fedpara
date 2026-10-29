# Kagami v80 — Validation record

## Test suite

The complete v80 suite, before the final addition of one extra recipient-Actor drift regression, ran successfully:

```
910 passed, 3 skipped in 68.43s
```

The final additional regression (`test_receiver_recomputes_probability_after_its_actor_changes`) was run with the full v80-specific module and all nine tests passed. A final full-suite rerun after this addition should be used to produce a single unified number; do not portray 910 as that final rerun.

Python bytecode compilation and `node --check web/desktop_webview.js` passed.

## Newly exercised behavior

- 9 v80 module tests: authenticated signed KTR80 wire, certified source Actor-epoch gating, reward checks from reference tokens, bounded behavior probabilities, direct two-peer experience ingestion and Critic-only update, source Actor unchanged, duplicate protection and equivocation, missing-event/frontier gap repair, crash after append before replay, restored head + AdamW state after restart, storage failure rollback, corrupted checkpoint rejection, v80 objective/GUI CLI selection, and changes in receiving Actor affecting computed importance ratio.
- Existing v78/v79 Critic / PPO / GUI tests were included in both the related subset and full-suite run.

## Not tested

Live WebRTC between independent physical machines, authentic heterogeneous pretrained Hugging Face GPUs and multiday WAN/Gossip convergence, source Actor probability forensic verification, adversarial bounded traffic over thousands of peers, unbounded-stale importance ratios. Do **not** claim mathematical exactness for Byzantine sources, rejected out-of-range trajectories, or finite samples.

## Sample wire size

A synthetic 2-token rollout (two prompt IDs, two action IDs, two behavior log probabilities, two reference IDs) produced a 480-byte signed wire using a short test-signature stub. This is not a measured bandwidth-per-learning-step for real workloads; actual wire size varies with prompt, trajectory length and signature representation.
