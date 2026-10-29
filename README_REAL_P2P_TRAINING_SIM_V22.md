# Fedpara v22 — per-round validation and ideal-parallel compute time

The v21 P2P training simulator now writes **validation metrics for every round** by default, with baseline at round 0. The default `p2p-training-loss.png` plots loss versus training compute seconds:

- **Centralized X:** accumulated measured time for its optimizer steps.
- **P2P X:** accumulated measured time for ALL peer optimizer steps divided by `--peers` (e.g. 16), not raw serial time. CUDA/MPS synchronize before and after every timed step to avoid asynchronous dispatch undermeasurement.
- **Not included:** validation, network transfer/synchronization, snapshot/merge/reinit, event application, or actual multiprocess contention. The P2P curve is a *theoretical perfectly balanced parallel compute-time projection*, not measured cluster wall clock. Churn/heterogeneous peers can make the real time worse; compare `p2p_compute_max_peer_s` too.

`p2p-training-trace.csv` includes the separate per-round time series `central_compute_s`, `p2p_compute_sum_s`, `p2p_compute_div_peers_s`, `p2p_compute_max_peer_s` plus the previous Loss/token/coverage series. The alternative `p2p-training-loss-tokens.png` preserves the equal-token comparison.

## Example: 16 peers, every round

```bash
python -m scripts.simulate_p2p_training \
  --model tinystories-1m --dataset tinystories \
  --peers 16 --rounds 200 --local-steps 4 \
  --reinit-every 20 --rank 16 --wide-lora-rank 4 \
  --lr 0.0001 --batch-size 4 --context 64 \
  --eval-every 1 --output-dir benchmark-results/relora_16
```

At 16 peers, computing **17 heldout losses each round** is expensive. If speed matters more than plot density, use `--eval-every 5` or 10. The timing series is recorded on all validation rows, and central/P2P compute exclude evaluation overhead.

Note that the 16-peer loss plateau is **not solved simply by fixing the chart**: centralized learning takes 16 times as many optimizer steps on a shared adapter, whereas each P2P adapter receives only its own steps. A flat orange line may represent a real optimization gap; the new loss and update-coverage trace help diagnose it.
