# Real P2P training peer sweep (v23)

`scripts/sweep_p2p_training.py` runs the actual-LM P2P training simulator for
multiple peer counts and aggregates the outputs into two graphs:

- `peer-sweep-loss.png`: one P2P mean line per peer count over compute seconds
  (x axis = sum of measured peer training compute divided by peer count).
- `peer-sweep-vs-peers.png`: x axis = peer count, showing centralized final
  loss, P2P mean final loss, and P2P worst final loss.

It also writes:

- `peer-sweep-summary.csv`
- `peer-sweep-trace.csv`
- `peer-sweep-result.json`

Example:

```bash
python -m scripts.sweep_p2p_training   --model tinystories-1m   --dataset tinystories   --peers-list 2,4,8,16   --rounds 200   --local-steps 4   --reinit-every 20   --rank 16   --wide-lora-rank 4   --lr 0.0001   --eval-every 1   --output-dir benchmark-results/peer-sweep
```
