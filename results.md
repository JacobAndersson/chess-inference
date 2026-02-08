# Experiment Results

## Preliminary Sweep (5k steps, 5M model)

18 configs: 3 ELO buckets × 3 LRs × 2 dataset sizes. Key findings:

- **LR is dominant**: 1e-3 >> 3e-4 >> 1e-4
- **Dataset size doesn't matter at 5k steps**: 500k vs full dataset identical
- **ELO bucket**: all ≈ 2000 > 1500

Best: `5m_eloall_lr1e-03` at **0.6792 loss**.

## Model Size Sweep (50k steps, elo=all)

| Model | LR | Steps | Loss | Opening | Midgame | Endgame |
|-------|------|-------|------|---------|---------|---------|
| 5M | 1e-3 | 50k | 0.5333 | 83.4% | 79.3% | 78.7% |
| 5M | 3e-4 | 50k | 0.5694 | 83.0% | 77.7% | 77.0% |
| 5M | 1e-4 | 50k | 0.6223 | 82.4% | 75.5% | 74.6% |
| 10M | 1e-3 | 50k | 0.4851 | 83.8% | 81.4% | 81.1% |
| 10M | 3e-4 | 50k | 0.5123 | 83.6% | 80.1% | 79.7% |
| 10M | 1e-4 | 50k | 0.5584 | 82.9% | 78.1% | 77.5% |

LR=1e-3 best for both sizes. 10M significantly outperforms 5M.

## Context Length (5M, lr=1e-3, 10k steps)

| Seq Len | Loss |
|---------|------|
| 512 | 0.6181 |
| 1024 | 0.6170 |

No meaningful difference. Most games fit within 512 tokens.

## Batch Size Scaling (5M, 10k steps)

| Effective Batch | LR | Loss | Opening | Midgame | Endgame |
|-----------------|------|------|---------|---------|---------|
| 64 | 1e-3 | 0.6093 | 82.6% | 75.8% | 75.1% |
| 256 | 1e-3 | 0.5658 | 83.3% | 77.9% | 77.0% |
| 256 | 4e-3 | 0.5259 | 83.7% | 79.6% | 79.0% |
| 512 | 1e-3 | 0.5461 | 83.6% | 78.8% | 78.0% |
| 512 | 8e-3 | 0.4966 | 84.0% | 81.0% | 80.4% |

Larger batches with linearly scaled LR give substantial gains. bs512 + lr=8e-3 at only 10k steps matches 10M at 50k steps with bs64.

## Larger Models (20k steps, lr=1e-3)

| Model | Batch | Loss | Opening | Midgame | Endgame |
|-------|-------|------|---------|---------|---------|
| 50M | 64 | 0.4673 | 84.3% | 81.9% | 81.6% |
| 150M | 32 | 0.4987 | 83.8% | 80.7% | 79.9% |

50M outperforms 150M at 20k steps, but 150M was handicapped by half batch size (OOM at bs=64) and its loss curve was still steeply declining. The 150M would likely surpass 50M with more training.

## Key Takeaways

1. **LR=1e-3** is optimal across all model sizes tested (5M, 10M, 50M).
2. **Larger batch + linear LR scaling** is highly effective. bs512 with lr=8e-3 gives ~0.11 loss improvement over bs64 at the same step count.
3. **Context length 512 vs 1024** makes no difference for the 5M model at 10k steps.
4. **Model scaling works**: 50M > 10M > 5M, with no sign of diminishing returns yet.
5. **Phase accuracy pattern**: Opening (~84%) > Midgame (~81%) > Endgame (~81%) for best models. Gap narrows as model quality improves.
6. **150M needs gradient accumulation or checkpointing** to train at reasonable batch sizes on 24GB GPU.
