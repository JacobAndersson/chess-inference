- [x] Setup logging infra so that logs are saved to ./logs for each training run.
- [x] Setup better test logging. I want to measure move accuracy bucketed by start, middle and endgame.
- [x] default to always use wandb
- [x] setup model snapshotting so that we can test models at different steps.
- [x] Perform hparam sweep on 5M model, varying the datasetsize, elobuckets to get preliminary results, learning rate and schedule.
- [x] setup test suite to let the model play agains stockfish at different elo levels to test its ability.

## Match paper (Transcendence, arXiv 2406.11741)

- [x] **Game packing (concatenate games into fixed-length blocks)**: Added `PackedChessDataset` that concatenates games into fixed-size blocks (default 1023 tokens). No padding used. Configurable via `packed=True` in `create_dataloader`.

- [x] **Dropout = 0.0**: Changed default to 0.0 in `ModelConfig`.

- [x] **Disable bias in linear layers**: Added `bias` flag to `ModelConfig` (default `False`). Wired through attention (qkv, proj), MLP (fc1/fc2 or gate/up/down), output head already had bias=False.

- [x] **Residual path scaling**: `_init_weights` now scales attn `proj` and MLP output layer (`fc2`/`down`) by `0.02 / sqrt(2 * n_layers)`.

- [x] **Cosine schedule minimum LR**: Added `min_lr` parameter to `TrainingConfig` (default 3e-5 = 10% of peak). LR lambda now decays to `min_lr` instead of 0.

- [x] **Warmup steps = 2000**: Updated default from 1000 to 2000 in both config and CLI.

- [x] **Illegal move retry in evaluation**: Changed from 3 retries + random fallback to 5 retries + forfeit (matching paper). On illegal move, model retries without making a move.

- [x] **Glicko-2 rating calculation**: Added `glicko2.py` with full Glicko-2 implementation. Integrated into `stockfish_eval.py` to compute model rating from game results.

## Convergence improvements (beyond paper)

- [x] **Data shuffling (streaming shuffle buffer)**: Added `_shuffled_iterator` with configurable buffer size (default 10,000). Available in both `ChessTokenDataset` and `PackedChessDataset`.

- [x] **SwiGLU activation**: Added `SwiGLUMLP` with 8/3x expansion (rounded to nearest 256). Controlled via `use_swiglu` flag in `ModelConfig` (default `True`). Keeps param count similar to GELU MLP.

## Training speed improvements

- [x] **Binary data format**: Added `OutputFormat::Binary` to Rust `TrainingWriter` (raw bytes, one byte per token). Added `BinaryChessDataset` using numpy memmap for near-instant loading. CLI flag: `--binary`.

- [x] **Multi-worker data loading**: Changed default `num_workers` from 0 to 1, added `pin_memory=True` when workers > 0.

## Scaling law experiments

Roadmap for producing Chinchilla-style scaling law plots. See `results.md` for detailed findings.

### Phase 1: Fix training setup (DONE)
- [x] **Context length**: 512 vs 1024 — no difference. Keeping 512.
- [x] **Batch size**: bs512 + linear LR scaling is best. Lock in bs=512 (grad_accum=8) for final runs.

### Phase 2: LR sweep per model size (DONE)
- [x] **5M and 10M**: lr=1e-3 best for both (50k steps).
- [x] **50M**: lr=1e-3 best (20k steps, loss=0.467).
- [x] **150M**: lr=1e-3 tested at bs=32 (20k steps, loss=0.499). OOM at bs=64 on 24GB GPU.

### Phase 3: Final scaling runs
- [ ] **Fix 150M training**: Use gradient accumulation (bs=32, grad_accum=2 or 4) to get effective batch size up while fitting in memory. Retest with larger effective batch + scaled LR.
- [ ] **Final training per model size**: Train each model (5m, 10m, 50m, 150m) at bs=512 equivalent (via grad_accum), lr scaled linearly from 1e-3 base at bs=64. Run long enough for loss to plateau, logging loss vs tokens_seen.
- [ ] **Dataset size variation**: For each model size, run with 1M, 10M, 100M, and full dataset to measure data scaling.

### Phase 4: Visualization
- [ ] **Scaling law plots**: Script to pull runs from wandb and produce loss vs compute (FLOPs = 6 × param_count × tokens_seen) for each model size, showing the optimal compute frontier.

## Future tasks

- [ ] **Benchmark packed vs unpacked training**: Compare convergence speed and tokens/sec with packed blocks vs padded individual games to quantify the improvement.
- [ ] **Tune SwiGLU hidden dimension**: The 8/3x expansion rounded to 256 alignment is a good default but may benefit from tuning per model size.
- [ ] **Add learning rate as CLI arg for min_lr**: Currently min_lr is only configurable via config, add `--min-lr` CLI argument.
- [ ] **Profile binary vs text data loading**: Benchmark the binary format loading speed against text CSV to quantify improvement.
- [ ] **Add Glicko-2 rating to wandb logging**: Log the computed Glicko-2 rating alongside win rates during stockfish evaluation.
