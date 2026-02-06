- [x] Setup logging infra so that logs are saved to ./logs for each training run.
- [x] Setup better test logging. I want to measure move accuracy bucketed by start, middle and endgame.
- [x] default to always use wandb
- [x] setup model snapshotting so that we can test models at different steps.
- [x] Perform hparam sweep on 5M model, varying the datasetsize, elobuckets to get preliminary results, learning rate and schedule.
- [x] setup test suite to let the model play agains stockfish at different elo levels to test its ability.

## Convergence improvements

- [ ] **Data shuffling (streaming shuffle buffer)**: The IterableDataset reads games sequentially from disk with no shuffling. When the iterator resets between epochs, games are read in the exact same order. This ordering bias hurts convergence. Implement a shuffle buffer that holds N games in memory and yields randomly from it, so we get good shuffling without loading the full dataset into memory.

- [ ] **Use the attention mask in the model**: The model receives `attention_mask` but ignores it entirely (model.py:104, ARG002 noqa). Padding tokens participate in self-attention, wasting compute and injecting noise. Pass the mask through to `scaled_dot_product_attention` so padding is properly excluded.

- [ ] **Residual path scaling**: GPT-2 scales the output projection of attention and MLP by `1/sqrt(2*n_layers)` for training stability at depth. Currently all linear layers are initialized with the same std=0.02. Add proper residual scaling to the projection layers in attention and MLP.

- [ ] **Cosine schedule minimum LR**: The LR schedule decays all the way to 0.0. Standard practice is to set a floor at ~10% of peak LR to prevent training from stalling in late stages. Add a `min_lr` parameter to the cosine schedule.

- [ ] **SwiGLU activation**: Replace the standard GELU MLP with SwiGLU (used in LLaMA, Gemma, etc.). SwiGLU consistently improves convergence in transformers. The MLP expansion factor changes from 4x to 8/3x to keep param count similar.

## Training speed improvements

- [ ] **Binary data format**: Training data is stored as comma-separated text files. Every epoch the dataloader re-parses CSV text into integers, which is slow. Switch to memory-mapped binary format (numpy .npy or torch .pt) for near-instant loading.

- [ ] **Multi-worker data loading**: DataLoader defaults to num_workers=0, meaning data loading blocks the main training thread. With proper IterableDataset worker sharding (or a map-style dataset), adding workers would overlap data loading with GPU compute.

- [ ] **KV-cache for generation**: The generate() method recomputes the full sequence from scratch at every token step. Adding a KV-cache would make Stockfish evaluation and inference significantly faster, especially for long games.
