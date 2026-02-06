- [x] Setup logging infra so that logs are saved to ./logs for each training run.
- [x] Setup better test logging. I want to measure move accuracy bucketed by start, middle and endgame.
- [x] default to always use wandb
- [x] setup model snapshotting so that we can test models at different steps.
- [x] Perform hparam sweep on 5M model, varying the datasetsize, elobuckets to get preliminary results, learning rate and schedule.
- [x] setup test suite to let the model play agains stockfish at different elo levels to test its ability.

## Convergence improvements

- [ ] **Move-level tokenization**: Currently using character-level tokenizer (vocab_size=32) where each character is a token ("Nf3" = 3 tokens). Switch to a move-level tokenizer where each SAN move is a single token (~1800-2000 unique moves). This would cut sequence lengths by ~3-4x, let the model reason at the move level directly, and dramatically reduce compute per game. Requires changes in both the Rust data-processing pipeline and the Python model/dataset code.

- [ ] **Data shuffling**: The IterableDataset reads games sequentially from disk with no shuffling. When the iterator resets between epochs, games are read in the exact same order. This ordering bias hurts convergence. Either implement a shuffle buffer for streaming, or switch to a map-style dataset that shuffles between epochs.

- [ ] **Use the attention mask in the model**: The model receives `attention_mask` but ignores it entirely (model.py:104, ARG002 noqa). Padding tokens participate in self-attention, wasting compute and injecting noise. Pass the mask through to `scaled_dot_product_attention` so padding is properly excluded.

- [ ] **Residual path scaling**: GPT-2 scales the output projection of attention and MLP by `1/sqrt(2*n_layers)` for training stability at depth. Currently all linear layers are initialized with the same std=0.02. Add proper residual scaling to the projection layers in attention and MLP.

- [ ] **Cosine schedule minimum LR**: The LR schedule decays all the way to 0.0. Standard practice is to set a floor at ~10% of peak LR to prevent training from stalling in late stages. Add a `min_lr` parameter to the cosine schedule.

- [ ] **Game result conditioning**: The training data includes game results (win/loss/draw) but they're discarded during tokenization. Prepending a result token (e.g. "W", "B", "D") to each game would let the model learn to condition on outcome, improving move quality and enabling result-conditional generation at inference.

- [ ] **ELO conditioning prefix**: The model has no way to distinguish skill levels within a dataset. Adding an ELO bucket token at the start of each sequence (e.g. a special token for "1500" or "2000") would let a single model learn to play at different strength levels.

- [ ] **SwiGLU activation**: Replace the standard GELU MLP with SwiGLU (used in LLaMA, Gemma, etc.). SwiGLU consistently improves convergence in transformers. The MLP expansion factor changes from 4x to 8/3x to keep param count similar.

- [ ] **Rotary positional embeddings (RoPE)**: Replace learned absolute positional embeddings with RoPE. RoPE gives better performance and enables length generalization beyond the training context window.

## Training speed improvements

- [ ] **Binary data format**: Training data is stored as comma-separated text files. Every epoch the dataloader re-parses CSV text into integers, which is slow. Switch to memory-mapped binary format (numpy .npy or torch .pt) for near-instant loading.

- [ ] **Multi-worker data loading**: DataLoader defaults to num_workers=0, meaning data loading blocks the main training thread. With proper IterableDataset worker sharding (or a map-style dataset), adding workers would overlap data loading with GPU compute.

- [ ] **KV-cache for generation**: The generate() method recomputes the full sequence from scratch at every token step. Adding a KV-cache would make Stockfish evaluation and inference significantly faster, especially for long games.

- [ ] **Process more training data**: Only ~115k games are currently processed (from 2013-2015 Lichess). The raw games directory contains 1.5TB of PGN files. Processing more data (especially higher-ELO games) would improve model quality significantly.
