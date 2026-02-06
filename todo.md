- [x] Setup logging infra so that logs are saved to ./logs for each training run.
- [x] Setup better test logging. I want to measure move accuracy bucketed by start, middle and endgame.
- [x] default to always use wandb
- [x] setup model snapshotting so that we can test models at different steps.
- [x] Perform hparam sweep on 5M model, varying the datasetsize, elobuckets to get preliminary results, learning rate and schedule.
- [x] setup test suite to let the model play agains stockfish at different elo levels to test its ability.

## Match paper (Transcendence, arXiv 2406.11741)

- [ ] **Game packing (concatenate games into fixed-length blocks)**: The paper concatenates multiple games into fixed-size blocks of 1023 tokens, separated by ";" delimiters. No padding is used — every token in a batch is meaningful. Our current approach pads each game individually, wasting compute on padding tokens. Need to: add ";" to the vocabulary (vocab 33), implement a streaming block packer that concatenates games until the block is full, and remove padding/attention-mask logic from the dataloader.

- [ ] **Dropout = 0.0**: The paper uses no dropout at all. We default to 0.1. Change the default and ensure the 50M preset uses dropout=0.0.

- [ ] **Disable bias in linear layers**: The paper sets bias=False on all linear layers. We currently have bias enabled everywhere. Add a `bias` flag to ModelConfig and wire it through attention, MLP, and the output head.

- [ ] **Residual path scaling**: The paper scales residual projection weights by `0.02 / sqrt(2 * n_layers)` during initialization, matching the GPT-2 approach. We initialize everything at std=0.02. Fix `_init_weights` to apply this scaling to the `proj` layer in attention and `fc2` in MLP.

- [ ] **Cosine schedule minimum LR**: The paper uses min_lr = 3e-5 (10% of peak lr=3e-4). Our schedule decays to 0.0. Add a `min_lr` parameter to TrainingConfig and the LR lambda.

- [ ] **Warmup steps = 2000**: The paper's 50M config uses 2000 warmup steps. Our default is 1000. Update the default.

- [ ] **Illegal move retry in evaluation**: When playing Stockfish, the paper retries up to 5 times if the model generates an illegal move before forfeiting. Verify our Stockfish eval handles this and matches the paper's protocol (100 games per Stockfish level, levels 1/3/5, 100ms timeout).

- [ ] **Glicko-2 rating calculation**: The paper uses Glicko-2 to compute model ELO from games against Stockfish. Add or verify we have a Glicko-2 implementation for consistent rating estimation.

## Convergence improvements (beyond paper)

- [ ] **Data shuffling (streaming shuffle buffer)**: The IterableDataset reads games sequentially from disk with no shuffling. When the iterator resets between epochs, games are read in the exact same order. This ordering bias hurts convergence. Implement a shuffle buffer that holds N games in memory and yields randomly from it, so we get good shuffling without loading the full dataset into memory.

- [ ] **SwiGLU activation**: Replace the standard GELU MLP with SwiGLU (used in LLaMA, Gemma, etc.). SwiGLU consistently improves convergence in transformers. The MLP expansion factor changes from 4x to 8/3x to keep param count similar.

## Training speed improvements

- [ ] **Binary data format**: Training data is stored as comma-separated text files. Every epoch the dataloader re-parses CSV text into integers, which is slow. Switch to memory-mapped binary format (numpy .npy or torch .pt) for near-instant loading.

- [ ] **Multi-worker data loading**: DataLoader defaults to num_workers=0, meaning data loading blocks the main training thread. The paper uses num_workers=1 with pin_memory. At minimum match that.
