"""Phase 1 experiments: context length and batch size A/B tests."""

import logging

from emsearch.config import ExperimentConfig, TrainingConfig
from emsearch.presets import get_preset
from emsearch.train import train
from emsearch.utils import setup_logging

logger = logging.getLogger("emsearch")

DATA_DIR = "/workspace/games/processed"
CHECKPOINT_BASE = "/workspace/checkpoints/phase1"
WANDB_PROJECT = "chess-transformer-phase1"
MAX_STEPS = 10_000


def context_length_experiments() -> list[tuple[str, ExperimentConfig]]:
    """Context length: 512 vs 1024."""
    configs = []
    for seq_len in [512, 1024]:
        name = f"ctx{seq_len}"
        model = get_preset("5m")
        model.max_seq_len = seq_len
        training = TrainingConfig(
            data_dir=DATA_DIR,
            learning_rate=1e-3,
            max_steps=MAX_STEPS,
            checkpoint_dir=f"{CHECKPOINT_BASE}/{name}",
            wandb_project=WANDB_PROJECT,
        )
        configs.append((name, ExperimentConfig(model=model, training=training)))
    return configs


def batch_size_experiments(seq_len: int) -> list[tuple[str, ExperimentConfig]]:
    """Batch size: 64, 256, 512 with multiple LRs each."""
    batch_configs = [
        (64, 1, [1e-3]),
        (64, 4, [1e-3, 4e-3]),
        (64, 8, [1e-3, 8e-3]),
    ]
    configs = []
    for batch_size, grad_accum, lrs in batch_configs:
        effective_batch = batch_size * grad_accum
        for lr in lrs:
            name = f"bs{effective_batch}_lr{lr:.0e}"
            model = get_preset("5m")
            model.max_seq_len = seq_len
            training = TrainingConfig(
                data_dir=DATA_DIR,
                learning_rate=lr,
                batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                max_steps=MAX_STEPS,
                checkpoint_dir=f"{CHECKPOINT_BASE}/{name}",
                wandb_project=WANDB_PROJECT,
            )
            configs.append((name, ExperimentConfig(model=model, training=training)))
    return configs


def main() -> None:
    """Run all Phase 1 experiments."""
    setup_logging("phase1")

    # Part 1: Context length
    ctx_configs = context_length_experiments()
    logger.info("=== Context Length Experiments: %d runs ===", len(ctx_configs))
    for i, (name, config) in enumerate(ctx_configs):
        config.device = "cuda"
        logger.info("=== Run %d/%d: %s ===", i + 1, len(ctx_configs), name)
        train(config, run_name=name)
        logger.info("=== Completed: %s ===", name)

    # Part 2: Batch size (using seq_len=512 as default, change if ctx1024 wins)
    bs_configs = batch_size_experiments(seq_len=512)
    logger.info("=== Batch Size Experiments: %d runs ===", len(bs_configs))
    for i, (name, config) in enumerate(bs_configs):
        config.device = "cuda"
        logger.info("=== Run %d/%d: %s ===", i + 1, len(bs_configs), name)
        train(config, run_name=name)
        logger.info("=== Completed: %s ===", name)


if __name__ == "__main__":
    main()
