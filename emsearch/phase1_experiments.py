"""Phase 1 + Phase 2 experiments: setup tests then LR sweeps for larger models."""

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


def lr_sweep_experiments(
    preset: str,
    seq_len: int,
    max_steps: int = 20_000,
) -> list[tuple[str, ExperimentConfig]]:
    """LR sweep for a given model size."""
    configs = []
    for lr in [1e-3, 3e-4, 1e-4]:
        name = f"{preset}_lr{lr:.0e}"
        model = get_preset(preset)
        model.max_seq_len = seq_len
        training = TrainingConfig(
            data_dir=DATA_DIR,
            learning_rate=lr,
            max_steps=max_steps,
            checkpoint_dir=f"/workspace/checkpoints/phase2/{name}",
            wandb_project="chess-transformer-phase2",
        )
        configs.append((name, ExperimentConfig(model=model, training=training)))
    return configs


def run_configs(label: str, configs: list[tuple[str, ExperimentConfig]]) -> None:
    """Run a list of experiment configs sequentially."""
    logger.info("=== %s: %d runs ===", label, len(configs))
    for i, (name, config) in enumerate(configs):
        config.device = "cuda"
        logger.info("=== Run %d/%d: %s ===", i + 1, len(configs), name)
        train(config, run_name=name)
        logger.info("=== Completed: %s ===", name)


def main() -> None:
    """Run Phase 1 (setup tests) then Phase 2 (LR sweeps for 50M/150M)."""
    setup_logging("phase1_and_2")

    # Phase 1a: Context length
    run_configs("Context Length", context_length_experiments())

    # Phase 1b: Batch size
    run_configs("Batch Size", batch_size_experiments(seq_len=512))

    # Phase 2: LR sweeps for larger models (20k steps each)
    run_configs("50M LR Sweep", lr_sweep_experiments("50m", seq_len=512))
    run_configs("150M LR Sweep", lr_sweep_experiments("150m", seq_len=512))


if __name__ == "__main__":
    main()
