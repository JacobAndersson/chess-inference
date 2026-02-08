"""Scaling law experiments: vary model size, step count, and ELO bucket."""

import argparse
import logging

from emsearch.config import ExperimentConfig, TrainingConfig
from emsearch.presets import get_preset
from emsearch.train import train
from emsearch.utils import setup_logging

logger = logging.getLogger("emsearch")

DATA_DIR = "/workspace/games/processed"
CHECKPOINT_BASE = "/workspace/checkpoints/scaling"
WANDB_PROJECT = "emsearch"

# Effective batch size = batch_size * grad_accum = 512 for all models
# LR scaled linearly: base 1e-3 at bs=64 -> 8e-3 at bs=512
EFFECTIVE_BATCH = 512
LR = 8e-3

# Per-model batch size and grad_accum to fit in 24GB GPU
MODEL_BATCH_CONFIG = {
    "5m":   (64, 8),
    "10m":  (64, 8),
    "50m":  (64, 8),
    "100m": (32, 16),
    "270m": (16, 32),
}

VERIFICATION_PRESETS = ["5m", "10m"]
FULL_PRESETS = ["5m", "10m", "50m", "100m", "270m"]
STEP_COUNTS = [50_000, 100_000, 1_000_000]
ELO_BUCKETS = ["1200", "2500", "all"]


def generate_configs(
    presets: list[str],
    step_counts: list[int],
    elo_buckets: list[str],
) -> list[tuple[str, ExperimentConfig]]:
    """Generate all experiment configs from the parameter grid."""
    configs = []
    for preset in presets:
        batch_size, grad_accum = MODEL_BATCH_CONFIG[preset]
        for max_steps in step_counts:
            for elo in elo_buckets:
                steps_label = f"{max_steps // 1000}k" if max_steps < 1_000_000 else f"{max_steps // 1_000_000}M"
                name = f"{preset}_elo{elo}_{steps_label}"

                model = get_preset(preset)
                training = TrainingConfig(
                    data_dir=DATA_DIR,
                    elo_bucket=elo,
                    batch_size=batch_size,
                    gradient_accumulation_steps=grad_accum,
                    learning_rate=LR,
                    max_steps=max_steps,
                    warmup_steps=2000,
                    checkpoint_dir=f"{CHECKPOINT_BASE}/{name}",
                    save_every=max(max_steps // 10, 5000),
                    eval_every=max(max_steps // 50, 1000),
                    wandb_project=WANDB_PROJECT,
                )
                configs.append((name, ExperimentConfig(model=model, training=training)))
    return configs


def main() -> None:
    """CLI entry point for scaling experiments."""
    parser = argparse.ArgumentParser(description="Run scaling law experiments")
    parser.add_argument(
        "--mode",
        choices=["verify", "full"],
        default="verify",
        help="verify=5M+10M only, full=all model sizes",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging("scaling")
    presets = VERIFICATION_PRESETS if args.mode == "verify" else FULL_PRESETS
    configs = generate_configs(presets, STEP_COUNTS, ELO_BUCKETS)

    logger.info("Scaling experiments: %d runs (%s mode)", len(configs), args.mode)

    if args.dry_run:
        for name, config in configs:
            bs = config.training.batch_size
            ga = config.training.gradient_accumulation_steps
            logger.info(
                "  %s  (bs=%dx%d=%d, lr=%s, steps=%d)",
                name, bs, ga, bs * ga, config.training.learning_rate, config.training.max_steps,
            )
        return

    for i, (name, config) in enumerate(configs):
        config.device = args.device
        logger.info("=== Run %d/%d: %s ===", i + 1, len(configs), name)
        train(config, run_name=name)
        logger.info("=== Completed: %s ===", name)


if __name__ == "__main__":
    main()
