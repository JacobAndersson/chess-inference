"""Hyperparameter sweep runner for chess transformer training."""

import argparse
import itertools
import json
import logging
from pathlib import Path

from emsearch.config import ExperimentConfig, TrainingConfig
from emsearch.presets import get_preset
from emsearch.train import train
from emsearch.utils import setup_logging

logger = logging.getLogger("emsearch")

DEFAULT_SWEEP = {
    "preset": ["5m"],
    "elo_bucket": ["all", "1500", "2000"],
    "learning_rate": [1e-3, 3e-4, 1e-4],
    "max_games": [500_000, None],
}


def load_sweep_config(path: str | None) -> dict:
    """Load sweep config from JSON or use defaults."""
    if path is None:
        return DEFAULT_SWEEP

    with Path(path).open() as f:
        return json.load(f)


def generate_configs(
    sweep: dict,
    data_dir: str,
    base_checkpoint_dir: str,
    max_steps: int,
) -> list[tuple[str, ExperimentConfig]]:
    """Generate all experiment configs from sweep parameter grid."""
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    configs = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo, strict=True))

        preset = params.pop("preset", "5m")
        model_config = get_preset(preset)

        lr = params.pop("learning_rate", 3e-4)
        elo = params.pop("elo_bucket", "all")
        max_games = params.pop("max_games", None)

        games_label = f"{max_games // 1000}k" if max_games else "full"
        run_name = f"{preset}_elo{elo}_{games_label}_lr{lr:.0e}"

        training_config = TrainingConfig(
            data_dir=data_dir,
            elo_bucket=elo,
            max_games=max_games,
            learning_rate=lr,
            max_steps=max_steps,
            checkpoint_dir=f"{base_checkpoint_dir}/{run_name}",
            wandb_project="chess-transformer-sweep",
            **params,
        )

        config = ExperimentConfig(model=model_config, training=training_config)
        configs.append((run_name, config))

    return configs


def main() -> None:
    """CLI entry point for sweep."""
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sweep-config", type=str, default=None, help="JSON sweep config file")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/sweep")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true", help="Print configs without training")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging("sweep")
    sweep = load_sweep_config(args.sweep_config)
    configs = generate_configs(sweep, args.data_dir, args.checkpoint_dir, args.max_steps)

    logger.info("Sweep: %d configurations", len(configs))

    if args.dry_run:
        for run_name, config in configs:
            logger.info("  %s", run_name)
            logger.info(
                "    lr=%s elo=%s games=%s warmup=%d",
                config.training.learning_rate,
                config.training.elo_bucket,
                config.training.max_games,
                config.training.warmup_steps,
            )
        return

    for i, (run_name, config) in enumerate(configs):
        config.device = args.device
        logger.info("=== Run %d/%d: %s ===", i + 1, len(configs), run_name)
        train(config, run_name=run_name)
        logger.info("=== Completed: %s ===", run_name)


if __name__ == "__main__":
    main()
