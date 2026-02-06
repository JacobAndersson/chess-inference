"""Evaluate saved model checkpoints."""

import argparse
import json
import logging
from pathlib import Path

import torch

from emsearch.config import ModelConfig
from emsearch.evaluate import run_phase_accuracy
from emsearch.model import ChessTransformer
from emsearch.train import run_validation
from emsearch.utils import log_metrics, setup_logging

logger = logging.getLogger("emsearch")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ChessTransformer, dict]:
    """Load a model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config_dict = checkpoint["config"]
    model_config = ModelConfig(**config_dict["model"])

    model = ChessTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    elo_bucket: str,
    batch_size: int,
    device: torch.device,
    max_batches: int = 50,
) -> dict[str, float]:
    """Run full evaluation on a single checkpoint."""
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    step = checkpoint.get("step", 0)

    logger.info("Evaluating checkpoint: %s (step %d)", checkpoint_path.name, step)
    logger.info("Model params: %s", f"{model.count_parameters():,}")

    val_loss = run_validation(model, data_dir, elo_bucket, batch_size, device, max_batches)
    phase_metrics = run_phase_accuracy(
        model, str(data_dir), elo_bucket, batch_size, device, max_batches
    )

    return {"step": step, "eval/loss": val_loss, **phase_metrics}


def main() -> None:
    """CLI entry point for evaluating checkpoints."""
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint file(s) to evaluate")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--elo-bucket", type=str, default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    setup_logging("eval_checkpoints")
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    all_results = []
    for cp_path in sorted(args.checkpoints):
        results = evaluate_checkpoint(
            Path(cp_path), data_dir, args.elo_bucket, args.batch_size, device, args.max_batches
        )
        log_metrics(results, results["step"], use_wandb=False)
        all_results.append(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
