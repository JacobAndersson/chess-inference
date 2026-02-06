"""Main training loop for chess transformer."""

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.nn import functional as fn
from tqdm import tqdm

from emsearch.chess_dataset import PAD_TOKEN, create_dataloader
from emsearch.config import ExperimentConfig, TrainingConfig
from emsearch.evaluate import run_phase_accuracy
from emsearch.model import ChessTransformer
from emsearch.presets import get_preset
from emsearch.utils import (
    configure_optimizer,
    get_lr_scheduler,
    log_metrics,
    save_checkpoint,
    save_config,
    set_seed,
    setup_logging,
    setup_wandb,
)

logger = logging.getLogger("emsearch")


def compute_loss(
    model: ChessTransformer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute cross-entropy loss on a batch."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    logits = model(input_ids, attention_mask)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss = fn.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=PAD_TOKEN,
        reduction="none",
    )

    # Mask out padding and compute mean
    loss = loss.view(shift_labels.shape)
    return (loss * shift_mask).sum() / shift_mask.sum()


@torch.no_grad()
def run_validation(
    model: ChessTransformer,
    data_dir: Path,
    elo_bucket: str,
    batch_size: int,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Run validation on test set and return loss."""
    model.eval()
    loader = create_dataloader(
        data_dir,
        elo_bucket=elo_bucket,
        batch_size=batch_size,
        max_seq_len=model.config.max_seq_len,
        split="test",
    )

    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        loss = fn.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=PAD_TOKEN,
            reduction="none",
        )
        loss = loss.view(shift_labels.shape)

        total_loss += (loss * shift_mask).sum().item()
        total_tokens += shift_mask.sum().item()

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def _setup_model(config: ExperimentConfig, device: torch.device) -> ChessTransformer:
    """Initialize and optionally compile the model."""
    model = ChessTransformer(config.model).to(device)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def _run_training_step(
    model: ChessTransformer,
    data_iter: iter,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int,
) -> tuple[iter, float, int]:
    """Run one training step with gradient accumulation."""
    optimizer.zero_grad()
    accum_loss = 0.0
    total_tokens = 0

    for _ in range(grad_accum_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss = compute_loss(model, batch, device)
            loss = loss / grad_accum_steps

        loss.backward()
        accum_loss += loss.item() * grad_accum_steps
        total_tokens += batch["attention_mask"].sum().item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return data_iter, accum_loss, total_tokens


def train(config: ExperimentConfig, run_name: str | None = None) -> None:
    """Main training function."""
    set_seed(config.seed)

    if run_name is None:
        run_name = f"elo{config.training.elo_bucket}"

    log_file = setup_logging(run_name)
    logger.info("Run: %s", run_name)
    logger.info("Log file: %s", log_file)
    logger.info("Config: %s", config)

    device = torch.device(config.device)
    data_dir = Path(config.training.data_dir)
    checkpoint_dir = Path(config.training.checkpoint_dir)

    model = _setup_model(config, device)
    optimizer = configure_optimizer(
        model, config.training.learning_rate, config.training.weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, config.training.warmup_steps, config.training.max_steps)
    use_wandb = setup_wandb(config.training.wandb_project, config, run_name)

    save_config(checkpoint_dir / "config.json", config)

    train_loader = create_dataloader(
        data_dir,
        elo_bucket=config.training.elo_bucket,
        batch_size=config.training.batch_size,
        max_seq_len=config.model.max_seq_len,
        max_games=config.training.max_games,
        split="train",
    )

    model.train()
    step = 0
    accum_loss = 0.0
    accum_count = 0
    tokens_since_log = 0
    time_since_log = time.time()

    pbar = tqdm(total=config.training.max_steps, desc="Training")
    data_iter = iter(train_loader)

    while step < config.training.max_steps:
        data_iter, loss, tokens = _run_training_step(
            model,
            data_iter,
            train_loader,
            optimizer,
            device,
            config.training.gradient_accumulation_steps,
        )
        scheduler.step()
        step += 1
        pbar.update(1)

        accum_loss += loss
        accum_count += 1
        tokens_since_log += tokens

        if step % config.training.log_every == 0:
            elapsed = time.time() - time_since_log
            log_metrics(
                {
                    "train/loss": accum_loss / accum_count,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/tokens_per_sec": tokens_since_log / elapsed if elapsed > 0 else 0,
                },
                step,
                use_wandb,
            )
            accum_loss, accum_count, tokens_since_log = 0.0, 0, 0
            time_since_log = time.time()

        if step % config.training.eval_every == 0:
            val_loss = run_validation(
                model, data_dir, config.training.elo_bucket, config.training.batch_size, device
            )
            log_metrics({"eval/loss": val_loss}, step, use_wandb)

            phase_metrics = run_phase_accuracy(
                model, data_dir, config.training.elo_bucket, config.training.batch_size, device
            )
            log_metrics(phase_metrics, step, use_wandb)

        if step % config.training.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"step_{step}.pt", model, optimizer, scheduler, config, step, 0.0
            )

    pbar.close()
    save_checkpoint(checkpoint_dir / "final.pt", model, optimizer, scheduler, config, step, 0.0)
    logger.info("Training complete. Checkpoints saved to %s", checkpoint_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train chess transformer")
    parser.add_argument("--preset", type=str, default="50m", help="Model size preset")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--elo-bucket", type=str, default="all", help="ELO bucket")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to use")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this run")
    args = parser.parse_args()

    model_config = get_preset(args.preset)

    training_config = TrainingConfig(
        data_dir=args.data_dir,
        elo_bucket=args.elo_bucket,
        max_games=args.max_games,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        wandb_project=args.wandb_project,
    )

    config = ExperimentConfig(
        model=model_config,
        training=training_config,
        seed=args.seed,
        device=args.device,
        compile=not args.no_compile,
    )

    train(config, run_name=args.run_name)


if __name__ == "__main__":
    main()
