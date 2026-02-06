"""Evaluation utilities for chess transformer models."""

import logging

import torch

from emsearch.chess_dataset import create_dataloader
from emsearch.model import ChessTransformer

logger = logging.getLogger("emsearch")

SPACE_TOKEN = 27

# Phase boundaries in half-moves (plies)
OPENING_END = 20  # moves 1-10
MIDDLEGAME_END = 60  # moves 11-30
PHASES = ("opening", "middlegame", "endgame")


def _get_phase_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Map each token position to a game phase (0=opening, 1=middlegame, 2=endgame).

    Phase is determined by counting space tokens (which separate half-moves).
    """
    is_space = (input_ids == SPACE_TOKEN).int()
    ply_count = is_space.cumsum(dim=1)

    phase = torch.zeros_like(ply_count)
    phase[ply_count >= OPENING_END] = 1
    phase[ply_count >= MIDDLEGAME_END] = 2
    return phase


@torch.no_grad()
def run_phase_accuracy(
    model: ChessTransformer,
    data_dir: str,
    elo_bucket: str,
    batch_size: int,
    device: torch.device,
    max_batches: int = 50,
) -> dict[str, float]:
    """Compute move prediction accuracy bucketed by game phase.

    Returns dict with keys like "opening/top1", "middlegame/top5", etc.
    """
    model.eval()
    loader = create_dataloader(
        data_dir,
        elo_bucket=elo_bucket,
        batch_size=batch_size,
        max_seq_len=model.config.max_seq_len,
        split="test",
    )

    # Counters per phase: [correct_top1, correct_top5, total]
    stats = {p: [0, 0, 0] for p in PHASES}

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]
        shift_phases = _get_phase_mask(input_ids)[:, 1:]

        preds_top1 = shift_logits.argmax(dim=-1)
        preds_top5 = shift_logits.topk(min(5, shift_logits.size(-1)), dim=-1).indices

        top1_correct = (preds_top1 == shift_labels) & shift_mask
        top5_correct = (preds_top5 == shift_labels.unsqueeze(-1)).any(dim=-1) & shift_mask

        for phase_idx, phase_name in enumerate(PHASES):
            phase_mask = (shift_phases == phase_idx) & shift_mask
            count = phase_mask.sum().item()
            if count > 0:
                stats[phase_name][0] += (top1_correct & phase_mask).sum().item()
                stats[phase_name][1] += (top5_correct & phase_mask).sum().item()
                stats[phase_name][2] += count

    model.train()

    results = {}
    for phase_name in PHASES:
        correct1, correct5, total = stats[phase_name]
        if total > 0:
            results[f"eval/{phase_name}/top1_acc"] = correct1 / total
            results[f"eval/{phase_name}/top5_acc"] = correct5 / total
            results[f"eval/{phase_name}/tokens"] = total
        else:
            results[f"eval/{phase_name}/top1_acc"] = 0.0
            results[f"eval/{phase_name}/top5_acc"] = 0.0
            results[f"eval/{phase_name}/tokens"] = 0

    return results
