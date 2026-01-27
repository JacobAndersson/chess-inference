"""Streaming dataset for tokenized chess games."""

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

VOCAB_SIZE = 32
EOS_TOKEN = 30
PAD_TOKEN = 31

VOCAB_CHARS = "abcdefgh12345678KQRBNx+#=O- 0/"
ENCODE_MAP = {c: i for i, c in enumerate(VOCAB_CHARS)}


def encode(text: str) -> list[int]:
    """Encode a chess move sequence to token IDs."""
    return [ENCODE_MAP[c] for c in text if c in ENCODE_MAP]


def encode_with_eos(text: str) -> list[int]:
    """Encode a chess move sequence with EOS token appended."""
    return [*encode(text), EOS_TOKEN]


def decode(tokens: list[int]) -> str:
    """Decode token IDs back to text."""
    return "".join(
        VOCAB_CHARS[t] for t in tokens if t < len(VOCAB_CHARS) and t not in (EOS_TOKEN, PAD_TOKEN)
    )


class ChessTokenDataset(IterableDataset):
    """Streaming dataset for tokenized chess games."""

    def __init__(
        self,
        data_dir: Path | str,
        elo_bucket: str = "all",
        max_seq_len: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Directory containing token files
            elo_bucket: ELO bucket to load ("1200", "1500", "1800", "2000", "2500", "all")
            max_seq_len: Maximum sequence length (truncates if exceeded)
        """
        self.data_dir = Path(data_dir)
        self.elo_bucket = elo_bucket
        self.max_seq_len = max_seq_len
        self.file_path = self.data_dir / f"tokens_elo_{elo_bucket}.txt"

        if not self.file_path.exists():
            msg = f"Token file not found: {self.file_path}"
            raise FileNotFoundError(msg)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over tokenized games, yielding one tensor per game."""
        with self.file_path.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                tokens = [int(t) for t in line.split(",")]

                if self.max_seq_len is not None and len(tokens) > self.max_seq_len:
                    tokens = tokens[: self.max_seq_len]

                yield torch.tensor(tokens, dtype=torch.long)


def collate_chess_games(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collate function that pads sequences to the max length in the batch."""
    max_len = max(len(seq) for seq in batch)

    padded = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, seq in enumerate(batch):
        padded[i, : len(seq)] = seq
        attention_mask[i, : len(seq)] = True

    return {
        "input_ids": padded,
        "attention_mask": attention_mask,
    }


def create_dataloader(
    data_dir: Path | str,
    elo_bucket: str = "all",
    batch_size: int = 32,
    max_seq_len: int | None = None,
    num_workers: int = 0,
    shuffle_buffer: int | None = None,  # noqa: ARG001
) -> DataLoader:
    """Create a DataLoader for tokenized chess games.

    Args:
        data_dir: Directory containing token files
        elo_bucket: ELO bucket to load
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle_buffer: If provided, shuffle using a buffer of this size (not implemented)

    Returns:
        DataLoader yielding batches with 'input_ids' and 'attention_mask'
    """
    dataset = ChessTokenDataset(data_dir, elo_bucket, max_seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_chess_games,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chess_dataset.py <data_dir> [elo_bucket]")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    elo_bucket = sys.argv[2] if len(sys.argv) > 2 else "all"

    print(f"Loading data from {data_dir}")
    print(f"ELO bucket: {elo_bucket}")

    loader = create_dataloader(data_dir, elo_bucket, batch_size=4)

    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")

        for j in range(min(2, len(batch["input_ids"]))):
            tokens = batch["input_ids"][j].tolist()
            decoded = decode(tokens)
            print(f"  Game {j}: {decoded[:50]}...")

        if i >= 2:
            break

    print("\nDataset test passed!")
