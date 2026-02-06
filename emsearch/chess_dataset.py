"""Streaming dataset for tokenized chess games."""

import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

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


def _read_token_lines(
    file_path: Path,
    max_seq_len: int | None,
    max_games: int | None,
) -> Iterator[list[int]]:
    """Read tokenized games from a text file, yielding token lists."""
    games_yielded = 0
    with file_path.open() as f:
        for raw_line in f:
            if max_games is not None and games_yielded >= max_games:
                return

            line = raw_line.strip()
            if not line:
                continue

            tokens = [int(t) for t in line.split(",")]

            if max_seq_len is not None and len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]

            games_yielded += 1
            yield tokens


def _shuffled_iterator(
    source: Iterator[list[int]],
    buffer_size: int,
) -> Iterator[list[int]]:
    """Yield items from source in shuffled order using a buffer."""
    buf: list[list[int]] = []

    for item in source:
        buf.append(item)
        if len(buf) >= buffer_size:
            idx = random.randrange(len(buf))  # noqa: S311
            buf[idx], buf[-1] = buf[-1], buf[idx]
            yield buf.pop()

    random.shuffle(buf)
    yield from buf


class ChessTokenDataset(IterableDataset):
    """Streaming dataset for tokenized chess games."""

    def __init__(
        self,
        data_dir: Path | str,
        elo_bucket: str = "all",
        max_seq_len: int | None = None,
        max_games: int | None = None,
        split: str = "train",
        shuffle_buffer_size: int = 0,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Directory containing token files
            elo_bucket: ELO bucket to load ("1200", "1500", "1800", "2000", "2500", "all")
            max_seq_len: Maximum sequence length (truncates if exceeded)
            max_games: Maximum number of games to load (None for all)
            split: "train" or "test"
            shuffle_buffer_size: Size of shuffle buffer (0 to disable)
        """
        self.data_dir = Path(data_dir)
        self.elo_bucket = elo_bucket
        self.max_seq_len = max_seq_len
        self.max_games = max_games
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size
        self.file_path = self.data_dir / f"tokens_elo_{elo_bucket}_{split}.txt"

        if not self.file_path.exists():
            msg = f"Token file not found: {self.file_path}"
            raise FileNotFoundError(msg)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over tokenized games, yielding one tensor per game."""
        source = _read_token_lines(self.file_path, self.max_seq_len, self.max_games)

        if self.shuffle_buffer_size > 0:
            source = _shuffled_iterator(source, self.shuffle_buffer_size)

        for tokens in source:
            yield torch.tensor(tokens, dtype=torch.long)


class PackedChessDataset(IterableDataset):
    """Streaming dataset that packs multiple games into fixed-length blocks.

    Games are concatenated with EOS delimiters into blocks of `block_size` tokens.
    No padding is used - every token is meaningful.
    """

    def __init__(
        self,
        data_dir: Path | str,
        block_size: int = 1023,
        elo_bucket: str = "all",
        max_games: int | None = None,
        split: str = "train",
        shuffle_buffer_size: int = 0,
    ) -> None:
        """Initialize the packed dataset.

        Args:
            data_dir: Directory containing token files
            block_size: Fixed block size for packed sequences
            elo_bucket: ELO bucket to load
            max_games: Maximum number of games to load
            split: "train" or "test"
            shuffle_buffer_size: Size of shuffle buffer (0 to disable)
        """
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.elo_bucket = elo_bucket
        self.max_games = max_games
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size
        self.file_path = self.data_dir / f"tokens_elo_{elo_bucket}_{split}.txt"

        if not self.file_path.exists():
            msg = f"Token file not found: {self.file_path}"
            raise FileNotFoundError(msg)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield fixed-size blocks of packed games."""
        source = _read_token_lines(self.file_path, None, self.max_games)

        if self.shuffle_buffer_size > 0:
            source = _shuffled_iterator(source, self.shuffle_buffer_size)

        buffer: list[int] = []

        for game_tokens in source:
            buffer.extend(game_tokens)

            while len(buffer) >= self.block_size:
                yield torch.tensor(buffer[: self.block_size], dtype=torch.long)
                buffer = buffer[self.block_size :]

        # Drop the remainder (incomplete block) to avoid padding


class BinaryChessDataset(Dataset):
    """Memory-mapped dataset for binary token files.

    Binary files contain raw token bytes with EOS (30) delimiting games.
    Supports packed block mode for efficient training.
    """

    def __init__(
        self,
        data_dir: Path | str,
        block_size: int = 1023,
        elo_bucket: str = "all",
        split: str = "train",
    ) -> None:
        """Initialize the binary dataset.

        Args:
            data_dir: Directory containing binary token files
            block_size: Fixed block size for yielded sequences
            elo_bucket: ELO bucket to load
            split: "train" or "test"
        """
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.file_path = self.data_dir / f"tokens_elo_{elo_bucket}_{split}.bin"

        if not self.file_path.exists():
            msg = f"Binary token file not found: {self.file_path}"
            raise FileNotFoundError(msg)

        self.data = np.memmap(self.file_path, dtype=np.uint8, mode="r")
        self.num_blocks = len(self.data) // block_size

    def __len__(self) -> int:
        """Return number of complete blocks."""
        return self.num_blocks

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a fixed-size block of tokens."""
        start = idx * self.block_size
        end = start + self.block_size
        return torch.from_numpy(self.data[start:end].astype(np.int64))


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


def collate_packed_blocks(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collate function for packed blocks (no padding needed)."""
    return {"input_ids": torch.stack(batch)}


def create_dataloader(
    data_dir: Path | str,
    elo_bucket: str = "all",
    batch_size: int = 32,
    max_seq_len: int | None = None,
    max_games: int | None = None,
    split: str = "train",
    num_workers: int = 1,
    packed: bool = False,
    block_size: int = 1023,
    shuffle_buffer_size: int = 10_000,
) -> DataLoader:
    """Create a DataLoader for tokenized chess games.

    Args:
        data_dir: Directory containing token files
        elo_bucket: ELO bucket to load
        batch_size: Batch size
        max_seq_len: Maximum sequence length (only for unpacked mode)
        max_games: Maximum number of games to load
        split: "train" or "test"
        num_workers: Number of data loading workers
        packed: Whether to use game packing
        block_size: Block size for packed mode
        shuffle_buffer_size: Size of shuffle buffer (0 to disable)

    Returns:
        DataLoader yielding batches with 'input_ids' (and 'attention_mask' if unpacked)
    """
    if packed:
        dataset = PackedChessDataset(
            data_dir, block_size, elo_bucket, max_games, split, shuffle_buffer_size
        )
        collate_fn = collate_packed_blocks
    else:
        dataset = ChessTokenDataset(
            data_dir, elo_bucket, max_seq_len, max_games, split, shuffle_buffer_size
        )
        collate_fn = collate_chess_games

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
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
        if "attention_mask" in batch:
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")

        for j in range(min(2, len(batch["input_ids"]))):
            tokens = batch["input_ids"][j].tolist()
            decoded = decode(tokens)
            print(f"  Game {j}: {decoded[:50]}...")

        if i >= 2:
            break

    print("\nDataset test passed!")
