import tempfile
from pathlib import Path

import pytest
import torch

from emsearch.chess_dataset import (
    EOS_TOKEN,
    PAD_TOKEN,
    BinaryChessDataset,
    ChessTokenDataset,
    PackedChessDataset,
    collate_chess_games,
    collate_packed_blocks,
    create_dataloader,
    decode,
    encode,
    encode_with_eos,
)


def _make_token_file(path, games):
    with Path(path).open("w") as f:
        f.writelines(",".join(str(t) for t in game) + "\n" for game in games)


def test_encode_simple():
    tokens = encode("e4")
    assert tokens == [4, 11]


def test_encode_with_eos():
    tokens = encode_with_eos("e4")
    assert tokens == [4, 11, EOS_TOKEN]


def test_decode():
    text = decode([4, 11])
    assert text == "e4"


def test_decode_skips_special_tokens():
    text = decode([4, 11, EOS_TOKEN, PAD_TOKEN])
    assert text == "e4"


def test_chess_token_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        game1 = [4, 11, 27, 4, 12, EOS_TOKEN]  # "e4 e5"
        game2 = [3, 11, 27, 3, 12, EOS_TOKEN]  # "d4 d5"
        _make_token_file(Path(tmpdir) / "tokens_elo_all_train.txt", [game1, game2])

        dataset = ChessTokenDataset(tmpdir, elo_bucket="all", split="train")
        items = list(dataset)
        assert len(items) == 2
        assert items[0].tolist() == game1
        assert items[1].tolist() == game2


def test_chess_token_dataset_max_games():
    with tempfile.TemporaryDirectory() as tmpdir:
        games = [[4, 11, EOS_TOKEN] for _ in range(10)]
        _make_token_file(Path(tmpdir) / "tokens_elo_all_train.txt", games)

        dataset = ChessTokenDataset(tmpdir, elo_bucket="all", split="train", max_games=3)
        items = list(dataset)
        assert len(items) == 3


def test_chess_token_dataset_max_seq_len():
    with tempfile.TemporaryDirectory() as tmpdir:
        game = [*list(range(20)), EOS_TOKEN]
        _make_token_file(Path(tmpdir) / "tokens_elo_all_train.txt", [game])

        dataset = ChessTokenDataset(tmpdir, elo_bucket="all", split="train", max_seq_len=10)
        items = list(dataset)
        assert len(items[0]) == 10


def test_chess_token_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        ChessTokenDataset("/nonexistent", elo_bucket="all", split="train")


def test_shuffle_buffer():
    with tempfile.TemporaryDirectory() as tmpdir:
        games = [[i, EOS_TOKEN] for i in range(100)]
        _make_token_file(Path(tmpdir) / "tokens_elo_all_train.txt", games)

        dataset = ChessTokenDataset(
            tmpdir, elo_bucket="all", split="train", shuffle_buffer_size=50
        )
        items = [t.tolist() for t in dataset]
        assert len(items) == 100

        original = [[i, EOS_TOKEN] for i in range(100)]
        assert items != original


def test_packed_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        games = [[4, 11, 27, 4, 12, EOS_TOKEN] for _ in range(20)]
        _make_token_file(Path(tmpdir) / "tokens_elo_all_train.txt", games)

        dataset = PackedChessDataset(tmpdir, block_size=10, elo_bucket="all", split="train")
        blocks = list(dataset)
        assert len(blocks) > 0
        assert all(len(b) == 10 for b in blocks)
        assert len(blocks) == 12


def test_packed_dataset_drops_remainder():
    with tempfile.TemporaryDirectory() as tmpdir:
        game = [4, 11, EOS_TOKEN]
        _make_token_file(Path(tmpdir) / "tokens_elo_all_train.txt", [game])

        dataset = PackedChessDataset(tmpdir, block_size=10, elo_bucket="all", split="train")
        blocks = list(dataset)
        assert len(blocks) == 0


def test_binary_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = bytes(list(range(100)))
        (Path(tmpdir) / "tokens_elo_all_train.bin").write_bytes(data)

        dataset = BinaryChessDataset(tmpdir, block_size=10, elo_bucket="all", split="train")
        assert len(dataset) == 10
        block = dataset[0]
        assert block.shape == (10,)
        assert block.tolist() == list(range(10))


def test_binary_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        BinaryChessDataset("/nonexistent", block_size=10, elo_bucket="all", split="train")


def test_collate_chess_games():
    batch = [
        torch.tensor([4, 11, EOS_TOKEN]),
        torch.tensor([4, 11, 27, 4, 12, EOS_TOKEN]),
    ]
    result = collate_chess_games(batch)
    assert result["input_ids"].shape == (2, 6)
    assert result["attention_mask"].shape == (2, 6)
    assert result["attention_mask"][0].tolist() == [True, True, True, False, False, False]
    assert result["attention_mask"][1].tolist() == [True, True, True, True, True, True]
    assert result["input_ids"][0, 3].item() == PAD_TOKEN


def test_collate_packed_blocks():
    batch = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([6, 7, 8, 9, 10])]
    result = collate_packed_blocks(batch)
    assert result["input_ids"].shape == (2, 5)
    assert "attention_mask" not in result


def test_create_dataloader_unpacked():
    with tempfile.TemporaryDirectory() as tmpdir:
        games = [[4, 11, 27, 4, 12, EOS_TOKEN] for _ in range(10)]
        _make_token_file(Path(tmpdir) / "tokens_elo_all_train.txt", games)

        loader = create_dataloader(
            tmpdir, batch_size=4, split="train", num_workers=0, shuffle_buffer_size=0
        )
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 4


def test_create_dataloader_packed():
    with tempfile.TemporaryDirectory() as tmpdir:
        games = [[4, 11, 27, 4, 12, EOS_TOKEN] for _ in range(100)]
        _make_token_file(Path(tmpdir) / "tokens_elo_all_train.txt", games)

        loader = create_dataloader(
            tmpdir,
            batch_size=4,
            split="train",
            packed=True,
            block_size=10,
            num_workers=0,
            shuffle_buffer_size=0,
        )
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert "attention_mask" not in batch
        assert batch["input_ids"].shape == (4, 10)
