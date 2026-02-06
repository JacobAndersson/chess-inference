"""Test suite for playing the model against Stockfish at different ELO levels."""

import argparse
import logging
from pathlib import Path

import chess
import chess.engine
import torch

from emsearch.chess_dataset import ENCODE_MAP, EOS_TOKEN, PAD_TOKEN, decode
from emsearch.eval_checkpoint import load_model_from_checkpoint
from emsearch.model import ChessTransformer
from emsearch.utils import setup_logging

logger = logging.getLogger("emsearch")

STOCKFISH_PATH = "/usr/games/stockfish"
SPACE_TOKEN = ENCODE_MAP[" "]


def _encode_move_text(text: str) -> list[int]:
    """Encode a move string (e.g. 'e4') to token IDs."""
    return [ENCODE_MAP[c] for c in text if c in ENCODE_MAP]


def _game_tokens_to_text(tokens: list[int]) -> str:
    """Convert token list to PGN-like move text."""
    return decode(tokens)


def _extract_model_move(
    model: ChessTransformer,
    context_tokens: list[int],
    device: torch.device,
    board: chess.Board,
) -> chess.Move | None:
    """Generate the next move from the model given the game context.

    Generates tokens until a space or EOS is produced, then tries to
    parse the result as a legal chess move.
    """
    max_new = 10  # a single move is at most ~7 chars (e.g. "Qxd8+# ")

    # Use space token as seed for first move since model needs at least one input token
    seed = context_tokens or [SPACE_TOKEN]

    input_ids = torch.tensor([seed], dtype=torch.long, device=device)

    generated = model.generate(input_ids, max_new_tokens=max_new, temperature=0.1, top_k=5)
    new_tokens = generated[0, len(seed) :].tolist()

    # Extract tokens up to space or EOS
    move_tokens = []
    for t in new_tokens:
        if t in (SPACE_TOKEN, EOS_TOKEN, PAD_TOKEN):
            break
        move_tokens.append(t)

    move_text = decode(move_tokens)

    try:
        move = board.parse_san(move_text)
    except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
        logger.warning("Model produced illegal move: '%s' in position %s", move_text, board.fen())
        return None
    else:
        return move


def play_game(
    model: ChessTransformer,
    device: torch.device,
    stockfish_path: str,
    stockfish_elo: int,
    model_plays_white: bool = True,
    max_moves: int = 200,
) -> dict:
    """Play a single game between model and Stockfish.

    Returns dict with result info.
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})

    board = chess.Board()
    context_tokens: list[int] = []
    move_count = 0
    illegal_moves = 0
    result_reason = "normal"

    while not board.is_game_over() and move_count < max_moves:
        is_model_turn = (board.turn == chess.WHITE) == model_plays_white

        if is_model_turn:
            move = _extract_model_move(model, context_tokens, device, board)
            if move is None:
                illegal_moves += 1
                if illegal_moves >= 3:
                    result_reason = "illegal_moves"
                    break
                # Pick a random legal move as fallback
                move = next(iter(board.legal_moves))
        else:
            sf_result = engine.play(board, chess.engine.Limit(time=0.1))
            move = sf_result.move

        san = board.san(move)
        if context_tokens:
            context_tokens.append(SPACE_TOKEN)
        context_tokens.extend(_encode_move_text(san))

        board.push(move)
        move_count += 1

    engine.quit()

    # Determine result
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            game_result = "draw"
        elif outcome.winner == model_plays_white:
            game_result = "win"
        else:
            game_result = "loss"
    elif result_reason == "illegal_moves":
        game_result = "loss"
    else:
        game_result = "draw"  # max moves reached

    return {
        "result": game_result,
        "moves": move_count,
        "illegal_moves": illegal_moves,
        "reason": result_reason,
        "pgn": _game_tokens_to_text(context_tokens),
        "model_color": "white" if model_plays_white else "black",
        "stockfish_elo": stockfish_elo,
    }


def run_stockfish_suite(
    model: ChessTransformer,
    device: torch.device,
    stockfish_path: str = STOCKFISH_PATH,
    elo_levels: list[int] | None = None,
    games_per_level: int = 10,
) -> dict:
    """Run the full Stockfish evaluation suite.

    Plays games at each ELO level with the model as both white and black.
    """
    if elo_levels is None:
        elo_levels = [1350, 1500, 1800, 2000, 2200]

    model.eval()
    results = {}

    for elo in elo_levels:
        level_results = {"wins": 0, "draws": 0, "losses": 0, "games": [], "illegal_total": 0}

        for i in range(games_per_level):
            model_white = i % 2 == 0
            game = play_game(model, device, stockfish_path, elo, model_white)
            level_results["games"].append(game)
            level_results[f"{game['result']}s"] += 1
            level_results["illegal_total"] += game["illegal_moves"]

            logger.info(
                "ELO %d game %d/%d: %s (%s as %s, %d moves)",
                elo,
                i + 1,
                games_per_level,
                game["result"],
                game["reason"],
                game["model_color"],
                game["moves"],
            )

        total = games_per_level
        win_rate = level_results["wins"] / total
        draw_rate = level_results["draws"] / total
        loss_rate = level_results["losses"] / total

        logger.info(
            "ELO %d summary: W=%.0f%% D=%.0f%% L=%.0f%% (illegal=%d)",
            elo,
            win_rate * 100,
            draw_rate * 100,
            loss_rate * 100,
            level_results["illegal_total"],
        )

        results[elo] = {
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "illegal_total": level_results["illegal_total"],
            "games": level_results["games"],
        }

    return results


def main() -> None:
    """CLI entry point for Stockfish evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model against Stockfish")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--stockfish", type=str, default=STOCKFISH_PATH)
    parser.add_argument(
        "--elo-levels",
        type=int,
        nargs="+",
        default=[1350, 1500, 1800, 2000, 2200],
    )
    parser.add_argument("--games-per-level", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging("stockfish_eval")
    device = torch.device(args.device)

    model, checkpoint = load_model_from_checkpoint(Path(args.checkpoint), device)
    step = checkpoint.get("step", 0)
    logger.info("Loaded checkpoint at step %d (%s params)", step, f"{model.count_parameters():,}")

    results = run_stockfish_suite(
        model, device, args.stockfish, args.elo_levels, args.games_per_level
    )

    logger.info("=== Final Results ===")
    for elo, data in sorted(results.items()):
        logger.info(
            "  ELO %4d: W=%.0f%% D=%.0f%% L=%.0f%%",
            elo,
            data["win_rate"] * 100,
            data["draw_rate"] * 100,
            data["loss_rate"] * 100,
        )


if __name__ == "__main__":
    main()
