from emsearch.glicko2 import rate, rate_from_stockfish_results


def test_rate_no_games():
    rating, rd, vol = rate(1500, 350, 0.06, [], [])
    assert rating == 1500
    assert rd > 350  # RD increases with no games
    assert vol == 0.06


def test_rate_all_wins():
    opponents = [(1400, 50)] * 10
    scores = [1.0] * 10
    rating, _rd, _vol = rate(1500, 200, 0.06, opponents, scores)
    assert rating > 1500


def test_rate_all_losses():
    opponents = [(1600, 50)] * 10
    scores = [0.0] * 10
    rating, _rd, _vol = rate(1500, 200, 0.06, opponents, scores)
    assert rating < 1500


def test_rate_mixed_results():
    opponents = [(1400, 50), (1600, 50), (1500, 50)]
    scores = [1.0, 0.0, 0.5]
    rating, _rd, _vol = rate(1500, 200, 0.06, opponents, scores)
    assert abs(rating - 1500) < 100


def test_rd_decreases_with_games():
    opponents = [(1500, 50)] * 20
    scores = [0.5] * 20
    _, rd, _ = rate(1500, 350, 0.06, opponents, scores)
    assert rd < 350


def test_rate_from_stockfish_results_all_wins():
    results = {
        1500: {
            "games": [{"result": "win"} for _ in range(10)],
        }
    }
    rating, _rd, _vol = rate_from_stockfish_results(results)
    assert rating > 1500


def test_rate_from_stockfish_results_all_losses():
    results = {
        2000: {
            "games": [{"result": "loss"} for _ in range(10)],
        }
    }
    rating, _rd, _vol = rate_from_stockfish_results(results)
    assert rating < 1500


def test_rate_from_stockfish_results_multiple_levels():
    results = {
        1350: {
            "games": [{"result": "win"} for _ in range(10)],
        },
        1800: {
            "games": [{"result": "loss"} for _ in range(10)],
        },
    }
    rating, _rd, _vol = rate_from_stockfish_results(results)
    assert 1350 < rating < 1800


def test_rate_from_stockfish_results_empty():
    results = {}
    rating, rd, _vol = rate_from_stockfish_results(results)
    assert rating == 1500
    assert rd > 350


def test_glicko2_paper_example():
    opponents = [(1400, 30), (1550, 100), (1700, 300)]
    scores = [1.0, 0.0, 0.0]
    rating, rd, _vol = rate(1500, 200, 0.06, opponents, scores)
    assert 1400 < rating < 1520
    assert 100 < rd < 200
