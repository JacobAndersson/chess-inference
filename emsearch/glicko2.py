"""Glicko-2 rating calculation for evaluating model strength.

Implementation based on Mark Glickman's Glicko-2 paper:
http://www.glicko.net/glicko/glicko2.pdf
"""

import math

# Glicko-2 constants
TAU = 0.5  # system volatility constraint
EPSILON = 1e-6
CONVERGENCE_TOLERANCE = 1e-6
GLICKO2_SCALE = 173.7178


def _g(phi: float) -> float:
    """Compute g(phi) = 1 / sqrt(1 + 3*phi^2 / pi^2)."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)


def _expected_score(mu: float, mu_j: float, phi_j: float) -> float:
    """Compute expected score E(mu, mu_j, phi_j)."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _compute_variance(mu: float, opponents: list[tuple[float, float]]) -> float:
    """Compute estimated variance of the player's rating (v)."""
    total = 0.0
    for mu_j, phi_j in opponents:
        g_val = _g(phi_j)
        e_val = _expected_score(mu, mu_j, phi_j)
        total += g_val**2 * e_val * (1.0 - e_val)
    return 1.0 / total if total > 0 else 1e10


def _compute_delta(
    mu: float,
    opponents: list[tuple[float, float]],
    scores: list[float],
    v: float,
) -> float:
    """Compute delta = v * sum(g(phi_j) * (s_j - E))."""
    total = 0.0
    for (mu_j, phi_j), s_j in zip(opponents, scores, strict=True):
        total += _g(phi_j) * (s_j - _expected_score(mu, mu_j, phi_j))
    return v * total


def _new_volatility(
    sigma: float,
    phi: float,
    v: float,
    delta: float,
) -> float:
    """Compute new volatility using the Illinois algorithm (Section 5.4)."""
    a = math.log(sigma**2)
    delta_sq = delta**2
    phi_sq = phi**2

    def f(x: float) -> float:
        ex = math.exp(x)
        top = ex * (delta_sq - phi_sq - v - ex)
        bottom = 2.0 * (phi_sq + v + ex) ** 2
        return top / bottom - (x - a) / TAU**2

    # Initial bounds
    big_a = a
    if delta_sq > phi_sq + v:
        big_b = math.log(delta_sq - phi_sq - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        big_b = a - k * TAU

    fa = f(big_a)
    fb = f(big_b)

    while abs(big_b - big_a) > CONVERGENCE_TOLERANCE:
        big_c = big_a + (big_a - big_b) * fa / (fb - fa)
        fc = f(big_c)
        if fc * fb <= 0:
            big_a = big_b
            fa = fb
        else:
            fa /= 2.0
        big_b = big_c
        fb = fc

    return math.exp(big_a / 2.0)


def rate(
    rating: float,
    rd: float,
    volatility: float,
    opponents: list[tuple[float, float]],
    scores: list[float],
) -> tuple[float, float, float]:
    """Update a player's Glicko-2 rating.

    Args:
        rating: Current Glicko rating (e.g. 1500)
        rd: Current rating deviation (e.g. 350)
        volatility: Current volatility (e.g. 0.06)
        opponents: List of (rating, rd) for each opponent
        scores: List of scores (1.0=win, 0.5=draw, 0.0=loss)

    Returns:
        Tuple of (new_rating, new_rd, new_volatility)
    """
    if not opponents:
        # No games: RD increases
        phi = rd / GLICKO2_SCALE
        new_phi = math.sqrt(phi**2 + volatility**2)
        return rating, new_phi * GLICKO2_SCALE, volatility

    # Convert to Glicko-2 scale
    mu = (rating - 1500.0) / GLICKO2_SCALE
    phi = rd / GLICKO2_SCALE
    opp_glicko2 = [((_r - 1500.0) / GLICKO2_SCALE, _rd / GLICKO2_SCALE) for _r, _rd in opponents]

    v = _compute_variance(mu, opp_glicko2)
    delta = _compute_delta(mu, opp_glicko2, scores, v)

    new_sigma = _new_volatility(volatility, phi, v, delta)

    phi_star = math.sqrt(phi**2 + new_sigma**2)
    new_phi = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
    new_mu = mu + new_phi**2 * sum(
        _g(phi_j) * (s_j - _expected_score(mu, mu_j, phi_j))
        for (mu_j, phi_j), s_j in zip(opp_glicko2, scores, strict=True)
    )

    new_rating = new_mu * GLICKO2_SCALE + 1500.0
    new_rd = new_phi * GLICKO2_SCALE

    return new_rating, new_rd, new_sigma


def rate_from_stockfish_results(
    results: dict[int, dict],
    initial_rating: float = 1500.0,
    initial_rd: float = 350.0,
    initial_volatility: float = 0.06,
) -> tuple[float, float, float]:
    """Compute Glicko-2 rating from Stockfish evaluation results.

    Args:
        results: Dict mapping Stockfish ELO -> {games: [{result: "win"/"draw"/"loss", ...}]}
        initial_rating: Starting rating
        initial_rd: Starting rating deviation
        initial_volatility: Starting volatility

    Returns:
        Tuple of (rating, rd, volatility)
    """
    opponents: list[tuple[float, float]] = []
    scores: list[float] = []

    score_map = {"win": 1.0, "draw": 0.5, "loss": 0.0}

    for elo, data in results.items():
        sf_rd = 50.0  # Stockfish has low uncertainty at a given level
        for game in data.get("games", []):
            result = game.get("result", "loss")
            opponents.append((float(elo), sf_rd))
            scores.append(score_map.get(result, 0.0))

    return rate(initial_rating, initial_rd, initial_volatility, opponents, scores)
