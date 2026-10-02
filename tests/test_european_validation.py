"""Additional integration checks for the Milestone 4 validation study."""

import runpy
from pathlib import Path

from numpy.testing import assert_allclose

from mc_pricer import (
    BlackScholesModel, EuropeanOption, black_scholes_price, make_rng,
    price_from_terminal_samples, price_monte_carlo, simulate_terminal,
)


def test_validation_rows_replay_from_a_single_largest_terminal_sample():
    """Catch accidental stream continuation/reset errors in the experiment."""
    source = Path(__file__).resolve().parents[1] / "experiments" / "european_validation.py"
    collect_validation = runpy.run_path(str(source))["collect_validation"]
    model = BlackScholesModel(100, 0.05, 0.2)
    counts, seed = (256, 2048), 20260924
    rows = collect_validation(model, 100, 1, counts, seed=seed)
    terminal = simulate_terminal(model, 1, counts[-1], rng=make_rng(seed))
    assert len(rows) == 2 * len(counts)
    for row in rows.itertuples(index=False):
        reference = price_from_terminal_samples(
            EuropeanOption(100, row.product), terminal[:row.n_paths], rate=0.05, maturity=1,
        )
        assert_allclose(row.mc_price, reference.price, rtol=0, atol=0)
        assert_allclose(row.standard_error, reference.standard_error, rtol=0, atol=0)
        assert_allclose((row.ci_lower, row.ci_upper), reference.confidence_interval, rtol=0, atol=0)


def test_european_pricing_beyond_the_at_the_money_baseline():
    """Fixed-seed regression over rates, moneyness, maturities and volatility."""
    cases = [(80.0, -0.02, 0.35, 0.25),
             (100.0, 0.0, 0.20, 2.0),
             (130.0, 0.07, 0.40, 0.5)]
    for spot, rate, volatility, maturity in cases:
        model = BlackScholesModel(spot, rate, volatility)
        for kind in ("call", "put"):
            result = price_monte_carlo(model, EuropeanOption(100, kind), maturity, 100_000,
                                       rng=make_rng(20260925))
            analytical = black_scholes_price(model, 100, maturity, kind)
            assert result.standard_error > 0
            assert abs(result.price - analytical) < 6 * result.standard_error
