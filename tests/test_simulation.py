"""Terminal GBM and RNG checks, including fixed-seed statistical diagnostics."""

from dataclasses import replace
from math import exp, expm1, sqrt

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

from mc_pricer import (
    BlackScholesModel, make_rng, simulate_terminal,
    standard_normal_draws, terminal_from_normals,
)


def test_same_seed_replays_draws_and_prices():
    model = BlackScholesModel(100, 0.05, 0.2)
    assert_array_equal(standard_normal_draws(100, rng=make_rng(123)),
                       standard_normal_draws(100, rng=make_rng(123)))
    assert_array_equal(simulate_terminal(model, 1, 100, rng=make_rng(123)),
                       simulate_terminal(model, 1, 100, rng=make_rng(123)))


def test_generator_advances_and_preserves_draw_order():
    rng = make_rng(123)
    first = standard_normal_draws(17, rng=rng)
    second = standard_normal_draws(23, rng=rng)
    expected = make_rng(123).standard_normal(40)
    assert_array_equal(np.concatenate((first, second)), expected)
    # This checks normal draws for this bit generator/environment, not a promise
    # about every distribution, NumPy version or platform.


def test_generator_interleaving_does_not_change_another_stream():
    rng = make_rng(123)
    first = standard_normal_draws(10, rng=rng)
    standard_normal_draws(100, rng=make_rng(456))
    second = standard_normal_draws(10, rng=rng)
    assert_array_equal(np.concatenate((first, second)), make_rng(123).standard_normal(20))


def test_rng_validation():
    for seed in (True, 1.5, "123", [123]):
        with assert_raises(TypeError):
            make_rng(seed)
    with assert_raises(ValueError):
        make_rng(-1)
    assert isinstance(make_rng(None), np.random.Generator)
    assert isinstance(make_rng(np.int64(0)), np.random.Generator)
    with assert_raises(TypeError):
        standard_normal_draws(10, rng=123)


def test_n_paths_validation():
    for count in (0, -1):
        with assert_raises(ValueError):
            standard_normal_draws(count, rng=make_rng(1))
    for count in (True, 1.0, "10", None):
        with assert_raises(TypeError):
            standard_normal_draws(count, rng=make_rng(1))
    assert standard_normal_draws(np.int64(3), rng=make_rng(1)).shape == (3,)


def test_terminal_shape_dtype_and_positivity():
    model = BlackScholesModel(100, -0.02, 0.3)
    for count in (1, 17, 1000):
        terminal = simulate_terminal(model, 2, count, rng=make_rng(987))
        assert terminal.shape == (count,)
        assert terminal.dtype == np.float64
        assert np.all(np.isfinite(terminal))
        assert np.all(terminal > 0)


def test_terminal_formula_from_fixed_shocks():
    model = BlackScholesModel(80, -0.03, 0.3)
    shocks = np.array([-2, -1, 0, 1, 2])
    expected = np.array([80 * exp((-0.03 - 0.3**2 / 2) * 2 + 0.3 * sqrt(2) * z)
                         for z in shocks])
    assert_allclose(terminal_from_normals(model, 2, shocks), expected, rtol=2e-15, atol=0)


def test_wrapper_matches_supplied_shocks_and_rng_state():
    model = BlackScholesModel(100, 0.05, 0.2)
    rng_a, rng_b = make_rng(91), make_rng(91)
    normals = standard_normal_draws(137, rng=rng_a)
    assert_array_equal(simulate_terminal(model, 1, 137, rng=rng_b),
                       terminal_from_normals(model, 1, normals))
    assert_array_equal(rng_a.standard_normal(5), rng_b.standard_normal(5))


def test_supplied_shocks_are_not_modified_or_aliased():
    model = BlackScholesModel(100, 0.05, 0.2)
    original = np.linspace(-2, 2, 21)
    before = original.copy()
    view = original[::2]
    view.setflags(write=False)
    result = terminal_from_normals(model, 1, view)
    assert_array_equal(original, before)
    assert not np.shares_memory(result, original)
    assert_allclose(result, terminal_from_normals(model, 1, view.tolist()), rtol=0, atol=0)


def test_terminal_deterministic_limits():
    model = BlackScholesModel(100, 0.05, 0.2)
    shocks = [-3, 0, 4]
    assert_array_equal(terminal_from_normals(model, 0, shocks), np.full(3, 100.0))
    for rate in (-0.05, 0, 0.05):
        deterministic = replace(model, rate=rate, volatility=0)
        assert_allclose(terminal_from_normals(deterministic, 2, shocks),
                        np.full(3, 100 * exp(2 * rate)), rtol=1e-15, atol=0)


def test_deterministic_limits_still_consume_one_normal_per_path():
    for volatility, maturity in ((0.2, 0), (0, 1), (0, 0)):
        rng_a, rng_b = make_rng(18), make_rng(18)
        simulate_terminal(BlackScholesModel(100, 0.05, volatility), maturity, 31, rng=rng_a)
        rng_b.standard_normal(31)
        assert_array_equal(rng_a.standard_normal(7), rng_b.standard_normal(7))


def test_invalid_maturity_does_not_consume_random_draws():
    model = BlackScholesModel(100, 0.05, 0.2)
    for maturity, error in ((-1, ValueError), (np.nan, ValueError), (np.inf, ValueError),
                            (True, TypeError), ("1", TypeError)):
        rng_a, rng_b = make_rng(4), make_rng(4)
        with assert_raises(error):
            simulate_terminal(model, maturity, 10, rng=rng_a)
        assert_array_equal(rng_a.standard_normal(10), rng_b.standard_normal(10))
        with assert_raises(error):
            terminal_from_normals(model, maturity, [0.0])


def test_invalid_normal_arrays():
    model = BlackScholesModel(100, 0.05, 0.2)
    for normals in ([], 1.0, [[0.0, 1.0]], [np.nan], [np.inf]):
        with assert_raises(ValueError):
            terminal_from_normals(model, 1, normals)
    for normals in ([True], [1 + 2j], ["0.0"], np.array([0], dtype=object)):
        with assert_raises(TypeError):
            terminal_from_normals(model, 1, normals)


def test_unrepresentable_prices_raise_instead_of_returning_zero_or_infinity():
    model = BlackScholesModel(100, 0.05, 0.2)
    for shocks in ([1e6], [-1e6]):
        with assert_raises(FloatingPointError):
            terminal_from_normals(model, 1, shocks)
    with assert_raises(FloatingPointError):
        terminal_from_normals(replace(model, volatility=1e308), 1, [0.0])


def test_shared_shocks_preserve_spot_scaling():
    model = BlackScholesModel(100, 0.05, 0.2)
    shocks = standard_normal_draws(1000, rng=make_rng(13))
    baseline = terminal_from_normals(model, 1, shocks)
    bumped = terminal_from_normals(replace(model, spot=101), 1, shocks)
    assert_allclose(bumped, 1.01 * baseline, rtol=3e-15, atol=0)


def test_terminal_risk_neutral_and_log_return_moments():
    """Fixed seed and conservative six-standard-error bounds, not a 95% CI test."""
    model = BlackScholesModel(100, 0.05, 0.2)
    count, maturity = 200_000, 1.0
    terminal = simulate_terminal(model, maturity, count, rng=make_rng(20260924))
    log_returns = np.log(terminal / model.spot)
    q = model.volatility**2 * maturity
    mean = model.spot * exp(model.rate * maturity)
    variance = mean**2 * expm1(q)
    fourth_central = variance**2 * (exp(4*q) + 2*exp(3*q) + 3*exp(2*q) - 3)
    variance_se = sqrt((fourth_central - (count-3)/(count-1) * variance**2) / count)
    assert abs(np.mean(terminal) - mean) < 6 * sqrt(variance / count)
    assert abs(np.var(terminal, ddof=1) - variance) < 6 * variance_se
    assert abs(np.mean(log_returns) - (model.rate - 0.5*model.volatility**2)*maturity) < 6*sqrt(q/count)
    assert abs(np.var(log_returns, ddof=1) - q) < 6*q*sqrt(2/(count-1))
    discounted = exp(-model.rate*maturity) * terminal
    assert abs(np.mean(discounted) - model.spot) < 6*exp(-model.rate*maturity)*sqrt(variance/count)
