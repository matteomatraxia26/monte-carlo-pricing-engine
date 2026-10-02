"""Discrete arithmetic Asian payoff, pricing and exact moment validation."""

from dataclasses import FrozenInstanceError, replace
from math import exp, expm1, log, sqrt

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

from mc_pricer import (
    ArithmeticAsianOption, BlackScholesModel, EuropeanOption,
    arithmetic_average_moments, black_scholes_price, make_rng,
    price_from_path_samples, price_from_payoff_samples, price_monte_carlo, simulate_paths,
)


def test_asian_hand_computed_payoffs_exclude_initial_column():
    paths = [[999, 80, 100], [7, 100, 100], [1, 110, 130], [42, 0, 0]]
    call, put = ArithmeticAsianOption(100), ArithmeticAsianOption(100, "put")
    assert_array_equal(call.average(paths), [90, 100, 120, 0])
    assert_array_equal(call.payoff(paths), [0, 0, 20, 0])
    assert_array_equal(put.payoff(paths), [10, 0, 0, 100])
    assert_array_equal(call.payoff_from_average([90, 100, 120, 0]), call.payoff(paths))


def test_asian_payoff_shape_immutability_and_noncontiguous_views():
    source = np.arange(80, 140, dtype=float).reshape(5, 12)
    before = source.copy()
    view = source[::2, ::2]
    view.setflags(write=False)
    for kind in ("call", "put"):
        result = ArithmeticAsianOption(100, kind).payoff(view)
        assert result.shape == (3,) and result.dtype == np.float64
        assert not np.shares_memory(result, source)
    assert_array_equal(source, before)


def test_asian_product_validation_and_immutability():
    for strike in (0, -1, np.inf, np.nan):
        with assert_raises(ValueError):
            ArithmeticAsianOption(strike)
    for strike in (True, "100", [100]):
        with assert_raises(TypeError):
            ArithmeticAsianOption(strike)
    for kind in ("CALL", "digital", None):
        with assert_raises(ValueError):
            ArithmeticAsianOption(100, kind)
    with assert_raises(FrozenInstanceError):
        ArithmeticAsianOption(100).strike = 110


def test_asian_rejects_invalid_paths_and_averages():
    product = ArithmeticAsianOption(100)
    for paths in ([], [100, 110], [[100]], [[100, -1]], [[100, np.nan]], np.empty((0, 3))):
        with assert_raises(ValueError):
            product.payoff(paths)
    for paths in ([[True, False]], [["100", "110"]], [[100 + 1j, 110]]):
        with assert_raises(TypeError):
            product.payoff(paths)
    for averages in ([], [-1, 100], [[100]], [np.inf]):
        with assert_raises(ValueError):
            product.payoff_from_average(averages)


def test_pathwise_asian_parity_and_convexity_bounds():
    paths = simulate_paths(BlackScholesModel(100, .05, .2), 1, 1000, 12, rng=make_rng(716))
    call, put = ArithmeticAsianOption(100), ArithmeticAsianOption(100, "put")
    assert_allclose(call.payoff(paths) - put.payoff(paths), call.average(paths) - 100, atol=1e-14, rtol=0)
    assert np.all(call.payoff(paths) <= np.maximum(paths[:, 1:] - 100, 0).mean(axis=1) + 1e-13)
    assert np.all(put.payoff(paths) <= np.maximum(100 - paths[:, 1:], 0).mean(axis=1) + 1e-13)


def test_asian_pricing_arithmetic_counts_independent_paths():
    # Means 80, 100, 110, 130: discounted call payoffs 0, 0, 5, 15.
    paths = [[100, 80, 80], [100, 100, 100], [100, 100, 120], [100, 120, 140]]
    result = price_from_path_samples(ArithmeticAsianOption(100), paths, rate=log(2), maturity=1)
    assert_allclose([result.price, result.sample_variance, result.standard_error], [5, 50, sqrt(12.5)])
    assert result.n_paths == result.n_independent_samples == 4  # Not 4*2 observations.
    assert result.n_steps == 2


def test_generic_payoff_summary_supports_signed_paired_differences():
    payoffs = np.array([-1.0, 0, 1, 2])
    result = price_from_payoff_samples(payoffs, rate=log(2), maturity=1)
    assert_allclose([result.price, result.sample_variance, result.standard_error], [.25, 5 / 12, sqrt(5 / 48)])
    assert_array_equal(payoffs, [-1, 0, 1, 2])
    assert result.confidence_interval[0] < 0
    for values in ([1], [1, np.inf], [[1, 2]]):
        with assert_raises(ValueError):
            price_from_payoff_samples(values, rate=0, maturity=1)


def test_asian_engine_replays_supplied_paths_and_rng_state():
    model, product = BlackScholesModel(100, .05, .2), ArithmeticAsianOption(100, "put")
    rng_a, rng_b = make_rng(27), make_rng(27)
    paths = simulate_paths(model, 1, 512, 12, rng=rng_a)
    expected = price_from_path_samples(product, paths, rate=model.rate, maturity=1, confidence_level=.99)
    actual = price_monte_carlo(model, product, 1, 512, n_steps=12, rng=rng_b, confidence_level=.99)
    assert replace(actual, runtime_seconds=0) == replace(expected, runtime_seconds=0)
    assert_array_equal(rng_a.standard_normal(5), rng_b.standard_normal(5))


def test_single_observation_asian_is_exactly_european():
    model = BlackScholesModel(100, .05, .2)
    for kind in ("call", "put"):
        asian = price_monte_carlo(model, ArithmeticAsianOption(100, kind), 1, 10_000,
                                  n_steps=1, rng=make_rng(382))
        european = price_monte_carlo(model, EuropeanOption(100, kind), 1, 10_000, rng=make_rng(382))
        assert replace(asian, n_steps=None, runtime_seconds=0) == replace(european, runtime_seconds=0)


def test_asian_expiry_and_zero_volatility_have_deterministic_prices():
    for rate, volatility, maturity in ((.05, .2, 0), (.05, 0, 1), (-.03, 0, 2), (0, 0, 1)):
        model = BlackScholesModel(105, rate, volatility)
        average = np.mean(105 * np.exp(rate * np.linspace(0, maturity, 13)[1:]))
        for kind, sign in (("call", 1), ("put", -1)):
            result = price_monte_carlo(model, ArithmeticAsianOption(100, kind), maturity, 17,
                                       n_steps=12, rng=make_rng(9))
            assert_allclose(result.price, exp(-rate * maturity) * max(sign * (average - 100), 0), atol=2e-13)
            assert result.standard_error == 0


def test_asian_grid_validation_fails_before_consuming_rng():
    model, asian = BlackScholesModel(100, .05, .2), ArithmeticAsianOption(100)
    for steps, error in ((None, ValueError), (0, ValueError), (-1, ValueError), (True, TypeError), (12.0, TypeError)):
        rng_a, rng_b = make_rng(16), make_rng(16)
        with assert_raises(error):
            price_monte_carlo(model, asian, 1, 100, n_steps=steps, rng=rng_a)
        assert_array_equal(rng_a.standard_normal(4), rng_b.standard_normal(4))
    rng_a, rng_b = make_rng(16), make_rng(16)
    with assert_raises(ValueError):
        price_monte_carlo(model, EuropeanOption(100), 1, 100, n_steps=12, rng=rng_a)
    assert_array_equal(rng_a.standard_normal(4), rng_b.standard_normal(4))
    with assert_raises(ValueError):
        price_from_path_samples(asian, [[100, 110]], rate=.05, maturity=1)
    with assert_raises(TypeError):
        price_from_path_samples(EuropeanOption(100), [[100, 110], [100, 90]], rate=.05, maturity=1)


def test_average_moments_match_explicit_covariance_double_sum():
    for rate in (-.03, 0, .05):
        model = BlackScholesModel(90, rate, .3)
        times = np.linspace(0, 2, 5)[1:]
        expected_mean = sum(90 * exp(rate * t) for t in times) / 4
        expected_variance = sum(90**2 * exp(rate * (s + t)) * expm1(.3**2 * min(s, t))
                                for s in times for t in times) / 16
        actual = arithmetic_average_moments(model, 2, 4)
        assert_allclose([actual.mean, actual.variance], [expected_mean, expected_variance], rtol=3e-15)


def test_average_moments_single_date_and_deterministic_limits():
    model = BlackScholesModel(100, .05, .2)
    single = arithmetic_average_moments(model, 1, 1)
    assert_allclose([single.mean, single.variance], [100 * exp(.05), 100**2 * exp(.1) * expm1(.04)])
    assert arithmetic_average_moments(model, 0, 12).variance == 0
    assert arithmetic_average_moments(model, 0, 12).mean == 100
    assert arithmetic_average_moments(replace(model, volatility=0), 1, 12).variance == 0


def test_average_moments_input_validation():
    model = BlackScholesModel(100, .05, .2)
    for maturity, count in ((-1, 12), (np.nan, 12), (1, 0)):
        with assert_raises(ValueError):
            arithmetic_average_moments(model, maturity, count)
    for count in (True, 12.0):
        with assert_raises(TypeError):
            arithmetic_average_moments(model, 1, count)
    with assert_raises(TypeError):
        arithmetic_average_moments(None, 1, 12)


def test_asian_mean_and_price_parity_against_exact_targets():
    model = BlackScholesModel(100, .05, .2)
    count, steps = 100_000, 12
    paths = simulate_paths(model, 1, count, steps, rng=make_rng(20260927))
    call, put = ArithmeticAsianOption(100), ArithmeticAsianOption(100, "put")
    average = call.average(paths)
    moments = arithmetic_average_moments(model, 1, steps)
    assert abs(average.mean() - moments.mean) < 6 * sqrt(moments.variance / count)
    c = price_from_path_samples(call, paths, rate=model.rate, maturity=1)
    p = price_from_path_samples(put, paths, rate=model.rate, maturity=1)
    assert_allclose(c.price - p.price, exp(-.05) * (average.mean() - 100), atol=5e-14, rtol=0)
    assert abs(c.price - p.price - exp(-.05) * (moments.mean - 100)) < 6 * exp(-.05) * sqrt(moments.variance / count)
    times = np.linspace(0, 1, steps + 1)[1:]
    for kind, result, sign in (("call", c, 1), ("put", p, -1)):
        lower = exp(-.05) * max(sign * (moments.mean - 100), 0)
        upper = sum(exp(-.05 * (1 - t)) * black_scholes_price(model, 100, t, kind) for t in times) / steps
        assert lower - 6 * result.standard_error <= result.price <= upper + 6 * result.standard_error
