"""Deterministic estimator arithmetic and fixed-seed European price validation."""

from dataclasses import asdict, FrozenInstanceError, replace
from math import exp, log, sqrt

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

from mc_pricer import (
    BlackScholesModel, EuropeanOption, black_scholes_price, make_rng,
    price_from_terminal_samples, price_monte_carlo, simulate_terminal,
)
from mc_pricer.statistics import normal_confidence_interval


def test_discounted_sample_statistics_against_hand_calculation():
    # Raw payoff [0, 0, 10, 30]; discount 1/2; X=[0,0,5,15].
    result = price_from_terminal_samples(EuropeanOption(100), [80, 100, 110, 130],
                                         rate=log(2), maturity=1)
    assert_allclose(result.price, 5, rtol=0, atol=1e-14)
    assert_allclose(result.sample_variance, 50, rtol=0, atol=1e-13)
    assert_allclose(result.estimator_variance, 12.5, rtol=0, atol=1e-13)
    assert_allclose(result.standard_error, sqrt(12.5), rtol=0, atol=1e-14)
    half_width = 1.959963984540054 * sqrt(12.5)
    assert_allclose(result.confidence_interval, (5-half_width, 5+half_width), rtol=0, atol=1e-13)
    assert result.confidence_interval[0] < 0  # Normal intervals are not clipped.
    assert result.n_paths == result.n_independent_samples == 4
    assert result.confidence_level == 0.95
    assert result.runtime_seconds >= 0


def test_normal_interval_known_quantiles_and_zero_se():
    assert_allclose(normal_confidence_interval(10, 2),
                    (6.080072030919892, 13.919927969080108), rtol=0, atol=1e-13)
    assert_allclose(normal_confidence_interval(10, 2, 0.99),
                    (4.848341392902199, 15.1516586070978), rtol=0, atol=1e-12)
    assert normal_confidence_interval(-3, 0) == (-3, -3)
    assert np.all(np.isfinite(normal_confidence_interval(0, 1, np.nextafter(1.0, 0.0))))


def test_confidence_interval_input_validation():
    for level in (0, 1, -0.1, 1.1, np.nan, np.inf):
        with assert_raises(ValueError):
            normal_confidence_interval(1, 0.1, level)
    for estimate, se in ((np.nan, 1), (np.inf, 1), (1, -1), (1, np.inf)):
        with assert_raises(ValueError):
            normal_confidence_interval(estimate, se)
    for estimate, se, level in ((True, 1, 0.95), (1, "1", 0.95), (1, 1, True)):
        with assert_raises(TypeError):
            normal_confidence_interval(estimate, se, level)
    with assert_raises(FloatingPointError):
        normal_confidence_interval(1e308, 1e308)


def test_discount_scales_price_se_and_variance_correctly():
    product, terminal = EuropeanOption(100), [80, 100, 110, 130]
    baseline = price_from_terminal_samples(product, terminal, rate=0, maturity=2)
    for rate in (-0.03, 0.07):
        result = price_from_terminal_samples(product, terminal, rate=rate, maturity=2)
        discount = exp(-rate*2)
        assert_allclose(result.price, discount*baseline.price, rtol=1e-14)
        assert_allclose(result.standard_error, discount*baseline.standard_error, rtol=1e-14)
        assert_allclose(result.sample_variance, discount**2*baseline.sample_variance, rtol=1e-14)


def test_engine_and_supplied_sample_route_agree():
    model, product = BlackScholesModel(100, 0.05, 0.2), EuropeanOption(100)
    terminal = simulate_terminal(model, 1, 10_000, rng=make_rng(42))
    expected = price_from_terminal_samples(product, terminal, rate=0.05, maturity=1)
    actual = price_monte_carlo(model, product, 1, 10_000, rng=make_rng(42))
    left, right = asdict(expected), asdict(actual)
    left.pop("runtime_seconds")
    right.pop("runtime_seconds")
    assert left == right


def test_price_reproducibility_excludes_wall_clock_runtime():
    model, product = BlackScholesModel(100, 0.05, 0.2), EuropeanOption(100, "put")
    first = price_monte_carlo(model, product, 1, 1000, rng=make_rng(33))
    second = price_monte_carlo(model, product, 1, 1000, rng=make_rng(33))
    assert replace(first, runtime_seconds=0) == replace(second, runtime_seconds=0)


def test_result_is_immutable_and_supplied_prices_are_unchanged():
    source = np.array([70, 80, 90, 100, 110, 120, 130], dtype=float)
    snapshot = source.copy()
    view = source[::2]
    view.setflags(write=False)
    result = price_from_terminal_samples(EuropeanOption(100), view, rate=0.05, maturity=1)
    assert_array_equal(source, snapshot)
    with assert_raises(FrozenInstanceError):
        result.price = 0


def test_constant_and_zero_payoffs_have_zero_empirical_variance():
    for terminal, expected in (([110]*17, 10*exp(-0.05)), ([80]*17, 0.0)):
        result = price_from_terminal_samples(EuropeanOption(100), terminal, rate=0.05, maturity=1)
        assert_allclose(result.price, expected, rtol=1e-15)
        assert result.sample_variance == result.standard_error == result.estimator_variance == 0
        assert result.confidence_interval == (result.price, result.price)


def test_expiry_and_zero_volatility_match_analytical_prices():
    for model, maturity in ((BlackScholesModel(110, 0.05, 0.2), 0),
                            (BlackScholesModel(100, 0.05, 0), 1),
                            (BlackScholesModel(100, -0.05, 0), 1)):
        for kind in ("call", "put"):
            result = price_monte_carlo(model, EuropeanOption(100, kind), maturity, 32, rng=make_rng(5))
            assert_allclose(result.price, black_scholes_price(model, 100, maturity, kind),
                            rtol=0, atol=2e-13)
            assert result.standard_error == 0


def test_invalid_pricer_metadata_fails_before_consuming_draws():
    model, product = BlackScholesModel(100, 0.05, 0.2), EuropeanOption(100)
    cases = [(-1, 10, 0.95, ValueError), (np.nan, 10, 0.95, ValueError),
             (1, 1, 0.95, ValueError), (1, 0, 0.95, ValueError),
             (1, 2.0, 0.95, TypeError), (1, True, 0.95, TypeError),
             (1, 10, 1, ValueError), (1, 10, np.nan, ValueError)]
    for maturity, count, level, error in cases:
        rng, reference = make_rng(123), make_rng(123)
        with assert_raises(error):
            price_monte_carlo(model, product, maturity, count, rng=rng, confidence_level=level)
        assert_array_equal(rng.standard_normal(10), reference.standard_normal(10))


def test_supplied_sample_pricer_validation():
    product = EuropeanOption(100)
    for terminal in ([], [100], [[100, 110]], [np.nan, 100], [-1, 100]):
        with assert_raises(ValueError):
            price_from_terminal_samples(product, terminal, rate=0.05, maturity=1)
    for rate, maturity in ((np.nan, 1), (np.inf, 1), (0.05, -1), (0.05, np.inf)):
        with assert_raises(ValueError):
            price_from_terminal_samples(product, [100, 110], rate=rate, maturity=maturity)
    for rate in (-1000, 1000):
        with assert_raises(FloatingPointError):
            price_from_terminal_samples(product, [100, 110], rate=rate, maturity=1)
    with assert_raises(FloatingPointError):
        price_from_terminal_samples(product, [1e308, 1e308], rate=-100, maturity=1)


def test_supported_model_product_and_rng_types_are_explicit():
    model, product = BlackScholesModel(100, 0.05, 0.2), EuropeanOption(100)
    for args in ((None, product, make_rng(1)), (model, None, make_rng(1)), (model, product, 1)):
        with assert_raises(TypeError):
            price_monte_carlo(args[0], args[1], 1, 10, rng=args[2])
    with assert_raises(TypeError):
        price_from_terminal_samples(None, [100, 110], rate=0.05, maturity=1)


def test_call_and_put_prices_are_close_to_black_scholes_in_se_units():
    # Do not demand that an approximate 95% CI covers on every random run.
    model = BlackScholesModel(100, 0.05, 0.2)
    for kind in ("call", "put"):
        result = price_monte_carlo(model, EuropeanOption(100, kind), 1, 200_000,
                                   rng=make_rng(20260924))
        expected = black_scholes_price(model, 100, 1, kind)
        assert result.standard_error > 0
        assert abs(result.price-expected) < 6*result.standard_error
        assert_allclose(result.standard_error**2, result.estimator_variance, rtol=2e-15)


def test_shared_sample_price_parity_equals_discounted_stock_minus_strike():
    model = BlackScholesModel(100, 0.05, 0.2)
    terminal = simulate_terminal(model, 1, 20_000, rng=make_rng(76))
    call = price_from_terminal_samples(EuropeanOption(100, "call"), terminal, rate=0.05, maturity=1)
    put = price_from_terminal_samples(EuropeanOption(100, "put"), terminal, rate=0.05, maturity=1)
    assert_allclose(call.price-put.price, exp(-0.05)*(terminal.mean()-100), rtol=0, atol=5e-14)
    # Sample parity is not necessarily exactly S0 - K exp(-rT).
