"""Deterministic checks of convergence diagnostics and their denominators."""

import numpy as np
from numpy.testing import assert_allclose, assert_raises
from scipy.stats import norm

from mc_pricer.statistics import (
    binomial_wilson_interval, fit_power_law, summarize_replications,
)


def test_power_law_recovers_known_exponent_and_units():
    sizes = np.array([100, 300, 1000, 10000], dtype=float)
    values = 3.25 / np.sqrt(sizes)
    original = values.copy()
    result = fit_power_law(sizes, values)
    assert_allclose([result.intercept, result.slope, result.r_squared], [np.log(3.25), -.5, 1], atol=1e-14)
    assert result.n_points == 4
    scaled = fit_power_law(sizes, 100 * values)
    assert_allclose(scaled.slope, result.slope, atol=1e-14)
    assert_allclose(scaled.intercept - result.intercept, np.log(100), atol=1e-14)
    assert_allclose(values, original, rtol=0, atol=0)


def test_constant_power_law_has_undefined_r_squared():
    result = fit_power_law([10, 20, 30, 40, 50], [7.3] * 5)
    assert result.slope == 0 and result.r_squared is None


def test_power_law_rejects_invalid_log_regression_inputs():
    for sizes, values in [([1, 2], [1, 2]), ([1, 2, 3], [1, 2, 3, 4]),
                          ([0, 1, 2], [3, 2, 1]), ([1, 1, 2], [3, 2, 1]),
                          ([3, 2, 1], [3, 2, 1]), ([1, 2, 3], [0, 1, 2]),
                          ([1, 2, 3], [np.nan, 1, 2]), ([[1, 2, 3]], [1, 2, 3])]:
        with assert_raises(ValueError):
            fit_power_law(sizes, values)
    with assert_raises(TypeError):
        fit_power_law([1, 2, 3], [1j, 2j, 3j])


def test_replication_summary_matches_hand_calculation_and_rmse_identity():
    estimates = np.array([1.0, 3.0, 5.0])
    result = summarize_replications(estimates, target=2)
    assert result.n_replications == 3
    assert_allclose([result.mean_estimate, result.empirical_bias, result.empirical_variance,
                     result.rmse, result.standard_error_of_mean],
                    [3, 1, 4, np.sqrt(11 / 3), np.sqrt(4 / 3)])
    assert_allclose(result.rmse ** 2,
                    result.empirical_bias ** 2 + (2 / 3) * result.empirical_variance)
    assert_allclose(result.bias_confidence_interval,
                    np.array([1, 1]) + np.array([-1, 1]) * norm.ppf(.975) * np.sqrt(4 / 3))
    assert_allclose(estimates, [1, 3, 5], rtol=0, atol=0)


def test_constant_replications_have_exact_zero_observed_dispersion():
    result = summarize_replications([.1] * 7, target=.1)
    assert result.empirical_variance == result.standard_error_of_mean == result.rmse == 0
    assert result.bias_confidence_interval == (0, 0)


def test_replication_summary_input_validation():
    for values in ([1], [1, np.inf], [[1, 2]]):
        with assert_raises(ValueError):
            summarize_replications(values, 0)
    with assert_raises(ValueError):
        summarize_replications([1, 2], np.nan)
    with assert_raises(ValueError):
        summarize_replications([1, 2], 0, confidence_level=1)


def test_wilson_interval_solves_binomial_score_equation():
    # Invert (p_hat - p)^2 = z^2 p(1-p)/R, independently of the closed form.
    k, n, level = 37, 50, .95
    lower, upper = binomial_wilson_interval(k, n, level)
    z = norm.ppf((1 + level) / 2)
    assert 0 < lower < k / n < upper < 1
    for endpoint in (lower, upper):
        assert_allclose((k / n - endpoint) ** 2, z * z * endpoint * (1 - endpoint) / n, atol=1e-15)
    complement = binomial_wilson_interval(n - k, n)
    assert_allclose(complement, [1 - upper, 1 - lower])


def test_wilson_interval_remains_informative_at_boundaries():
    z2 = norm.ppf(.975) ** 2
    assert_allclose(binomial_wilson_interval(0, 10), [0, z2 / (10 + z2)], atol=1e-15)
    assert_allclose(binomial_wilson_interval(10, 10), [10 / (10 + z2), 1], atol=1e-15)
    assert binomial_wilson_interval(0, 10)[0] == 0
    assert binomial_wilson_interval(10, 10)[1] == 1
    narrower = binomial_wilson_interval(95, 100, .90)
    wider = binomial_wilson_interval(95, 100, .99)
    assert wider[0] < narrower[0] < narrower[1] < wider[1]


def test_wilson_interval_input_validation():
    for successes, trials in [(-1, 10), (11, 10), (0, 0)]:
        with assert_raises(ValueError):
            binomial_wilson_interval(successes, trials)
    for successes, trials in [(True, 10), (1.0, 10), (1, 10.0)]:
        with assert_raises(TypeError):
            binomial_wilson_interval(successes, trials)
    with assert_raises(ValueError):
        binomial_wilson_interval(1, 10, confidence_level=0)
