"""Independent method streams and equal-budget statistical comparison checks."""

from pathlib import Path
import runpy

import numpy as np
from numpy.testing import assert_allclose, assert_raises

from mc_pricer import price_monte_carlo


def _experiment():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "experiments/variance_reduction.py"))


def test_antithetic_comparison_replays_distinct_method_streams_and_budgets():
    module = _experiment()
    raw = module["collect_comparison"](64, 3, 991)
    assert len(raw) == 30 and raw.spawn_key.nunique() == 30
    cases = module["study_cases"]()
    for row in raw.itertuples():
        case = cases[row.case_index]
        rng = module["stream"](991, row.case_index, row.replication, int(row.method == "antithetic"))
        expected = price_monte_carlo(case.model, case.product, case.maturity, 64,
                                     n_steps=case.n_steps, rng=rng, variance_reduction=row.method)
        assert_allclose([row.price, row.standard_error, row.estimator_variance],
                        [expected.price, expected.standard_error, expected.estimator_variance], rtol=0, atol=0)
        assert row.n_independent_samples == (32 if row.method == "antithetic" else 64)


def test_antithetic_comparison_variance_ratio_uses_estimates_not_payoff_variance():
    module = _experiment()
    raw = module["collect_comparison"](128, 4, 992)
    summary, comparisons = module["summarize_comparison"](raw)
    for row in comparisons.itertuples():
        runs = raw.loc[raw.case == row.case]
        plain = runs.loc[runs.method == "none", "price"].to_numpy()
        anti = runs.loc[runs.method == "antithetic", "price"].to_numpy()
        assert_allclose(row.empirical_vrf, np.var(plain, ddof=1) / np.var(anti, ddof=1))
        assert row.vrf_ci_lower < row.empirical_vrf < row.vrf_ci_upper
        assert_allclose(row.difference_standard_error, np.sqrt((np.var(plain,ddof=1)+np.var(anti,ddof=1))/4))
    unknown = summary.loc[summary.case.isin(("asian_call_12", "barrier_call_96"))]
    assert unknown.empirical_bias.isna().all() and unknown.rmse.isna().all()
    bad = raw.copy()
    bad.loc[bad.method == "antithetic", "n_paths"] = 256
    with assert_raises(ValueError):
        module["summarize_comparison"](bad)
    degenerate = raw.copy()
    degenerate[["price", "standard_error", "sample_variance", "estimator_variance"]] = 0.
    _, undefined = module["summarize_comparison"](degenerate)
    assert undefined.empirical_vrf.isna().all() and undefined.reported_variance_vrf.isna().all()


def test_antithetic_study_rejects_invalid_design():
    collect = _experiment()["collect_comparison"]
    for paths, runs, seed in ((3,4,1),(2,4,1),(8,1,1),(8,4,-1)):
        with assert_raises(ValueError):
            collect(paths,runs,seed)
    with assert_raises(TypeError):
        collect(8.,4,1)


def test_symmetric_barrier_counterexample_has_positive_population_covariance():
    row = _experiment()["symmetric_window_benchmark"](10000).iloc[0]
    assert_allclose(row.quadrature_price, row.analytical_price, atol=1e-12, rtol=1e-12)
    assert row.pair_covariance > 0 and row.pair_correlation > 0
    assert row.antithetic_estimator_variance > row.plain_estimator_variance
    assert 0 < row.population_vrf_quadrature < 1
    assert row.max_quadrature_reported_absolute_error < 1e-9
