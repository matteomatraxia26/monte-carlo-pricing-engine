"""Small integration checks of the independent-stream research design."""

import json
from pathlib import Path
import runpy

import numpy as np
from numpy.testing import assert_allclose, assert_raises

from mc_pricer import BlackScholesModel, EuropeanOption, price_from_terminal_samples, simulate_terminal


def _experiment():
    source = Path(__file__).resolve().parents[1] / "experiments" / "mc_convergence.py"
    return runpy.run_path(str(source))


def test_replication_streams_replay_from_recorded_spawn_keys():
    module = _experiment()
    model = BlackScholesModel(100, .05, .2)
    seed = 7142
    raw = module["collect_replications"](model, 100, 1, [128, 256, 512], n_replications=3, seed=seed)
    assert len(raw) == 18
    assert raw["spawn_key"].nunique() == 9
    for row in raw.itertuples(index=False):
        stream = np.random.SeedSequence(seed, spawn_key=tuple(json.loads(row.spawn_key)))
        terminal = simulate_terminal(model, 1, row.n_paths, rng=np.random.default_rng(stream))
        reference = price_from_terminal_samples(EuropeanOption(100, row.product), terminal, rate=.05, maturity=1)
        assert_allclose([row.estimate, row.standard_error, row.ci_lower, row.ci_upper],
                        [reference.price, reference.standard_error, *reference.confidence_interval], rtol=0, atol=0)
    assert raw.loc[raw["product"] == "call", "estimate"].nunique() == 9


def test_study_summary_preserves_replication_counts_and_denominators():
    module = _experiment()
    raw = module["collect_replications"](BlackScholesModel(100, .05, .2), 100, 1,
                                          [128, 256, 512], n_replications=4, seed=8192)
    summary, fits, curves = module["summarize_study"](raw)
    assert len(summary) == 6 and len(fits) == 2 and len(curves) == 8
    assert (summary["n_replications"] == 4).all()
    for row in summary.itertuples(index=False):
        group = raw.loc[(raw["product"] == row.product) & (raw["n_paths"] == row.n_paths)]
        errors = group["estimate"].to_numpy() - row.analytical_value
        assert_allclose(row.rmse ** 2, np.mean(errors ** 2))
        assert_allclose(row.rmse ** 2, row.empirical_bias ** 2 + .75 * row.empirical_variance)
        assert_allclose(row.rms_reported_se ** 2, row.mean_reported_variance)
        assert row.covered_runs == int(group["covers_target"].sum())
        assert row.empirical_coverage == row.covered_runs / 4
        assert row.coverage_wilson_lower <= row.empirical_coverage <= row.coverage_wilson_upper


def test_study_rejects_invalid_design_before_simulation():
    collect = _experiment()["collect_replications"]
    for counts, replications, seed in [([1, 2, 3], 3, 1), ([2, 2, 3], 3, 1),
                                       ([2, 3], 3, 1), ([2, 3, 4], 1, 1), ([2, 3, 4], 3, -1)]:
        with assert_raises(ValueError):
            collect(BlackScholesModel(100, .05, .2), 100, 1, counts,
                    n_replications=replications, seed=seed)
    with assert_raises(TypeError):
        collect(BlackScholesModel(100, .05, .2), 100, 1, [2, 3, 4], n_replications=3, seed=None)
