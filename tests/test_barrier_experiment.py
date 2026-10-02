"""Calendar nesting, stored summaries and paired barrier diagnostics."""

from pathlib import Path
import runpy

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

from mc_pricer import BlackScholesModel, UpAndOutCall, simulate_paths


def _experiment():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "experiments/barrier_pricing.py"))


def test_barrier_study_replays_initial_and_nested_calendar_maxima():
    model, product = BlackScholesModel(100,.05,.2), UpAndOutCall(100,120)
    terminal, maxima, raw = _experiment()["collect_extrema"](
        model,1,product,(1,2,4),n_paths=32,replications=3,seed=88)
    assert terminal.shape == (3,32) and maxima.shape == (3,32,3)
    assert len(raw) == 9
    for run in range(3):
        paths = simulate_paths(model,1,32,4,rng=np.random.default_rng(np.random.SeedSequence(88,spawn_key=(run,))))
        assert_array_equal(terminal[run],paths[:,-1])
        for column, dates in enumerate(([0,4],[0,2,4],[0,1,2,3,4])):
            expected = np.array([max(path[j] for j in dates) for path in paths])
            assert_array_equal(maxima[run,:,column],expected)
    assert np.all(np.diff(maxima,axis=2)>=0)
    assert not np.array_equal(terminal[0],terminal[1])


def test_barrier_summary_prices_paired_differences_and_missed_hits():
    terminal = np.array([[110.,115.],[105.,90.]])
    maxima = np.array([[[110.,125.],[115.,115.]],[[105.,120.],[100.,100.]]])
    table = _experiment()["summarize_extrema"](terminal,maxima,BlackScholesModel(100,0,.2),
                                               1,UpAndOutCall(100,120),(1,2))
    coarse,fine = table.iloc[0],table.iloc[1]
    assert coarse.n_paths == 4 and coarse.price == 7.5 and fine.price == 3.75
    assert coarse.difference_to_reference == 3.75
    assert_allclose(coarse.difference_standard_error,np.sqrt(275/48),rtol=1e-15)
    assert coarse.missed_reference_hit_fraction == .5
    assert coarse.knockout_fraction == 0 and fine.knockout_fraction == .5
    assert fine.difference_to_reference == fine.difference_standard_error == 0
    assert (table.max_pathwise_in_out_residual == 0).all()
    assert np.isnan(fine.terminal_only_exact_price)


def test_barrier_study_rejects_non_nested_grids_and_invalid_run_counts():
    collect = _experiment()["collect_extrema"]
    args = (BlackScholesModel(100,.05,.2),1,UpAndOutCall(100,120))
    # 2 and 3 both divide 6, but are not nested with each other.
    for counts in ((2,3,6),(1,4,4),(0,4),(4,2),(1,)):
        with assert_raises(ValueError):
            collect(*args,counts,n_paths=4,replications=2,seed=1)
    with assert_raises(TypeError):
        collect(*args,(1,4.),n_paths=4,replications=2,seed=1)
    with assert_raises(ValueError):
        collect(*args,(1,4),n_paths=1,replications=2,seed=1)
