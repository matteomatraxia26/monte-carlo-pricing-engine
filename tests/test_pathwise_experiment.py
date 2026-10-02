"""Pathwise experiment: transparent costs and distinct independent/paired designs."""

from pathlib import Path
import json
import runpy

import numpy as np
from numpy.testing import assert_allclose, assert_raises

from mc_pricer import price_from_payoff_samples


def _experiment():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/'experiments/pathwise_delta.py'))


def test_pathwise_comparison_replays_independent_cells_and_both_budgets():
    module=_experiment()
    raw=module['collect_comparison'](128,3,415)
    assert len(raw)==24 and raw.spawn_key.nunique()==24
    for row in raw.itertuples():
        mi=[x[0] for x in module['METHODS']].index(row.method)
        result=module['_estimate'](mi,row.n_paths,module['stream'](415,tuple(json.loads(row.spawn_key))))
        assert result.estimate==row.estimate and result.standard_error==row.standard_error
        if row.budget=='equal_rows':
            assert row.n_paths==128
            assert row.n_terminal_evaluations==(128 if mi==0 else 256)
        else:
            assert row.n_terminal_evaluations==256
            assert row.n_paths==(256 if mi==0 else 128)


def test_pathwise_comparison_summarizes_run_variances_and_preserves_fd_bias():
    module=_experiment()
    raw=module['collect_comparison'](256,4,416)
    summary=module['summarize'](raw)
    comparison=module['compare_variance'](summary)
    assert summary.loc[summary.method=='pathwise','deterministic_bias'].eq(0).all()
    assert (summary.loc[summary.method=='fd_h10','deterministic_bias'].abs()>.008).all()
    for row in comparison.itertuples():
        group=raw.loc[raw.budget==row.budget]
        numerator=group.loc[group.method==row.method,'estimate'].var(ddof=1)
        denominator=group.loc[group.method=='pathwise','estimate'].var(ddof=1)
        assert_allclose(row.fd_over_pw_variance,numerator/denominator)
        assert row.ratio_ci_lower<row.fd_over_pw_variance<row.ratio_ci_upper
    assert_allclose(summary.rmse**2,summary.empirical_bias**2+.75*summary.empirical_variance,rtol=1e-12)
    degenerate=summary.copy()
    degenerate[['empirical_variance','mean_reported_estimator_variance']]=0.
    undefined=module['compare_variance'](degenerate)
    assert undefined.fd_over_pw_variance.isna().all() and undefined.reported_variance_ratio.isna().all()
    bad=summary.copy()
    bad.loc[(bad.budget=='equal_terminal_evaluations')&(bad.method=='pathwise'),'n_terminal_evaluations']=128
    with assert_raises(ValueError):
        module['compare_variance'](bad)


def test_pathwise_paired_study_uses_difference_samples_and_independent_replications():
    module=_experiment()
    raw=module['collect_paired'](256,4,417)
    assert len(raw)==24 and raw.spawn_key.nunique()==4  # Same stream across bumps by design.
    for row in raw.itertuples():
        z=module['stream'](417,tuple(json.loads(row.spawn_key))).standard_normal(256)
        pw,fd=module['paired_samples'](z,row.bump)
        expected=price_from_payoff_samples(fd-pw,rate=.05,maturity=1.)
        assert row.fd_minus_pw==expected.price and row.paired_standard_error==expected.standard_error
    summary=module['summarize_paired'](raw)
    for row in summary.itertuples():
        values=raw.loc[raw.bump==row.bump,'fd_minus_pw'].to_numpy()
        assert_allclose(row.mean_difference,values.mean())
        assert_allclose(row.empirical_standard_error_of_mean,np.std(values,ddof=1)/2)
    convergence=module['collect_convergence'](3,417,(32,64,128))
    assert len(convergence)==18 and convergence.spawn_key.nunique()==18
    for args in ((1,4,1),(4,1,1),(4,4,-1)):
        with assert_raises(ValueError):
            module['collect_comparison'](*args)
    with assert_raises(TypeError):
        module['collect_comparison'](4.,4,1)
    with assert_raises(ValueError):
        module['collect_convergence'](4,1,(128,64,32))
