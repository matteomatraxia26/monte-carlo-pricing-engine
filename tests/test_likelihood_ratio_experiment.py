"""Test M14 moment benchmarks, experimental targets, costs and saved summaries."""
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose, assert_raises
from scipy.integrate import quad
from scipy.stats import norm
from mc_pricer import (
    BlackScholesModel, DigitalOption, EuropeanOption, digital_delta,
    likelihood_ratio_delta, likelihood_ratio_delta_samples, black_scholes_greeks,
)
from experiments import likelihood_ratio as e


def test_lr_tilted_moments_against_independent_quadrature():
    for b in (0., .2, .4):
        for c in (-1., .3, 4.2):
            for upper in (True, False):
                lo, hi = (c, 12.) if upper else (-12., c)
                actual = e.tilted_normal_moments(b, c, upper)
                expected = [quad(lambda z: z**j*np.exp(b*z)*norm.pdf(z), lo, hi, epsabs=1e-13)[0] for j in (1, 2)]
                assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)


def test_lr_population_moments_match_implementation_quadrature_and_greeks():
    for case in (*e.CASES, e.Case('digital_put', digital=True, kind='put')):
        mean, variance = e.lr_population_moments(case)
        a = case.volatility*np.sqrt(case.maturity)
        c = (np.log(case.strike/case.spot)-(case.rate-.5*case.volatility**2)*case.maturity)/a
        lo, hi = (c, 12.) if case.kind=='call' else (-12., c)
        second = quad(lambda z: (np.exp(-case.rate*case.maturity)*likelihood_ratio_delta_samples(case.model,case.product,case.maturity,[z])[0])**2*norm.pdf(z), lo, hi, epsabs=1e-13)[0]
        exact = e.targets(case,'lr')[0]
        assert_allclose(mean, exact, rtol=2e-10, atol=1e-13)
        assert_allclose(variance, second-mean**2, rtol=2e-10, atol=1e-13)


def test_lr_simulation_matches_exact_delta_and_population_variance():
    for ci in (0,1,3):
        case = e.CASES[ci]
        mean, variance = e.lr_population_moments(case)
        out = likelihood_ratio_delta(case.model,case.product,case.maturity,200000,rng=np.random.default_rng(145+ci))
        assert abs(out.estimate-mean)<6*np.sqrt(variance/out.n_paths)
        assert_allclose(out.sample_variance,variance,rtol=.04)


def test_lr_digital_fd_variance_uses_boundary_crossing_probability():
    case = e.CASES[3]
    for method in e.BUMPS:
        exact, mean, variance = e.targets(case,method)
        h = e.BUMPS[method]
        a = case.volatility*np.sqrt(case.maturity)
        drift = (case.rate-.5*case.volatility**2)*case.maturity
        lo = (np.log(case.strike/(case.spot+h))-drift)/a
        hi = (np.log(case.strike/(case.spot-h))-drift)/a
        p = quad(norm.pdf,lo,hi,epsabs=1e-13)[0]
        scale = np.exp(-case.rate*case.maturity)/(2*h)
        assert_allclose(mean,scale*p,rtol=1e-10)
        assert_allclose(variance,scale**2*p*(1-p),rtol=1e-10)
        assert exact==digital_delta(case.model,case.strike,case.maturity)
    small = e.targets(case,'fd_h0.01')[2]
    large = e.targets(case,'fd_h0.1')[2]
    assert 9<small/large<11  # Variance diverges as O(1/h) for this digital.


def test_lr_study_replay_budgets_and_summary_denominators():
    raw = e.collect_comparison(64,3,146)
    assert len(raw)==72
    assert (raw.terminal_evaluations==128).all()
    for row in raw.itertuples():
        mi = e.methods(e.CASES[row.case_index]).index(row.method)
        result = e.estimate(e.CASES[row.case_index],row.method,row.n_paths,e.stream(146,(0,row.case_index,row.replication,mi)))
        assert result.estimate==row.estimate and result.standard_error==row.standard_error
    summary = e.summarize(raw)
    assert len(summary)==24
    assert_allclose(summary.rmse**2,summary.empirical_bias**2+2/3*summary.empirical_variance,rtol=1e-12,atol=1e-18)
    ratios = e.comparisons(raw,146,resamples=50)
    assert len(ratios)==18
    assert set(ratios.reference)=={'lr'}
    broken = raw.copy(); broken.loc[0,'terminal_evaluations']=1
    with assert_raises(ValueError):
        e.comparisons(broken,146,resamples=50)


def test_lr_study_retains_zero_events_and_replays_convergence():
    raw = e.collect_comparison(32,3,147)
    mask = (raw.scenario=='digital_rare')&(raw.method=='lr')
    raw.loc[mask,['estimate','standard_error','ci_lower','ci_upper']]=0.
    summary = e.summarize(raw)
    row = summary.loc[(summary.scenario=='digital_rare')&(summary.method=='lr')].iloc[0]
    assert row.greek_coverage==0 and row.zero_se_rate==1
    assert_allclose(row.rmse,abs(row.analytical_delta))
    convergence = e.collect_convergence(3,148,counts=(32,64,128))
    assert len(convergence)==27
    assert (convergence.n_paths==convergence.terminal_evaluations).all()
    fits = e.slopes(e.summarize(convergence))
    assert len(fits)==3
    for args in ((1,3,0),(32,1,0),(32,3,-1),(True,3,0)):
        with assert_raises((ValueError,TypeError)):
            e.validate_design(*args)
    with assert_raises(ValueError):
        e.targets(e.CASES[3],'pw')
