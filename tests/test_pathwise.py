"""Pathwise Delta: derivative samples, statistics, regularity and stream checks."""

from dataclasses import FrozenInstanceError, replace
import numpy as np
from numpy.testing import assert_allclose, assert_raises
from scipy.integrate import quad
from scipy.stats import norm

from mc_pricer import (
    ArithmeticAsianOption, BlackScholesModel, EuropeanOption, UpAndOutCall,
    black_scholes_greeks, european_pathwise_delta_samples, finite_difference_from_normals,
    pathwise_delta, pathwise_delta_from_terminal_samples, terminal_from_normals,
)

MODEL=BlackScholesModel(100.,.05,.2)


def test_pathwise_call_put_samples_and_strike_tie_convention():
    terminal=np.array([80.,100.,120.,150.])
    call=european_pathwise_delta_samples(EuropeanOption(100.),terminal,spot=100.)
    put=european_pathwise_delta_samples(EuropeanOption(100.,'put'),terminal,spot=100.)
    assert_allclose(call,[0.,0.,1.2,1.5],rtol=0,atol=0)
    assert_allclose(put,[-.8,0.,0.,0.],rtol=0,atol=0)
    assert call.dtype==np.float64 and call.shape==(4,)


def test_pathwise_hand_discounted_statistics_and_result_immutability():
    result=pathwise_delta_from_terminal_samples(BlackScholesModel(100.,np.log(2),.2),
                                               EuropeanOption(100.),1.,[80.,100.,120.,160.])
    x=np.array([0.,0.,.6,.8])
    assert_allclose(result.estimate,x.mean(),rtol=0,atol=1e-16)
    assert_allclose(result.sample_variance,x.var(ddof=1))
    assert_allclose(result.standard_error,np.sqrt(x.var(ddof=1)/4))
    assert result.n_terminal_evaluations==4
    assert_allclose(result.estimator_variance,result.standard_error**2)
    assert_allclose(result.confidence_interval,x.mean()+np.array([-1,1])*norm.ppf(.975)*result.standard_error)
    with assert_raises(FrozenInstanceError):
        result.estimate=1.


def test_pathwise_normal_quadrature_matches_analytical_delta_and_second_moment():
    for spot,rate,volatility,time in ((80.,-.02,.35,.4),(100.,.05,.2,1.),(140.,.01,.4,2.)):
        model=BlackScholesModel(spot,rate,volatility)
        drift=(rate-.5*volatility**2)*time
        scale=volatility*np.sqrt(time)
        cutoff=(np.log(100./spot)-drift)/scale
        d1=(np.log(spot/100.)+(rate+.5*volatility**2)*time)/scale
        for kind in ('call','put'):
            lower,upper=(cutoff,12.) if kind=='call' else (-12.,cutoff)
            sign=1 if kind=='call' else -1
            # Integrate the derivative directly, independently of the vector helper.
            mean,_=quad(lambda z: sign*np.exp(-rate*time+drift+scale*z)*norm.pdf(z),lower,upper,epsabs=1e-11)
            second,_=quad(lambda z: np.exp(-2*rate*time+2*drift+2*scale*z)*norm.pdf(z),lower,upper,epsabs=1e-11)
            assert_allclose(mean,black_scholes_greeks(model,100.,time,kind).delta,atol=1e-10)
            assert_allclose(second,np.exp(volatility**2*time)*norm.cdf(sign*(d1+scale)),rtol=1e-10)


def test_pathwise_engine_replays_terminal_draws_and_rng_continuation():
    for kind in ('call','put'):
        rng=np.random.default_rng(412)
        result=pathwise_delta(MODEL,EuropeanOption(100.,kind),1.,4096,rng=rng)
        replay=np.random.default_rng(412)
        terminal=terminal_from_normals(MODEL,1.,replay.standard_normal(4096))
        expected=pathwise_delta_from_terminal_samples(MODEL,EuropeanOption(100.,kind),1.,terminal)
        assert result.estimate==expected.estimate and result.standard_error==expected.standard_error
        assert rng.standard_normal()==replay.standard_normal()


def test_pathwise_delta_mc_agrees_with_analytic_value_and_variance():
    for kind in ('call','put'):
        result=pathwise_delta(MODEL,EuropeanOption(100.,kind),1.,200000,rng=np.random.default_rng(413))
        exact=black_scholes_greeks(MODEL,100.,1.,kind).delta
        sign=1 if kind=='call' else -1
        variance=np.exp(.04)*norm.cdf(sign*(.35+.2))-exact**2
        assert abs(result.estimate-exact)<6*np.sqrt(variance/result.n_paths)
        assert_allclose(result.sample_variance,variance,rtol=.02)


def test_pathwise_put_call_sample_parity_away_from_strike():
    terminal=np.array([50.,95.,100.001,130.,250.])
    call=european_pathwise_delta_samples(EuropeanOption(100.),terminal,spot=100.)
    put=european_pathwise_delta_samples(EuropeanOption(100.,'put'),terminal,spot=100.)
    assert_allclose(call-put,terminal/100.,rtol=0,atol=0)


def test_pathwise_equals_local_fd_for_fixed_paths_without_kink_crossings():
    z=np.array([-2.,-1.,0.,1.,2.])
    terminal=terminal_from_normals(MODEL,1.,z)
    for kind in ('call','put'):
        product=EuropeanOption(100.,kind)
        pw=pathwise_delta_from_terminal_samples(MODEL,product,1.,terminal)
        fd=finite_difference_from_normals(MODEL,product,1.,z,greek='delta',bump=.001)
        assert_allclose([pw.estimate,pw.standard_error],[fd.estimate,fd.standard_error],rtol=2e-10,atol=1e-11)


def test_pathwise_kink_crossing_fd_is_not_identical_at_finite_bump():
    terminal=np.array([99.5,100.5,120.])
    z=(np.log(terminal/100.)-.03)/.2
    pw=pathwise_delta_from_terminal_samples(MODEL,EuropeanOption(100.),1.,terminal)
    fd=finite_difference_from_normals(MODEL,EuropeanOption(100.),1.,z,greek='delta',bump=1.)
    assert abs(pw.estimate-fd.estimate)>1e-5


def test_pathwise_readonly_strided_inputs_and_inactive_overflow_safety():
    terminal=np.arange(80.,160.)[::3]
    before=terminal.copy()
    terminal.flags.writeable=False
    result=european_pathwise_delta_samples(EuropeanOption(100.),terminal,spot=100.)
    assert np.array_equal(terminal,before) and not np.shares_memory(result,terminal)
    assert european_pathwise_delta_samples(EuropeanOption(1.,'put'),[1e300],spot=1e-100)[0]==0.
    with assert_raises(FloatingPointError):
        european_pathwise_delta_samples(EuropeanOption(1.),[1e300],spot=1e-100)


def test_pathwise_invalid_configuration_rejected_before_rng_draws():
    cases=[dict(model=object()),dict(model=replace(MODEL,volatility=0.)),dict(maturity=0.),dict(maturity=-1.),
           dict(n_paths=1),dict(n_paths=True),dict(product=ArithmeticAsianOption(100.)),
           dict(product=UpAndOutCall(100.,120.)),dict(confidence_level=1.)]
    for changes in cases:
        rng=np.random.default_rng(414)
        kwargs=dict(model=MODEL,product=EuropeanOption(100.),maturity=1.,n_paths=8,rng=rng)
        kwargs.update(changes)
        with assert_raises((TypeError,ValueError)):
            pathwise_delta(**kwargs)
        assert rng.standard_normal()==np.random.default_rng(414).standard_normal()
    with assert_raises(TypeError):
        pathwise_delta(MODEL,EuropeanOption(100.),1.,8,rng=1)


def test_pathwise_sample_validation_and_zero_event_uncertainty():
    for terminal in ([],[0.,100.],[-1.,100.],[np.nan,100.],[[100.,110.]]):
        with assert_raises(ValueError):
            european_pathwise_delta_samples(EuropeanOption(100.),terminal,spot=100.)
    for terminal in ([True,False],['100','110'],[1+2j]):
        with assert_raises(TypeError):
            european_pathwise_delta_samples(EuropeanOption(100.),terminal,spot=100.)
    with assert_raises(ValueError):
        pathwise_delta_from_terminal_samples(MODEL,EuropeanOption(100.),1.,[110.])
    with assert_raises(ValueError):
        european_pathwise_delta_samples(EuropeanOption(100.),[110.],spot=0.)
    result=pathwise_delta_from_terminal_samples(MODEL,EuropeanOption(100.),1.,[60.,70.,80.])
    assert result.estimate==0. and result.standard_error==0.
    assert black_scholes_greeks(MODEL,100.,1.).delta>0.  # Missing events are not proof of zero risk.
