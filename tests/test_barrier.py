"""Discrete barrier identities, endpoint benchmark and engine integration."""

from dataclasses import FrozenInstanceError, replace
from math import exp, log, sqrt

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises
from scipy.integrate import quad
from scipy.stats import norm

from mc_pricer import (BlackScholesModel, EuropeanOption, UpAndOutCall,
                       black_scholes_price, make_rng, price_from_path_samples,
                       price_monte_carlo, simulate_paths,
                       terminal_monitored_up_and_out_price)


def test_barrier_hand_payoffs_include_initial_hit_and_equality():
    paths = [[100,110,115], [100,120,110], [100,121,110], [120,90,110],
             [121,90,110], [100,110,120], [100,110,90], [100,119,119]]
    assert_array_equal(UpAndOutCall(100,120).payoff(paths), [15,0,0,0,0,0,0,19])


def test_barrier_recovery_after_hit_does_not_restore_option():
    product = UpAndOutCall(100,120)
    paths = np.array([[100,125,110], [100,115,110]])
    assert_array_equal(product.payoff(paths), [0,10])
    # Removing the intermediate observation changes the monitored contract.
    assert_array_equal(product.payoff(paths[:,::2]), [10,10])


def test_barrier_product_validation_and_frozen_fields():
    for name in ('strike','barrier'):
        for value in (0,-1,np.inf,np.nan):
            with assert_raises(ValueError):
                UpAndOutCall(**{'strike':100,'barrier':120,name:value})
        for value in (True,'120',[120]):
            with assert_raises(TypeError):
                UpAndOutCall(**{'strike':100,'barrier':120,name:value})
    with assert_raises(FrozenInstanceError):
        UpAndOutCall(100,120).barrier = 130


def test_barrier_invalid_paths_and_extrema_are_rejected():
    product = UpAndOutCall(100,120)
    for paths in ([], [100,110], [[100]], [[100,-1]], [[100,np.nan]], np.empty((0,2))):
        with assert_raises(ValueError):
            product.payoff(paths)
    for paths in ([[True,False]], [['100','110']], [[100+1j,110]]):
        with assert_raises(TypeError):
            product.payoff(paths)
    for terminal, maxima in (([110],[109]), ([-1],[100]), ([100,110],[120]), ([100],[np.inf])):
        with assert_raises(ValueError):
            product.payoff_from_extrema(terminal,maxima)


def test_barrier_supports_readonly_strided_paths_without_mutation():
    source = np.arange(90.,150.).reshape(5,12)
    before = source.copy()
    view = source[::2,::2]
    view.setflags(write=False)
    product = UpAndOutCall(100,125)
    output = product.payoff(view)
    assert output.shape == (3,) and output.dtype == np.float64
    assert not np.shares_memory(output,source)
    assert_array_equal(output,product.payoff_from_extrema(view[:,-1],view.max(axis=1)))
    assert_array_equal(source,before)


def test_barrier_payoff_bounds_in_out_identity_and_barrier_ordering():
    paths=simulate_paths(BlackScholesModel(100,.05,.2),1,2048,24,rng=make_rng(812))
    vanilla=EuropeanOption(100).payoff(paths[:,-1])
    low=UpAndOutCall(100,115).payoff(paths)
    high=UpAndOutCall(100,130).payoff(paths)
    assert np.all((0<=low)&(low<=high)&(high<=vanilla))
    knocked_in=vanilla*(paths.max(axis=1)>=115)
    assert_array_equal(low+knocked_in,vanilla)
    assert_array_equal(UpAndOutCall(100,1e6).payoff(paths),vanilla)
    # Monitoring includes maturity: positive payoff implies ST>K>=B, a hit.
    assert_array_equal(UpAndOutCall(120,115).payoff(paths),np.zeros(2048))


def test_barrier_nested_grid_payoffs_are_pathwise_nonincreasing():
    paths=simulate_paths(BlackScholesModel(100,.05,.2),1,4096,48,rng=make_rng(814))
    product=UpAndOutCall(100,120)
    previous=product.payoff(paths[:,::48])
    for steps in (2,4,12,24,48):
        current=product.payoff(paths[:,::48//steps])
        assert np.all(current<=previous)
        previous=current


def test_barrier_discounted_statistics_from_hand_payoffs():
    paths=[[100,105,110],[100,110,115],[100,120,110],[100,95,90]]
    result=price_from_path_samples(UpAndOutCall(100,120),paths,rate=log(2),maturity=1)
    # Discounted samples are [5,7.5,0,0], N=4 rather than 8 monitoring values.
    assert_allclose([result.price,result.sample_variance,result.standard_error],
                    [3.125,14.0625,sqrt(14.0625/4)])
    assert result.n_paths==result.n_independent_samples==4 and result.n_steps==2


def test_barrier_engine_replays_paths_and_consumes_draws_even_if_initially_hit():
    model=BlackScholesModel(100,.05,.2)
    for barrier in (100,120):
        product=UpAndOutCall(90,barrier)
        a,b=make_rng(815),make_rng(815)
        paths=simulate_paths(model,1,1000,12,rng=a)
        expected=price_from_path_samples(product,paths,rate=.05,maturity=1,confidence_level=.99)
        actual=price_monte_carlo(model,product,1,1000,n_steps=12,rng=b,confidence_level=.99)
        assert replace(expected,runtime_seconds=0)==replace(actual,runtime_seconds=0)
        assert_array_equal(a.standard_normal(4),b.standard_normal(4))
        if barrier==100:
            assert actual.price==actual.standard_error==0


def test_barrier_expiry_and_deterministic_limits():
    for rate,vol,time,barrier in ((.05,.2,0,120),(.05,.2,0,100),
                                 (.05,0,1,120),(.2,0,1,120),(-.1,0,1,99),(-.1,0,1,120)):
        model=BlackScholesModel(100,rate,vol)
        result=price_monte_carlo(model,UpAndOutCall(80,barrier),time,32,n_steps=12,rng=make_rng(1))
        terminal=100*exp(rate*time)
        expected=exp(-rate*time)*max(terminal-80,0)*(max(100,terminal)<barrier)
        assert_allclose(result.price,expected,atol=2e-13)
        assert result.standard_error==0


def test_barrier_grid_validation_before_rng_consumption():
    model,product=BlackScholesModel(100,.05,.2),UpAndOutCall(100,120)
    for steps,error in ((None,ValueError),(0,ValueError),(True,TypeError),(12.,TypeError)):
        a,b=make_rng(12),make_rng(12)
        with assert_raises(error):
            price_monte_carlo(model,product,1,10,n_steps=steps,rng=a)
        assert_array_equal(a.standard_normal(4),b.standard_normal(4))


def test_terminal_barrier_formula_against_lognormal_quadrature():
    for spot,strike,barrier,rate,vol,time in ((100,100,120,.05,.2,1),
                                            (90,80,105,-.03,.35,2), (100,105,125,0,.15,.5)):
        model=BlackScholesModel(spot,rate,vol)
        mu=log(spot)+(rate-.5*vol**2)*time
        scale=vol*sqrt(time)
        integrand=lambda s: (s-strike)*norm.pdf((log(s)-mu)/scale)/(s*scale)
        expected=exp(-rate*time)*quad(integrand,strike,barrier,epsabs=1e-11)[0]
        actual=terminal_monitored_up_and_out_price(model,strike,barrier,time)
        assert_allclose(actual,expected,atol=3e-13,rtol=3e-13)
        assert 0<=actual<=black_scholes_price(model,strike,time)


def test_terminal_barrier_formula_limits_and_validation():
    model=BlackScholesModel(100,.05,.2)
    for strike,barrier,time,expected in ((90,100,1,0),(120,110,1,0),(90,120,0,10)):
        assert terminal_monitored_up_and_out_price(model,strike,barrier,time)==expected
    deterministic=replace(model,volatility=0)
    assert terminal_monitored_up_and_out_price(deterministic,90,104,1)==0
    assert_allclose(terminal_monitored_up_and_out_price(deterministic,90,120,1),100-90*exp(-.05))
    for strike,barrier,time in ((0,120,1),(100,0,1),(100,np.inf,1),(100,120,-1)):
        with assert_raises(ValueError):
            terminal_monitored_up_and_out_price(model,strike,barrier,time)
    with assert_raises(TypeError):
        terminal_monitored_up_and_out_price(None,100,120,1)


def test_one_step_barrier_monte_carlo_matches_exact_target_in_se_units():
    model=BlackScholesModel(100,.05,.2)
    result=price_monte_carlo(model,UpAndOutCall(100,120),1,150_000,n_steps=1,rng=make_rng(20260928))
    target=terminal_monitored_up_and_out_price(model,100,120,1)
    assert abs(result.price-target)<6*result.standard_error
