"""Geometric control formula, independent-pilot correction and cost accounting."""

from dataclasses import replace
from math import exp, log, sqrt

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises
from scipy.integrate import quad
from scipy.stats import norm

from mc_pricer import (
    ArithmeticAsianOption, GeometricAsianOption, EuropeanOption, BlackScholesModel,
    UpAndOutCall, apply_control_variate, fit_control_variate, geometric_asian_price,
    geometric_average_log_moments, black_scholes_price, make_rng, simulate_paths,
    price_monte_carlo, price_from_payoff_samples, price_from_control_samples,
)


def test_geometric_payoffs_exclude_spot_and_match_hand_values():
    paths = [[999,25,100],[1,100,400],[7,16,100]]
    call, put = GeometricAsianOption(50), GeometricAsianOption(50,"put")
    assert_allclose(call.average(paths),[50,200,40],atol=1e-12)
    assert_allclose(call.payoff(paths),[0,150,0],atol=1e-12)
    assert_allclose(put.payoff(paths),[0,0,10],atol=1e-12)
    assert_array_equal(call.average([[100,121],[200,81]]),[121,81])


def test_geometric_average_am_gm_ordering_views_and_log_stability():
    source = np.arange(80.,140.).reshape(5,12)
    before = source.copy()
    view = source[:,::2]
    view.setflags(write=False)
    geo, arith = GeometricAsianOption(100), ArithmeticAsianOption(100)
    assert np.all(geo.average(view) <= arith.average(view)+1e-12)
    assert np.all(geo.payoff(view) <= arith.payoff(view)+1e-12)
    assert np.all(GeometricAsianOption(100,"put").payoff(view) >= ArithmeticAsianOption(100,"put").payoff(view)-1e-12)
    assert_array_equal(source,before)
    assert not np.shares_memory(geo.average(view),source)
    assert_allclose(geo.average([[1,1e200,1e200]]),[1e200],rtol=1e-13)


def test_geometric_product_and_input_validation():
    for strike in (0,-1,np.inf,np.nan):
        with assert_raises(ValueError):
            GeometricAsianOption(strike)
    with assert_raises(ValueError):
        GeometricAsianOption(100,"CALL")
    with assert_raises(TypeError):
        GeometricAsianOption(True)
    product = GeometricAsianOption(100)
    for paths in ([[100]], [[100,0]], [[0,100]], [[100,-1]], [[100,np.nan]], [100,120]):
        with assert_raises(ValueError):
            product.payoff(paths)
    with assert_raises(ValueError):
        product.payoff_from_average([0,100])


def test_geometric_log_moments_match_explicit_brownian_covariance_sum():
    model = BlackScholesModel(95,-.03,.35)
    for m in (1,4,12,48):
        times = np.arange(1,m+1)*2/m
        expected_mean = log(95)+(-.03-.35**2/2)*times.mean()
        expected_var = .35**2*sum(min(s,t) for s in times for t in times)/m**2
        moments = geometric_average_log_moments(model,2,m)
        assert_allclose([moments.log_mean,moments.log_variance],[expected_mean,expected_var],rtol=3e-15)


def test_geometric_price_against_independent_normal_quadrature_and_parity():
    for spot,rate,vol,time,m in ((100,.05,.2,1,12),(90,-.03,.35,2,4)):
        model = BlackScholesModel(spot,rate,vol)
        moments = geometric_average_log_moments(model,time,m)
        a,b = moments.log_mean,sqrt(moments.log_variance)
        strike = 100
        cutoff = (log(strike)-a)/b
        call_integral = exp(-rate*time)*quad(lambda z:(exp(a+b*z)-strike)*norm.pdf(z),cutoff,12,epsabs=1e-11)[0]
        put_integral = exp(-rate*time)*quad(lambda z:(strike-exp(a+b*z))*norm.pdf(z),-12,cutoff,epsabs=1e-11)[0]
        call = geometric_asian_price(model,strike,time,m)
        put = geometric_asian_price(model,strike,time,m,"put")
        assert_allclose([call,put],[call_integral,put_integral],atol=3e-12,rtol=3e-12)
        assert_allclose(call-put,exp(-rate*time)*(exp(a+b*b/2)-strike),atol=1e-12)


def test_geometric_price_single_date_expiry_deterministic_limits_and_validation():
    model = BlackScholesModel(105,.05,.2)
    for kind in ("call","put"):
        assert geometric_asian_price(model,100,1,1,kind)==black_scholes_price(model,100,1,kind)
        assert geometric_asian_price(model,100,0,12,kind)==black_scholes_price(model,100,0,kind)
        expected=exp(-.05)*max((1 if kind=='call' else -1)*(105*exp(.05*13/24)-100),0)
        assert_allclose(geometric_asian_price(replace(model,volatility=0),100,1,12,kind),expected,atol=1e-12)
    for time,m in ((-1,12),(1,0),(np.nan,12)):
        with assert_raises(ValueError):
            geometric_average_log_moments(model,time,m)
    with assert_raises(TypeError):
        geometric_average_log_moments(model,1,12.)


def test_geometric_simulation_matches_log_moments_and_exact_option_prices():
    model = BlackScholesModel(100,.05,.2)
    paths = simulate_paths(model,1,50000,12,rng=make_rng(1001))
    logs = np.log(GeometricAsianOption(100).average(paths))
    moments = geometric_average_log_moments(model,1,12)
    assert abs(logs.mean()-moments.log_mean)<6*sqrt(moments.log_variance/logs.size)
    for kind in ("call","put"):
        result = price_monte_carlo(model,GeometricAsianOption(100,kind),1,50000,n_steps=12,rng=make_rng(1001))
        target = geometric_asian_price(model,100,1,12,kind)
        assert abs(result.price-target)<6*result.standard_error
        anti = price_monte_carlo(model,GeometricAsianOption(100,kind),1,20000,n_steps=12,
                                 rng=make_rng(1002),variance_reduction="antithetic")
        assert abs(anti.price-target)<6*anti.standard_error


def test_control_fit_recovers_linear_relation_and_discount_invariance():
    y = np.array([0.,1,2,3,4])
    x = 2*y+1
    fit = fit_control_variate(x,y)
    assert fit.beta==2 and fit.n_pilot_samples==5 and fit.status=='fitted'
    assert_allclose([fit.correlation,fit.control_sample_variance],[1,2.5])
    discounted = fit_control_variate(.95*x,.95*y)
    assert_allclose([discounted.beta,discounted.correlation],[2,1],rtol=1e-15)
    assert_allclose(apply_control_variate([1,5,9],[0,2,4],control_expectation=2,beta=fit.beta),[5,5,5])


def test_constant_control_pilot_falls_back_without_false_certainty():
    fit = fit_control_variate([1,3,5],[0,0,0])
    assert fit.beta==0 and fit.status=='zero_control_variance' and fit.correlation is None
    result = price_from_control_samples([1,3,5],[0,1,2],beta=fit.beta,control_expectation=1,rate=0,maturity=1)
    assert result.price==3 and result.standard_error>0
    assert fit_control_variate([1,1,1],[1,2,3]).beta==0


def test_control_samples_discounted_variance_and_expectation_shift():
    result = price_from_control_samples([1,3,5,7],[0,1,2,3],control_expectation=1.5,beta=1,rate=log(2),maturity=1)
    assert_allclose([result.price,result.sample_variance,result.standard_error],[2,5/12,sqrt(5/48)])
    assert result.n_paths==result.n_independent_samples==4 and result.n_pilot_paths==0
    assert result.control_expectation==1.5 and result.control_beta==1
    wrong = price_from_control_samples([1,3,5,7],[0,1,2,3],control_expectation=2.5,beta=1,rate=log(2),maturity=1)
    assert_allclose(wrong.price-result.price,.5)


def test_control_input_validation_and_nonmutation():
    x,y=np.array([1.,3,7]),np.array([1.,2,3])
    x.setflags(write=False)
    saved=y.copy()
    fit=fit_control_variate(x,y)
    adjusted=apply_control_variate(x,y,control_expectation=2,beta=fit.beta)
    assert_array_equal(y,saved)
    assert not np.shares_memory(adjusted,x)
    for a,b in (([1],[1]),([1,2],[1,2,3]),([1,np.nan],[1,2])):
        with assert_raises(ValueError):
            fit_control_variate(a,b)
    with assert_raises(ValueError):
        apply_control_variate([1,2],[1],control_expectation=0,beta=1)
    with assert_raises(ValueError):
        apply_control_variate([1,2],[1,2],control_expectation=np.nan,beta=1)


def test_control_engine_replays_disjoint_pilot_production_and_full_rng_budget():
    model,product=BlackScholesModel(100,.05,.2),ArithmeticAsianOption(100)
    a,b=make_rng(1003),make_rng(1003)
    pilot=simulate_paths(model,1,128,12,rng=a)
    control=GeometricAsianOption(100)
    fit=fit_control_variate(product.payoff(pilot),control.payoff(pilot))
    production=simulate_paths(model,1,896,12,rng=a)
    expected=geometric_asian_price(model,100,1,12)/exp(-.05)
    reference=price_from_control_samples(product.payoff(production),control.payoff(production),
                                         beta=fit.beta,control_expectation=expected,rate=.05,maturity=1)
    actual=price_monte_carlo(model,product,1,1024,n_steps=12,n_pilot_paths=128,
                             rng=b,variance_reduction="control_variate")
    assert replace(actual,n_paths=896,n_steps=None,n_pilot_paths=0,pilot_correlation=None,
                   control_fit_status=None,runtime_seconds=0)==replace(reference,runtime_seconds=0)
    assert actual.n_independent_samples==896 and actual.n_paths==1024
    assert_array_equal(a.standard_normal(5),b.standard_normal(5))


def test_one_date_control_reduces_to_analytical_european_with_unit_beta():
    model=BlackScholesModel(100,.05,.2)
    for kind in ('call','put'):
        result=price_monte_carlo(model,ArithmeticAsianOption(100,kind),1,2000,n_steps=1,
                                 n_pilot_paths=200,rng=make_rng(1004),variance_reduction="control_variate")
        assert result.control_beta==1
        assert_allclose(result.price,black_scholes_price(model,100,1,kind),atol=1e-13)
        assert result.standard_error<1e-14


def test_control_deterministic_limits_and_zero_pilot_variance():
    for time,vol in ((0,.2),(1,0)):
        model=BlackScholesModel(105,.05,vol)
        result=price_monte_carlo(model,ArithmeticAsianOption(100),time,100,n_steps=12,
                                 n_pilot_paths=20,rng=make_rng(1005),variance_reduction="control_variate")
        assert result.control_beta==0 and result.control_fit_status=='zero_control_variance'
        assert result.standard_error==0
        expected=exp(-.05*time)*max(np.mean(105*np.exp(.05*time*np.arange(1,13)/12))-100,0)
        assert_allclose(result.price,expected,atol=1e-12)


def test_invalid_control_budget_product_and_grid_fail_before_rng():
    model=BlackScholesModel(100,.05,.2)
    asian=ArithmeticAsianOption(100)
    cases=[(asian,100,12,None,ValueError),(asian,100,12,1,ValueError),
           (asian,100,12,99,ValueError),(asian,100,None,20,ValueError),
           (asian,100,12,20.,TypeError),(GeometricAsianOption(100),100,12,20,TypeError),
           (EuropeanOption(100),100,None,20,TypeError),(UpAndOutCall(100,120),100,12,20,TypeError)]
    for product,budget,steps,pilot,error in cases:
        a,b=make_rng(1006),make_rng(1006)
        with assert_raises(error):
            price_monte_carlo(model,product,1,budget,n_steps=steps,n_pilot_paths=pilot,rng=a,variance_reduction='control_variate')
        assert_array_equal(a.standard_normal(3),b.standard_normal(3))
    with assert_raises(ValueError):
        price_monte_carlo(model,asian,1,100,n_steps=12,n_pilot_paths=20,rng=make_rng(1))


def test_pilot_conditioned_linear_control_preserves_known_target():
    # Exactly X=4+2Y: fitting only the independent pilot must yield beta=2.
    # Production has a deliberately different sample mean; same-sample centering
    # of Y would leave a detectable error instead of returning the known target.
    pilot_y=np.array([-2.,0,2,4])
    production_y=np.array([5.,6,7,8])
    fit=fit_control_variate(4+2*pilot_y,pilot_y)
    result=price_from_control_samples(4+2*production_y,production_y,control_expectation=0,
                                      beta=fit.beta,rate=0,maturity=1)
    assert result.price==4 and result.standard_error==0
