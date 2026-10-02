"""Pair alignment, sampling units, RNG budgets and antithetic validation."""

from dataclasses import replace
from math import exp, log, sqrt

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

from mc_pricer import (
    ArithmeticAsianOption, BlackScholesModel, EuropeanOption, UpAndOutCall,
    black_scholes_price, make_rng, pair_means, paths_from_normals,
    price_from_antithetic_samples, price_from_payoff_samples, price_monte_carlo,
    simulate_antithetic_payoffs, terminal_from_normals,
)


def test_antithetic_summary_uses_discounted_pair_units_and_total_budget():
    result = price_from_antithetic_samples([0,2,4,6], [2,4,6,8], rate=log(2), maturity=1)
    assert_allclose([result.price, result.sample_variance, result.standard_error,
                     result.estimator_variance], [2,5/3,sqrt(5/12),5/12])
    assert result.n_paths == 8 and result.n_independent_samples == 4
    assert result.variance_reduction == "antithetic" and result.n_steps is None


def test_antithetic_flattening_gives_wrong_uncertainty():
    first, second = np.array([0.,2,4,6]), np.array([6.,4,2,0])
    paired = price_from_antithetic_samples(first,second,rate=0,maturity=1)
    naive = price_from_payoff_samples(np.concatenate((first,second)),rate=0,maturity=1)
    assert paired.price == naive.price == 3
    assert paired.standard_error == 0 < naive.standard_error
    # Positive dependence can instead increase variance at the same 2M budget.
    positive = price_from_antithetic_samples(first,first,rate=0,maturity=1)
    assert positive.estimator_variance > np.var(first,ddof=1)/8


def test_pair_mean_covariance_identity_and_input_immutability():
    first = np.array([1.,3,8,2,6,10])[::2]
    second = np.array([8.,4,2])
    before = first.copy()
    first.setflags(write=False)
    averages = pair_means(first,second)
    covariance = np.cov(first,second,ddof=1)[0,1]
    assert_allclose(averages.var(ddof=1), (first.var(ddof=1)+second.var(ddof=1)+2*covariance)/4)
    assert_array_equal(first,before)
    assert not np.shares_memory(averages,first)
    assert_allclose(pair_means([1e308,1e308],[1e308,-1e308]),[1e308,0])


def test_pair_validation_rejects_misaligned_or_insufficient_samples():
    for a,b in (([1],[2,3]),([],[]),([[1,2]],[1,2]),([1,np.nan],[1,2])):
        with assert_raises(ValueError):
            pair_means(a,b)
    with assert_raises(TypeError):
        pair_means([True,False],[1,2])
    with assert_raises(ValueError):
        price_from_antithetic_samples([1],[2],rate=0,maturity=1)


def test_antithetic_terminal_shocks_alignment_and_rng_consumption():
    model, product = BlackScholesModel(100,.05,.2), EuropeanOption(100)
    a,b = make_rng(901),make_rng(901)
    normals = a.standard_normal(256)
    positive,negative = simulate_antithetic_payoffs(model,product,1,256,rng=b)
    assert_array_equal(positive,product.payoff(terminal_from_normals(model,1,normals)))
    assert_array_equal(negative,product.payoff(terminal_from_normals(model,1,-normals)))
    assert_array_equal(a.standard_normal(4),b.standard_normal(4))


def test_antithetic_negates_entire_path_shock_vector_for_all_path_products():
    model = BlackScholesModel(100,.05,.2)
    for product in (ArithmeticAsianOption(100),ArithmeticAsianOption(100,"put"),UpAndOutCall(100,120)):
        a,b = make_rng(902),make_rng(902)
        normals = a.standard_normal((128,12))
        positive,negative = simulate_antithetic_payoffs(model,product,1,128,n_steps=12,rng=b)
        assert_array_equal(positive,product.payoff(paths_from_normals(model,1,normals)))
        assert_array_equal(negative,product.payoff(paths_from_normals(model,1,-normals)))
        assert_array_equal(a.standard_normal(4),b.standard_normal(4))


def test_antithetic_engine_equals_supplied_pair_summary_for_all_products():
    model = BlackScholesModel(100,.05,.2)
    for product,steps in ((EuropeanOption(100),None),(ArithmeticAsianOption(100),12),(UpAndOutCall(100,120),24)):
        a,b = make_rng(903),make_rng(903)
        positive,negative = simulate_antithetic_payoffs(model,product,1,512,n_steps=steps,rng=a)
        expected = price_from_antithetic_samples(positive,negative,rate=.05,maturity=1,confidence_level=.99)
        actual = price_monte_carlo(model,product,1,1024,n_steps=steps,rng=b,
                                   variance_reduction="antithetic",confidence_level=.99)
        assert replace(actual,n_steps=None,runtime_seconds=0) == replace(expected,runtime_seconds=0)
        assert actual.n_steps == steps
        assert_array_equal(a.standard_normal(4),b.standard_normal(4))


def test_antithetic_invalid_budget_and_method_fail_before_rng_advances():
    model,product = BlackScholesModel(100,.05,.2),EuropeanOption(100)
    for count,method,error in ((2,"antithetic",ValueError),(5,"antithetic",ValueError),
                              (8,"unknown",ValueError),(8,None,ValueError),(8.,"antithetic",TypeError)):
        a,b = make_rng(904),make_rng(904)
        with assert_raises(error):
            price_monte_carlo(model,product,1,count,rng=a,variance_reduction=method)
        assert_array_equal(a.standard_normal(4),b.standard_normal(4))
    for product,steps in ((EuropeanOption(100),12),(ArithmeticAsianOption(100),None),(UpAndOutCall(100,120),0)):
        a,b = make_rng(904),make_rng(904)
        with assert_raises(ValueError):
            price_monte_carlo(model,product,1,8,rng=a,n_steps=steps,variance_reduction="antithetic")
        assert_array_equal(a.standard_normal(4),b.standard_normal(4))


def test_antithetic_simulator_input_validation():
    model,product = BlackScholesModel(100,.05,.2),EuropeanOption(100)
    for bad_model,bad_product,time,pairs,error in ((None,product,1,4,TypeError),
         (model,None,1,4,TypeError),(model,product,-1,4,ValueError),(model,product,1,0,ValueError),
         (model,product,1,True,TypeError)):
        with assert_raises(error):
            simulate_antithetic_payoffs(bad_model,bad_product,time,pairs,rng=make_rng(1))
    with assert_raises(TypeError):
        simulate_antithetic_payoffs(model,product,1,4,rng=None)


def test_antithetic_deterministic_limits_and_pair_rng_budget():
    for model,time,product,steps in (
        (BlackScholesModel(105,.05,.2),0,EuropeanOption(100),None),
        (BlackScholesModel(105,.05,0),1,ArithmeticAsianOption(100),12),
        (BlackScholesModel(105,.05,.2),1,UpAndOutCall(100,105),12),
    ):
        a,b = make_rng(905),make_rng(905)
        result = price_monte_carlo(model,product,time,20,n_steps=steps,rng=a,variance_reduction="antithetic")
        b.standard_normal(10 if steps is None else (10,steps))
        assert_array_equal(a.standard_normal(4),b.standard_normal(4))
        assert result.standard_error == 0 and result.n_independent_samples == 10


def test_antithetic_european_prices_match_analytical_targets_in_se_units():
    model = BlackScholesModel(100,.05,.2)
    for kind in ("call","put"):
        result = price_monte_carlo(model,EuropeanOption(100,kind),1,100_000,
                                   rng=make_rng(906),variance_reduction="antithetic")
        target = black_scholes_price(model,100,1,kind)
        assert abs(result.price-target) < 6*result.standard_error


def test_antithetic_symmetry_cancels_linear_normal_component_exactly():
    # Analytic toy estimator: f(Z)=3+2Z. Pair means have zero variance;
    # marginal payoffs do not. An independent check of the statistics layer.
    shocks = np.array([-2.,-1.,0.,1.,2.])
    result = price_from_antithetic_samples(3+2*shocks,3-2*shocks,rate=.05,maturity=1)
    assert_allclose(result.price,3*exp(-.05))
    assert result.standard_error == result.sample_variance == 0
