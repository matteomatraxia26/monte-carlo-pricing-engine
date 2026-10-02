"""Independent Riccati-ODE, deterministic limits and Fourier-cutoff checks."""
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose, assert_raises
from scipy.integrate import solve_ivp
from mc_pricer import (HestonModel,BlackScholesModel,black_scholes_price,
    deterministic_integrated_variance,heston_centered_characteristic,heston_european_reference)
MODEL=HestonModel(100.,.05,.04,2.,.04,.3,-.7)


def test_heston_characteristic_against_independent_riccati_ode():
    for model in (MODEL,replace(MODEL,xi=.9),replace(MODEL,kappa=.5,xi=1.,rho=.8)):
        for u in (0.,1.,8.,1-.5j,12-.5j):
            def rhs(t,y):
                c,d=y
                return [model.kappa*model.theta*d,.5*model.xi**2*d*d+(model.rho*model.xi*1j*u-model.kappa)*d-.5*(u*u+1j*u)]
            solution=solve_ivp(rhs,[0,1],[0j,0j],rtol=1e-10,atol=1e-12)
            assert solution.success
            expected=np.exp(solution.y[0,-1]+solution.y[1,-1]*model.v0)
            assert_allclose(heston_centered_characteristic(model,1,u),expected,rtol=3e-9,atol=1e-11)
    for u in (1.,4.,10.):
        assert_allclose(heston_centered_characteristic(MODEL,1,-u),np.conj(heston_centered_characteristic(MODEL,1,u)))
    assert heston_centered_characteristic(MODEL,1,0)==1


def test_heston_reference_xi_zero_and_small_xi_match_black_scholes():
    for v0 in (.04,.09):
        model=replace(MODEL,xi=0.,v0=v0)
        total=deterministic_integrated_variance(model,1.)
        for kind in ('call','put'):
            expected=black_scholes_price(BlackScholesModel(100,.05,np.sqrt(total)),100,1,kind)
            assert heston_european_reference(model,100,1,kind).price==expected
            close=heston_european_reference(replace(model,xi=1e-5),100,1,kind).price
            assert_allclose(close,expected,atol=2e-5)
    assert heston_european_reference(MODEL,110,0,'put').price==10


def test_heston_fourier_cutoff_stability_parity_and_bounds():
    for model in (MODEL,replace(MODEL,xi=.9),replace(MODEL,rho=.3)):
        for strike in (80.,100.,120.):
            a=heston_european_reference(model,strike,1,cutoff=200)
            b=heston_european_reference(model,strike,1,cutoff=400,tolerance=1e-11)
            assert abs(a.price-b.price)<2e-6
            put=heston_european_reference(model,strike,1,'put',cutoff=400)
            assert_allclose(b.price-put.price,100-strike*np.exp(-.05),atol=1e-8)
            assert max(100-strike*np.exp(-.05),0)<=b.price<=100


def test_heston_reference_input_and_domain_validation():
    for kwargs in (dict(strike=0),dict(maturity=-1),dict(cutoff=0),dict(tolerance=0),dict(option_type='x'),dict(model=replace(MODEL,rho=1.))):
        args=dict(model=MODEL,strike=100,maturity=1);args.update(kwargs)
        with assert_raises((TypeError,ValueError)):
            heston_european_reference(**args)
    for u in (1j,-1j,np.nan,[1],True):
        with assert_raises((TypeError,ValueError)):
            heston_centered_characteristic(MODEL,1,u)
    with assert_raises(ValueError):deterministic_integrated_variance(MODEL,1)
