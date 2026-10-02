"""M13: predeclared multi-scenario estimator study and rare-event diagnostics.

No new pricing estimators: reuse M1-M12 APIs. Arithmetic Asian prices have no
exact target here; report variance/cost and geometric-lower-bound violations,
never assign their unknown bias, RMSE or exact coverage.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm

from mc_pricer import (ArithmeticAsianOption, BlackScholesModel, EuropeanOption,
                       black_scholes_greeks, black_scholes_price, finite_difference_greek,
                       geometric_asian_price, pathwise_delta, price_monte_carlo)
from mc_pricer.statistics.confidence_intervals import binomial_wilson_interval
from mc_pricer.statistics.diagnostics import summarize_replications


@dataclass(frozen=True)
class Scenario:
    name: str
    spot: float
    volatility: float
    maturity: float
    rate: float = .05
    strike: float = 100.

    @property
    def model(self) -> BlackScholesModel:
        return BlackScholesModel(self.spot,self.rate,self.volatility)


GREEK_CASES=(Scenario('atm',100.,.2,1.),Scenario('itm',120.,.2,1.),
             Scenario('otm',80.,.2,1.),Scenario('short',100.,.2,.04),
             Scenario('low_vol',100.,.08,1.),Scenario('high_vol',100.,.5,1.),
             Scenario('long',100.,.2,3.),Scenario('rare',80.,.1,.25))
ASIAN_CASES=(Scenario('atm',100.,.2,1.),Scenario('itm',120.,.2,1.),
             Scenario('high_vol',100.,.5,1.),Scenario('rare',85.,.15,.25))
# name, Greek, bump specification, common random numbers
METHODS=(('pw_delta','delta',None,True),('fd_delta_1pct','delta',.01,True),
         ('fd_delta_0.1pct','delta',.001,True),('ind_delta_1pct','delta',.01,False),
         ('fd_gamma_1pct','gamma',.01,True),('fd_gamma_0.1pct','gamma',.001,True),
         ('fd_vega_0.01','vega',.01,True),('fd_vega_0.001','vega',.001,True))
PRICE_METHODS=('none','antithetic','control_variate')
REFERENCES={'delta':'fd_delta_1pct','gamma':'fd_gamma_1pct','vega':'fd_vega_0.01'}


def stream(seed: int, key: tuple[int,...]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed,spawn_key=key))


def validate_design(n_paths: int, replications: int, seed: int, n_pilot: int) -> None:
    if any(isinstance(x,bool) or not isinstance(x,Integral) for x in (n_paths,replications,seed,n_pilot)):
        raise TypeError('counts and seed must be integers')
    if n_paths<4 or n_paths%2 or replications<2 or seed<0 or not 2<=n_pilot<=n_paths-2:
        raise ValueError('require even paths>=4, replications>=2, seed>=0 and 2<=pilot<=paths-2')


def greek_targets(case: Scenario, method_index: int) -> tuple[float,float,float | None]:
    _,greek,fraction,_=METHODS[method_index]
    exact=getattr(black_scholes_greeks(case.model,case.strike,case.maturity),greek)
    if fraction is None:
        return exact,exact,None
    h=fraction if greek=='vega' else fraction*case.spot
    field='volatility' if greek=='vega' else 'spot'
    center=getattr(case.model,field)
    plus=black_scholes_price(replace(case.model,**{field:center+h}),case.strike,case.maturity)
    minus=black_scholes_price(replace(case.model,**{field:center-h}),case.strike,case.maturity)
    if greek=='gamma':
        base=black_scholes_price(case.model,case.strike,case.maturity)
        target=((plus-base)+(minus-base))/h**2
    else:
        target=(plus-minus)/(2*h)
    return exact,target,h


def run_greek(case: Scenario, method_index: int, n_paths: int, rng: np.random.Generator):
    name,greek,fraction,crn=METHODS[method_index]
    product=EuropeanOption(case.strike)
    if fraction is None:
        return pathwise_delta(case.model,product,case.maturity,2*n_paths,rng=rng)
    h=fraction if greek=='vega' else fraction*case.spot
    return finite_difference_greek(case.model,product,case.maturity,n_paths,greek=greek,bump=h,
                                  common_random_numbers=crn,rng=rng)


def collect_greeks(n_paths: int, replications: int, seed: int) -> pd.DataFrame:
    validate_design(n_paths,replications,seed,2)
    rows=[]
    for ci,case in enumerate(GREEK_CASES):
        targets=[greek_targets(case,mi) for mi in range(len(METHODS))]
        for mi in range(len(METHODS)):
            run_greek(case,mi,128,stream(seed,(99,0,ci,mi)))
        for replication in range(replications):
            for index in np.roll(np.arange(len(METHODS)),replication%len(METHODS)):
                mi=int(index)
                method,greek,fraction,crn=METHODS[mi]
                key=(0,ci,replication,mi)
                result=run_greek(case,mi,n_paths,stream(seed,key))
                exact,target,h=targets[mi]
                rows.append(dict(scenario=case.name,case_index=ci,method=method,method_index=mi,greek=greek,
                                 replication=replication,spawn_key=json.dumps(key),n_paths=result.n_paths,
                                 terminal_evaluations=(3 if greek=='gamma' else 2)*n_paths,
                                 bump=h,estimate=result.estimate,standard_error=result.standard_error,
                                 estimator_variance=result.estimator_variance,ci_lower=result.confidence_interval[0],
                                 ci_upper=result.confidence_interval[1],runtime_seconds=result.runtime_seconds,
                                 analytical_value=exact,estimator_target=target,deterministic_bias=target-exact))
    return pd.DataFrame(rows)


def collect_asian(n_paths: int, n_pilot: int, replications: int, seed: int) -> pd.DataFrame:
    validate_design(n_paths,replications,seed,n_pilot)
    rows=[]
    for ci,case in enumerate(ASIAN_CASES):
        product=ArithmeticAsianOption(case.strike)
        lower_bound=geometric_asian_price(case.model,case.strike,case.maturity,12)
        for mi,method in enumerate(PRICE_METHODS):
            price_monte_carlo(case.model,product,case.maturity,128,n_steps=12,rng=stream(seed,(99,1,ci,mi)),
                              variance_reduction=method,n_pilot_paths=16 if method=='control_variate' else None)
        for replication in range(replications):
            for index in np.roll(np.arange(3),replication%3):
                mi=int(index)
                method=PRICE_METHODS[mi]
                key=(1,ci,replication,mi)
                result=price_monte_carlo(case.model,product,case.maturity,n_paths,n_steps=12,
                                        variance_reduction=method,n_pilot_paths=n_pilot if mi==2 else None,
                                        rng=stream(seed,key))
                rows.append(dict(scenario=case.name,case_index=ci,method=method,method_index=mi,
                                 replication=replication,spawn_key=json.dumps(key),n_paths=n_paths,
                                 n_pilot_paths=result.n_pilot_paths,n_independent_samples=result.n_independent_samples,
                                 payoff_evaluations=n_paths*(2 if mi==2 else 1),estimate=result.price,
                                 standard_error=result.standard_error,estimator_variance=result.estimator_variance,
                                 ci_lower=result.confidence_interval[0],ci_upper=result.confidence_interval[1],
                                 runtime_seconds=result.runtime_seconds,geometric_lower_bound=lower_bound,
                                 beta=result.control_beta,pilot_correlation=result.pilot_correlation,
                                 fit_status=result.control_fit_status))
    return pd.DataFrame(rows)


def _common_summary(group: pd.DataFrame) -> dict:
    if len(group)<2 or group.n_paths.nunique()!=1:
        raise ValueError('need independent replications with identical method budgets')
    variance=group.estimate.var(ddof=1)
    zeros=int((group.standard_error==0).sum())
    lo,hi=binomial_wilson_interval(zeros,len(group))
    return dict(n_replications=len(group),n_paths=int(group.n_paths.iloc[0]),
                mean_estimate=group.estimate.mean(),empirical_variance=variance,
                empirical_standard_error_of_mean=np.sqrt(variance/len(group)),
                rms_reported_standard_error=np.sqrt(group.estimator_variance.mean()),
                mean_reported_estimator_variance=group.estimator_variance.mean(),
                mean_runtime_seconds=group.runtime_seconds.mean(),zero_se_runs=zeros,
                zero_se_rate=zeros/len(group),zero_se_ci_lower=lo,zero_se_ci_upper=hi,
                min_estimate=group.estimate.min(),max_estimate=group.estimate.max())


def summarize_greeks(raw: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (scenario,greek,method),group in raw.groupby(['scenario','greek','method'],sort=False):
        target=group.analytical_value.iloc[0]
        expectation=group.estimator_target.iloc[0]
        if group.analytical_value.nunique()!=1 or group.estimator_target.nunique()!=1:
            raise ValueError('inconsistent targets within a replication cell')
        stats=summarize_replications(group.estimate.to_numpy(),target)
        row=dict(scenario=scenario,greek=greek,method=method,**_common_summary(group),
                 analytical_value=target,estimator_target=expectation,
                 deterministic_bias=expectation-target,empirical_bias=stats.empirical_bias,
                 empirical_bias_ci_lower=stats.bias_confidence_interval[0],empirical_bias_ci_upper=stats.bias_confidence_interval[1],
                 rmse=stats.rmse,relative_rmse=stats.rmse/abs(target) if target else np.nan,
                 terminal_evaluations=int(group.terminal_evaluations.iloc[0]),
                 bump=group.bump.iloc[0])
        for label,value in (('greek',target),('estimator',expectation)):
            hits=int(((group.ci_lower<=value)&(value<=group.ci_upper)).sum())
            lo,hi=binomial_wilson_interval(hits,len(group))
            row.update({f'{label}_coverage':hits/len(group),f'{label}_coverage_ci_lower':lo,f'{label}_coverage_ci_upper':hi})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_asian(raw: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (scenario,method),group in raw.groupby(['scenario','method'],sort=False):
        violations=int((group.ci_upper<group.geometric_lower_bound).sum())
        lo,hi=binomial_wilson_interval(violations,len(group))
        rows.append(dict(scenario=scenario,method=method,**_common_summary(group),
                         geometric_lower_bound=group.geometric_lower_bound.iloc[0],
                         upper_ci_below_lower_bound_rate=violations/len(group),
                         violation_ci_lower=lo,violation_ci_upper=hi,
                         fallback_rate=float((group.fit_status=='zero_control_variance').mean()) if method=='control_variate' else np.nan,
                         median_beta=group.beta.median() if group.beta.notna().any() else np.nan,
                         max_abs_beta=group.beta.abs().max(),
                         negative_estimate_runs=int((group.estimate<0).sum()),
                         n_pilot_paths=int(group.n_pilot_paths.iloc[0]),
                         payoff_evaluations=int(group.payoff_evaluations.iloc[0])))
    return pd.DataFrame(rows)


def bootstrap_variance_ratio(reference: np.ndarray, candidate: np.ndarray, *,
                             rng: np.random.Generator, resamples: int=2000) -> dict:
    """Independent run-level percentile bootstrap, not a remedy for unseen tails.

    Undefined ratios stay NaN. Zero-variance resamples are reported and omitted
    from quantiles; valid fraction then matters and interval is conditional on
    validity. No distribution-free population-variance guarantee is implied.
    """
    x=np.asarray(reference,dtype=float)
    y=np.asarray(candidate,dtype=float)
    if x.ndim!=1 or y.ndim!=1 or min(x.size,y.size)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('need finite one-dimensional repeated estimates')
    if isinstance(resamples,bool) or not isinstance(resamples,Integral) or resamples<2:
        raise ValueError('resamples must be an integer >=2')
    vx,vy=x.var(ddof=1),y.var(ddof=1)
    if min(vx,vy)<=0:
        return dict(empirical_vrf=np.nan,bootstrap_lower=np.nan,bootstrap_upper=np.nan,
                    bootstrap_valid_fraction=0.,status='zero_empirical_variance')
    xb=x[rng.integers(0,x.size,size=(resamples,x.size))].var(axis=1,ddof=1)
    yb=y[rng.integers(0,y.size,size=(resamples,y.size))].var(axis=1,ddof=1)
    valid=(xb>0)&(yb>0)
    low,high=np.quantile(xb[valid]/yb[valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
    return dict(empirical_vrf=vx/vy,bootstrap_lower=low,bootstrap_upper=high,
                bootstrap_valid_fraction=valid.mean(),status='ok' if valid.all() else 'some_resamples_undefined')


def comparisons(greek_raw: pd.DataFrame, asian_raw: pd.DataFrame, seed: int,
                resamples: int=2000) -> pd.DataFrame:
    rows=[]
    groups=[]
    for (scenario,greek),group in greek_raw.groupby(['scenario','greek'],sort=False):
        groups.append(('greeks',scenario,greek,group,REFERENCES[greek]))
    for scenario,group in asian_raw.groupby('scenario',sort=False):
        groups.append(('asian',scenario,'price',group,'none'))
    for gi,(study,scenario,quantity,group,ref_name) in enumerate(groups):
        reference=group.loc[group.method==ref_name]
        if reference.empty:
            raise ValueError('missing reference method')
        budget='terminal_evaluations' if study=='greeks' else 'n_paths'
        if group[budget].nunique()!=1:
            raise ValueError('unequal comparison budgets')
        for mi,(method,candidate) in enumerate(group.loc[group.method!=ref_name].groupby('method',sort=True)):
            key=(2,gi,mi)
            ratio=bootstrap_variance_ratio(reference.estimate.to_numpy(),candidate.estimate.to_numpy(),
                                           rng=stream(seed,key),resamples=resamples)
            row=dict(study=study,scenario=scenario,quantity=quantity,reference=ref_name,method=method,
                     bootstrap_spawn_key=json.dumps(key),**ratio,
                     time_adjusted_efficiency=ratio['empirical_vrf']*reference.runtime_seconds.mean()/candidate.runtime_seconds.mean())
            if study=='greeks':
                exact=reference.analytical_value.iloc[0]
                ref_mse=np.mean((reference.estimate-exact)**2)
                cand_mse=np.mean((candidate.estimate-exact)**2)
                row['empirical_mse_ratio']=ref_mse/cand_mse if cand_mse>0 else np.nan
            else:
                row['empirical_mse_ratio']=np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def rare_event_diagnostic(n_paths: int, raw: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for case in GREEK_CASES:
        a=case.volatility*np.sqrt(case.maturity)
        d2=(np.log(case.spot/case.strike)+(case.rate-.5*case.volatility**2)*case.maturity)/a
        p=norm.cdf(d2)
        count=2*n_paths
        group=raw.loc[(raw.scenario==case.name)&(raw.method=='pw_delta')]
        rows.append(dict(scenario=case.name,exercise_probability=p,pathwise_paths=count,
                         expected_active_paths=count*p,theoretical_no_active_probability=float(np.exp(count*np.log1p(-p))),
                         observed_zero_se_rate=float((group.standard_error==0).mean())))
    return pd.DataFrame(rows)


def figures(greeks: pd.DataFrame, asian: pd.DataFrame, ratios: pd.DataFrame, folder: Path) -> None:
    names=[c.name for c in GREEK_CASES]
    methods=[x[0] for x in METHODS]
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),constrained_layout=True)
    coverage=greeks.pivot(index='method',columns='scenario',values='greek_coverage').loc[methods,names]
    zeros=greeks.pivot(index='method',columns='scenario',values='zero_se_rate').loc[methods,names]
    for ax,data,title in ((axes[0],coverage,'Coverage of true Greek'),(axes[1],zeros,'Runs reporting zero SE')):
        im=ax.imshow(data,vmin=0,vmax=1,cmap='viridis')
        ax.set_xticks(range(len(names)),names,rotation=45,ha='right')
        ax.set_yticks(range(len(methods)),methods)
        ax.set(title=title,xlabel='Predeclared scenario')
        for i in range(len(methods)):
            for j in range(len(names)):
                v=data.iloc[i,j]
                ax.text(j,i,f'{v:.0%}',ha='center',va='center',fontsize=7,color='white' if v<.5 else 'black')
        fig.colorbar(im,ax=ax,shrink=.7,label='Fraction of independent runs')
    fig.savefig(folder/'robustness_coverage.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,quantity in zip(axes,('delta','gamma','vega')):
        data=greeks.loc[greeks.greek==quantity]
        for method,group in data.groupby('method',sort=False):
            group=group.set_index('scenario').loc[names]
            ax.plot(names,group.relative_rmse,'o-',label=method)
        ax.set_yscale('log')
        ax.set(title=quantity.capitalize(),ylabel='RMSE / absolute analytical Greek',xlabel='Scenario')
        ax.tick_params(axis='x',rotation=45)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(folder/'robustness_greek_rmse.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),constrained_layout=True)
    cases=[c.name for c in ASIAN_CASES]
    x=np.arange(len(cases))
    for mi,method in enumerate(PRICE_METHODS):
        data=asian.loc[asian.method==method].set_index('scenario').loc[cases]
        axes[0].bar(x+(mi-1)*.24,data.empirical_variance,width=.23,label=method)
        axes[1].plot(x,data.upper_ci_below_lower_bound_rate,'o-',label=method)
    axes[0].set_yscale('log')
    axes[0].set(title='Arithmetic Asian estimator variance',ylabel='Empirical variance (currency²)')
    axes[1].set(title='Intervals provably missing the Asian price',ylabel='Upper CI below geometric lower bound',ylim=(-.03,1.03))
    for ax in axes:
        ax.set_xticks(x,cases)
        ax.set_xlabel('Scenario (12 monitoring dates)')
        ax.legend(fontsize=8)
        ax.grid(axis='y',alpha=.2)
    fig.savefig(folder/'robustness_asian_variance.png',dpi=160)
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-paths',type=int,default=10000)
    parser.add_argument('--n-pilot',type=int,default=1000)
    parser.add_argument('--replications',type=int,default=100)
    parser.add_argument('--bootstrap-resamples',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20261003)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    args=parser.parse_args()
    validate_design(args.n_paths,args.replications,args.seed,args.n_pilot)
    if args.bootstrap_resamples<2:
        parser.error('bootstrap-resamples must be >=2')
    for name in ('raw','tables','figures'):
        (args.output_dir/name).mkdir(parents=True,exist_ok=True)
    greek_raw=collect_greeks(args.n_paths,args.replications,args.seed)
    asian_raw=collect_asian(args.n_paths,args.n_pilot,args.replications,args.seed)
    greeks=summarize_greeks(greek_raw)
    asian=summarize_asian(asian_raw)
    ratios=comparisons(greek_raw,asian_raw,args.seed,args.bootstrap_resamples)
    rare=rare_event_diagnostic(args.n_paths,greek_raw)
    artifacts=[]
    for directory,name,frame in (('raw','robustness_greek_replications',greek_raw),('raw','robustness_asian_replications',asian_raw),
                                 ('tables','robustness_greeks',greeks),('tables','robustness_asian',asian),
                                 ('tables','robustness_comparisons',ratios),('tables','robustness_rare_events',rare)):
        path=args.output_dir/directory/f'{name}.csv'
        frame.to_csv(path,index=False)
        artifacts.append(path)
    figures(greeks,asian,ratios,args.output_dir/'figures')
    metadata=dict(milestone=13,greek_scenarios=[asdict(c) for c in GREEK_CASES],asian_scenarios=[asdict(c) for c in ASIAN_CASES],
                  greek_methods=METHODS,price_methods=PRICE_METHODS,n_paths=args.n_paths,n_pilot=args.n_pilot,
                  replications=args.replications,seed=args.seed,bootstrap_resamples=args.bootstrap_resamples,
                  stream_design='PCG64: Greeks (0,case,rep,method), Asian (1,case,rep,method), bootstrap (2,group,method), warmup prefix 99',
                  budgets='Delta/Vega 2N terminal evaluations; Gamma 3N; PW uses 2N samples; Asians N total stock paths including pilot; CV two payoffs/path',
                  uncertainty='pointwise normal pricing/Greek CIs; Wilson coverage intervals; independent run-level percentile bootstrap variance ratios; no simultaneous or rare-tail guarantee',
                  timing='full pricing/Greek call including draws; excludes Generator construction, warmups, analytical targets, bootstrap and output',
                  limitations='finite R cannot establish unconditional moments of unbounded pilot beta; bootstrap cannot recover unseen rare events; no exact arithmetic-Asian bias/RMSE/coverage',
                  versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
                  sha256={str(p.relative_to(args.output_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    (args.output_dir/'raw/robustness_study.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(greeks.loc[(greeks.method=='pw_delta')|(greeks.scenario.isin(('short','rare'))),
                    ['scenario','method','analytical_value','mean_estimate','deterministic_bias','rmse','greek_coverage','estimator_coverage','zero_se_rate']].to_string(index=False))
    print(asian[['scenario','method','mean_estimate','empirical_variance','zero_se_rate','fallback_rate','max_abs_beta','upper_ci_below_lower_bound_rate']].to_string(index=False))
    print(rare.to_string(index=False))


if __name__=='__main__':
    main()
