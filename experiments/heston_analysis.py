"""M15: coupled timestep refinement at fixed contractual monitoring dates."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform
from time import perf_counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from mc_pricer import (
    HestonModel, EuropeanOption, ArithmeticAsianOption, UpAndOutCall,
    heston_paths_from_normals, simulate_heston_paths, correlate_normals,
    heston_european_reference, price_from_payoff_samples, price_heston_monte_carlo,
)
from mc_pricer.statistics.confidence_intervals import normal_confidence_interval

BASE=HestonModel(100.,.05,.04,2.,.04,.3,-.7)
CASES=(('baseline',BASE),('high_xi',replace(BASE,xi=.9)),
       ('rho_zero',replace(BASE,rho=0.)),('rho_positive',replace(BASE,rho=.5)),
       ('deterministic_variance',replace(BASE,xi=0.,v0=.09)))
GRIDS=(12,24,48,96,192,384)
OBSERVATIONS=12
PRODUCTS=(('european_call',EuropeanOption(100.)),('european_put',EuropeanOption(100.,'put')),
          ('asian_call',ArithmeticAsianOption(100.)),('up_out_call',UpAndOutCall(100.,120.)))
RHOS=(-.9,-.7,0.,.5,.9)


def stream(seed: int, key: tuple[int,...]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed,spawn_key=key))


def validate_design(n_paths: int, replications: int, seed: int, grids: tuple[int,...]) -> None:
    if any(isinstance(x,bool) or not isinstance(x,Integral) for x in (n_paths,replications,seed,*grids)):
        raise TypeError('counts and seed must be integers')
    if n_paths<2 or replications<2 or seed<0 or not grids:
        raise ValueError('require paths/replications>=2, nonnegative seed, nonempty grid list')
    if any(g<OBSERVATIONS or g%OBSERVATIONS for g in grids) or any(b<=a for a,b in zip(grids,grids[1:])) or any(grids[-1]%g for g in grids):
        raise ValueError('increasing nested grids must be multiples of 12 and divide the finest')


def aggregate_normals(values: np.ndarray, n_steps: int) -> np.ndarray:
    """Sum fine Brownian increments into coarse standardized normals (N,m)."""
    n,fine=values.shape
    if n_steps<=0 or fine%n_steps:
        raise ValueError('coarse count must divide fine count')
    block=fine//n_steps
    return values.reshape(n,n_steps,block).sum(axis=2)/np.sqrt(block)


def references() -> pd.DataFrame:
    rows=[]
    for name,model in CASES:
        for kind in ('call','put'):
            first=heston_european_reference(model,100.,1.,kind,cutoff=200.)
            second=heston_european_reference(model,100.,1.,kind,cutoff=400.,tolerance=1e-11)
            if abs(first.price-second.price)>2e-6:
                raise ArithmeticError('Fourier cutoff stability gate failed')
            rows.append(dict(scenario=name,product='european_'+kind,price=second.price,
                             cutoff_200_price=first.price,cutoff_difference=second.price-first.price,
                             quadrature_error_estimate=second.quadrature_error_estimate,
                             method=second.method,cutoff=second.cutoff))
    return pd.DataFrame(rows)


def collect_grids(n_paths: int, replications: int, seed: int, reference: pd.DataFrame,
                  grids: tuple[int,...]=GRIDS) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Use one fine-grid pair of independent drivers per case/replication.

    Across grids and products estimates are correlated. Across replications they
    are independent. Raw paired payoff differences preserve that correlation.
    Timing here excludes generation/aggregation of shared shocks and is not a
    standalone pricer comparison; a separate table times the full library API.
    """
    validate_design(n_paths,replications,seed,grids)
    prices=[]; differences=[]
    ref=reference.set_index(['scenario','product']).price
    for ci,(name,model) in enumerate(CASES):
        for rep in range(replications):
            key=(0,ci,rep)
            rng=stream(seed,key)
            zs=rng.standard_normal((n_paths,grids[-1]))
            zi=rng.standard_normal((n_paths,grids[-1]))
            samples={}
            for step in grids:
                coarse_s=aggregate_normals(zs,step);coarse_i=aggregate_normals(zi,step)
                start=perf_counter()
                paths=heston_paths_from_normals(model,1.,coarse_s,coarse_i,n_observations=OBSERVATIONS)
                simulation_seconds=perf_counter()-start
                discounted_stock=np.exp(-model.rate)*paths.stock_paths[:,-1]
                variance=paths.variance_paths[:,-1]
                for pi,(product_name,product) in enumerate(PRODUCTS):
                    payoff=product.payoff(paths.stock_paths[:,-1] if pi<2 else paths.stock_paths)
                    samples[step,product_name]=payoff
                    out=price_from_payoff_samples(payoff,rate=model.rate,maturity=1.)
                    prices.append(dict(scenario=name,case_index=ci,product=product_name,replication=rep,
                        spawn_key=json.dumps(key),n_paths=n_paths,n_steps=step,n_observations=OBSERVATIONS if pi>=2 else 1,
                        price=out.price,standard_error=out.standard_error,sample_variance=out.sample_variance,
                        ci_lower=out.confidence_interval[0],ci_upper=out.confidence_interval[1],
                        reference_price=ref.get((name,product_name),np.nan),
                        negative_variance_fraction=paths.negative_variance_fraction,
                        discounted_stock_mean=discounted_stock.mean(),discounted_stock_se=discounted_stock.std(ddof=1)/np.sqrt(n_paths),
                        terminal_effective_variance_mean=variance.mean(),terminal_raw_variance_mean=paths.raw_variance_paths[:,-1].mean(),
                        simulation_seconds=simulation_seconds,feller_margin=model.feller_margin))
            for step in grids:
                for product_name,_ in PRODUCTS:
                    difference=samples[step,product_name]-samples[grids[-1],product_name]
                    out=price_from_payoff_samples(difference,rate=model.rate,maturity=1.)
                    differences.append(dict(scenario=name,product=product_name,replication=rep,n_paths=n_paths,n_steps=step,
                        finest_steps=grids[-1],difference=out.price,sample_variance=out.sample_variance,standard_error=out.standard_error))
    return pd.DataFrame(prices),pd.DataFrame(differences)


def pooled_summary(group: pd.DataFrame, column: str) -> dict:
    """Recover all-path variance from equal-size independent run moments."""
    if group.n_paths.nunique()!=1 or len(group)<2:
        raise ValueError('need independent equal-path replications')
    n=int(group.n_paths.iloc[0]); r=len(group); total=n*r
    values=group[column].to_numpy(); mean=float(values.mean())
    m2=float(((n-1)*group.sample_variance+n*(group[column]-mean)**2).sum())
    variance=m2/(total-1); se=np.sqrt(variance/total)
    lo,hi=normal_confidence_interval(mean,se)
    return dict(n_paths_per_run=n,n_replications=r,total_paths=total,mean_estimate=mean,
                pooled_sample_variance=variance,pooled_standard_error=se,ci_lower=lo,ci_upper=hi,
                empirical_estimator_variance=values.var(ddof=1),empirical_se_of_mean=values.std(ddof=1)/np.sqrt(r))


def summarize_prices(raw: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (case,product,steps),group in raw.groupby(['scenario','product','n_steps'],sort=False):
        target=group.reference_price.iloc[0]
        row=dict(scenario=case,product=product,n_steps=steps,n_observations=int(group.n_observations.iloc[0]),
                 **pooled_summary(group,'price'),reference_price=target,
                 mean_negative_variance_fraction=group.negative_variance_fraction.mean(),
                 mean_discounted_stock=group.discounted_stock_mean.mean(),
                 mean_effective_terminal_variance=group.terminal_effective_variance_mean.mean(),
                 mean_raw_terminal_variance=group.terminal_raw_variance_mean.mean(),
                 feller_margin=group.feller_margin.iloc[0],mean_simulation_seconds=group.simulation_seconds.mean())
        row['estimated_bias']=row['mean_estimate']-target
        row['bias_in_pooled_se']=row['estimated_bias']/row['pooled_standard_error'] if np.isfinite(target) and row['pooled_standard_error']>0 else np.nan
        row['run_rmse']=np.sqrt(np.mean((group.price-target)**2)) if np.isfinite(target) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_differences(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([dict(scenario=case,product=product,n_steps=steps,finest_steps=int(group.finest_steps.iloc[0]),
                              **pooled_summary(group,'difference'))
        for (case,product,steps),group in raw.groupby(['scenario','product','n_steps'],sort=False)])


def correlation_study(seed: int, replications: int=30, n_pairs: int=100000) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]
    for ri,rho in enumerate(RHOS):
        for rep in range(replications):
            key=(1,ri,rep);z=stream(seed,key).standard_normal((n_pairs,2))
            zv=correlate_normals(z[:,0],z[:,1],rho)
            corr=np.corrcoef(z[:,0],zv)[0,1]
            half=norm.ppf(.975)/np.sqrt(n_pairs-3)
            lo,hi=np.tanh(np.arctanh(corr)+np.array([-half,half]))
            rows.append(dict(rho=rho,replication=rep,spawn_key=json.dumps(key),n_pairs=n_pairs,
                             sample_correlation=corr,mean_variance_shock=zv.mean(),variance_shock_variance=zv.var(ddof=1),
                             ci_lower=lo,ci_upper=hi))
    raw=pd.DataFrame(rows);table=[]
    for rho,group in raw.groupby('rho',sort=False):
        table.append(dict(rho=rho,n_replications=replications,n_pairs=n_pairs,
                          mean_correlation=group.sample_correlation.mean(),empirical_sd=group.sample_correlation.std(ddof=1),
                          mean_shock_variance=group.variance_shock_variance.mean(),
                          fisher_interval_coverage=((group.ci_lower<=rho)&(rho<=group.ci_upper)).mean()))
    return raw,pd.DataFrame(table)


def pricing_examples(seed: int, n_paths: int) -> pd.DataFrame:
    rows=[]
    for ci,(name,model) in enumerate(CASES):
        for pi,(product_name,product) in enumerate(PRODUCTS):
            out=price_heston_monte_carlo(model,product,1.,n_paths,384,rng=stream(seed,(2,ci,pi)),
                                       n_observations=12 if pi>=2 else None)
            rows.append(dict(scenario=name,product=product_name,**asdict(out)))
    return pd.DataFrame(rows)


def figures(prices: pd.DataFrame, differences: pd.DataFrame, correlations: pd.DataFrame, folder: Path) -> None:
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for name,_ in CASES:
        data=prices.loc[(prices.scenario==name)&(prices['product']=='european_call')]
        axes[0].errorbar(data.n_steps,data.estimated_bias,yerr=1.96*data.pooled_standard_error,marker='o',label=name)
        axes[1].plot(data.n_steps,data.mean_negative_variance_fraction,'o-',label=name)
    axes[0].axhline(0,color='k',ls=':')
    axes[0].set(title='European call error against Fourier / deterministic reference',ylabel='Estimated bias ± 1.96 pooled SE (currency)')
    axes[1].set(title='Negative auxiliary variance updates',ylabel='Fraction of all path-step updates')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks(GRIDS,[str(g) for g in GRIDS]);ax.set_xlabel('Integration steps (T=1 year)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(folder/'heston_time_convergence.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,product in zip(axes,('asian_call','up_out_call')):
        for name,_ in CASES:
            data=differences.loc[(differences.scenario==name)&(differences['product']==product)]
            ax.errorbar(data.n_steps,data.mean_estimate,yerr=1.96*data.pooled_standard_error,marker='o',label=name)
        ax.axhline(0,color='k',ls=':');ax.set_xscale('log',base=2);ax.set_xticks(GRIDS,[str(g) for g in GRIDS])
        ax.set(title=f'{product}: fixed 12 monitoring dates',xlabel='Integration steps',ylabel='Paired price difference vs 384 steps (currency)')
        ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(folder/'heston_exotic_convergence.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4),constrained_layout=True)
    ax.errorbar(correlations.rho,correlations.mean_correlation-correlations.rho,
                yerr=1.96*correlations.empirical_sd/np.sqrt(correlations.n_replications),fmt='o')
    ax.axhline(0,color='k',ls=':');ax.set(title='Correlated Gaussian drivers: repeated validation',xlabel='Target rho',ylabel='Mean sample correlation minus rho ± 1.96 SE');ax.grid(alpha=.2)
    fig.savefig(folder/'heston_driver_correlation.png',dpi=160);plt.close(fig)


def save_outputs(raw: pd.DataFrame, paired: pd.DataFrame, ref: pd.DataFrame,
                 corr_raw: pd.DataFrame, corr: pd.DataFrame, examples: pd.DataFrame,
                 *, output_dir: Path, n_paths: int, replications: int, seed: int) -> None:
    """Summarize and persist completed runs; also permits recovery without rerunning timings."""
    price_summary=summarize_prices(raw);paired_summary=summarize_differences(paired)
    paths=[]
    for directory,name,frame in (('raw','heston_grid_replications',raw),('raw','heston_paired_replications',paired),
        ('raw','heston_correlation_replications',corr_raw),('tables','heston_references',ref),
        ('tables','heston_convergence',price_summary),('tables','heston_paired_convergence',paired_summary),
        ('tables','heston_correlations',corr),('tables','heston_exotic_prices',examples)):
        path=output_dir/directory/f'{name}.csv';frame.to_csv(path,index=False);paths.append(path)
    figures(price_summary,paired_summary,corr,output_dir/'figures')
    metadata=dict(milestone=15,seed=seed,n_paths=n_paths,replications=replications,grids=GRIDS,
                  maturity=1.,monitoring_dates=OBSERVATIONS,cases={name:asdict(model) for name,model in CASES},
                  products={name:asdict(product) for name,product in PRODUCTS},correlation_replications=30,correlation_pairs=100000,
                  streams='PCG64 SeedSequence: grids (0,case,rep), driver correlation (1,rho,rep), end-to-end examples (2,case,product)',
                  coupling='Each run draws all stock normals then independent normals at finest grid, both (N,384); coarse normals sum blocks/sqrt(block); fixed 12 dates for exotics',
                  timing='Grid simulation_seconds excludes RNG/aggregation and is shared across products; examples table times complete independent library calls',
                  scheme='auxiliary-state full truncation + left-point exponential stock; no state absorption',
                  uncertainty='Plain IID normal sampling intervals exclude integration bias. Grid differences use paired paths. 384 is a numerical reference for exotics, not exact.',
                  reference='Lewis inversion of affine characteristic function; cutoff 200 vs 400 checks, independent Riccati-ODE tests, deterministic-variance limits; quadrature error is not a tail bound',
                  versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
                  sha256={str(p.relative_to(output_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (output_dir/'raw/heston_study.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(price_summary.loc[(price_summary['product']=='european_call')&(price_summary.n_steps.isin([12,384])),
         ['scenario','n_steps','mean_estimate','reference_price','estimated_bias','pooled_standard_error','mean_negative_variance_fraction']].to_string(index=False))
    print(paired_summary.loc[(paired_summary.scenario=='high_xi')&(paired_summary.n_steps==12),['product','mean_estimate','pooled_standard_error']].to_string(index=False))
    print(corr.to_string(index=False))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-paths',type=int,default=4000)
    parser.add_argument('--replications',type=int,default=30)
    parser.add_argument('--seed',type=int,default=20261005)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    args=parser.parse_args();validate_design(args.n_paths,args.replications,args.seed,GRIDS)
    for directory in ('raw','tables','figures'):(args.output_dir/directory).mkdir(parents=True,exist_ok=True)
    ref=references()
    raw,paired=collect_grids(args.n_paths,args.replications,args.seed,ref)
    corr_raw,corr=correlation_study(args.seed)
    examples=pricing_examples(args.seed,args.n_paths)
    save_outputs(raw,paired,ref,corr_raw,corr,examples,output_dir=args.output_dir,
                 n_paths=args.n_paths,replications=args.replications,seed=args.seed)


if __name__=='__main__':main()
