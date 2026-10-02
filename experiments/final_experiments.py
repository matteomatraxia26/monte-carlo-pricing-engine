"""M16: fresh replicated benchmarks and an auditable index of final evidence.

Run: python -m experiments.final_experiments
Rebuild figures/tables without resimulation: add --rebuild.
This consolidates completed studies; it does not claim to rerun their raw draws.
"""
import argparse
from dataclasses import asdict, replace
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

from mc_pricer import (
    BlackScholesModel, EuropeanOption, black_scholes_price, black_scholes_greeks,
    price_monte_carlo, finite_difference_greek, pathwise_delta, likelihood_ratio_delta,
)
from mc_pricer.statistics.confidence_intervals import binomial_wilson_interval
from experiments.variance_reduction import study_cases
from experiments.estimator_robustness import bootstrap_variance_ratio

BASE = BlackScholesModel(100., .05, .2)
GREEKS = ('delta', 'gamma', 'vega')
# Frozen before running the new seed; paths include control pilot costs.
DEFAULT_BUDGET = 12000
DEFAULT_REPLICATIONS = 50
DEFAULT_SEED = 20261006
DEFAULT_PILOT = 1200

# Requirement, figure, numerical source, exact reproduction command.
EVIDENCE = (
 ('MC estimate vs N', 'mc_price_vs_n', 'mc_convergence', 'mc_convergence --replications 200 --seed 20260925'),
 ('Absolute pricing error vs N', 'mc_absolute_error_vs_n', 'mc_convergence', 'mc_convergence --replications 200 --seed 20260925'),
 ('Standard error vs N', 'mc_standard_error_vs_n', 'mc_convergence', 'mc_convergence --replications 200 --seed 20260925'),
 ('Log-log SE convergence', 'mc_loglog_convergence', 'mc_convergence_slopes', 'mc_convergence --replications 200 --seed 20260925'),
 ('Antithetic variance', 'antithetic_variance_reduction', 'antithetic_comparison', 'variance_reduction --n-paths 10000 --replications 200 --seed 20260929'),
 ('Control-variate variance', 'control_variate_variance_reduction', 'control_comparison', 'control_variates --n-paths 10000 --n-pilot 1000 --replications 200 --seed 20260930'),
 ('Delta convergence', 'pathwise_delta_convergence', 'pathwise_convergence', 'pathwise_delta --n-paths 20000 --replications 200 --seed 20261002'),
 ('FD vs pathwise Delta', 'finite_difference_vs_pathwise_delta', 'pathwise_variance_comparison', 'pathwise_delta --n-paths 20000 --replications 200 --seed 20261002'),
 ('Greek error vs bump', 'greek_error_vs_bump', 'fd_greek_bumps', 'greeks_comparison --n-paths 20000 --replications 100 --seed 20261001'),
 ('Asian monitoring convergence', 'asian_monitoring_convergence', 'asian_monitoring', 'asian_pricing --n-paths 5000 --replications 20 --seed 20260927'),
 ('Barrier monitoring convergence', 'barrier_monitoring_convergence', 'barrier_monitoring', 'barrier_pricing --n-paths 5000 --replications 30 --seed 20260928'),
 ('Heston integration convergence', 'heston_time_convergence', 'heston_convergence', 'heston_analysis --n-paths 4000 --replications 30 --seed 20261005'),
)
EXTRA_SOURCES = ('tables/fd_greek_comparison.csv', 'tables/lr_comparisons.csv',
                 'tables/european_validation.csv', 'tables/heston_paired_convergence.csv')


def stream(seed: int, key: tuple[int, ...]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def validate_design(budget: int, replications: int, seed: int, pilot: int) -> None:
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (budget, replications, seed, pilot)):
        raise TypeError('budget, replications, seed and pilot must be integers')
    if budget < 12 or budget % 6 or replications < 2 or seed < 0 or not 2 <= pilot <= budget-2:
        raise ValueError('budget>=12 divisible by 6, replications>=2, seed>=0, 2<=pilot<=budget-2 required')


def methods(case_index: int) -> tuple[str, ...]:
    if case_index < 5:
        return ('none', 'antithetic', 'control_variate') if case_index == 2 else ('none', 'antithetic')
    return ('fd', 'pw', 'lr') if case_index == 5 else ('fd',)


def run_cell(case_index: int, method: str, budget: int, pilot: int,
             rng: np.random.Generator) -> dict:
    """One full estimator call; quantities and computational budgets stay distinct."""
    if case_index not in range(8) or method not in methods(case_index):
        raise ValueError('unknown case/method')
    if case_index < 5:
        case = study_cases()[case_index]
        result = price_monte_carlo(case.model, case.product, case.maturity, budget,
            n_steps=case.n_steps, variance_reduction=method, rng=rng,
            n_pilot_paths=pilot if method == 'control_variate' else None)
        target = np.nan if case.analytical_target is None else case.analytical_target
        config = dict(case=case.name, quantity='price', units='currency',
            budget_unit='simulated_paths_including_pilot', budget=budget,
            n_paths=result.n_paths, n_independent_samples=result.n_independent_samples,
            n_payoff_evaluations=budget*(2 if method == 'control_variate' else 1),
            n_pilot_paths=result.n_pilot_paths, n_steps=case.n_steps or 1,
            analytical_value=target, estimator_target=target, bump=np.nan)
        estimate = result.price
    else:
        greek = GREEKS[case_index-5]
        product = EuropeanOption(100.)
        exact = getattr(black_scholes_greeks(BASE, 100., 1.), greek)
        h = .01 if greek == 'vega' else 1.
        target = exact
        if method == 'fd':
            legs = 3 if greek == 'gamma' else 2
            result = finite_difference_greek(BASE, product, 1., budget//legs,
                greek=greek, bump=h, rng=rng)
            field = 'volatility' if greek == 'vega' else 'spot'
            center = getattr(BASE, field)
            up = black_scholes_price(replace(BASE, **{field:center+h}), 100., 1.)
            down = black_scholes_price(replace(BASE, **{field:center-h}), 100., 1.)
            base = black_scholes_price(BASE, 100., 1.)
            target = ((up-base)+(down-base))/h**2 if greek == 'gamma' else (up-down)/(2*h)
        else:
            fn = pathwise_delta if method == 'pw' else likelihood_ratio_delta
            result = fn(BASE, product, 1., budget, rng=rng)
        config = dict(case='european_call_'+greek, quantity=greek,
            units={'delta':'price / spot', 'gamma':'price / spot^2', 'vega':'price / unit volatility'}[greek],
            budget_unit='terminal_evaluations', budget=budget, n_paths=result.n_paths,
            n_independent_samples=result.n_paths, n_payoff_evaluations=budget,
            n_pilot_paths=0, n_steps=1, analytical_value=exact, estimator_target=target,
            bump=h if method == 'fd' else np.nan)
        estimate = result.estimate
    return dict(**config, method=method, estimate=estimate, standard_error=result.standard_error,
        reported_estimator_variance=result.estimator_variance, ci_lower=result.confidence_interval[0],
        ci_upper=result.confidence_interval[1], runtime_seconds=result.runtime_seconds)


def collect(budget: int, replications: int, seed: int, pilot: int) -> pd.DataFrame:
    validate_design(budget, replications, seed, pilot)
    rows = []
    for ci in range(8):
        choices = methods(ci)
        for mi, method in enumerate(choices):
            run_cell(ci, method, 120, 12, stream(seed, (99, ci, mi)))
        for rep in range(replications):
            for index in np.roll(np.arange(len(choices)), rep % len(choices)):
                mi = int(index)
                key = (0, ci, rep, mi)
                row = run_cell(ci, choices[mi], budget, pilot, stream(seed, key))
                rows.append(dict(case_index=ci, method_index=mi, replication=rep,
                                 spawn_key=json.dumps(key), **row))
    return pd.DataFrame(rows)


def validate_raw(raw: pd.DataFrame) -> None:
    """Reject mixed targets/budgets or missing runs before statistical summaries."""
    if raw.empty or set(raw.case_index) != set(range(8)) or raw.duplicated(['case_index', 'method', 'replication']).any():
        raise ValueError('empty or duplicate replications')
    for ci, case in raw.groupby('case_index'):
        if set(case.method) != set(methods(ci)) or case.budget.nunique() != 1:
            raise ValueError('missing method or mixed budgets')
        for column in ('case', 'quantity', 'units', 'budget_unit', 'n_steps', 'analytical_value'):
            if case[column].nunique(dropna=False) != 1:
                raise ValueError('mixed case configuration: '+column)
        expected = None
        for _, group in case.groupby('method'):
            reps = set(group.replication)
            if len(reps) < 2 or reps != set(range(len(reps))) or (expected is not None and reps != expected):
                raise ValueError('incomplete independent replication panel')
            expected = reps
            for column in ('case', 'quantity', 'units', 'budget_unit', 'n_paths', 'n_independent_samples',
                           'n_payoff_evaluations', 'n_pilot_paths', 'n_steps', 'analytical_value', 'estimator_target', 'bump'):
                if group[column].nunique(dropna=False) != 1:
                    raise ValueError('mixed configuration: '+column)
            if not np.isfinite(group[['estimate','standard_error','reported_estimator_variance','runtime_seconds']]).all().all():
                raise ValueError('nonfinite estimate or uncertainty')
            if (group[['standard_error','reported_estimator_variance','runtime_seconds']] < 0).any().any():
                raise ValueError('negative uncertainty or timing')


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    validate_raw(raw)
    rows = []
    for (case, method), group in raw.groupby(['case','method'], sort=False):
        first = group.iloc[0]; values = group.estimate.to_numpy(); r = len(values)
        mean = values.mean(); variance = values.var(ddof=1); se_mean = np.sqrt(variance/r)
        exact = first.analytical_value; known = np.isfinite(exact)
        target = first.estimator_target
        covered = int(((group.ci_lower<=exact)&(exact<=group.ci_upper)).sum()) if known else 0
        lo, hi = binomial_wilson_interval(covered,r) if known else (np.nan,np.nan)
        rows.append(dict(case=case, method=method, quantity=first.quantity, units=first.units,
            budget_unit=first.budget_unit, budget=first.budget, n_paths=first.n_paths,
            n_independent_samples=first.n_independent_samples, n_payoff_evaluations=first.n_payoff_evaluations,
            n_pilot_paths=first.n_pilot_paths, n_steps=first.n_steps, bump=first.bump,
            n_replications=r, mean_estimate=mean, analytical_value=exact, estimator_target=target,
            empirical_bias=mean-exact if known else np.nan,
            deterministic_fd_bias=target-exact if known else np.nan,
            empirical_variance=variance, rmse=np.sqrt(np.mean((values-exact)**2)) if known else np.nan,
            empirical_se_of_mean=se_mean, mean_ci_lower=mean-1.96*se_mean, mean_ci_upper=mean+1.96*se_mean,
            mean_reported_se=group.standard_error.mean(), mean_reported_variance=group.reported_estimator_variance.mean(),
            exact_target_coverage=covered/r if known else np.nan, coverage_ci_lower=lo, coverage_ci_upper=hi,
            estimator_target_coverage=((group.ci_lower<=target)&(target<=group.ci_upper)).mean() if known else np.nan,
            zero_se_fraction=(group.standard_error==0).mean(), mean_runtime_seconds=group.runtime_seconds.mean(),
            median_runtime_seconds=group.runtime_seconds.median(), runtime_p90_seconds=group.runtime_seconds.quantile(.9)))
    return pd.DataFrame(rows)


def compare(raw: pd.DataFrame, seed: int, resamples: int = 2000) -> pd.DataFrame:
    validate_raw(raw)
    rows = []
    for ci, group in raw.groupby('case_index', sort=True):
        reference_name = 'none' if ci<5 else 'fd'
        reference = group.loc[group.method==reference_name]
        for mi, candidate in group.loc[group.method!=reference_name].groupby('method_index',sort=True):
            ratio = bootstrap_variance_ratio(reference.estimate.to_numpy(), candidate.estimate.to_numpy(),
                rng=stream(seed,(1,int(ci),int(mi))),resamples=resamples)
            exact = reference.analytical_value.iloc[0]
            mse_candidate = np.mean((candidate.estimate-exact)**2)
            rows.append(dict(case=group.case.iloc[0],reference=reference_name,candidate=candidate.method.iloc[0],
                budget=group.budget.iloc[0],budget_unit=group.budget_unit.iloc[0],
                bootstrap_spawn_key=json.dumps((1,int(ci),int(mi))), **ratio,
                empirical_variance_reduction_percent=100*(1-1/ratio['empirical_vrf']) if np.isfinite(ratio['empirical_vrf']) else np.nan,
                empirical_mse_ratio=np.mean((reference.estimate-exact)**2)/mse_candidate if np.isfinite(exact) and mse_candidate>0 else np.nan,
                mean_runtime_ratio=reference.runtime_seconds.mean()/candidate.runtime_seconds.mean(),
                time_adjusted_efficiency=ratio['empirical_vrf']*reference.runtime_seconds.mean()/candidate.runtime_seconds.mean()))
    return pd.DataFrame(rows)


def evidence_index(source: Path) -> pd.DataFrame:
    rows = []
    for requirement, figure, table, command in EVIDENCE:
        fig = source/'figures'/f'{figure}.png'; csv = source/'tables'/f'{table}.csv'
        if not fig.is_file() or not csv.is_file():
            raise FileNotFoundError(f'missing required evidence: {fig} or {csv}')
        if fig.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n' or pd.read_csv(csv).empty:
            raise ValueError('invalid or empty evidence artifact')
        rows.append(dict(requirement=requirement,figure=str(fig.relative_to(source)),
            table=str(csv.relative_to(source)),figure_sha256=hashlib.sha256(fig.read_bytes()).hexdigest(),
            table_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
            reproduce_command='python -m experiments.'+command))
    for name in EXTRA_SOURCES:
        if not (source/name).is_file(): raise FileNotFoundError(source/name)
    return pd.DataFrame(rows)


def historical_highlights(source: Path) -> pd.DataFrame:
    """Selected earlier results keep their source, budget and interval semantics."""
    rows = []
    def add(file: str, selectors: dict, metric: str, scope: str, lower: str | None=None, upper: str | None=None):
        frame=pd.read_csv(source/'tables'/file)
        for key,value in selectors.items(): frame=frame.loc[frame[key]==value]
        if len(frame)!=1: raise ValueError('historical selection is not unique: '+file)
        row=frame.iloc[0]
        rows.append(dict(source_table='tables/'+file,selection=json.dumps(selectors,sort_keys=True),
            metric=metric,value=row[metric],interval_lower=row[lower] if lower else np.nan,
            interval_upper=row[upper] if upper else np.nan,scope=scope))
    for kind in ('call','put'):
        add('mc_convergence_slopes.csv',{'product':kind},'log_mean_se_slope','M5: descriptive SE exponent; 200 runs/N; 10 counts; no interval for this fitted aggregate slope')
    add('antithetic_comparison.csv',{'case':'european_call'},'empirical_vrf','M9: plain/anti variance; 200 runs; 10000 payoff evaluations; approximate F interval','vrf_ci_lower','vrf_ci_upper')
    add('control_comparison.csv',{'case':'asian_call_12','method':'control_variate'},'empirical_vrf','M10: plain/control variance; 200 runs; 10000 paths including pilot; approximate F interval','vrf_ci_lower','vrf_ci_upper')
    add('fd_greek_comparison.csv',{'greek':'delta','bump':1.},'empirical_vrf','M11: independent/CRN variance; 100 runs; 40000 terminal evaluations; approximate F interval','vrf_ci_lower','vrf_ci_upper')
    add('pathwise_variance_comparison.csv',{'budget':'equal_terminal_evaluations','method':'fd_h1'},'fd_over_pw_variance','M12: FD/PW variance; 200 runs; 40000 terminal evaluations; approximate F interval','ratio_ci_lower','ratio_ci_upper')
    add('lr_comparisons.csv',{'scenario':'digital_call','candidate':'fd_h1'},'empirical_vrf','M14: LR/FD variance; 200 runs; 20000 terminal evaluations; independent percentile bootstrap','bootstrap_lower','bootstrap_upper')
    for steps in (12,384):
        add('heston_convergence.csv',{'scenario':'high_xi','product':'european_call','n_steps':steps},'estimated_bias','M15: 30 runs x 4000 paths; Fourier target; pooled SE in source table; sampling error excludes timestep bias')
    return pd.DataFrame(rows)


def figures(summary: pd.DataFrame, comparisons: pd.DataFrame, folder: Path) -> None:
    fig,ax=plt.subplots(figsize=(10,5.2),constrained_layout=True)
    positions=np.arange(len(comparisons)); values=comparisons.empirical_vrf.to_numpy()
    # Endpoints are drawn directly; an asymmetric bootstrap interval need not contain its point estimate.
    ax.hlines(positions,comparisons.bootstrap_lower,comparisons.bootstrap_upper,color='tab:blue')
    ax.plot(values,positions,'o',color='tab:blue');ax.axvline(1,color='black',ls=':')
    ax.set_yticks(positions,[f'{r.case}: {r.reference}/{r.candidate}' for r in comparisons.itertuples()])
    ax.set_xscale('log');ax.set(xlabel='Reference variance / candidate variance (log scale)',
        title='Final benchmark: empirical variance ratios and 95% bootstrap intervals')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.savefig(folder/'final_variance_comparisons.png',dpi=160);plt.close(fig)
    data=summary.loc[summary.analytical_value.notna() & (summary.empirical_se_of_mean>0)]
    fig,ax=plt.subplots(figsize=(9,5.5),constrained_layout=True)
    pos=np.arange(len(data));z=data.empirical_bias/data.empirical_se_of_mean
    ax.errorbar(z,pos,xerr=1.96,fmt='o');ax.axvline(0,color='black',ls=':')
    ax.set_yticks(pos,[f'{r.case}: {r.method}' for r in data.itertuples()]);ax.invert_yaxis()
    ax.set(xlabel='(Mean estimate − exact target) / empirical SE of mean',
        title='Final benchmark: estimated bias with pointwise normal uncertainty')
    ax.grid(axis='x',alpha=.2);fig.savefig(folder/'final_bias_diagnostics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for ax,case in zip(axes,('asian_call_12','european_call_delta')):
        data=summary.loc[summary.case==case]
        for row in data.itertuples():
            ax.scatter(row.mean_runtime_seconds*1000,row.empirical_variance,s=40)
            ax.annotate(row.method,(row.mean_runtime_seconds*1000,row.empirical_variance),xytext=(5,5),textcoords='offset points')
        ax.set(xlabel='Mean full estimator-call runtime (ms)',
               ylabel='Estimator variance (currency²)' if case=='asian_call_12' else 'Estimator variance ((price / spot)²)',title=case)
        ax.set_yscale('log');ax.margins(x=.3,y=.3);ax.grid(alpha=.2)
    fig.savefig(folder/'final_runtime_variance.png',dpi=160);plt.close(fig)


def run(source: Path, output: Path, *, budget: int, replications: int, seed: int,
        pilot: int, resamples: int, rebuild: bool=False) -> None:
    validate_design(budget,replications,seed,pilot)
    if isinstance(resamples,bool) or not isinstance(resamples,Integral) or resamples<2:
        raise ValueError('bootstrap resamples must be an integer >=2')
    index=evidence_index(source)  # Fail before simulations if published evidence is missing.
    source_files=set(index.figure)|set(index.table)|set(EXTRA_SOURCES)
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in sorted(source_files)}
    for sub in ('raw','tables','figures'): (output/sub).mkdir(parents=True,exist_ok=True)
    raw_path=output/'raw/final_benchmark_replications.csv'; meta_path=output/'raw/final_study.json'
    if rebuild:
        old=json.loads(meta_path.read_text())
        if hashes!=old['source_sha256'] or hashlib.sha256(raw_path.read_bytes()).hexdigest()!=old['output_sha256']['raw/final_benchmark_replications.csv']:
            raise ValueError('source evidence or raw benchmark changed since the recorded run')
        budget,replications,seed,pilot,resamples=(old[k] for k in ('budget','replications','seed','pilot','bootstrap_resamples'))
        raw=pd.read_csv(raw_path,float_precision='round_trip')
    else:
        raw=collect(budget,replications,seed,pilot)
    summary=summarize(raw);comparison=compare(raw,seed,resamples);highlights=historical_highlights(source)
    outputs=[]
    for relative,frame in [('raw/final_benchmark_replications.csv',raw),('tables/final_benchmark.csv',summary),
        ('tables/final_comparisons.csv',comparison),('tables/final_evidence_index.csv',index),('tables/final_historical_highlights.csv',highlights)]:
        path=output/relative;frame.to_csv(path,index=False);outputs.append(path)
    figures(summary,comparison,output/'figures')
    outputs.extend(output/'figures'/f'{name}.png' for name in ('final_variance_comparisons','final_bias_diagnostics','final_runtime_variance'))
    metadata=dict(milestone=16,budget=budget,replications=replications,seed=seed,pilot=pilot,bootstrap_resamples=resamples,
        benchmark_rows=len(raw),summary_rows=len(summary),required_figures=len(index),source_sha256=hashes,
        price_cases=[asdict(case) for case in study_cases()],greek_model=asdict(BASE),greek_strike=100.,greek_maturity=1.,
        code_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((Path(__file__).resolve().parents[1]/'src/mc_pricer').rglob('*.py'))
            +[Path(__file__).resolve(),Path(__file__).with_name('variance_reduction.py'),Path(__file__).with_name('estimator_robustness.py')]},
        source_directory=str(source.resolve()),output_sha256={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
        design='Five fixed M9 price scenarios; ATM European call Delta/Gamma/Vega. New independent method/replication streams; rotating order; no retuning.',
        budgets='Price comparisons use equal simulated paths INCLUDING pilot; control evaluates two payoffs/path. Greek comparisons use equal terminal evaluations. No cross-case variance ranking.',
        streams='PCG64 SeedSequence: (0,case,rep,method) simulation; (1,case,candidate_index) bootstrap; (99,case,method) 120-row warmup.',
        timing='Full estimator API runtime, excluding Generator construction, reference calculations, warmup, bootstrap and I/O. Environment-specific observations, no timing confidence interval.',
        uncertainty='Empirical variance ddof=1; RMSE divisor R; normal intervals for repeated mean; Wilson run coverage; independent run-level percentile bootstrap. Pointwise, no simultaneous/rare-tail guarantee.',
        scope='Fresh GBM benchmark; original convergence/monitoring/Heston evidence indexed and hashed, not resimulated. Unknown Asian/barrier exact bias and RMSE remain missing.',
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__))
    meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    print(summary[['case','method','mean_estimate','empirical_bias','empirical_variance','rmse','mean_runtime_seconds']].to_string(index=False))
    print(comparison[['case','reference','candidate','empirical_vrf','bootstrap_lower','bootstrap_upper','time_adjusted_efficiency']].to_string(index=False))
    print(f'{len(index)}/12 required figures verified; {len(raw)} new benchmark rows.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    default=Path(__file__).resolve().parents[1]/'results'
    parser.add_argument('--source-dir',type=Path,default=default)
    parser.add_argument('--output-dir',type=Path,default=default)
    parser.add_argument('--budget',type=int,default=DEFAULT_BUDGET)
    parser.add_argument('--replications',type=int,default=DEFAULT_REPLICATIONS)
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--pilot',type=int,default=DEFAULT_PILOT)
    parser.add_argument('--bootstrap-resamples',type=int,default=2000)
    parser.add_argument('--rebuild',action='store_true',help='Verify hashes and rebuild from saved rows; recorded design overrides CLI counts/seed.')
    args=parser.parse_args()
    run(args.source_dir,args.output_dir,budget=args.budget,replications=args.replications,
        seed=args.seed,pilot=args.pilot,resamples=args.bootstrap_resamples,rebuild=args.rebuild)


if __name__=='__main__': main()
