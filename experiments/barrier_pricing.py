"""Milestone 8: nested discrete barrier monitoring and barrier-level sensitivity.

Run: python -m experiments.barrier_pricing
The finest grid remains discrete. No continuous-barrier correction is applied.
"""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from mc_pricer import (BlackScholesModel, UpAndOutCall, black_scholes_price,
                       price_from_payoff_samples, price_monte_carlo, simulate_paths,
                       terminal_monitored_up_and_out_price)
from mc_pricer.statistics.confidence_intervals import binomial_wilson_interval
from mc_pricer.statistics.diagnostics import summarize_replications

DEFAULT_MONITORING = (1, 4, 12, 24, 48, 96, 192, 384)


def collect_extrema(
    model: BlackScholesModel, maturity: float, product: UpAndOutCall,
    counts: Sequence[int], *, n_paths: int, replications: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return terminals (R,N), maxima (R,N,k), and per-run pricing records.

    Every successive grid must refine the preceding grid, not merely divide
    the finest grid. One independent child stream per run; grids share paths.
    Maxima include S0 and ST. Retain compact summaries, release full paths.
    """
    counts = tuple(counts)
    if len(counts) < 2:
        raise ValueError("require at least two monitoring counts")
    if any(isinstance(x, bool) or not isinstance(x, Integral)
           for x in (*counts, n_paths, replications, seed)):
        raise TypeError("counts and seed must be integers")
    if any(m < 1 for m in counts) or any(b <= a or b % a for a,b in zip(counts, counts[1:])):
        raise ValueError("monitoring counts must be strictly increasing nested divisors")
    if n_paths < 2 or replications < 2 or seed < 0:
        raise ValueError("require paths >= 2, replications >= 2 and seed >= 0")
    terminal = np.empty((replications, n_paths))
    maxima = np.empty((replications, n_paths, len(counts)))
    rows = []
    for replication, stream in enumerate(np.random.SeedSequence(seed).spawn(replications)):
        start = perf_counter()
        paths = simulate_paths(model, maturity, n_paths, counts[-1], rng=np.random.default_rng(stream))
        simulation_seconds = perf_counter() - start
        terminal[replication] = paths[:, -1]
        for column, m in enumerate(counts):
            start = perf_counter()
            maximum = paths[:, ::counts[-1] // m].max(axis=1)
            maxima[replication, :, column] = maximum
            cashflow = product.payoff_from_extrema(terminal[replication], maximum)
            result = price_from_payoff_samples(cashflow, rate=model.rate, maturity=maturity)
            rows.append({"replication": replication, "spawn_key": json.dumps(list(stream.spawn_key)),
                         "n_paths": n_paths, "n_steps": m, "barrier": product.barrier,
                         "price": result.price, "standard_error": result.standard_error,
                         "sample_variance": result.sample_variance,
                         "ci_lower": result.confidence_interval[0], "ci_upper": result.confidence_interval[1],
                         "knockout_fraction": float(np.mean(maximum >= product.barrier)),
                         "shared_simulation_seconds": simulation_seconds,
                         "evaluation_seconds": perf_counter() - start})
        del paths
    return terminal, maxima, pd.DataFrame(rows)


def summarize_extrema(
    terminal: np.ndarray, maxima: np.ndarray, model: BlackScholesModel,
    maturity: float, product: UpAndOutCall, counts: Sequence[int],
) -> pd.DataFrame:
    """Summarize IID paths and paired differences from the finest discrete grid."""
    if terminal.ndim != 2 or maxima.shape != (*terminal.shape, len(counts)):
        raise ValueError("expected terminal (R,N) and maxima (R,N,k)")
    replications, paths_per_run = terminal.shape
    terminal_flat = terminal.reshape(-1)
    reference_maximum = maxima[:, :, -1].reshape(-1)
    reference = product.payoff_from_extrema(terminal_flat, reference_maximum)
    vanilla = np.maximum(terminal_flat - product.strike, 0)
    discount = np.exp(-model.rate * maturity)
    rows = []
    for column, m in enumerate(counts):
        maximum = maxima[:, :, column].reshape(-1)
        payoffs = product.payoff_from_extrema(terminal_flat, maximum)
        result = price_from_payoff_samples(payoffs, rate=model.rate, maturity=maturity)
        difference = price_from_payoff_samples(payoffs-reference, rate=model.rate, maturity=maturity)
        hit = maximum >= product.barrier
        missed = (~hit) & (reference_maximum >= product.barrier)
        hit_ci = binomial_wilson_interval(int(hit.sum()), len(hit))
        run_prices = discount * payoffs.reshape(replications, paths_per_run).mean(axis=1)
        empirical_sd = run_prices.std(ddof=1)
        target = terminal_monitored_up_and_out_price(model, product.strike, product.barrier, maturity) if m == 1 else None
        # In/out is an algebraic validation identity, not another supported product.
        in_out_residual = np.max(np.abs(payoffs + vanilla*hit - vanilla))
        rows.append({"product": "up_and_out_call", "barrier": product.barrier,
                     "n_paths": terminal.size, "n_paths_per_replication": paths_per_run,
                     "n_replications": replications, "n_steps": m,
                     "price": result.price, "standard_error": result.standard_error,
                     "sample_variance": result.sample_variance, "estimator_variance": result.estimator_variance,
                     "ci_lower": result.confidence_interval[0], "ci_upper": result.confidence_interval[1],
                     "reference_steps": counts[-1], "is_reference": m == counts[-1],
                     "difference_to_reference": difference.price, "difference_standard_error": difference.standard_error,
                     "difference_ci_lower": difference.confidence_interval[0],
                     "difference_ci_upper": difference.confidence_interval[1],
                     "knockout_fraction": float(hit.mean()), "knockout_ci_lower": hit_ci[0], "knockout_ci_upper": hit_ci[1],
                     "missed_reference_hit_fraction": float(missed.mean()),
                     "empirical_replication_price_sd": float(empirical_sd),
                     "replication_mean_se": float(empirical_sd / np.sqrt(replications)),
                     "terminal_only_exact_price": target,
                     "terminal_only_error_in_se": (result.price-target)/result.standard_error if target is not None and result.standard_error else None,
                     "max_pathwise_in_out_residual": float(in_out_residual),
                     "pathwise_payoff_excess_over_vanilla": float(np.max(payoffs-vanilla)),
                     "pathwise_reference_excess": float(np.max(reference-payoffs))})
    return pd.DataFrame(rows)


def plot_results(table: pd.DataFrame, sweep: pd.DataFrame, destination: Path) -> None:
    """Prices, paired differences, hits and barrier-level sensitivity."""
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 8), constrained_layout=True)
        price_ax, diff_ax, hit_ax, level_ax = axes.flat
        price_ax.errorbar(table.n_steps, table.price, yerr=table.ci_upper-table.price,
                         fmt="o-", capsize=3, label="Price ± 95% normal CI")
        price_ax.axhline(table.price.iloc[-1], color="gray", ls="--", label="Finest discrete estimate")
        price_ax.set(xscale="log", xlabel="Future monitoring dates m (log scale)",
                     ylabel="Price (currency units)", title="Up-and-out call: monitoring frequency")
        price_ax.legend(frameon=False, fontsize=8)
        coarse = table.loc[~table.is_reference]
        diff_ax.errorbar(coarse.n_steps, coarse.difference_to_reference,
                        yerr=coarse.difference_ci_upper-coarse.difference_to_reference, fmt="o-", capsize=3)
        diff_ax.axhline(0, color="gray", ls="--")
        diff_ax.set(xscale="log", xlabel="Future monitoring dates m (log scale)",
                    ylabel="Paired price difference (currency units)", title="Difference from finest discrete grid ± 95% CI")
        hit_ax.plot(table.n_steps, 100*table.knockout_fraction, "o-", label="Observed knockout")
        hit_ax.plot(table.n_steps, 100*table.missed_reference_hit_fraction, "s--", label="Hits missed relative to finest grid")
        hit_ax.set(xscale="log", xlabel="Future monitoring dates m (log scale)",
                   ylabel="Fraction of paths (%)", title="Discrete hit diagnostics (includes S0)")
        hit_ax.legend(frameon=False, fontsize=8)
        level_ax.errorbar(sweep.barrier, sweep.price, yerr=sweep.ci_upper-sweep.price, fmt="o-", capsize=3)
        level_ax.set(xlabel="Barrier B (currency units)", ylabel="Price (currency units)",
                     title=f"Barrier sensitivity: {int(table.n_steps.iloc[-1])} dates ± 95% CI")
        for axis in axes.flat:
            axis.grid(alpha=.2)
        fig.suptitle(f"Discrete zero-rebate barrier | S₀=K=100, B=120, T=1, r=5%, σ=20%\n"
                     f"{int(table.n_paths.iloc[0]):,} IID paths per contract; shared paths across grids and barriers", fontsize=11)
        fig.savefig(destination / "barrier_monitoring_convergence.png", dpi=170)
        plt.close(fig)


def run(output_dir: Path, n_paths: int, replications: int, counts: Sequence[int], seed: int) -> None:
    model, product, maturity = BlackScholesModel(100,.05,.2), UpAndOutCall(100,120), 1.
    counts = tuple(counts)
    start = perf_counter()
    terminal, maxima, raw = collect_extrema(model, maturity, product, counts,
                                            n_paths=n_paths, replications=replications, seed=seed)
    collection_seconds = perf_counter()-start
    table = summarize_extrema(terminal, maxima, model, maturity, product, counts)
    assert np.all(np.diff(maxima, axis=2) >= 0), "nested maxima cannot decrease"
    assert np.all(np.diff(table.price) <= 0), "nested barrier prices cannot increase"
    assert (table.max_pathwise_in_out_residual == 0).all()
    sweep_rows = []
    for barrier in (100,105,110,120,130,150):
        quote = price_from_payoff_samples(UpAndOutCall(100,barrier).payoff_from_extrema(
            terminal.reshape(-1), maxima[:,:,-1].reshape(-1)), rate=model.rate, maturity=maturity)
        sweep_rows.append({"barrier": barrier, "n_paths": terminal.size, "n_steps": counts[-1],
                           "price": quote.price, "standard_error": quote.standard_error,
                           "ci_lower": quote.confidence_interval[0], "ci_upper": quote.confidence_interval[1]})
    sweep = pd.DataFrame(sweep_rows)
    assert np.all(np.diff(sweep.price) >= 0)
    engine = price_monte_carlo(model, product, maturity, n_paths, n_steps=96,
                              rng=np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(replications,))))
    quote_table = pd.DataFrame([{"product":"up_and_out_call", "barrier":120,
        "price":engine.price, "standard_error":engine.standard_error,
        "n_paths":engine.n_paths, "n_steps":engine.n_steps, "runtime_seconds":engine.runtime_seconds,
        "ci_lower":engine.confidence_interval[0], "ci_upper":engine.confidence_interval[1],
        "spawn_key":json.dumps([replications])}])
    benchmark = None
    if 1 in counts:
        target=terminal_monitored_up_and_out_price(model,100,120,1)
        benchmark=asdict(summarize_replications(raw.loc[raw.n_steps==1,"price"].to_numpy(),target))
        benchmark.update({"target":target,"scope":"Independent N-path runs, inception/expiry monitoring only"})
    for folder in ("raw","tables","figures"):
        (output_dir/folder).mkdir(parents=True,exist_ok=True)
    raw.to_csv(output_dir/"raw/barrier_replications.csv",index=False)
    table.to_csv(output_dir/"tables/barrier_monitoring.csv",index=False)
    sweep.to_csv(output_dir/"tables/barrier_levels.csv",index=False)
    quote_table.to_csv(output_dir/"tables/barrier_engine_quotes.csv",index=False)
    sample_file=output_dir/"raw/barrier_extrema.npz"
    np.savez_compressed(sample_file,terminal=terminal,maxima=maxima,monitoring_counts=np.asarray(counts))
    metadata={"milestone":8,"model":asdict(model),"product":asdict(product),"maturity_years":maturity,
        "root_seed":seed,"bit_generator":"PCG64","n_paths_per_replication":n_paths,"n_replications":replications,
        "monitoring_counts":list(counts),"total_paths_per_contract":terminal.size,
        "contract":"Zero rebate, equality knocks out; monitor S0 and j*T/m, j=1,...,m; payoff paid at T.",
        "rng_design":"SeedSequence(seed).spawn(R)[run]; one fine grid per independent run. All grids/levels share paths. Separate 96-date engine quote uses spawn_key=(R,).",
        "draw_call":"Generator.standard_normal((n_paths_per_replication, max(monitoring_counts)), dtype=float64)",
        "reference_scope":"Finest discrete grid is NOT a continuous-barrier benchmark. Missed hit fraction is relative to that finite grid.",
        "collection_seconds":collection_seconds,"timing_scope":"Collection excludes pooling/output/plots. Raw simulation is shared across grids; do not sum repeated times. Engine quote measures a separate full call, excluding caller RNG construction.",
        "terminal_bytes":terminal.nbytes,"maxima_bytes":maxima.nbytes,
        "fine_path_output_bytes_per_replication":n_paths*(counts[-1]+1)*8,
        "fine_normal_bytes_per_replication":n_paths*counts[-1]*8,
        "memory_scope":"Explicit arrays, not measured peak RAM. Library is not batched; study releases paths between runs.",
        "terminal_only_replication_benchmark":benchmark,
        "vanilla_analytical_upper_bound":black_scholes_price(model,100,1),
        "raw_samples_sha256":hashlib.sha256(sample_file.read_bytes()).hexdigest(),
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__},
        "limitations":"One baseline. Normal price CIs cover sampling error only. No continuous correction, numerical Greeks, variance reduction or general batching."}
    (output_dir/"raw/barrier_pricing.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    plot_results(table,sweep,output_dir/"figures")
    print(table[["n_steps","price","standard_error","knockout_fraction","difference_to_reference","difference_standard_error"]].to_string(index=False,float_format=lambda x:f"{x:.8g}"))
    print(sweep.to_string(index=False,float_format=lambda x:f"{x:.8g}"))
    print(quote_table.to_string(index=False))
    print("Terminal-only replication benchmark:",benchmark)
    print(f"Collected {len(raw)} estimates in {collection_seconds:.3f}s; saved to {output_dir.resolve()}")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("results"))
    parser.add_argument("--n-paths",type=int,default=5000)
    parser.add_argument("--replications",type=int,default=30)
    parser.add_argument("--monitoring-counts",type=int,nargs="+",default=DEFAULT_MONITORING)
    parser.add_argument("--seed",type=int,default=20260928)
    args=parser.parse_args()
    run(args.output_dir,args.n_paths,args.replications,args.monitoring_counts,args.seed)


if __name__=="__main__":
    main()
