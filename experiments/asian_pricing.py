"""Milestone 7: discrete arithmetic Asians and paired monitoring-frequency study.

Run: python -m experiments.asian_pricing
The finest grid is a discrete reference, not an exact continuous-Asian price.
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

from mc_pricer import (
    ArithmeticAsianOption, BlackScholesModel, arithmetic_average_moments,
    black_scholes_price, price_from_payoff_samples, price_monte_carlo, simulate_paths,
)

DEFAULT_MONITORING = (1, 4, 12, 24, 48, 96, 192, 384)


def collect_averages(
    model: BlackScholesModel, maturity: float, strike: float,
    monitoring_counts: Sequence[int], *, n_paths: int, n_replications: int, seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return averages (R,N,k) and prices per replication/product/grid.

    The maximum observation count is the fine grid; all other counts must
    divide it. Each replication has a distinct SeedSequence child. Grids and
    products share paths within a replication; distinct replications are independent.
    Only one full path matrix is retained at a time; all small average vectors
    are saved for pooled estimation and exact replay of reported statistics.
    """
    counts = tuple(monitoring_counts)
    if len(counts) < 2:
        raise ValueError("at least two monitoring counts are required")
    for name, values in (("monitoring_counts", counts), ("run counts", (n_paths, n_replications, seed))):
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
            raise TypeError(f"{name} must contain integers")
    if any(n < 1 for n in counts) or any(a >= b for a, b in zip(counts, counts[1:])):
        raise ValueError("monitoring counts must be positive and strictly increasing")
    if any(counts[-1] % n for n in counts):
        raise ValueError("each monitoring count must divide the finest grid")
    if n_paths < 2 or n_replications < 2 or seed < 0:
        raise ValueError("require n_paths >= 2, n_replications >= 2, seed >= 0")
    products = [ArithmeticAsianOption(strike, kind) for kind in ("call", "put")]
    averages = np.empty((n_replications, n_paths, len(counts)), dtype=np.float64)
    rows = []
    streams = np.random.SeedSequence(int(seed)).spawn(int(n_replications))
    for replication, stream in enumerate(streams):
        rng = np.random.default_rng(stream)
        start = perf_counter()
        paths = simulate_paths(model, maturity, n_paths, counts[-1], rng=rng)
        simulation_seconds = perf_counter() - start
        for index, observations in enumerate(counts):
            monitored_paths = paths[:, ::counts[-1] // observations]
            start = perf_counter()
            average = products[0].average(monitored_paths)
            averaging_seconds = perf_counter() - start
            averages[replication, :, index] = average
            for product in products:
                start = perf_counter()
                result = price_from_payoff_samples(product.payoff_from_average(average),
                                                   rate=model.rate, maturity=maturity)
                evaluation_seconds = averaging_seconds + perf_counter() - start
                rows.append({"replication": replication, "spawn_key": json.dumps(list(stream.spawn_key)),
                             "product": product.option_type, "n_paths": n_paths, "n_observations": observations,
                             "reference_observations": counts[-1], "price": result.price,
                             "standard_error": result.standard_error, "sample_variance": result.sample_variance,
                             "ci_lower": result.confidence_interval[0], "ci_upper": result.confidence_interval[1],
                             "shared_fine_simulation_seconds": simulation_seconds,
                             "evaluation_seconds": evaluation_seconds})
        # Views must be released too, otherwise they keep the full grid alive.
        del monitored_paths, paths
    return averages, pd.DataFrame(rows)


def summarize_averages(
    averages: np.ndarray, model: BlackScholesModel, maturity: float,
    strike: float, monitoring_counts: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled IID estimates, paired grid differences and exact moment/parity checks.

    Input is the (R,N,k) output of collect_averages. Pooled N is R*N; standard
    errors use the independent path averages, not the k monitoring grids.
    The final column is the discrete reference. No true Asian-price bias or
    RMSE is reported because a general exact arithmetic-Asian target is absent.
    """
    replications, paths_per_run, columns = averages.shape
    if columns != len(monitoring_counts):
        raise ValueError("average columns must match monitoring_counts")
    total = replications * paths_per_run
    discount = float(np.exp(-model.rate * maturity))
    reference = averages[:, :, -1].reshape(-1)
    rows, validation = [], []
    for index, observations in enumerate(monitoring_counts):
        average = averages[:, :, index].reshape(-1)
        moments = arithmetic_average_moments(model, maturity, observations)
        mean_se = float(np.sqrt(moments.variance / total))
        call_payoffs = ArithmeticAsianOption(strike).payoff_from_average(average)
        put_payoffs = ArithmeticAsianOption(strike, "put").payoff_from_average(average)
        times = maturity * np.arange(1, observations + 1, dtype=float) / observations
        prices = {}
        for kind, sign, payoffs in (("call", 1, call_payoffs), ("put", -1, put_payoffs)):
            start = perf_counter()
            result = price_from_payoff_samples(payoffs, rate=model.rate, maturity=maturity)
            reference_payoffs = ArithmeticAsianOption(strike, kind).payoff_from_average(reference)
            difference = price_from_payoff_samples(payoffs - reference_payoffs, rate=model.rate, maturity=maturity)
            pooled_statistics_seconds = perf_counter() - start
            # Equal-sized independent runs: their mean equals the pooled price.
            run_prices = discount * payoffs.reshape(replications, paths_per_run).mean(axis=1)
            empirical_sd = float(run_prices.std(ddof=1))
            lower = discount * max(sign * (moments.mean - strike), 0)
            upper = sum(np.exp(-model.rate * (maturity - t)) * black_scholes_price(model, strike, t, kind)
                        for t in times) / observations
            prices[kind] = result.price
            rows.append({"product": f"arithmetic_asian_{kind}", "option_type": kind,
                         "n_paths": total, "n_paths_per_replication": paths_per_run,
                         "n_replications": replications, "n_steps": observations,
                         "price": result.price, "standard_error": result.standard_error,
                         "sample_variance": result.sample_variance, "estimator_variance": result.estimator_variance,
                         "ci_lower": result.confidence_interval[0], "ci_upper": result.confidence_interval[1],
                         "jensen_lower_bound": lower, "convexity_upper_bound": float(upper),
                         "reference_steps": monitoring_counts[-1], "is_reference": observations == monitoring_counts[-1],
                         "difference_to_reference": difference.price, "difference_standard_error": difference.standard_error,
                         "difference_ci_lower": difference.confidence_interval[0],
                         "difference_ci_upper": difference.confidence_interval[1],
                         "empirical_replication_price_sd": empirical_sd,
                         "replication_mean_se": empirical_sd / np.sqrt(replications),
                         "pooled_statistics_seconds": pooled_statistics_seconds})
        theoretical_parity = discount * (moments.mean - strike)
        observed_parity = prices["call"] - prices["put"]
        validation.append({"n_observations": observations, "n_paths": total,
                           "mean_average": float(average.mean()), "exact_mean_average": moments.mean,
                           "sample_average_variance": float(average.var(ddof=1)), "exact_average_variance": moments.variance,
                           "average_mean_theoretical_se": mean_se,
                           "average_mean_error_in_se": (average.mean() - moments.mean) / mean_se if mean_se else None,
                           "observed_call_minus_put": observed_parity, "exact_call_minus_put": theoretical_parity,
                           "parity_theoretical_se": discount * mean_se,
                           "parity_error_in_se": (observed_parity - theoretical_parity) / (discount * mean_se) if mean_se else None,
                           "sample_parity_residual": observed_parity - discount * (average.mean() - strike),
                           "max_pathwise_parity_residual": float(np.max(np.abs(call_payoffs - put_payoffs - (average - strike))))})
    return pd.DataFrame(rows), pd.DataFrame(validation)


def plot_study(table: pd.DataFrame, destination: Path) -> None:
    """Prices and paired differences; no extrapolation to continuous monitoring."""
    colors = {"call": "#136F8A", "put": "#CB6632"}
    reference_steps = int(table["reference_steps"].iloc[0])
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
        for axis, kind in zip(axes, ("call", "put")):
            rows = table.loc[table["option_type"] == kind].sort_values("n_steps")
            axis.errorbar(rows["n_steps"], rows["price"], yerr=rows["ci_upper"] - rows["price"],
                          fmt="o-", capsize=4, color=colors[kind], label="Pooled price ± 95% normal CI")
            axis.axhline(rows["price"].iloc[-1], ls="--", color="#454B53", label=f"{reference_steps}-observation estimate")
            axis.set(xscale="log", xlabel="Number of observations m (log scale)",
                     ylabel="Option price (currency units)", title=f"Arithmetic Asian {kind}")
            axis.grid(alpha=.18)
            axis.legend(frameon=False, fontsize=8)
        total = int(table.n_paths.iloc[0])
        fig.suptitle(f"Discrete monitoring-frequency study | {total:,} independent paths per contract\n"
                     "S₀ excluded; equal weights; estimates across grids share paths", fontsize=11)
        fig.savefig(destination / "asian_monitoring_convergence.png", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), constrained_layout=True)
        for col, kind in enumerate(("call", "put")):
            rows = table.loc[(table["option_type"] == kind) & ~table["is_reference"]].sort_values("n_steps")
            for row_index, subset in enumerate((rows, rows.tail(4))):
                axis = axes[row_index, col]
                axis.errorbar(subset["n_steps"], subset["difference_to_reference"],
                              yerr=subset["difference_ci_upper"] - subset["difference_to_reference"],
                              fmt="o-", capsize=4, color=colors[kind], label="Paired difference ± 95% CI")
                axis.axhline(0, color="#454B53", ls="--", lw=1)
                axis.set(xscale="log", xlabel="Number of observations m (log scale)",
                         ylabel="Price difference (currency units)",
                         title=f"{kind.capitalize()}: {'full range' if row_index == 0 else 'finer-grid detail'}")
                axis.grid(alpha=.18)
                if row_index == 0:
                    axis.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Price differences from the discrete {reference_steps}-observation contract\n"
                     "Uncertainty uses pathwise payoff differences, including shared-path covariance", fontsize=11)
        fig.savefig(destination / "asian_paired_monitoring_differences.png", dpi=170)
        plt.close(fig)


def run(output_dir: Path, n_paths: int, n_replications: int, counts: Sequence[int], seed: int) -> None:
    """Run the baseline, save all averages/results and an end-to-end engine quote."""
    model, maturity, strike = BlackScholesModel(100.0, .05, .20), 1.0, 100.0
    start = perf_counter()
    averages, raw = collect_averages(model, maturity, strike, counts, n_paths=n_paths,
                                     n_replications=n_replications, seed=seed)
    collection_seconds = perf_counter() - start
    table, validation = summarize_averages(averages, model, maturity, strike, counts)
    # A separate full-pricer quote records genuine end-to-end runtime. It is
    # excluded from the monitoring study and uses a distinct spawn key.
    quote_rows = []
    for kind in ("call", "put"):
        stream = np.random.SeedSequence(seed, spawn_key=(int(n_replications),))
        result = price_monte_carlo(model, ArithmeticAsianOption(strike, kind), maturity,
                                   n_paths, n_steps=12, rng=np.random.default_rng(stream))
        quote_rows.append({"product": f"arithmetic_asian_{kind}", "price": result.price,
                           "standard_error": result.standard_error, "n_paths": result.n_paths,
                           "n_steps": result.n_steps, "ci_lower": result.confidence_interval[0],
                           "ci_upper": result.confidence_interval[1], "runtime_seconds": result.runtime_seconds,
                           "spawn_key": json.dumps(list(stream.spawn_key))})
    quotes = pd.DataFrame(quote_rows)
    for folder in ("raw", "tables", "figures"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "raw" / "asian_monitoring_replications.csv", index=False)
    table.to_csv(output_dir / "tables" / "asian_monitoring.csv", index=False)
    validation.to_csv(output_dir / "tables" / "asian_moment_parity_validation.csv", index=False)
    quotes.to_csv(output_dir / "tables" / "asian_engine_quotes.csv", index=False)
    sample_path = output_dir / "raw" / "asian_average_samples.npz"
    np.savez_compressed(sample_path, averages=averages, monitoring_counts=np.asarray(counts))
    metadata = {
        "milestone": 7, "model": asdict(model), "maturity_years": maturity, "strike": strike,
        "root_seed": seed, "bit_generator": type(np.random.default_rng(seed).bit_generator).__name__,
        "n_paths_per_replication": n_paths, "n_replications": n_replications,
        "total_paths_per_contract": n_paths * n_replications, "monitoring_counts": list(counts),
        "monitoring_convention": "Equal-weight observations j*T/m, j=1,...,m; exclude S0, include T.",
        "rng_design": "SeedSequence(seed).spawn(R)[replication]; one fine path matrix per independent run. "
                      "All grids/products share that run's paths. Separate engine quote uses spawn_key=(R,).",
        "draw_call": "Generator.standard_normal((n_paths_per_replication, max(monitoring_counts)), dtype=float64)",
        "pooled_estimation": "Pool R*N IID path averages at each fixed m. Prices across m are dependent. "
                             "Pair payoff differences against the finest discrete grid; do not treat separate price CIs as independent.",
        "reference_scope": "Finest discrete contract, not an exact continuous-monitoring benchmark; no exact Asian-price bias/RMSE claim.",
        "averages_shape": list(averages.shape), "saved_average_bytes": averages.nbytes,
        "fine_path_output_bytes_per_replication": n_paths * (counts[-1] + 1) * 8,
        "fine_normal_bytes_per_replication": n_paths * counts[-1] * 8,
        "memory_scope": "Explicit arrays, not measured peak memory. Full grid released per replication; only averages retained.",
        "collection_seconds": collection_seconds,
        "runtime_scope": "Raw rows separately record shared fine simulation and per-contract evaluation. "
                         "Shared work must not be summed across grids/products. Pooled-table timing covers statistics only. "
                         "asian_engine_quotes.csv records separate complete pricer calls, excluding caller RNG construction.",
        "raw_samples_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "limitations": "One parameter baseline, finite replication count and finite reference monitoring. "
                       "Normal CIs quantify sampling error, not model error or continuous-monitoring bias. "
                       "No antithetic/control variates or numerical Greeks implemented at this milestone.",
    }
    (output_dir / "raw" / "asian_pricing.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_study(table, output_dir / "figures")
    print(table[["option_type", "n_steps", "n_paths", "price", "standard_error", "difference_to_reference",
                 "difference_standard_error"]].to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print(validation[["n_observations", "mean_average", "exact_mean_average", "average_mean_error_in_se",
                      "sample_parity_residual"]].to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print(quotes.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print(f"Collected {len(raw)} contract estimates in {collection_seconds:.3f}s; saved to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-paths", type=int, default=5000)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--monitoring-counts", type=int, nargs="+", default=DEFAULT_MONITORING)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.n_paths < 2 or args.replications < 2 or args.seed < 0:
        parser.error("require n_paths >= 2, replications >= 2, seed >= 0")
    run(args.output_dir, args.n_paths, args.replications, args.monitoring_counts, args.seed)


if __name__ == "__main__":
    main()
