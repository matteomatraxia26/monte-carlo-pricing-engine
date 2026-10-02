"""Milestone 2: exact GBM terminal distribution and risk-neutral moments.

Run: python -m experiments.terminal_simulation --n-paths 200000 --seed 20260924
This validates the simulator; option payoffs and pricing arrive in Milestone 3.
"""

import argparse
from dataclasses import asdict
import json
from math import exp, expm1, sqrt
from pathlib import Path
import platform
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import lognorm, norm

from mc_pricer import BlackScholesModel, make_rng, standard_normal_draws, terminal_from_normals


def run(output_dir: Path, n_paths: int, seed: int) -> None:
    """Save raw draws/states, moment diagnostics, metadata and a density plot."""
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2 for sample variance")
    model = BlackScholesModel(spot=100.0, rate=0.05, volatility=0.20)
    maturity = 1.0
    rng = make_rng(seed)
    start = perf_counter()
    normals = standard_normal_draws(n_paths, rng=rng)
    terminal = terminal_from_normals(model, maturity, normals)
    runtime = perf_counter() - start

    log_returns = np.log(terminal / model.spot)
    log_mean = (model.rate - 0.5 * model.volatility**2) * maturity
    log_variance = model.volatility**2 * maturity
    terminal_mean = model.spot * exp(model.rate * maturity)
    terminal_variance = terminal_mean**2 * expm1(log_variance)
    # Exact central fourth moment of a lognormal, expressed through its kurtosis.
    q = log_variance
    fourth_central = terminal_variance**2 * (exp(4*q) + 2*exp(3*q) + 3*exp(2*q) - 3)
    variance_se = sqrt((fourth_central - (n_paths-3)/(n_paths-1)*terminal_variance**2) / n_paths)
    discount = exp(-model.rate * maturity)
    cases = [
        ("Terminal mean", "currency", float(terminal.mean()), terminal_mean,
         sqrt(terminal_variance / n_paths)),
        ("Terminal variance", "currency squared", float(terminal.var(ddof=1)), terminal_variance,
         variance_se),
        ("Log-return mean", "dimensionless", float(log_returns.mean()), log_mean,
         sqrt(log_variance / n_paths)),
        ("Log-return variance", "dimensionless squared", float(log_returns.var(ddof=1)), log_variance,
         log_variance * sqrt(2 / (n_paths-1))),
        ("Discounted terminal mean", "currency", float((discount * terminal).mean()), model.spot,
         discount * sqrt(terminal_variance / n_paths)),
    ]
    rows = [{"metric": name, "units": units, "estimate": estimate, "theoretical": target,
             "absolute_error": abs(estimate-target), "standard_error": se,
             "error_in_standard_errors": (estimate-target)/se,
             "n_paths": n_paths, "seed": seed} for name, units, estimate, target, se in cases]
    table = pd.DataFrame(rows)
    for directory in ("raw", "tables", "figures"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "tables" / "terminal_moments.csv", index=False)
    np.savez_compressed(output_dir / "raw" / "terminal_samples.npz",
                        normals=normals, terminal=terminal)
    metadata = {
        "milestone": 2, "model": asdict(model), "maturity_years": maturity,
        "n_paths": n_paths, "seed": seed,
        "bit_generator": type(rng.bit_generator).__name__,
        "draw_call": "Generator.standard_normal(n_paths, dtype=float64)",
        "array_shape": list(terminal.shape), "dtype": str(terminal.dtype),
        "terminal_array_bytes": terminal.nbytes,
        "draw_and_transform_runtime_seconds": runtime,
        "timing_excludes": "Moment analysis, serialization and plotting",
        "standard_error_basis": "Exact theoretical sampling variance of each sample moment; "
                                "variance estimates use ddof=1 and the fourth central moment.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__},
        "moments": rows,
        "scope": "Single fixed-seed simulator validation, not an estimator-comparison study.",
    }
    (output_dir / "raw" / "terminal_simulation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        axes[0].hist(log_returns, bins=90, density=True, color="#13738B", alpha=0.4,
                     label="Simulated samples")
        grid = np.linspace(log_returns.min(), log_returns.max(), 500)
        axes[0].plot(grid, norm.pdf(grid, loc=log_mean, scale=sqrt(log_variance)),
                     color="#CB6632", lw=2, label="Theoretical normal density")
        axes[0].set(title="Log returns", xlabel="Log(S_T / S_0)",
                    ylabel="Density (per unit log return)")
        axes[1].hist(terminal, bins=90, density=True, color="#13738B", alpha=0.4,
                     label="Simulated samples")
        grid = np.linspace(terminal.min(), terminal.max(), 500)
        axes[1].plot(grid, lognorm.pdf(grid, s=sqrt(log_variance),
                                      scale=model.spot * exp(log_mean)),
                     color="#CB6632", lw=2, label="Theoretical lognormal density")
        axes[1].set(title="Terminal stock prices", xlabel="S_T (currency units)",
                    ylabel="Density (per currency unit)")
        for axis in axes:
            axis.grid(alpha=0.18)
            axis.legend(frameon=False)
        fig.suptitle(f"Exact GBM terminal simulation | N = {n_paths:,}, seed = {seed}\n"
                     "S₀ = 100, r = 5%, σ = 20%, T = 1 year", fontsize=13)
        fig.savefig(output_dir / "figures" / "terminal_distribution.png", dpi=170)
        plt.close(fig)

    print(table[["metric", "estimate", "theoretical", "standard_error",
                 "error_in_standard_errors"]].to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print(f"Draw and transformation runtime: {runtime:.4f} s")
    print(f"Terminal array: {terminal.nbytes:,} bytes; additional arrays/temporaries excluded.")
    print(f"Saved results to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-paths", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.n_paths < 2:
        parser.error("--n-paths must be at least 2")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    run(args.output_dir, args.n_paths, args.seed)


if __name__ == "__main__":
    main()
