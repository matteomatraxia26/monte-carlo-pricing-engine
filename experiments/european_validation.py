"""Milestone 4: European validation at several Monte Carlo sample sizes.

Run: python -m experiments.european_validation
The same seed deliberately couples sample sizes and contracts. Rows are not
independent replications; this script does not estimate coverage or a slope.
"""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from mc_pricer import (
    BlackScholesModel, EuropeanOption, black_scholes_price,
    make_rng, price_monte_carlo, simulate_terminal,
)


def collect_validation(
    model: BlackScholesModel,
    strike: float,
    maturity: float,
    path_counts: Sequence[int],
    *,
    seed: int,
) -> pd.DataFrame:
    """Collect call/put benchmark rows without writing files or changing globals.

    Counts must be strictly increasing integers >= 2. Restarting the same RNG
    seed gives nested normal prefixes in the recorded NumPy/PCG64 environment.
    Timings include the full pricer call; each row is a single timing observation.
    Relative error is a fraction, or None when the analytical price is zero.
    Standardized error is None when observed SE is zero.
    """
    counts = tuple(path_counts)
    if not counts:
        raise ValueError("path_counts must not be empty")
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in counts):
        raise TypeError("path_counts must contain integers")
    if any(n < 2 for n in counts):
        raise ValueError("every path count must be at least 2")
    if any(left >= right for left, right in zip(counts, counts[1:])):
        raise ValueError("path_counts must be strictly increasing")

    rows = []
    for option_type in ("call", "put"):
        product = EuropeanOption(strike, option_type)
        analytical = black_scholes_price(model, strike, maturity, option_type)
        for count in counts:
            result = price_monte_carlo(model, product, maturity, int(count), rng=make_rng(seed))
            error = result.price - analytical
            rows.append({
                "product": option_type, "n_paths": result.n_paths,
                "n_independent_samples": result.n_independent_samples,
                "mc_price": result.price, "bs_price": analytical,
                "signed_error": error, "absolute_error": abs(error),
                "relative_error": abs(error) / abs(analytical) if analytical != 0 else None,
                "standard_error": result.standard_error,
                "sample_variance": result.sample_variance,
                "estimator_variance": result.estimator_variance,
                "confidence_level": result.confidence_level,
                "ci_lower": result.confidence_interval[0],
                "ci_upper": result.confidence_interval[1],
                "ci_contains_bs": result.confidence_interval[0] <= analytical <= result.confidence_interval[1],
                "signed_error_in_se": error / result.standard_error if result.standard_error > 0 else None,
                "runtime_seconds": result.runtime_seconds,
                "seed": seed,
            })
    return pd.DataFrame(rows)


def _plot_validation(table: pd.DataFrame, destination: Path, seed: int) -> None:
    """Plot prices with CIs, then absolute and standardized errors; no fitting."""
    colors = {"call": "#136F8A", "put": "#CB6632"}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
        for axis, option_type in zip(axes, ("call", "put")):
            rows = table.loc[table["product"] == option_type]
            axis.errorbar(rows["n_paths"], rows["mc_price"],
                          yerr=rows["ci_upper"].to_numpy() - rows["mc_price"].to_numpy(),
                          fmt="o-", capsize=5, color=colors[option_type], lw=1.5,
                          label="Monte Carlo ± 95% normal CI")
            axis.axhline(rows["bs_price"].iloc[0], color="#40464D", ls="--", label="Black–Scholes")
            axis.set(xscale="log", xlabel="Number of paths N (log scale)",
                     ylabel="Option price (currency units)", title=f"European {option_type}")
            axis.grid(alpha=0.18)
            axis.legend(frameon=False, fontsize=9)
        fig.suptitle(f"European price validation | nested samples, seed {seed}\n"
                     "S₀ = K = 100, r = 5%, σ = 20%, T = 1 year", fontsize=12)
        fig.savefig(destination / "european_validation_prices.png", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
        for option_type in ("call", "put"):
            rows = table.loc[table["product"] == option_type]
            axes[0].loglog(rows["n_paths"], rows["absolute_error"], "o-",
                           color=colors[option_type], label=option_type.capitalize())
            axes[1].semilogx(rows["n_paths"], rows["signed_error_in_se"], "o-",
                             color=colors[option_type], label=option_type.capitalize())
        axes[0].set(title="Realized absolute pricing error", xlabel="Number of paths N (log scale)",
                    ylabel="Absolute error (currency units; log scale)")
        axes[1].axhline(0, color="#40464D", lw=1)
        for level in (-1.959963984540054, 1.959963984540054):
            axes[1].axhline(level, color="#8A929A", ls="--", lw=1,
                           label="±1.96 reference levels" if level > 0 else None)
        axes[1].set(title="Error relative to estimated uncertainty",
                    xlabel="Number of paths N (log scale)", ylabel="(MC − BS) / estimated SE")
        for axis in axes:
            axis.grid(alpha=0.18)
            axis.legend(frameon=False, fontsize=9)
        fig.suptitle("One nested sample sequence; points are not independent replications", fontsize=12)
        fig.savefig(destination / "european_validation_errors.png", dpi=170)
        plt.close(fig)


def run(output_dir: Path, path_counts: Sequence[int], seed: int) -> None:
    """Run the baseline and save the complete table, replay sample and figures."""
    model = BlackScholesModel(100.0, 0.05, 0.20)
    strike, maturity = 100.0, 1.0
    table = collect_validation(model, strike, maturity, path_counts, seed=seed)
    counts = [int(n) for n in path_counts]
    # Save only the largest terminal vector; smaller samples are its prefixes.
    # Replay and serialization are outside reported pricer runtimes.
    terminal = simulate_terminal(model, maturity, counts[-1], rng=make_rng(seed))
    for directory in ("tables", "raw", "figures"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "tables" / "european_validation.csv", index=False)
    np.savez_compressed(output_dir / "raw" / "european_validation_samples.npz", terminal=terminal)
    # pandas JSON conversion maps undefined diagnostics to JSON null, not NaN.
    raw_rows = json.loads(table.to_json(orient="records", double_precision=15))
    metadata = {
        "milestone": 4, "model": asdict(model), "strike": strike,
        "maturity_years": maturity, "path_counts": counts, "seed": seed,
        "bit_generator": type(make_rng(seed).bit_generator).__name__,
        "draw_call_per_row": "Generator.standard_normal(n_paths, dtype=float64)",
        "sampling_design": "Same seed restarted for every count and both products. "
                           "Nested normal prefixes in the recorded environment. Rows are correlated.",
        "runtime_scope": "One full pricing call per row, including validation, simulation, "
                         "payoff, discount and statistics. Excludes caller RNG creation, "
                         "analytical target, replay, plots and serialization. No warmup exclusion.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__},
        "raw_terminal_shape": list(terminal.shape), "raw_terminal_dtype": str(terminal.dtype),
        "raw_terminal_bytes": terminal.nbytes,
        "raw_terminal_sha256": hashlib.sha256(terminal.tobytes()).hexdigest(),
        "raw_sample_usage": "terminal[:N] reproduces each count. Recompute call/put payoffs "
                            "with the recorded strike and discount factor.",
        "relative_error_units": "Fraction, not percent. Multiply by 100 for percentage error.",
        "table": raw_rows,
        "limitations": "No empirical coverage rate, bias, RMSE, convergence slope or speed ranking "
                       "is inferred from these correlated single-run estimates.",
    }
    (output_dir / "raw" / "european_validation.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _plot_validation(table, output_dir / "figures", seed)
    print(table[["product", "n_paths", "mc_price", "bs_price", "absolute_error", "relative_error",
                 "standard_error", "ci_lower", "ci_upper", "signed_error_in_se", "runtime_seconds"]]
          .to_string(index=False, float_format=lambda value: f"{value:.9g}"))
    print("Rows share random draws; interval containment here is not an estimated coverage rate.")
    print(f"Largest terminal vector: {terminal.nbytes:,} bytes, excluding temporaries.")
    print(f"Saved results to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--path-counts", type=int, nargs="+",
                        default=[1_000, 10_000, 100_000, 1_000_000])
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if any(n < 2 for n in args.path_counts) or any(a >= b for a, b in zip(args.path_counts, args.path_counts[1:])):
        parser.error("--path-counts must be strictly increasing integers >= 2")
    run(args.output_dir, args.path_counts, args.seed)


if __name__ == "__main__":
    main()
