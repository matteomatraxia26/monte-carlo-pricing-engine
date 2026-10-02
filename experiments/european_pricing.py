"""Milestone 3: one baseline European call/put Monte Carlo experiment.

Run: python -m experiments.european_pricing --n-paths 200000 --seed 20260924
The multi-N validation and convergence studies are Milestones 4 and 5.
"""

import argparse
from dataclasses import asdict
import json
from math import exp
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


def run(output_dir: Path, n_paths: int, seed: int) -> None:
    """Save prices, uncertainty, raw samples, metadata and a CI figure."""
    model = BlackScholesModel(spot=100.0, rate=0.05, volatility=0.20)
    strike, maturity = 100.0, 1.0
    rows = []
    raw_results = {}
    for option_type in ("call", "put"):
        product = EuropeanOption(strike, option_type)
        # Restart deliberately so both contracts use identical terminal draws.
        # These are paired contract valuations, not independent replications.
        result = price_monte_carlo(model, product, maturity, n_paths, rng=make_rng(seed))
        analytical = black_scholes_price(model, strike, maturity, option_type)
        absolute_error = abs(result.price - analytical)
        rows.append({
            "product": option_type, "n_paths": result.n_paths,
            "n_independent_samples": result.n_independent_samples,
            "mc_price": result.price, "bs_price": analytical,
            "absolute_error": absolute_error, "relative_error": absolute_error / analytical,
            "standard_error": result.standard_error, "sample_variance": result.sample_variance,
            "estimator_variance": result.estimator_variance,
            "confidence_level": result.confidence_level,
            "ci_lower": result.confidence_interval[0], "ci_upper": result.confidence_interval[1],
            "ci_contains_bs": result.confidence_interval[0] <= analytical <= result.confidence_interval[1],
            "signed_error_in_se": (result.price - analytical) / result.standard_error
                                  if result.standard_error > 0 else None,
            "runtime_seconds": result.runtime_seconds,
        })
        raw_results[option_type] = asdict(result)
    table = pd.DataFrame(rows)

    # Replay for raw-output storage, outside both reported pricing runtimes.
    terminal = simulate_terminal(model, maturity, n_paths, rng=make_rng(seed))
    discount = exp(-model.rate * maturity)
    discounted_call = discount * EuropeanOption(strike, "call").payoff(terminal)
    discounted_put = discount * EuropeanOption(strike, "put").payoff(terminal)
    sample_difference = rows[0]["mc_price"] - rows[1]["mc_price"]
    sample_parity_target = discount * (float(terminal.mean()) - strike)
    theoretical_parity_target = model.spot - discount * strike

    for directory in ("tables", "raw", "figures"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "tables" / "european_pricing.csv", index=False)
    np.savez_compressed(output_dir / "raw" / "european_pricing_samples.npz",
                        terminal=terminal, discounted_call=discounted_call,
                        discounted_put=discounted_put)
    metadata = {
        "milestone": 3, "model": asdict(model), "strike": strike,
        "maturity_years": maturity, "n_paths": n_paths, "seed": seed,
        "bit_generator": type(make_rng(seed).bit_generator).__name__,
        "draw_call": "Generator.standard_normal(n_paths, dtype=float64)",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__},
        "pricing_results": raw_results,
        "table": rows,
        "runtime_scope": "Each price includes terminal simulation, payoff, discount and statistics; "
                         "excludes caller RNG construction, raw replay, plotting and serialization.",
        "sample_parity_residual": sample_difference - sample_parity_target,
        "theoretical_parity_error": sample_difference - theoretical_parity_target,
        "scope": "One fixed-seed baseline for each contract. Both use the same shocks. "
                 "No empirical coverage rate, convergence slope or estimator ranking is inferred.",
    }
    (output_dir / "raw" / "european_pricing.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axis = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
        estimates = table["mc_price"].to_numpy() - table["bs_price"].to_numpy()
        half_widths = table["ci_upper"].to_numpy() - table["mc_price"].to_numpy()
        axis.errorbar(estimates, [1, 0], xerr=half_widths, fmt="o", capsize=6,
                      color="#136F8A", lw=2, markersize=7,
                      label="MC error ± 95% normal interval half-width")
        axis.axvline(0, color="#CB6632", lw=1.5, ls="--", label="Analytical target")
        axis.set(yticks=[0, 1], yticklabels=["European put", "European call"],
                 ylim=(-0.65, 1.65), xlabel="MC price − Black–Scholes price (currency units)")
        axis.grid(axis="x", alpha=0.2)
        axis.legend(frameon=False, loc="upper left", fontsize=9)
        axis.set_title(f"European pricing: sampling error and uncertainty\n"
                       f"N = {n_paths:,}, seed = {seed}; S₀ = K = 100, r = 5%, σ = 20%, T = 1",
                       fontsize=11)
        fig.savefig(output_dir / "figures" / "european_pricing.png", dpi=170)
        plt.close(fig)

    print(table[["product", "mc_price", "bs_price", "absolute_error", "standard_error",
                 "ci_lower", "ci_upper", "signed_error_in_se", "runtime_seconds"]].to_string(
                     index=False, float_format=lambda value: f"{value:.9g}"))
    print(f"Sample put–call parity residual: {metadata['sample_parity_residual']:.3e}")
    print(f"Difference from theoretical put–call parity: {metadata['theoretical_parity_error']:.9g}")
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
