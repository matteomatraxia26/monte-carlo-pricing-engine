# Completed scope and development roadmap

All **17 milestones are complete**, research release **0.17.0**.
The tables below retain the development dependency map. Unimplemented optional
items remain explicit extensions, not missing core deliverables.

## Finished project

A small Python library that prices European, discretely monitored arithmetic
Asian and up-and-out options under Black–Scholes, estimates Greeks, and reports
statistically meaningful uncertainty. Reproducible experiments distinguish
sampling noise, estimator bias and monitoring error. An advanced extension adds
likelihood-ratio Delta and Heston pricing with calibration-parameter import.

The interview narrative is: derive the model, build and verify the estimator,
reduce its variance, assess sensitivities, measure numerical error, and explain
the method's limitations. A single carefully defended experiment is more useful
than many unvalidated model variants.

## Skills demonstrated at completion

Mathematics: no-arbitrage and martingale pricing; Itô calculus; lognormal
distributions; laws of large numbers and CLT; covariance-based variance reduction;
finite-difference truncation; differentiation under the integral; statistical
study design; stochastic-volatility discretization.

Python: src-layout packaging; dataclasses and type hints; explicit array shape
contracts; vectorization and broadcasting; reductions and cumulative sums;
boolean payoff masks; explicit RNG streams; independent tests; pandas tables;
matplotlib figures; memory accounting and profiling.

## Repository map

`A` = advanced, `O` = optional and not implemented. The map groups the main
modules; see the repository for all tests and historical experiment artifacts.

```text
monte-carlo-pricing-engine/
  README.md
  pyproject.toml
  requirements.txt
  src/mc_pricer/
    __init__.py
    _validation.py
    models/
      __init__.py
      black_scholes.py
      heston.py                         # A
    products/
      __init__.py
      european.py
      digital.py                        # M14 cash-or-nothing
      asian.py
      barrier.py
      lookback.py                       # O
    simulation/
      __init__.py
      random.py
      paths.py
      heston.py                         # M15
    pricing/
      __init__.py
      analytical.py
      digital.py                        # M14 exact digital price/Delta
      monte_carlo.py
      heston_monte_carlo.py              # M15 plain IID
      heston_analytical.py               # M15 Fourier reference
    variance_reduction/
      __init__.py
      antithetic.py
      control_variates.py
    greeks/
      __init__.py
      finite_difference.py
      pathwise.py
      likelihood_ratio.py               # A
    statistics/
      __init__.py
      confidence_intervals.py
      diagnostics.py
  tests/
    test_black_scholes.py
    test_payoffs.py
    test_simulation.py
    test_paths.py
    test_asian.py
    test_asian_experiment.py
    test_barrier.py
    test_barrier_experiment.py
    test_monte_carlo.py
    test_european_validation.py
    test_diagnostics.py
    test_convergence_experiment.py
    test_variance_reduction.py
    test_antithetic_experiment.py
    test_control_variates.py
    test_control_experiment.py
    test_greeks.py
    test_greeks_experiment.py
    test_pathwise.py
    test_pathwise_experiment.py
    test_estimator_robustness.py         # M13
    test_likelihood_ratio.py            # M14
    test_likelihood_ratio_experiment.py # M14
    test_heston.py                      # A
  experiments/
    __init__.py
    analytical_benchmark.py             # M1 deterministic benchmark
    terminal_simulation.py              # M2 stock-distribution validation
    path_simulation.py                  # M6 exact-grid and joint-law validation
    asian_pricing.py                    # M7 paired monitoring-frequency study
    barrier_pricing.py                  # M8 nested grids and barrier levels
    european_pricing.py                 # M3 baseline option-pricing experiment
    european_validation.py
    mc_convergence.py
    variance_reduction.py
    control_variates.py                 # M10 independent pilot and exact control
    greeks_comparison.py
    pathwise_delta.py                   # M12 cost and paired comparisons
    estimator_robustness.py             # M13 rare-event and multi-scenario study
    likelihood_ratio.py                 # M14 vanilla/digital Delta
    exotic_pricing.py
    heston_analysis.py                  # A
    final_experiments.py                # M16
    check_presentation.py               # M17
  notebooks/
    project_overview.ipynb              # final presentation only
  results/
    figures/
    tables/
    raw/
  docs/
    roadmap.md
    milestone_01.md
    milestone_02.md
    milestone_03.md
    milestone_04.md
    milestone_05.md
    milestone_06.md
    milestone_07.md
    milestone_08.md
    milestone_09.md
    milestone_10.md
    milestone_11.md
    milestone_12.md
    milestone_13.md
    milestone_14.md
    milestone_15.md
    milestone_16.md
    milestone_17.md
    mathematical_background.md
    methodology.md
    results.md
    interview_questions.md
    cv_targets.md
```

## Dependency map and acceptance gates

| Milestone | Class | Depends on | Deliverable and acceptance criterion |
|---|---|---|---|
| 1. Skeleton and analytical BS | Core | None | Prices and Delta/Gamma/Vega; identities, quadrature and derivative tests |
| 2. RNG and terminal GBM | Core | 1 | Implemented: explicit Generator and supplied draws; shapes, moments and reproducibility |
| 3. European MC | Core | 2 | Implemented: pure payoffs, structured result, SE and normal CI from discounted samples |
| 4. European validation | Core | 3 | Implemented: baseline call/put table at 1k–1m paths; standardized errors and replayable samples |
| 5. Convergence and CIs | Core | 4 | Implemented: log-log SE/RMSE fits, 200 independent replications per N, bias/variance and Wilson coverage intervals |
| 6. Vectorized paths | Core | 2, 5 | Implemented: exact grid paths, shape `(N,m+1)`, joint-law, stream and coupled-grid checks |
| 7. Asians | Core | 6, 3 | Implemented: arithmetic call/put, exact average moments, parity and paired frequency study |
| 8. Barriers | Core | 6, 3 | Implemented: zero-rebate up-and-out, equality/initial hits, endpoint benchmark and paired monitoring study |
| 9. Antithetics | Core | 3, 6 | Implemented: pair-mean uncertainty, 200-run equal-budget comparison and positive-covariance counterexample |
| 10. Control variates | Core | 7, 9 | Implemented: exact discrete geometric Asian expectation, independent beta pilot and 200-run cost-aware variance study |
| 11. FD Greeks with CRN | Core | 3, 5 | Implemented: Delta/Gamma/Vega, paired uncertainty, 100-run CRN/independent bump study and Delta convergence |
| 12. Pathwise Delta | Core | 11 | Implemented: call/put Delta, dominated-convergence derivation, exact variance, 200-run independent and paired comparisons |
| 13. Repeated estimator studies | Advanced | 9–12 | Implemented: 7,600 estimates, eight Greek/four Asian scenarios, bootstrap comparisons, rare-event and pilot diagnostics |
| 14. Likelihood ratio | Advanced | 12, 13 | Implemented: vanilla/digital call-put spot score, exact moments, 6,300 repeated estimates, rare-event diagnostics |
| 15. Heston | Advanced | 6–8, 11, 13 | Implemented: full truncation, calibration import, Fourier references, correlation and coupled step-size studies |
| 16. Final experiments | Core + selected advanced | 1–15 selected | Implemented: 800 new estimates, 12-figure source index, bootstrap/runtime comparisons, hash-verified rebuild |
| 17. Final report and interview pack | Core | 16 | Completed: concise README/report, executed notebook, 25 interview answers, timed walkthrough and evidence-backed CV bullets |

Dependencies are a map, not a request to implement everything now. Development
follows the numbered milestones, with an explanation and inspection at each gate.
Documentation and meaningful tests are maintained throughout, not deferred to 17.

## Core / advanced / optional

**Core:** Black–Scholes, European calls/puts, arithmetic Asian calls/puts, discrete
up-and-out calls, Monte Carlo uncertainty, convergence, antithetics, geometric
Asian controls, CRN Delta/Gamma/Vega, pathwise call Delta, tests and empirical
benchmarks. Small repeated-seed comparisons belong to the core evidence.

**Advanced:** larger repeated statistical studies, likelihood-ratio Delta,
Heston full truncation Euler and parameter import, batching, continuous barrier
bridge corrections. Complete the core before implementing these extensions.

**Optional:** lookbacks, Sobol/QMC, Numba, parallelism and alternative Heston
schemes. No optional feature is required for a strong first portfolio release.

## Recommended realistic final scope

The release contains the complete core plus likelihood-ratio spot Delta and
Heston pricing as focused advanced studies. Heston finite-difference sensitivities can reuse
the CRN design after validation. Heston calibration is outside this repository:
only the parameter handoff is included. Defer lookbacks, QMC and accelerators.

This supports a 10–15 minute interview discussion organized around one vanilla
benchmark, one Asian variance-reduction study, one Greek-estimator comparison
and a clearly qualified Heston example. The scientific contribution is the
validation and error analysis, not a claim to invent a new pricing method.

## Target output checklist, created progressively

Core figures: MC price/absolute error/SE vs N, log-log SE regression, standard
vs antithetic estimator variance, Asian control variance, Delta convergence,
FD vs pathwise Delta, Greek error vs bump, Asian monitoring convergence and
barrier monitoring convergence. Advanced: Heston timestep convergence.

Core tables: European validation including CI/runtime; variance reduction
including total cost; Greeks including analytical targets and SE; exotic
prices including monitoring schedule; replicated bias/variance/RMSE summaries.
Every script will save raw data, tables, figures, inputs and concise output.

Before leaving a milestone: explain objective and mathematics, name files,
review implementation decisions and manageable code, run tests and the stated
experiment, compare actual with expected output, and document limitations.
