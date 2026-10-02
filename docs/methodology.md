# Numerical methodology and conventions

This is the complete convention and study-design record through Milestone 16.
Milestone 17 consolidates presentation without changing the numerical estimators.
Sections retain their chronological scope; the final support matrix is in the README.

## Implemented in Milestone 1

- No dividends; constant continuously compounded rate and annualized volatility.
- Time in years, prices in the same currency units as spot/strike.
- Scalar analytical API, positive finite spot and strike, nonnegative finite
  maturity and volatility; finite negative rates are valid.
- Immutable model dataclass. Use `dataclasses.replace` for a parameter change;
  this will also make Greek bumps explicit later.
- Call/put prices at expiry and zero volatility use direct limits.
- Greeks require $T>0$ and $\sigma>0$. Vega is per 1.00 volatility.
- Ordinary float64 formulas and SciPy normal CDF/PDF. Extremely deep tails,
  underflowing $\sigma\sqrt T$, and overflowing exponentials are not covered
  by a specialized stability algorithm. Finite input validation is not a promise
  of accurate prices for every representable float. No silent price clipping is
  used to hide a numerical problem. Tests cover realistic parameter ranges.
- Tests include direct payoff quadrature, put–call parity, no-arbitrage bounds,
  independent derivative checks, boundary prices and invalid inputs.
- The spot sweep uses a readable Python loop across 201 **parameter scenarios**;
  this is unrelated to loops across millions of simulated paths. Vectorization
  will be essential for the latter, not forced into this small scalar benchmark.
- The benchmark records inputs, package versions and raw numerical results.
  Per-call timings are illustrative only, including first-call overhead.

## Simulation: implemented terminal/grid conventions and planned extensions

- Modern explicit `numpy.random.Generator`; no global random state.
- Separate random-number generation from deterministic mapping of supplied
  Gaussian draws to states. Estimators can share the same draws without guessing
  how many random numbers each code path consumed.
- Implemented in M2: `make_rng`, `standard_normal_draws`, `terminal_from_normals`
  and `simulate_terminal`. Output is a new float64 array; input shocks are not
  modified. Maturity/count/RNG validation precedes random draws. The wrapper
  consumes N normal draws even for zero maturity or zero volatility.
- M2 rejects numerical overflow and underflow to zero terminal prices instead
  of returning invalid states. This is not an extreme-parameter stability solver.
- GBM terminal arrays: `(n_paths,)`. Full paths: `(n_paths, n_steps + 1)`, with
  column zero equal to spot; Gaussian increments: `(n_paths, n_steps)`.
  M2 implements terminal arrays; M6 adds uniform-grid paths, including t=0.
- M6 exposes `standard_normal_matrix`, `paths_from_normals` and `simulate_paths`.
  The deterministic mapping infers N/m from a real finite matrix. It uses
  `cumsum(axis=1)` and reuses its own output buffer, preserving caller inputs
  including read-only or noncontiguous views. The output is C-contiguous float64.
- M6 consumes N*m normals even at zero maturity/volatility. Model, maturity,
  counts and RNG validation precede sampling. Numerical range errors during
  transformation can advance the stream; no rollback is promised.
- Grid coupling sums contiguous blocks of fine shocks and divides by sqrt(block
  size). Same-seed restarts with different matrix widths do not by themselves
  couple common calendar dates. Mathematical common-date prices are identical;
  measured tiny discrepancies reflect floating-point reduction order.
- The M6 study checks increment means/variances, adjacent-increment covariance,
  joint log-level covariance sigma^2 min(s,t), discounted means and European
  terminal prices. Distinct dates on the same paths are dependent. Per-date
  CIs are pointwise; their containment does not estimate coverage or prove the
  conditional martingale property.
- M6 records 50,000 paths at 252 steps: 101.2 MB output and 100.8 MB shocks.
  These array sizes exclude temporary diagnostics/validation allocations and
  do not measure peak process memory. Only the first 32 path/shock rows, all
  terminal prices and aggregated shocks are saved; the full grid is replayable
  from the seed, dimensions, draw order and recorded environment.
- European contracts need only terminal values.
- The implemented M7 Asian convention uses $m$ dates $jT/m$, $j=1,...,m$; **exclude
  $S_0$** from the average. A monitoring schedule is a contract input.
- The implemented M8 up-and-out call has zero rebate and is knocked out on `S >= barrier`,
  including at inception. Start with explicitly **discrete monitoring**.
- `UpAndOutCall.payoff` checks all supplied path columns, including S0 and T.
  The compact `payoff_from_extrema` route accepts matching terminal/maximum
  vectors and validates 0 <= terminal <= maximum. Callers must compute maxima
  on the intended dates, including inception; missing dates cannot be inferred.
- Barrier pricing uses the existing path route and explicit `n_steps`. Even an
  initially knocked-out option simulates paths and consumes N*m normals, matching
  the existing RNG convention; an early-exit optimization is not implemented.
- `terminal_monitored_up_and_out_price` integrates the lognormal terminal payoff
  over K<ST<B, after checking S0<B. This benchmarks m=1 only, not continuous
  monitoring. T=0, sigma=0, B<=S0 and B<=K are explicit limiting cases. Ordinary
  CDF differences are not an extreme-tail/narrow-band stability algorithm.
- Increasing Asian monitoring frequency generally changes the discrete contract;
  the limiting contract must be specified before calling this numerical bias.
- Discrete barrier monitoring misses inter-date crossings. For a nonnegative
  knock-out payoff it overprices the corresponding continuously monitored claim
  in expectation. Nested-date comparisons under GBM separate this monitoring
  effect from sampling noise. Brownian bridge correction is advanced.

## Estimators and uncertainty: implemented plain MC and planned extensions

- Define $X$ as a **discounted** payoff; never discount the standard error twice.
- Record sample variance of the independent sampling units, estimated variance
  of their mean, standard error, CI, path count, independent-unit count and time.
- Implemented in M3: European call/put payoffs, `price_monte_carlo` and
  `price_from_terminal_samples`, with at least two IID samples, sample variance
  using `ddof=1` and normal critical values for configurable confidence levels.
- M7 adds pure `ArithmeticAsianOption` payoffs and `price_from_path_samples`.
  A single `price_from_payoff_samples` implementation handles discounting and
  statistics for all routes. It accepts signed cashflows for paired comparisons.
  The full pricer requires explicit `n_steps` for Asians and rejects it for
  Europeans, which retain terminal-only simulation. Result `n_steps` records
  the observation count for path pricing and is None for terminal pricing.
- One independent row produces one Asian payoff; its correlated monitoring
  observations are not independent samples. Path input has shape (N,m+1),
  averaging `paths[:, 1:]` over axis 1 produces (N,). Products do not simulate,
  discount or generate dates. Supplied-path callers own the observation schedule.
- `arithmetic_average_moments` computes exact GBM mean/variance on the same
  schedule using O(m) prefix sums of covariance terms. This provides moment,
  put–call parity and convexity checks; it is not an exact Asian price formula.
- `n_paths == n_independent_samples` in plain MC. Supplied-sample pricing assumes
  independence; it must not be used to estimate uncertainty from correlated
  antithetic outputs without first defining the proper independent units.
- Product payoffs are undiscounted. The pricer discounts once before both mean
  and variance. Normal CI endpoints are not clipped at zero. A zero observed
  sample variance produces a point interval, not a general certainty guarantee.
- The full pricing entry point times simulation and estimation. The supplied-
  sample entry point times only payoff/discount/statistical work and validation.
- Implemented in M9: for antithetic sampling, $M$ pairs cost $2M$ paths but provide $M$ independent
  units $A_i=(X(Z_i)+X(-Z_i))/2$. Compute $s_A/\sqrt M$, not the standard error
  of $2M$ incorrectly treated independent payoffs. Negative covariance helps;
  benefit is not guaranteed for every non-monotone exotic payoff.
- `variance_reduction="antithetic"` requires an even total n_paths>=4; no silent
  rounding or unpaired remainder. Draw M (or M*m) IID Gaussian values and negate
  the entire owned vector/matrix for the second branch, leaving drift unchanged.
  Pay off one branch before constructing the other. The two full path matrices
  are not retained simultaneously. RNG consumption remains fixed at M or M*m
  even at expiry, zero volatility or an initial barrier hit.
- `sample_variance` is discounted pair-mean variance for antithetic results.
  Compare `estimator_variance=sample_variance/n_independent_samples`, not raw
  sample variances of different unit types. The low-level paired entry point
  accepts aligned undiscounted payoff vectors and excludes simulation from runtime.
- For control variates, use $X-\widehat\beta(Y-\mu_Y)$, where $Y$ and $\mu_Y$
  follow the same discount convention. Use an **independent pilot** for
  $\widehat\beta$; conditional on that pilot, unbiasedness follows from
  $\mathbb E[Y-\mu_Y]=0$. Fitting and applying beta on the same sample does
  not in general give exact finite-sample unbiasedness.
- The geometric Asian analytic control must use precisely the arithmetic
  contract's monitoring dates. A continuous-geometric formula is the wrong
  expectation for a discretely observed control.
- Pilot paths count toward the computational budget. Report both variance ratios
  at equal total path costs and time-adjusted efficiency. VRF compares variances
  of **price estimators**, not unmatched payoff variances.
- Finite-difference standard errors are computed from the path-level paired
  difference samples, including covariance induced by common random numbers.
- Spot bumps are in price units; volatility bumps are decimal volatility units.
  The suggested spot bumps 10, 5, 1, ... cannot be reused as volatility bumps.
- Smooth central-difference bias is $O(h^2)$ locally. Small-bump noise depends
  on the estimator: CRN call Delta approaches a finite-variance pathwise limit;
  Gamma and discontinuous payoffs can remain much more delicate. Floating-point
  cancellation is a distinct additional source of error.
- Use pathwise Delta only when differentiation can be interchanged with
  expectation. A European payoff kink has probability zero for $T,\sigma>0$;
  domination/integrability is also needed. A moving barrier indicator requires
  more care and cannot be naively differentiated as zero.

## Research evidence: implemented validation and planned studies

- M9 compares 200 independent runs of 10,000 total payoff evaluations per method
  in five pre-specified cases: European call/put, 12-date Asian call, 96-date
  B=120 barrier call, and a terminal-only symmetric-window barrier. Default
  baseline is S0=K=100, r=.05, sigma=.2, T=1. The counterexample instead uses
  r=.02, K=100/1.1 and B=110, so both antithetic legs share a central support window.
- Every (case,replication,method) uses a distinct SeedSequence spawn key. Methods
  are independent, not paired by sharing their first half. Both methods receive
  equal path/payoff budgets; antithetic uses half the Gaussian draws. Full-pricer
  timings exclude RNG construction, warmups, replayed covariance diagnostics
  and output. Method execution order alternates; scenarios remain in fixed order.
- Report empirical variance across run estimates, mean reported estimator
  variance, RMS reported SE, runtimes, and VRF for equal budgets. Empirical VRF
  and a ratio of mean reported variances are distinct noisy statistics. Approximate
  F intervals for empirical ratios assume independent approximately normal run
  estimates; they are pointwise, not simultaneous. Zero estimated variances make
  ratio comparisons undefined; missing values replace misleading efficiency claims.
- Bias/RMSE/coverage use exact targets only for Europeans and the terminal-only
  counterexample. Coverage rates have Wilson intervals. No unknown-target Asian
  or 96-date barrier bias is inferred from a difference between methods.
- The symmetric-window case also has independently integrated Gaussian payoff
  moments. They establish positive covariance and population VRF<1, despite the
  200-run empirical ratio interval still containing 1. Retain the original run
  evidence; do not rerun until a desired significance result appears.
- Store all 2,000 estimator rows and seeds, not all path samples. Price/SE replay
  requires recorded draw order and environment. Timing-adjusted efficiency is
  empirical VRF times mean runtime_plain/runtime_anti and is environment-specific.

- M8 studies nested counts 1,4,12,24,48,96,192,384 at B=120 using 30 independent
  child streams, 5,000 paths each (150,000 pooled paths per contract). Every
  successive grid refines its predecessor. Testing only that both divide the
  finest count is insufficient: grids of 2 and 3 dates both divide 6 but are
  not nested with each other. Tests explicitly reject that design.
- The terminal value is shared across grids. The monitored maximum increases
  pathwise with nested dates, so a zero-rebate up-and-out payoff can only fall.
  Raising B on a fixed grid has the opposite ordering. The study verifies both
  identities and the synthetic in/out decomposition against the vanilla payoff;
  an up-and-in product is not advertised as implemented.
- Store terminal values (30,5000) and maxima (30,5000,8). The pooled table gives
  paired differences against 384 dates, hit proportions with Wilson intervals,
  and the fraction of all paths hit on that reference but missed on each coarse
  grid. These missed-hit rates are not relative to a continuous barrier and do
  not weight losses by payoff magnitude. Price differences do weight cashflows.
- The m=1 exact target supports a 30-replication bias/variance/RMSE summary for
  5,000-path runs. Do not transfer that benchmark to other monitoring grids.
  No exact continuous-price bias/RMSE or convergence exponent is claimed.
- A separate 5,000-path, 96-date engine call uses spawn key (30,) and reports
  full runtime. Monitoring timings are shared work. Saved arrays use 1.2 MB
  terminals and 9.6 MB maxima; each fine output uses 15.4 MB, normals 15.36 MB.
  All figures are array sizes, not measured peak memory. No general batching.

- M7 uses 20 independent SeedSequence child streams, each with 5,000 full paths
  on 384 dates. The grids 1,4,12,24,48,96,192,384 are nested subsets of each
  path; both call and put reuse the corresponding average. At each fixed grid,
  pooling supplies 100,000 independent payoffs. Cross-grid prices are correlated.
- Estimate differences against the discrete 384-date contract using discounted
  pathwise payoff differences and their sample SE. Include reference covariance;
  do not add two standalone price variances as if they were independent. A zero
  difference and SE for reference-minus-itself say nothing about reference bias.
- Save averages with shape (20,5000,8), all 320 per-run estimates and pooled
  statistics. Replication price SD concerns a 5,000-path estimator; divide by
  sqrt(20) for an alternative SE of the pooled price. Twenty replications give
  noisy empirical dispersion estimates. No exact Asian price bias/RMSE is claimed.
- The explicit path matrix uses 15.4 MB per run, normal matrix 15.36 MB, retained
  averages 6.4 MB, excluding other allocations; these are not peak memory values.
  The library itself still allocates a full grid for one pricing call. The study
  releases paths between independent runs; general library batching is future work.
- Monitoring-study simulation is shared; raw row timings cannot be summed as
  independent pricing costs. `asian_engine_quotes.csv` separately records full
  5,000-path, 12-date pricing calls with spawn key (20,), independent of the study.
  Pooled-table timings cover summary calculations only. Timings are illustrative.

- Implemented in M4: European call/put validation at $N=10^3,10^4,10^5,10^6$,
  including absolute/relative errors, SE, approximate 95% CI and runtime.
  Relative error is a fraction in the CSV. Signed error/SE is a diagnostic, not
  a measured bias. Undefined ratios are represented as missing values.
- Each M4 row restarts the same seed. Within a row the draws are IID; across
  counts they form nested prefixes, and call/put share the same terminal sample.
  These rows are correlated and are not independent replications. Saving the
  largest terminal array permits replay via `terminal[:N]`; software versions,
  generator, draw convention and an array checksum are recorded.
- M4 timings include one full pricing call and exclude RNG construction by the
  caller, analytical pricing, file output and plotting. They are single wall-clock
  observations including first-call overhead, not a performance comparison.
- Implemented in M5: 200 independent replications per N on 10 counts from
  1,000 to 1,000,000. Fit log mean reported SE against log N; compare against
  the predicted slope -1/2 and the log-log slope of empirical RMSE. Also fit
  each independent replication's SE curve to measure the distribution of
  finite-grid slopes. The normal CI for the mean of these slopes does not
  target an unknown asymptotic exponent; the exponent is known from theory.
- M5 uses independent `SeedSequence` children for every (count, replication)
  cell; call and put share that cell's terminal draws. Spawn keys are recorded.
  Rows at different N are no longer nested. No classical homoskedastic OLS
  standard errors are reported for noisy, heteroskedastic SE measurements.
- M5 measures price-CI coverage at each fixed product/N using Bernoulli indicators
  over independent runs. A 95% Wilson score interval quantifies uncertainty in
  each estimated coverage probability. These are pointwise intervals, not a
  simultaneous confidence band; call and put's coverage indicators are paired.
- M5 raw files retain all estimator outputs, including misses and zero-SE flags.
  The displayed single-run curve is replication 0, chosen before inspection.
  Terminal arrays are discarded after each cell; no R-by-N price-state matrix
  is allocated. Raw terminal replay uses the root seed and recorded spawn key.
- A mean reported SE and the RMS reported SE differ: the latter is the square
  root of the average reported variance. Both are saved, alongside the empirical
  SD of repeated price estimates. A slope near -1/2 partly reflects the built-in
  factor N^(-1/2), so dispersion and RMSE provide additional validation evidence.
- Use a small repeated-seed comparison in the core (at least 10 replications)
  so no comparative claim relies on one lucky run. Milestone 13 extends the M5
  repeated-run framework to estimator comparisons, with at least 30 replications,
  RMSE, uncertainty and runtime comparisons.
- Across $R$ independent replications, report mean, empirical bias against a
  known target, sample variance across estimates, and
  $\sqrt{R^{-1}\sum_r(\widehat V_r-V_*)^2}$. This empirical RMSE uses denominator
  $R$, whereas an unbiased sample variance uses $R-1$; distinguish the conventions.
- Pair methods within a replication when useful; make replications independent.
  M5 implements RNG stream construction using `SeedSequence.spawn` and records
  its configuration. A seed alone is not a full reproducibility specification.
- When no exact exotic price is available, do not call deviation from another
  noisy estimate “bias” without reporting the reference uncertainty. The
  arithmetic Asian control study can compare variance without a known exact price.
- Separate sampling error, monitoring approximation, dynamics time stepping,
  finite-difference bias, floating-point error and model error. Confidence
  intervals quantify only sampling uncertainty for the implemented estimator.

## Advanced extension plans

- Full truncation Euler: retain the raw variance state $\widetilde v$ and use
  $\widetilde v^+=\max(\widetilde v,0)$ in both variance drift and diffusion.
  Do not silently conflate this with clamping the stored state after each step.
- Use a log-price update with the same positive variance proxy to maintain
  positive stock prices; record that exact GBM steps with frozen variance do not
  make the Heston discretization exact.
- Heston validation: correlated normals, discounted-stock martingale check,
  constant-variance Black–Scholes limit ($\xi=0$, $v_0=\theta$), timestep study
  with coupled Brownian increments and multiple replications. Timestep agreement
  alone is not an independent proof of correct Heston prices.
- Heston parameters received from calibration must be risk-neutral, use the same
  time/rate/dividend conventions, and distinguish variance from volatility.
- One float64 path matrix costs $8N(m+1)$ bytes; at $N=10^6,m=252$, about
  2.024 GB (1.89 GiB), before increments, temporary arrays and payoffs. Batching
  will aggregate counts, means and centered sums of squares stably. Preserve
  entire antithetic pairs, CRN alignment and the independent pilot/production
  split across batches. Batching preserves the estimator but changing draw order
  need not preserve bit-for-bit paths for the same seed.

## M10 implemented control-variate protocol

- Arithmetic Asian call/put is the target; the matching discrete geometric
  Asian is the control. Dates are jT/m, exclude S0, include T. Exact lognormal
  moments and prices are independently tested with covariance sums and quadrature.
- `GeometricAsianOption` validates strictly positive paths; `exp(mean(log(S)))`
  avoids multiplying many prices. The m=1 case returns the terminal exactly.
- The engine requires explicit `n_pilot_paths>=2` and at least two production
  paths. `n_paths` includes both blocks. Unsupported products and budgets fail
  before consuming random numbers. Pilot and production are successive disjoint
  blocks of the supplied Generator; paired X and Y share a path within each block.
- The pilot fits beta only. Its payoffs never enter the price mean. Zero sample
  control variance uses beta=0 and records `zero_control_variance`; production
  SE is still computed from production payoffs, not set to zero automatically.
- `sample_variance` is the variance of a discounted production residual,
  conditional on the pilot. `n_independent_samples=n_paths-n_pilot_paths`.
  The result retains pilot count, beta, pilot correlation, known **undiscounted**
  control mean and fit status. Low-level supplied-sample functions cannot verify
  statistical independence: callers own this contract and any external pilot cost.
- The experiment fixes 200 independent runs per method, total paths 10,000,
  pilot 1,000, S0=K=100, r=.05, sigma=.2, T=1, call/put on 12 dates and call
  on 48 dates. Root seed 20260930, PCG64, spawn keys (case, replication, method).
  Distinct method streams permit independent variance-ratio comparisons.
- Compare empirical variance across run prices, mean reported variance and
  RMS reported SE. Report approximate F intervals (near-normal independent run
  estimates assumed), differences of independent mean prices and runtime.
  Do not assign an exact arithmetic-Asian bias, RMSE or coverage without a target.
- Every replication refits beta. The geometric production estimator is separately
  checked against its exact price, with bias, RMSE and Wilson coverage intervals.
  This diagnostic replays existing production draws; it is not another independent
  sample to pool with the arithmetic estimates.
- Full pricer timing includes exact control pricing, pilot simulation and fitting,
  production simulation, both payoffs and uncertainty. Excludes RNG construction,
  warmups, diagnostic replays, CSV writing and plotting. Rotate execution order.
  Report VRF times plain/control runtime as a measured time-adjusted efficiency.
- CV evaluates two payoffs on every path, versus one for plain/antithetic.
  Equal **stock-path** budget includes the pilot but is not equal payoff cost.
  At the largest production case the output grid occupies 9,000*49*8=3.528 MB,
  and shocks 3.456 MB; additional log/exp/validation arrays use more memory.
  Pilot and production full grids are never simultaneously retained. This is
  not general-purpose batching, and no measured peak-RAM claim is made.
- Save all 1,800 estimator rows, 600 geometric diagnostic rows and one prespecified
  pilot/production payoff sample (case0/run0). Metadata includes hashes, versions
  and draw conventions. Seed replay reproduces numerical estimates, not timings.

## M11 implemented finite-difference protocol

- `finite_difference_greek` estimates one central Delta, Gamma or Vega with
  CRN by default. `finite_difference_from_normals` reuses supplied shocks.
  Pure first/second-difference helpers do no discounting or RNG work.
- Rates, maturity, strike and monitoring grid are held fixed. Bump initial spot
  for Delta/Gamma; bump decimal volatility for Vega. Require T>0, sigma>0,
  0<hS<S0, 0<hSigma<sigma, at least two independent rows, finite denominators
  and representable parameter changes. No silent bump clipping or one-sided
  boundary rules. Fail ordinary validation before RNG consumption.
- European shocks have shape (N,); path-product shocks have shape (N,m).
  Reuse the entire matrix under CRN. For independent legs, draw up then down
  then base (Gamma only). Different rows must be IID. Supplying explicit arrays
  cannot prove independence, so the low-level caller owns that assumption.
- Simulate each perturbed model through existing GBM transforms and pure product
  payoffs. Discount the path-level differences exactly once with the common
  sample-summary routine. Covariance is captured by taking sample variance after
  differencing; never combine independent-price SEs for a CRN result.
- Greek `n_paths` is the independent tuple count N; `n_payoff_evaluations` is 2N
  for Delta/Vega, 3N for Gamma. This differs from total simulated-path counting
  in the M9/M10 price estimator. Compare methods within a Greek at equal payoff
  evaluations; do not interpret shared randomness as extra independent samples.
- The dedicated result has estimate, sample variance, SE, CI, bump, Greek,
  sample count, payoff-evaluation count, CRN flag, grid, level and runtime.
  Variance units are squared Greek units. CI targets the finite-bump expectation,
  not the derivative after truncation bias. Supplied-shock runtime excludes RNG;
  full-call runtime includes all normal draws, valuations and statistics.
- Support European call/put, arithmetic/geometric Asian call/put and discrete
  up-and-out call through the same route. Tests verify pathwise replay for each.
  The main benchmark is a European call with S0=K=100, r=.05, sigma=.2, T=1.
  No exact exotic-Greek accuracy or continuous-barrier claim is made.
- Fix N=20,000 and R=100 independent replications per bump/method. Spot bumps
  are 10,5,1,.5,.1,.01; vol bumps .1,.05,.01,.005,.001,.0001. Root seed 20261001,
  PCG64, bump-study keys (0,Greek,bump,replication,method); method 0 is CRN.
  Convergence keys (1,count_index,replication,method), fixed Delta h=1 and
  N=1,000,3,000,10,000,30,000,100,000. Warmups use prefix 99. Cells are independent.
- Alternate method execution order. Timings include validation, shock generation,
  GBM transforms, payoffs and uncertainty; exclude Generator construction,
  warmups, analytical targets, summaries, plotting and I/O. The benchmark does
  not optimize or share valuations across separate Greeks or bump settings.
- For each setting, calculate true Greek analytically and finite-bump expectation
  using analytical prices. Report deterministic truncation bias, empirical bias,
  empirical variance, RMSE, absolute error of mean, reported SE, runtime and
  coverage for both targets with Wilson intervals. R-run variance uses R-1;
  empirical RMSE uses R. Mean-estimate SE is distinct from single-run RMS SE.
- Approximate F intervals compare independent run-level variances and assume
  near-normality. Gamma with tiny bumps can be skewed/sparse, weakening this
  approximation. Record all settings; do not select favorable seeds or increase
  replication counts after inspecting significance.
- Save 3,600 bump rows and 1,000 convergence rows with exact stream keys. Raw
  estimates, uncertainty, runtimes and targets reconstruct all summary tables.
  Metadata records versions, parameters and SHA256 hashes. Seeds regenerate
  underlying shocks; no large path matrices are persisted.
- A separate deterministic Gamma diagnostic differences float64 analytical prices
  down to h=1e-7, exposing cancellation. These tiny-bump values are not reliable
  reference derivatives. More Monte Carlo samples cannot cure this roundoff.
- Terminal sensitivities use O(N) storage. Path sensitivities use O(Nm): retain
  one shock matrix and only one full perturbed path grid at a time in CRN mode,
  plus payoff vectors. Independent mode retains up to three shock matrices.
  This avoids simultaneous full path outputs but is not batching or measured
  peak-memory optimization. Antithetic/control combinations are deferred.

## M12 implemented pathwise Delta protocol

- Under exact GBM, dST/dS0=ST/S0. Call derivative samples are
  (ST/S0)*1{ST>K}; put samples are -(ST/S0)*1{ST<K}. Discount once when
  summarizing. The pure sample helper returns a new float64 (N,) array.
- Use a strict mask and set strike ties to zero, a null event under stipulated
  T>0, sigma>0. The expectation interchange relies on both almost-sure
  differentiability and Lipschitz domination by the integrable GBM growth factor.
  Do not generalize the argument from probability-zero kinks alone.
- `PathwiseDeltaResult` reports estimate, sample variance, SE, CI, N, confidence
  level, runtime and terminal-evaluation count. There is no artificial bump or
  payoff-price field. Supplied-sample runtime excludes simulation; full-call
  runtime includes N normal draws, one terminal transform and derivative summary.
- Full and supplied-terminal APIs support European call/put only. Require positive
  T, sigma and strictly positive terminal prices; N>=2 for statistics. Reject
  unsupported products and ordinary input errors before random draws.
  IID sampling from the stated model is a low-level caller responsibility.
- `np.divide(..., out=zeros, where=mask)` leaves inactive entries at zero and
  avoids unnecessary inactive ratios. Arrays are unchanged; both masks and
  output use O(N) space. No full time grid or finite-difference valuation is needed.
- For call Delta, exact discounted sample variance is
  exp(sigma²T)*Phi(d1+sigma*sqrt(T))-Phi(d1)². Validate the first and second
  moments with independent normal quadrature and fixed-seed MC checks. Test
  put analogues, sample parity, local FD agreement away from kink crossings,
  stream continuation, shapes, boundaries and input immutability.
- Main experiment: baseline call S0=K=100, r=.05, sigma=.2, T=1. PW and CRN
  FD bumps 10,1,.1; R=200 independent runs per method/budget. Equal-row design
  uses N=20,000 each (PW 20,000 vs FD 40,000 terminal evaluations). Equal-terminal
  design uses PW N=40,000 vs FD N=20,000 (40,000 terminal evaluations each).
- Terminal-evaluation counts are a transparent proxy, not equal elapsed time,
  equal Gaussian counts or an optimal computational scheme. FD uses two direct
  model transforms; caching/scaling a single terminal draw for spot bumps is
  a possible future optimization. Do not interpret this count as a universal
  twofold computational lower bound for finite differences.
- Independent comparison streams use PCG64, root seed 20261002, keys
  (0,budget,replication,method). Rotate method order and use warmup prefix 99.
  Convergence uses (1,count,replication,method) at five counts 1,000 to 100,000,
  PW versus FD h=1 with equal rows. Fit RMS-SE exponents and compare PW to its
  exact sampling SE. Full-call timing excludes Generator construction, reference
  formulas, warmups, diagnostic replays, plotting and I/O.
- Compare empirical mean, bias, variance, RMSE, reported SE and runtime. F
  variance-ratio intervals assume independent approximately normal run means;
  no same-path cross-method data enter these ratios. Report coverage both of
  analytical Delta and each method's expectation, with Wilson intervals.
- Separate paired diagnostic: keys (2,replication), same N=20,000 shocks for
  PW and FD at six bumps 10,5,1,.5,.1,.01. Summarize signed FD-minus-PW samples
  with covariance, then summarize independent run means. Its target is the
  exact FD bias. Bumps share paths within a run and must not be pooled as IID.
  This diagnostic is not included in method timing comparisons.
- Save 1,600 independent comparison, 2,000 convergence and 1,200 paired rows,
  plus summary tables, figures, metadata and SHA256 hashes. Seeds replay numeric
  results, not runtimes. Earlier milestone results retain their original streams.
- PW is theoretically unbiased under its assumptions but is not guaranteed to
  have lower variance or runtime than smoothed FD at equal N. Large bumps can
  reduce FD variance while increasing total MSE. Report these trade-offs rather
  than selecting a single favorable metric. No pathwise Gamma/Vega or exotic
  sensitivities are introduced. M13 will extend research comparisons; M14 adds LR.


## M13 implemented multi-scenario protocol

The design is declared as frozen Scenario objects and method tuples before the
experiment. It reuses M1–M12 estimators without selecting a winning seed or
retuning the control after inspecting stress outcomes. Every cell has 100 runs.
All calls have K=100, r=.05. Greek (S0,sigma,T) scenarios are atm (100,.2,1),
itm (120,.2,1), otm (80,.2,1), short (100,.2,.04), low_vol (100,.08,1),
high_vol (100,.5,1), long (100,.2,3), rare (80,.1,.25). Asian scenarios use
atm, itm, high_vol, and a different rare case (85,.15,.25); 12 future dates.

Eight Greek methods: pathwise Delta, CRN Delta at spot bumps 1% and 0.1%,
independent-leg Delta at 1%, CRN Gamma at 1% and 0.1%, CRN Vega at absolute
volatility bumps .01 and .001. FD uses N=10,000 rows. PW uses 2N paths, making
Delta comparisons equal in terminal evaluations. Gamma comparisons cost 3N;
Vega comparisons cost 2N. Do not compare quantities of different units.

Asian plain/antithetic/control methods each consume 10,000 stock paths. The
control includes 1,000 pilot paths and 9,000 production paths and evaluates two
payoffs per path; antithetics use 5,000 independent pairs. Equal stock paths
therefore do not mean equal payoff evaluations or runtime. Runtime efficiency
is empirical VRF times mean reference runtime divided by candidate runtime.

SeedSequence entropy is 20261003. Spawn keys are (0,case,rep,method) for Greeks,
(1,case,rep,method) for Asians, (2,group,method) for bootstrap, and prefix 99
for warmups. All method streams are independent. Method order rotates by run.
Timing covers the full estimator call including draws, excludes Generator
construction, analytical targets, warmups, bootstrap and output. Millisecond
measurements are environment-dependent and are not exact replay targets.

Store all 7,600 run-level outputs, targets, streams, runtime and control metadata.
Summaries retain zero-SE and negative corrected estimates. A negative CV estimate
is possible; silently clipping it changes the estimator and introduces bias.
Use R-1 for sample variance and R for empirical MSE. Report finite-bump and
exact-Greek interval coverage separately with Wilson uncertainty. No exact
arithmetic-Asian bias, RMSE or coverage is asserted; use the geometric lower bound
only to detect some intervals that certainly miss.

For each of 48 within-quantity comparisons, use 2,000 independent run-level
bootstrap resamples, save the valid fraction and flag undefined ratios. Intervals
are pointwise, approximate and unable to recover unseen tails. They are not
multiple-testing corrections. Avoid treating a single low coverage estimate
among many cells as an isolated definitive rejection. The rare-event diagnostic
is stronger: it also has a precomputable no-exercise probability.

Pilot beta remains the existing unbounded M10 coefficient. No moment guarantee
is inferred from finite observed slopes or sample variances. Conditional pilot
centering is exact, while unconditional moment claims require assumptions.
Report pilot fallback frequency and extreme slopes alongside CV results.
Memory remains O(N*m) for one Asian run, rather than O(R*N*m) for the study;
retained run-level evidence is O(R*number_of_cells). Bootstrap arrays have shape
(B,R), here (2,000,100), and the loop is over experiments, never stock paths.


## M14 implemented likelihood-ratio protocol

- Scope: LR spot Delta for fixed-strike vanilla and fixed-cash digital calls/puts
  under GBM. APIs retain the original Gaussian draw. Supplied-normal entry points
  require independent standard-normal rows; dependent/antithetic rows cannot be
  passed to the IID summary. No LR path score, Gamma, Vega or score control.
- Digital call uses ST>K, put ST<=K. Pricing handles T=0/sigma=0; analytical Delta
  and LR require both positive. The payoff also uses the existing terminal MC,
  antithetic and FD routes. Pathwise digital input is explicitly rejected.
- Predeclared comparisons: vanilla ATM call/put and short call (T=.04); digital
  ATM call, short call and rare call (S0=80,sigma=.10,T=.25). Otherwise S0=K=100,
  r=.05,sigma=.20,T=1; digital payout=1. See frozen CASES in the experiment.
- Vanilla methods: LR, PW, FD h=1 and .1. Digital methods: LR, FD h=1,.1,.01.
  Each cell has 200 independent runs. LR/PW use 20,000 draws; FD uses 10,000 CRN
  tuples, all costing 20,000 terminal evaluations. Counts do not equate runtime;
  no claim of optimized terminal transformation is made.
- Convergence: vanilla LR/PW and ATM digital LR at N=1k,3k,10k,30k,100k with 100
  independent runs per point. These series measure N^-1/2 sampling-SE behavior,
  not comparable relative accuracy across different payoff scales.
- Root seed 20261004; comparison keys (0,case,rep,method), convergence
  (1,series,count,rep), bootstrap (2,case,candidate), warmup prefix 99. Independent
  method streams; rotating order. Report full estimator-call times, excluding
  Generator construction, analytical benchmarks, warmups, bootstrap and writing.
- 4,800 comparison and 1,500 convergence rows are saved, with exact Delta,
  finite-bump target, reported variance and available population moment references.
  Comparisons use LR variance divided by candidate variance; values above one
  favor the candidate. Bootstrap uses 2,000 independent run-level resamples.
- All runs, including zero outcomes, enter summaries. Known population variance
  is reported separately from finite-study variance. Undefined empirical ratios
  remain NaN and are flagged. Display population and empirical RMSE for digital
  bumps; an all-zero empirical study is not evidence of zero population variance.
- The probability of no digital crossings across the entire rare small-bump study
  is a mathematical diagnostic added during interpretation; it does not retune
  parameters or remove unfavorable runs. Pointwise bootstrap and Wilson intervals
  do not establish simultaneous or rare-tail coverage. Earlier M13 caveats persist.
- LR sample arrays use O(N) memory; paths are terminal only. Loops iterate scenarios
  and independent runs. Vector operations evaluate all paths in a run at once.
- M14 adds 16 test functions; the full 197 pass via the standard-library runner.
  The runner and pytest configuration include the repository root for importing
  experiment modules, while installed library imports remain separate.

## M15 — Heston integration study

- Risk-neutral baseline: S0=K=100,r=.05,T=1,v0=theta=.04,kappa=2,xi=.3,rho=-.7.
  Scenarios change xi to .9, rho to 0 or .5, or xi to zero with v0=.09.
  Feller failure is retained and reported, not repaired.
- Full-truncation auxiliary variance with exponential old-variance stock update.
  Integration grids 12,24,48,96,192,384; every exotic has 12 future monitoring
  dates, with inception additionally monitored for the zero-rebate barrier B=120.
- 30 independent replications of 4,000 paths per scenario. A replication draws
  two independent (N,384) normal matrices. Aggregate each driver separately into
  coarse shocks. Products/grids share paths; replications/scenarios are independent.
- Root seed 20261005, PCG64 SeedSequence keys (0,case,rep) for the grid study,
  (1,rho-index,rep) for correlation, (2,case,product) for standalone pricing.
  The library's streaming draw order differs from the coupled study's matrix order.
- European references use affine Fourier inversion and deterministic integrated
  variance at xi=0; compare cutoffs 200/400 and tolerances 1e-9/1e-11. Separate
  ODE, parity, bounds and limiting-case tests validate the implementation.
  Quadrature error estimates do not bound the omitted infinite tail.
- Pooled variance reconstructs within- and between-run sums of squares:
  M2=sum_r[(N-1)*s_r²+N*(mean_r-grand_mean)²], s²=M2/(RN-1).
  Pooled SE=sqrt(s²/(RN)); also report empirical run variance and SE of the run
  mean. Preserve paired payoff differences when comparing integration grids.
- Report estimated European bias and run RMSE against the independent reference.
  Exotic differences use 384 steps as a numerical reference, never as an exact
  value. Sampling CIs exclude grid bias, monitoring-contract differences and
  model error. No numerical weak-convergence order is fitted to unresolved bias.
- 3,600 price and 3,600 paired rows share simulations; they are not 7,200
  independent experiments. Correlation adds 150 independent driver samples.
  Five summary/reference tables and three figures accompany raw rows and hashes.
- Grid simulation timing excludes shock generation/aggregation and is shared
  across products. A separate 20-row table times the complete standalone API;
  these one-run timings are examples, not a robust speed comparison.
- Generated-shock simulation uses O(N) working state and O(N*m_obs) output.
  The coupled experiment deliberately stores O(N*m_steps) shocks, one replication
  at a time. No general batching, Heston variance reduction or Greeks are added.
- 20 new test functions bring the suite to 217, all executed with the standard-
  library fallback. Installed-package and artifact checks are saved separately.

## M16 — Final benchmark and evidence index

- Predeclared eight cases, sixteen method/configuration cells, 50 independent
  replications each, new root seed 20261006: 800 run-level estimates. No new
  pricing method or post-result seed/bump tuning. Reuse M9's five price cases
  and add ATM call Delta/Gamma/Vega; exact parameters are in `final_study.json`.
- Pricing uses 12,000 paths including a 1,200-path pilot for control; the control
  evaluates two payoffs/path. Greeks use 12,000 terminal evaluations: 6,000 FD
  Delta/Vega rows, 4,000 Gamma rows, 12,000 PW/LR rows. FD spot h=1, vol h=.01.
- Streams are PCG64 SeedSequence (0,case,rep,method), bootstrap (1,case,candidate),
  warmup (99,case,method). Rotate method order; separate 120-path warmups.
  All method streams are independent. CRN operates inside FD calls only.
- Save every estimate, reported variance, normal CI, independent count, path
  count, payoff work, target, bump, key and full estimator-call runtime. Targets
  are computed outside that timing. No paths are retained between calls.
- Empirical variance uses R-1, empirical RMSE R, and SE of the mean sqrt(s²/R).
  Known-target bias/coverage use exact prices/Greeks; FD also keeps its finite-
  bump target. Exact Asian/96-date barrier bias, RMSE and coverage stay missing.
- Compare reference/candidate variance within each case. 2,000 independent
  run-level bootstrap resamples; report validity fraction and flag degeneracy.
  Coverage uses Wilson intervals. All intervals are pointwise, not simultaneous.
  Time-adjusted ratios are descriptive and have no timing-uncertainty interval.
- Historical results retain their source tables, sample counts and original
  confidence semantics. Twelve required figures are indexed with table/figure
  hashes and reconstruction commands. M16 checks presence/provenance and reuses
  those studies, not their full simulations. Nine historical headline rows are
  selected explicitly; no ratios are pooled across different budgets.
- `--rebuild` checks raw/new-study and historical-source hashes before regenerating
  outputs. It preserves recorded measurements, including timing. Source-code
  hashes identify the implementation used in generating the current output.
- Seven new tests bring the suite to 224; the standard-library runner executes
  the plain pytest-discoverable functions. Numerical and installed-package
  checks are recorded separately. Portfolio editing remains M17.
