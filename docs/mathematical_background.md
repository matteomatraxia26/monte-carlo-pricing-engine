# Mathematical background

Status: derivations for implemented Milestones 1–16.

## 1. Physical and risk-neutral probabilities

Under the physical measure $\mathbb P$, an idealized stock might satisfy

$$dS_t=\mu S_t\,dt+\sigma S_t\,dW_t^{\mathbb P}.$$

$\mu$ describes expected real-world returns. Derivative valuation does not
generally equal the physical expected payoff discounted at the risk-free rate:
that would omit compensation for risk.

In the frictionless complete Black–Scholes market, dynamic replication and
no-arbitrage identify a unique equivalent martingale measure $\mathbb Q$.
With money-market account $B_t=e^{rt}$, discounted traded assets are martingales
under $\mathbb Q$ in this model. For a replicable integrable payoff $\Phi$,

$$\frac{V_t}{B_t}=\mathbb E^{\mathbb Q}\!\left[
\frac{\Phi}{B_T}\middle|\mathcal F_t\right],\qquad
V_0=e^{-rT}\mathbb E^{\mathbb Q}[\Phi].$$

This is a change of probability weights, not a claim that investors are
psychologically risk-neutral. Under $\mathbb Q$ the stock drift is $r$ (no
dividends). In incomplete models such as Heston, no-arbitrage alone does not
select a unique $\mathbb Q$; calibrated risk-neutral parameters specify the
pricing model. Historical variance parameters cannot simply be substituted.

Monte Carlo approximates this expectation by averaging simulated payoffs. It is
particularly useful when path dependence makes analytical integration difficult.
Its $N^{-1/2}$ rate persists in high dimension, although the variance and cost per
sample can grow substantially with dimension.

## 2. Exact GBM solution

Under $\mathbb Q$,

$$dS_t=rS_t\,dt+\sigma S_t\,dW_t^{\mathbb Q}.$$

For $f(s)=\log s$, $f'(s)=1/s$ and $f''(s)=-1/s^2$. Itô's formula yields

$$d\log S_t=\frac{dS_t}{S_t}-\frac{1}{2S_t^2}(dS_t)^2
=\left(r-\frac12\sigma^2\right)dt+\sigma dW_t^{\mathbb Q}.$$

Integrating over a time interval and using independent Gaussian increments:

$$S_{t+\Delta t}=S_t\exp\left[
\left(r-\frac12\sigma^2\right)\Delta t+\sigma\sqrt{\Delta t}Z\right],
\qquad Z\sim\mathcal N(0,1).$$

The $-\sigma^2/2$ correction comes from quadratic variation. It ensures
$\mathbb E^{\mathbb Q}[S_T]=S_0e^{rT}$, using
$\mathbb E[e^{aZ}]=e^{a^2/2}$.

Exact GBM transitions preserve positivity and have no time-stepping bias at the
chosen observation dates. Euler–Maruyama would use
$S_{t+\Delta t}=S_t(1+r\Delta t+\sigma\sqrt{\Delta t}Z)$, which can become
negative and introduces discretization error. Exact transitions do **not** detect
barrier crossings between grid dates and do not make discrete Asian monitoring
equivalent to a continuous-time average.

## 3. Monte Carlo foundation (implemented in Milestone 3)

Let $X_i=e^{-rT}\Phi_i$ be independent, identically distributed discounted
payoffs with finite variance $\sigma_X^2$. Then

$$\widehat V_N=\bar X_N=\frac1N\sum_{i=1}^N X_i,\qquad
\mathbb E[\widehat V_N]=V_0,\qquad
\operatorname{Var}(\widehat V_N)=\frac{\sigma_X^2}{N}.$$

The unbiased sample variance estimates the variance of **one discounted payoff**:

$$s_X^2=\frac{1}{N-1}\sum_{i=1}^N(X_i-\bar X_N)^2,
\qquad \widehat{\mathrm{SE}}=\frac{s_X}{\sqrt N}.$$

Thus $s_X^2$ is not the variance of the price estimator: that estimate is
$s_X^2/N$. With positive finite variance the central limit theorem gives

$$\frac{\widehat V_N-V_0}{\sigma_X/\sqrt N}
\xrightarrow{d}\mathcal N(0,1),$$

and a large-sample approximate 95% interval is
$\widehat V_N\pm1.96\widehat{\mathrm{SE}}$.
Across repeated independent runs, approximately 95% of these constructed
intervals cover the fixed model value. It is not a guarantee for a specific run,
nor an interval for future option payoffs, nor protection against model bias.
Sparse rare-event payoffs can make the normal approximation poor.

The root-mean-square error is $\sigma_X/\sqrt N$ for an unbiased estimator:
halving it requires four times as many independent samples. Realized absolute
error can increase with $N$; the convergence statement concerns the distribution
of error, not monotonic improvement of a single trajectory.

## 4. European call formula derived from the expectation

For $S_0,K,T,\sigma>0$, define

$$d_1=\frac{\log(S_0/K)+(r+\sigma^2/2)T}{\sigma\sqrt T},
\qquad d_2=d_1-\sigma\sqrt T.$$

Let $\Phi$ and $\varphi$ now denote the standard normal CDF and density (the
payoff will be written explicitly). A call is exercised when $Z>-d_2$. Hence

$$C=e^{-rT}\left(\mathbb E[S_T\mathbf1_{S_T>K}]
-K\mathbb Q(S_T>K)\right).$$

The probability term is $\Phi(d_2)$. For the first term, put
$a=\sigma\sqrt T$ and complete the square:

$$e^{az-a^2/2}\varphi(z)=\varphi(z-a),\qquad
e^{-rT}\mathbb E[S_T\mathbf1_{Z>-d_2}]
=S_0\int_{-d_2}^{\infty}\varphi(z-a)\,dz=S_0\Phi(d_1).$$

Therefore

$$C=S_0\Phi(d_1)-Ke^{-rT}\Phi(d_2),$$
$$P=Ke^{-rT}\Phi(-d_2)-S_0\Phi(-d_1).$$

The put also follows from the pathwise identity
$(S_T-K)^+-(K-S_T)^+=S_T-K$, giving
$C-P=S_0-Ke^{-rT}$.

## 5. Analytical Greeks

Holding the other model and contract parameters fixed,

$$\Delta_C=\Phi(d_1),\qquad\Delta_P=\Phi(d_1)-1,$$
$$\Gamma_C=\Gamma_P=\frac{\varphi(d_1)}{S_0\sigma\sqrt T},$$
$$\mathrm{Vega}_C=\mathrm{Vega}_P=S_0\varphi(d_1)\sqrt T.$$

To derive Delta, differentiate the call formula and use
$S_0\varphi(d_1)=Ke^{-rT}\varphi(d_2)$: the density terms cancel, leaving
$\Phi(d_1)$. Differentiating Delta and using
$\partial d_1/\partial S_0=1/(S_0\sigma\sqrt T)$ gives Gamma. For Vega,
$\partial d_1/\partial\sigma-\partial d_2/\partial\sigma=\sqrt T$, so the
same density identity leaves $S_0\varphi(d_1)\sqrt T$.

Vega is per **1.00 absolute volatility**, so the approximate price change for
20% to 21% volatility is $0.01\times\mathrm{Vega}$, not $\mathrm{Vega}$.
This is a local approximation; Gamma describes second-order spot curvature.

At $T=0$, prices equal intrinsic values. At $\sigma=0$ and $T>0$, prices are
$(S_0-Ke^{-rT})^+$ and $(Ke^{-rT}-S_0)^+$. The Greeks API rejects both
boundaries: some derivatives exist away from a kink, but a uniform convention
would require additional explicit choices.

## 6. Joint GBM paths and coupled grids

For a uniform grid $t_j=jT/m$ with $\Delta t=T/m$, let $Z_{i,j}$ be independent
standard normals across paths and time increments. Exact grid paths satisfy

$$S_{i,j}=S_0\exp\left[(r-\sigma^2/2)t_j+
\sigma\sqrt{\Delta t}\sum_{k=1}^j Z_{i,k}\right].$$

The sum is along time within one path. Levels at different dates are dependent:
writing $L_t=\log(S_t/S_0)$, independent Brownian increments give

$$\operatorname{Cov}(L_s,L_t)=\sigma^2\min(s,t).$$

Correct terminal marginals alone do not validate a path simulator: drawing
independent lognormal levels at each date would destroy this joint law.

For a coarse step comprising q fine steps, set
$Z^c=q^{-1/2}\sum_{k=1}^qZ^f_k$. This is standard normal and
$\sqrt{q\Delta t}Z^c=\sqrt{\Delta t}\sum_kZ^f_k$. The drift also adds, so coarse
and fine GBM paths agree at shared dates. Setting q=m gives the equivalent
terminal shock. Floating-point summation can introduce tiny numerical differences.

Exact grid simulation does not supply the unobserved between-date trajectory.
Asian monitoring conventions and continuous barrier crossings are distinct
questions. The [Milestone 6 guide](milestone_06.md) maps
these identities to array axes, tests, memory accounting and measured output.

## 7. Discrete arithmetic Asian options

For equally weighted monitoring dates $t_j=jT/m$, excluding inception, define

$$A_m=\frac1m\sum_{j=1}^m S_{t_j},\qquad
C_m=e^{-rT}\mathbb E[(A_m-K)^+],\qquad
P_m=e^{-rT}\mathbb E[(K-A_m)^+].$$

The cashflow is paid at T. Values at earlier dates enter the average without
discounting; discount the resulting payoff once at its payment date. A sum of
correlated lognormals is not generally lognormal. Substituting the average's
mean into the ordinary Black–Scholes formula is not an exact pricing method.

For $s\leq t$, independent future GBM increments give
$\mathbb E[S_t\mid S_s]=S_s e^{r(t-s)}$, and hence

$$\mathbb E[S_s S_t]=S_0^2 e^{r(s+t)+\sigma^2 s},\qquad
\operatorname{Cov}(S_s,S_t)=S_0^2 e^{r(s+t)}
\big(e^{\sigma^2\min(s,t)}-1\big).$$

Consequently

$$\mu_m=\mathbb E[A_m]=\frac{S_0}{m}\sum_j e^{rt_j},\qquad
\operatorname{Var}(A_m)=\frac1{m^2}\sum_{j,k}
\operatorname{Cov}(S_{t_j},S_{t_k}).$$

Let $g_j=e^{rt_j}$ and $h_j=e^{\sigma^2t_j}-1$. The implementation evaluates
the double sum as

$$\operatorname{Var}(A_m)=\left(\frac{S_0}{m}\right)^2
\left[\sum_j g_j^2h_j+2\sum_k g_k\sum_{j<k}g_jh_j\right].$$

Prefix sums calculate all inner sums in O(m) operations and storage.
`expm1` evaluates $e^x-1$ accurately near zero. These moments provide validation
targets, not a complete law or an arithmetic-Asian closed-form option price.

From $(a-K)^+-(K-a)^+=a-K$,

$$C_m-P_m=e^{-rT}(\mu_m-K).$$

On shared simulated paths, the exact sample identity instead uses $\bar A_m$.
Its deviation from theoretical parity is the discounted mean-average sampling
error, with theoretical SE $e^{-rT}\sqrt{\operatorname{Var}(A_m)/N}$.

For either convex payoff $\phi$, Jensen's inequality twice gives

$$e^{-rT}\phi(\mu_m)\leq V_m\leq
\frac1m\sum_j e^{-r(T-t_j)}V_{\rm European}(t_j).$$

The upper bound pays each single-date payoff at T: a vanilla price expiring at
$t_j$ therefore needs the extra discount $e^{-r(T-t_j)}$. A noisy MC estimate
can fall outside an exact bound; uncertainty must be considered, not clipped.

Across N independent paths, define $X_i=e^{-rT}\phi(A_m^{(i)})$ and use
$\widehat V_m=\bar X$, $\mathrm{SE}=s_X/\sqrt N$. The m observations within a
path are correlated; there are N independent cashflows, not Nm.

For shared paths and a finer discrete reference M, form
$D_i=e^{-rT}[\phi(A_m^{(i)})-\phi(A_M^{(i)})]$. Then

$$\widehat\Delta_{m,M}=\bar D,\qquad
\mathrm{SE}(\bar D)=s_D/\sqrt N,\qquad
\operatorname{Var}(D)=\operatorname{Var}(X_m)+\operatorname{Var}(X_M)
-2\operatorname{Cov}(X_m,X_M).$$

With $T>0$, GBM's continuous paths imply the right-endpoint Riemann averages
converge almost surely to $A_{\rm cont}=T^{-1}\int_0^T S_t\,dt$. The positive-part
payoffs are Lipschitz and dominated by an integrable multiple of
$K+\sup_{t\leq T}S_t$, so dominated convergence also gives convergence of prices.
It gives no general pathwise monotonicity under refinement. For fixed m, exact
GBM simulation has no dynamics time-discretization bias. Changing m changes the
discrete contract; differences from finite M do not measure exact bias relative
to the continuous contract. See [Milestone 7](milestone_07.md) for the experiment.

## 8. Discrete zero-rebate up-and-out calls

For monitoring set $D_m=\{0,T/m,\ldots,T\}$, define

$$M_m=\max_{t\in D_m}S_t,\qquad
\Phi_m=(S_T-K)^+\mathbf 1_{\{M_m<B\}},\qquad
V_m=e^{-rT}\mathbb E^{\mathbb Q}[\Phi_m].$$

There are m future observations and m+1 monitored points including inception.
An observation equal to B knocks out; there is no rebate. If $S_0\ge B$ the
claim is already worthless. If $B\le K$, survival implies $S_T<B\le K$ and
the payoff is again zero. A later return below B never restores a knocked-out
claim. Rebate timing and alternative equality conventions would change the
contract, so they are deliberately not implicit options in the API.

Let $D_a\subset D_b$ with the same terminal date. Then
$M_a\le M_b$ and the nonnegative terminal call payoff is common, so

$$0\le\Phi_b\le\Phi_a\le(S_T-K)^+,\qquad
0\le V_b\le V_a\le C_{\rm European}.$$

This ordering holds for each coupled sample path and hence for finite-sample
price estimates on shared paths. Independent samples could violate the apparent
ordering through Monte Carlo noise. Larger B weakly increases the payoff on a
fixed path. For $B>K$, the surviving payoff is also bounded above by $B-K$.

The synthetic complementary cashflow
$\Phi_{\rm in}=(S_T-K)^+\mathbf 1_{\{M_m\ge B\}}$ satisfies the pathwise
identity $\Phi_{\rm out}+\Phi_{\rm in}=(S_T-K)^+$. This is an algebraic test;
it does not require a second product implementation or independent simulation.

For a continuously monitored maximum $M=\sup_{0\le t\le T}S_t$,
$M_m\le M$, so $\Phi_m\ge\Phi_{\rm cont}$ pathwise. Exact GBM values on a
grid do not reveal between-date crossings. Discrete pricing therefore gives
an upper bound in expectation for the corresponding continuously monitored
zero-rebate call, even though it is unbiased for its own discrete contract.
Adding paths reduces sampling SE, not this monitoring approximation error.

On nested uniform grids whose mesh tends to zero, continuous sample paths give
$M_m\to M$. For $T,\sigma>0$ and $B>S_0$, the GBM maximum has no atom at B,
so the survival indicators converge almost surely. Domination by the integrable
vanilla payoff then gives convergence of prices. This result does not say that
a finite grid is exact or supply an error rate for the implemented experiment.

### Exact benchmark with inception and expiry monitoring only

If $S_0<B$, $K<B$ and $T,\sigma>0$, m=1 implies

$$V_1=e^{-rT}\mathbb E[(S_T-K)\mathbf 1_{\{K<S_T<B\}}].$$

For threshold a, let $d_1(a)=[\log(S_0/a)+(r+\sigma^2/2)T]/(\sigma\sqrt T)$
and $d_2(a)=d_1(a)-\sigma\sqrt T$. The lognormal tail identities used for
European pricing give

$$\mathbb P(S_T>a)=\Phi(d_2(a)),\qquad
e^{-rT}\mathbb E[S_T\mathbf 1_{\{S_T>a\}}]=S_0\Phi(d_1(a)).$$

Subtract the tails at K and B:

$$V_1=S_0[\Phi(d_1(K))-\Phi(d_1(B))]
-Ke^{-rT}[\Phi(d_2(K))-\Phi(d_2(B))].$$

The library names this `terminal_monitored_up_and_out_price` to distinguish it
from a continuous barrier formula. Tests independently integrate the lognormal
density over (K,B). At T=0 the initial hit check precedes intrinsic value. At
sigma=0, check the deterministic terminal value as well as S0 before paying.

### Statistical interpretation

There is one discounted payoff per independent path. The existing IID estimator
and normal CI apply. For a paired monitoring comparison, use differences of
discounted path payoffs, including their covariance. In particular, for nested
grids m and M,

$$\Phi_m-\Phi_M=(S_T-K)^+
\mathbf 1_{\{M_m<B\le M_M\}}.$$

The fraction of paths satisfying this event measures missed reference hits;
the price difference also weights them by their terminal call cashflow. Paths
ending out of the money can miss a hit without contributing to the price gap.
The finest discrete grid itself misses some continuous crossings. Its paired
difference from itself has zero variance but its price still has sampling error.
See [Milestone 8](milestone_08.md) for measured results and limitations.

## 9. Antithetic variates and the independent sampling unit

Let Z be a scalar standard normal for terminal GBM or a vector of independent
standard normals for an entire path. Symmetry gives $Z\overset d=-Z$. For the
discounted payoff $X(Z)=e^{-rT}\Phi(S(Z))$, define independent pair means

$$A_i=\frac{X(Z_i)+X(-Z_i)}2,\qquad i=1,\ldots,M.$$

Both marginal cashflows have expectation V, so $\mathbb E[A_i]=V$. Pairing
does not change the target or require estimating a parameter. With variance
$v=\operatorname{Var}(X(Z))>0$ and paired-payoff correlation rho,

$$\operatorname{Var}(A_i)=\frac14(2v+2\rho v)=\frac v2(1+\rho).$$

At an equal budget of N=2M payoff evaluations,

$$\operatorname{Var}(\widehat V_{\rm plain})=\frac v{2M},\qquad
\operatorname{Var}(\widehat V_{\rm anti})=\frac{v(1+\rho)}{2M},\qquad
\mathrm{VRF}=\frac1{1+\rho}.$$

Negative correlation improves variance, positive correlation worsens it, and
zero correlation gives no variance benefit at equal payoff cost. Comparing M
plain paths with M antithetic pairs would give the latter twice the budget.
Variance reduction does not change the $M^{-1/2}$ standard-error order; it
changes the multiplicative constant unless the pair mean is exactly constant.

Estimate uncertainty using M independent pair means:

$$s_A^2=\frac1{M-1}\sum_i(A_i-\bar A)^2,\qquad
\widehat V_{\rm anti}=\bar A,\qquad
\mathrm{SE}=s_A/\sqrt M.$$

The price equals the mean of the 2M individual cashflows, but the IID standard
error from flattening them omits dependence. It can overstate or understate
uncertainty depending on the covariance sign. For M>=2 a sample identity is

$$s_A^2=\frac{s_+^2+s_-^2+2s_{+-}}4,$$

which tests the estimator independently. The low-level helper averages
undiscounted cashflows; constant discounting commutes with this average and is
applied once by the common statistics routine.

For a monotone scalar payoff f and an independent copy Z',

$$\operatorname{Cov}(f(Z),f(-Z))=
\frac12\mathbb E[(f(Z)-f(Z'))(f(-Z)-f(-Z'))]\le0.$$

The factors have opposite signs. European call and put payoffs under scalar
GBM meet this condition. An up-and-out payoff is not monotone: higher states
can improve intrinsic value or trigger knockout. Negating all path shocks
preserves each path's correct marginal law but does not guarantee negative
payoff covariance. Never negate asset prices or the risk-neutral drift.

### A constructed positive-covariance example

Set S0=100, sigma=.2, T=1, r=sigma^2/2=.02, K=100/1.1 and B=110; monitor
only inception and expiry. The terminal log drift vanishes, so ST=100*exp(.2Z).
The positive-payoff support is

$$-a<Z<a,\qquad a=\log(1.1)/.2.$$

Both f(Z) and f(-Z) vanish outside the same window. To verify the covariance,
integrate $\mu=\int f(z)\varphi(z)\,dz$, $q=\int f(z)^2\varphi(z)\,dz$ over
(-a,a), and $c=\int f(z)f(-z)\varphi(z)\,dz$ over the shared support. Then
v=q-mu^2 and covariance=c-mu^2. Gaussian quadrature yields v=29.395563,
covariance=8.682342 and rho=0.295362 for discounted payoffs. Consequently
population VRF is approximately 0.771985: antithetic variance is about 29.5%
higher at equal N. These are deterministic quadrature benchmarks, not empirical
variance ratios from repeated MC. The mean independently agrees with the
terminal-barrier analytical price. Quadrature error is also recorded.

### Replicated comparisons and uncertainty of variance ratios

For independent method runs, estimate each price-estimator variance from R
replication prices with denominator R-1. This differs from an average of the
per-run estimated variances; both are useful diagnostics. For approximately
normal run estimates, an approximate confidence interval for the population
ratio q is

$$\left[\frac{\widehat q}{F_{.975}(R_p-1,R_a-1)},
\frac{\widehat q}{F_{.025}(R_p-1,R_a-1)}\right].$$

The two variance estimates must be independent for this F construction. It is
not exact for finite-N option-price distributions and is pointwise across the
multiple scenarios. A difference of method means has SE
$\sqrt{s_p^2/R_p+s_a^2/R_a}$ in this independent design. If methods share paths,
their covariance must instead be included; this study deliberately uses separate
streams. Equal payoff budgets need not give equal runtimes. Approximate efficiency
per unit time is $\mathrm{VRF}\,t_p/t_a$, an environment-dependent measurement.

## 10. Geometric Asian controls with an independent pilot

Let monitoring dates be $t_j=jT/m$, excluding inception and including maturity.
Write $A=m^{-1}\sum_j S_{t_j}$ and $G=(\prod_j S_{t_j})^{1/m}$.
For GBM,

$$\log G=\log S_0+(r-\tfrac12\sigma^2)\bar t+
\frac{\sigma}{m}\sum_{j=1}^m W_{t_j},\qquad
\bar t=\frac{T(m+1)}{2m}.$$

Joint Gaussianity and $\operatorname{Cov}(W_s,W_t)=\min(s,t)$ give

$$\log G\sim N(\mu_G,v_G),\quad
\mu_G=\log S_0+(r-\tfrac12\sigma^2)\frac{T(m+1)}{2m},$$

$$v_G=\frac{\sigma^2}{m^2}\sum_{i,j}\min(t_i,t_j)
=\sigma^2T\frac{(m+1)(2m+1)}{6m^2}.$$

One way to evaluate the double sum is to express each $W_{t_j}$ as a sum of
independent increments. Increment k appears m-k+1 times; the variance uses
$\sum_{k=1}^m k^2=m(m+1)(2m+1)/6$. No simulation time-discretization bias is
introduced: these are the exact moments of the **discrete** geometric average.

Define $d=e^{-rT}$, $M_G=e^{\mu_G+v_G/2}$,
$d_2=(\mu_G-\log K)/\sqrt{v_G}$ and $d_1=d_2+\sqrt{v_G}$.
Integrating the lognormal payoff gives

$$C_G=d[M_G\Phi(d_1)-K\Phi(d_2)],\qquad
P_G=d[K\Phi(-d_2)-M_G\Phi(-d_1)].$$

For example, completing the square in the normal density yields
$E[G\mathbf1_{G>K}]=M_G\Phi(d_1)$ and $P(G>K)=\Phi(d_2)$.
The parity identity is $C_G-P_G=d(M_G-K)$. With one date this is ordinary
Black–Scholes; with zero variance it is discounted deterministic intrinsic value.
A continuous-geometric formula would center a different random variable and
bias the control. The geometric-average process need not itself be traded;
the formula follows from its terminal distribution, not replication of G.

For a call, set $X=(A-K)^+$, $Y=(G-K)^+$ and $\mu_Y=C_G/d$; use the analogous
put payoff and $P_G/d$ for puts. X, Y and the known mean are **undiscounted**.
They use the same path, strike, option type, dates and payment date.
For deterministic beta,

$$X_{cv}=X-\beta(Y-\mu_Y),\qquad E[X_{cv}]=E[X],$$

$$\operatorname{Var}(X_{cv})=v_X-2\beta c+\beta^2v_Y,\qquad
\beta^*=c/v_Y,\qquad
\operatorname{Var}(X_{cv};\beta^*)=v_X(1-\rho_{XY}^2).$$

A useful control has a precisely known expectation, high payoff correlation and
low evaluation cost. High correlation of underlying averages alone is insufficient;
it is the covariance of the actual payoffs that enters the variance identity.

Estimate beta from P independent pilot paths:

$$\widehat\beta=\frac{\sum_{i=1}^P(X_i-\bar X)(Y_i-\bar Y)}
{\sum_{i=1}^P(Y_i-\bar Y)^2}.$$

If the observed control variance is zero, use beta=0 and record the fallback.
Then simulate n **new** paired payoffs and report

$$\widehat V_{cv}=\frac d n\sum_{i=1}^n
[X_i-\widehat\beta(Y_i-\mu_Y)],\qquad
\widehat{SE}=d\frac{s_{cv}}{\sqrt n}.$$

Conditional on the pilot, beta is fixed and the production residuals are IID
with expectation E[X]. The estimator is therefore conditionally and, under
integrability, unconditionally unbiased. Fitting beta on the production sample
instead generally loses exact finite-sample unbiasedness. Reusing or resetting
the pilot's seed for production is not independent sampling.

By total variance, when the relevant second moments exist,

$$\operatorname{Var}(\widehat V_{cv})=
\frac{d^2}{n}\left[v_X(1-\rho_{XY}^2)
+v_Y E[(\widehat\beta-\beta^*)^2]\right].$$

There is no extra variance-of-conditional-mean term because that mean is always V.
The production residual variance estimates the appropriate conditional variance;
it is not necessary to add an unrelated beta standard error. Nevertheless an
inaccurate pilot worsens residual variance and random slopes create an
unconditional mixture. Normal CIs are approximate, not exact small-sample guarantees.

At a fixed total stock-path budget B=P+n, plain variance is $d^2v_X/B$.
Even with the oracle beta, the budget-adjusted ratio is

$$VRF_{oracle}=\frac{n/B}{1-\rho_{XY}^2}.$$

Finite pilot error reduces this oracle gain. There is no guarantee that an
estimated control improves variance, especially with weak correlation or a
small unrepresentative pilot. Choosing the pilot size is a cost-accuracy trade-off;
M10 fixes P/B=10% before observing results and does not claim this is optimal.
Both payoffs are evaluated on each pilot and production path. Equal path budgets
therefore do not mean equal payoff evaluations or runtime.

Using an erroneous control mean $\mu_Y+\delta$ shifts the conditional price
expectation by $d\widehat\beta\delta$. A small formula or discounting error can
thus dominate a very small reported SE. Reduced sampling uncertainty does not
remove model error, monitoring differences or floating-point error.

## 11. Central finite-difference Greeks and common random numbers

Hold strike, rate, maturity, product convention and monitoring dates fixed.
Delta and Gamma differentiate with respect to S0; Vega differentiates with
respect to decimal volatility sigma. Vega/100 is sensitivity to one volatility
percentage point. For sufficiently smooth option values,

$$D_hV=\frac{V(S_0+h)-V(S_0-h)}{2h}
=V_S+\frac{h^2}{6}V_{SSS}+O(h^4),$$
$$G_hV=\frac{V(S_0+h)-2V(S_0)+V(S_0-h)}{h^2}
=V_{SS}+\frac{h^2}{12}V_{SSSS}+O(h^4),$$
$$\mathcal V_h=\frac{V(\sigma+h)-V(\sigma-h)}{2h}
=V_\sigma+\frac{h^2}{6}V_{\sigma\sigma\sigma}+O(h^4).$$

Taylor expansions give these expressions by cancellation of odd/even powers.
The bias is O(h²), assuming the necessary higher derivatives exist locally.
The price can be smooth even when its payoff has a kink. These expansions
need not apply at a discontinuity, such as certain inception-barrier boundaries.
The implemented boundary convention requires T>0, sigma>0, 0<hS<S0 and
0<hSigma<sigma. There are no automatic one-sided differences or clipped bumps.

Let Phi_i+, Phi_i0 and Phi_i- be **undiscounted payoffs** generated by the
same standard-normal vector Zi for the perturbed models. For European options
Zi is scalar. For Asians and barriers it is the whole m-step shock row.
Do not regenerate independent Zi inside each bumped valuation when using CRN.
Discount d=exp(-rT) is unchanged by spot or volatility bumps. Define

$$D_i=\frac{\Phi_i^+-\Phi_i^-}{2h_S},\qquad
G_i=\frac{(\Phi_i^+-\Phi_i^0)+(\Phi_i^--\Phi_i^0)}{h_S^2},$$
$$U_i=\frac{\Phi_i^{\sigma+}-\Phi_i^{\sigma-}}{2h_\sigma}.$$

For any of these samples H, the estimate and its sampling uncertainty are

$$\widehat g_h=\frac dN\sum_{i=1}^N H_i,\qquad
\widehat{SE}(\widehat g_h)=\frac{d s_H}{\sqrt N}.$$

Distinct rows are independent. Legs within a row are dependent under CRN,
and that covariance is captured automatically by computing sample variance
**after** differencing. The expectation is the finite-bump price difference,
not in general the true derivative. A normal confidence interval quantifies
sampling error around g_h; it does not account for finite-difference bias.

For Delta, the variance identity is

$$\operatorname{Var}(\widehat\Delta_h)=\frac{d^2}{4h^2N}
\left[\operatorname{Var}(\Phi^+)+\operatorname{Var}(\Phi^-)
-2\operatorname{Cov}(\Phi^+,\Phi^-)\right].$$

With independent legs the covariance is zero. With shared shocks, nearby
vanilla payoffs move together and much of their noise cancels. Gamma uses
weights a=(1,-2,1) on (plus,base,minus), giving
$d^2 a^\top\Sigma a/(Nh^4)$. Merely adding three price variances would omit
all covariance terms and give the wrong CRN uncertainty. Independent Gamma
has numerator Var(Phi+)+4 Var(Phi0)+Var(Phi-).

### Bump size is estimator-dependent

Without shared shocks, small-h first-derivative variance is typically
O(1/(N h²)); second-derivative variance is O(1/(N h⁴)). Thus an unbiased
price estimator alone is not enough to obtain a stable numerical derivative.

For continuous vanilla payoffs under GBM, CRN Delta and Vega difference
samples approach finite-variance path derivatives as h decreases. Their
sampling SE can plateau rather than diverge. The statement "a smaller bump
always produces more Monte Carlo noise" is therefore false without specifying
the estimator and payoff. Pathwise estimators themselves are introduced in M12.

CRN Gamma of a call is different. Write ST=S0*R(Z). As a function of S0,
the path payoff (S0*R-K)+ is piecewise linear. A central second difference is
zero away from the strike-switching window, but of order 1/h inside a window
whose probability is O(h). Its second moment is O(1/h), so its sampling variance
is typically O(1/(N h)). For tiny bumps, too few paths cross the payoff kink;
a run can miss the important events and report misleadingly small uncertainty.
The pointwise second derivative is zero almost surely, but expected Gamma is
positive: exchanging second differentiation and expectation this way fails.

For discontinuous barrier payoffs, CRN does not smooth the jump. A bump can
change knockout status; first differences may already have rare large terms.
These finite-difference APIs estimate sensitivities of the **specified discrete
contract**, with barrier level and grid fixed, not continuous-barrier Greeks.

### Sampling error, truncation and rounding

Decompose the realized error as

$$\widehat g_h-g=(\widehat g_h-g_h)+(g_h-g).$$

The first term is sampling error; the second is deterministic finite-difference
bias. Add floating-point and any model/monitoring errors separately. In the
baseline study g_h is computed using analytical BS prices, at moderate bumps
where subtraction is sufficiently accurate. For tiny h, even differencing
closed-form prices loses digits. Rounding in a second difference can be
amplified roughly by 1/h². Resolved input bumps are necessary but not sufficient
for an accurate output. More paths do not repair deterministic roundoff.

For independent replications r=1,...,R, empirical bias is mean(estimate)-g,
unbiased empirical variance uses R-1, and empirical RMSE uses R. Consequently

$$\widehat{RMSE}^2=\widehat{Bias}^2+\frac{R-1}{R}s_{runs}^2.$$

The population identity RMSE²=Bias²+Variance is exact; substituting an unbiased
sample variance requires this finite-R factor to recover empirical MSE exactly.
Reported study bias contains Monte Carlo uncertainty; g_h-g is the separate
known truncation bias. Coverage of g_h and of g are reported separately.

A fixed nonzero bump leaves a bias floor as N increases: sampling SE can scale
as N^-1/2 while total error need not converge to zero. Balancing h and N requires
knowledge of the payoff and estimator. There is no universally optimal bump,
and a single noisy minimum across a tested grid is not sufficient to choose one.

## 12. Pathwise European Delta and differentiation under expectation

For exact GBM at T>0 and sigma>0, write

$$S_T=S_0R,\qquad R=\exp[(r-\sigma^2/2)T+\sigma\sqrt T Z].$$

With Z fixed, R is independent of the spot parameter, so dST/dS0=R=ST/S0.
For f(x)=(x-K)+, f'(x)=1{x>K} away from the kink. Therefore

$$\Delta_C=e^{-rT}E\left[R\mathbf1_{S_0R>K}\right],\qquad
\widehat\Delta_C=\frac{e^{-rT}}{N}\sum_{i=1}^N
\frac{S_T^{(i)}}{S_0}\mathbf1_{S_T^{(i)}>K}.$$

The corresponding put formula is

$$\Delta_P=-e^{-rT}E\left[R\mathbf1_{S_0R<K}\right].$$

### Why the interchange is valid here

The GBM terminal distribution is continuous, so P(ST=K)=0. This gives almost
sure differentiability of the payoff with respect to S0, but **alone it is not
a sufficient interchange argument**. A call or put payoff is 1-Lipschitz:

$$\left|\frac{f((S_0+h)R)-f(S_0R)}h\right|\le R.$$

Since E[R]=exp(rT)<infinity, dominated convergence justifies taking the limit
inside the expectation. Moreover R has a finite second moment, giving finite
variance and the usual IID CLT for the sample mean. In exact arithmetic the
pathwise estimator is unbiased for Delta itself: there is no bump parameter
or finite-difference truncation bias. Model error and numerical rounding remain.
The finite-sample normal interval is still approximate.

At ST=K the code chooses zero for either derivative. This convention changes
no expectation in the stipulated continuous distribution, but is not a general
rule at an atom. Boundary T=0 and sigma=0 cases remain outside the Greek APIs.

### Exact variance benchmark

Let a=sigma*sqrt(T), d1=[log(S0/K)+(r+sigma²/2)T]/a and
D=exp(-rT) R 1{ST>K}. Completing the square in the normal density yields

$$E[D]=\Phi(d_1),\qquad E[D^2]=e^{\sigma^2T}\Phi(d_1+a),$$
$$\operatorname{Var}(D)=e^{\sigma^2T}\Phi(d_1+a)-\Phi(d_1)^2.$$

For the put, replace the mean by -Phi(-d1) and the second moment by
exp(sigma²T)*Phi(-d1-a). Thus both the expected Delta and its estimator variance
are available independently of simulation. At the baseline S0=K=100, r=.05,
sigma=.2, T=1, call sample variance is 0.3322153569. The exact sampling SE for
N independent paths is sqrt(0.3322153569/N).

### Relation to common-random-number finite differences

For a fixed R, the central difference of f(SR) tends almost surely to its
pathwise first derivative. With sufficient second-moment domination this also
explains why their variances converge as h decreases. For call spot Delta,
a finite bump replaces the sharp exercise switch by a transition over the
strike-crossing region. That smoothing can reduce variance at fixed N while
introducing bias. Pathwise is not automatically the minimum-variance estimator.

Our basic FD implementation transforms the shared shock vector twice for the
perturbed spots; PW transforms it once. We report both equal independent-row
and equal terminal-evaluation budgets. These are implementation-level counts,
not equal runtime or fundamental complexity bounds: an optimized spot-FD
routine could scale a single simulated terminal vector. Such optimization is
not included in the measured comparison.

For a paired diagnostic, use the same Zi for PW and FD and define Hi=Di,FD-Di,PW.
Its SE comes from the variance of H, including cross-method covariance. Its
expectation is exactly the FD bias because PW has expectation Delta. Repeated
runs are independent, but different bumps within one shared-path run are
correlated and cannot be pooled as additional replications. Separate independent
method streams are used for the experiment's variance-ratio F intervals.

### Why discontinuities and second derivatives are different

For a cash-or-nothing digital, differentiating 1{S0R>K} along a fixed path gives
zero almost surely, whereas its price exp(-rT)*Phi(d2) has spot derivative
exp(-rT)*phi(d2)/(S0*sigma*sqrt(T)), which is positive. Almost-sure
differentiability has not justified the expectation interchange: crossing the
moving boundary contributes to the derivative. The payoff is not Lipschitz,
and the needed uniform integrable bound on difference quotients fails.

The same warning applies to differentiating the pathwise call Delta again:
R is spot-independent and the indicator is locally constant almost surely,
so a naive second path derivative would be zero. It misses true positive
Gamma from the moving exercise boundary. Barrier knockout indicators create
related problems, even on discrete monitoring grids. The implementation thus
supports European Delta only; it does not silently apply this formula to
digitals, barriers, Gamma, Vega or other models. Likelihood-ratio estimators
are implemented for terminal vanilla/digital Delta in section 14; they do not
make the naive pathwise differentiation valid.


## 13. Replicated robustness and rare-event diagnostics

For R independent estimates U_r of a known target u, use

$$\widehat b=\bar U-u,\quad s_U^2=\frac{1}{R-1}\sum_r(U_r-\bar U)^2,\quad
\widehat{RMSE}^2=\frac1R\sum_r(U_r-u)^2
=\widehat b^2+\frac{R-1}{R}s_U^2.$$

The empirical bias is noisy: its approximate standard error is s_U/sqrt(R).
An FD method estimates the analytical finite-bump target u_h; u_h-u is its
known deterministic bias here. Coverage of u_h and coverage of u answer different
questions. Empirical estimator variance uses run-level estimates, not path-level
payoff variance. Units must match before forming ratios.

For an interval I_r, measured coverage is R^-1 sum 1{u in I_r}. Independent runs
make these indicators Bernoulli; the Wilson interval quantifies uncertainty in
that coverage probability. It does not fix the original intervals. The normal
Greek interval and the Wilson interval for its coverage are different objects.
With many cells, pointwise intervals are not simultaneous confidence statements.
Zero-SE runs remain in both coverage and RMSE: deleting them selects on outcomes.

For call pathwise Delta, a contribution is positive iff S_T>K. Thus

$$p=\mathbb Q(S_T>K)=\Phi(d_2),\qquad
\mathbb P(\text{no active path among }n)=(1-p)^n.$$

If no path is active, the empirical Delta and its SE are both zero even though
true Delta is positive. Finite variance and an asymptotic CLT do not guarantee
useful normal intervals at a fixed n with np much smaller than one. For the M13
rare scenario, n=20,000 and np=0.22565, giving a no-active probability 0.7980.
This can be predicted without simulating any path. Compute it stably as
`exp(n * log1p(-p))`.

For positive monitored stock levels, AM-GM implies A>=G, hence for the same call
strike and dates, (A-K)^+ >= (G-K)^+ and V_arithmetic >= V_geometric.
If an arithmetic price interval has upper endpoint below the exact geometric
price, it certainly misses the true arithmetic price. The observed fraction of
such intervals is a **lower bound on empirical noncoverage**, not a measurement
of full coverage. Bias and RMSE remain unknown for the arithmetic price.

The M13 variance comparison resamples the R estimator values, independently for
each method, with replacement. In each of 2,000 bootstrap repetitions it forms
s_reference^2/s_candidate^2. The 2.5% and 97.5% quantiles give an approximate
percentile interval. Independence is appropriate because the compared methods
use independent streams. Paired methods would require joint resampling instead.
Zero-variance bootstrap draws are flagged and excluded from quantiles; if any
are excluded, the interval is conditional on a valid ratio. Degenerate original
samples produce an undefined comparison, not an infinite improvement claim.

This is not an exact distribution-free interval for a variance ratio. It is
sensitive to the number of runs and tails, and resampling cannot invent a rare
event absent from the data. Even 100% valid bootstrap draws do not establish tail
reliability. Normal bias intervals also require an adequate run-level CLT.

A further issue arises with an unbounded random pilot slope. Conditional on each
independent pilot, E[U_cv | pilot]=V. Unconditional unbiasedness additionally
requires integrability; the total-variance identity in section 10 requires the
relevant second moments, including E[beta_hat^2], to exist. A small *nonzero*
pilot control variance can give a very large slope. The zero-variance fallback
alone does not bound it. Finite observed slopes over R runs do not establish
these population moment assumptions, and M13 does not prove or disprove their
existence for its particular controls. Its CV variance ratios are empirical
summaries with this limitation. A bounded coefficient chosen using only the
independent pilot would preserve conditional centering and control its moments;
that is a possible separately validated extension, not a change made in M13.


## 14. Likelihood-ratio spot Delta

Let X have density p_theta(x) on a support independent of theta. For a payoff f
with no explicit parameter dependence, differentiation under the integral gives

$$\frac{d}{d\theta}E_\theta[f(X)]
=\int f(x)\partial_\theta p_\theta(x)\,dx
=E_\theta[f(X)\,\partial_\theta\log p_\theta(X)].$$

The **score** is the logarithmic density derivative. This requires a valid
interchange of derivative and integration, not differentiability of f. With an
explicit f_theta, add E_theta[partial_theta f_theta(X)]. Parameter-dependent
support or discount factors can add other terms; do not ignore them.

For GBM let a=sigma*sqrt(T)>0 and m=(r-sigma²/2)T. At fixed terminal value s>0,

$$p(s;S_0)=\frac{1}{sa\sqrt{2\pi}}
\exp\left[-\frac{(\log s-\log S_0-m)^2}{2a^2}\right],\qquad
z(s;S_0)=\frac{\log s-\log S_0-m}{a}.$$

The density prefactor contains s, not S0. Since partial_S0 z=-1/(S0*a),

$$\partial_{S_0}\log p(s;S_0)=\frac{z(s;S_0)}{S_0a}.$$

Only **after differentiating at fixed s** do we evaluate at the simulated S_T,
so z(S_T;S0)=Z. Differentiating the normal sampling density with respect to spot
would give zero; that is a different calculation. Retaining original Z avoids
recovering it by a potentially unstable log subtraction.

$$\boxed{\widehat\Delta_{LR}=\frac{e^{-rT}}N\sum_{i=1}^N
f(S_T^{(i)})\frac{Z_i}{S_0\sigma\sqrt T}.}$$

For the call, f(s)=(s-K)^+; for the put, (K-s)^+; for the digital call,
f(s)=q*1{s>K}. Discounting is constant under a spot derivative. The score has
mean zero and variance 1/(S0²*sigma²*T). The sample mean uses the variance of
**payoff times score**, which is not the payoff variance times the score variance.
They are dependent. No finite bump is used.

For positive T and sigma and spot restricted to a compact neighborhood inside
(0,infinity), derivatives of this lognormal density admit integrable envelopes
when multiplied by a bounded payoff or one growing at most linearly in s.
Normal tails dominate polynomial factors in log(s), and lognormal moments of
all fixed real orders exist. This justifies the interchange here and finite
second moments of these LR samples. It does not guarantee a good finite-N CLT
approximation, especially for rare exercise. T=0 and sigma=0 are excluded.

### Digital benchmark and why pathwise fails

With d2=(log(S0/K)+m)/a,

$$V_{digital,call}=q e^{-rT}\Phi(d_2),\qquad
\Delta_{digital,call}=\frac{q e^{-rT}\phi(d_2)}{S_0a}.$$

Put price is q*exp(-rT)*Phi(-d2), and put Delta is minus call Delta. The code
assigns equality to the put, making call+put=q even for deterministic inputs.
Differentiating 1{S0*R>K} along a fixed path gives zero almost surely. That misses
the moving-boundary probability and is not a valid interchange with expectation.
LR differentiates the density instead and recovers the positive call Delta.

### Exact moment checks

Let c=-d2, and define M_j(b,c)=E[Z^j exp(bZ)1{Z>c}]. Completing the square and
differentiating in b gives

$$M_0=e^{b^2/2}\Phi(b-c),\quad
M_1=e^{b^2/2}[b\Phi(b-c)+\phi(c-b)],$$
$$M_2=e^{b^2/2}[(1+b^2)\Phi(b-c)+(b+c)\phi(c-b)].$$

For lower-tail moments replace Phi(b-c) with Phi(c-b) and change the density
terms' sign. For a vanilla call, with F=S0*exp(m), d=exp(-rT),

$$E[D_{LR}]=\frac d{S_0a}[F M_1(a,c)-K M_1(0,c)],$$
$$E[D_{LR}^2]=\frac{d^2}{S_0^2a^2}
[F^2M_2(2a,c)-2FKM_2(a,c)+K^2M_2(0,c)].$$

For puts, use the lower tail, reverse the first-moment bracket's sign, and keep
the squared expression. For digital calls, moments simplify to

$$E[D_{LR}]=\frac{dq}{S_0a}\phi(c),\quad
E[D_{LR}^2]=\left(\frac{dq}{S_0a}\right)^2[\Phi(-c)+c\phi(c)].$$

Subtract mean squared for sample variance and divide by N for estimator variance.
The experiment implements unit digital cash q=1. Tests compare these expressions
against direct quadrature of the implementation. Float64 cancellation can limit
extreme-tail moment arithmetic; this is not a universal tail-accurate library.

### Digital finite differences and missing events

With CRN and spot bump h, the digital call difference sample is d*q/(2h) times
an indicator that only the bumped-up path exercises. Its probability is

$$p_h=\Phi(z_+)-\Phi(z_-),\quad
z_-=\frac{\log(K/(S_0+h))-m}{a},\quad
z_+=\frac{\log(K/(S_0-h))-m}{a}.$$

Its mean is d*q*p_h/(2h), equal to the analytical central price difference,
and variance is (d*q/(2h))²*p_h*(1-p_h). At fixed regular parameters, p_h=O(h),
so variance grows as O(1/h) and SE as O((Nh)^-1/2). CRN helps but does not prevent
this small-bump divergence for a digital. Put has the opposite mean and same
variance. For R runs of N rows, no crossing anywhere has probability
(1-p_h)^(RN). In M14's rare h=.01 case that probability is 60.51%: an entirely
zero empirical study is plausible even with 200 replications. Known population
variance exposes the failure that empirical variance and bootstrap alone miss.

## 15. Heston dynamics, full truncation and an independent reference

Under a selected risk-neutral measure, with constant rate and no dividends:

$$dS_t=rS_tdt+\sqrt{v_t}S_t dW_t^S,\qquad
 dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\qquad
 d\langle W^S,W^v\rangle_t=\rho\,dt.$$

Here v0 and theta are variances (0.04 corresponds to 20% volatility), kappa is
mean-reversion speed, xi is volatility of variance, and rho is instantaneous
Brownian correlation. Heston is generally incomplete: no-arbitrage alone does
not determine volatility risk compensation. Import risk-neutral calibrated
parameters; physical estimates cannot be relabeled Q without an assumption.

For independent standard normals Z1,Z2, use ZS=Z1 and
Zv=rho*Z1+sqrt(1-rho²)*Z2. Then Var(Zv)=rho²+1-rho²=1 and
Cov(ZS,Zv)=rho. This is not a statement about Corr(ST,vT).

### The auxiliary variance must survive a negative update

Euler's Gaussian increment can make its variance approximation negative.
Full truncation uses a raw auxiliary state, initially tilde-v0=v0, and its
positive part x+=max(x,0):

$$\widetilde v_{n+1}=\widetilde v_n+
\kappa(\theta-\widetilde v_n^+)\Delta t+
\xi\sqrt{\widetilde v_n^+\Delta t}\,Z_n^v,$$
$$S_{n+1}=S_n\exp\!\left[(r-\tfrac12\widetilde v_n^+)\Delta t+
\sqrt{\widetilde v_n^+\Delta t}\,Z_n^S\right].$$

Carry the raw next variance, including a negative value; truncation acts inside
the drift/diffusion coefficients. Replacing the carried state with zero changes
the scheme. Both updates use the **old** effective variance. Conditional on the
past, the exponential stock increment has mean exp(r*dt), so the discounted
stock is a discrete-time martingale. A stock update using the new variance
would not inherit that conditional-normal argument.

The CIR variance process remains nonnegative. For positive initial variance
and xi>0, the Feller condition 2*kappa*theta>=xi² prevents hitting zero;
failure permits boundary access, not negative physical variance. It does not
invalidate Heston parameters. Finite-step Euler auxiliary states can be negative
even when Feller holds. `feller_margin` and the negative-update fraction are
diagnostics, not rejection criteria or measures of model validity.

### Exact deterministic-variance limit

At xi=0, variance solves an ODE:

$$v(t)=\theta+(v_0-\theta)e^{-\kappa t},\qquad
I(T)=\theta T+(v_0-\theta)\frac{1-e^{-\kappa T}}{\kappa}.$$

The terminal law is lognormal with integrated variance I(T), giving a European
Black–Scholes reference with sigma_eff=sqrt(I(T)/T). If v0=theta, the simulator
matches exact GBM at grid dates pathwise. If v0 differs from theta, Euler still
approximates the deterministic variance ODE and retains integration error.
For stochastic variance, E[vT]=theta+(v0-theta)*exp(-kappa*T), a separate moment
check; the truncated discrete approximation need not satisfy it exactly.

### Affine characteristic function

An independent European benchmark separates sampling error from integration
bias. For centered log return X=log(ST/(S0*exp(rT))), set
phi(u,T)=E[exp(i*u*X)]=exp(C(T,u)+D(T,u)*v0). The affine generator gives

$$D'=\tfrac12\xi^2D^2+(\rho\xi iu-\kappa)D-\tfrac12(u^2+iu),
\qquad C'=\kappa\theta D,\qquad C(0)=D(0)=0.$$

For xi>0 define b=kappa-rho*xi*i*u,
d=sqrt(b²+xi²*(u²+i*u)) with nonnegative real part, and g=(b-d)/(b+d). Then

$$D=\frac{b-d}{\xi^2}\frac{1-e^{-dT}}{1-ge^{-dT}},$$
$$C=\frac{\kappa\theta}{\xi^2}\left[(b-d)T-
2\log\!\left(\frac{1-ge^{-dT}}{1-g}\right)\right].$$

Use the decaying exponential, `expm1`, and b-d=-xi²*(u²+i*u)/(b+d) to avoid
subtraction loss. With w=g*(1-exp(-dT))/(1-g), compute the real logarithm as
0.5*log1p(2*Re(w)+Re(w)²+Im(w)²) and the imaginary part via atan2.
This avoids loss of a tiny real part before division by xi². The xi=0 limit is
handled exactly, not by silently replacing a small positive xi with zero.
Tests integrate the Riccati ODE independently with `solve_ivp`.

### Fourier price and its error budget

With k=log(S0/K)+rT, inversion on the half-unit shifted contour gives

$$C_0=S_0-\frac{\sqrt{S_0K}e^{-rT/2}}{\pi}
\int_0^\infty\frac{\Re[e^{iuk}\phi(u-i/2,T)]}{u^2+1/4}\,du.$$

To see the payoff identity, use integral_0^infinity cos(u*x)/(u²+1/4) du
=pi*exp(-abs(x)/2), together with
min(ST,K)=sqrt(ST*K)*exp(-abs(log(ST/K))/2) and
(ST-K)+=ST-min(ST,K). Discounting and the stock martingale give the formula.
Put follows from C-P=S0-K*exp(-rT).

`heston_european_reference` integrates to a finite cutoff with SciPy quadrature.
Its error field estimates finite-interval quadrature error, not the omitted
Fourier tail, branch/roundoff error, or a confidence interval. M15 compares
cutoffs 200 and 400, tighter tolerances, Riccati solutions, parity, bounds and
deterministic limits. This establishes evidence for the tested parameter range,
not a universal extreme-parameter guarantee. The stochastic Fourier routine
requires abs(rho)<1; simulation permits the degenerate endpoints. The exposed
characteristic function restricts Im(u) to [-0.5,0].

### Integration, monitoring and sampling are distinct

Fix the Asian/barrier contract at 12 equally spaced future observations while
varying integration steps 12,24,48,96,192,384. Stock states are saved only on
contractual dates (plus inception); an Asian excludes inception and a barrier
includes it. Coarse shocks sum fine independent shocks divided by sqrt(block).
This preserves each grid's law and enables paired payoff differences.
Increasing integration frequency does not impose pathwise barrier monotonicity.
Increasing monitoring dates would change the discrete contract instead.

Normal Monte Carlo intervals cover a discretized expectation approximately;
they exclude integration bias and model error. A difference from 384 steps
measures grid sensitivity, not bias against an exact exotic price. No Heston
continuous-barrier crossing correction is implemented.

Sources: [Heston (1993), original paper](https://wwwf.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf);
[Lord, Koekkoek and van Dijk, full-truncation study, working paper](https://papers.tinbergen.nl/06046.pdf).
The implementation, Fourier payoff derivation and independent numerical checks
are included so the benchmark can be inspected without a finance library.

## 16. Final evidence: independent units, costs and repeated-study uncertainty

An estimator's sample variance must refer to its independent units. A plain
payoff, antithetic pair mean, pilot-conditioned control residual and CRN Greek
row are different units. Compare the variances of their final estimates at a
stated cost, not their raw sample variances.

For R independent estimates with common configuration, define grand mean m and
run sample variance s² with denominator R-1. The SE of m is sqrt(s²/R), whereas
the uncertainty of a single run is approximately sqrt(s²). These answer different
questions. When exact target V is available:

$$\widehat{\mathrm{RMSE}}^2=\frac1R\sum_j(\widehat V_j-V)^2
=(m-V)^2+\frac{R-1}{R}s^2.$$

For two independent estimator studies, VRF=s_ref²/s_candidate². Resample whole
run estimates independently within each method to approximate its sampling
uncertainty. Percentile bootstrap intervals are pointwise and can fail for
unobserved tails or degenerate empirical distributions. Zero-variance cases
remain flagged. Comparing many intervals does not produce simultaneous 95%
coverage. A realized empirical bias is not the estimator's population bias.

At equal paths, a control with a pilot has fewer production residuals and more
payoff calculations than plain MC. At equal terminal evaluations, central FD
Delta has half the rows of pathwise/LR Delta and Gamma uses one-third the rows.
Explicitly include these counts. Conditional control SE and empirical variance
across pilots need not agree closely in a finite replication study.

For independent calls of cost t, averaging approximately tau/t runs gives
variance approximately s²*t/tau. The descriptive efficiency ratio is therefore
VRF*t_ref/t_candidate. Finite-difference bias does not vanish by repeating calls;
variance efficiency alone does not rank RMSE. Runtime variability and machine
conditions prevent interpreting that ratio as a certified speedup.

The [M16 guide](milestone_16.md) states all budgets, actual outcomes and the
provenance rules for historical versus fresh experiments.

## References

The [Milestone 5 guide](milestone_05.md) derives the log-log slope, repeated-run
RMSE/variance identity, empirical bias uncertainty and Wilson interval for
coverage. It explains which independent sampling units each statistic uses.

The [Milestone 4 guide](milestone_04.md) derives validation diagnostics and the
correlation between nested sample means. It distinguishes realized pricing error
from standard error, repeated-run bias and interval coverage.

The [Milestone 3 guide](milestone_03.md) derives the sample-variance denominator,
studentized CLT interval, discount scaling and shared-sample put–call parity,
and maps each formula to the implemented result fields.

The terminal-simulation validation in [Milestone 2](milestone_02.md) derives
lognormal moments and the sampling uncertainty of means and sample variances.

- Leonid Kogan, MIT 15.450, [Stochastic Calculus and Option Pricing](https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/0d1260b891a96241316d883d4f5bfaec_MIT15_450F10_lec02.pdf): background on Itô calculus, replication and Black–Scholes.
- SciPy, [`scipy.stats.norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html): the CDF and density primitives used by the implementation.
- NumPy, [Generator](https://numpy.org/doc/stable/reference/random/generator.html) and [random compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html): explicit state, construction and reproducibility limits.

The algebra above is presented explicitly so the implementation can be derived
and defended without relying on a pricing package.
