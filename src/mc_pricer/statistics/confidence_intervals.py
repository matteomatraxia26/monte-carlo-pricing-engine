"""Normal-approximation intervals for an estimated mean."""

from math import isfinite
from numbers import Integral

from scipy.stats import norm

from mc_pricer._validation import finite_real


def validate_confidence_level(confidence_level: float) -> float:
    """Validate a finite real probability strictly between zero and one."""
    level = finite_real("confidence_level", confidence_level)
    if not 0 < level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    return level


def normal_confidence_interval(
    estimate: float, standard_error: float, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Return estimate +/- z * SE using a central normal critical value.

    This is a CLT-based, approximate interval; it is not an exact finite-sample
    coverage guarantee for skewed payoffs. Endpoints are not clipped at zero.
    An observed SE of zero gives a point interval, which alone does not prove
    absence of uncertainty (e.g. an unsampled rare event).
    """
    estimate = finite_real("estimate", estimate)
    standard_error = finite_real("standard_error", standard_error)
    if standard_error < 0:
        raise ValueError("standard_error must be nonnegative")
    level = validate_confidence_level(confidence_level)
    # isf avoids rounding (1 + level)/2 to exactly 1 near the upper boundary.
    critical_value = float(norm.isf((1.0 - level) / 2.0))
    half_width = critical_value * standard_error
    lower, upper = estimate - half_width, estimate + half_width
    if not isfinite(lower) or not isfinite(upper):
        raise FloatingPointError("confidence interval exceeds finite float64 range")
    return lower, upper


def binomial_wilson_interval(
    successes: int, trials: int, confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Approximate score interval for a binomial success probability.

    Trials must be independent Bernoulli observations with the same probability.
    Here each trial can be an independent pricing run whose CI covers a target.
    This interval quantifies uncertainty of the *coverage rate*, not the price.
    It has nonzero width even when successes is zero or equals trials.
    """
    for name, value in (("successes", successes), ("trials", trials)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require trials >= 1 and 0 <= successes <= trials")
    successes, trials = int(successes), int(trials)
    level = validate_confidence_level(confidence_level)
    z = float(norm.isf((1.0 - level) / 2.0))
    proportion = successes / trials
    z2_over_n = z * z / trials
    denominator = 1.0 + z2_over_n
    center = (proportion + 0.5 * z2_over_n) / denominator
    half_width = z * (proportion * (1.0 - proportion) / trials
                      + z * z / (4.0 * trials * trials)) ** 0.5 / denominator
    # Remove possible rounding excursions outside the probability domain.
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == trials else min(1.0, center + half_width)
    return lower, upper
