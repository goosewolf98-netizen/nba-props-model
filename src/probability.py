import math
import numpy as np
from scipy.stats import poisson

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def calculate_probs(stat: str, projection: float, line: float, rmse: float = None) -> dict:
    """
    Calculate the probability of going Over/Under a line given a projection.

    Args:
        stat: The statistic name (e.g., 'pts', 'reb', 'ast').
        projection: The projected value (mean).
        line: The betting line.
        rmse: Root Mean Squared Error of the model (standard deviation for Normal dist).
              Required for Normal distribution stats (e.g., 'pts').
              Ignored for Poisson distribution stats (variance = mean).

    Returns:
        dict: {'p_over': float, 'p_under': float, 'rmse_used': float}
    """
    stat = str(stat).strip().lower()

    # Discrete stats use Poisson
    # PRA is often high enough to be Normal, but existing logic used Poisson.
    # Let's support both or stick to Poisson for consistency with prior code if preferred.
    # However, PRA variance != mean usually (covariance of P, R, A).
    # But for now, we simply consolidate existing logic.
    discrete_stats = ["reb", "ast", "stl", "blk", "tpm", "pra"]

    if stat in discrete_stats:
        if projection <= 0:
            return {'p_over': 0.0, 'p_under': 1.0, 'rmse_used': 0.0}

        # Poisson CDF(k, lambda) is prob X <= k
        # P(Over) = P(X > line) = 1 - P(X <= line)
        # For integer lines, P(X > 25) is 1 - CDF(25).
        # For float lines like 25.5, CDF(25.5) is same as CDF(25).
        # We usually want P(X > line).
        # If line is 25.5, we want P(X >= 26).
        # 1 - CDF(25.5) gives P(X > 25.5) -> P(X >= 26). Correct.

        p_over = 1.0 - poisson.cdf(line, projection)

        # P(Under) = P(X < line).
        # If line is 25.5, P(X <= 25).
        # CDF(25.5) gives P(X <= 25). Correct.
        p_under = poisson.cdf(line, projection)

        # Note: Poisson implies Variance = Mean, so StdDev (RMSE proxy) = sqrt(Mean)
        rmse_used = math.sqrt(projection)

    else:
        # Normal distribution (Points, or undefined stats default to Normal)
        if rmse is None or rmse <= 1e-9:
            # Fallback if no RMSE provided, though caller should usually provide it.
            rmse = 1.0

        z = (line - projection) / rmse
        p_under = normal_cdf(z)
        p_over = 1.0 - p_under
        rmse_used = rmse

    return {
        'p_over': float(p_over),
        'p_under': float(p_under),
        'rmse_used': float(rmse_used)
    }
