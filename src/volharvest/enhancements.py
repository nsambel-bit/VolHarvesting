"""Strategy enhancements motivated by the real-data drawdown analysis.

All features are causal: the value on rebalance day t uses data up to and including t.

- residual_momentum: 12-1 month residual momentum (a genuine slow trend signal)
- jump_mask: flags stocks that had an extreme residual day recently (earnings-type jumps),
  so the reversal signals don't fade them
- guard_runaway: forbids the reversal sleeve from betting against extreme persistent trends
- blend: combines sleeves by cross-sectionally standardized alpha
"""
import numpy as np


def xs_z(a: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Cross-sectional z-score along the last axis; NaN -> 0."""
    m = np.nanmean(a, axis=-1, keepdims=True)
    s = np.nanstd(a, axis=-1, keepdims=True)
    return np.clip(np.nan_to_num((a - m) / np.where(s > 0, s, 1)), -clip, clip)


def residual_momentum(e: np.ndarray, vol: np.ndarray, dates: np.ndarray,
                      lookback: int = 252, skip: int = 21) -> np.ndarray:
    """Sum of residual returns over days (t-lookback, t-skip], scaled by residual vol at t.

    Shape (len(dates), N). NaN where history is insufficient.
    """
    cs = np.concatenate([np.zeros((1, e.shape[1])), np.cumsum(e, 0)])
    out = np.full((len(dates), e.shape[1]), np.nan)
    ok = dates + 1 - lookback >= 0
    d = dates[ok]
    out[ok] = (cs[d + 1 - skip] - cs[d + 1 - lookback]) / (vol[d] * np.sqrt(lookback - skip))
    return out


def jump_mask(z: np.ndarray, dates: np.ndarray, threshold: float = 4.0, hold: int = 20) -> np.ndarray:
    """True where the stock had |standardized residual| > threshold in days [t-hold+1, t]."""
    big = (np.abs(np.nan_to_num(z)) > threshold).astype(float)
    cs = np.concatenate([np.zeros((1, z.shape[1])), np.cumsum(big, 0)])
    lo = np.maximum(dates + 1 - hold, 0)
    return (cs[dates + 1] - cs[lo]) > 0


def guard_runaway(alpha_rev: np.ndarray, mom_z: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """Reversal sleeve may not short extreme residual winners or buy extreme residual losers."""
    a = alpha_rev.copy()
    a = np.where(mom_z > threshold, np.maximum(a, 0.0), a)
    a = np.where(mom_z < -threshold, np.minimum(a, 0.0), a)
    return a


def blend(alpha_rev: np.ndarray, mom: np.ndarray, w_mom: float) -> np.ndarray:
    """Risk-blend two sleeves: (1-w_mom) * z(reversal alpha) + w_mom * z(momentum)."""
    return (1 - w_mom) * xs_z(alpha_rev) + w_mom * xs_z(mom)
