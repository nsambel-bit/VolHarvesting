"""Factor residuals: strip market beta and sector moves from each stock's return.

Everything at day t uses only data up to t (beta from the window ending t-1),
so residuals can feed regime detection without look-ahead.
"""
import numpy as np


def _window_sum(c: np.ndarray, t_idx: np.ndarray, w: int) -> np.ndarray:
    """Sum over [t-w, t) from a zero-padded cumulative sum array c (len T+1)."""
    lo = np.maximum(t_idx - w, 0)
    return c[t_idx] - c[lo]


def rolling_beta(returns: np.ndarray, market: np.ndarray, window: int, min_obs: int = 60) -> np.ndarray:
    """Beta of each stock to the market, estimated on days [t-window, t). Shape (T, N)."""
    T, _ = returns.shape
    pad = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
    c_rm, c_r = pad(returns * market[:, None]), pad(returns)
    c_m, c_mm = pad(market), pad(market ** 2)
    t = np.arange(T)
    n = np.minimum(t, window).astype(float)
    s_rm, s_r = _window_sum(c_rm, t, window), _window_sum(c_r, t, window)
    s_m, s_mm = _window_sum(c_m, t, window), _window_sum(c_mm, t, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = s_rm - s_r * (s_m / n)[:, None]
        var = s_mm - s_m ** 2 / n
        beta = cov / var[:, None]
    beta[n < min_obs] = 1.0
    return beta


def residualize(returns: np.ndarray, sector: np.ndarray, beta_window: int):
    """Residual = return - beta_hat * market - same-day sector average of (return - beta_hat * market).

    Market proxy is the equal-weighted universe return. Returns (residuals, beta_hat, market).
    """
    market = returns.mean(axis=1)
    beta = rolling_beta(returns, market, beta_window)
    e = returns - beta * market[:, None]
    for s in np.unique(sector):
        cols = sector == s
        e[:, cols] -= e[:, cols].mean(axis=1, keepdims=True)
    return e, beta, market


def rolling_vol(x: np.ndarray, window: int, floor: float = 1e-4) -> np.ndarray:
    """Std of x over days [t-window+1, t] (inclusive of t). Shape (T, N)."""
    c1 = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])
    c2 = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x ** 2, 0)])
    t = np.arange(1, x.shape[0] + 1)
    lo = np.maximum(t - window, 0)
    n = (t - lo)[:, None].astype(float)
    m = (c1[t] - c1[lo]) / n
    v = (c2[t] - c2[lo]) / n - m ** 2
    return np.sqrt(np.maximum(v, floor ** 2))
