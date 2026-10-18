"""Regime identification methods.

Each method maps residual returns up to day t to a score at day t, where a
HIGHER score means MORE LIKELY TRENDING. Scores are not probabilities; the
ensemble layer standardizes them cross-sectionally and lets a walk-forward
regression decide how much each one is worth. No true regime labels are used.

Input z is standardized residual returns: z_t = e_t / vol_hat_{t-1}.
"""
import numpy as np


def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Mean over [t-w+1, t]; NaN before a full window."""
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])
    out = np.full_like(x, np.nan, dtype=float)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def variance_ratio(z: np.ndarray, q: int = 5, window: int = 120) -> np.ndarray:
    """Lo-MacKinlay variance ratio VR(q) as a z-statistic. VR > 1 means trending."""
    zz = np.nan_to_num(z)
    cs = np.concatenate([np.zeros((1, zz.shape[1])), np.cumsum(zz, 0)])
    sq = np.full_like(zz, 0.0)
    sq[q - 1:] = cs[q:] - cs[:-q]                     # overlapping q-day sums
    v1 = _roll_mean(zz ** 2, window)
    vq = _roll_mean(sq ** 2, window)
    vr = vq / (q * v1)
    se = np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * window))
    return (vr - 1) / se


def efficiency_ratio(z: np.ndarray, window: int = 20) -> np.ndarray:
    """Kaufman efficiency ratio |net move| / path length, in [0, 1]. High = trending."""
    zz = np.nan_to_num(z)
    net = np.abs(_roll_mean(zz, window)) * window
    path = _roll_mean(np.abs(zz), window) * window
    return net / np.maximum(path, 1e-12)


def autocorr_sum(z: np.ndarray, max_lag: int = 5, window: int = 60) -> np.ndarray:
    """Sum of autocorrelations at lags 1..max_lag over a rolling window."""
    zz = np.nan_to_num(z)
    denom = _roll_mean(zz ** 2, window)
    tot = np.zeros_like(zz)
    for k in range(1, max_lag + 1):
        prod = np.zeros_like(zz)
        prod[k:] = zz[k:] * zz[:-k]
        tot += _roll_mean(prod, window) / np.maximum(denom, 1e-12)
    return tot


def _ewma(z: np.ndarray, lam: float) -> np.ndarray:
    out = np.zeros_like(z)
    acc = np.zeros(z.shape[1])
    for t in range(z.shape[0]):
        acc = lam * acc + np.nan_to_num(z[t])
        out[t] = acc
    return out


def hamilton_filter(z: np.ndarray, fit_until: int, stay: float = 1 - 1 / 120,
                    slow_lam: float = 0.95, fast_lam: float = 0.80) -> np.ndarray:
    """Two-state regression-switching filter, returning log-odds of 'trending'.

    Trending state:       z_t ~ N(b_trend * slow_ewma_{t-1}, 1)
    Mean-reverting state: z_t ~ N(b_rev   * fast_ewma_{t-1}, 1)
    b_trend, b_rev are fitted by pooled least squares on days < fit_until
    (doubled, since each effect is present only about half the time).
    """
    zz = np.nan_to_num(z)
    slow, fast = _ewma(zz, slow_lam), _ewma(zz, fast_lam)
    X = np.stack([slow[:fit_until - 1].ravel(), fast[:fit_until - 1].ravel()], 1)
    y = zz[1:fit_until].ravel()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    b_trend, b_rev = 2 * max(b[0], 1e-4), 2 * min(b[1], -1e-4)

    T, N = zz.shape
    q = np.full(N, 0.5)
    out = np.zeros((T, N))
    for t in range(1, T):
        prior = q * stay + (1 - q) * (1 - stay)
        # log-likelihood ratio (log space avoids 0/0 when both likelihoods underflow)
        llr = -0.5 * (zz[t] - b_trend * slow[t - 1]) ** 2 + 0.5 * (zz[t] - b_rev * fast[t - 1]) ** 2
        logit = np.log(prior / (1 - prior)) + np.clip(llr, -30, 30)
        logit = np.clip(logit, -13.8, 13.8)
        q = 1 / (1 + np.exp(-logit))
        out[t] = logit
    return out


def kalman_drift_strength(z: np.ndarray, rho: float = 0.98, eta: float = 0.015) -> np.ndarray:
    """Scalar Kalman filter for a slowly varying drift s_t in z_t = s_t + noise.

    Score = s_hat^2 / P (squared t-statistic of the drift). A significant
    drift in either direction is evidence of a trending regime.
    """
    T, N = z.shape
    s, P = np.zeros(N), np.ones(N) * eta ** 2 / (1 - rho ** 2)
    out = np.zeros((T, N))
    for t in range(T):
        s, P = rho * s, rho ** 2 * P + eta ** 2           # predict
        K = P / (P + 1.0)                                   # observation noise var = 1
        s = s + K * (np.nan_to_num(z[t]) - s)               # update
        P = (1 - K) * P
        out[t] = s ** 2 / P
    return np.log1p(out)


def all_scores(z: np.ndarray, methods: tuple, fit_until: int) -> dict:
    fns = {
        "var_ratio": lambda: variance_ratio(z),
        "efficiency": lambda: efficiency_ratio(z),
        "autocorr": lambda: autocorr_sum(z),
        "hamilton": lambda: hamilton_filter(z, fit_until),
        "kalman": lambda: kalman_drift_strength(z),
    }
    return {m: fns[m]() for m in methods}
