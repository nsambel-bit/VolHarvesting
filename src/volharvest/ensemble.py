"""Turn regime scores into return forecasts, walk-forward.

Forecast model (pooled across stocks and dates, ridge):
    y = b_T*T + b_R*R + sum_k [ c_k * T*s_k + d_k * R*s_k ]
where
    T   = past-month residual return / vol   (trend signal)
    R   = past-week residual return / vol    (reversal signal)
    s_k = regime score of method k, cross-sectionally standardized
    y   = next-week residual return / vol

The interaction terms let the data decide how much to lean on trend vs reversal
given each regime score. Because the target is realized returns, not regime
labels, the same code runs on real data where the true regime is unknown.
"""
import numpy as np

from .config import ModelConfig


def _xs_z(a: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Cross-sectional z-score along the last axis; NaN -> 0."""
    m = np.nanmean(a, axis=-1, keepdims=True)
    s = np.nanstd(a, axis=-1, keepdims=True)
    out = (a - m) / np.where(s > 0, s, 1)
    return np.clip(np.nan_to_num(out), -clip, clip)


class Panel:
    """Features and targets on the weekly rebalance grid."""

    def __init__(self, e, vol, scores: dict, cfg: ModelConfig, regime=None, start: int = 130):
        T = e.shape[0]
        self.dates = np.arange(start, T - cfg.rebalance_every - 1, cfg.rebalance_every)
        d, h = self.dates, cfg.rebalance_every
        cs = np.concatenate([np.zeros((1, e.shape[1])), np.cumsum(e, 0)])
        v = vol[d]
        self.vol = v
        self.trend = (cs[d + 1] - cs[d + 1 - cfg.trend_lookback]) / (v * np.sqrt(cfg.trend_lookback))
        self.rev = (cs[d + 1] - cs[d + 1 - cfg.reversal_lookback]) / (v * np.sqrt(cfg.reversal_lookback))
        self.y = (cs[d + 1 + h] - cs[d + 1]) / (v * np.sqrt(h))
        self.z = {k: _xs_z(s[d]) for k, s in scores.items()}
        zs = np.stack(list(self.z.values()))
        self.combined = {
            "average": _xs_z(zs.mean(0)),
            "vote": np.sign(zs).mean(0),
        }
        if regime is not None:
            self.combined["oracle"] = 2.0 * regime[d] - 1.0

    def design(self, variant: str) -> np.ndarray:
        """Feature tensor (D, N, F) for a model variant."""
        T, R = self.trend, self.rev
        if variant == "trend_only":
            return T[..., None]
        if variant == "reversal_only":
            return R[..., None]
        if variant == "no_regime":
            return np.stack([T, R], -1)
        if variant == "stack":
            feats = [T, R] + [T * s for s in self.z.values()] + [R * s for s in self.z.values()]
            return np.stack(feats, -1)
        s = self.z[variant] if variant in self.z else self.combined[variant]
        return np.stack([T, R, T * s, R * s], -1)


def walk_forward(panel: Panel, variant: str, cfg: ModelConfig):
    """Expanding-window ridge, refit every `refit_every_weeks`, trading from warmup_days.

    Only rows whose target window has fully closed before the refit date are used.
    Returns (trade_idx, alpha_std, coef_history).
    """
    X = panel.design(variant)
    y = panel.y
    D, N, F = X.shape
    trade = np.where(panel.dates >= cfg.warmup_days)[0]
    alpha = np.zeros((len(trade), N))
    coefs, coef = [], None
    for j, d in enumerate(trade):
        if j % cfg.refit_every_weeks == 0:
            t_now = panel.dates[d]
            train = np.where(panel.dates + cfg.rebalance_every < t_now)[0]
            Xt, yt = X[train].reshape(-1, F), y[train].ravel()
            coef = np.linalg.solve(Xt.T @ Xt + cfg.ridge * np.eye(F), Xt.T @ yt)
            coefs.append((int(t_now), coef.copy()))
        alpha[j] = X[d] @ coef
    return trade, alpha, coefs
