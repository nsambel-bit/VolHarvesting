"""Mean-variance weights with exact beta and sector neutrality.

With a diagonal (idiosyncratic) risk model D and linear constraints A w = 0,
the mean-variance optimum is closed-form:
    w = D^-1 (alpha - A' lam),   lam = (A D^-1 A')^-1 A D^-1 alpha
A has one row of estimated betas and one indicator row per sector, so the
portfolio is beta-neutral, sector-neutral, and therefore also dollar-neutral.
"""
import numpy as np

from .config import PortfolioConfig


def constraint_matrix(beta_hat: np.ndarray, sector: np.ndarray) -> np.ndarray:
    sectors = np.unique(sector)
    rows = [beta_hat] + [(sector == s).astype(float) for s in sectors]
    return np.vstack(rows)


def _project(w: np.ndarray, A: np.ndarray, dinv: np.ndarray) -> np.ndarray:
    """Smallest risk-weighted change to w that satisfies A w = 0."""
    M = (A * dinv) @ A.T
    return w - dinv * (A.T @ np.linalg.solve(M, A @ w))


def neutral_weights(alpha, var, beta_hat, sector, cfg: PortfolioConfig) -> np.ndarray:
    """alpha: expected next-period residual return; var: its variance (same horizon)."""
    A = constraint_matrix(beta_hat, sector)
    dinv = 1.0 / var
    w = _project(dinv * alpha, A, dinv)
    ann_var = var / 5.0 * 252                            # var is weekly
    vol = np.sqrt(np.sum(w ** 2 * ann_var))
    if vol > 0:
        w = w * cfg.target_vol / vol
    # Dykstra's alternating projections onto {|w| <= cap} and {A w = 0}, in the risk metric.
    # Ends exactly neutral and (to tolerance) within caps. If caps bind, ex-ante vol ends
    # below target rather than breaking a constraint.
    x, p, q = w, np.zeros_like(w), np.zeros_like(w)
    for _ in range(300):
        y = np.clip(x + p, -cfg.max_weight, cfg.max_weight)
        p = x + p - y
        x_new = _project(y + q, A, dinv)
        q = y + q - x_new
        if np.abs(x_new - x).max() < 1e-7:
            x = x_new
            break
        x = x_new
    return x
