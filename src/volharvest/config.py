"""Central configuration. Every tunable lives here so experiments are reproducible."""
from dataclasses import dataclass, field


@dataclass
class UniverseConfig:
    n_stocks: int = 500
    n_sectors: int = 10
    n_days: int = 2520            # 10 years of trading days
    seed: int = 11
    mkt_drift: float = 0.0003     # daily
    mkt_vol: float = 0.010
    sector_vol: float = 0.006
    idio_vol_median: float = 0.018
    regime_days: float = 120.0    # mean regime duration (days)
    trend_rho: float = 0.985      # persistence of residual drift in trending regime
    trend_eta: float = 0.008      # drift innovation, in units of idio vol
    mr_lambda: float = 0.80       # EWMA decay of recent residual move
    mr_kappa: float = 0.03        # strength of reversal in mean-reverting regime


@dataclass
class ModelConfig:
    rebalance_every: int = 5      # trading days (weekly)
    beta_window: int = 252
    vol_window: int = 60
    trend_lookback: int = 20      # trend signal: past ~month residual return
    reversal_lookback: int = 5    # reversal signal: past week residual return
    warmup_days: int = 504        # 2 years before first trade
    refit_every_weeks: int = 52   # walk-forward refit frequency
    ridge: float = 10.0
    methods: tuple = ("var_ratio", "efficiency", "autocorr", "hamilton", "kalman")


@dataclass
class PortfolioConfig:
    target_vol: float = 0.10      # annualized, ex-ante
    max_weight: float = 0.02      # per-name cap (fraction of capital, each side)
    trade_fraction: float = 1.0   # 1 = jump to target; <1 = partial trading
    cost_bps: float = 5.0         # one-way cost per unit traded


@dataclass
class Config:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
