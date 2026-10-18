# AGENTS.md

Guidance for coding agents (and humans) working in this repository.

## Setup and commands

```bash
pip install -r requirements.txt && pip install -e .
pytest                                    # must pass before any commit
python -m volharvest.run --scenario realistic --trade-fraction 0.5 --out output/check
```

A full synthetic run takes a few seconds. Re-run it after changing anything in `src/` and compare the printed table with `docs/REPORT.md`.

## Architecture

Data flows one way: `universe -> factors -> regimes -> ensemble -> portfolio -> backtest`, orchestrated by `run.py`. Arrays are `(T, N)`: days × stocks. The ensemble works on a weekly grid (`Panel.dates`, indices into the daily axis).

## Rules that protect the backtest

1. **No look-ahead.** Anything computed for day `t` may use data up to and including `t` only. Trades placed at the close of `t` earn returns from `t+1`. `tests/test_pipeline.py::test_no_lookahead` perturbs future returns and checks past signals are unchanged. Extend it when you add a signal or feature.
2. **No regime labels in modeling.** `Universe.regime` exists only for synthetic diagnostics (AUC, oracle). Regime scores and forecasts must never read it, so the pipeline runs unchanged on real data.
3. **Neutrality is a hard constraint.** Weights from `portfolio.neutral_weights` satisfy `A @ w = 0` (beta and sector rows) exactly. If you add constraints or factors, add rows to `constraint_matrix` and keep `test_neutrality_and_caps` passing.
4. **Walk-forward only.** Models are fitted on rows whose target window closed before the refit date (`ensemble.walk_forward`). Never fit on the full sample.
5. **Report gross and net.** Any new variant goes through `backtest.summarize`, so costs, turnover and factor checks are always reported.

## Conventions

- All tunable numbers live in `config.py` dataclasses. Don't hard-code parameters in modules.
- Regime methods in `regimes.py` take standardized residuals `z` of shape `(T, N)` and return a `(T, N)` score where **higher means more likely trending**. Register new methods in `all_scores` and in `ModelConfig.methods`.
- Keep modules numpy-vectorized across stocks; loops over time are fine.
- Data files and `output/` are git-ignored. Never commit price data.
- When results change materially, update `docs/REPORT.md` and the README table, and regenerate `docs/summary.png` with `python -m volharvest.charts` and a copy into `docs/`.

## Adding a regime method (checklist)

1. Implement `my_method(z) -> scores` in `regimes.py`, causal and vectorized.
2. Add it to `all_scores` and `ModelConfig.methods`.
3. Run `pytest`; the no-look-ahead test covers it automatically.
4. Run both scenarios and record its AUC and net Sharpe against `no_regime` and `average`.
