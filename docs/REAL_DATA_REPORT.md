# Real-data study: 88 US large caps, 2005–2025

![Real-data summary](real_data_summary.png)

## Bottom line

On real prices, **the regime layer does not help.** No regime method or ensemble beats the simple no-regime model (a fixed blend of residual trend and reversal), and the stacked ensemble is significantly worse. The base strategy itself is thin: a gross Sharpe of 0.80 becomes 0.32 after 5bp costs, and it has been in a drawdown since May 2021 that was still at its low in August 2025.

## Data and setup

- **Prices:** daily adjusted closes, 25 Aug 2005 – 22 Aug 2025. The final row (25 Aug 2025) was dropped as an apparent intraday snapshot.
- **Universe:** 88 stocks. GOOG was dropped as a near-duplicate of GOOGL (daily return correlation 0.997). The sector ETFs and indices are not traded.
- **Sectors:** GICS (2025), with small groups merged: Linde into "Industrials & Materials"; utilities with real estate. See `data/sectors.csv`.
- **Market proxy:** S&P 500 index (`^GSPC`).
- **Portfolio:** weekly rebalance, half of the gap to target traded each week, 10% ex-ante volatility target, 5% per-name cap, exactly beta- and sector-neutral at each rebalance.
- **Trading period:** Sep 2007 – Aug 2025, after a 2-year warmup. Forecasts are refit yearly on an expanding window, so every result is out of sample.
- **Data checks:** no gaps or non-positive prices. GE's 2023 and 2024 spin-offs are correctly adjusted. The largest unexplained daily moves (VRTX, AMD, NFLX, C, CDNS, NVDA) are genuine events, not bad prints.

Reproduce with:

```bash
python -m volharvest.run --prices data/prices.csv --sectors data/sectors.csv --market ^GSPC \
    --drop-last-row --trade-fraction 0.5 --max-weight 0.05 --out output/real_tf0.5
python scripts/real_data_study.py --prices data/prices.csv --sectors data/sectors.csv
python scripts/real_data_charts.py
```

## Results

**Headline (5bp one-way costs):**

| Variant | Gross Sharpe | Net Sharpe | Net return / yr | Volatility | Gross leverage | Turnover / week |
|---|---|---|---|---|---|---|
| Trend only | 0.57 | 0.25 | 2.1% | 8.4% | 2.3× | 1.07 |
| Reversal only | 1.00 | 0.31 | 1.8% | 5.7% | 1.5× | 1.56 |
| **No regime** | **0.80** | **0.32** | **2.3%** | **7.2%** | **1.9×** | **1.35** |
| Hamilton filter | 0.75 | 0.28 | 2.0% | 7.1% | 1.8× | 1.32 |
| Kalman drift | 0.79 | 0.32 | 2.2% | 6.9% | 1.7× | 1.29 |
| Ensemble: vote | 0.78 | 0.31 | 2.1% | 6.8% | 1.7× | 1.29 |
| Ensemble: average | 0.77 | 0.29 | 2.0% | 6.9% | 1.7× | 1.29 |
| Ensemble: stacked | 0.67 | 0.19 | 1.2% | 6.5% | 1.7× | 1.25 |

Realized volatility (about 7%) is below the 10% target because the 5% per-name cap binds.

**Cost sensitivity (net Sharpe):**

| Variant | 0bp | 2bp | 5bp | 10bp |
|---|---|---|---|---|
| No regime | 0.80 | 0.61 | 0.32 | −0.14 |
| Ensemble: average | 0.77 | 0.58 | 0.29 | −0.18 |
| Ensemble: stacked | 0.67 | 0.48 | 0.19 | −0.29 |

The strategy breaks even at roughly 8bp one-way. Costs, not signal, decide whether it is worth running.

**Significance versus no regime** (net Sharpe difference at 5bp; 20-day block bootstrap, 2,000 resamples):

| Variant | Difference | 95% interval | p (better than baseline) |
|---|---|---|---|
| Kalman drift | −0.006 | −0.06 to +0.05 | 0.58 |
| Ensemble: vote | −0.019 | −0.08 to +0.04 | 0.73 |
| Efficiency ratio | −0.026 | −0.06 to +0.01 | 0.91 |
| Ensemble: average | −0.031 | −0.09 to +0.02 | 0.88 |
| Hamilton filter | −0.040 | −0.08 to 0.00 | 0.96 |
| Variance ratio | −0.042 | −0.09 to 0.00 | 0.97 |
| Autocorrelation | −0.045 | −0.12 to +0.03 | 0.88 |
| Ensemble: stacked | −0.136 | −0.25 to −0.01 | 0.99 |

Every regime variant is at or below the baseline. Stacking is reliably worse: its fitted weights overfit noise, which is the failure mode the synthetic study predicted for weak regimes.

**Net Sharpe by subperiod (5bp):**

| Variant | 2007–09 | 2010–14 | 2015–19 | 2020–22 | 2023–25 |
|---|---|---|---|---|---|
| Trend only | 0.34 | 0.54 | 0.74 | 0.26 | −0.83 |
| Reversal only | 1.55 | 0.53 | 0.26 | 0.34 | −1.10 |
| No regime | 1.02 | 0.75 | 0.56 | 0.26 | −1.01 |
| Ensemble: average | 0.93 | 0.68 | 0.54 | 0.25 | −0.96 |

The edge decays over time and turns clearly negative from 2023. The no-regime model peaked in May 2021 and was in August 2025 was 28.7 percentage points of cumulative return below that peak, about four times its annual volatility, with no recovery yet.

## Why 2023–25 went wrong

The no-regime model's 2023–25 losses are concentrated in a few stocks with large, persistent moves: GE (−3.6% of capital), NVDA (−3.4%), APH (−2.7%) and ORCL (−2.2%). By sector, technology lost 6.3%, industrials and materials 3.7% and financials 3.2%. This was a market of narrow, sustained leadership in which both the one-week reversal and the one-month trend signals lost money. Regime detection did not help: it couldn't distinguish these persistent moves early enough to change positioning.

## Survivorship bias

Every stock trades for the full 20 years, so the list is today's largest companies, chosen with hindsight.

- **The biggest winners didn't flatter the results.** Removing the 10 largest full-sample compounders (NVDA, NFLX, BKNG, AAPL, AMZN, REGN, CRM, APH, LRCX, INTU) *raised* net Sharpe: no regime from 0.32 to 0.46, trend only from 0.25 to 0.47. The strategy lost money on those names.
- **The missing failures probably do flatter it.** Companies that collapsed or were acquired (for example, in 2008) are absent. The reversal leg buys recent losers; in a point-in-time universe, some of those losers would have kept falling. Expect real results to be somewhat worse than shown.

## Factor exposure

The portfolio is exactly beta- and sector-neutral at each rebalance using estimated betas, but realized exposure leaks: realized beta of about 0.06, and market plus sector returns explain about 6% of P&L variance (versus at most 0.5% in the synthetic study). Betas estimated from one year of daily data are noisy for volatile names.

## Recommendations

1. **Don't trade this as is.** The regime layer adds no value here, and the base signal has been losing since 2021.
2. **If continuing the research:**
   - test on a broad, point-in-time universe (e.g. historical S&P 500 or Russell 1000 membership, with delisted stocks), since 88 names gives little breadth;
   - cut turnover (longer holding periods, a turnover penalty), because costs consume about 60% of gross return;
   - neutralize momentum and size, and exclude names around earnings;
   - shrink betas toward 1 or estimate them on weekly data to reduce beta leakage.
3. **Simplest robust choice:** if regime information is used at all, use the plain average, never stacking.
