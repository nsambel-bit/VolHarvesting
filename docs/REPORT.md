# Regime-switching long/short equity: research report

## What the pipeline does

1. **Residualize** each stock's daily return: remove rolling-estimated market beta (252-day window, ending the previous day) and the same-day sector move.
2. **Score regimes** on those residuals with five methods. Each outputs a score where higher means "more likely trending":
   - Variance ratio VR(5), 120-day window, as a z-statistic
   - Kaufman efficiency ratio, 20 days
   - Sum of autocorrelations at lags 1–5, 60 days
   - Two-state Hamilton filter (trend state follows a slow moving average, reversal state fades a fast one), log-odds output
   - Kalman filter for a slowly varying drift; score is the drift's squared t-statistic
3. **Forecast** next-week residual returns with a pooled ridge regression on a trend signal (past month), a reversal signal (past week), and their interactions with the regime scores. It is refit yearly on an expanding window and trades only after 2 years of warmup. The target is realized returns, not regime labels, so the same code runs unchanged on real data.
4. **Combine** methods three ways:
   - **Vote:** average sign of the standardized scores
   - **Average:** mean of the standardized scores
   - **Stacked:** every method's interactions enter the regression separately and get their own fitted weights
5. **Build the portfolio** weekly: closed-form mean-variance weights under exact beta- and sector-neutrality, 10% ex-ante volatility target, 2% per-name cap (Dykstra projection), 5bp one-way costs, optional partial trading.

## Synthetic test universe

500 stocks, 10 sectors, 10 years of daily data. Each stock switches between a trending regime (a persistent residual drift) and a mean-reverting one (reversal of the recent move), with regimes lasting about 6 months on average. Two signal strengths:

- **Realistic:** weekly information coefficient (IC) of about 0.02 without regime information, in the range reported for short-term equity signals.
- **Strong:** about twice that, with regimes easier to detect.

Because the true regimes are known, each method's accuracy can be measured directly (AUC: 0.5 is a coin flip, 1.0 is perfect).

## Results (half-speed trading, 5bp costs)

| Variant | Net Sharpe, realistic | Net Sharpe, strong | Regime AUC, realistic / strong |
|---|---|---|---|
| No regime (trend + reversal) | 1.04 | 2.87 | – |
| Efficiency ratio | 1.04 | 3.40 | 0.54 / 0.57 |
| Kalman drift | 1.04 | 4.13 | 0.54 / 0.58 |
| Variance ratio | 1.09 | 3.72 | 0.55 / 0.61 |
| Autocorrelation | 1.00 | 3.50 | 0.56 / 0.62 |
| Hamilton filter | 1.08 | 4.17 | 0.58 / 0.65 |
| Ensemble: vote | 0.99 | 3.83 | 0.57 / 0.63 |
| Ensemble: average | 1.07 | 4.14 | 0.57 / 0.64 |
| Ensemble: stacked | 1.01 | 4.41 | – |
| Oracle (true regime) | 3.81 | 9.30 | 1.00 |

Absolute Sharpe ratios are inflated by the clean synthetic setup and the breadth of 500 independent names. **Compare variants against each other, not against real-world expectations.**

## Findings

1. **The regime layer only pays when regimes are detectable.** With realistic signal strength, the best methods reach AUC of about 0.55–0.58 and add essentially nothing over the no-regime model (1.04 to about 1.08), even though perfect regime knowledge would be worth 3.8. With strong regimes (AUC 0.58–0.65), the same pipeline adds +1.3 to +1.5 Sharpe. This matches the earlier single-stock result: Sharpe scales with how confidently and correctly the regime is called.
2. **Averaging beats voting,** by 0.3 Sharpe in the strong case. Voting discards how strong each method's evidence is.
3. **Stacking is best when there's signal (4.41) and slightly worst when there isn't (1.01).** Fitted weights help when the methods carry real information and overfit noise when they don't. Equal-weight averaging is the safer default until the uplift is proven.
4. **The methods overlap a lot.** Hamilton correlates 0.71–0.75 with the variance ratio and autocorrelation scores, so five methods carry closer to two or three independent views. Efficiency ratio and Kalman drift are the most distinct.
5. **Costs dominate.** Full-speed trading turns over about 7× capital per week; trading half the gap each week cuts that to about 3× and improves net Sharpe in most variants.
6. **Factor neutrality holds out of sample.** Realized beta stays within ±0.015 in every variant, and market plus sector returns explain at most 0.5% of P&L variance, even though betas are estimated rather than known.

## Checks performed
- `tests/test_pipeline.py`: (a) weights satisfy the beta and sector constraints to 1e-10 and respect caps; (b) **no look-ahead**: perturbing all returns after a cutoff date leaves every residual, beta, regime score and feature up to that date unchanged.
- The oracle variant recovers a large, stable Sharpe, confirming the forecasting and portfolio layers can use regime information when it exists.

## Before running on real data
- **Data:** adjusted daily prices including delisted names (to avoid survivorship bias), plus a sector map. Run `python -m volharvest.run --prices data/prices.csv --sectors data/sectors.csv` (formats in `data/README.md`). With real data, regime AUC can't be measured, so judge each method by its out-of-sample Sharpe and IC uplift over the no-regime baseline.
- **Costs:** 5bp is optimistic for small caps. Add short-borrow fees and a market-impact model before trusting net numbers.
- **Parameters:** the Hamilton filter's coefficients are fitted on the warmup period; the Kalman filter's are fixed assumptions. Estimate both properly (by maximum likelihood) on real data.
- **Factors:** only market and sector are neutralized. Size, value and momentum exposures can leak in; adding them needs fundamentals data.
- **Multiple testing:** with many methods and combinations, report the best variant against a data-snooping test such as White's Reality Check or Hansen's SPA, not just the raw best Sharpe.
