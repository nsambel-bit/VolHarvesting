# data/

Input data lives here and is **not committed** (see `.gitignore`). The default run needs no data: it generates a synthetic universe in memory (`volharvest.universe.synthetic_universe`).

To run on real stocks, add two files:

**`prices.csv`**: wide table of daily adjusted closes.

```
date,AAPL,MSFT,XOM,...
2015-01-02,24.53,40.07,72.32,...
```

- One column per ticker, dates ascending.
- Adjusted for splits and dividends.
- **Include delisted stocks** for the periods they traded; dropping them creates survivorship bias.
- Gaps are forward-filled for up to 5 days, after which the return is set to 0.

**`sectors.csv`**: one row per ticker.

```
ticker,sector
AAPL,Information Technology
XOM,Energy
```

Tickers missing from `sectors.csv` are dropped. Then run:

```bash
python -m volharvest.run --prices data/prices.csv --sectors data/sectors.csv --trade-fraction 0.5 --out output/real
```
