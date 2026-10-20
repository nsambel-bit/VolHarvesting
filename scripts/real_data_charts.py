"""Charts for the real-data study. Requires output/real_study (scripts/real_data_study.py).

    python scripts/real_data_charts.py   ->  docs/real_data_summary.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TXT, TXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b2"
ORDER = ["trend_only", "reversal_only", "no_regime", "efficiency", "kalman", "var_ratio", "autocorr",
         "hamilton", "vote", "average", "stack"]
LABEL = {"trend_only": "Trend only", "reversal_only": "Reversal only", "no_regime": "No regime (trend + reversal)",
         "efficiency": "Efficiency ratio", "kalman": "Kalman drift", "var_ratio": "Variance ratio",
         "autocorr": "Autocorrelation", "hamilton": "Hamilton filter", "vote": "Ensemble: vote",
         "average": "Ensemble: average", "stack": "Ensemble: stacked"}

study = json.load(open("output/real_study/study.json"))
cost = pd.DataFrame(study["cost_sensitivity"])
net = pd.read_csv("output/real_study/daily_net_5bp.csv", index_col=0, parse_dates=True)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2})
fig, ax = plt.subplots(2, 1, figsize=(6.6, 10), facecolor=SURF, gridspec_kw={"height_ratios": [1.2, 1]})
for a in ax:
    a.set_facecolor(SURF); a.grid(color=GRID, lw=.6); a.set_axisbelow(True)

y = np.arange(len(ORDER))[::-1]
hgt = 0.38
for off, col, c, lab in ((hgt / 2, BLUE, "2bp", "2bp costs"), (-hgt / 2, ORANGE, "5bp", "5bp costs")):
    vals = [cost.loc[v, c] for v in ORDER]
    ax[0].barh(y + off, vals, height=hgt - 0.04, color=col, label=lab)
    for yi, v in zip(y + off, vals):
        ax[0].text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=7.5, color=TXT2)
ax[0].axvline(0, color=TXT2, lw=.8)
ax[0].set_yticks(y, [LABEL[v] for v in ORDER], fontsize=8.5, color=TXT)
ax[0].set_xlabel("Net Sharpe ratio, Sep 2007 – Aug 2025")
ax[0].set_title("88 large caps: regime layer adds nothing", loc="left", fontsize=12, color=TXT)
ax[0].legend(frameon=False, fontsize=8, labelcolor=TXT2, loc="lower right")
ax[0].set_xlim(0, 0.85)

for v, col in (("no_regime", GRAY), ("average", BLUE), ("stack", AQUA)):
    c = net[v].cumsum() * 100
    ax[1].plot(c.index, c.values, color=col, lw=2, label=LABEL[v])
ax[1].axvspan(pd.Timestamp("2021-05-27"), net.index[-1], color=GRID, alpha=.6, lw=0)
ax[1].text(pd.Timestamp("2021-08-01"), ax[1].get_ylim()[0] + 3, "Drawdown\nsince May 2021", fontsize=8, color=TXT2, va="bottom")
ax[1].set_ylabel("Cumulative net return, 5bp costs  [%]")
ax[1].set_title("Gains until 2021, then a long drawdown", loc="left", fontsize=12, color=TXT)
ax[1].legend(frameon=False, fontsize=8, labelcolor=TXT2, loc="upper left")
plt.tight_layout(h_pad=2.5)
Path("docs").mkdir(exist_ok=True)
plt.savefig("docs/real_data_summary.png", dpi=170, facecolor=SURF)
print("saved docs/real_data_summary.png")
