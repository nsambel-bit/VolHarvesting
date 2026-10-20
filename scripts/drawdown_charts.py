"""Chart for the drawdown analysis. Requires output/drawdown/drawdown.json.

    python scripts/drawdown_charts.py   ->  docs/drawdown_analysis.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TXT, TXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b8b7b2"

d = json.load(open("output/drawdown/drawdown.json"))
yr = pd.DataFrame(d["by_year_pct"]).loc[[str(y) for y in range(2008, 2026)]]
yr.index = yr.index.astype(int)
ic = pd.DataFrame(d["ic_by_year"]).loc[[str(y) for y in range(2008, 2026)]]
ic.index = ic.index.astype(int)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2})
fig, ax = plt.subplots(2, 1, figsize=(6.6, 9), facecolor=SURF, sharex=True)
for a in ax:
    a.set_facecolor(SURF); a.grid(color=GRID, lw=.6, axis="y"); a.set_axisbelow(True)
    a.axvspan(2020.5, 2025.5, color=GRID, alpha=.5, lw=0)

x = yr.index.values
w = 0.38
ax[0].bar(x - w / 2, yr["gross"], width=w - 0.04, color=BLUE, label="Gross P&L")
ax[0].bar(x + w / 2, -yr["cost"], width=w - 0.04, color=ORANGE, label="Trading costs")
ax[0].axhline(0, color=TXT2, lw=.8)
ax[0].set_ylabel("% of capital per year")
ax[0].set_title("Costs never let up; gross edge shrank, then 2024 hit", loc="left", fontsize=12, color=TXT)
ax[0].legend(frameon=False, fontsize=8, labelcolor=TXT2, loc="upper right")
ax[0].text(2020.6, ax[0].get_ylim()[0] + 1, "2021–25\n(2025 to Aug)", fontsize=8, color=TXT2, va="bottom")

ax[1].plot(x, ic["trend_ic"], color=BLUE, lw=2, marker="o", ms=5, mec=SURF, label="1-month move vs. next week")
ax[1].plot(x, ic["reversal_ic"], color=ORANGE, lw=2, marker="o", ms=5, mec=SURF, label="1-week move vs. next week")
ax[1].axhline(0, color=TXT2, lw=.8)
ax[1].set_ylabel("Average weekly IC")
ax[1].set_title("Below zero: past moves reverse. 2024: they continued", loc="left", fontsize=12, color=TXT)
ax[1].legend(frameon=False, fontsize=8, labelcolor=TXT2, loc="lower left")
ax[1].set_xticks(np.arange(2008, 2026, 2))
plt.tight_layout(h_pad=2.5)
plt.savefig("docs/drawdown_analysis.png", dpi=170, facecolor=SURF)
print("saved docs/drawdown_analysis.png")
