"""Chart for the improvements study. Requires output/improvements/improvements.json.

    python scripts/improvements_charts.py   ->  docs/improvements.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TXT, TXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

r = pd.DataFrame(json.load(open("output/improvements/improvements.json"))["results"]).T
r = r.iloc[::-1]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2})
fig, ax = plt.subplots(figsize=(6.6, 6.2), facecolor=SURF)
ax.set_facecolor(SURF); ax.grid(color=GRID, lw=.6, axis="x"); ax.set_axisbelow(True)
y = np.arange(len(r))
h = 0.38
for off, col, key, lab in ((h / 2, BLUE, "dev_net_sharpe", "Development, 2007–2020"),
                           (-h / 2, ORANGE, "test_net_sharpe", "Final test, 2021–2025")):
    vals = r[key].astype(float).values
    ax.barh(y + off, vals, height=h - 0.04, color=col, label=lab)
    for yi, v in zip(y + off, vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=TXT2)
ax.axvline(0, color=TXT2, lw=.8)
ax.set_yticks(y, r.index, fontsize=8.5, color=TXT)
ax.set_xlim(-0.75, 0.9)
ax.set_xlabel("Net Sharpe ratio (5bp costs)")
ax.set_title("No change makes 2021–25 profitable", loc="left", fontsize=12, color=TXT)
ax.legend(frameon=False, fontsize=8, labelcolor=TXT2, loc="lower right")
plt.tight_layout()
plt.savefig("docs/improvements.png", dpi=170, facecolor=SURF)
print("saved docs/improvements.png")
