"""Summary chart from saved runs.

Requires output/realistic_tf0.5 and output/strong_tf0.5 (see README). Run: python -m volharvest.charts
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TXT, TXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b8b7b2"
ORDER = ["no_regime", "efficiency", "kalman", "var_ratio", "autocorr", "hamilton", "vote", "average", "stack", "oracle"]
LABEL = {"no_regime": "No regime (trend + reversal)", "efficiency": "Efficiency ratio", "kalman": "Kalman drift",
         "var_ratio": "Variance ratio", "autocorr": "Autocorrelation", "hamilton": "Hamilton filter",
         "vote": "Ensemble: vote", "average": "Ensemble: average", "stack": "Ensemble: stacked", "oracle": "Oracle (true regime)"}

def main(out: str = "output/summary.png"):
    res = {s: json.load(open(f"output/{s}_tf0.5/results.json"))["results"] for s in ("realistic", "strong")}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
                         "xtick.color": TXT2, "ytick.color": TXT2})
    fig, ax = plt.subplots(2, 1, figsize=(6.6, 10), facecolor=SURF, gridspec_kw={"height_ratios": [1.25, 1]})
    for a in ax:
        a.set_facecolor(SURF); a.grid(color=GRID, lw=.6); a.set_axisbelow(True)

    y = np.arange(len(ORDER))[::-1]
    hgt = 0.38
    for off, scen, col in ((hgt / 2, "realistic", BLUE), (-hgt / 2, "strong", ORANGE)):
        vals = [res[scen][v]["sharpe_net"] for v in ORDER]
        ax[0].barh(y + off, vals, height=hgt - 0.04, color=col, label=f"{scen.capitalize()} regimes")
        for yi, v in zip(y + off, vals):
            ax[0].text(v + 0.08, yi, f"{v:.1f}", va="center", fontsize=7.5, color=TXT2)
    ax[0].set_yticks(y, [LABEL[v] for v in ORDER], fontsize=8.5, color=TXT)
    ax[0].set_xlabel("Net Sharpe ratio (5bp costs, half-speed trading)")
    ax[0].set_title("Regime layer pays only when regimes are detectable", loc="left", fontsize=12, color=TXT)
    ax[0].legend(frameon=False, fontsize=8, labelcolor=TXT2, loc="upper right")
    ax[0].set_xlim(0, 10.6)

    curves = np.load("output/strong_tf0.5/curves.npz")
    for v, col in (("no_regime", GRAY), ("vote", YELLOW), ("average", BLUE), ("stack", AQUA)):
        c = curves[v] * 100
        ax[1].plot(np.arange(len(c)) / 252, c, color=col, lw=2, label=LABEL[v])
    ax[1].set_xlabel("Years of trading (after 2-year warmup)")
    ax[1].set_ylabel("Cumulative net return  [%]")
    ax[1].set_title("Strong regimes: averaging beats voting", loc="left", fontsize=12, color=TXT)
    ax[1].legend(frameon=False, fontsize=8, labelcolor=TXT2, loc="upper left")
    plt.tight_layout(h_pad=2.5)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=170, facecolor=SURF)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
