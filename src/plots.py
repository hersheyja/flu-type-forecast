# makes the 2 charts:
# 1. which flu won each region each season
# 2. how well each method guessed the winner

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

out = Path("data/processed")
fig_dir = Path("figures")
viruses = ["H1N1", "H3N2", "B"]
colors = ["lightgray", "steelblue", "darkorange", "seagreen"]  # no data, H1N1, H3N2, B


def winner_chart():
    rs = pd.read_csv(out / "region_season.csv")
    seasons = list(range(rs["season"].min(), rs["season"].max() + 1))

    # grid of regions x seasons, 0 = no data, 1 = H1N1, 2 = H3N2, 3 = B
    grid = np.zeros((10, len(seasons)))
    for _, r in rs.iterrows():
        grid[r["region"] - 1, seasons.index(r["season"])] = viruses.index(r["winner"]) + 1

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.imshow(grid, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")

    # write H1 / H3 / B in each square
    labels = {1: "H1", 2: "H3", 3: "B"}
    for i in range(10):
        for j in range(len(seasons)):
            if grid[i, j]:
                ax.text(j, i, labels[grid[i, j]], ha="center", va="center", fontsize=7, color="white")

    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels([f"{s}-{(s + 1) % 100:02d}" for s in seasons], rotation=60, fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Region {r}" for r in range(1, 11)], fontsize=8)
    ax.set_title("Which flu won each HHS region each season (gray = not enough data)")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in
                       zip(colors[1:] + colors[:1], viruses + ["Not enough data"])],
              bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "winner_by_region_season.png", dpi=200)


def results_chart():
    res = pd.read_csv(out / "results.csv")
    y = np.arange(len(res))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(y - 0.2, res["accuracy"], height=0.4, color="steelblue", label="Exact winner correct")
    ax.barh(y + 0.2, res["top2"].fillna(0), height=0.4, color="lightsteelblue", label="Winner in top 2 guesses")

    # write the % next to each bar
    for i, r in res.iterrows():
        ax.text(r["accuracy"] + 0.01, i - 0.2, f"{r['accuracy']:.0%}", va="center")
        if not np.isnan(r["top2"]):
            ax.text(r["top2"] + 0.01, i + 0.2, f"{r['top2']:.0%}", va="center")

    ax.axvline(1 / 3, color="gray", linestyle="--", label="Random guess (33%)")
    ax.set_yticks(y)
    ax.set_yticklabels(res["method"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("% of region-seasons")
    ax.set_title("How often each method guessed the winning flu")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_results.png", dpi=200)


def main():
    fig_dir.mkdir(exist_ok=True)
    winner_chart()
    results_chart()
    print("saved charts to figures/")


if __name__ == "__main__":
    main()
