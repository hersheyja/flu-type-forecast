# makes the charts: winner grid, results bar chart, confusion matrix

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

    # 0 = no data, 1 = H1N1, 2 = H3N2, 3 = B
    grid = np.zeros((10, len(seasons)))
    for _, r in rs.iterrows():
        grid[r["region"] - 1, seasons.index(r["season"])] = viruses.index(r["winner"]) + 1

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.imshow(grid, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")

    # label each square
    labels = {1: "H1", 2: "H3", 3: "B"}
    for i in range(10):
        for j in range(len(seasons)):
            if grid[i, j]:
                ax.text(j, i, labels[grid[i, j]], ha="center", va="center", fontsize=7, color="white")

    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels([f"{s}-{str(s + 1)[2:]}" for s in seasons], rotation=60, fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Region {r}" for r in range(1, 11)], fontsize=8)
    ax.set_title("Which flu won each HHS region each season (gray = not enough data)")
    legend_items = [Patch(color="steelblue", label="H1N1"),
                    Patch(color="darkorange", label="H3N2"),
                    Patch(color="seagreen", label="B"),
                    Patch(color="lightgray", label="Not enough data")]
    ax.legend(handles=legend_items, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "winner_by_region_season.png", dpi=200)


def results_chart():
    res = pd.read_csv(out / "results.csv")
    y = np.arange(len(res))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(y - 0.2, res["accuracy"], height=0.4, color="steelblue", label="Exact winner correct")
    ax.barh(y + 0.2, res["top2"].fillna(0), height=0.4, color="lightsteelblue", label="Winner in top 2 guesses")

    # % next to each bar
    for i, r in res.iterrows():
        ax.text(r["accuracy"] + 0.01, i - 0.2, f"{r['accuracy']:.0%}", va="center")
        if not np.isnan(r["top2"]):
            ax.text(r["top2"] + 0.01, i + 0.2, f"{r['top2']:.0%}", va="center")

    ax.axvline(1 / 3, color="gray", linestyle="--", label="Random guess (33%)")
    ax.set_yticks(y)
    ax.set_yticklabels(res["method"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("% of region-seasons")
    ax.set_title("How often each method guessed the winning flu")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_results.png", dpi=200)


def confusion_chart():
    preds = pd.read_csv(out / "predictions.csv")

    # rows = actual, columns = svm guess
    counts = pd.crosstab(preds["winner"], preds["svm_pred"])
    counts = counts.loc[viruses, viruses]  # same order as other charts

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(counts, cmap="Blues")

    # count in each square
    for i in range(3):
        for j in range(3):
            n = counts.iloc[i, j]
            # white text on dark squares
            if n > 50:
                ax.text(j, i, n, ha="center", va="center", color="white")
            else:
                ax.text(j, i, n, ha="center", va="center", color="black")

    ax.set_xticks(range(3))
    ax.set_xticklabels(viruses)
    ax.set_yticks(range(3))
    ax.set_yticklabels(viruses)
    ax.set_xlabel("What the SVM guessed")
    ax.set_ylabel("What actually won")
    ax.set_title("Linear SVM: right guesses are on the diagonal")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=200)


def main():
    fig_dir.mkdir(exist_ok=True)
    winner_chart()
    results_chart()
    confusion_chart()
    print("saved charts to figures/")


if __name__ == "__main__":
    main()
