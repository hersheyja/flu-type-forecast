# makes the charts: winner grid, results bar chart, confusion matrices, accuracy by date

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

    # before and after balancing, side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, col, title in [(axes[0], "svm_pred", "Linear SVM"),
                           (axes[1], "svm_bal_pred", "Linear SVM (balanced)")]:
        # rows = actual, columns = svm guess
        counts = pd.crosstab(preds["winner"], preds[col])
        counts = counts.loc[viruses, viruses]  # same order as other charts
        ax.imshow(counts, cmap="Blues", vmin=0, vmax=110)

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
        ax.set_xlabel("What the model guessed")
        ax.set_ylabel("What actually won")
        ax.set_title(title)

    fig.suptitle("Right guesses are on the diagonal")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=200)


def date_chart():
    res = pd.read_csv(out / "date_results.csv")
    x = range(len(res))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, res["svm_balanced"], marker="o", color="steelblue", label="Linear SVM (balanced)")
    ax.plot(x, res["leader_so_far"], marker="o", color="darkorange", linestyle="--",
            label="Guess whatever is winning so far")
    ax.axhline(1 / 3, color="gray", linestyle=":", label="Random guess (33%)")

    # % on each point, higher one gets its label on top so they dont overlap
    for i, r in res.iterrows():
        svm, leader = r["svm_balanced"], r["leader_so_far"]
        if np.isnan(leader) or svm >= leader:
            ax.text(i, svm + 0.025, f"{svm:.0%}", ha="center", color="steelblue")
            if not np.isnan(leader):
                ax.text(i, leader - 0.05, f"{leader:.0%}", ha="center", color="darkorange")
        else:
            ax.text(i, leader + 0.025, f"{leader:.0%}", ha="center", color="darkorange")
            ax.text(i, svm - 0.05, f"{svm:.0%}", ha="center", color="steelblue")

    ax.set_xticks(list(x))
    ax.set_xticklabels(res["date"])
    ax.set_ylim(0.25, 1.0)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])
    ax.set_xlabel("Date the prediction is made")
    ax.set_ylabel("% of region-seasons right")
    ax.set_title("Predicting the full-season winner: the later you guess, the easier it gets")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_by_date.png", dpi=200)


def main():
    fig_dir.mkdir(exist_ok=True)
    winner_chart()
    results_chart()
    confusion_chart()
    date_chart()
    print("saved charts to figures/")


if __name__ == "__main__":
    main()
