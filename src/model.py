# trains the model and checks how often they guess the right winner
# leaves out one whole season at a time, trains on the rest, then guesses that season
# so the model never sees the season its guessing

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

out = Path("data/processed")
viruses = ["H1N1", "H3N2", "B"]

# the clues from features.py
features = ([f"prev_share_{v}" for v in viruses]
            + [f"nat_prev_share_{v}" for v in viruses]
            + [f"seasons_since_{v}" for v in viruses]
            + [f"summer_share_{v}" for v in viruses]
            + ["summer_log_total"])


def top2(scores, classes):
    # the model's 2 highest scoring flus for each row
    guesses = []
    for s in scores:
        best_two = np.argsort(s)[::-1][:2]
        guesses.append([classes[i] for i in best_two])
    return guesses


def main():
    df = pd.read_csv(out / "features.csv")
    X = df[features].fillna(df[features].median())
    y = df["winner"]

    results = {"Always the most common virus": [], "Same as last season": [], "Linear SVM": []}
    results_top2 = {"Always the most common virus": [], "Linear SVM": []}

    for season in sorted(df["season"].unique()):
        # hold out this season, train on all the others
        test = df["season"] == season
        X_train, y_train = X[~test], y[~test]
        X_test, y_test = X[test], y[test]

        # baseline 1: always guess the flu that won the most in training
        most_common = y_train.value_counts().index
        results["Always the most common virus"] += list(y_test == most_common[0])
        results_top2["Always the most common virus"] += list(y_test.isin(most_common[:2]))

        # baseline 2: guess the same flu that won last season
        results["Same as last season"] += list(y_test == df.loc[test, "prev_winner"])

        # linear svm (scale the clues first so theyre all on the same scale)
        scaler = StandardScaler().fit(X_train)
        svm = SVC(kernel="linear", C=0.1)
        svm.fit(scaler.transform(X_train), y_train)
        results["Linear SVM"] += list(svm.predict(scaler.transform(X_test)) == y_test)
        guesses = top2(svm.decision_function(scaler.transform(X_test)), svm.classes_)
        results_top2["Linear SVM"] += [w in g for w, g in zip(y_test, guesses)]

    # % right for each method
    table = pd.DataFrame({
        "method": list(results),
        "accuracy": [np.mean(results[m]) for m in results],
        "top2": [np.mean(results_top2[m]) if m in results_top2 else np.nan for m in results],
    })
    table.to_csv(out / "results.csv", index=False)
    print(f"tested on {len(df)} region-seasons ({df['season'].nunique()} seasons)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.1%}"))


if __name__ == "__main__":
    main()
