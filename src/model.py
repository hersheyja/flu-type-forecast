# trains the models and checks how often they guess the right winner
# leaves out one whole season at a time, trains on the rest, then guesses that season
# so the model never sees the season its guessing

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

out = Path("data/processed")
viruses = ["H1N1", "H3N2", "B"]

# the clues from features.py
features = ([f"prev_share_{v}" for v in viruses]
            + [f"nat_prev_share_{v}" for v in viruses]
            + [f"seasons_since_{v}" for v in viruses]
            + [f"summer_share_{v}" for v in viruses]
            + ["summer_log_total"])


def top2(model, X):
    # the model's 2 most likely flus for each row
    probs = model.predict_proba(X)
    guesses = []
    for p in probs:
        best_two = np.argsort(p)[::-1][:2]
        guesses.append([model.classes_[i] for i in best_two])
    return guesses


def main():
    df = pd.read_csv(out / "features.csv")
    X = df[features].fillna(df[features].median())
    y = df["winner"]

    results = {"Always the most common virus": [], "Same as last season": [],
               "Logistic regression": [], "Random forest": []}
    results_top2 = {"Always the most common virus": [], "Logistic regression": [],
                    "Random forest": []}

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

        # logistic regression (scale the clues first so theyre all on the same scale)
        scaler = StandardScaler().fit(X_train)
        logreg = LogisticRegression(C=0.5, max_iter=2000)
        logreg.fit(scaler.transform(X_train), y_train)
        results["Logistic regression"] += list(logreg.predict(scaler.transform(X_test)) == y_test)
        guesses = top2(logreg, scaler.transform(X_test))
        results_top2["Logistic regression"] += [w in g for w, g in zip(y_test, guesses)]

        # random forest
        forest = RandomForestClassifier(n_estimators=300, max_depth=4,
                                        min_samples_leaf=5, random_state=0)
        forest.fit(X_train, y_train)
        results["Random forest"] += list(forest.predict(X_test) == y_test)
        guesses = top2(forest, X_test)
        results_top2["Random forest"] += [w in g for w, g in zip(y_test, guesses)]

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
