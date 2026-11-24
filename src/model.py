# trains the models + checks how often they get the winner right
# hide one season, train on the rest, guess the hidden one, repeat

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

out = Path("data/processed")
viruses = ["H1N1", "H3N2", "B"]

# features from features.py
features = ([f"prev_share_{v}" for v in viruses]
            + [f"nat_prev_share_{v}" for v in viruses]
            + [f"seasons_since_{v}" for v in viruses]
            + [f"summer_share_{v}" for v in viruses]
            + ["summer_log_total"])


def top2(scores, classes):
    # top 2 flus the model picked for each row
    guesses = []
    for s in scores:
        best_two = np.argsort(s)[::-1][:2]
        guesses.append([classes[i] for i in best_two])
    return guesses


# predicting the full season winner at different dates
# number = how many weeks into the season (oct 1 = 0, nov 1 = 4, dec 1 = 8 ...)
prediction_dates = {"Oct 1": None, "Nov 1": 4, "Dec 1": 8, "Jan 1": 13, "Feb 1": 17}


def early_season(weekly, weeks_in):
    # how much of each flu showed up in the first few weeks of the season
    w = weekly.copy()
    w["weeks_in"] = np.where(w["week"] >= 40, w["week"] - 40, w["week"] + 12)
    in_season = (w["week"] >= 40) | (w["week"] <= 20)
    w = w[in_season & (w["weeks_in"] <= weeks_in)]

    early = w.groupby(["region", "season"])[viruses].sum().reset_index()
    total = early[viruses].sum(axis=1)
    for v in viruses:
        early[f"early_share_{v}"] = (early[v] + 1) / (total + 3)
    early["early_log_total"] = np.log1p(total)

    # simple guess to compare to: whatever is winning so far
    early["leader"] = early[viruses].idxmax(axis=1)
    return early.drop(columns=viruses)


def try_dates(df):
    # same balanced svm, but also using the early season weeks if we wait to guess
    weekly = pd.read_csv(out / "weekly.csv")
    rows = []
    for date, weeks_in in prediction_dates.items():
        if weeks_in is None:
            data = df.copy()
            feats = features
        else:
            data = df.merge(early_season(weekly, weeks_in), on=["region", "season"], how="left")
            feats = features + [f"early_share_{v}" for v in viruses] + ["early_log_total"]

        X = data[feats].fillna(data[feats].median())
        y = data["winner"]
        right = []
        for season in sorted(data["season"].unique()):
            test = data["season"] == season
            scaler = StandardScaler().fit(X[~test])
            svm = SVC(kernel="linear", C=0.1, class_weight="balanced")
            svm.fit(scaler.transform(X[~test]), y[~test])
            right += list(svm.predict(scaler.transform(X[test])) == y[test])

        if weeks_in is None:
            leader_accuracy = np.nan  # no season data yet on oct 1
        else:
            leader_accuracy = (data["leader"] == y).mean()
        rows.append({"date": date, "svm_balanced": np.mean(right), "leader_so_far": leader_accuracy})

    table = pd.DataFrame(rows)
    table.to_csv(out / "date_results.csv", index=False)
    print("accuracy by prediction date:")
    print(table.round(3))


def try_svm_c(df, X, y):
    # testing different C values for the svm
    # small C = simpler line, big C = tries to fit every training point
    rows = []
    for c in [0.01, 0.1, 1, 10]:
        right = []
        for season in sorted(df["season"].unique()):
            test = df["season"] == season
            scaler = StandardScaler().fit(X[~test])
            svm = SVC(kernel="linear", C=c)
            svm.fit(scaler.transform(X[~test]), y[~test])
            right += list(svm.predict(scaler.transform(X[test])) == y[test])
        rows.append({"C": c, "accuracy": np.mean(right)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "svm_c_results.csv", index=False)
    print("svm C values:")
    print(table.round(3))


def main():
    df = pd.read_csv(out / "features.csv")
    X = df[features].fillna(df[features].median())
    y = df["winner"]

    results = {"Always the most common virus": [], "Same as last season": [],
               "Linear SVM": [], "Logistic regression": [], "Linear SVM (balanced)": []}
    results_top2 = {"Always the most common virus": [], "Linear SVM": [],
                    "Logistic regression": [], "Linear SVM (balanced)": []}

    # save guesses for the confusion matrix
    preds = df[["region", "season", "winner"]].copy()
    preds["svm_pred"] = ""
    preds["logreg_pred"] = ""
    preds["svm_bal_pred"] = ""

    for season in sorted(df["season"].unique()):
        # hold out this season, train on all the others
        test = df["season"] == season
        X_train, y_train = X[~test], y[~test]
        X_test, y_test = X[test], y[test]

        # baseline 1: always guess the most common flu
        most_common = y_train.value_counts().index
        results["Always the most common virus"] += list(y_test == most_common[0])
        results_top2["Always the most common virus"] += list(y_test.isin(most_common[:2]))

        # baseline 2: guess whatever won last season
        results["Same as last season"] += list(y_test == df.loc[test, "prev_winner"])

        # scale first, svm cares about distance
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        # linear svm
        svm = SVC(kernel="linear", C=0.1)
        svm.fit(X_train_s, y_train)
        svm_guess = svm.predict(X_test_s)
        preds.loc[test, "svm_pred"] = svm_guess
        results["Linear SVM"] += list(svm_guess == y_test)
        guesses = top2(svm.decision_function(X_test_s), svm.classes_)
        results_top2["Linear SVM"] += [w in g for w, g in zip(y_test, guesses)]

        # logistic regression
        logreg = LogisticRegression(max_iter=2000)
        logreg.fit(X_train_s, y_train)
        logreg_guess = logreg.predict(X_test_s)
        preds.loc[test, "logreg_pred"] = logreg_guess
        results["Logistic regression"] += list(logreg_guess == y_test)
        guesses = top2(logreg.predict_proba(X_test_s), logreg.classes_)
        results_top2["Logistic regression"] += [w in g for w, g in zip(y_test, guesses)]

        # balanced svm: mistakes on rare flus (like B) count more
        # so it doesnt just guess H3N2 all the time (same idea as worksheet 9)
        svm_bal = SVC(kernel="linear", C=0.1, class_weight="balanced")
        svm_bal.fit(X_train_s, y_train)
        svm_bal_guess = svm_bal.predict(X_test_s)
        preds.loc[test, "svm_bal_pred"] = svm_bal_guess
        results["Linear SVM (balanced)"] += list(svm_bal_guess == y_test)
        guesses = top2(svm_bal.decision_function(X_test_s), svm_bal.classes_)
        results_top2["Linear SVM (balanced)"] += [w in g for w, g in zip(y_test, guesses)]

    preds.to_csv(out / "predictions.csv", index=False)

    # accuracy for each method
    rows = []
    for method in results:
        accuracy = np.mean(results[method])
        # same as last season only has 1 guess so no top 2
        if method in results_top2:
            top2_accuracy = np.mean(results_top2[method])
        else:
            top2_accuracy = np.nan
        rows.append({"method": method, "accuracy": accuracy, "top2": top2_accuracy})
    table = pd.DataFrame(rows)
    table.to_csv(out / "results.csv", index=False)
    print(table.round(3))

    # % right for each flu, svm before vs after balancing
    print("% right for each flu:")
    for v in viruses:
        rows_v = preds[preds["winner"] == v]
        before = (rows_v["svm_pred"] == v).mean()
        after = (rows_v["svm_bal_pred"] == v).mean()
        print(f"  {v}: svm {before:.0%}, balanced svm {after:.0%}")

    try_svm_c(df, X, y)
    try_dates(df)


if __name__ == "__main__":
    main()
