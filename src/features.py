# makes the features for the model
# only stuff we'd know by oct 1 so the model isnt seeing the future

from pathlib import Path

import numpy as np
import pandas as pd

out = Path("data/processed")
viruses = ["H1N1", "H3N2", "B"]

# summer = late may - sept (between seasons)
summer_weeks = (21, 39)

# cap seasons since at 10, anything older doesnt really matter
max_seasons_since = 10


def summer_features(weekly):
    # how much of each flu showed up the summer before
    s = weekly[(weekly["week"] >= summer_weeks[0]) & (weekly["week"] <= summer_weeks[1])]
    s = s.groupby(["region", "year"])[viruses].sum().reset_index()

    # summer 2015 goes with the 2015-16 season
    s = s.rename(columns={"year": "season"})
    s["summer_total"] = s[viruses].sum(axis=1)

    # +1 so we dont divide by 0 when a summer has no tests
    for v in viruses:
        s[f"summer_share_{v}"] = (s[v] + 1) / (s["summer_total"] + 3)
    s["summer_log_total"] = np.log1p(s["summer_total"])
    return s[["region", "season", "summer_log_total"] + [f"summer_share_{v}" for v in viruses]]


def history_features(rs):
    # last season (region + whole US) and seasons since each flu won
    rows = []
    # % of each flu in the whole US each season
    national = rs.groupby("season")[viruses].sum()
    national_total = national.sum(axis=1)
    for v in viruses:
        national[v] = national[v] / national_total

    for region, grp in rs.groupby("region"):
        grp = grp.sort_values("season")
        last_won = {v: None for v in viruses}
        prev = None
        for _, r in grp.iterrows():
            feat = {"region": region, "season": r["season"]}

            # last season in this region
            if prev is not None:
                feat["prev_winner"] = prev["winner"]
                for v in viruses:
                    feat[f"prev_share_{v}"] = prev[f"share_{v}"]

            # last season in the whole US
            nat_prev = national[national.index < r["season"]]
            if len(nat_prev):
                for v in viruses:
                    feat[f"nat_prev_share_{v}"] = nat_prev.iloc[-1][v]

            # seasons since each flu last won here
            for v in viruses:
                if last_won[v] is None:
                    since = max_seasons_since
                else:
                    since = min(r["season"] - last_won[v], max_seasons_since)
                feat[f"seasons_since_{v}"] = since

            rows.append(feat)
            last_won[r["winner"]] = r["season"]
            prev = r
    return pd.DataFrame(rows)


def main():
    weekly = pd.read_csv(out / "weekly.csv")
    rs = pd.read_csv(out / "region_season.csv")
    feats = history_features(rs).merge(summer_features(weekly), on=["region", "season"], how="left")
    data = rs[["region", "season", "season_label", "winner"]].merge(
        feats, on=["region", "season"])

    # first season in each region has no last season so drop it
    data = data.dropna(subset=["prev_winner"]).reset_index(drop=True)
    data.to_csv(out / "features.csv", index=False)
    print("feature rows:", len(data))


if __name__ == "__main__":
    main()
