# makes the clues (features) the model uses to guess each season's winner
# only uses stuff we would know by oct 1, before the season starts

from pathlib import Path

import numpy as np
import pandas as pd

out = Path("data/processed")
viruses = ["H1N1", "H3N2", "B"]

# summer = late may to september, the gap between flu seasons
summer_weeks = (21, 39)

# cap "seasons since it last won" at 10 so really old stuff doesnt matter
max_seasons_since = 10


def summer_features(weekly):
    # how much of each flu showed up over the summer right before the season
    s = weekly[(weekly["week"] >= summer_weeks[0]) & (weekly["week"] <= summer_weeks[1])]
    s = s.groupby(["region", "year"])[viruses].sum().reset_index()

    # summer 2015 comes right before the 2015-16 season
    s = s.rename(columns={"year": "season"})
    s["summer_total"] = s[viruses].sum(axis=1)

    # +1 so a summer with 0 tests doesnt divide by 0
    for v in viruses:
        s[f"summer_share_{v}"] = (s[v] + 1) / (s["summer_total"] + 3)
    s["summer_log_total"] = np.log1p(s["summer_total"])
    return s[["region", "season", "summer_log_total"] + [f"summer_share_{v}" for v in viruses]]


def history_features(rs):
    # what happened last season (in the region and the whole US)
    # and how many seasons since each flu last won
    rows = []
    national = rs.groupby("season")[viruses].sum()
    national = national.div(national.sum(axis=1), axis=0)

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
                since = max_seasons_since if last_won[v] is None else min(
                    r["season"] - last_won[v], max_seasons_since)
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

    # first season in each region has no "last season" so skip it
    data = data.dropna(subset=["prev_winner"]).reset_index(drop=True)
    data.to_csv(out / "features.csv", index=False)
    print(f"feature rows: {len(data)}, columns: {data.shape[1]}")


if __name__ == "__main__":
    main()
