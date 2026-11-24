# cleans the cdc data -> one row per region per season + which flu won

from pathlib import Path

import pandas as pd

raw = Path("data/raw")
out = Path("data/processed")
viruses = ["H1N1", "H3N2", "B"]

# flu season is week 40 (oct) to week 20 (may)
season_start_week = 40
season_end_week = 20

# under 50 positive tests the winner is basically random
min_positives = 50


def load_csv(path):
    # first line of the cdc files is just a note so skip it
    df = pd.read_csv(path, skiprows=1, na_values=["X"])
    df.columns = [c.strip() for c in df.columns]
    return df


def season_of(year, week):
    # season = the year it starts, so oct 2015 - may 2016 is 2015
    return year if week >= season_start_week else year - 1


def load_weekly():
    old = load_csv(raw / "ICL_NREVSS_Combined_prior_to_2015_16.csv")
    new = load_csv(raw / "ICL_NREVSS_Public_Health_Labs.csv")

    for df in (old, new):
        for col in df.columns[4:]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # old and new files dont have the same columns so match them up
    # counting old H1 and 2009 H1N1 both as H1N1
    old_w = pd.DataFrame({
        "region": old["REGION"], "year": old["YEAR"], "week": old["WEEK"],
        "H1N1": old["A (2009 H1N1)"] + old["A (H1)"],
        "H3N2": old["A (H3)"],
        "B": old["B"],
    })
    # add up all the B types
    new_w = pd.DataFrame({
        "region": new["REGION"], "year": new["YEAR"], "week": new["WEEK"],
        "H1N1": new["A (2009 H1N1)"],
        "H3N2": new["A (H3)"],
        "B": new["B"] + new["BVic"] + new["BYam"],
    })
    weekly = pd.concat([old_w, new_w], ignore_index=True)
    weekly["region"] = weekly["region"].str.replace("Region ", "", regex=False).astype(int)
    weekly["season"] = [season_of(y, w) for y, w in zip(weekly["year"], weekly["week"])]
    weekly = weekly.drop_duplicates(["region", "year", "week"])
    return weekly.sort_values(["region", "year", "week"]).reset_index(drop=True)


def in_flu_season(week):
    return (week >= season_start_week) | (week <= season_end_week)


def build_region_season(weekly):
    flu = weekly[in_flu_season(weekly["week"])]
    totals = flu.groupby(["region", "season"])[viruses].sum().reset_index()

    # skip seasons that arent done yet (need week 20 of the next year)
    finished = []
    for season in totals["season"].unique():
        end_weeks = weekly[(weekly["year"] == season + 1) & (weekly["week"] >= season_end_week)]
        if len(end_weeks) > 0:
            finished.append(season)
    totals = totals[totals["season"].isin(finished)]

    totals["total"] = totals[viruses].sum(axis=1)

    # drop anything under 50 tests
    # this gets rid of region 10 before 2008 and 2020 (covid)
    totals = totals[totals["total"] >= min_positives].copy()

    # winner = flu with the most positive tests
    for v in viruses:
        totals[f"share_{v}"] = totals[v] / totals["total"]
    totals["winner"] = totals[viruses].idxmax(axis=1)

    # ex: 2015-16
    totals["season_label"] = [f"{s}-{str(s + 1)[2:]}" for s in totals["season"]]
    return totals


def main():
    out.mkdir(parents=True, exist_ok=True)
    weekly = load_weekly()
    weekly.to_csv(out / "weekly.csv", index=False)
    rs = build_region_season(weekly)
    rs.to_csv(out / "region_season.csv", index=False)
    print(f"weekly rows: {len(weekly)}")
    print("region-seasons:", len(rs))


if __name__ == "__main__":
    main()
