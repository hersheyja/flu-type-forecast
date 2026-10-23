# cleans the CDC flu data and makes one row per region per season
# with which flu won (H1N1, H3N2, or B)

from pathlib import Path

import pandas as pd

raw = Path("data/raw")
out = Path("data/processed")
viruses = ["H1N1", "H3N2", "B"]

# flu season is week 40 (oct) to week 20 (may)
season_start_week = 40
season_end_week = 20

# need at least this many positive tests or the winner isnt reliable
min_positives = 50


def load_csv(path):
    # first line of the cdc files is just a note so skip it
    df = pd.read_csv(path, skiprows=1, na_values=["X"])
    df.columns = [c.strip() for c in df.columns]
    return df


def season_of(year, week):
    # label each season by the year it starts (oct 2015 - may 2016 = 2015)
    return year if week >= season_start_week else year - 1


def load_weekly():
    old = load_csv(raw / "ICL_NREVSS_Combined_prior_to_2015_16.csv")
    new = load_csv(raw / "ICL_NREVSS_Public_Health_Labs.csv")

    for df in (old, new):
        for col in df.columns[4:]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # old and new files have different columns so match them up
    # old H1 (before 2009) and 2009 H1N1 both count as H1N1
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

    # skip seasons that arent over yet
    finished = weekly[(weekly["year"] == weekly["season"] + 1)
                      & (weekly["week"] >= season_end_week)]["season"].unique()
    totals = totals[totals["season"].isin(finished)]

    totals["total"] = totals[viruses].sum(axis=1)

    # drop region-seasons with too few tests
    # (region 10 had no data before 2008, and 2020 had almost no flu)
    totals = totals[totals["total"] >= min_positives].copy()

    # winner = the flu with the most positive tests
    for v in viruses:
        totals[f"share_{v}"] = totals[v] / totals["total"]
    totals["winner"] = totals[viruses].idxmax(axis=1)

    totals["season_label"] = totals["season"].astype(str) + "-" + (
        (totals["season"] + 1) % 100).astype(str).str.zfill(2)
    return totals


def main():
    out.mkdir(parents=True, exist_ok=True)
    weekly = load_weekly()
    weekly.to_csv(out / "weekly.csv", index=False)
    rs = build_region_season(weekly)
    rs.to_csv(out / "region_season.csv", index=False)
    print(f"weekly rows: {len(weekly)}")
    print(f"region-seasons: {len(rs)} "
          f"({rs['season'].nunique()} seasons x {rs['region'].nunique()} regions)")
    print(f"seasons: {rs['season_label'].iloc[0]} to {rs['season_label'].iloc[-1]}")


if __name__ == "__main__":
    main()
