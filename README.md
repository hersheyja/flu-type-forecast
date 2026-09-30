# Flu Type Forecast

CS 506 Final Project Proposal
Name: Hershey Jamla

## Project Description

Every flu season, three main seasonal flu viruses circulate in the U.S.:
influenza A(H1N1), influenza A(H3N2), and influenza B (currently B/Victoria).
Usually, one or two of these become predominant, and which virus predominates
can vary from year to year and by region.

For this project, I will try to predict which virus will predominate in each
U.S. region before the flu season starts. I'll use past U.S. lab data along
with flu data from the Southern Hemisphere, whose flu season happens during
our summer.

If this turns out to be too ambitious, I'll simplify it to predicting one
virus for the whole country, or whether the season will be mostly flu A
or flu B.

## Project Timeline

- Weeks 1–2: Write scripts to collect the CDC and WHO data
- Weeks 3–4: Clean the data and build a dataset with one row per region per season (October check-in)
- Weeks 5–7: Create features, train models, and compare them to baselines (November check-in)
- Week 8: Predict the 2026–27 season and compare with early season data
- Weeks 9–10: Makefile, tests, GitHub workflow, final report, and presentation

## Project Goals

The goal is to predict, as of October 1, which virus (A(H1N1), A(H3N2), or B)
will have the most positive lab tests in each of the CDC's 10 HHS regions over
the October–May flu season. Most flu activity usually peaks between December and February,
but the full season is used to capture late-season influenza B activity.

Features I plan to look at:
- Which virus predominated in the region last season
- Which virus predominated in the most recent Southern Hemisphere season
- How many seasons it has been since each virus last predominated
- Flu detections in the region over the summer

I'll measure how often the model predicts the correct virus, and how often
the correct virus is in its top two predictions, since some seasons have two
viruses circulating at similar levels. To be useful, the model needs to beat
two simple guesses: last season's virus, and the Southern Hemisphere's virus.

## Data Collection Plan

- [CDC FluView](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html):
  weekly positive flu tests by virus subtype and HHS region, going back to the
  1997–98 season. Recent seasons will be pulled through the
  [Delphi Epidata API](https://cmu-delphi.github.io/delphi-epidata/) from
  Carnegie Mellon, and older seasons will be downloaded from FluView directly.
- [WHO FluNet](https://www.who.int/tools/flunet): weekly flu detections by
  subtype for countries around the world. I'll use data from Southern
  Hemisphere countries like Australia.

All data will be collected with Python scripts, and I'll save a copy of the
raw data in the repo so results can be reproduced if a source changes.
