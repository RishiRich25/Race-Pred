# Setup Guide

This guide explains how to set up the environment and run the full pipeline end to end.

The steps are intentionally linear so you can stop at any stage and inspect outputs.

---

## Requirements

* Python **3.9 or newer**
* Stable internet connection (FastF1 data download)
* Enough disk space for cached timing data

---

## Install Dependencies

Create a virtual environment if you prefer, then install:

```
pip install -r requirements.txt
```

---

## FastF1 Cache (Important)

FastF1 requires a cache directory to avoid repeated downloads. While a `./cache` directory is in the repository, with `.gitkeep`, please create one in case it was accidentally deleted.

```
mkdir cache
```

---

## Running the Pipeline

### 1. Build Historical Data

This step downloads past race data and updates Elo ratings.

```
python history.py
```

Outputs:

* `history_driver.csv`
* `history_team.csv`
* `history_race.csv`

This step can take time on first run.

---

### 2. Assign Race Groups

Add race-level grouping used by the ranking model:

```
python race_id.py
```

This modifies `history_race.csv` by adding `Race_Id`.

---

### 3. Train the Model

Open the notebook:

```
jupyter notebook model.ipynb
```

The notebook:

* Encodes categorical variables
* Performs grouped train / test / holdout splits
* Trains an XGBoost ranker
* Reports NDCG scores

---

### 4. Update Current Season Elo

To compute Elo ratings for the ongoing season:

```
python current_year.py
```

Outputs:

* `this_year_driver.csv`
* `this_year_team.csv`

These can be used for live or future predictions.

---

## Common Pitfalls

* **Slow first run**: FastF1 downloads a lot of data initially
* **Missing qualifying times**: Filled with zero by design
* **DNFs**: Ranked last intentionally

---

