# 🏎️ RacePred

This project predicts Formula 1 race outcomes using a **hybrid approach** that combines:

* 📊 **Custom Elo ratings** for drivers and teams
* 🌦️ Contextual race features (qualifying pace, weather, track)
* 🤖 **XGBoost Learning-to-Rank (NDCG)** model

The system continuously updates historical and current-season Elo ratings using FastF1 data, then trains a ranking model to predict race finishing order.

---

## Core Concepts

### Elo Rating System

* Drivers start at **1200 Elo**, teams at **1800 Elo**
* Elo updates are based on:

  * Grid position vs finishing position
  * Race vs Sprint weighting
  * Rain-adjusted volatility (higher K-factor)


---

### Feature Engineering

Each race entry includes:

| Feature  | Description                    |
| -------- | ------------------------------ |
| Driver   | Encoded driver ID              |
| Team     | Encoded constructor            |
| Track    | Circuit name                   |
| Rain     | Boolean weather flag           |
| Q1/Q2/Q3 | Qualifying lap times (seconds) |
| Start    | Grid position                  |
| D_Elo    | Driver Elo before race         |
| T_Elo    | Team Elo before race           |

Target variable:

```
Finish_rank = 24 - Finish_Position
```

(Higher is better)

---

### Machine Learning Model

* **Model:** XGBoost (rank:ndcg)
* **Validation:** GroupKFold (by Race_Id)
* **Metrics:** NDCG

**Performance:**

* Mean CV NDCG ≈ **0.85**
* Final Test NDCG ≈ **0.84**
* Holdout NDCG ≈ **0.90**

---

## Data Flow Pipeline

```
FastF1 API
   ↓
Elo Updates (history.py)
   ↓
history_race.csv
   ↓
Race Grouping (race_id.py)
   ↓
Feature Encoding
   ↓
XGBoost Ranker (model.ipynb)
```

---

