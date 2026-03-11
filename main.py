import streamlit as st
import pandas as pd
import os
import xgboost as xgb
import fastf1
from datetime import datetime

from preprocessing import Preprocessor

cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)    

fastf1.Cache.enable_cache("cache")

st.set_page_config(page_title="Live F1 Race Predictor", layout="wide",page_icon="🏎️")
st.title("🏎️ Automatic Live F1 Race Predictor")
st.markdown(
"""
RacePred uses **machine learning + FastF1 session data** to predict the finishing
order of Formula 1 races based on qualifying results and historical performance.
(Scroll to the end to see live weekend predictions available on Saturday)
"""
)


st.title("🏎️ RacePred Project Workflow")
st.subheader("How the Formula 1 Race Prediction System Works")

st.markdown("""
The **complete workflow of the RacePred project** —
from collecting Formula 1 data to generating race predictions using machine learning.
""")
st.divider()


st.header("📊 Pipeline Overview")

st.code("""
FastF1 Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Driver & Team Elo Ratings
        │
        ▼
Machine Learning Model (XGBoost)
        │
        ▼
Prediction Scores
        │
        ▼
Sorted Finishing Order
""")

st.divider()

st.sidebar.title("Workflow Navigation")

step = st.sidebar.radio(
    "Select a Stage",
    [
        "1️⃣ Data Collection",
        "2️⃣ Data Cleaning",
        "3️⃣ Feature Engineering",
        "4️⃣ Elo Ratings",
        "5️⃣ Model Training",
        "6️⃣ Race Prediction",
        "7️⃣ Ranking Drivers",
        "8️⃣ Output Results"
    ]
)

# -----------------------------------
# STEP 1
# -----------------------------------

if step == "1️⃣ Data Collection":

    st.header("1️⃣ Data Collection")

    st.write("""
RacePred retrieves official Formula 1 session data using the **FastF1 API**.

This includes:
- qualifying results
- race results
- driver information
- team information
- lap timing data
""")

    st.code("""
import fastf1

fastf1.Cache.enable_cache("cache")

session = fastf1.get_session(2024, 5, "Q")
session.load()

results = session.results
""", language="python")


# -----------------------------------
# STEP 2
# -----------------------------------

elif step == "2️⃣ Data Cleaning":

    st.header("2️⃣ Data Cleaning")

    st.write("""
Raw data from FastF1 must be cleaned and formatted before being used
in a machine learning model.

Typical cleaning steps:
- remove missing values
- standardize driver/team names
- convert results into tabular format
""")

    st.code("""
import pandas as pd

df = pd.DataFrame({
    "Driver": results["Abbreviation"],
    "Team": results["TeamName"],
    "QualiPos": results["Position"]
})
""", language="python")


# -----------------------------------
# STEP 3
# -----------------------------------

elif step == "3️⃣ Feature Engineering":

    st.header("3️⃣ Feature Engineering")

    st.write("""
Machine learning models require **numerical features**.

The project generates additional variables that help the model
learn patterns in race outcomes.
""")

    st.code("""
df["AvgFinish"] = historical_data.groupby("Driver")["Finish"].mean()

df["GridAdvantage"] = df["QualiPos"] - df["AvgFinish"]
""", language="python")


# -----------------------------------
# STEP 4
# -----------------------------------

elif step == "4️⃣ Elo Ratings":

    st.header("4️⃣ Driver & Team Elo Ratings")

    st.write("""
The project uses **Elo ratings** to estimate the relative strength
of drivers and teams.

Higher Elo → stronger expected performance.
""")

    st.code("""
race_df = race_df.merge(
    driver_elo[["Driver", "Elo"]],
    on="Driver",
    how="left"
).rename(columns={"Elo": "D_Elo"})

race_df = race_df.merge(
    team_elo[["Team", "Elo"]],
    on="Team",
    how="left"
).rename(columns={"Elo": "T_Elo"})
""", language="python")


# -----------------------------------
# STEP 5
# -----------------------------------

elif step == "5️⃣ Model Training":

    st.header("5️⃣ Model Training")

    st.write("""
RacePred trains a machine learning model to learn the relationship
between race features and finishing positions.

The current implementation uses **XGBoost**.
""")

    st.code("""
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)
""", language="python")


# -----------------------------------
# STEP 6
# -----------------------------------

elif step == "6️⃣ Race Prediction":

    st.header("6️⃣ Race Prediction")

    st.write("""
Once qualifying is completed, the model generates predictions
for the upcoming race.
""")

    st.code("""
features = df[["QualiPos", "D_Elo", "T_Elo"]]

predictions = model.predict(features)
""", language="python")


# -----------------------------------
# STEP 7
# -----------------------------------

elif step == "7️⃣ Ranking Drivers":

    st.header("7️⃣ Ranking Drivers")

    st.write("""
The model outputs a **performance score** for each driver.

Drivers are sorted by this score to determine the predicted
finishing order.
""")

    st.code("""
df["PredictionScore"] = predictions

df = df.sort_values("PredictionScore")

df["PredictedPosition"] = range(1, len(df) + 1)
""", language="python")


# -----------------------------------
# STEP 8
# -----------------------------------

elif step == "8️⃣ Output Results":

    st.header("8️⃣ Output Results")

    st.write("""
The final result is a predicted race finishing order.

Example output:
""")

    st.code("""
Predicted Race Finish

P1  Max Verstappen
P2  Charles Leclerc
P3  Lando Norris
P4  Lewis Hamilton
""")

# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

# ===============================
# Load Model + Encoders (cached)
# ===============================
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model("f1_rank_model.json")
    return model

@st.cache_resource
def load_preprocessor():
    prep = Preprocessor()
    prep.load_encoders("encoders.pkl")
    return prep

model = load_model()
prep = load_preprocessor()

# ===============================
# Detect Current Event
# ===============================
current_year = datetime.now().year
try:
    schedule = fastf1.get_event_schedule(current_year, backend="ergast")
except:
    st.error("Could not load F1 schedule")
    st.stop()
now = pd.Timestamp.now()

next_event = schedule[schedule["EventDate"] >= now].iloc[0]
event_name = next_event["EventName"]
round_number = next_event["RoundNumber"]

st.write(f"### 📍 Next Event: {event_name} (Round {round_number})")

# ===============================
# Load Qualifying
# ===============================
try:
    session = fastf1.get_session(current_year, round_number, "Q")
    session.load()
    quali = session.results
except Exception:
    st.warning("Qualifying not available yet.")
    st.stop()

if len(quali) == 0:
    st.write(f"### Quali Data Not Available")
    st.write(f"#### Come back after qualifying is over on a Saturday")

else:
    st.success("Qualifying session detected ✅")

    # ===============================
    # Build Feature Table
    # ===============================
    race_df = pd.DataFrame({
        "Driver": quali["DriverId"],
        "Team": quali["TeamId"],
        "Q1": quali["Q1"].dt.total_seconds().fillna(0),
        "Q2": quali["Q2"].dt.total_seconds().fillna(0),
        "Q3": quali["Q3"].dt.total_seconds().fillna(0),
        "Start": quali["Position"],
        "Track": event_name,
        "Rain": 0
    })

    # ===============================
    # 🔥 LOAD ELO HERE (Dynamic)
    # ===============================
    race_df = race_df["Driver"].dropna()
    driver_elo = pd.read_csv("this_year_driver.csv", encoding="latin1")
    driver_elo = driver_elo.rename(columns={
        "Name": "Driver"
    })
    team_elo = pd.read_csv("this_year_team.csv", encoding="latin1")
    team_elo = team_elo.rename(columns={
        "Name": "Team"
    })

    race_df["Driver_Name"] = race_df["Driver"]
    race_df["Team_Name"] = race_df["Team"]

    race_df = race_df.merge(
        driver_elo[["Driver", "Elo"]],
        on="Driver",
        how="left"
    ).rename(columns={"Elo": "D_Elo"})

    race_df = race_df.merge(
        team_elo[["Team", "Elo"]],
        on="Team",
        how="left"
    ).rename(columns={"Elo": "T_Elo"})

    race_df["D_Elo"] = race_df["D_Elo"].fillna(1200)
    race_df["T_Elo"] = race_df["T_Elo"].fillna(1800)

    # ===============================
    # Encode
    # ===============================
    FEAT = [
        "Driver", "Team", "Start",
        "D_Elo", "T_Elo"
    ]
    st.write(f"#### Elo for Event: {event_name}")
    st.dataframe(race_df[FEAT])

    race_df = prep.encode(
        race_df,
        mode="update",
        save=True
    )

    FEATURES = [
        "Driver", "Team", "Track", "Rain",
        "Q1", "Q2", "Q3", "Start",
        "D_Elo", "T_Elo"
    ]

    dtest = xgb.DMatrix(race_df[FEATURES])
    dtest.set_group([len(race_df)])

    scores = model.predict(dtest)

    race_df["Predicted_Score"] = scores
    race_df = race_df.sort_values(
        by="Predicted_Score",
        ascending=False
    ).reset_index(drop=True)

    race_df["Predicted_Position"] = race_df.index + 1

    st.write("## 🏁 Predicted Race Classification")
    st.dataframe(
        race_df[[
            "Predicted_Position",
            "Driver_Name",
            "Team_Name",
            "Start"
        ]].rename(columns={
            "Driver_Name": "Driver",
            "Team_Name": "Team",
            "Start":"Starting Position"
        })
    )
st.markdown(
"""
⚠️ **Disclaimer**

This project is a data science experiment and is not affiliated with Formula 1 or the FIA.
Predictions are probabilistic and may not reflect actual race outcomes.
"""
)