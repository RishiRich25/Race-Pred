import streamlit as st
import pandas as pd
import os
import xgboost as xgb
import fastf1
from datetime import datetime

from current_year import run_elo_update
from preprocessing import Preprocessor

cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)    

fastf1.Cache.enable_cache("cache")

st.set_page_config(page_title="Live F1 Race Predictor", layout="wide")
st.title("🏎️ Automatic Live F1 Race Predictor")

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
    (f"### Quali Data Not Available")
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
    "Start": quali["GridPosition"],
    "Track": event_name,
    "Rain": 0
})

# ===============================
# 🔥 LOAD ELO HERE (Dynamic)
# ===============================

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
