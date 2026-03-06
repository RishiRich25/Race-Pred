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
schedule = fastf1.get_event_schedule(current_year)
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

st.success("Qualifying session detected ✅")

# ===============================
# Build Feature Table
# ===============================
race_df = pd.DataFrame({
    "Driver": quali["FullName"],
    "Team": quali["TeamName"],
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
run_elo_update()
driver_elo = pd.read_csv("this_year_driver.csv")
team_elo = pd.read_csv("this_year_team.csv")

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

race_df["D_Elo"] = race_df["D_Elo"].fillna(race_df["D_Elo"].mean())
race_df["T_Elo"] = race_df["T_Elo"].fillna(race_df["T_Elo"].mean())

# ===============================
# Encode
# ===============================
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
        "Driver",
        "Team",
        "Predicted_Score"
    ]]
)

