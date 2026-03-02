import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

FEATURES = [
    "Driver", "Team", "Track", "Rain",
    "Q1", "Q2", "Q3", "Start",
    "D_Elo", "T_Elo"
]


class F1RacePredictor:

    def __init__(self, model_path="f1_rank_model.json", encoder_path="encoders.pkl"):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        with open(encoder_path, "rb") as f:
            self.encoders = pickle.load(f)

    def encode_features(self, df):
        df = df.copy()
        for col in ["Driver", "Team", "Track"]:
            df[col] = self.encoders[col].transform(df[col])
        return df

    def predict_race(self, race_df):
        """
        race_df = dataframe containing ALL drivers of ONE race
        """

        # Encode categorical features
        race_df = self.encode_features(race_df)

        # Keep only model features
        X = race_df[FEATURES]

        # Create DMatrix
        dmatrix = xgb.DMatrix(X)

        # Set group size (IMPORTANT for ranking)
        dmatrix.set_group([len(race_df)])

        # Predict ranking scores
        scores = self.model.predict(dmatrix)

        race_df["Predicted_Score"] = scores

        # Higher score = better predicted finish
        race_df = race_df.sort_values(
            "Predicted_Score",
            ascending=False
        ).reset_index(drop=True)

        race_df["Predicted_Position"] = race_df.index + 1

        return race_df[[
            "Driver",
            "Team",
            "Predicted_Position",
            "Predicted_Score"
        ]]
    
    