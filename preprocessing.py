import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

class Preprocessor:

    def __init__(self):
        self.encoders = {}

    def load_data(self, path):
        return pd.read_csv(path, encoding="latin")

    def clean_data(self, df):
        df["Finish"] = df["Finish"].fillna(23).astype(int)
        df[["Q1", "Q2", "Q3"]] = df[["Q1", "Q2", "Q3"]].fillna(0)
        return df

    def feature_engineering(self, df):
        df["Finish_rank"] = 24 - df["Finish"]
        return df

    def fit_encoders(self, df):
        for col in ["Driver", "Team", "Track"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
        return df

    def transform_encoders(self, df):
        for col in ["Driver", "Team", "Track"]:
            df[col] = self.encoders[col].transform(df[col])
        return df

    def save_encoders(self, path="encoders.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.encoders, f)