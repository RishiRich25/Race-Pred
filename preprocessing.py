import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder


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

    def encode(self,
               df,
               mode="transform",  # "fit" | "transform" | "update"
               path="encoders.pkl",
               categorical_cols=["Driver", "Team", "Track"],
               save=False):

        for col in categorical_cols:

            # TRAINING MODE
            if mode == "fit":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.encoders[col] = le

            # UPDATE MODE (for new unseen categories)
            elif mode == "update":
                if col not in self.encoders:
                    raise ValueError(f"Encoder for '{col}' not initialized.")

                new_values = set(df[col].unique())
                current_classes = set(self.encoders[col].classes_)
                unseen = new_values - current_classes

                if unseen:
                    print(f"Updating {col} encoder with: {unseen}")
                    updated_classes = np.unique(
                        np.concatenate([self.encoders[col].classes_, list(unseen)])
                    )
                    self.encoders[col].classes_ = updated_classes

                df[col] = self.encoders[col].transform(df[col])

            # TRANSFORM MODE (normal inference)
            elif mode == "transform":
                if col not in self.encoders:
                    raise ValueError(f"Encoder for '{col}' not loaded.")
                df[col] = self.encoders[col].transform(df[col])

            else:
                raise ValueError("Mode must be 'fit', 'transform', or 'update'.")

        if save:
            self.save_encoders(path)

        return df
    

    def save_encoders(self, path="encoders.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.encoders, f)


    def load_encoders(self, path="encoders.pkl"):
        with open(path, "rb") as f:
            self.encoders = pickle.load(f)