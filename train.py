from preprocessing import Preprocessor
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import xgboost as xgb

prep = Preprocessor()


df = prep.load_data("history_race.csv")
df = prep.clean_data(df)
df = prep.feature_engineering(df)
df = prep.fit_encoders(df)


prep.save_encoders()


FEATURES = [
    "Driver", "Team", "Track", "Rain",
    "Q1", "Q2", "Q3", "Start",
    "D_Elo", "T_Elo"
]

X = df[FEATURES]
y = df["Finish_rank"]
groups = df["Race_Id"]

gss_1 = GroupShuffleSplit(
    n_splits=1, train_size=0.7, random_state=42
)
train_idx, temp_idx = next(gss_1.split(X, y, groups))

gss_2 = GroupShuffleSplit(
    n_splits=1, train_size=2/3, random_state=42
)
test_sub_idx, hold_sub_idx = next(
    gss_2.split(
        X.iloc[temp_idx],
        y.iloc[temp_idx],
        groups.iloc[temp_idx]
    )
)

test_idx = temp_idx[test_sub_idx]
hold_idx = temp_idx[hold_sub_idx]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test,  y_test  = X.iloc[test_idx],  y.iloc[test_idx]
X_hold,  y_hold  = X.iloc[hold_idx],  y.iloc[hold_idx]

train_groups = groups.iloc[train_idx]
test_groups  = groups.iloc[test_idx]
hold_groups  = groups.iloc[hold_idx]

params = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg",
    "learning_rate": 0.07,
    "max_depth": 9,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.2,
    "reg_lambda": 1,
    "tree_method": "hist",
    "seed": 42
}

gkf = GroupKFold(n_splits=5)
cv_ndcg = []
best_iters = []

for fold, (tr_idx, val_idx) in enumerate(
    gkf.split(X_train, y_train, train_groups), 1
):
    X_tr = X_train.iloc[tr_idx]
    y_tr = y_train.iloc[tr_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    tr_group_sizes = (
        df.iloc[train_idx].iloc[tr_idx]
        .groupby("Race_Id").size().values
    )
    val_group_sizes = (
        df.iloc[train_idx].iloc[val_idx]
        .groupby("Race_Id").size().values
    )

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtrain.set_group(tr_group_sizes)

    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(val_group_sizes)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    ndcg = float(model.eval(dval).split(":")[1])
    cv_ndcg.append(ndcg)
    best_iters.append(model.best_iteration)

    print(f"Fold {fold} CV NDCG: {ndcg:.4f}")

print("Mean CV NDCG:", np.mean(cv_ndcg))

final_groups = (
    df.iloc[train_idx]
    .groupby("Race_Id").size().values
)

dtrain_full = xgb.DMatrix(X_train, label=y_train)
dtrain_full.set_group(final_groups)

model = xgb.train(
    params,
    dtrain_full,
    num_boost_round=int(np.mean(best_iters))
)

test_group_sizes = (
    df.iloc[test_idx]
    .groupby("Race_Id").size().values
)

dtest = xgb.DMatrix(X_test, label=y_test)
dtest.set_group(test_group_sizes)

test_ndcg = float(model.eval(dtest).split(":")[1])
print("Final TEST NDCG:", test_ndcg)


hold_group_sizes = (
    df.iloc[hold_idx]
    .groupby("Race_Id").size().values
)

dhold = xgb.DMatrix(X_hold, label=y_hold)
dhold.set_group(hold_group_sizes)

hold_ndcg = float(model.eval(dhold).split(":")[1])
print("HOLDOUT NDCG:", hold_ndcg)


model.save_model("f1_rank_model.json")