"""train the model per file and score it on the test part, next to the baselines."""
import numpy as np                                                  # NumPy: arrays and maths
import pandas as pd                                                 # pandas: tables
from kpi import config as C, data, evaluate as E, model as M        # our config, loader, evaluation functions and model

PARAMS = M.PARAMS                                                   # the model settings chosen in the tuning step

rows, fc = [], []                                                   # rows = metric table, fc = saved test forecasts
for name in data.list_series():                                     # loop over the 17 files
    df = data.load_series(name)                                     # load the cleaned series
    train, test = data.split(df)                                    # 70% train, 30% test
    y = df["value"].to_numpy()                                      # the series as a plain array
    imputed = df["imputed"].to_numpy().astype(float)                # 1.0 where a value was filled in on Day 1, else 0.0

    o = E.origins(len(y), len(train))                               # the ~100 hourly forecast origins on the test part
    truth = E.actuals(y, o)                                         # the real values after each origin (answer key)
    skip = E.actuals(imputed, o) > 0                                # True where the real value was interpolated (ignore these)
    scale = E.mase_scale(train["value"].to_numpy())                 # MASE divisor, from the train part only

    preds = {"last_value": E.forecast_last(y, o),                   # baseline 1
             "yesterday": E.forecast_yesterday(y, o),               # baseline 2
             "mean_1h": E.forecast_mean(y, o),                      # baseline 3 (average of the last hour)
             "gboost": M.fit_predict(y, len(train), o, **PARAMS)}   # our model: trained on train only, forecasts the test origins

    for model, pred in preds.items():                               # score each of the four forecasters
        mae = E.mae_per_horizon(truth, pred, skip)                  # 12 MAE values, one per horizon
        for h in range(C.HORIZON):                                  # one table row per horizon
            rows.append(dict(file=name, model=model, minutes=5 * (h + 1),   # which file, model and horizon
                             MAE=mae[h], MASE=mae[h] / scale, origins=len(o)))   # the two metrics and the number of forecasts

    for h in range(C.HORIZON):                                      # keep every test forecast for the error analysis
        fc.append(pd.DataFrame(dict(
            file=name, origin_time=df.index[o], minutes=5 * (h + 1),                # when and how far ahead
            actual=truth[:, h], gboost=preds["gboost"][:, h],                       # the real values and the model forecasts
            last_value=preds["last_value"][:, h], yesterday=preds["yesterday"][:, h],   # the two baseline forecasts
            imputed=skip[:, h])))                                                   # which targets were interpolated
    print("done", name)                                             # progress message

long = pd.DataFrame(rows)                                           # the full metric table (17 files x 4 models x 12 horizons)
long.to_csv(C.TABLES / "models_by_horizon.csv", index=False)        # save it
pd.concat(fc).to_csv(C.TABLES / "test_forecasts.csv", index=False)  # save all test forecasts

by_file = long.groupby(["file", "model"])[["MAE", "MASE"]].mean().unstack()   # average over the 12 horizons: one line per file
by_file.columns = [f"{a}_{b}" for a, b in by_file.columns]          # flatten column names, e.g. MAE_gboost
maes = by_file[["MAE_last_value", "MAE_yesterday", "MAE_mean_1h", "MAE_gboost"]]   # just the four MAE columns
by_file["winner"] = maes.idxmin(axis=1).str.replace("MAE_", "")     # the forecaster with the lowest MAE in each file
by_file["gboost_vs_best_baseline"] = by_file["MAE_gboost"] / maes[["MAE_last_value", "MAE_yesterday"]].min(axis=1)   # model MAE / better required baseline
by_file.to_csv(C.TABLES / "models_by_file.csv")                     # save the per-file table

pd.set_option("display.width", 220)                                 # print wide tables on one line
print(by_file[["MASE_last_value", "MASE_yesterday", "MASE_mean_1h", "MASE_gboost",
               "gboost_vs_best_baseline", "winner"]].round(3).to_string())   # the main results table
print(by_file["winner"].value_counts())                             # how many files each forecaster wins
print("gboost beats BOTH required baselines on",
      int((by_file["gboost_vs_best_baseline"] < 1).sum()), "of 17 files")   # count of files where the ratio is below 1