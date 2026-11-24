import numpy as np
import pandas as pd
from kpi import config as C, data, evaluate as E

rows = []
for name in data.list_series():
    df = data.load_series(name)
    train, test = data.split(df)
    y = df["value"].to_numpy()
    imputed = df["imputed"].to_numpy().astype(float)

    o = E.origins(len(y), len(train))
    truth = E.actuals(y, o)
    skip = E.actuals(imputed, o) > 0
    scale = E.mase_scale(train["value"].to_numpy())

    for model, pred in [("last_value", E.forecast_last(y, o)),
                        ("yesterday", E.forecast_yesterday(y, o))]:
        mae = E.mae_per_horizon(truth, pred, skip)
        for h in range(C.HORIZON):
            rows.append(dict(file=name, model=model, minutes=5 * (h + 1),
                             MAE=mae[h], MASE=mae[h] / scale,
                             origins=len(o), scale=scale))

long = pd.DataFrame(rows)
long.to_csv(C.TABLES / "baselines_by_horizon.csv", index=False)

by_file = long.groupby(["file", "model"])[["MAE", "MASE"]].mean().unstack()
by_file.columns = [f"{a}_{b}" for a, b in by_file.columns]
by_file["winner"] = np.where(by_file["MAE_last_value"] <= by_file["MAE_yesterday"],
                             "last_value", "yesterday")
by_file.to_csv(C.TABLES / "baselines_by_file.csv")

pd.set_option("display.width", 200)
print(by_file.round(3).to_string())
print(by_file["winner"].value_counts())
print("origins per file:", long.groupby("file")["origins"].first().min(), "to",
      long.groupby("file")["origins"].first().max())