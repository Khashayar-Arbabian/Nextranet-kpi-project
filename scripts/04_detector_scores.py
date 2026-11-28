"""compute both anomaly scores for every row of every file."""
import pandas as pd                                                 # pandas: tables
from kpi import config as C, data, detect as D                      # our config, loader and detectors

out = []                                                            # one table per file, collected here
for name in data.list_series():                                     # loop over the 17 files
    df = data.load_series(name)                                     # load the cleaned series
    train, _ = data.split(df)                                       # we only need the length of the train part
    y = df["value"].to_numpy()                                      # the series as a plain array
    r = D.one_step_errors(y, len(train))                            # forecast error at every row (train: cross-fitted, test: train-only model)
    out.append(pd.DataFrame(dict(                                   # build this file's score table
        file=name, time=df.index, value=y,                          # which file, each timestamp, the real value
        z72=D.zscore(y, 72),                                        # z-score against the previous 6 hours
        z288=D.zscore(y, 288),                                      # z-score against the previous 24 hours
        err=D.error_score(r, len(train)))))                         # forecast-error score
    print("done", name)                                             # progress message
pd.concat(out).to_csv(C.TABLES / "detector_scores.csv", index=False)   # stack all files and save one big table