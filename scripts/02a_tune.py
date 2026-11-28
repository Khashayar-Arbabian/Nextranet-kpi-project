"""Pick the model settings using only a validation slice inside the TRAIN part."""
import numpy as np                                                  # NumPy: arrays and maths
from kpi import data, evaluate as E, model as M                     # our loader, evaluation functions and model

candidates = {                                                      # four sets of settings to compare
    "A": dict(max_iter=100, learning_rate=0.05, max_depth=3, min_samples_leaf=40),   # 100 trees, small steps
    "B": dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=40),   # same but 200 trees
    "C": dict(max_iter=100, learning_rate=0.1, max_depth=2, min_samples_leaf=100),   # bigger steps, shallower trees
    "D": dict(max_iter=50, learning_rate=0.05, max_depth=3, min_samples_leaf=100),   # only 50 trees
}
ratios = {k: [] for k in candidates}                                # one list of results per candidate

for name in data.list_series():                                     # loop over the 17 files
    df = data.load_series(name)                                     # load the cleaned series
    train, _ = data.split(df)                                       # keep only the train part (the test part is never used here)
    y = df["value"].to_numpy()                                      # the series as a plain array
    cut_val = int(len(train) * 0.8)                                 # first 80% of train = used for training the model
    o = np.arange(cut_val - 1, len(train) - 12, 12)                 # hourly forecast origins inside the LAST 20% of train
    truth = E.actuals(y, o)                                         # the real values after each origin (answer key)
    base = E.mae_per_horizon(truth, E.forecast_last(y, o)).mean()   # error of the last-value baseline on this slice
    for k, params in candidates.items():                            # try each set of settings
        pred = M.fit_predict(y, cut_val, o, **params)               # train on the first 80% of train, forecast the last 20%
        ratios[k].append(E.mae_per_horizon(truth, pred).mean() / base)   # model error divided by last-value error (below 1 = better)

for k, v in ratios.items():                                         # print one line per candidate
    print(k, "median error vs last value:", round(float(np.median(v)), 3),     # typical ratio over the 17 files
          "| files better than last value:", int((np.array(v) < 1).sum()), "of 17")   # in how many files the ratio is below 1