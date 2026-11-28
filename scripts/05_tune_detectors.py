"""choose the alarm threshold of each detector using ONLY the train part."""
import matplotlib                                                   # plotting library
matplotlib.use("Agg")                                               # draw into image files (no window)
import matplotlib.pyplot as plt                                     # the plotting functions
import pandas as pd                                                 # pandas: tables
from kpi import config as C, data, detect as D                      # our config, loader and detectors

scores = pd.read_csv(C.TABLES / "detector_scores.csv", parse_dates=["time"])   # the scores computed by 04_detector_scores.py
GRID = {"global": [3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000],     # global: the same number for every file
        "per_file": [0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 0.99995, 1.0]}  # per_file: a quantile of each file's own train scores
DETECTORS = ["z72", "z288", "err"]                                  # the three score columns to tune

rows = []                                                           # one result row per file, detector, method and value
for name in data.list_series():                                     # loop over the 17 files
    g = scores[scores["file"] == name].reset_index(drop=True)       # the scores of this file
    cut = len(data.split(data.load_series(name))[0])                # number of train rows
    ev = data.load_events(name)                                     # labels and windows of this file
    times = g["time"].to_numpy()                                    # the timestamps as an array
    for det in DETECTORS:                                           # try each detector
        s = g[det].to_numpy()                                       # its scores as an array
        for method, values in GRID.items():                         # try each threshold method
            for v in values:                                        # and each candidate value
                d = D.detections(s, times, D.threshold(s, cut, method, v), 0, cut)   # detections in the TRAIN part only
                table, fp = D.score_events(d, ev, g["time"].iloc[0], g["time"].iloc[cut])   # score them against the train windows
                rows.append(dict(file=name, method=method, detector=det, value=v,   # remember the setting ...
                                 tp=int(table["detected"].sum()),                   # ... windows hit
                                 fn=int((~table["detected"]).sum()), fp=fp))        # ... windows missed and false alarms

res = pd.DataFrame(rows).groupby(["method", "detector", "value"])[["tp", "fn", "fp"]].sum().reset_index()   # add up over the 17 files
res[["precision", "recall", "f1"]] = [D.prf(a, b, c) for a, b, c in zip(res.tp, res.fp, res.fn)]   # precision, recall and F1 for each setting
res.to_csv(C.TABLES / "tuning_grid_train.csv", index=False)         # save the whole grid
pd.set_option("display.width", 160)                                 # print wide tables on one line
print(res[res.method == "per_file"].round({"precision": 2, "recall": 2, "f1": 2}).to_string(index=False))   # show the per-file grid

# one z-score detector (window 72 or 288 is a tuned setting) and one forecast-error detector, per method
res["family"] = res["detector"].map(lambda d: "error" if d == "err" else "zscore")   # group z72 and z288 as one family
best = res.sort_values(["f1", "value"], ascending=[False, False]).groupby(["method", "family"]).head(1)   # highest F1; ties -> larger value
best.to_csv(C.TABLES / "detector_params.csv", index=False)          # save the chosen settings
print("\nCHOSEN ON TRAIN (highest F1; if tied, the larger value = fewer alarms):")   # section title
print(best.round({"precision": 2, "recall": 2, "f1": 2}).to_string(index=False))    # show the chosen settings

fig, axes = plt.subplots(2, 3, figsize=(16, 7))                     # a 2x3 grid of plots
for r, method in enumerate(["global", "per_file"]):                 # one row of plots per method
    for ax, det in zip(axes[r], ["z288", "z72", "err"]):            # one plot per detector
        g = res[(res.method == method) & (res.detector == det)]     # the grid results for this method and detector
        for col in ["precision", "recall", "f1"]:                   # three curves
            ax.plot(g["value"], g[col], "o-", label=col)            # value on the x axis, metric on the y axis
        if method == "global":                                      # global thresholds span a huge range
            ax.set_xscale("log")                                    # so use a log axis
        ax.set_title(f"{det}, {method} threshold (train)")          # plot title
        ax.legend(fontsize=8)                                       # show the legend
plt.tight_layout()                                                  # avoid overlapping labels
fig.savefig(C.FIGURES / "tuning_train.png", dpi=90)                 # save the figure