"""where and why do the forecasts fail? Which metric types are predictable?"""
import matplotlib                                                   # plotting library
matplotlib.use("Agg")                                               # draw into image files (no window needed)
import matplotlib.pyplot as plt                                     # the plotting functions
import numpy as np                                                  # NumPy: arrays and maths
import pandas as pd                                                 # pandas: tables
from kpi import config as C, data                                   # our paths and loader

fc = pd.read_csv(C.TABLES / "test_forecasts.csv", parse_dates=["origin_time"])   # all test forecasts saved by 02_model.py
fc = fc[~fc["imputed"]].copy()                                      # drop targets that were interpolated on Day 1
fc["err"] = (fc["actual"] - fc["gboost"]).abs()                     # absolute error of the model at every forecast point


def kind(name):                                                     # label a file with its metric type, using its name
    for prefix, label in [("ec2_cpu", "EC2 CPU"), ("ec2_disk", "disk write"), ("ec2_network", "network in"),
                          ("iio", "network in"), ("elb", "request count"), ("rds", "RDS CPU"),
                          ("grok", "auto-scaling")]:                # known name prefixes and their labels
        if name.startswith(prefix):                                 # does this file name start with the prefix?
            return label                                            # yes: return the label


# ---- 1. which metric types are predictable? (model error / baseline error; below 1 = model better)
bf = pd.read_csv(C.TABLES / "models_by_file.csv", index_col=0)      # per-file results from the model run
bf["type"] = [kind(n) for n in bf.index]                            # metric type of each file
bf["vs_last"] = bf["MAE_gboost"] / bf["MAE_last_value"]             # model error divided by last-value error
bf["vs_mean_1h"] = bf["MAE_gboost"] / bf["MAE_mean_1h"]             # model error divided by 1-hour-mean error
by_type = bf.groupby("type")[["vs_last", "vs_mean_1h"]].mean().round(2)   # average the two ratios per metric type
by_type.insert(0, "files", bf.groupby("type").size())               # add a column with the number of files per type
print("\n1) BY METRIC TYPE (ratio of gboost MAE to the baseline MAE; lower is better)")   # section title
print(by_type.sort_values("vs_last").to_string())                   # print the table, best type first
by_type.to_csv(C.TABLES / "error_by_type.csv")                      # save it

# ---- 2. does the model's advantage change with the horizon? (median over the 17 files)
bh = pd.read_csv(C.TABLES / "models_by_horizon.csv")                # metrics per file, model and horizon
p = bh.pivot_table(index=["file", "minutes"], columns="model", values="MAE").reset_index()   # one column per model
p["ratio"] = p["gboost"] / p["last_value"]                          # model error divided by last-value error
by_h = p.groupby("minutes")["ratio"].median().round(3)              # median ratio over the files, for each horizon
print("\n2) MEDIAN gboost/last_value MAE RATIO BY HORIZON")          # section title
print(by_h.to_string())                                             # print the 12 numbers

# ---- 3. where do the errors concentrate?
rows = []                                                           # one result row per file
for name, g in fc.groupby("file"):                                  # loop over the files
    per_origin = g.groupby("origin_time")["err"].mean().sort_index()   # one error number per forecast (average of its 12 points)
    top = per_origin.sort_values(ascending=False)                   # forecasts sorted from worst to best
    k = max(1, int(round(0.05 * len(top))))                         # how many forecasts are the worst 5%
    share = top.iloc[:k].sum() / top.sum()                          # share of all error caused by the worst 5%
    ev = data.load_events(name).drop_duplicates("window_id")        # the labeled windows of this file
    inside = np.zeros(len(per_origin), dtype=bool)                  # True when a forecast was made inside a window
    for _, w in ev.iterrows():                                      # loop over the windows
        inside |= ((per_origin.index >= w.window_start) & (per_origin.index <= w.window_end))   # mark forecasts inside this window
    ratio = per_origin[inside].mean() / per_origin[~inside].mean() if inside.any() and (~inside).any() else np.nan   # error inside / outside windows
    rows.append(dict(file=name, top5pct_share=round(share, 2), in_window_origins=int(inside.sum()),   # collect the results
                     err_in_vs_out_window=round(ratio, 1) if ratio == ratio else np.nan,
                     worst_origin=top.index[0]))                    # time of the single worst forecast
where = pd.DataFrame(rows).set_index("file")                        # turn the list into a table
where.to_csv(C.TABLES / "error_concentration.csv")                  # save it
print("\n3) WHERE ERRORS CONCENTRATE (top5pct_share = share of total error from the worst 5% of forecasts)")   # section title
print(where.to_string())                                            # print the table

# ---- 4. figure: residuals over time at 60 minutes ahead
fig, axes = plt.subplots(6, 3, figsize=(20, 22))                    # a 6x3 grid of small plots
axes = axes.ravel()                                                 # flatten the grid into one list
for ax, (name, g) in zip(axes, fc[fc["minutes"] == 60].groupby("file")):   # one plot per file, using the +60 min forecasts
    ev = data.load_events(name)                                     # labels and windows of this file
    ax.plot(g["origin_time"], g["actual"] - g["gboost"], lw=0.8, color="#1f4e79")   # residual = real minus forecast
    ax.axhline(0, color="k", lw=0.6)                                # a line at zero error
    for _, w in ev.drop_duplicates("window_id").iterrows():         # shade every labeled window
        ax.axvspan(w.window_start, w.window_end, color="#e8a33d", alpha=0.25)   # orange band
    for t in ev["label_time"]:                                      # mark every exact anomaly time
        ax.axvline(t, color="red", lw=1)                            # red line
    ax.set_xlim(g["origin_time"].min(), g["origin_time"].max())     # show only the test period
    ax.set_title(name, fontsize=11)                                 # file name as the title
    ax.tick_params(labelsize=8)                                     # small tick labels
for ax in axes[17:]:                                                # the grid has 18 slots but only 17 files
    ax.axis("off")                                                  # hide the unused slot
fig.suptitle("Residuals (actual - forecast) at 60 min ahead, test period. Orange = windows, red = anomalies", fontsize=15)   # overall title
plt.tight_layout()                                                  # avoid overlapping labels
fig.savefig(C.FIGURES / "residuals_over_time.png", dpi=90)          # save the figure

# ---- 5. figure: the worst forecast of the 4 files with the most concentrated error
worst4 = where.sort_values("top5pct_share", ascending=False).head(4)   # the 4 files where a few forecasts cause most error
fig, axes = plt.subplots(2, 2, figsize=(14, 8))                     # a 2x2 grid of plots
axes = axes.ravel()                                                 # flatten the grid
for ax, (name, r) in zip(axes, worst4.iterrows()):                  # one plot per file
    df = data.load_series(name)                                     # the cleaned series
    t0 = r["worst_origin"]                                          # time of the worst forecast
    seg = df.loc[t0 - pd.Timedelta("6h"): t0 + pd.Timedelta("2h"), "value"]   # the series from 6 h before to 2 h after it
    g = fc[(fc.file == name) & (fc.origin_time == t0)].sort_values("minutes")   # the 12 forecasts made at that moment
    ax.plot(seg.index, seg.values, color="#1f4e79", lw=1, label="actual")       # the real series
    when = t0 + pd.to_timedelta(g["minutes"], unit="m")             # the 12 target times
    ax.plot(when, g["gboost"], "o-", color="tab:red", ms=3, label="gboost forecast")   # the model forecast
    ax.plot(when, g["last_value"], "--", color="grey", label="last value")     # the last-value forecast
    ax.axvline(t0, color="k", lw=0.8)                               # mark the moment the forecast was made
    ax.set_title(f"{name}\nforecast made at {t0}", fontsize=10)     # title with file and time
    ax.legend(fontsize=8)                                           # show the legend
plt.tight_layout()                                                  # avoid overlapping labels
fig.savefig(C.FIGURES / "worst_forecasts.png", dpi=90)              # save the figure
print("\nsaved figures and tables")                                  # confirmation message