"""group the detected anomalies into types by shape (Task 2.2)."""
import matplotlib                                                   # plotting library
matplotlib.use("Agg")                                               # draw into image files (no window needed)
import matplotlib.pyplot as plt                                     # the plotting functions
import pandas as pd                                                 # pandas: tables
from kpi import config as C, data, shape as S                       # our config, loader and shape tools

scores = pd.read_csv(C.TABLES / "detector_scores.csv", parse_dates=["time"])   # the anomaly scores from Day 1
params = pd.read_csv(C.TABLES / "detector_params.csv")              # the thresholds chosen on train
params = params[params["method"] == "per_file"]                     # use the per-file detectors (z-score and error)

rows = []                                                           # one row per detected event
for name in data.list_series():                                     # loop over the 17 files
    df = data.load_series(name)                                     # the cleaned series
    y = df["value"].to_numpy()                                      # the values as an array
    cut = len(data.split(df)[0])                                    # number of train rows
    R = S.normal_range(y[:cut])                                     # the file's normal range (from train only)
    windows = data.load_events(name).drop_duplicates("window_id")   # the labeled windows of this file
    for t in S.detection_events(name, scores, params):              # every detected event of this file
        i = df.index.get_loc(t)                                     # the row number of its start
        f = S.shape_features(y, i, R)                               # its two shape numbers
        if f is None:                                               # too close to the start or the end of the file
            continue                                                # skip it
        inside = any((t >= w.window_start) and (t <= w.window_end) for _, w in windows.iterrows())   # inside a labeled window?
        rows.append(dict(file=name, time=t, part="test" if i >= cut else "train",   # which file, when, train or test
                         peak=f["peak"], shift=f["shift"],          # the two shape numbers
                         type=S.categorize(f["peak"], f["shift"]), in_window=inside))   # the type and the window check

events = pd.DataFrame(rows)                                         # all events as a table
events.to_csv(C.TABLES / "anomaly_events.csv", index=False)         # save it
SEVERITY = {"level shift": "High", "spike": "Medium",              # suggested severity for each type
            "moderate excursion": "Low", "minor deviation": "Info (log only)"}   # (the other two types)

summary = events.groupby("type").agg(count=("type", "size"),        # how many events of each type
                                     in_labeled_window=("in_window", "mean"),   # share of them inside a labeled window
                                     files=("file", "nunique")).round(2)         # in how many files they occur
summary["suggested_severity"] = summary.index.map(SEVERITY)         # add the severity column
order = ["level shift", "spike", "moderate excursion", "minor deviation"]   # the order we want to show
summary = summary.loc[order]                                        # sort the table
summary.to_csv(C.TABLES / "anomaly_types.csv")                      # save it
pd.set_option("display.width", 200)                                 # print wide tables on one line
print(len(events), "events in total")                               # total number of events
print(summary.to_string())                                          # the type table
print("level shifts up / down:", int(((events.type == "level shift") & (events["shift"] > 0)).sum()),   # level shifts upwards
      "/", int(((events.type == "level shift") & (events["shift"] < 0)).sum()))   # direction of the level shifts
print("spikes up / down:", int(((events.type == "spike") & (events.peak > 0)).sum()),   # spikes upwards
      "/", int(((events.type == "spike") & (events.peak < 0)).sum()))             # direction of the spikes
print("\nspikes per file:\n", events[events.type == "spike"].groupby("file").size().to_string())   # which files have the spikes

# sanity check: the same rules applied directly to the 31 labeled anomalies (not to detections)
lab = []                                                            # one row per labeled anomaly
for name in data.list_series():                                     # loop over the files
    df = data.load_series(name)                                     # the cleaned series
    y = df["value"].to_numpy()                                      # the values
    cut = len(data.split(df)[0])                                    # number of train rows
    R = S.normal_range(y[:cut])                                     # the normal range from train
    for L in data.load_events(name)["label_time"]:                  # each exact anomaly time
        i = int(df.index.get_indexer([L], method="nearest")[0])     # the nearest row of the series
        f = S.shape_features(y, i, R)                               # its shape numbers
        if f is not None:                                           # skip labels too close to the edges
            lab.append(dict(file=name[:28], label=L, part="test" if i >= cut else "train",   # which anomaly, train or test
                            peak=round(f["peak"], 2), shift=round(f["shift"], 2),   # its shape numbers
                            type=S.categorize(f["peak"], f["shift"])))   # the type of this known anomaly
lab = pd.DataFrame(lab)                                             # as a table
print("\nTYPES OF THE LABELED ANOMALIES (", len(lab), "of 31 with enough data around them )")   # section title
print(lab["type"].value_counts().to_string())                       # how many labeled anomalies of each type
print(lab[lab.part == "test"].sort_values("label").to_string(index=False))   # the labeled anomalies of the test part

# figure: the most pronounced example of each type
fig, axes = plt.subplots(2, 2, figsize=(14, 8))                     # a 2x2 grid of plots
for ax, tname in zip(axes.ravel(), order):                          # one plot per type
    g = events[events.type == tname]                                # the events of this type
    key = g["shift"].abs() if tname == "level shift" else g["peak"].abs()   # level shifts: biggest shift; others: biggest peak
    e = g.loc[key.idxmax()]                                         # the most pronounced example
    df = data.load_series(e.file)                                   # its series
    seg = df.loc[e.time - pd.Timedelta("12h"): e.time + pd.Timedelta("12h"), "value"]   # 12 hours before and after
    ax.plot(seg.index, seg.values, lw=0.8, color="#1f4e79")         # the series
    for _, w in data.load_events(e.file).drop_duplicates("window_id").iterrows():   # the labeled windows
        ax.axvspan(w.window_start, w.window_end, color="#e8a33d", alpha=0.25)        # orange bands
    ax.axvline(e.time, color="tab:purple", ls="--", lw=1.2)         # the detection time
    ax.set_xlim(seg.index[0], seg.index[-1])                        # show only the 24 hours around the detection
    ax.set_title(f"{tname}: {e.file[:30]}\n{e.time}  (peak {e.peak:.1f}, shift {e['shift']:.1f})", fontsize=9)   # title
    ax.tick_params(labelsize=8)                                     # small tick labels
plt.tight_layout()                                                  # avoid overlapping labels
fig.savefig(C.FIGURES / "anomaly_types.png", dpi=90)                # save the figure