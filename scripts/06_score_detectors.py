"""score the detectors ONCE on the test part, with the thresholds chosen on train."""
import matplotlib                                                   # plotting library
matplotlib.use("Agg")                                               # draw into image files (no window)
import matplotlib.pyplot as plt                                     # the plotting functions
import pandas as pd                                                 # pandas: tables
from kpi import config as C, data, detect as D                      # our config, loader and detectors

scores = pd.read_csv(C.TABLES / "detector_scores.csv", parse_dates=["time"])   # the scores from 04_detector_scores.py
params = pd.read_csv(C.TABLES / "detector_params.csv")              # the settings chosen on train by 05_tune_detectors.py
print(params[["method", "family", "detector", "value"]].to_string(index=False))   # show which settings we are using

events, per_file, found = [], [], {}                                # collected results
for name in data.list_series():                                     # loop over the 17 files
    g = scores[scores["file"] == name].reset_index(drop=True)       # the scores of this file
    cut = len(data.split(data.load_series(name))[0])                # number of train rows = first test row
    ev = data.load_events(name)                                     # labels and windows of this file
    times = g["time"].to_numpy()                                    # the timestamps as an array
    test_days = (len(g) - cut) / 288                                # length of the test part in days
    for _, p in params.iterrows():                                  # one pass per chosen setting (4 of them)
        s = g[p["detector"]].to_numpy()                             # the scores of the chosen detector
        d = D.detections(s, times, D.threshold(s, cut, p["method"], p["value"]), cut, len(g))   # detections in the TEST part only
        found[(name, p["method"], p["family"])] = d                 # remember them for the figure
        table, fp = D.score_events(d, ev, g["time"].iloc[cut], g["time"].iloc[-1] + pd.Timedelta("1min"))   # score vs the test windows
        table.insert(0, "file", name)                               # add the file name as a column
        table.insert(1, "method", p["method"])                      # add the threshold method
        table.insert(2, "family", p["family"])                      # add the detector family
        events.append(table)                                        # keep the per-window results
        per_file.append(dict(file=name, method=p["method"], family=p["family"], windows=len(table),   # keep the per-file counts
                             hit=int(table["detected"].sum()), false_alarms=fp, test_days=test_days))

events, per_file = pd.concat(events), pd.DataFrame(per_file)        # stack the lists into tables
events.to_csv(C.TABLES / "detection_events_test.csv", index=False)  # save the per-window results
per_file.to_csv(C.TABLES / "detection_by_file_test.csv", index=False)   # save the per-file counts

summary = []                                                        # one summary row per method and detector
for (method, family), g in per_file.groupby(["method", "family"]):  # loop over the four combinations
    tp, fp = g["hit"].sum(), g["false_alarms"].sum()                # total windows hit and total false alarms
    fn = g["windows"].sum() - tp                                    # windows missed
    pr, rc, f1 = D.prf(tp, fp, fn)                                  # precision, recall, F1
    delays = events[(events.method == method) & (events.family == family) & events.detected]["delay_min"]   # delays of the hit windows
    summary.append(dict(method=method, detector=family, windows=int(g["windows"].sum()), detected=int(tp),   # collect the numbers
                        false_alarms=int(fp), precision=round(pr, 2), recall=round(rc, 2), f1=round(f1, 2),
                        median_delay_min=delays.median(),                                   # typical delay in minutes
                        false_alarms_per_file_day=round(fp / g["test_days"].sum(), 2)))     # false alarms per file per day
summary = pd.DataFrame(summary)                                     # turn the list into a table
summary.to_csv(C.TABLES / "detection_summary_test.csv", index=False)   # save it
pd.set_option("display.width", 220)                                 # print wide tables on one line
print("\nTEST SUMMARY")                                             # section title
print(summary.to_string(index=False))                               # print the summary table

print("\nPER EVENT (test), per_file method; delay in minutes from the exact label (negative = before it)")   # section title
e = events[events.method == "per_file"]                             # only the per-file method
cols = ["file", "label", "detected", "delay_min"]                   # the columns we want to show
a = e[e.family == "error"][cols].rename(columns={"detected": "error_hit", "delay_min": "error_delay"})       # error detector results
b = e[e.family == "zscore"][cols].rename(columns={"detected": "zscore_hit", "delay_min": "zscore_delay"})    # z-score detector results
out = a.merge(b, on=["file", "label"]).sort_values("label")         # put both detectors side by side, sorted by time
out["file"] = out["file"].str.slice(0, 30)                          # shorten long file names
print(out.to_string(index=False))                                   # print the per-event table
print("\nFALSE ALARMS PER FILE (test), per_file method")            # section title
print(per_file[per_file.method == "per_file"].pivot(index="file", columns="family", values="false_alarms").to_string())   # false alarms per file

# ---- figure: the 9 test events with the first alarms of the per_file detectors
lab = events.drop_duplicates(["file", "label"])[["file", "label", "window_start", "window_end"]].sort_values("label")   # the 9 test events
fig, axes = plt.subplots(3, 3, figsize=(18, 11))                    # a 3x3 grid of plots
axes = axes.ravel()                                                 # flatten the grid
for ax, (_, e) in zip(axes, lab.iterrows()):                        # one plot per event
    df = data.load_series(e.file)                                   # the cleaned series
    seg = df.loc[e.label - pd.Timedelta("18h"): e.label + pd.Timedelta("18h"), "value"]   # 18 hours before and after the label
    ax.plot(seg.index, seg.values, lw=0.7, color="#1f4e79")         # the series
    ax.axvspan(e.window_start, e.window_end, color="#e8a33d", alpha=0.25)   # the labeled window (orange)
    ax.axvline(e.label, color="red", lw=1.2)                        # the exact anomaly time (red)
    for family, colr in [("zscore", "tab:green"), ("error", "tab:purple")]:   # one colour per detector
        for t in found[(e.file, "per_file", family)]:               # every detection of this detector in this file
            if seg.index[0] <= t <= seg.index[-1]:                  # only those inside the plotted range
                ax.axvline(t, color=colr, ls="--", lw=1, label=family)   # dashed line
    h, l = ax.get_legend_handles_labels()                           # collect the legend entries
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=8)   # legend without duplicates
    ax.set_title(f"{e.file[:32]}\n{e.label}", fontsize=9)           # title with file and label time
plt.tight_layout()                                                  # avoid overlapping labels
fig.savefig(C.FIGURES / "detections_test_events.png", dpi=90)       # save the figure