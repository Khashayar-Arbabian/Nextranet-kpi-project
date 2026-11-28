"""tune the early-warning model and the simple rule using ONLY the train part."""
import matplotlib                                                   # plotting library
matplotlib.use("Agg")                                               # draw into image files (no window needed)
import matplotlib.pyplot as plt                                     # the plotting functions
import numpy as np                                                  # NumPy: arrays and maths
import pandas as pd                                                 # pandas: tables
from sklearn.metrics import roc_auc_score, average_precision_score  # threshold-free quality measures
from kpi import config as C, data, warn as W                        # our config, loader and warning tools

scores = pd.read_csv(C.TABLES / "detector_scores.csv", parse_dates=["time"])   # the anomaly scores from Day 1
params = pd.read_csv(C.TABLES / "detector_params.csv")              # the thresholds chosen on train
pe = params[(params.method == "per_file") & (params.family == "error")].iloc[0]   # the forecast-error detector (per-file threshold)

tables = {n: W.file_table(n, W.merged_detection_times(n, pe, scores)) for n in data.list_series()}   # inputs and answers per file
files = list(tables)                                                # the 17 file names
fold = {n: i % 3 for i, n in enumerate(files)}                      # split the FILES into 3 groups (for cross-fitting)
VARIANTS = {"labels": lambda t: t["y_label"],                       # variant A: learn from the exact labels only
            "labels+detections": lambda t: t["y_label"] | t["y_det"]}   # variant B: also from detector-found anomalies
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]  # candidate alarm thresholds for the model's probability
RULE_P = [0.9, 0.95, 0.99, 0.995, 0.999]                            # candidate percentiles for the simple rule

oof = {}                                                            # out-of-fold model scores for each variant
for vname, target in VARIANTS.items():                              # for each variant
    oof[vname] = {}                                                 # one dictionary of scores per file
    for f in range(3):                                              # each group of files takes a turn as "held out"
        learn = [n for n in files if fold[n] != f]                  # the files the model learns from
        held = [n for n in files if fold[n] == f]                   # the files it is checked on
        X = np.vstack([tables[n]["X"][tables[n]["is_train"]] for n in learn])   # inputs: train origins of the learning files
        y = np.concatenate([target(tables[n])[tables[n]["is_train"]] for n in learn])   # answers for those origins
        model = W.make_model().fit(X, y)                            # teach the model
        for n in held:                                              # score the held-out files (never seen by this model)
            oof[vname][n] = model.predict_proba(tables[n]["X"][tables[n]["is_train"]])[:, 1]   # probability of "anomaly in the next 30 min"

rows = []                                                           # one result row per method and setting
for vname in VARIANTS:                                              # the two model variants
    for thr in THRESHOLDS:                                          # each candidate threshold
        results = [W.score_warnings(oof[vname][n] > thr, tables[n]["times"][tables[n]["is_train"]], tables[n]["labels"])   # alarm if probability > threshold
                   for n in files]                                  # score the warnings of every file against the exact labels
        rows.append(dict(method=vname, value=thr, **W.pool(results)))   # add up over the files
for P in RULE_P:                                                    # the simple rule: alarm when the value is extreme for this file
    results = []                                                    # results per file
    for n in files:                                                 # loop over the files
        t = tables[n]                                               # this file's table
        v = t["value"][t["is_train"]]                               # the current value at each train origin
        hi = np.quantile(t["train_values"], P)                      # the upper limit: P-th percentile of the train values
        lo = np.quantile(t["train_values"], 1 - P)                  # the lower limit: (1-P)-th percentile of the train values
        results.append(W.score_warnings((v > hi) | (v < lo), t["times"][t["is_train"]], t["labels"]))   # alarm outside the limits
    rows.append(dict(method="rule", value=P, **W.pool(results)))    # add up over the files

grid = pd.DataFrame(rows)                                           # the whole tuning grid
grid.to_csv(C.TABLES / "warning_tuning_train.csv", index=False)     # save it
pd.set_option("display.width", 200)                                 # print wide tables on one line
print(grid.to_string(index=False))                                  # show the grid
best = grid.sort_values(["f1", "value"], ascending=[False, False]).groupby("method").head(1)   # best F1 per method; ties -> larger value
best.to_csv(C.TABLES / "warning_params.csv", index=False)           # save the chosen settings
print("\nCHOSEN ON TRAIN (highest F1; if tied, the larger value = fewer alarms):")   # section title
print(best.to_string(index=False))                                  # show the chosen settings

# threshold-free check: is there ANY signal in the last 2 hours?
yy = np.concatenate([tables[n]["y_label"][tables[n]["is_train"]] for n in files])   # the true answers at all train origins
print("\nTHRESHOLD-FREE CHECK on train (cross-fitted); base rate of positives: %.4f" % yy.mean())   # how rare positives are
for vname in VARIANTS:                                              # the two variants
    s = np.concatenate([oof[vname][n] for n in files])              # their out-of-fold scores
    print(f"{vname:26s} AUC {roc_auc_score(yy, s):.3f}   average precision {average_precision_score(yy, s):.4f}")   # AUC 0.5 = coin flip
sc = []                                                             # the simple rule's continuous score
for n in files:                                                     # loop over the files
    t = tables[n]                                                   # this file's table
    ref = np.sort(t["train_values"])                                # the train values, sorted
    p = np.searchsorted(ref, t["value"][t["is_train"]], side="right") / len(ref)   # percentile of the current value
    sc.append(np.maximum(p, 1 - p))                                 # 0.5 = typical, 1 = most extreme (high or low)
sc = np.concatenate(sc)                                             # one array over all files
print(f"{'simple rule (extremeness)':26s} AUC {roc_auc_score(yy, sc):.3f}   average precision {average_precision_score(yy, sc):.4f}")   # same measures

# figure: the average model score in the 2 hours before each labeled train anomaly
curve = {k: [] for k in range(-120, 31, 5)}                         # minutes relative to the label -> scores
for n in files:                                                     # loop over the files
    t = tables[n]                                                   # this file's table
    tt = t["times"][t["is_train"]]                                  # origin times in the train part
    for L in t["labels"]:                                           # each exact anomaly time
        L = np.datetime64(L)                                        # as a NumPy time
        if not (tt[0] < L <= tt[-1]):                               # skip labels outside the train origins
            continue                                                # (skip it)
        for k in curve:                                             # each minute offset
            j = np.searchsorted(tt, L + np.timedelta64(k, "m"))     # the origin closest to that moment
            if 0 <= j < len(tt) and abs((tt[j] - (L + np.timedelta64(k, "m"))) / np.timedelta64(1, "m")) < 3:   # it must be near enough
                curve[k].append(oof["labels"][n][j])                # remember the model's score there
xs = list(curve)                                                    # the minute offsets
ys = [np.mean(curve[k]) for k in xs]                                # the average score at each offset
overall = np.concatenate([oof["labels"][n] for n in files]).mean()  # the average score over all origins
fig, ax = plt.subplots(figsize=(8, 4))                              # one plot
ax.plot(xs, ys, "o-", label="average score before/after a labeled anomaly")   # the curve
ax.axhline(overall, color="grey", ls="--", label="average score overall")    # the overall average
ax.axvline(0, color="red", lw=1)                                    # the anomaly time
ax.set_xlabel("minutes relative to the labeled anomaly"); ax.set_ylabel("model score")   # axis labels
ax.set_title("Is there a precursor? (train, cross-fitted)"); ax.legend(fontsize=8)       # title and legend
plt.tight_layout()                                                  # avoid overlapping labels
fig.savefig(C.FIGURES / "warning_signal_train.png", dpi=90)         # save the figure