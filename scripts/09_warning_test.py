"""score the early warnings ONCE on the test part, with the settings chosen on train."""
import numpy as np                                                  # NumPy: arrays and maths
import pandas as pd                                                 # pandas: tables
from sklearn.metrics import roc_auc_score                           # threshold-free quality measure
from kpi import config as C, data, warn as W                        # our config, loader and warning tools

scores = pd.read_csv(C.TABLES / "detector_scores.csv", parse_dates=["time"])   # the anomaly scores from Day 1
dparams = pd.read_csv(C.TABLES / "detector_params.csv")             # the detector thresholds chosen on train
pe = dparams[(dparams.method == "per_file") & (dparams.family == "error")].iloc[0]   # the forecast-error detector
chosen = pd.read_csv(C.TABLES / "warning_params.csv").set_index("method")["value"]   # the warning settings chosen on train
print(chosen.to_string())                                           # show which settings we use

tables = {n: W.file_table(n, W.merged_detection_times(n, pe, scores)) for n in data.list_series()}   # inputs and answers per file
files = list(tables)                                                # the 17 file names
VARIANTS = {"labels": lambda t: t["y_label"],                       # variant A: learn from the exact labels only
            "labels+detections": lambda t: t["y_label"] | t["y_det"]}   # variant B: also from detector-found anomalies

summary, test_scores = [], {}                                       # results table, and test scores for the AUC
days = sum((~tables[n]["is_train"]).sum() for n in files) / 288 / len(files)   # average test length in days per file
for vname, target in VARIANTS.items():                              # for each model variant
    X = np.vstack([tables[n]["X"][tables[n]["is_train"]] for n in files])      # inputs: ALL train origins of ALL files
    y = np.concatenate([target(tables[n])[tables[n]["is_train"]] for n in files])   # their answers
    model = W.make_model().fit(X, y)                                # teach the final model (train rows only)
    results, test_scores[vname] = [], []                            # per-file results and scores
    for n in files:                                                 # score every file's TEST part
        t = tables[n]                                               # this file's table
        te = ~t["is_train"]                                         # the test origins
        p = model.predict_proba(t["X"][te])[:, 1]                   # probability of "anomaly in the next 30 min"
        test_scores[vname].append(p)                                # keep it
        results.append(W.score_warnings(p > chosen[vname], t["times"][te], t["labels"]))   # score the warnings against the exact labels
    summary.append(dict(method=vname, setting=chosen[vname], **W.pool(results)))   # add up over the files
results, test_scores["rule"] = [], []                               # now the simple rule
for n in files:                                                     # loop over the files
    t = tables[n]                                                   # this file's table
    te = ~t["is_train"]                                             # the test origins
    v = t["value"][te]                                              # the current value at each test origin
    hi = np.quantile(t["train_values"], chosen["rule"])             # upper limit, from the TRAIN values
    lo = np.quantile(t["train_values"], 1 - chosen["rule"])         # lower limit, from the TRAIN values
    results.append(W.score_warnings((v > hi) | (v < lo), t["times"][te], t["labels"]))   # alarm outside the limits
    ref = np.sort(t["train_values"])                                # the train values, sorted
    pp = np.searchsorted(ref, v, side="right") / len(ref)           # percentile of the current value
    test_scores["rule"].append(np.maximum(pp, 1 - pp))              # the rule's continuous score (for the AUC)
summary.append(dict(method="rule", setting=chosen["rule"], **W.pool(results)))   # add up over the files

out = pd.DataFrame(summary)                                         # the summary table
yy = np.concatenate([tables[n]["y_label"][~tables[n]["is_train"]] for n in files])   # true answers at all test origins
out["false_per_file_day"] = (out["false"] / (days * len(files))).round(2)   # false warnings per file per day
out["AUC"] = [round(roc_auc_score(yy, np.concatenate(test_scores[m])), 3) for m in out["method"]]   # threshold-free quality
out.to_csv(C.TABLES / "warning_summary_test.csv", index=False)      # save it
pd.set_option("display.width", 220)                                 # print wide tables on one line
print("\nTEST SUMMARY (", int(yy.sum()), "positive test origins,", int(len(yy)), "origins in total )")   # section title
print(out.to_string(index=False))                                   # the main result