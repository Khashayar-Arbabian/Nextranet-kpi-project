"""Early warning: from the last 2 hours, will an exact anomaly time fall in the next 30 minutes?"""
import numpy as np                                                  # NumPy: arrays and maths
import pandas as pd                                                 # pandas: tables and time series
from sklearn.ensemble import HistGradientBoostingClassifier         # gradient boosting for yes/no questions

from . import config as C, data, detect as D                        # our config, loader and detection tools

AHEAD = np.timedelta64(30, "m")                                     # "the next 30 minutes"
STEPS_AHEAD = 6                                                     # 30 minutes = 6 steps of 5 minutes
FIRST = 24                                                          # first usable row: we need 2 hours (24 values) of history
LOOK = np.arange(1, 25)                                             # look back 1..24 steps = 2 hours


def raw_features(y, o):                                             # describe the last 2 hours before each origin
    """Numbers describing the last 2 hours before each origin `o` (nothing after the origin is used)."""
    last = y[o]                                                     # "now": the value at each origin
    lags = y[o[:, None] - LOOK] - last[:, None]                     # the last 24 values, each minus now -> 24 columns
    two_h = y[o[:, None] - np.arange(24)]                           # the last 24 values (2 hours), including now
    one_h = y[o[:, None] - np.arange(12)]                           # the last 12 values (1 hour), including now
    mean1 = one_h.mean(axis=1) - last                               # average of the last hour minus now
    std1 = one_h.std(axis=1)                                        # noise level of the last hour
    std2 = two_h.std(axis=1)                                        # noise level of the last 2 hours
    jump = np.abs(np.diff(two_h, axis=1)).max(axis=1)               # the biggest one-step change in the last 2 hours
    z = (last - two_h.mean(axis=1)) / (std2 + 1e-9)                 # how far now is from the 2-hour average, in standard deviations
    return np.column_stack([lags, mean1, std1, std2, jump, z])      # 24 + 5 = 29 columns


def to_percentiles(X, ref):                                         # make the numbers comparable between files
    """Replace every number by its percentile among the reference rows (this file's TRAIN rows)."""
    out = np.empty_like(X, dtype=float)                             # empty result with the same shape
    for j in range(X.shape[1]):                                     # one column at a time
        sorted_ref = np.sort(ref[:, j])                             # the reference values of this column, sorted
        out[:, j] = np.searchsorted(sorted_ref, X[:, j], side="right") / len(sorted_ref)   # share of reference values <= this value
    return out                                                      # all columns now run from 0 to 1, whatever the file's units


def zone(times, event_times):                                       # which origins have an event coming within 30 minutes?
    """True for origins that have an event in their next 30 minutes."""
    out = np.zeros(len(times), dtype=bool)                          # start with "no"
    for e in event_times:                                           # for each event time
        e = np.datetime64(e)                                        # as a NumPy time
        out |= (times < e) & (times >= e - AHEAD)                   # origins up to 30 minutes before the event
    return out                                                      # one True/False per origin


def merged_detection_times(name, params, scores):                   # anomalies found by our detector in the train part
    """Start times of the forecast-error detector's detections in the TRAIN part of a file."""
    g = scores[scores["file"] == name].reset_index(drop=True)       # the scores of this file
    cut = len(data.split(data.load_series(name))[0])                # number of train rows
    s = g[params["detector"]].to_numpy()                            # the chosen detector's scores
    thr = D.threshold(s, cut, params["method"], params["value"])    # its threshold (from train scores)
    return D.detections(s, g["time"].to_numpy(), thr, 0, cut)       # detections inside the train part only


def file_table(name, det_times):                                    # everything we need for one file
    """All the inputs and answers for one file: train origins and test origins."""
    df = data.load_series(name)                                     # the cleaned series
    cut = len(data.split(df)[0])                                    # number of train rows
    y = df["value"].to_numpy()                                      # the series as an array
    ev = data.load_events(name)                                     # exact labels of this file
    o = np.arange(FIRST, len(y) - STEPS_AHEAD)                      # all origins that still have 30 minutes after them
    times = df.index.to_numpy()[o]                                  # the timestamp of each origin
    is_train = o <= cut - 1 - STEPS_AHEAD                           # train origins: their next 30 minutes stay inside train
    X = raw_features(y, o)                                          # features from the last 2 hours
    Xp = to_percentiles(X, X[is_train])                             # make them comparable between files (percentiles of train)
    return dict(name=name, times=times, o=o, is_train=is_train, X=Xp,   # the origins and the inputs
                value=y[o],                                         # the current value at each origin (for the simple rule)
                train_values=y[:cut],                               # the train part of the series (for the simple rule)
                labels=ev["label_time"].to_numpy(),                 # exact anomaly times
                y_label=zone(times, ev["label_time"]),              # answer from exact labels only
                y_det=zone(times, det_times))                       # answer from detector-found anomalies


def score_warnings(alarm, times, labels):                           # score the warnings of ONE file
    """Event-level scoring of warnings for ONE file. `alarm` is True/False per origin.
    A warning is the START of an alarm episode (the first alarm after at least 1 hour without alarms)."""
    lab = [np.datetime64(L) for L in labels                         # keep only labels whose 30-minute zone has origins
           if times[0] < np.datetime64(L) <= times[-1] + AHEAD]     # (the label must lie inside the range we can warn about)
    idx = np.flatnonzero(alarm)                                     # origins that raised an alarm
    if len(idx):                                                    # if there were any alarms
        onset = idx[np.r_[True, np.diff(idx) > D.GAP]]              # the first alarm, and every alarm more than 1 hour after the previous
    else:                                                           # otherwise
        onset = np.array([], dtype=int)                             # no alarms, no warnings
    in_zone = zone(times, lab)                                      # origins that have a labeled anomaly within 30 minutes
    correct = int(in_zone[onset].sum())                             # warnings that start inside a "30 minutes before" zone
    warned, minutes = 0, []                                         # labels warned in time, and the warning times
    for L in lab:                                                   # for each labeled anomaly
        hit = onset[(times[onset] < L) & (times[onset] >= L - AHEAD)]   # warnings that started in the 30 minutes before it
        if len(hit):                                                # at least one: it was warned in time
            warned += 1                                             # count it
            minutes.append((L - times[hit[0]]) / np.timedelta64(1, "m"))   # warning time = label minus the warning start
    return dict(labels=len(lab), warned=warned,                     # collect the counts: labels and labels warned
                warnings=len(onset), correct=correct,               # all warnings and correct warnings
                false=len(onset) - correct, minutes=minutes)        # false warnings and the warning times


def pool(results):                                                  # add up the results of all files
    """Add up per-file results and compute precision, recall, F1 and average warning time."""
    labels = sum(r["labels"] for r in results)                      # total labeled anomalies
    warned = sum(r["warned"] for r in results)                      # anomalies warned in time
    correct = sum(r["correct"] for r in results)                    # correct warnings
    false = sum(r["false"] for r in results)                        # false warnings
    minutes = [m for r in results for m in r["minutes"]]            # all warning times
    p = correct / (correct + false) if correct + false else 0.0     # precision: correct warnings / all warnings
    rc = warned / labels if labels else 0.0                         # recall: anomalies warned / all anomalies
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0                   # F1: balance of the two
    return dict(labels=labels, warned=warned,                       # collect the numbers: labels and labels warned
                warnings=correct + false, correct=correct, false=false,   # warnings, correct, false
                precision=round(p, 3), recall=round(rc, 2), f1=round(f1, 3),   # the three quality numbers
                avg_warning_min=round(float(np.mean(minutes)), 1) if minutes else np.nan)   # average warning time in minutes


def make_model():                                                   # the warning model
    return HistGradientBoostingClassifier(                          # gradient boosting classifier
        max_iter=100, learning_rate=0.05, max_depth=3, min_samples_leaf=40,   # small, careful settings
        class_weight="balanced",                                    # positives are very rare: count them more
        early_stopping=False, random_state=0)                       # same result on every run