"""Shape features and simple rules to group detected anomalies into types."""
import numpy as np                                                  # NumPy: arrays and maths
import pandas as pd                                                 # pandas: tables and time series

from . import data, detect as D                                     # our loader and detection tools

SHIFT_MIN = 0.5       # a level change of at least half the file's normal range = level shift
SPIKE_MIN = 3.0       # a peak of at least 3 normal ranges = spike
EXCURSION_MIN = 1.0   # a peak of at least 1 normal range = moderate excursion


def normal_range(y_train):                                          # the yardstick that makes files comparable
    """The file's normal range: the 99th percentile minus the 1st percentile of its TRAIN values."""
    return max(np.percentile(y_train, 99) - np.percentile(y_train, 1), 1e-9)   # never zero, so we can divide by it


def shape_features(y, i, R):                                        # describe the shape around row i
    """Two numbers that describe the shape around row i, in units of the normal range R."""
    pre = y[max(0, i - 36): max(0, i - 2)]                          # about 3 hours before the detection (ends 10 minutes before)
    post = y[i + 3: i + 39]                                         # about 3 hours after the detection (starts 15 minutes after)
    zone = y[max(0, i - 2): i + 4]                                  # the 30 minutes around the detection
    if len(pre) < 6 or len(post) < 6:                               # too close to the start or the end of the file
        return None                                                 # then we cannot describe the shape
    base = np.median(pre)                                           # the usual level just before
    j = np.argmax(np.abs(zone - base))                              # the point that is furthest from the usual level
    return dict(peak=(zone[j] - base) / R,                          # peak: how far that point is from the usual level (with sign)
                shift=(np.median(post) - base) / R)                 # shift: how much the level changed afterwards (with sign)


def categorize(peak, shift):                                        # the rules that name the type
    """Turn the two shape numbers into a type name."""
    if abs(shift) >= SHIFT_MIN:                                     # the level stayed changed afterwards
        return "level shift"                                        # type 1
    if abs(peak) >= SPIKE_MIN:                                      # big and short-lived
        return "spike"                                              # type 2
    if abs(peak) >= EXCURSION_MIN:                                  # medium and short-lived
        return "moderate excursion"                                 # type 3
    return "minor deviation"                                        # type 4: small and short-lived


def detection_events(name, scores, params):                         # list the detected events of one file
    """Detections of both per-file detectors over the whole file; detections less than 1 hour apart are one event."""
    g = scores[scores["file"] == name].reset_index(drop=True)       # the scores of this file
    cut = len(data.split(data.load_series(name))[0])                # number of train rows (needed for the per-file threshold)
    found = []                                                      # all detection times of both detectors
    for _, p in params.iterrows():                                  # the two chosen per-file settings (z-score and error)
        s = g[p["detector"]].to_numpy()                             # this detector's scores
        thr = D.threshold(s, cut, p["method"], p["value"])          # its threshold, from the train scores
        found += [pd.Timestamp(t) for t in D.detections(s, g["time"].to_numpy(), thr, 0, len(g))]   # detections over the whole file
    found.sort()                                                    # earliest first
    events = []                                                     # the merged events
    for t in found:                                                 # go through the detections in time order
        if events and t - events[-1] < pd.Timedelta("1h"):          # less than 1 hour after the previous event?
            continue                                                # then it is the same event: skip it
        events.append(t)                                            # otherwise it starts a new event
    return events                                                   # the start times of the events