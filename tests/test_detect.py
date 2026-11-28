import numpy as np                                                  # NumPy: arrays and maths
import pandas as pd                                                 # pandas: tables and dates
from kpi import detect as D                                         # the module we are testing


def test_nearby_alarms_become_one_detection():
    score = np.zeros(100)                                           # 100 points, no alarms yet
    score[[10, 11, 12, 40]] = 5                                     # alarms at rows 10, 11, 12 and 40
    times = pd.date_range("2014-01-01", periods=100, freq="5min").to_numpy()   # a timestamp for each row
    found = D.detections(score, times, 1, 0, 100)                   # threshold 1, look at rows 0..99
    assert found == [times[10], times[40]]                          # rows 10-12 merge into one detection; row 40 is another


def test_event_scoring_hit_delay_and_false_alarm():
    ev = pd.DataFrame(dict(label_time=[pd.Timestamp("2014-01-02 12:00")], window_id=[0],   # one anomaly at 12:00 ...
                           window_start=[pd.Timestamp("2014-01-02 06:00")],                # ... in a window from 06:00 ...
                           window_end=[pd.Timestamp("2014-01-02 18:00")]))                 # ... to 18:00
    det = [pd.Timestamp("2014-01-02 12:10"), pd.Timestamp("2014-01-05")]   # one detection in the window, one days later
    table, fp = D.score_events(det, ev, pd.Timestamp("2014-01-01"), pd.Timestamp("2014-02-01"))   # score January
    assert table["detected"].tolist() == [True]                     # the window was hit
    assert table["delay_min"].iloc[0] == 10                         # 10 minutes after the label
    assert fp == 1                                                  # the detection on Jan 5 is a false alarm


def test_detection_inside_an_unscored_window_is_ignored():
    ev = pd.DataFrame(dict(label_time=[pd.Timestamp("2014-01-02 12:00")], window_id=[0],
                           window_start=[pd.Timestamp("2014-01-02 06:00")],
                           window_end=[pd.Timestamp("2014-01-02 18:00")]))             # the same window as above
    table, fp = D.score_events([pd.Timestamp("2014-01-02 12:10")], ev,
                               pd.Timestamp("2014-01-03"), pd.Timestamp("2014-02-01"))   # but we only score from Jan 3
    assert len(table) == 0 and fp == 0                              # the window is not scored, and the detection is NOT a false alarm


def test_zscore_does_not_use_the_future():
    y = np.random.default_rng(0).normal(size=1000)                  # a fake series of 1000 random values
    y2 = y.copy()                                                   # a copy of it
    y2[500:] += 1000                                                # change everything from row 500 on
    assert np.allclose(D.zscore(y, 72)[:500], D.zscore(y2, 72)[:500])   # scores before row 500 must not change


def test_test_errors_do_not_depend_on_later_data():
    y = np.random.default_rng(1).normal(size=1000).cumsum()         # a fake random-walk series
    y2 = y.copy()                                                   # a copy of it
    y2[706:] += 100                                                 # change the data after row 705
    r1, r2 = D.one_step_errors(y, 700), D.one_step_errors(y2, 700)  # forecast errors for both versions (train part = 700 rows)
    assert np.isclose(r1[705], r2[705])                             # the error at row 705 must be the same


def test_per_file_threshold_uses_only_train_scores():
    score = np.r_[np.ones(700), np.full(300, 999.0)]                # train scores all 1, test scores huge
    assert D.threshold(score, 700, "per_file", 1.0) == 1.0          # the per-file threshold ignores the huge test scores
    assert D.threshold(score, 700, "global", 30) == 30              # a global threshold is just the number itself