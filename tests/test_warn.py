import numpy as np                                                  # NumPy: arrays and maths
from kpi import warn as W                                           # the module we are testing


def make_times(n=100):                                              # helper: a list of origin times
    return np.datetime64("2014-01-01T00:00") + np.arange(n) * np.timedelta64(5, "m")   # n origins, 5 minutes apart


def test_zone_marks_the_six_origins_before_an_event():              # check 1: the 30-minute zone
    times = make_times()                                            # 100 origin times
    z = W.zone(times, [times[50]])                                  # an event exactly at origin 50
    assert np.flatnonzero(z).tolist() == [44, 45, 46, 47, 48, 49]   # the 30 minutes (6 origins) before it, and nothing else


def test_features_do_not_use_the_future():                          # check 2: no peeking
    y = np.random.default_rng(0).normal(size=500)                   # a fake series
    o = np.array([300])                                             # one origin at row 300
    y2 = y.copy()                                                   # a copy of the series
    y2[301:] += 999                                                 # change everything after the origin
    assert np.allclose(W.raw_features(y, o), W.raw_features(y2, o))   # the features must not change


def test_percentiles():                                             # check 3: percentiles among the reference values
    ref = np.array([[1.0], [2.0], [3.0], [4.0]])                    # four reference values
    X = np.array([[0.0], [2.0], [4.0], [10.0]])                     # four values to convert
    assert W.to_percentiles(X, ref)[:, 0].tolist() == [0.0, 0.5, 1.0, 1.0]   # share of reference values that are <= each value


def test_warning_scoring():                                         # check 4: a correct and a false warning
    times = make_times()                                            # 100 origin times
    labels = [times[50]]                                            # one labeled anomaly at origin 50
    alarm = np.zeros(100, dtype=bool)                               # no alarms yet
    alarm[[10, 47]] = True                                          # one alarm far too early, one alarm 15 minutes before the anomaly
    r = W.score_warnings(alarm, times, labels)                      # score it
    assert (r["labels"], r["warned"], r["warnings"], r["correct"], r["false"]) == (1, 1, 2, 1, 1)   # 2 warnings: 1 correct, 1 false
    assert r["minutes"] == [15.0]                                   # the anomaly was warned 15 minutes ahead


def test_an_alarm_that_was_already_on_does_not_count_as_early_warning():   # check 5: a long-running alarm
    times = make_times()                                            # 100 origin times
    alarm = np.zeros(100, dtype=bool)                               # no alarms yet
    alarm[20:61] = True                                             # one long alarm from origin 20 to 60 (it started before the zone)
    r = W.score_warnings(alarm, times, [times[50]])                 # the anomaly is at origin 50
    assert (r["warned"], r["warnings"], r["correct"], r["false"]) == (0, 1, 0, 1)   # the episode started too early: not a warning in time