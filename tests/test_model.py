import numpy as np                                      # NumPy: arrays and maths
from kpi import evaluate as E, model as M               # the two modules we are testing


def test_features_do_not_use_the_future():
    rng = np.random.default_rng(0)                      # random number generator with a fixed seed (same numbers every run)
    y = rng.normal(size=1000)                           # a fake series of 1000 random values
    o = np.array([700, 712])                            # two forecast origins
    before = M.features(y, o, h=6)                      # features computed on the original series
    y2 = y.copy()                                       # make a copy of the series
    y2[701:] += 999                                     # change EVERYTHING after row 700 (the future of the first origin)
    after = M.features(y2, o[:1], h=6)                  # features of the first origin, computed on the changed series
    assert np.allclose(before[0], after[0])             # they must be identical: the features never looked at the future


def test_model_on_constant_series_equals_last_value():
    y = np.full(1000, 5.0)                              # a series that never changes: always 5
    o = E.origins(1000, 700)                            # forecast origins for a train part of 700 rows
    pred = M.fit_predict(y, 700, o, max_iter=10)        # train quickly (10 trees) and forecast
    assert pred.shape == (len(o), 12)                   # one row per forecast, 12 columns
    assert np.allclose(pred, 5.0)                       # nothing ever changes, so every forecast must be 5


def test_mean_baseline_is_mean_of_last_hour():
    y = np.arange(1000, dtype=float)                    # a series that counts 0, 1, 2, ... 999
    o = np.array([700])                                 # one origin at row 700
    assert np.isclose(E.forecast_mean(y, o)[0, 0], y[689:701].mean())   # forecast = average of rows 689..700