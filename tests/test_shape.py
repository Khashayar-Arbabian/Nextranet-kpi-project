import numpy as np                                                  # NumPy: arrays and maths
from kpi import shape as S                                          # the module we are testing


def test_a_step_is_a_level_shift():                                 # check 1: a jump that stays is a level shift
    y = np.zeros(200)                                               # a flat series at 0 ...
    y[100:] = 10                                                    # ... that jumps to 10 at row 100 and stays there
    f = S.shape_features(y, 100, R=1.0)                             # describe the shape around row 100 (normal range 1)
    assert f["shift"] == 10 and S.categorize(f["peak"], f["shift"]) == "level shift"   # the level stayed changed


def test_one_high_point_is_a_spike():                               # check 2: one high point is a spike
    y = np.zeros(200)                                               # a flat series at 0 ...
    y[100] = 10                                                     # ... with one single high point at row 100
    f = S.shape_features(y, 100, R=1.0)                             # describe the shape around row 100
    assert f["shift"] == 0 and S.categorize(f["peak"], f["shift"]) == "spike"   # it comes back at once: a spike


def test_one_low_point_has_a_negative_peak():                       # check 3: the sign of the peak is kept
    y = np.full(200, 10.0)                                          # a flat series at 10 ...
    y[100] = 0                                                      # ... with one single low point at row 100
    assert S.shape_features(y, 100, R=1.0)["peak"] == -10           # the peak is 10 units below the usual level (negative)


def test_small_and_medium_deviations():                             # check 4: the thresholds between the types
    assert S.categorize(peak=0.4, shift=0.0) == "minor deviation"   # below 1 normal range
    assert S.categorize(peak=2.0, shift=0.1) == "moderate excursion"   # between 1 and 3 normal ranges
    assert S.categorize(peak=-5.0, shift=0.0) == "spike"            # the sign does not matter for the type