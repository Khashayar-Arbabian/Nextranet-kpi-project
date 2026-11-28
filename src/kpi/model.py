"""Gradient boosting on lag features. The model predicts the CHANGE from the last value."""
import numpy as np                                             # NumPy: fast arrays and maths
from sklearn.ensemble import HistGradientBoostingRegressor     # the gradient boosting model from scikit-learn
from . import config as C                                      # our paths and constants (HORIZON = 12, ...)
from . import evaluate as E                                    # our Day 2 functions (SEASON, steps, ...)

PARAMS = dict(max_iter=100, learning_rate=0.05, max_depth=3, min_samples_leaf=40)  # model settings
LAGS = np.arange(1, 25)      # the look-back steps 1..24 (24 x 5 min = 2 hours)
START = E.SEASON             # first usable row = 288, because we need one full day of history


def features(y, o, h):
    """Inputs for forecasts made at rows `o`, looking `h` steps ahead."""
    last = y[o]                                                # "now": the value at each origin
    lags = y[o[:, None] - LAGS] - last[:, None]                # the 24 past values, each minus now -> 24 columns
    hour = y[o[:, None] - np.arange(12)]                       # the last 12 values (the last hour), incl. now
    mean = hour.mean(axis=1) - last                            # average of the last hour minus now -> 1 column
    std = hour.std(axis=1)                                     # how noisy the last hour was -> 1 column
    yday = y[o + h - E.SEASON] - last                          # value 24 h before the target time, minus now -> 1 column
    clock = o % E.SEASON                                       # position inside the day (0..287) -> 1 column
    return np.column_stack([lags, mean, std, yday, clock])     # glue all columns together: 24+1+1+1+1 = 28


def fit_rows(y, t, o, h, **params):
    """Train on the origins `t`; forecast `h` steps ahead at the origins `o`."""
    model = HistGradientBoostingRegressor(                     # create an empty model
        loss="absolute_error",                                 # learn to minimise MAE, our metric
        early_stopping=False,                                  # always build exactly max_iter trees
        random_state=0,                                        # same result on every run
        **params)                                              # add the settings from PARAMS
    model.fit(features(y, t, h), y[t + h] - y[t])              # learn: inputs at time t -> change that happened h steps later
    return y[o] + model.predict(features(y, o, h))             # forecast = last value + predicted change


def fit_predict(y, cut, o, **params):
    """Train on rows before `cut`; forecast at origins `o` -> array (len(o), 12)."""
    pred = np.empty((len(o), C.HORIZON))                       # empty result table: one row per forecast, 12 columns
    for h in E.steps():                                        # repeat for h = 1, 2, ..., 12 (one model per horizon)
        t = np.arange(START, cut - h)                          # training origins; the answer row t+h stays inside train
        pred[:, h - 1] = fit_rows(y, t, o, h, **params)        # train this horizon's model and store its forecasts in column h-1
    return pred                                                # the (n_forecasts, 12) table, same shape as the baselines