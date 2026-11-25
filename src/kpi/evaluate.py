"""Forecast evaluation: origins, two baselines, MAE and MASE per horizon."""
import numpy as np
from . import config as C

SEASON = 288   # one day = 288 steps of 5 minutes
EVERY = 12     # make one forecast per hour


def origins(n, cut):
    """Row numbers of the last known value for each forecast (cut-1, cut-1+12, ...)."""
    return np.arange(cut - 1, n - C.HORIZON, EVERY)


def steps():
    return np.arange(1, C.HORIZON + 1)          # 1..12 steps ahead


def actuals(y, o):
    """True values for the next 12 steps after each origin -> shape (n_origins, 12)."""
    return y[o[:, None] + steps()]


def forecast_last(y, o):
    """Baseline 1: repeat the last known value."""
    return np.repeat(y[o][:, None], C.HORIZON, axis=1)


def forecast_yesterday(y, o):
    """Baseline 2: the value at the same time yesterday (288 steps earlier)."""
    return y[o[:, None] + steps() - SEASON]


def mae_per_horizon(truth, pred, skip=None):
    """Mean absolute error for each of the 12 horizons. `skip` marks points to ignore."""
    err = np.abs(truth - pred)
    if skip is not None:
        err = np.where(skip, np.nan, err)
    return np.nanmean(err, axis=0)


def mase_scale(train_y):
    """MASE denominator: average one-step jump of the series in the TRAIN part."""
    return np.abs(np.diff(train_y)).mean()

def forecast_mean(y, o):
    """Extra baseline: average of the last hour (12 values)."""
    m = y[o[:, None] - np.arange(12)].mean(axis=1)    
    return np.repeat(m[:, None], C.HORIZON, axis=1) 