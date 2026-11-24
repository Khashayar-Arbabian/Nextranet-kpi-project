import numpy as np
from kpi import evaluate as E


def test_origins_targets_stay_inside_series():
    o = E.origins(n=1000, cut=700)
    assert o[0] == 699                      # first forecast starts right at the end of train
    assert o[-1] + 12 <= 999                # last target exists
    assert (np.diff(o) == 12).all()         # one forecast per hour


def test_last_value_is_perfect_on_constant_series():
    y = np.full(1000, 5.0)
    o = E.origins(1000, 700)
    mae = E.mae_per_horizon(E.actuals(y, o), E.forecast_last(y, o))
    assert (mae == 0).all()


def test_yesterday_is_perfect_on_daily_pattern():
    y = np.tile(np.arange(288, dtype=float), 4)      # exact daily cycle, 4 days
    o = E.origins(len(y), int(len(y) * 0.7))
    mae = E.mae_per_horizon(E.actuals(y, o), E.forecast_yesterday(y, o))
    assert (mae == 0).all()


def test_forecasts_never_use_the_future():
    # every value a forecast reads must be at or before the origin
    o = np.array([700])
    assert (o[:, None] + E.steps() - E.SEASON <= o[:, None]).all()