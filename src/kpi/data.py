from __future__ import annotations
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C

def list_series() -> list[str]:
    return sorted(p.stem for p in C.SERIES_DIR.glob("*.csv"))


def _restamp_block(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    ts = df["timestamp"]
    phase = (ts.dt.minute % 5).mode().iloc[0]
    bad = ts.duplicated(keep=False) | ((ts.dt.minute % 5) != phase)
    idx = np.flatnonzero(bad.to_numpy())
    if idx.size == 0:
        return df, 0
    assert (np.diff(idx) == 1).all(), "bad rows are not one contiguous block"
    assert 0 < idx[0] and idx[-1] < len(df) - 1
    new = pd.date_range(ts.iloc[idx[0] - 1] + pd.Timedelta(minutes=5),
                        periods=idx.size, freq=C.STEP)
    assert new[-1] < ts.iloc[idx[-1] + 1], "re-stamped block would overlap next row"
    df = df.copy()
    df.loc[idx, "timestamp"] = new.to_numpy()
    return df, int(idx.size)


def load_series(name: str, max_fill: int = 6) -> pd.DataFrame:
    df = pd.read_csv(C.SERIES_DIR / f"{name}.csv", parse_dates=["timestamp"])
    df, n_fixed = _restamp_block(df)
    assert df["timestamp"].is_monotonic_increasing
    assert not df["timestamp"].duplicated().any()

    s = df.set_index("timestamp")["value"].astype(float)
    grid = pd.date_range(s.index[0], s.index[-1], freq=C.STEP)
    assert s.index.isin(grid).all(), "timestamps off the 5-min grid"
    s = s.reindex(grid)
    imputed = s.isna()
    s = s.interpolate(method="time", limit=max_fill, limit_direction="both")
    assert s.notna().all()

    out = pd.DataFrame({"value": s, "imputed": imputed})
    out.attrs.update(name=name, restamped_rows=n_fixed)
    return out


def split(df: pd.DataFrame, frac: float = C.TRAIN_FRAC):
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


@lru_cache(maxsize=1)
def _raw_labels():
    with open(C.LABELS, encoding="utf-8") as f1, open(C.WINDOWS, encoding="utf-8") as f2:
        return json.load(f1), json.load(f2)


def load_events(name: str) -> pd.DataFrame:
    labels, windows = _raw_labels()
    key = f"realAWSCloudwatch/{name}.csv"
    L = [pd.Timestamp(x) for x in labels[key]]
    W = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in windows[key]]
    rows = []
    for t in L:
        inside = [i for i, (a, b) in enumerate(W) if a <= t <= b]
        w = inside[0] if inside else int(np.argmin([abs((a - t).total_seconds()) for a, _ in W]))
        rows.append(dict(label_time=t, window_id=w,
                         window_start=W[w][0], window_end=W[w][1]))
    return pd.DataFrame(rows, columns=["label_time", "window_id", "window_start", "window_end"])