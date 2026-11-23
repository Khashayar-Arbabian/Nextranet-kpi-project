import pandas as pd
from kpi import data


def test_all_series_load_on_regular_grid():
    names = data.list_series()
    assert len(names) == 17
    for n in names:
        df = data.load_series(n)
        assert not df.index.duplicated().any()
        assert (df.index.to_series().diff().dropna() == pd.Timedelta("5min")).all()
        assert df["value"].notna().all()


def test_dst_block_restamped():
    for n in ["ec2_disk_write_bytes_1ef3de", "ec2_network_in_5abac7"]:
        df = data.load_series(n)
        assert df.attrs["restamped_rows"] == 12
        assert df["imputed"].sum() == 0


def test_label_counts_match_brief():
    train = test = files_with_test = 0
    for n in data.list_series():
        _, te = data.split(data.load_series(n))
        ev = data.load_events(n)
        k = int((ev.label_time >= te.index[0]).sum())
        test += k
        train += len(ev) - k
        files_with_test += k > 0
    assert (train, test, files_with_test) == (22, 9, 7)