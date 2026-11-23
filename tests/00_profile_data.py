import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from kpi import config as C, data

C.TABLES.mkdir(parents=True, exist_ok=True)
C.FIGURES.mkdir(parents=True, exist_ok=True)

names = data.list_series()
fig, axes = plt.subplots(6, 3, figsize=(20, 22))
axes = axes.ravel()
rows = []

for ax, name in zip(axes, names):
    df = data.load_series(name)
    tr, te = data.split(df)
    ev = data.load_events(name)
    cut, v = te.index[0], df["value"]
    n_te = int((ev.label_time >= cut).sum())

    rows.append(dict(
        file=name, rows=len(df), start=df.index[0], end=df.index[-1],
        restamped=df.attrs["restamped_rows"], imputed=int(df["imputed"].sum()),
        mean=v.mean(), std=v.std(), min=v.min(), max=v.max(),
        zero_pct=round((v == 0).mean() * 100, 1),
        train_rows=len(tr), test_rows=len(te), test_start=cut,
        labels_train=len(ev) - n_te, labels_test=n_te,
        windows=ev["window_id"].nunique()))

    straddle = ev[(ev.window_end >= cut) & (ev.label_time < cut)]
    for _, w in straddle.iterrows():
        print("STRADDLING WINDOW:", name, "label", w.label_time, "window", w.window_start, "->", w.window_end, "| test starts", cut)

    ax.plot(v.index, v.values, lw=0.5, color="#1f4e79")
    for _, w in ev.drop_duplicates("window_id").iterrows():
        ax.axvspan(w.window_start, w.window_end, color="#e8a33d", alpha=0.25)
    for t in ev["label_time"]:
        ax.axvline(t, color="red", lw=1)
    ax.axvline(cut, color="k", ls="--", lw=1.2)
    ax.axvspan(cut, v.index[-1], color="grey", alpha=0.08)
    if v.max() / (v.median() + 1e-9) > 200 or name.startswith("ec2_network"):
        ax.set_yscale("symlog", linthresh=max(v.median(), 1))
    ax.set_title(name, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(labelsize=8)

for ax in axes[len(names):]:
    ax.axis("off")
fig.suptitle("17 series: orange = windows, red = exact anomalies, dashed = 70/30 split", fontsize=15)
plt.tight_layout()
fig.savefig(C.FIGURES / "overview.png", dpi=110)

prof = pd.DataFrame(rows)
prof.to_csv(C.TABLES / "data_profile.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(prof[["file","rows","restamped","imputed","zero_pct","train_rows","test_rows","labels_train","labels_test","windows"]].to_string())
print(prof[["imputed", "restamped", "labels_train", "labels_test", "windows"]].sum())
print("files with imputed:", int((prof.imputed>0).sum()))