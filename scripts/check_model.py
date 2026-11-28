import numpy as np                                             # NumPy: arrays and maths
from kpi import data, evaluate as E, model as M                # our own modules from Days 1-3

df = data.load_series("ec2_cpu_utilization_5f5533")            # load the cleaned series (Day 1 loader)
train, _ = data.split(df)                                      # split 70/30; we keep only the train part (for its length)
y = df["value"].to_numpy()                                     # the whole series as a plain array
o = E.origins(len(y), len(train))                              # the 100 forecast origins on the test part (Day 2)

X = M.features(y, o, 1)                                        # model inputs for horizon 1: one row per forecast
print("features:", X.shape)                                    # expect (100, 28)

p = M.fit_predict(y, len(train), o[:1], **M.PARAMS)[0]         # train the 12 models, forecast only the first origin
t = E.actuals(y, o)[0]                                         # the 12 real values after the first origin (answer key)
print("forecast:", np.round(p, 2))                             # the model's 12 forecasts
print("actual:  ", np.round(t, 2))                             # the 12 real values
print("model error: %.2f | last value error: %.2f" % (np.abs(p - t).mean(), np.abs(y[o[0]] - t).mean()))  # average error: model vs last value