import pickle
import pandas as pd

from features import to_returns
from dm import fz0_daily, dm_test

with open("forecast_ml.pkl", "rb") as f:
    forecasts = pickle.load(f)

df = pd.read_csv("portfolio_risk_dataset.csv", parse_dates = ["Date"], index_col = "Date").iloc[:-1]
ASSETS = ["SPY", "KRE", "JPM", "AAPL", "TSLA", "ES3.SI", "BTC-USD"]
PAIRS = [("har_fhs", "garch_fhs"), ("gbm_fhs", "garch_fhs"), ("gbm_fhs", "har_fhs"), ("ewma", "garch_fhs")]

rows = []
for asset in ASSETS:
    r = to_returns(df[asset])
    for alpha in (0.05, 0.01):
        loss = {m: fz0_daily(r, forecasts[(asset, alpha, m)], alpha) for m in {x for p in PAIRS for x in p}}
        for a, b in PAIRS:
            dbar, stat, p = dm_test(loss[a], loss[b])
            rows.append({"asset": asset, "alpha": alpha, "A": a, "B":b, "mean_duff": dbar, "dm_stat": stat, "p_value": p})

res = pd.DataFrame(rows)
res.to_csv("dm_results.csv", index = False)
print(res.round(4).to_string(index = False))