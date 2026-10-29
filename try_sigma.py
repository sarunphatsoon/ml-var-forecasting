import pandas as pd
from features import to_returns
from ml_models import HARS_COL, har_model, gbm_model, forecast_sigma, fhs_var_es
from backtest import hits, kupiec

df = pd.read_csv("portfolio_risk_dataset.csv", parse_dates = ["Date"], index_col = "Date").iloc[:-1]
r = to_returns(df["SPY"])

s_har = forecast_sigma(r, har_model, cols = HARS_COL)
s_gbm = forecast_sigma(r, gbm_model)
print(len(s_har), s_har.index[0].date())

for name, s in (("har_fhs", s_har), ("gbm_fhs", s_gbm)):
    for a in (0.05, 0.01):
        fc = fhs_var_es(r, s, a)
        h = hits(r, fc["VaR"])
        print(f"{name} alpha = {a}: hits = {h.sum()} expected = {a * len(h):.0f} kupiec = {kupiec(h, a)[1]:.3f}")

