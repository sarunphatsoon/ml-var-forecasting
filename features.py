import numpy as np
import pandas as pd

def to_returns(prices):
    return np.log(prices.dropna()).diff().dropna()

def make_features(returns):
    a = returns.abs()
    X = pd.DataFrame(index = returns.index)
    X["abs_lag1"] = a.shift(1)
    X["abs_mean5"] = a.rolling(5).mean().shift(1)
    X["abs_mean22"] = a.rolling(22).mean().shift(1)
    X["abs_mean66"] = a.rolling(66).mean().shift(1)
    X["down_mean22"] = (-returns.clip(upper=0)).rolling(22).mean().shift(1)
    X["ret_lag1"] = returns.shift(1)
    return X.dropna()

def make_target(returns, X):
    return returns.abs().loc[X.index]

if __name__ == "__main__":
    df = pd.read_csv("portfolio_risk_dataset.csv", parse_dates = ["Date"], index_col = "Date").iloc[:-1]
    r = to_returns(df["SPY"])
    X = make_features(r); y = make_target(r, X)
    print(X.shape, y.shape)

    Xc = make_features(r.iloc[:-50])
    print(np.allclose(X.loc[Xc.index].values, Xc.values))

    t = r.index[1000]
    r2 = r.copy(); r2.loc[t] += 0.20
    print(np.allclose(X.loc[t].values, make_features(r2).loc[t].values))
