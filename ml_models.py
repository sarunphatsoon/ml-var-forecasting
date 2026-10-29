import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from features import make_features, make_target

HARS_COL = ['abs_lag1', "abs_mean5", "abs_mean22"]

def har_model():
    return LinearRegression()

def gbm_model():
    return HistGradientBoostingRegressor(max_depth = 3, max_iter = 200, learning_rate = 0.05, min_samples_leaf = 50, l2_regularization = 1.0, random_state = 0)

def forecast_sigma(returns, make_model, cols = None, init = 1000, step = 20):
    X = make_features(returns)
    if cols is not None:
        X = X[cols]
    y = make_target(returns, X)
    out = []
    for start in range(init, len(X), step):
        model = make_model()
        model.fit(X.iloc[:start], y.iloc[:start])
        stop = min(start + step, len(X))
        pred = model.predict(X.iloc[start:stop])
        pred = np.maximum(pred, 0.2 * y.iloc[:start].mean())
        out.append(pd.Series(pred, index = X.index[start:stop]))
    return pd.concat(out)

def fhs_var_es(returns, sigma, alpha, min_oos = 500):
    r = returns.loc[sigma.index].values
    s = sigma.values
    z = r / s
    rows = []
    for i in range(min_oos, len(z)):
        past = z[:i]
        q = np.quantile(past, alpha)
        rows.append((sigma.index[i], -s[i] * q, -s[i] * past[past <=q].mean()))
    return pd.DataFrame(rows, columns = ["Date", "VaR", "ES"]).set_index("Date")

