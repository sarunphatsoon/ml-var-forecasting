import numpy as np
import pandas as pd
from scipy.stats import norm

def fz0_daily(returns, fc, alpha):
    r = returns.loc[fc.index]
    v, e = -fc["VaR"], -fc["ES"]
    return -(1 / (alpha * e)) * (r <= v) * (v - r) + v / e + np.log(-e) - 1

def dm_test(loss_a, loss_b, lags = None):
    d = (loss_a - loss_b).values
    n = len(d)
    if lags is None:
        lags = int(np.floor(n ** (1/3)))
    dbar = d.mean()
    dc = d - dbar
    lrv = (dc @ dc) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        lrv +=2 * w * (dc[k:] @ dc[:-k]) / n
    stat = dbar / np.sqrt(lrv / n)
    return dbar, stat, 2 * norm.sf(abs(stat))

