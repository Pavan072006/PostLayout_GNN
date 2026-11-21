import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import TERMS, TYPE_LIST, NET_TYPES


def net_features(g):
    W = np.exp(g["x_dev"][:, 1])
    L = np.exp(g["x_dev"][:, 2])
    NF = np.exp(g["x_dev"][:, 4])
    is_p = g["x_dev"][:, 6]
    n = len(g["nets"])
    F = np.zeros((n, 10))
    for i, j, t in zip(g["e_dev"], g["e_net"], g["e_t"]):
        T = TERMS[t]
        if T in "ds":
            F[j, 0] += W[i]; F[j, 1] += NF[i]; F[j, 2] += W[i] * is_p[i]
        elif T == "g":
            F[j, 3] += W[i]; F[j, 4] += W[i] * L[i]
        else:
            F[j, 5] += W[i]
        F[j, 6 + t] += 1
    typ = np.zeros((n, len(TYPE_LIST)))
    for j, name in enumerate(g["nets"]):
        typ[j, TYPE_LIST.index(NET_TYPES[name])] = 1
    totW = np.full((n, 1), W.sum())
    topo = np.tile(g["glob"][:2], (n, 1))
    return np.hstack([F, np.log1p(F[:, :6]), typ, totW, np.log(totW), topo])


def pair_features(g, Fn=None):
    Fn = net_features(g) if Fn is None else Fn
    i, j = g["pair_i"], g["pair_j"]
    dev_nets = [set() for _ in g["devs"]]
    for d, n in zip(g["e_dev"], g["e_net"]):
        dev_nets[d].add(n)
    shared = np.array([sum((a in s) and (b in s) for s in dev_nets) for a, b in zip(i, j)])
    return np.hstack([Fn[i] + Fn[j], np.abs(Fn[i] - Fn[j]), shared[:, None], g["pair_f"]])


def stack_nets(graphs):
    X = np.vstack([net_features(g) for g in graphs])
    y = np.concatenate([g["y_tot"] for g in graphs])
    names = np.concatenate([g["nets"] for g in graphs])
    return X, y, names


class TypeMean:
    def fit(self, names, y):
        self.m = {n: y[names == n].mean() for n in set(names)}
        self.glob = y.mean()
        return self

    def predict(self, names):
        return np.array([self.m.get(n, self.glob) for n in names])


def fit_net_baselines(train):
    X, y, names = stack_nets(train)
    b0 = TypeMean().fit(names, y)
    b1 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    b2 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X, np.log(y + 1e-3))
    return dict(
        type_mean=lambda gs: np.concatenate([b0.predict(np.array(g["nets"])) for g in gs]),
        ridge=lambda gs: np.maximum(b1.predict(np.vstack([net_features(g) for g in gs])), 0),
        gbm=lambda gs: np.exp(b2.predict(np.vstack([net_features(g) for g in gs]))) - 1e-3,
    )


def fit_pair_gbm(train):
    X = np.vstack([pair_features(g) for g in train])
    y = np.concatenate([g["y_pair"] for g in train])
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0).fit(X, np.log1p(y))
    return lambda g: np.maximum(np.expm1(m.predict(pair_features(g))), 0)


def fit_graph_gbm(train, target="dgbw"):
    def gf(g):
        Fn = net_features(g)
        sig = [k for k, n in enumerate(g["nets"]) if NET_TYPES[n] not in ("supply", "gnd")]
        return np.concatenate([g["glob"], Fn[sig, :6].sum(0), [Fn[0, 24]]])
    X = np.array([gf(g) for g in train])
    y = np.array([g[target] for g in train])
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X, y)
    return lambda gs: m.predict(np.array([gf(g) for g in gs]))


def fit_geo_gbm(train):
    def df(g):
        tied = []
        for k in range(len(g["devs"])):
            nets = {TERMS[t]: n for d, n, t in zip(g["e_dev"], g["e_net"], g["e_t"]) if d == k}
            tied.append([nets["d"] == nets["g"], nets["s"] == nets["b"]])
        return np.hstack([g["x_dev"], np.array(tied, float), np.tile(g["glob"][:2], (len(g["devs"]), 1))])
    X = np.vstack([df(g) for g in train])
    Y = np.vstack([g["y_geo"] for g in train])
    ms = [HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X, np.log1p(Y[:, k]))
          for k in range(4)]
    return lambda g: np.stack([np.expm1(m.predict(df(g))) for m in ms], 1)
