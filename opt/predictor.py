#!/usr/bin/env python3
"""Final GNN ensemble: train on all data, save, and predict parasitics for a 5T OTA sizing"""
import argparse
import math
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "model"))
sys.path.insert(0, str(PROJ / "gen"))
import gnn as G  # noqa: E402
from dataset import load_all, build_graph, GEO_KEYS, GEO_SCALE  # noqa: E402

MODEL = PROJ / "models" / "gnn_final.pt"
GRID, WF_MAX, WF_MIN = 0.01, 8.0, 0.5


def snap(x):
    return round(x / GRID) * GRID


def to_layout_params(sz):
    W1, W3, W5 = sz["W1"] * 1e6, sz["W3"] * 1e6, sz["W5"] * 1e6
    L1, L3, L5 = (round(sz[k] * 1e6, 2) for k in ("L1", "L3", "L5"))
    f1 = max(2, math.ceil(W1 / (2 * WF_MAX)))
    w1 = max(WF_MIN, snap(W1 / (2 * f1)))
    c3 = max(1, math.ceil(W3 / WF_MAX))
    w3 = max(WF_MIN, snap(W3 / c3))
    c5 = max(1, math.ceil(W5 / WF_MAX))
    w5 = max(WF_MIN, snap(W5 / c5))
    p = dict(w1=w1, f1=f1, L1=L1, w3=w3, c3=c3, L3=L3, w5=w5, c5=c5, L5=L5)
    p["W1_drawn"], p["W3_drawn"], p["W5_drawn"] = 2 * f1 * w1, c3 * w3, c5 * w5
    return p


def reference_netlist(p):
    n, pf = "nfet_03v3", "pfet_03v3"
    L1, L3, L5 = p["L1"], p["L3"], p["L5"]
    W1, W3, W5 = p["W1_drawn"], p["W3_drawn"], p["W5_drawn"]
    return f"""* 5T OTA reference netlist (from sizing)
.subckt ota5t VP VN OUT IBIAS VDD VSS B
XM1 X VP TAIL B {n} w={W1:.3f}u l={L1}u
XM2 OUT VN TAIL B {n} w={W1:.3f}u l={L1}u
XM3 X X VDD VDD {pf} w={W3:.3f}u l={L3}u
XM4 OUT X VDD VDD {pf} w={W3:.3f}u l={L3}u
XM5 TAIL IBIAS VSS B {n} w={W5:.3f}u l={L5}u
XM6 IBIAS IBIAS VSS B {n} w={W5:.3f}u l={L5}u
XD1 B B B B {n} w={4*p['w1']:.3f}u l={L1}u
XD2 B B B B {n} w={2*p['w5']:.3f}u l={L5}u
XD3 VDD VDD VDD VDD {pf} w={2*p['w3']:.3f}u l={L3}u
.ends ota5t
"""


def finger_plan(p):
    return [2 * p["f1"], 2 * p["f1"], p["c3"], p["c3"], p["c5"], p["c5"]]


def train_final(seeds=5, epochs=1500):
    graphs = load_all()
    idx = np.random.default_rng(0).permutation(len(graphs))
    nva = len(graphs) // 10
    va, tr = [graphs[i] for i in idx[:nva]], [graphs[i] for i in idx[nva:]]
    mns = []
    for s in range(seeds):
        mns.append(G.train(tr, va, seed=s, epochs=epochs))
        print(f"seed {s} done", flush=True)
    MODEL.parent.mkdir(exist_ok=True)
    torch.save([(m.state_dict(), norm) for m, norm in mns], MODEL)


_ENS = None


def load():
    global _ENS
    if _ENS is None:
        mns = []
        for state, norm in torch.load(MODEL, weights_only=False):
            m = G.NetlistGNN()
            m.load_state_dict(state)
            m.eval()
            mns.append((m, norm))
        _ENS = G.Ensemble(mns)
    return _ENS


def predict(sz, CL):
    p = to_layout_params(sz)
    ref = reference_netlist(p)
    nfs = finger_plan(p)
    dummy = {"A0_dB": 40.0, "GBW_Hz": 1e7, "PM_deg": 60.0, "power_W": 1e-4}
    rec = dict(id="query", topology="ota5t", ref_netlist=ref, layout=p, caps=[], sizing=sz, CL=CL)
    rs = dict(nfs=nfs, geo=[dict.fromkeys(GEO_KEYS, 0.0) for _ in nfs],
              sims={k: dummy for k in ("pre", "post", "oracle_caps", "oracle_full")})
    g = build_graph(rec, rs)
    q = load().predict([g])[0]
    caps = [(g["nets"][i], g["nets"][j], c * 1e-15)
            for i, j, c in zip(g["pair_i"], g["pair_j"], q["pair"]) if c > 1e-6]
    geo = [dict(zip(GEO_KEYS, row / GEO_SCALE)) for row in q["geo"]]
    return dict(ref=ref, layout=p, nfs=nfs, caps=caps, geo=geo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1500)
    a = ap.parse_args()
    if a.train:
        train_final(a.seeds, a.epochs)
