"""Turn the generated samples (data/<topo>/*.json) into graphs"""
import glob
import json
import math
import pathlib
import re

import numpy as np

PROJ = pathlib.Path(__file__).resolve().parent.parent
TERMS = ("d", "g", "s", "b")
NET_TYPES = {"VP": "in", "VN": "in", "OUT": "out", "OUT1": "hiz", "X": "int", "TAIL": "int",
             "IBIAS": "bias", "VDD": "supply", "VSS": "gnd", "B": "gnd"}
TYPE_LIST = ["in", "out", "hiz", "int", "bias", "supply", "gnd"]
TOPOS = ["ota5t", "ts2"]
SIGNAL_NETS = ["VP", "VN", "OUT", "OUT1", "X", "TAIL", "IBIAS"]
DEV_FINGERS = {
    "XM1": ("w1", lambda p: 2 * p["f1"]), "XM2": ("w1", lambda p: 2 * p["f1"]),
    "XM3": ("w3", lambda p: p["c3"]), "XM4": ("w3", lambda p: p["c3"]),
    "XM5": ("w5", lambda p: p["c5"]), "XM6": ("w5", lambda p: p["c5"]),
    "XM7": ("w7", lambda p: p["f7"]), "XM8": ("w8", lambda p: p["f8"]),
}
DEV_FEAT_DIM = 7
NET_FEAT_DIM = len(TYPE_LIST) + len(TERMS) + 1
GLOB_FEAT_DIM = 8
PAIR_FEAT_DIM = 17


def parse_ref(txt):
    pins, devs = [], []
    for ln in txt.splitlines():
        t = ln.split()
        if not t or t[0].startswith("*"):
            continue
        if t[0].lower() == ".subckt":
            pins = t[2:]
        elif t[0].startswith("XM"):
            kv = dict(x.split("=") for x in t[6:])
            devs.append(dict(name=t[0], d=t[1], g=t[2], s=t[3], b=t[4], model=t[5],
                             w=float(kv["w"].rstrip("u")), l=float(kv["l"].rstrip("u"))))
    return pins, devs


GEO_KEYS = ("ad", "as", "pd", "ps")
GEO_SCALE = np.array([1e12, 1e12, 1e6, 1e6])


def build_graph(rec, rs):
    pins, devs = parse_ref(rec["ref_netlist"])
    nets = sorted({d[t] for d in devs for t in TERMS} | set(pins), key=lambda n: list(NET_TYPES).index(n))
    nid = {n: i for i, n in enumerate(nets)}
    p = rec["layout"]

    # device features
    X_dev = []
    for d in devs:
        wkey, nf_rule = DEV_FINGERS[d["name"]]
        nf = nf_rule(p)
        X_dev.append([d["model"].startswith("n"), math.log(d["w"]), math.log(d["l"]),
                      math.log(p[wkey]), math.log(nf), math.log(d["w"] * d["l"]), d["model"].startswith("p")])
    # edges (device -> net with terminal type)
    E_dev, E_net, E_t = [], [], []
    for i, d in enumerate(devs):
        for k, t in enumerate(TERMS):
            E_dev.append(i); E_net.append(nid[d[t]]); E_t.append(k)
    # net features
    X_net = np.zeros((len(nets), NET_FEAT_DIM))
    for j, n in enumerate(nets):
        X_net[j, TYPE_LIST.index(NET_TYPES[n])] = 1
    for i, j, k in zip(E_dev, E_net, E_t):
        X_net[j, len(TYPE_LIST) + k] += 1
    X_net[:, -1] = [n in pins for n in nets]

    # targets: pair caps (fF) over all unordered pairs
    C = np.zeros((len(nets), len(nets)))
    for a, b, c in rec["caps"]:
        if a in nid and b in nid and a != b:
            C[nid[a], nid[b]] += c * 1e15
            C[nid[b], nid[a]] += c * 1e15
    iu = np.triu_indices(len(nets), 1)
    term_of = [{} for _ in devs]
    for i, j, k in zip(E_dev, E_net, E_t):
        term_of[i].setdefault(j, set()).add(k)
    pair_f = np.zeros((len(iu[0]), 1 + 16), dtype=np.float32)
    for q, (a_, b_) in enumerate(zip(*iu)):
        for tm in term_of:
            if a_ in tm and b_ in tm:
                pair_f[q, 0] += 1
                for ta in tm[a_]:
                    for tb_ in tm[b_]:
                        pair_f[q, 1 + 4 * min(ta, tb_) + max(ta, tb_)] += 1
    S = rs["sims"]
    pre, post, orc, orf = S["pre"], S["post"], S["oracle_caps"], S["oracle_full"]
    y_geo = np.array([[g[k] for k in GEO_KEYS] for g in rs["geo"]]) * GEO_SCALE
    sch_geo = []
    for d, nf in zip(devs, rs["nfs"]):
        W, D = d["w"], 0.48
        sch_geo.append([W * D, W * D, 2 * (W + nf * D), 2 * (W + nf * D)])
    glob_f = [rec["topology"] == "ota5t", rec["topology"] == "ts2",
              math.log10(rec["CL"]), math.log10(rec.get("Cc", 1e-15)) if rec.get("Cc") else 0.0,
              math.log10(rec["sizing"]["Itail"]), math.log10(pre["GBW_Hz"]), pre["PM_deg"] / 90, pre["A0_dB"] / 50]
    return dict(
        id=rec["id"], topo=rec["topology"], nets=nets, devs=[d["name"] for d in devs],
        x_dev=np.array(X_dev, dtype=np.float32), x_net=X_net.astype(np.float32),
        e_dev=np.array(E_dev), e_net=np.array(E_net), e_t=np.array(E_t),
        pair_i=iu[0], pair_j=iu[1], pair_f=pair_f, y_pair=C[iu].astype(np.float32),
        y_tot=C.sum(1).astype(np.float32), glob=np.array(glob_f, dtype=np.float32),
        dgbw=100 * (post["GBW_Hz"] / pre["GBW_Hz"] - 1), dpm=post["PM_deg"] - pre["PM_deg"],
        dgbw_oracle=100 * (orc["GBW_Hz"] / pre["GBW_Hz"] - 1), dpm_oracle=orc["PM_deg"] - pre["PM_deg"],
        dgbw_full=100 * (orf["GBW_Hz"] / pre["GBW_Hz"] - 1), dpm_full=orf["PM_deg"] - pre["PM_deg"],
        y_geo=y_geo.astype(np.float32), sch_geo=np.array(sch_geo, dtype=np.float32), nfs=rs["nfs"],
        rec_ref=rec["ref_netlist"], sizing=rec["sizing"], CL=rec["CL"], Cc=rec.get("Cc"), sims=S,
    )


def load_all(topos=TOPOS):
    graphs = []
    for t in topos:
        for f in sorted(glob.glob(str(PROJ / "data" / t / f"{t}_*.json"))):
            rp = PROJ / "data" / "resim" / pathlib.Path(f).name
            if not rp.exists():
                continue
            rec, rs = json.loads(pathlib.Path(f).read_text()), json.loads(rp.read_text())
            if rs.get("finger_mismatch") or not all(
                    math.isfinite(rs["sims"][k][m]) for k in rs["sims"] for m in ("GBW_Hz", "PM_deg")):
                continue
            graphs.append(build_graph(rec, rs))
    return graphs


def split(graphs, seed=0, frac=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for t in TOPOS:
        idx = [i for i, g in enumerate(graphs) if g["topo"] == t]
        rng.shuffle(idx)
        n1, n2 = int(frac[0] * len(idx)), int((frac[0] + frac[1]) * len(idx))
        tr += idx[:n1]; va += idx[n1:n2]; te += idx[n2:]
    return tr, va, te


if __name__ == "__main__":
    gs = load_all()
    print(len(gs), "graphs:", {t: sum(g["topo"] == t for g in gs) for t in TOPOS})
    g = gs[0]
    print(g["id"], g["nets"], g["devs"])
    print("net totals fF:", dict(zip(g["nets"], g["y_tot"].round(2))))
    print("dGBW% post", round(g["dgbw"], 2), "oracle", round(g["dgbw_oracle"], 2))
