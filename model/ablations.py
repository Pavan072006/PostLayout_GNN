#!/usr/bin/env python3
"""Ablations (caps + junction geometry only, no re-simulation):"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import load_all, split, SIGNAL_NETS  # noqa: E402
import baselines as B  # noqa: E402
import gnn as G  # noqa: E402
from run_all import metrics, RES  # noqa: E402

EPOCHS = int(os.environ.get("ABL_EPOCHS", 1000))
SEEDS = int(os.environ.get("ABL_SEEDS", 2))


def sig_metrics(test, tot_pred):
    names = np.concatenate([g["nets"] for g in test])
    y = np.concatenate([g["y_tot"] for g in test])
    s = np.isin(names, SIGNAL_NETS)
    return metrics(y[s], np.asarray(tot_pred)[s])


def geo_metrics(test, geo_pred):
    return metrics(np.vstack([g["y_geo"] for g in test]).ravel(), np.vstack(geo_pred).ravel())


def run_gnn(train, val, test, K=3):
    ens = G.Ensemble([G.train(train, val, seed=s, epochs=EPOCHS, K=K) for s in range(SEEDS)])
    gp = ens.predict(test)
    return dict(net=sig_metrics(test, np.concatenate([q["tot"] for q in gp])),
                geo=geo_metrics(test, [q["geo"] for q in gp]))


def run_gbm(train, test):
    bl = B.fit_net_baselines(train)
    geo = B.fit_geo_gbm(train)
    return dict(net=sig_metrics(test, bl["gbm"](test)), geo=geo_metrics(test, [geo(g) for g in test]))


def main():
    graphs = load_all()
    tr, va, te = split(graphs, seed=0)
    train, val, test = [graphs[i] for i in tr], [graphs[i] for i in va], [graphs[i] for i in te]
    out = {}

    print("A. message-passing depth")
    out["depth"] = {f"K={K}": run_gnn(train, val, test, K) for K in (0, 1, 2, 3)}
    print(json.dumps(out["depth"], indent=0)[:600])

    print("B. topology transfer (train 5T OTA -> test two-stage)")
    ota = [g for g in graphs if g["topo"] == "ota5t"]
    ts2 = [g for g in graphs if g["topo"] == "ts2"]
    rng = np.random.default_rng(1)
    ts2 = [ts2[i] for i in rng.permutation(len(ts2))]
    n_ota_val = len(ota) // 6
    ota_tr, ota_va = ota[n_ota_val:], ota[:n_ota_val]
    k = min(20, len(ts2) // 4)
    ts2_test = ts2[2 * k:]
    out["transfer"] = {
        "zero-shot GNN": run_gnn(ota_tr, ota_va, ts2_test),
        "zero-shot GBM": run_gbm(ota_tr + ota_va, ts2_test),
        f"few-shot (+{k} ts2) GNN": run_gnn(ota_tr + ts2[:k], ota_va + ts2[k:2 * k], ts2_test),
        f"few-shot (+{k} ts2) GBM": run_gbm(ota_tr + ota_va + ts2[:2 * k], ts2_test),
    }
    out["transfer_n_test"] = len(ts2_test)

    print("C. learning curve")
    rng = np.random.default_rng(2)
    lc = {}
    for frac in (0.25, 0.5, 1.0):
        sub = [train[i] for i in rng.permutation(len(train))[: max(10, int(frac * len(train)))]]
        lc[frac] = {"n_train": len(sub), "gnn": run_gnn(sub, val, test), "gbm": run_gbm(sub + val, test)}
    out["learning_curve"] = lc
    (RES / "ablations.json").write_text(json.dumps(out, indent=1))

    L = ["# Ablations\n", f"(GNN: ensemble of {SEEDS} seeds, {EPOCHS} epochs; metrics on signal-net total caps and on all device-geometry values)\n",
         "## A. Message-passing depth\n", "| K | net cap MAPE % | net cap R² | geometry MAPE % | geometry R² |", "|---|---|---|---|---|"]
    for k, r in out["depth"].items():
        L.append(f"| {k} | {r['net']['MAPE']:.1f} | {r['net']['R2']:.3f} | {r['geo']['MAPE']:.1f} | {r['geo']['R2']:.3f} |")
    L += [f"\n## B. Topology transfer: train on 5T OTA, test on {out['transfer_n_test']} two-stage OTAs\n",
          "| setting | net cap MAPE % | net cap R² | geometry MAPE % | geometry R² |", "|---|---|---|---|---|"]
    for k, r in out["transfer"].items():
        L.append(f"| {k} | {r['net']['MAPE']:.1f} | {r['net']['R2']:.3f} | {r['geo']['MAPE']:.1f} | {r['geo']['R2']:.3f} |")
    L += ["\n## C. Learning curve (same test set)\n", "| train graphs | GNN net MAPE % | GBM net MAPE % | GNN geo MAPE % | GBM geo MAPE % |", "|---|---|---|---|---|"]
    for f, r in lc.items():
        L.append(f"| {r['n_train']} | {r['gnn']['net']['MAPE']:.1f} | {r['gbm']['net']['MAPE']:.1f} | {r['gnn']['geo']['MAPE']:.1f} | {r['gbm']['geo']['MAPE']:.1f} |")
    (RES / "ablations.md").write_text("\n".join(L) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    n = [r["n_train"] for r in lc.values()]
    for m, st in (("gnn", "o-"), ("gbm", "s--")):
        ax.plot(n, [r[m]["net"]["MAPE"] for r in lc.values()], st, label=f"{m.upper()} net caps")
        ax.plot(n, [r[m]["geo"]["MAPE"] for r in lc.values()], st, alpha=0.5, label=f"{m.upper()} junction geometry")
    ax.set_xlabel("training layouts")
    ax.set_ylabel("test MAPE [%]")
    ax.set_title("Learning curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "learning_curve.png", dpi=130)
    print((RES / "ablations.md").read_text())


if __name__ == "__main__":
    main()
