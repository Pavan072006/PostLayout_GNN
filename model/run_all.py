#!/usr/bin/env python3
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
import json
import math
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJ / "gen"))
from dataset import load_all, split, NET_TYPES, SIGNAL_NETS, TOPOS  # noqa: E402
import baselines as B  # noqa: E402
import gnn as G  # noqa: E402

RES = PROJ / "results"


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return {}
    ss = ((y - y.mean()) ** 2).sum()
    return dict(n=int(len(y)), MAE=float(np.abs(y - p).mean()),
                MAPE=float(100 * np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-9))),
                R2=float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan"))


def per_net_table(test, preds):
    names = np.concatenate([g["nets"] for g in test])
    topo = np.concatenate([[g["topo"]] * len(g["nets"]) for g in test])
    y = np.concatenate([g["y_tot"] for g in test])
    tab = {}
    for m, p in preds.items():
        tab[m] = {"ALL signal nets": metrics(y[np.isin(names, SIGNAL_NETS)], p[np.isin(names, SIGNAL_NETS)])}
        for n in SIGNAL_NETS + ["VDD", "VSS", "B"]:
            sel = names == n
            if sel.any():
                tab[m][n] = metrics(y[sel], p[sel])
        for t in TOPOS:
            sel = (topo == t) & np.isin(names, SIGNAL_NETS)
            if sel.any():
                tab[m][f"signal nets [{t}]"] = metrics(y[sel], p[sel])
    return tab


def _sim_one(args):
    import tb
    from netlist_utils import parse_spice, lumped_subckt
    gid, topo, ref_txt, caps, geo, nfs, Itail, CL, Cc = args
    ref = parse_spice(ref_txt)
    work = PROJ / "work" / "resim" / gid
    r = tb.simulate(topo, lumped_subckt(ref, caps, topo, geo, nfs), Itail, CL, Cc, workdir=work)
    return r.get("GBW_Hz", float("nan")), r.get("PM_deg", float("nan"))


def resim(test, pair_preds, geo_preds, label, workers=3):
    from dataset import GEO_KEYS, GEO_SCALE
    jobs = []
    for k, (g, pp) in enumerate(zip(test, pair_preds)):
        caps = [(g["nets"][i], g["nets"][j], c * 1e-15) for i, j, c in zip(g["pair_i"], g["pair_j"], pp) if c > 1e-6]
        geo = None if geo_preds is None else [dict(zip(GEO_KEYS, row / GEO_SCALE)) for row in geo_preds[k]]
        jobs.append((f"{label}_{g['id']}", g["topo"], g["rec_ref"], caps, geo, g["nfs"],
                     g["sizing"]["Itail"], g["CL"], g["Cc"]))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_sim_one, jobs))
    out = []
    for g, (gbw, pm) in zip(test, res):
        pre = g["sims"]["pre"]
        out.append((100 * (gbw / pre["GBW_Hz"] - 1), pm - pre["PM_deg"]))
    return np.array(out)


def plots(test, preds_tot, shift, tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = np.concatenate([g["nets"] for g in test])
    y = np.concatenate([g["y_tot"] for g in test])
    sig = np.isin(names, SIGNAL_NETS)
    colors = dict(zip(SIGNAL_NETS, plt.cm.tab10.colors))

    fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, m in zip(axs, ["gbm", "gnn"]):
        p = preds_tot[m]
        for n in SIGNAL_NETS:
            s = names == n
            if s.any():
                ax.scatter(y[s], p[s], s=12, alpha=0.7, color=colors[n], label=n)
        lim = [0, max(y[sig].max(), p[sig].max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
        r = tab[m]["ALL signal nets"]
        ax.set_title(f"{m.upper()}: MAE {r['MAE']:.2f} fF, MAPE {r['MAPE']:.1f}%, R$^2$ {r['R2']:.3f}")
        ax.set_xlabel("extracted net cap (Magic PEX) [fF]")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("predicted from schematic [fF]")
    axs[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Per-net parasitic capacitance, held-out test layouts")
    fig.tight_layout()
    fig.savefig(RES / "cap_scatter.png", dpi=130)
    plt.close(fig)

    # 2. error per net and model
    models = list(preds_tot)
    nets = [n for n in SIGNAL_NETS if (names == n).any()]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    wbar = 0.8 / len(models)
    for k, m in enumerate(models):
        ax.bar(np.arange(len(nets)) + k * wbar, [tab[m][n]["MAPE"] for n in nets], wbar, label=m)
    ax.set_xticks(np.arange(len(nets)) + 0.4 - wbar / 2)
    ax.set_xticklabels(nets)
    ax.set_ylabel("MAPE on test set [%]")
    ax.set_title("Per-net capacitance error by model (lower is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "error_per_net.png", dpi=130)
    plt.close(fig)

    # 3. GBW / PM shift: true (post-layout) vs predicted ways
    if shift:
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        for ax, (k, unit) in zip(axs, ((0, "GBW shift [%]"), (1, "PM shift [deg]"))):
            yt = shift["post"][:, k]
            for lab, mk in (("sim+GNN caps+geo", "o"), ("sim+GBM caps+geo", "^"), ("GNN direct", "s"),
                            ("oracle caps only", "x")):
                if lab in shift:
                    ax.scatter(yt, shift[lab][:, k], s=14, alpha=0.6, marker=mk, label=lab)
            lim = [min(yt.min(), 0) * 1.1, max(yt.max(), 0) * 1.1 + 1e-9]
            lo = min(lim[0], *(shift[l][:, k].min() for l in shift))
            hi = max(lim[1], *(shift[l][:, k].max() for l in shift))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel(f"true post-layout {unit}")
            ax.set_ylabel(f"predicted {unit}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Post-layout performance shift predicted from the schematic (test set)")
        fig.tight_layout()
        fig.savefig(RES / "shift_prediction.png", dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1500)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    graphs = load_all()
    tr, va, te = split(graphs, seed=0)
    train, val, test = [graphs[i] for i in tr], [graphs[i] for i in va], [graphs[i] for i in te]
    counts = {t: sum(g["topo"] == t for g in graphs) for t in TOPOS}
    print(f"{len(graphs)} graphs {counts}; train/val/test = {len(train)}/{len(val)}/{len(test)}")

    # baselines (fit on train+val: they have no early stopping)
    bl = B.fit_net_baselines(train + val)
    preds_tot = {k: f(test) for k, f in bl.items()}
    pair_gbm = B.fit_pair_gbm(train + val)

    # GNN ensemble
    mns = []
    for s in range(a.seeds):
        m, norm = G.train(train, val, seed=s, epochs=a.epochs, verbose=(s == 0))
        mns.append((m, norm))
        print(f"  GNN seed {s} done ({time.time()-t0:.0f}s)")
    ens = G.Ensemble(mns)
    gp = ens.predict(test)
    preds_tot["gnn"] = np.concatenate([q["tot"] for q in gp])
    # pair-sum totals from the GBM pair model, for comparison
    preds_tot["gbm_pairs"] = np.concatenate([
        np.bincount(np.r_[g["pair_i"], g["pair_j"]], weights=np.r_[pp, pp], minlength=len(g["nets"]))
        for g, pp in ((g, pair_gbm(g)) for g in test)])

    tab = per_net_table(test, preds_tot)
    def pair_metrics(pfun):
        y, p, ys, ps = [], [], [], []
        for g, pp in zip(test, pfun):
            for i, j, yy, q in zip(g["pair_i"], g["pair_j"], g["y_pair"], pp):
                y.append(yy); p.append(q)
                if g["nets"][i] in SIGNAL_NETS and g["nets"][j] in SIGNAL_NETS:
                    ys.append(yy); ps.append(q)
        return {"all pairs": metrics(y, p), "signal-signal coupling": metrics(ys, ps)}
    pair_tab = {"gbm": pair_metrics([pair_gbm(g) for g in test]), "gnn": pair_metrics([q["pair"] for q in gp])}

    # device junction geometry
    from dataset import GEO_KEYS
    geo_gbm = B.fit_geo_gbm(train + val)
    geo_preds = {"schematic estimate": [g["sch_geo"] for g in test], "gbm": [geo_gbm(g) for g in test],
                 "gnn": [q["geo"] for q in gp]}
    yg = np.vstack([g["y_geo"] for g in test])
    geo_tab = {m: {k: metrics(yg[:, i], np.vstack(p)[:, i]) for i, k in enumerate(GEO_KEYS)}
               for m, p in geo_preds.items()}

    # graph level: GBW / PM shift
    yt = np.array([[g["dgbw"], g["dpm"]] for g in test])
    shift = {"post": yt, "oracle caps only": np.array([[g["dgbw_oracle"], g["dpm_oracle"]] for g in test]),
             "oracle caps+geo": np.array([[g["dgbw_full"], g["dpm_full"]] for g in test]),
             "GNN direct": np.array([[q["dgbw"], q["dpm"]] for q in gp])}
    gb = [B.fit_graph_gbm(train + val, t)(test) for t in ("dgbw", "dpm")]
    shift["GBM direct"] = np.stack(gb, 1)
    shift["zero (pre=post)"] = np.zeros_like(yt)
    if not a.no_sim:
        print("re-simulating test circuits with predicted caps ...")
        shift["sim+GNN caps (no geo)"] = resim(test, [q["pair"] for q in gp], None, "gnnc")
        shift["sim+GNN caps+geo"] = resim(test, [q["pair"] for q in gp], geo_preds["gnn"], "gnn")
        shift["sim+GBM caps+geo"] = resim(test, [pair_gbm(g) for g in test], geo_preds["gbm"], "gbm")
    shift_tab = {}
    for k, v in shift.items():
        if k == "post":
            continue
        shift_tab[k] = {"dGBW%": metrics(yt[:, 0], v[:, 0]), "dPM_deg": metrics(yt[:, 1], v[:, 1])}
        for t in TOPOS:
            s = np.array([g["topo"] == t for g in test])
            if s.any():
                shift_tab[k][f"dGBW% [{t}]"] = metrics(yt[s, 0], v[s, 0])
                shift_tab[k][f"dPM_deg [{t}]"] = metrics(yt[s, 1], v[s, 1])

    out = dict(counts=counts, split=dict(train=len(train), val=len(val), test=len(test)),
               net_caps=tab, pair_caps=pair_tab, device_geometry=geo_tab, shift=shift_tab,
               shift_stats=dict(mean_dgbw=float(yt[:, 0].mean()), std_dgbw=float(yt[:, 0].std()),
                                mean_dpm=float(yt[:, 1].mean()), std_dpm=float(yt[:, 1].std())),
               runtime_s=time.time() - t0)
    (RES / "metrics.json").write_text(json.dumps(out, indent=1))
    (RES / "test_predictions.json").write_text(json.dumps([
        dict(id=g["id"], nets=g["nets"], y_tot=g["y_tot"].tolist(),
             **{f"pred_{m}": preds_tot[m][sum(len(h["nets"]) for h in test[:k]):][:len(g["nets"])].tolist() for m in preds_tot},
             **{f"shift_{s}": shift[s][k].tolist() for s in shift})
        for k, g in enumerate(test)]))
    plots(test, preds_tot, shift, tab)
    write_md(out)
    print((RES / "results.md").read_text())


def write_md(o):
    L = [f"# Results\n\nSamples: {o['counts']} (train/val/test = {o['split']['train']}/{o['split']['val']}/{o['split']['test']})\n",
         "## Per-net total capacitance (test set, signal nets)\n",
         "| model | MAE [fF] | MAPE [%] | R² |", "|---|---|---|---|"]
    for m, t in o["net_caps"].items():
        r = t["ALL signal nets"]
        L.append(f"| {m} | {r['MAE']:.2f} | {r['MAPE']:.1f} | {r['R2']:.3f} |")
    nets = [n for n in SIGNAL_NETS if n in o["net_caps"]["gnn"]]
    L += ["\n## MAPE [%] per net\n", "| model | " + " | ".join(nets) + " |", "|---" * (len(nets) + 1) + "|"]
    for m, t in o["net_caps"].items():
        L.append(f"| {m} | " + " | ".join(f"{t[n]['MAPE']:.1f}" for n in nets) + " |")
    L += ["\n## Pair (coupling) capacitance\n", "| model | pairs | MAE [fF] | R² |", "|---|---|---|---|"]
    for m, t in o["pair_caps"].items():
        for k, r in t.items():
            L.append(f"| {m} | {k} | {r['MAE']:.3f} | {r['R2']:.3f} |")
    L += ["\n## Device junction geometry (AD/AS um², PD/PS um), MAPE [%] / R²\n",
          "| model | AD | AS | PD | PS |", "|---|---|---|---|---|"]
    for m, t in o["device_geometry"].items():
        L.append(f"| {m} | " + " | ".join(f"{t[k]['MAPE']:.1f} / {t[k]['R2']:.3f}" for k in ("ad", "as", "pd", "ps")) + " |")
    s = o["shift_stats"]
    L += [f"\n## GBW / PM shift (post-layout minus pre-layout), test set\n",
          f"True shift on test set: dGBW = {s['mean_dgbw']:.2f} ± {s['std_dgbw']:.2f} %, dPM = {s['mean_dpm']:.2f} ± {s['std_dpm']:.2f} deg\n",
          "| method | dGBW MAE [%-pts] | dGBW R² | dPM MAE [deg] | dPM R² |", "|---|---|---|---|---|"]
    for m, t in o["shift"].items():
        L.append(f"| {m} | {t['dGBW%']['MAE']:.2f} | {t['dGBW%']['R2']:.2f} | {t['dPM_deg']['MAE']:.2f} | {t['dPM_deg']['R2']:.2f} |")
    (RES / "results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
