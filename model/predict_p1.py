#!/usr/bin/env python3
"""Predict post-layout shifts for the final designs of LLM_Analog_Sizing"""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys

PROJ = pathlib.Path(__file__).resolve().parent.parent
AA = PROJ.parent / "analog_agent"
REFINE = AA / "results" / "refine_groq-openai-gpt-oss-120b-effort-low.json"
OUT, OUT_RS = PROJ / "data" / "p1", PROJ / "data" / "p1_resim"
RES = PROJ / "results"
sys.path.insert(0, str(PROJ / "gen"))
sys.path.insert(0, str(PROJ / "model"))
BASE = 90000


def designs():
    sys.path.insert(0, str(AA))
    from agent.specs import SPECS
    cl = {s["id"]: s["CL"] for s in SPECS}
    out = []
    for k, r in enumerate(json.loads(REFINE.read_text()), 1):
        last = r["trace"][-1]
        p = last["proposal"]
        gmid = dict(gmid1=p["gmid1"], gmid3=p["gmid3"], gmid5=p["gmid5"])
        out.append((BASE + k, r["spec"], last["sizing"], gmid, cl[r["spec"]]))
    return out


# layout + PEX + sims, reusing the dataset generator
def _gen_one(job):
    import gen_ota5t as G
    idx, spec, sz, gmid, CL = job
    G.sample_sizing = lambda rng, lut: (sz, gmid, CL)
    return spec, G.one((idx, 0, True, OUT, PROJ / "work" / "p1"))


def gen(workers):
    import gen_ota5t as G
    import resim_dataset as R
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_RS.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(workers, initializer=G.init_worker) as pool:
        for spec, (idx, status, dt) in pool.imap_unordered(_gen_one, designs()):
            print(f"{spec} #{idx}: {status} ({dt:.0f}s)", flush=True)
    R.OUT = OUT_RS
    for f in sorted(OUT.glob("*.json")):
        print(f.name, R.one(str(f)), flush=True)


# train the GNN on the usual split, predict the P1 designs
def predict(seeds, epochs):
    import numpy as np
    from dataset import load_all, split, build_graph
    import gnn
    from run_all import resim
    graphs = load_all()
    tr, va, _ = split(graphs, seed=0)
    train, val = [graphs[i] for i in tr], [graphs[i] for i in va]
    ens = gnn.Ensemble([gnn.train(train, val, seed=s, epochs=epochs) for s in range(seeds)])
    spec_of = {f"ota5t_{i:05d}": s for i, s, *_ in designs()}
    p1 = []
    for f in sorted(OUT.glob("*.json")):
        rs = OUT_RS / f.name
        if rs.exists():
            p1.append(build_graph(json.loads(f.read_text()), json.loads(rs.read_text())))
    gp = ens.predict(p1)
    pred = resim(p1, [q["pair"] for q in gp], [q["geo"] for q in gp], "p1")
    rows = []
    for g, q, (dg, dp) in zip(p1, gp, pred):
        rows.append(dict(spec=spec_of[g["id"]], CL_pF=g["CL"] * 1e12,
                         W1_um=g["sizing"]["W1"] * 1e6,
                         true_dgbw=g["dgbw"], pred_dgbw=float(dg),
                         true_dpm=g["dpm"], pred_dpm=float(dp),
                         true_cout_fF=float(g["y_tot"][g["nets"].index("OUT")]),
                         pred_cout_fF=float(q["tot"][g["nets"].index("OUT")])))
    rows.sort(key=lambda r: int(r["spec"][1:]))
    eg = np.array([r["pred_dgbw"] - r["true_dgbw"] for r in rows])
    ep = np.array([r["pred_dpm"] - r["true_dpm"] for r in rows])
    summ = dict(n=len(rows), mae_dgbw=float(abs(eg).mean()), mae_dpm=float(abs(ep).mean()),
                mean_true_dgbw=float(np.mean([r["true_dgbw"] for r in rows])))
    (RES / "p1_designs.json").write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))
    lines = ["| spec | CL pF | W1 um | true dGBW % | GNN dGBW % | true dPM deg | GNN dPM deg | C(OUT) true / GNN fF |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['spec']} | {r['CL_pF']:.0f} | {r['W1_um']:.0f} | {r['true_dgbw']:+.2f} | {r['pred_dgbw']:+.2f} | "
                     f"{r['true_dpm']:+.2f} | {r['pred_dpm']:+.2f} | {r['true_cout_fF']:.1f} / {r['pred_cout_fF']:.1f} |")
    lines.append(f"\nMAE: dGBW {summ['mae_dgbw']:.2f} %-pts, dPM {summ['mae_dpm']:.2f} deg (n={len(rows)})")
    (RES / "p1_designs.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gen", "predict"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1500)
    a = ap.parse_args()
    gen(a.workers) if a.step == "gen" else predict(a.seeds, a.epochs)
