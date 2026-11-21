#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import math
import multiprocessing as mp
import os
import pathlib
import random
import shutil
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
AA = PROJ.parent / "analog_agent"
sys.path.insert(0, str(HERE))
from netlist_utils import parse_spice, match_nets, net_caps, lumped_subckt, pex_to_ngspice  # noqa: E402
from pex_tools import extract_pex, run_lvs  # noqa: E402
import tb  # noqa: E402

TOPO = "ts2"


def sample_sizing(rng, lut):
    while True:
        I = 10 ** rng.uniform(math.log10(5e-6), math.log10(150e-6))
        L = {k: round(math.exp(rng.uniform(math.log(0.28), math.log(2.0))), 2) for k in ("L1", "L3", "L5")}
        L7 = round(math.exp(rng.uniform(math.log(0.28), math.log(1.0))), 2)
        g1, g3, g5, g7 = rng.uniform(5, 22), rng.uniform(5, 22), rng.uniform(5, 22), rng.uniform(6, 16)
        k = rng.uniform(1.0, 4.0)
        try:
            sz = dict(L1=L["L1"] * 1e-6, L3=L["L3"] * 1e-6, L5=L["L5"] * 1e-6, L7=L7 * 1e-6, Itail=I,
                      W1=lut.width("nmos", L["L1"] * 1e-6, g1, I / 2),
                      W3=lut.width("pmos", L["L3"] * 1e-6, g3, I / 2),
                      W5=lut.width("nmos", L["L5"] * 1e-6, g5, I),
                      W7=lut.width("pmos", L7 * 1e-6, g7, k * I))
            sz["W8"] = k * sz["W5"]
        except Exception:
            continue
        Ws = [sz[w] for w in ("W1", "W3", "W5", "W7", "W8")]
        if all(1e-6 <= w <= 120e-6 for w in Ws) and math.isfinite(sum(Ws)):
            CL = 10 ** rng.uniform(math.log10(0.2e-12), math.log10(2e-12))
            r = 10 ** rng.uniform(0, math.log10(5))
            Cc = min(max(r * g1 * CL / (2 * k * g7), 0.1 * CL), 3 * CL)
            return sz, dict(gmid1=g1, gmid3=g3, gmid5=g5, gmid7=g7, k=k), CL, Cc


_G = {}


def init_worker():
    os.environ.setdefault("PDK_ROOT", "/foss/pdks")
    sys.path.insert(0, str(AA))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import twostage_layout as T  # noqa: E402
        from lut.lookup import LUT  # noqa: E402
    _G.update(T=T, lut=LUT())


def one(args):
    idx, seed, do_lvs, outdir, workroot = args
    out = outdir / f"{TOPO}_{idx:05d}.json"
    if out.exists():
        return idx, "skip", 0.0
    t0 = time.time()
    work = workroot / f"{TOPO}_{idx:05d}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    T = _G["T"]
    rng = random.Random(seed * 100003 + idx)
    try:
        sz, gmid, CL, Cc = sample_sizing(rng, _G["lut"])
        p = T.to_layout_params(sz)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            top = T.build(p)
            cell = f"{TOPO}_s{idx}"
            top.name = cell
            gds = work / f"{TOPO}.gds"
            top.write_gds(str(gds))
            from glayout.util.comp_utils import evaluate_bbox
            w, h = evaluate_bbox(top)
        ref_txt = T.reference_netlist(p)
        (work / "ref.spice").write_text(ref_txt)
        t_lay = time.time() - t0
        lvs = run_lvs(gds, work / "ref.spice", work, cell, TOPO) if do_lvs else None
        pex_txt = extract_pex(gds, work, cell).read_text()
        ref, ext = parse_spice(ref_txt), parse_spice(pex_txt)
        mapping = match_nets(ref, ext)
        caps = [(mapping.get(a, a), mapping.get(b, b), c) for a, b, c in ext["caps"]]
        nc = net_caps(ext["caps"], mapping)
        keys = ("A0_dB", "GBW_Hz", "PM_deg", "power_W")
        sims = {}
        for tag, dut in (("pre", lumped_subckt(ref, None, TOPO)),
                         ("oracle", lumped_subckt(ref, caps, TOPO)),
                         ("post", pex_to_ngspice(pex_txt, TOPO))):
            r = tb.simulate(TOPO, dut, sz["Itail"], CL, Cc, workdir=work)
            if not r.get("ok") or not math.isfinite(r["GBW_Hz"]):
                raise RuntimeError(f"{tag} sim failed: {r.get('error', 'no unity-gain crossing')}")
            sims[tag] = {k: r[k] for k in keys}
        rec = dict(id=f"{TOPO}_{idx:05d}", topology=TOPO, seed=seed, sizing=sz, gmid=gmid, CL=CL, Cc=Cc,
                   layout=p, area_um2=w * h, bbox_um=[w, h], lvs=lvs,
                   ref_netlist=ref_txt, pex_netlist=pex_txt, net_map=mapping,
                   caps=caps, net_caps=nc, sims=sims, t_layout_s=t_lay, t_total_s=time.time() - t0)
        out.write_text(json.dumps(rec))
        shutil.rmtree(work, ignore_errors=True)
        return idx, "ok" + ("" if lvs is None else f" lvs={'MATCH' if lvs else 'FAIL'}"), time.time() - t0
    except Exception as e:
        (work / "error.txt").write_text(traceback.format_exc())
        return idx, f"FAIL {type(e).__name__}: {str(e)[:120]}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--lvs-every", type=int, default=20)
    a = ap.parse_args()
    outdir = PROJ / "data" / TOPO
    outdir.mkdir(parents=True, exist_ok=True)
    workroot = PROJ / "work" / TOPO
    jobs = [(i, a.seed, a.lvs_every > 0 and i % a.lvs_every == 0, outdir, workroot)
            for i in range(a.start, a.start + a.n)]
    t0, nok = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker, maxtasksperchild=20) as pool:
        for k, (idx, status, dt) in enumerate(pool.imap_unordered(one, jobs), 1):
            nok += status.startswith("ok")
            print(f"[{k}/{len(jobs)}] #{idx} {status} ({dt:.0f}s)  ok={nok}  elapsed={time.time()-t0:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
