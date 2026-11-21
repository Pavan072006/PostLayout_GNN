#!/usr/bin/env python3
import argparse
import glob
import json
import math
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
import tb  # noqa: E402
from netlist_utils import parse_spice, lumped_subckt, device_geometry, finger_plan  # noqa: E402

OUT = PROJ / "data" / "resim"
KEYS = ("A0_dB", "GBW_Hz", "PM_deg", "power_W")


def one(path):
    r = json.loads(pathlib.Path(path).read_text())
    out = OUT / f"{r['id']}.json"
    if out.exists():
        return r["id"], "skip"
    topo = r["topology"]
    ref, ext = parse_spice(r["ref_netlist"]), parse_spice(r["pex_netlist"])
    nfs = finger_plan(ref, r["layout"])
    geo = device_geometry(ref, ext, r["net_map"])
    bad = [k for k, (g, n) in enumerate(zip(geo, nfs)) if g["nf"] != n]
    caps = [tuple(c) for c in r["caps"]]
    work = PROJ / "work" / "resim_ds" / r["id"]
    sims = {}
    for tag, c, g in (("pre", None, None), ("oracle_caps", caps, None), ("oracle_full", caps, geo)):
        s = tb.simulate(topo, lumped_subckt(ref, c, topo, g, nfs), r["sizing"]["Itail"], r["CL"], r.get("Cc"),
                        workdir=work)
        if not s.get("ok") or not math.isfinite(s["GBW_Hz"]):
            return r["id"], f"FAIL {tag}"
        sims[tag] = {k: s[k] for k in KEYS}
    sims["post"] = r["sims"]["post"]
    out.write_text(json.dumps(dict(id=r["id"], nfs=nfs, geo=geo, finger_mismatch=bad, sims=sims)))
    return r["id"], "ok" + (f" (finger count mismatch on devices {bad})" if bad else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(PROJ / "data" / "ota5t" / "*.json")) + glob.glob(str(PROJ / "data" / "ts2" / "*.json")))
    n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for gid, st in ex.map(one, files):
            if st != "skip":
                n += 1
                if st != "ok":
                    print(gid, st, flush=True)
    print(f"resimulated {n} new samples; total cached {len(list(OUT.glob('*.json')))}")


if __name__ == "__main__":
    main()
