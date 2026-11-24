#!/usr/bin/env python3
"""Size a 5T OTA for specs three ways, then compare after real layout + PEX (opt/verify.py)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
AA = PROJ.parent / "analog_agent"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJ / "gen"))
sys.path.insert(0, str(AA))
import tb  # noqa: E402
from netlist_utils import parse_spice, lumped_subckt  # noqa: E402
from lut.lookup import LUT  # noqa: E402
import predictor as PR  # noqa: E402
from optimise_specs import SPECS  # noqa: E402


MODES = {"schematic": (1.0, 0.0, False), "guard5": (1.05, 2.0, False), "gnn": (1.0, 0.0, True)}

LO = np.array([5, 5, 5, math.log(0.28), math.log(0.28), math.log(0.28), math.log10(5e-6)])
HI = np.array([22, 22, 22, math.log(2.0), math.log(2.0), math.log(2.0), math.log10(300e-6)])
_LUT = None


def decode(u):
    global _LUT
    _LUT = _LUT or LUT()
    x = LO + np.clip(u, 0, 1) * (HI - LO)
    L = [round(math.exp(v), 2) * 1e-6 for v in x[3:6]]
    I = 10 ** x[6]
    sz = dict(L1=L[0], L3=L[1], L5=L[2], Itail=I,
              W1=_LUT.width("nmos", L[0], x[0], I / 2),
              W3=_LUT.width("pmos", L[1], x[1], I / 2),
              W5=_LUT.width("nmos", L[2], x[2], I))
    return sz


def simulate(sz, CL, use_gnn):
    p = PR.to_layout_params(sz)
    ref_txt = PR.reference_netlist(p)
    caps = geo = None
    if use_gnn:
        q = PR.predict(sz, CL)
        caps, geo = q["caps"], q["geo"]
    ref = parse_spice(ref_txt)
    return tb.simulate("ota5t", lumped_subckt(ref, caps, "ota5t", geo, PR.finger_plan(p)), sz["Itail"], CL,
                       workdir=PROJ / "work" / "opt" / f"{os.getpid()}")


def violation(s, m, gbw_k, pm_add):
    if not m.get("ok") or not all(math.isfinite(m.get(k, float("nan"))) for k in ("GBW_Hz", "A0_dB", "PM_deg")):
        return 10.0
    v = max(0.0, 1 - m["GBW_Hz"] / (s["GBW_min"] * gbw_k))
    v += max(0.0, (s["A0_min"] - m["A0_dB"]) / 10)
    v += max(0.0, (s["PM_min"] + pm_add - m["PM_deg"]) / 30)
    v += max(0.0, m["power_W"] / s["P_max"] - 1)
    return v


def run(args):
    s, mode, seed, budget = args
    gbw_k, pm_add, use_gnn = MODES[mode]
    t0 = time.time()
    best = {"score": float("inf")}
    n = {"evals": 0}

    def f(u):
        n["evals"] += 1
        try:
            sz = decode(u)
        except Exception:
            return 1e3
        if not all(1e-6 <= sz[k] <= 120e-6 for k in ("W1", "W3", "W5")):
            return 1e3
        m = simulate(sz, s["CL"], use_gnn)
        v = violation(s, m, gbw_k, pm_add)
        score = 100 * v + (m["power_W"] / s["P_max"] if m.get("ok") else 10)
        if score < best["score"]:
            best.update(score=score, v=v, sz=sz, m=m)
        return score

    rng = np.random.default_rng(seed)
    starts = [rng.uniform(0.2, 0.8, 7) for _ in range(3)]
    for u0 in starts:
        simplex = [u0] + [np.clip(u0 + 0.15 * np.eye(7)[i], 0, 1) for i in range(7)]
        minimize(f, u0, method="Nelder-Mead", bounds=[(0, 1)] * 7,
                 options={"initial_simplex": np.array(simplex), "maxfev": budget // len(starts)})
    m = best["m"]
    return dict(spec=s["id"], mode=mode, seed=seed, evals=n["evals"], seconds=round(time.time() - t0, 1),
                sizing=best["sz"], predicted_pass=best["v"] == 0,
                predicted={k: m[k] for k in ("A0_dB", "GBW_Hz", "PM_deg", "power_W")})


def main():
    jobs = [(s, mode, seed, 450) for s in SPECS for mode in MODES for seed in range(2)]
    with ProcessPoolExecutor(6) as ex:
        out = list(ex.map(run, jobs))
    for r in out:
        p = r["predicted"]
        print(f"{r['spec']} {r['mode']:9} seed{r['seed']}  pass(as optimised)={r['predicted_pass']}  "
              f"GBW {p['GBW_Hz']/1e6:6.1f} MHz  PM {p['PM_deg']:5.1f}  A0 {p['A0_dB']:5.1f}  "
              f"P {p['power_W']*1e6:6.1f} uW  ({r['evals']} evals, {r['seconds']} s)")
    (PROJ / "results" / "opt_designs.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
