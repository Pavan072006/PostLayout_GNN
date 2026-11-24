#!/usr/bin/env python3
import contextlib
import io
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
AA = PROJ.parent / "analog_agent"
sys.path.insert(0, str(PROJ / "gen"))
from netlist_utils import pex_to_ngspice  # noqa: E402
from pex_tools import extract_pex, run_lvs  # noqa: E402
import tb  # noqa: E402

_G = {}


def init():
    os.environ.setdefault("PDK_ROOT", "/foss/pdks")
    sys.path.insert(0, str(AA / "layout"))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import ota5t_layout as L  # noqa: E402
    _G["L"] = L


def one(args):
    k, d, CL = args
    L = _G["L"]
    work = PROJ / "work" / "verify" / f"d{k}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    p = L.to_layout_params(d["sizing"])
    cell = f"ota5t_v{k}"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        top = L.build(p)
        top.name = cell
        gds = work / "ota5t.gds"
        top.write_gds(str(gds))
    (work / "ref.spice").write_text(L.reference_netlist(p))
    lvs = run_lvs(gds, work / "ref.spice", work, cell, "ota5t")
    pex = extract_pex(gds, work, cell).read_text()
    m = tb.simulate("ota5t", pex_to_ngspice(pex, "ota5t"), d["sizing"]["Itail"], CL, workdir=work)
    shutil.rmtree(work, ignore_errors=True)
    return k, lvs, {q: m.get(q) for q in ("A0_dB", "GBW_Hz", "PM_deg", "power_W")}


def main():
    sys.path.insert(0, str(HERE))
    designs = json.loads((PROJ / "results" / "opt_designs.json").read_text())
    from optimise_specs import SPECS
    cl = {s["id"]: s["CL"] for s in SPECS}
    jobs = [(k, d, cl[d["spec"]]) for k, d in enumerate(designs)]
    with mp.get_context("spawn").Pool(6, initializer=init) as pool:
        for k, lvs, post in pool.imap_unordered(one, jobs):
            designs[k]["lvs"] = lvs
            designs[k]["post"] = post
            print(f"{designs[k]['spec']} {designs[k]['mode']:9} seed{designs[k]['seed']} LVS={lvs} "
                  f"post GBW {post['GBW_Hz']/1e6:.1f} MHz PM {post['PM_deg']:.1f}", flush=True)
    (PROJ / "results" / "opt_verified.json").write_text(json.dumps(designs, indent=1))


if __name__ == "__main__":
    main()
