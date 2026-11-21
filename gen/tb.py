"""ngspice open-loop AC testbench for both topologies (typical corner by default)"""
import os
import pathlib
import re
import subprocess
import uuid

import numpy as np

_INC = None


def _model_includes():
    root = pathlib.Path(os.environ.get("PDK_ROOT", "/foss/pdks"))
    d = root / "gf180mcuD" / "libs.tech" / "ngspice"
    lib = next(d / n for n in ("sm141064.ngspice", "sm141064.spice") if (d / n).exists())
    design = next((d / n for n in ("design.ngspice", "design.spice") if (d / n).exists()), None)
    inc = f'.include "{design}"\n' if design else ""
    return inc + f'.lib "{lib}" {{corner}}'


NODEMAP = {"VP": "vp", "VN": "vn", "OUT": "out", "OUT1": "out1", "IBIAS": "nb",
           "VDD": "vdd", "VSS": "0", "B": "0"}


def netlist(topo, dut, Itail, CL, Cc=None, corner="typical", VDD=3.3, VCM=1.65):
    global _INC
    if _INC is None:
        _INC = _model_includes()
    sub, pins = dut
    xline = "Xdut " + " ".join(NODEMAP.get(q, f"int_{q.lower()}") for q in pins) + f" {topo}"
    if topo == "ota5t":
        drive = f"Vp vp 0 DC {VCM} AC 1\nLfb out vn 1T\nCfb vn 0 1T"
    else:
        drive = f"Vn vn 0 DC {VCM} AC 1\nLfb out vp 1T\nCfb vp 0 1T\nCc out1 out {Cc}"
    return f"""* {topo} testbench
{_INC.format(corner=corner)}
Vdd vdd 0 {VDD}
Iref vdd nb {Itail}
{drive}
{sub}
{xline}
CL out 0 {CL}
.control
op
print v(out) vdd#branch
ac dec 40 1 10G
let mag = vdb(out)
let ph = 180/pi*cph(v(out))
set wr_singlescale
set wr_vecnames
set numdgt=7
wrdata {{acfile}} mag ph
.endc
.end
"""


def simulate(topo, dut, Itail, CL, Cc=None, corner="typical", workdir=".", keep=False):
    work = pathlib.Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    acfile = f"ac_{tag}.txt"
    cir = work / f"tb_{tag}.cir"
    cir.write_text(netlist(topo, dut, Itail, CL, Cc, corner).replace("{acfile}", acfile))
    (work / ".spiceinit").write_text("set ngbehavior=hsa\nset ng_nomodcheck\n")
    res = subprocess.run(["ngspice", "-b", cir.name], cwd=work, capture_output=True, text=True, timeout=120)
    log = res.stdout + res.stderr
    out = {"ok": False}
    acpath = work / acfile
    if not acpath.exists():
        out["error"] = "ngspice failed: " + " | ".join(l for l in log.splitlines() if "rror" in l)[:400]
        return out
    d = np.loadtxt(acpath, skiprows=1)
    f, mag, ph = d[:, 0], d[:, 1], d[:, 2]
    out["A0_dB"] = float(mag[0])
    if mag[0] > 0 and (mag < 0).any():
        i = int(np.argmax(mag < 0))
        x0, x1 = np.log10(f[i - 1]), np.log10(f[i])
        t = mag[i - 1] / (mag[i - 1] - mag[i])
        out["GBW_Hz"] = float(10 ** (x0 + t * (x1 - x0)))
        ph_u = ph[i - 1] + t * (ph[i] - ph[i - 1])
        out["PM_deg"] = float(180 + (ph_u - ph[0]))
    else:
        out["GBW_Hz"] = out["PM_deg"] = float("nan")
    m = re.search(r"^vdd#branch\s*=\s*([-+0-9.eE]+)", log, re.MULTILINE)
    out["power_W"] = 3.3 * abs(float(m.group(1))) if m else float("nan")
    m = re.search(r"^v\(out\)\s*=\s*([-+0-9.eE]+)", log, re.MULTILINE)
    out["Vout_dc"] = float(m.group(1)) if m else float("nan")
    out["ok"] = True
    if not keep:
        for q in (cir, acpath):
            q.unlink(missing_ok=True)
    return out
