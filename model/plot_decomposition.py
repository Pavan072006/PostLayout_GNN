#!/usr/bin/env python3
"""Where does the post-layout shift come from? (all samples, no ML)"""
import glob
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJ = pathlib.Path(__file__).resolve().parent.parent
rows = []
for f in sorted(glob.glob(str(PROJ / "data" / "resim" / "*.json"))):
    rs = json.loads(pathlib.Path(f).read_text())
    topo = rs["id"].split("_")[0]
    raw = json.loads((PROJ / "data" / topo / f"{rs['id']}.json").read_text())["sims"]
    S = rs["sims"]
    sh = lambda k, ref=S["pre"]: 100 * (S[k]["GBW_Hz"] / ref["GBW_Hz"] - 1)
    pm = lambda k: S[k]["PM_deg"] - S["pre"]["PM_deg"]
    rows.append(dict(id=rs["id"], topo=topo, post=sh("post"), caps=sh("oracle_caps"), full=sh("oracle_full"),
                     naive=100 * (raw["post"]["GBW_Hz"] / raw["pre"]["GBW_Hz"] - 1),
                     pm_post=pm("post"), pm_caps=pm("oracle_caps"), pm_full=pm("oracle_full"),
                     dA0_naive=raw["post"]["A0_dB"] - raw["pre"]["A0_dB"],
                     dA0=S["post"]["A0_dB"] - S["pre"]["A0_dB"]))

a = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("id", "topo")}
mae = lambda x, y: float(np.mean(np.abs(x - y)))
summary = dict(n=len(rows),
               dGBW_post_mean=float(a["post"].mean()), dGBW_post_std=float(a["post"].std()),
               MAE_caps_only=mae(a["caps"], a["post"]), MAE_caps_geo=mae(a["full"], a["post"]),
               PM_MAE_caps_only=mae(a["pm_caps"], a["pm_post"]), PM_MAE_caps_geo=mae(a["pm_full"], a["pm_post"]),
               dA0_post_vs_naive_pre_mean_abs=float(np.abs(a["dA0_naive"]).mean()),
               dA0_post_vs_finger_pre_mean_abs=float(np.abs(a["dA0"]).mean()))
(PROJ / "results" / "shift_decomposition.json").write_text(json.dumps(summary, indent=1))

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
ax = axs[0]
ax.scatter(a["post"], a["caps"], s=10, alpha=0.6, label=f"wiring caps only (MAE {summary['MAE_caps_only']:.2f} %-pts)")
ax.scatter(a["post"], a["full"], s=10, alpha=0.6, label=f"caps + junction geometry (MAE {summary['MAE_caps_geo']:.3f})")
lo, hi = min(a["post"].min(), a["caps"].min()), max(a["post"].max(), a["caps"].max())
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax.set_xlabel("true post-layout GBW shift [%]")
ax.set_ylabel("schematic + true parasitics, GBW shift [%]")
ax.set_title("What explains the post-layout GBW shift?")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[1]
ax.hist(a["dA0_naive"], bins=40, alpha=0.6, label="vs single-finger schematic")
ax.hist(a["dA0"], bins=40, alpha=0.6, label="vs finger-accurate schematic")
ax.set_xlabel("post - pre open-loop gain A0 [dB]")
ax.set_ylabel("samples")
ax.set_title("Fake 'layout' gain loss from a wrong finger plan")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PROJ / "results" / "shift_decomposition.png", dpi=130)
print(json.dumps(summary, indent=1))
