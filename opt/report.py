#!/usr/bin/env python3
"""Table + plot: does each sizing method still meet spec after real layout + PEX?"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
from optimise_specs import SPECS  # noqa: E402

S = {s["id"]: s for s in SPECS}
D = json.loads((PROJ / "results" / "opt_verified.json").read_text())
LABEL = {"schematic": "schematic only", "guard5": "schematic + 5% guard band", "gnn": "schematic + GNN parasitics"}


def passes(s, m):
    return (m["GBW_Hz"] >= s["GBW_min"] and m["PM_deg"] >= s["PM_min"] and m["A0_dB"] >= s["A0_min"]
            and m["power_W"] <= s["P_max"])


rows = ["# Post-layout-aware sizing (project 1 optimiser idea + project 4 GNN)\n",
        "Each design is optimised for minimum power under the spec, then laid out with gLayout, "
        "extracted with Magic and re-simulated (typical corner).\n",
        "| spec | method | seed | GBW optimiser saw [MHz] | GBW after layout [MHz] | GBW min | PM after layout | power [uW] | LVS | meets spec after layout |",
        "|---|---|---|---|---|---|---|---|---|---|"]
summary = {}
for d in sorted(D, key=lambda d: (d["spec"], list(LABEL).index(d["mode"]), d["seed"])):
    s, pre, post = S[d["spec"]], d["predicted"], d["post"]
    ok = passes(s, post)
    summary.setdefault(d["mode"], []).append((ok, post["power_W"] / s["P_max"], abs(post["GBW_Hz"] / pre["GBW_Hz"] - 1)))
    rows.append(f"| {d['spec']} | {LABEL[d['mode']]} | {d['seed']} | {pre['GBW_Hz']/1e6:.1f} | {post['GBW_Hz']/1e6:.1f} | "
                f"{s['GBW_min']/1e6:g} | {post['PM_deg']:.1f} | {post['power_W']*1e6:.0f} | {'match' if d['lvs'] else 'FAIL'} | "
                f"{'yes' if ok else '**no**'} |")
rows += ["\n| method | designs meeting spec after layout | mean power | mean |predicted - real| GBW |", "|---|---|---|---|"]
for m, v in summary.items():
    rows.append(f"| {LABEL[m]} | {sum(x[0] for x in v)}/{len(v)} | {100*sum(x[1] for x in v)/len(v):.1f} % of budget | "
                f"{100*sum(x[2] for x in v)/len(v):.2f} % |")
(PROJ / "results" / "opt_report.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))

fig, ax = plt.subplots(figsize=(8, 4.5))
mk = {"schematic": "o", "guard5": "s", "gnn": "^"}
for d in D:
    s = S[d["spec"]]
    x = 100 * (d["post"]["GBW_Hz"] / s["GBW_min"] - 1)
    ax.scatter(x, d["post"]["power_W"] * 1e6, marker=mk[d["mode"]], s=50, alpha=0.75,
               color=plt.cm.tab10(list(S).index(d["spec"])),
               label=f"{LABEL[d['mode']]}" if d["spec"] == "P1" and d["seed"] == 0 else None)
ax.axvline(0, color="k", lw=0.8)
ax.text(0.3, ax.get_ylim()[1] * 0.97, "meets GBW spec →", va="top", fontsize=8)
ax.set_xlabel("post-layout GBW margin over spec [%]")
ax.set_ylabel("power [uW]")
ax.set_title("After real layout + PEX (marker = method, colour = spec P1/P2/P3)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PROJ / "results" / "opt_comparison.png", dpi=130)
