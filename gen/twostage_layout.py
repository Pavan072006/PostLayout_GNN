"""Topology 2: two-stage Miller OTA layout in GF180 with gLayout"""
import math
import os

os.environ.setdefault("PDK_ROOT", "/foss/pdks")
os.environ["GLAYOUT_NO_PIN_LABELS"] = "1"

from gdsfactory.component import Component  # noqa: E402
from gdsfactory.components import rectangle  # noqa: E402
from glayout import gf180, via_array  # noqa: E402
from glayout.cells.composite.diffpair_cmirror_bias.diff_pair_cmirrorbias import diff_pair_ibias  # noqa: E402
from glayout.cells.elementary.current_mirror.current_mirror import current_mirror  # noqa: E402
from glayout.primitives.fet import nmos, pmos  # noqa: E402
from glayout.routing.straight_route import straight_route  # noqa: E402
from glayout.util.comp_utils import align_comp_to_port, prec_ref_center  # noqa: E402

GRID, WF_MAX, WF_MIN = 0.01, 8.0, 0.5
TOPO = "ts2"


def snap(x):
    return round(x / GRID) * GRID


def to_layout_params(sz):
    W1, W3, W5 = sz["W1"] * 1e6, sz["W3"] * 1e6, sz["W5"] * 1e6
    L1, L3, L5 = (round(sz[k] * 1e6, 2) for k in ("L1", "L3", "L5"))
    f1 = max(2, math.ceil(W1 / (2 * WF_MAX)))
    w1 = max(WF_MIN, snap(W1 / (2 * f1)))
    c3 = max(1, math.ceil(W3 / WF_MAX))
    w3 = max(WF_MIN, snap(W3 / c3))
    c5 = max(1, math.ceil(W5 / WF_MAX))
    w5 = max(WF_MIN, snap(W5 / c5))
    W7, W8, L7 = sz["W7"] * 1e6, sz["W8"] * 1e6, round(sz["L7"] * 1e6, 2)
    f7 = max(1, math.ceil(W7 / WF_MAX))
    w7 = max(WF_MIN, snap(W7 / f7))
    f8 = max(1, math.ceil(W8 / WF_MAX))
    w8 = max(WF_MIN, snap(W8 / f8))
    p = dict(w1=w1, f1=f1, L1=L1, w3=w3, c3=c3, L3=L3, w5=w5, c5=c5, L5=L5,
             w7=w7, f7=f7, L7=L7, w8=w8, f8=f8)
    p["W1_drawn"], p["W3_drawn"], p["W5_drawn"] = 2 * f1 * w1, c3 * w3, c5 * w5
    p["W7_drawn"], p["W8_drawn"] = f7 * w7, f8 * w8
    return p


def add_pin(top, port, text, glayer):
    orient_to_align = {90: ("c", "b"), 270: ("c", "t"), 0: ("l", "c"), 180: ("r", "c")}
    lab = rectangle(layer=gf180.get_glayer(f"{glayer}_pin"), size=(0.27, 0.27), centered=True).copy()
    lab.add_label(text=text, layer=gf180.get_glayer(f"{glayer}_label"))
    top.add(align_comp_to_port(lab, port, alignment=orient_to_align[round(port.orientation) % 360]))


def rect(top, glayer, x0, y0, x1, y1):
    xl, xr, yb, yt = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    r = top << rectangle(size=(snap(xr - xl), snap(yt - yb)), layer=gf180.get_glayer(glayer))
    r.move((xl - r.xmin, yb - r.ymin))
    return r


def vias(top, lo, hi, x, y, w=0.5, h=0.5):
    v = top << via_array(gf180, lo, hi, size=(w, h), lay_bottom=False)
    v.move((x - v.center[0], y - v.center[1]))
    return v


def build_stage1(top, p):
    dp = diff_pair_ibias(gf180, half_diffpair_params=(p["w1"], p["L1"], p["f1"]),
                         diffpair_bias=(p["w5"], p["L5"], p["c5"]))
    dp_ref = prec_ref_center(dp)
    top.add(dp_ref)
    pm = current_mirror(gf180, numcols=p["c3"], device="pfet", width=p["w3"], length=p["L3"],
                        with_dummy=True, with_tie=True)
    pm_ref = prec_ref_center(pm)
    top.add(pm_ref)
    sep = 4 * gf180.util_max_metal_seperation()
    pm_ref.movey(dp_ref.ymax - pm_ref.ymin + sep)
    P = lambda r, n: r.ports[n]
    m4 = gf180.get_glayer("met4")
    wr = 0.8
    for a, b, side in (("tl_multiplier_0_drain_N", "fet_A_drain_N", -1),
                       ("tr_multiplier_0_drain_N", "fet_B_drain_N", +1)):
        src, bus = P(dp_ref, a), P(pm_ref, b)
        bus_y = bus.center[1] - 0.25
        bx0, bx1 = bus.center[0] - bus.width / 2, bus.center[0] + bus.width / 2
        x = min(max(src.center[0], bx0 + wr), bx1 - wr)
        if abs(x) < wr:
            x = side * wr
        y0 = src.center[1] - 0.25
        v = top << via_array(gf180, "met2", "met4", size=(src.width - 0.4, 0.5), lay_bottom=False)
        v.move((src.center[0] - v.center[0], y0 - v.center[1]))
        xl, xr = min(src.center[0] - src.width / 2, x - wr / 2), max(src.center[0] + src.width / 2, x + wr / 2)
        j = top << rectangle(size=(xr - xl, 0.5), layer=m4)
        j.move((xl - j.xmin, y0 - j.center[1]))
        r = top << rectangle(size=(wr, bus_y - y0), layer=m4)
        r.move((x - r.center[0], y0 - r.ymin))
        vt = top << via_array(gf180, "met2", "met4", size=(wr, 0.5), lay_bottom=False)
        vt.move((x - vt.center[0], bus_y - vt.center[1]))
    top << straight_route(gf180, P(pm_ref, "purposegndportscon_N"), P(pm_ref, "welltie_N_top_met_S"))
    return dp_ref, pm_ref


def build(p):
    gf180.activate()
    top = Component("ts2_build")
    dp_ref, pm_ref = build_stage1(top, p)
    P = lambda r, n: r.ports[n]

    m7 = pmos(gf180, width=p["w7"], fingers=p["f7"], length=p["L7"], with_tie=True,
              with_substrate_tap=False, with_dummy=True)
    m8 = nmos(gf180, width=p["w8"], fingers=p["f8"], length=p["L5"], with_tie=True,
              with_dnwell=False, with_substrate_tap=False, with_dummy=True)
    m7r = prec_ref_center(m7)
    m7r.rotate(180)
    top.add(m7r)
    m8r = prec_ref_center(m8)
    top.add(m8r)

    gap = 3.0
    xcol = max(pm_ref.xmax, dp_ref.xmax) + gap / 2
    x_left = max(pm_ref.xmax, dp_ref.xmax) + gap
    out1 = P(pm_ref, "fet_B_drain_E")
    m7r.movex(x_left - m7r.xmin)
    g7 = P(m7r, "multiplier_0_gate_E")
    m7r.movey(out1.center[1] - g7.center[1])
    if m7r.ymin < dp_ref.ymax + 3:
        m7r.movey(dp_ref.ymax + 3 - m7r.ymin)
    m8r.movex(x_left - m8r.xmin)
    m8r.movey(min(m7r.ymin - 3, dp_ref.ymax) - m8r.ymax)

    g7w = min((P(m7r, "multiplier_0_gate_E"), P(m7r, "multiplier_0_gate_W")), key=lambda q: q.center[0])
    y1, yg = out1.center[1], g7w.center[1]
    rect(top, "met2", out1.center[0], y1 - 0.25, xcol + 0.25, y1 + 0.25)
    vias(top, "met2", "met3", xcol, y1)
    rect(top, "met3", xcol - 0.25, y1, xcol + 0.25, yg)
    vias(top, "met2", "met3", xcol, yg)
    rect(top, "met2", xcol - 0.25, yg - 0.25, g7w.center[0] + 0.3, yg + 0.25)

    pe = P(pm_ref, "welltie_E_top_met_E")
    m7w = min((P(m7r, "tie_W_top_met_W"), P(m7r, "tie_E_top_met_E")), key=lambda q: q.center[0])
    ylo = max(pe.center[1] - pe.width / 2, m7w.center[1] - m7w.width / 2) + 0.6
    yhi = min(pe.center[1] + pe.width / 2, m7w.center[1] + m7w.width / 2) - 0.6
    if yhi - ylo < 0.5:
        yb = m7w.center[1]
        ya = min(max(yb, pe.center[1] - pe.width / 2 + 0.6), pe.center[1] + pe.width / 2 - 0.6)
        xm = xcol + 0.9
        rect(top, "met1", pe.center[0] - 0.2, ya - 0.3, xm + 0.3, ya + 0.3)
        rect(top, "met1", xm - 0.3, ya, xm + 0.3, yb)
        rect(top, "met1", xm - 0.3, yb - 0.3, m7w.center[0] + 0.2, yb + 0.3)
    else:
        yb = (ylo + yhi) / 2
        rect(top, "met1", pe.center[0] - 0.2, yb - 0.3, m7w.center[0] + 0.2, yb + 0.3)

    for ref in (m7r, m8r):
        s = max((P(ref, "multiplier_0_source_E"), P(ref, "multiplier_0_source_W")), key=lambda q: q.center[0])
        te = max((P(ref, "tie_E_top_met_E"), P(ref, "tie_W_top_met_W")), key=lambda q: q.center[0])
        xr = te.center[0] - 0.23
        rect(top, "met2", s.center[0] - 0.1, s.center[1] - 0.25, xr + 0.25, s.center[1] + 0.25)
        vias(top, "met1", "met2", xr, s.center[1])

    d8 = P(m8r, "multiplier_0_drain_N")
    d7 = P(m7r, "multiplier_0_drain_N")
    lo = max(d8.center[0] - d8.width / 2, d7.center[0] - d7.width / 2)
    hi = min(d8.center[0] + d8.width / 2, d7.center[0] + d7.width / 2)
    xo = lo + min(1.0, (hi - lo) / 2)
    yd8 = d8.center[1] - 0.25 if round(d8.orientation) % 360 == 90 else d8.center[1] + 0.25
    yd7 = d7.center[1] + 0.25 if round(d7.orientation) % 360 == 270 else d7.center[1] - 0.25
    vias(top, "met2", "met3", xo, yd8)
    vias(top, "met2", "met3", xo, yd7)
    rect(top, "met3", xo - 0.3, yd8, xo + 0.3, yd7)

    ib = P(dp_ref, "ibias_A_drain_E")
    g8 = max((P(m8r, "multiplier_0_gate_E"), P(m8r, "multiplier_0_gate_W")), key=lambda q: q.center[0])
    xg = g8.center[0] - 0.3
    vias(top, "met3", "met5", ib.center[0] - 0.3, ib.center[1], 0.6, 0.5)
    rect(top, "met5", ib.center[0] - 0.6, ib.center[1] - 0.4, xg + 0.4, ib.center[1] + 0.4)
    rect(top, "met5", xg - 0.4, ib.center[1], xg + 0.4, g8.center[1])
    vias(top, "met2", "met5", xg, g8.center[1], 0.6, 0.5)

    add_pin(top, P(dp_ref, "br_multiplier_0_gate_S"), "VP", "met2")
    add_pin(top, P(dp_ref, "bl_multiplier_0_gate_S"), "VN", "met2")
    add_pin(top, P(pm_ref, "fet_B_drain_E"), "OUT1", "met2")
    d7e = max((P(m7r, "multiplier_0_drain_E"), P(m7r, "multiplier_0_drain_W")), key=lambda q: q.center[0])
    add_pin(top, d7e, "OUT", "met2")
    add_pin(top, P(dp_ref, "ibias_A_drain_E"), "IBIAS", "met3")
    add_pin(top, P(dp_ref, "ibias_purposegndport"), "VSS", "met4")
    add_pin(top, P(dp_ref, "tap_N_top_met_S"), "B", "met1")
    add_pin(top, P(pm_ref, "welltie_N_top_met_S"), "VDD", "met2")
    top = top.flatten()
    stray = [lb for lb in top.labels if lb.text in ("VDD1", "VDD2")]
    if stray:
        top._cell.remove(*stray)
    top.name = TOPO
    return top


def reference_netlist(p):
    n, pf = "nfet_03v3", "pfet_03v3"
    L1, L3, L5, L7 = p["L1"], p["L3"], p["L5"], p["L7"]
    W1, W3, W5 = p["W1_drawn"], p["W3_drawn"], p["W5_drawn"]
    W7, W8 = p["W7_drawn"], p["W8_drawn"]
    return f"""* two-stage Miller OTA reference netlist (from sizing); Cc, CL are testbench parts
.subckt {TOPO} VP VN OUT1 OUT IBIAS VDD VSS B
XM1 X VP TAIL B {n} w={W1:.3f}u l={L1}u
XM2 OUT1 VN TAIL B {n} w={W1:.3f}u l={L1}u
XM3 X X VDD VDD {pf} w={W3:.3f}u l={L3}u
XM4 OUT1 X VDD VDD {pf} w={W3:.3f}u l={L3}u
XM5 TAIL IBIAS VSS B {n} w={W5:.3f}u l={L5}u
XM6 IBIAS IBIAS VSS B {n} w={W5:.3f}u l={L5}u
XM7 OUT OUT1 VDD VDD {pf} w={W7:.3f}u l={L7}u
XM8 OUT IBIAS B B {n} w={W8:.3f}u l={L5}u
XD1 B B B B {n} w={4*p['w1']:.3f}u l={L1}u
XD2 B B B B {n} w={2*p['w5']:.3f}u l={L5}u
XD3 VDD VDD VDD VDD {pf} w={2*p['w3']:.3f}u l={L3}u
XD7 VDD VDD VDD VDD {pf} w={2*p['w7']:.3f}u l={L7}u
XD8 B B B B {n} w={2*p['w8']:.3f}u l={L5}u
.ends {TOPO}
"""
