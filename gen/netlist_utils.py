import collections
import hashlib
import math
import re

SI = {"a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3}
SUPPLY = ("VDD", "VSS", "B")


def val(tok):
    tok = tok.split("=")[-1].strip().lower()
    if tok and tok[-1] in SI:
        return float(tok[:-1]) * SI[tok[-1]]
    return float(tok)


def parse_spice(text):
    pins, devs, caps = [], [], []
    for ln in text.splitlines():
        ln = ln.split(";")[0].strip()
        if not ln or ln.startswith("*"):
            continue
        t = ln.split()
        if t[0].lower() == ".subckt":
            pins = t[2:]
        elif t[0][0] in "XxMm" and len(t) >= 6 and re.match(r"[np]fet", t[5]):
            kv = {k.lower(): val(v) for k, v in (x.split("=") for x in t[6:] if "=" in x)}
            devs.append(dict(name=t[0], d=t[1], g=t[2], s=t[3], b=t[4], model=t[5], **kv))
        elif t[0][0] in "Cc" and len(t) >= 4:
            c = val(t[3])
            if c > 0:
                caps.append((t[1], t[2], c))
    return dict(pins=pins, devices=devs, caps=caps)


def is_dummy(d):
    return d["d"] == d["g"] == d["s"]


def _merge_parallel(devs):
    out = collections.OrderedDict()
    for d in devs:
        if is_dummy(d):
            continue
        k = (d["model"], frozenset((d["d"], d["s"])), d["g"], d["b"])
        if k in out:
            out[k]["w"] += d.get("w", 0)
            out[k]["nf"] += 1
        else:
            out[k] = dict(d, nf=1)
    return list(out.values())


def _h(*x):
    return hashlib.md5(repr(x).encode()).hexdigest()[:12]


def _colors(devs, pins, rounds=5):
    nets = sorted({d[t] for d in devs for t in "dgsb"})
    nc = {n: (n if n in pins else "?") for n in nets}
    for _ in range(rounds):
        dc = []
        for d in devs:
            ds = tuple(sorted((nc[d["d"]], nc[d["s"]])))
            dc.append(_h(d["model"], ds, nc[d["g"]], nc[d["b"]]))
        new = {}
        for n in nets:
            ms = []
            for d, c in zip(devs, dc):
                for t in "dgsb":
                    if d[t] == n:
                        ms.append(("ds" if t in "ds" else t, c))
            new[n] = n if n in pins else _h(sorted(ms))
        nc = new
    return nc


def match_nets(ref, ext):
    pins = set(ref["pins"])
    rc = _colors(_merge_parallel(ref["devices"]), pins)
    ec = _colors(_merge_parallel(ext["devices"]), pins)
    by_col = collections.defaultdict(list)
    for n, c in rc.items():
        by_col[c].append(n)
    m = {}
    for n, c in ec.items():
        cand = by_col.get(c, [])
        if len(cand) != 1:
            raise ValueError(f"net {n}: ambiguous/unmatched ({cand})")
        m[n] = cand[0]
    for p in pins:
        m.setdefault(p, p)
    return m


def net_caps(caps, mapping):
    tot, gnd, cpl = collections.Counter(), collections.Counter(), collections.Counter()
    for a, b, c in caps:
        a, b = mapping.get(a, a), mapping.get(b, b)
        if a == b:
            continue
        tot[a] += c
        tot[b] += c
        if "B" in (a, b):
            gnd[a if b == "B" else b] += c
        else:
            cpl["|".join(sorted((a, b)))] += c
    return dict(total=dict(tot), ground=dict(gnd), coupling=dict(cpl))


DIFF_EXT = 0.48e-6
MAX_FINGER_W = 20e-6


def _mos(name, d, g, s, b, model, W, L, geo=None, nf=None):
    nf = nf or max(1, math.ceil(W / MAX_FINGER_W))
    wf = W / nf
    if geo is None:
        ad = as_ = wf * DIFF_EXT
        pd = ps = 2 * (wf + DIFF_EXT)
    else:
        ad, as_, pd, ps = (geo[k] / nf for k in ("ad", "as", "pd", "ps"))
    return (f"M{name} {d} {g} {s} {b} {model} W={wf*1e6:.4f}u L={L*1e6:.4f}u m={nf} "
            f"ad={ad*1e12:.4f}p as={as_*1e12:.4f}p pd={pd*1e6:.4f}u ps={ps*1e6:.4f}u")


FINGER_RULE = {"XM1": lambda p: 2 * p["f1"], "XM2": lambda p: 2 * p["f1"],
               "XM3": lambda p: p["c3"], "XM4": lambda p: p["c3"],
               "XM5": lambda p: p["c5"], "XM6": lambda p: p["c5"],
               "XM7": lambda p: p["f7"], "XM8": lambda p: p["f8"]}


def finger_plan(ref, p):
    return [FINGER_RULE[d["name"]](p) for d in ref["devices"] if not is_dummy(d)]


def schematic_geometry(d, nf=None):
    W = d["w"]
    nf = nf or max(1, math.ceil(W / MAX_FINGER_W))
    a = W * DIFF_EXT
    p = 2 * (W + nf * DIFF_EXT)
    return {"ad": a, "as": a, "pd": p, "ps": p}


def device_geometry(ref, ext, mapping):
    rdev = [d for d in ref["devices"] if not is_dummy(d)]
    geo = [{"ad": 0.0, "as": 0.0, "pd": 0.0, "ps": 0.0, "nf": 0} for _ in rdev]
    for f in ext["devices"]:
        fd, fg, fs, fb = (mapping.get(f[t], f[t]) for t in "dgsb")
        for k, d in enumerate(rdev):
            if d["model"] != f["model"] or d["g"] != fg or {d["d"], d["s"]} != {fd, fs}:
                continue
            swap = fd != d["d"]
            geo[k]["ad"] += f.get("as" if swap else "ad", 0)
            geo[k]["as"] += f.get("ad" if swap else "as", 0)
            geo[k]["pd"] += f.get("ps" if swap else "pd", 0)
            geo[k]["ps"] += f.get("pd" if swap else "ps", 0)
            geo[k]["nf"] += 1
            break
    return geo


def lumped_subckt(ref, caps=None, name="ota5t", geo=None, nfs=None):
    lines = [f".subckt {name} " + " ".join(ref["pins"])]
    for i, d in enumerate(x for x in ref["devices"] if not is_dummy(x)):
        lines.append(_mos(f"{i}", d["d"], d["g"], d["s"], d["b"], d["model"], d["w"], d["l"],
                          None if geo is None else geo[i], None if nfs is None else nfs[i]))
    for i, (a, b, c) in enumerate(caps or []):
        if c > 0 and a != b:
            lines.append(f"Cp{i} {a} {b} {c:.6e}")
    lines.append(".ends")
    return "\n".join(lines), list(ref["pins"])


def pex_to_ngspice(text, name=None):
    out, pins = [], []
    for ln in text.splitlines():
        if ln.lower().startswith(".subckt"):
            pins = ln.split()[2:]
            if name:
                ln = f".subckt {name} " + " ".join(pins)
        if re.match(r"^X\S*\s", ln) and re.search(r"\b[np]fet_03v3\b", ln):
            ln = "M" + ln[1:]
        out.append(ln)
    return "\n".join(l for l in out if l.strip().lower() not in (".end",)), pins
