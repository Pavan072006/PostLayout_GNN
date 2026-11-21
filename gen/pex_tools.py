"""Magic PEX / LVS helpers with the cell name as a parameter"""
import os
import pathlib
import shutil
import subprocess

PDK_TECH = pathlib.Path(os.environ.get("PDK_ROOT", "/foss/pdks")) / "gf180mcuD" / "libs.tech"


def magic(script, cwd, log):
    m = subprocess.run(["magic", "-rcfile", str(PDK_TECH / "magic" / "gf180mcuD.magicrc"),
                        "-noconsole", "-dnull"], input=script, cwd=cwd,
                       capture_output=True, text=True, timeout=900)
    (cwd / log).write_text(m.stdout + m.stderr)


def extract_pex(gds, work, cell):
    pex = work / "pex.spice"
    magic(f"""gds read {gds}
load {cell}
select top cell
extract do local
extract all
ext2spice format ngspice
ext2spice cthresh 0
ext2spice subcircuit top on
ext2spice scale off
ext2spice -o {pex}
quit -noprompt
""", work, "pex_magic.log")
    if not pex.exists():
        raise RuntimeError("magic PEX failed")
    return pex


def run_lvs(gds, ref_path, work, cell, ref_cell=None):
    ext = work / "lvs_ext.spice"
    magic(f"""gds read {gds}
load {cell}
select top cell
extract do local
extract all
ext2spice lvs
ext2spice -o {ext}
quit -noprompt
""", work, "lvs_magic.log")
    rpt = work / "lvs.rpt"
    netgen = shutil.which("netgen") or shutil.which("netgen-lvs")
    subprocess.run([netgen, "-batch", "lvs", f"{ext} {cell}", f"{ref_path} {ref_cell or cell}",
                    str(PDK_TECH / "netgen" / "gf180mcuD_setup.tcl"), str(rpt)],
                   cwd=work, capture_output=True, text=True, timeout=900)
    txt = rpt.read_text() if rpt.exists() else ""
    return "match uniquely" in txt and "do not match" not in txt.split("Final result")[-1]
