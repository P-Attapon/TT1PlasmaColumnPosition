"""Command-line runner for the Biot-Savart inversion.

    python -m methods_script.biot_savart.cli 1641
    python -m methods_script.biot_savart.cli 2400 --probes 12 3 6 9
    python -m methods_script.biot_savart.cli 1643 --no-fit-ip

Writes `bs_results/<shot>_bs.npz` and, with --csv, a plain-text table.
`biot_savart_main.py` at the repository root is the equivalent entry point with
parameters edited in the file instead of passed as flags.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from . import adapter, invert

OUT_DIR = "bs_results"


def run(shot, probes=None, weights_source="auto", ip_min=None,
        search="grid", phi_xy=None, forward="internal", fit_ip=True,
        data_dir="data", quiet=False):
    geom = adapter.load_geometry(strict=True)
    fwd = adapter.load_forward(forward, geom)
    fwd_many = adapter.load_forward_many(geom)
    t, ip, B = adapter.load_shot(shot, data_dir=data_dir)
    w = adapter.load_weights(shot, weights_source)

    if not quiet:
        print(f"shot {shot}: {t.size} samples, {B.shape[1]} probes")
        print(f"geometry : {geom.source}")
        print(f"forward  : {forward}    fit_ip: {fit_ip}")
        print(f"probes   : {'all' if probes is None else probes}")
        print(f"weights  : {'uniform' if w is None else weights_source}")

    def prog(i, n):
        if not quiet:
            sys.stderr.write(f"\r  {i}/{n} samples")
            sys.stderr.flush()

    t0 = time.time()
    res = invert.invert_shot(fwd, fwd_many, t, ip, B, probes=probes, weights=w,
                             ip_min=ip_min, search=search, phi_xy=phi_xy,
                             fit_ip=fit_ip, I_ref=geom.I_ref,
                             progress=None if quiet else prog)
    dt = time.time() - t0
    if not quiet:
        sys.stderr.write("\r" + " " * 40 + "\r")

    res["shot"] = shot
    res["probes"] = np.array([] if probes is None else probes)
    res["seconds"] = dt
    res["I_ref"] = geom.I_ref
    return res


def summarise(res):
    ok = np.isfinite(res["dR_m"]) & ~res["at_wall"]
    n = int(ok.sum())
    T = res["dR_m"].size
    print(f"\nsolved {n}/{T} samples in {res['seconds']:.1f} s "
          f"({1e3 * res['seconds'] / max(T, 1):.1f} ms/sample)")
    if not n:
        print("nothing solved away from the fit bound")
        return
    dR = res["dR_m"][ok] * 1e3
    dZ = res["dZ_m"][ok] * 1e3
    print(f"  dR  {dR.min():8.1f} .. {dR.max():8.1f} mm   median {np.median(dR):8.1f}")
    print(f"  dZ  {dZ.min():8.1f} .. {dZ.max():8.1f} mm   median {np.median(dZ):8.1f}")
    rn = res["resid_norm"][ok]
    print(f"  normalised residual: median {np.median(rn):.4f}, "
          f"p95 {np.percentile(rn, 95):.4f}, max {rn.max():.4f}")

    nb = int(res["at_wall"].sum())
    if nb:
        print(f"  at the chamber wall: {nb} samples")
    amb = int((res["n_minima"] > 1).sum())
    print(f"  ambiguous samples: {amb}")
    if amb:
        sp = res["spread_m"][res["n_minima"] > 1] * 1e3
        print(f"    minima separated by up to {np.nanmax(sp):.1f} mm")
    gated = int(res["gated"].sum())
    if gated:
        print(f"  Ip-gated: {gated}")

    a, ipv = res["amp"][ok], res["ip"][ok]
    exp = ipv / res["I_ref"]
    m = np.isfinite(a) & (exp > 0)
    if m.sum() > 10:
        r = a[m] / exp[m]
        print(f"  amplitude / (Ip/I_ref): median {np.median(r):.3f} (expect 1.000)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="biot_savart",
                                description="Biot-Savart NLSQ displacement")
    p.add_argument("shot")
    p.add_argument("--probes", nargs="+", type=int, default=None,
                   help="1-based probe numbers (default: all 12)")
    p.add_argument("--weights", default="auto", choices=("auto", "last", "none"))
    p.add_argument("--ip-min", type=float, default=None)
    p.add_argument("--search", choices=("grid",), default="grid",
                   help="exhaustive lattice search; 'phi' needs the filament "
                        "answer and is only available through compare_methods")
    p.add_argument("--forward", choices=("internal", "cal_signal"),
                   default="internal")
    p.add_argument("--fit-ip", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="fit the amplitude (default) or fix it at Ip/I_ref "
                        "with --no-fit-ip")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--csv", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    res = run(a.shot, probes=a.probes,
              weights_source=(None if a.weights == "none" else a.weights),
              ip_min=a.ip_min, search=a.search, forward=a.forward,
              fit_ip=a.fit_ip, data_dir=a.data_dir, quiet=a.quiet)
    if not a.quiet:
        summarise(res)

    os.makedirs(OUT_DIR, exist_ok=True)
    npz = os.path.join(OUT_DIR, f"{a.shot}_bs.npz")
    np.savez_compressed(npz, **res)
    print(f"\nwrote {npz}")

    if a.csv:
        csv = os.path.join(OUT_DIR, f"{a.shot}_bs.csv")
        cols = np.column_stack([res["t"], res["dR_m"] * 1e3, res["dZ_m"] * 1e3,
                                res["resid_norm"], res["n_minima"],
                                res["at_wall"].astype(int), res["amp"]])
        np.savetxt(csv, cols, delimiter=",",
                   header="t,dR_mm,dZ_mm,resid_norm,n_minima,at_wall,amp",
                   comments="", fmt="%.6g")
        print(f"wrote {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
