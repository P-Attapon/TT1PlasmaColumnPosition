"""Biot-Savart NLSQ displacement -- edit the parameters below and run:

    python biot_savart_main.py

Fits (dR, dZ) directly to the probe signals by nonlinear least squares against
the exact filament field: no linear proxy, no polynomial, no Phi table, no
interpolation grid, no hull, no validity domain.

The method shares the probe calibration, the vacuum-field subtraction and the
single-filament ansatz with the Phi path. See
methods_script/biot_savart/README.md for what that means when reading agreement
between the two, and for the per-probe gain sensitivity that absolute-field
methods have and ratio methods do not.

Run the selftest at least once per checkout:

    python -m methods_script.biot_savart.selftest
"""

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from methods_script.biot_savart import adapter, invert

plt.style.use("seaborn-v0_8-dark-palette")

##################### Parameter setup ############################################
# shots to run. The data/<shot> directory must exist -- see README.
# shot_lst = [int(s) for s in os.listdir("data") if s.isnumeric()]
shot_lst = [1641]

# probes to fit, as a list of probe numbers; None uses all 12.
# Any subset of length >= 2 works and costs nothing but runtime -- there is no
# per-set map to build. With 2 probes the fit has no degrees of freedom left, so
# the residual carries no information; use >= 4 for resid_norm to mean anything.
#   use_probes = None
#   use_probes = [3, 4, 9, 10]
use_probes = None

# per-probe weights, from the same curation layer the filament method uses.
#   "auto" -> this shot's own pre-plasma window
#   "last" -> inherited from the previously stored shot
#   None   -> uniform
# Weights are rescaled to mean 1 internally, so only their ratios matter.
weights_source = "auto"

# amplitude handling. Same meaning as mprobe's fit_ip, so a run here and a run
# of the filament method are comparable only when the two agree.
#   True  -> the overall amplitude is eliminated analytically at every (dR, dZ),
#            so only ratios between probes enter the fit and the recovered
#            amplitude becomes an output. It should equal ip / I_ref.
#   False -> the amplitude is fixed at ip / I_ref, so the absolute signal level
#            enters the fit. This is what adaptive_select uses.
# With 4 probes, True leaves only one degree of freedom; False leaves two.
fit_ip = True

# search method. "grid" evaluates the residual at every point of a 1 mm lattice
# covering the chamber, refines every local minimum of that surface, and keeps
# the lowest. Subject to the lattice being fine enough to place a point in every
# basin, that is the global minimum inside the chamber rather than whichever
# minimum a starting guess happens to reach.
# The other method, "phi", descends from the filament method's own answer and is
# reached through compare_methods.py, which has the Phi result to seed it with.
search = "grid"

# extra plasma-current gate [A] on top of the Ip > 2500 A gate already applied
# by adaptive_select._load_shot. None disables it.
ip_min = None

# forward model. "internal" is field.py, vectorised; "cal_signal" calls the
# repository's scalar implementation. Selftest stage 2 checks they agree.
forward_model = "internal"

# overlay the filament (adaptive Phi) result and report |dR_BS - dR_Phi|.
# Costs a full adaptive_selection() run per shot.
compare_with_filament = True

# output
save_directory = os.path.join("result_plot", "biot_savart")
save_npz = True
save_csv = True
save_plot = True

##################################################################################

# ---- config consistency guards (fail fast on contradictory settings) ----------
def _validate_config():
    if not isinstance(fit_ip, bool):
        raise TypeError(f"fit_ip must be True or False, got {fit_ip!r}")
    if search != "grid":
        raise ValueError(f"search must be 'grid' here; 'phi' needs the filament "
                         f"answer and is reached through compare_methods.py. "
                         f"Got {search!r}")
    if forward_model not in ("cal_signal", "internal"):
        raise ValueError(f"forward_model must be 'cal_signal' or 'internal', "
                         f"got {forward_model!r}")
    if weights_source not in ("auto", "last", None):
        raise ValueError(f"weights_source must be 'auto', 'last' or None, "
                         f"got {weights_source!r}")
    if use_probes is not None:
        if not (isinstance(use_probes, (list, tuple)) and len(use_probes) >= 2):
            raise ValueError("use_probes must be None or a list of at least 2 "
                             f"probe numbers, got {use_probes!r}")
        bad = [p for p in use_probes if not (isinstance(p, int) and 1 <= p <= 12)]
        if bad:
            raise ValueError(f"use_probes contains invalid probe numbers {bad}; "
                             "probes are 1..12 (1-based).")
        if len(set(use_probes)) != len(use_probes):
            raise ValueError(f"use_probes contains duplicates: {use_probes!r}")

_validate_config()
# -------------------------------------------------------------------------------

geom = adapter.load_geometry(strict=True)
forward = adapter.load_forward(forward_model, geom)
forward_many = adapter.load_forward_many(geom)
print(f"geometry : {geom.source}")
print(f"           R0 = {geom.R0} m, probe circle = {geom.probe_radius} m, "
      f"I_ref = {geom.I_ref:.4g} A")
print(f"forward  : {forward_model}\n")

for shot_no in shot_lst:
    print(f"===== shot {shot_no} " + "=" * 40)
    try:
        t, ip, B = adapter.load_shot(shot_no)
    except Exception as exc:
        print(f"  cannot load shot {shot_no}: {exc}")
        continue
    if t.size == 0:
        print(f"  shot {shot_no}: no samples above the Ip gate; skipped")
        continue

    w = adapter.load_weights(shot_no, weights_source)
    if w is not None:
        dropped = [p for p, wi in zip(adapter.PROBES, w) if wi <= 0]
        print(f"  weights  : {weights_source}"
              + (f", probes dropped by curation: {dropped}" if dropped else ""))
    else:
        print("  weights  : uniform")
    print(f"  samples  : {t.size}  ({t[0]:.1f}-{t[-1]:.1f} ms)")
    print(f"  probes   : {'all 12' if use_probes is None else use_probes}")
    print(f"  fit_ip   : {fit_ip}    search: {search}")

    res = invert.invert_shot(
        forward, forward_many, t, ip, B,
        probes=use_probes, weights=w, ip_min=ip_min,
        search=search, fit_ip=fit_ip, I_ref=geom.I_ref,
        progress=lambda i, n: print(f"\r  solving  : {i}/{n}", end="", flush=True))
    print("\r" + " " * 40 + "\r", end="")

    # A solution on the chamber wall means the best fit inside the 0.20 m
    # limiter radius lies on it. Kept in the statistics but marked, since it is
    # a statement about the shot rather than about the solver.
    good = np.isfinite(res["dR_m"])
    print(f"  solved   : {int(good.sum())}/{t.size} samples "
          f"({int(res['at_wall'].sum())} at the chamber wall)")
    if not good.any():
        print("  nothing solved away from the bound; skipping output")
        continue

    dR_mm, dZ_mm = res["dR_m"] * 1e3, res["dZ_m"] * 1e3
    print(f"    dR  {dR_mm[good].min():8.1f} .. {dR_mm[good].max():8.1f} mm  "
          f"median {np.median(dR_mm[good]):8.1f}")
    print(f"    dZ  {dZ_mm[good].min():8.1f} .. {dZ_mm[good].max():8.1f} mm  "
          f"median {np.median(dZ_mm[good]):8.1f}")

    rn = res["resid_norm"][good]
    print(f"    normalised residual: median {np.median(rn):.4f}, "
          f"p95 {np.percentile(rn, 95):.4f}, max {rn.max():.4f}")

    amb = int((res["n_minima"] > 1).sum())
    if amb:
        sp = res["spread_m"][res["n_minima"] > 1] * 1e3
        print(f"    ambiguous: {amb} samples, minima up to {np.nanmax(sp):.1f} mm apart")

    if fit_ip:
        expected = ip / geom.I_ref
        m = good & np.isfinite(res["amp"]) & (expected > 0)
        if m.sum() > 10:
            r = res["amp"][m] / expected[m]
            print(f"    amplitude / (Ip/I_ref): median {np.median(r):.3f} "
                  f"(expect 1.000), IQR "
                  f"{np.percentile(r, 75) - np.percentile(r, 25):.3f}")

    dR_fil = dZ_fil = None
    if compare_with_filament:
        try:
            from adaptive_select import adaptive_selection
            fil = adaptive_selection(shot_no, weights_source=(weights_source or "auto"))
            dR_fil = np.interp(t, fil["t_ms"], fil["dR_m"] * 1e3,
                               left=np.nan, right=np.nan)
            dZ_fil = np.interp(t, fil["t_ms"], fil["dZ_m"] * 1e3,
                               left=np.nan, right=np.nan)
            gR = np.where(good, np.abs(dR_mm - dR_fil), np.nan)
            gZ = np.where(good, np.abs(dZ_mm - dZ_fil), np.nan)
            n_cmp = int(np.isfinite(gR).sum())
            print(f"    vs filament (adaptive), coverage {fil['coverage']:.1%}:")
            print(f"      |dR_BS - dR_Phi|  median {np.nanmedian(gR):6.2f} mm, "
                  f"p95 {np.nanpercentile(gR, 95):6.2f} mm  (n={n_cmp})")
            print(f"      |dZ_BS - dZ_Phi|  median {np.nanmedian(gZ):6.2f} mm, "
                  f"p95 {np.nanpercentile(gZ, 95):6.2f} mm")
            only_bs = int(np.sum(good & ~np.isfinite(dR_fil)))
            if only_bs:
                print(f"      {only_bs} samples solved here that Phi returned NaN for")
        except Exception as exc:
            print(f"    filament comparison unavailable: {exc}")

    os.makedirs(save_directory, exist_ok=True)

    if save_npz:
        p = os.path.join(save_directory, f"{shot_no}_bs.npz")
        np.savez_compressed(p, shot=shot_no, fit_ip=fit_ip, I_ref=geom.I_ref,
                            probes=np.array(use_probes if use_probes
                                            else adapter.PROBES),
                            **{k: v for k, v in res.items()})
        print(f"  wrote {p}")

    if save_csv:
        p = os.path.join(save_directory, f"{shot_no}_bs.csv")
        df = pd.DataFrame({"Time (ms)": t, "dR (mm)": dR_mm, "dZ (mm)": dZ_mm,
                           "resid_norm": res["resid_norm"],
                           "n_minima": res["n_minima"],
                           "at_wall": res["at_wall"].astype(int),
                           "amp": res["amp"], "IP (A)": ip})
        if dR_fil is not None:
            df["dR filament (mm)"] = dR_fil
            df["dZ filament (mm)"] = dZ_fil
        df.to_csv(p, index=False)
        print(f"  wrote {p}")

    if save_plot:
        fig, (axR, axZ, axr) = plt.subplots(
            3, 1, figsize=(9, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 3, 2]})

        plot_R = np.where(good, dR_mm, np.nan)
        plot_Z = np.where(good, dZ_mm, np.nan)
        axR.plot(t, plot_R, lw=1.4, label="Biot-Savart NLSQ")
        axZ.plot(t, plot_Z, lw=1.4, label="Biot-Savart NLSQ")
        if dR_fil is not None:
            axR.plot(t, dR_fil, lw=1.2, ls="--", label="filament (adaptive)")
            axZ.plot(t, dZ_fil, lw=1.2, ls="--", label="filament (adaptive)")

        for ax in (axR, axZ):
            if res["at_wall"].any():
                ax.axvspan(np.nan, np.nan, color="0.85")  # keeps legend order
        if res["at_wall"].any():
            for ax, y in ((axR, plot_R), (axZ, plot_Z)):
                ax.plot(t[res["at_wall"]],
                        np.full(int(res["at_wall"].sum()), 0.0), "|",
                        ms=6, color="0.55", label="at chamber wall")
        amb_m = res["n_minima"] > 1
        if amb_m.any():
            axR.plot(t[amb_m], dR_mm[amb_m], ".", ms=4, color="crimson",
                     label="ambiguous")
            axZ.plot(t[amb_m], dZ_mm[amb_m], ".", ms=4, color="crimson")

        axr.semilogy(t, res["resid_norm"], lw=1.0, color="0.35")
        axr.set_ylabel("resid / median|B|")
        axr.set_xlabel("time [ms]")
        axR.set_ylabel(r"$\Delta_R$ [mm]")
        axZ.set_ylabel(r"$\Delta_Z$ [mm]")
        axR.axhline(0, ls="--", color="gray", alpha=0.3)
        axZ.axhline(0, ls="--", color="gray", alpha=0.3)
        axR.legend(fontsize="small", frameon=False)
        for a, lab in zip((axR, axZ, axr), ("(a)", "(b)", "(c)")):
            a.text(0.02, 0.93, lab, transform=a.transAxes, fontsize=13,
                   fontweight="bold", va="top", ha="left")
        fig.suptitle(f"Shot {shot_no} -- Biot-Savart NLSQ ({search}, fit_ip={fit_ip}, "
                     f"{'all 12' if use_probes is None else use_probes})", y=0.95)
        fig.tight_layout()
        p = os.path.join(save_directory, f"{shot_no}_bs.png")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p}")
