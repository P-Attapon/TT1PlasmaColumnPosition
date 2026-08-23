"""
compare_methods.py
==================
Non-invasive wrapper that computes plasma horizontal (and vertical, where a
method provides it) displacement by MULTIPLE methods on the same shot and plots
them on shared axes for comparison.

Currently wired:
  * "filament"   - the M-probe toroidal-filament method with ADAPTIVE SELECTION
                   (adaptive_select.adaptive_selection). Provides dR and dZ in
                   metres from the vessel centre. Cannot refuse a shot: samples no
                   probe set can represent are NaN, and coverage is reported.
  * "position_c" - the real-time feedback controller's 2-probe method, ported
                   from position.c (position_c_displacement.py). Provides dR
                   only, in mm -> converted to metres. SAME centre (Rg=0.65).

  * "ai_camera"  - CCD-image AI edge/centre detection (<shot>_pred.txt). Provides
                   dR and dZ, converted from image pixels via AI_MMPERPX about a
                   reference centre pixel, and time-clipped to the Ip-gated plasma
                   window. Independent CROSS-CHECK only, never a ground truth.

  * "biot_savart" - the same toroidal-filament physics as "filament", but with
                   (dR, dZ) fitted directly to the probe signals by nonlinear
                   least squares instead of inverted through the linear proxy and
                   the Phi map (methods_script/biot_savart). Provides dR and dZ
                   in metres. It shares the calibration and the filament ansatz
                   with "filament", so the gap between the two measures the Phi
                   path's APPROXIMATION error, not its accuracy. Set
                   BS_FOLLOW_FILAMENT to make it match the filament method's
                   probes (fit_ip is already shared via FIT_IP), leaving
                   difference between the two curves.

Nothing in the existing method modules is modified. This script only imports and
calls them, aligns their time bases, and plots.

Run from the repo root (next to main.py):
    python compare_methods.py 1641
    python compare_methods.py 1641 1643

Each method is plotted on its OWN native time samples (no interpolation, no
resampling) so the overlaid curves show exactly where each method has real data.
No aggregate agreement statistic is computed: comparing methods on different
time bases would require interpolation, which can fabricate agreement or
disagreement where samples are sparse (e.g. the video-rate AI method). The
curves are left to speak for themselves.
"""
import os
import sys
import numpy as np
import pandas as pd   # REQUIRED: the AI Ip-gate below uses pd.read_csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ CONFIG
DATA_DIR = "data"

# --- which methods to plot -------------------------------------------------
# Master switch per method. A method set False is not loaded or plotted, and its
# own configuration block below is ignored -- this overrides everything else for
# that method. "biot_savart_phi" only appears when BS_PLOT_PHI_START is also set.
PLOT_METHODS = {
    "filament":    True,
    "biot_savart": True,
    "position_c":  False,
    "ai_camera":   True,
}

# Amplitude treatment, applied to BOTH magnetic methods so they are always
# compared on the same footing. True fits the amplitude (immune to a common gain
# error, but needs one more live probe); False fixes it at Ip/I_ref (exposes a
# calibration error as a position error). This single flag replaces the old
# separate FIL_FIT_IP / BS_FIT_IP -- they could disagree, which quietly made the
# filament-vs-Biot-Savart gap a function of configuration rather than of method.
FIT_IP = True
OUT_DIR = os.path.join("result_plot", "comparison")

# --- filament method configuration (mirror of main.py's mprobe block) ---
FIL_PROBES = "adaptive"  # "adaptive" (recommended) | list of probe-set lists
# "adaptive": ADAPTIVE SELECTION -- see adaptive_select.adaptive_selection().
#
#   Phase A, once per CONFIGURATION (no shot involved): load each candidate set's
#   cached Phi map, convex hull, and rt(u,v) field, then order the sets by
#   good-frac (share of the hull interior where the inversion is faithful),
#   tie-broken by median rt. Hulls and rt fields are shot-INDEPENDENT -- built from
#   the forward Biot-Savart model over the operating domain -- so they are computed
#   once and cached, and the order is known before a shot begins.
#
#   Phase B, per SAMPLE: accept a set only where it is inside its own hull AND
#   rt(u,v) <= RT_GOOD. Keep the current set while it stays acceptable
#   (hysteresis); otherwise walk the static order and take the first set that
#   qualifies. A sample no set accepts becomes NaN.
#
# WHY ACCURACY IS ASKED PER SAMPLE, NOT PER SHOT
# ----------------------------------------------
# An aggregate per-shot admission gate is the wrong instrument here, for two
# reasons. It is BINARY: a set marginally over a whole-shot ceiling refuses the
# entire shot, discarding every sample that was perfectly invertible -- the right
# treatment of an unreachable sample is a NaN at that sample. And it can only
# police an order inherited across shots, which Phase A does not produce, being
# shot-independent.
#
# Accuracy is therefore asked LOCALLY, per sample. Adaptive selection cannot
# refuse a shot; where it has no valid set it returns NaN there and reports the
# coverage it achieved.
FIL_ADAPTIVE_WEIGHTS = "auto"   # "last" (inherited -- what real time must use) |
                                # "auto" (this shot's own pre-shot window).
                                # ADAPTIVE ONLY. "last" is the default because
                                # nothing in adaptive selection needs to be
                                # computed on the live shot.
FIL_USE_MPROBE = True
FIL_WEIGHTS = "auto"          # "auto" | dict | None      (fixed-set path only)
# fit_ip is set once, at the top, by FIT_IP (see the "which methods" block).
FIL_GAINS = None
# NOTE: curation thresholds and the Phi grid are NOT exposed here. They have single
# settled values at their source of truth -- curation.py and mprobe.py -- shared by
# main.py and the adaptive path. Edit those modules to change a value once; that
# keeps all entry points consistent and avoids fragmenting the grid-keyed Phi cache.

# --- position.c options ---
POSC_PICKUP_SOURCE = "filament"   # "positionc" (faithful) | "filament" (error-separation)

# Sign convention, PER PICKUP SOURCE. These are two different conventions, not
# one: measured against the filament trace, the two sources come out with
# OPPOSITE polarity on every shot tested, and shot 4404 inverts both relative to
# 1643 and 2766. A single shared flag therefore cannot express the truth table
# and silently mis-signs whichever source it was not tuned for.
#
#   shot | positionc | filament
#   1643 |   False   |   True
#   2766 |   False   |   True
#   4404 |   True    |   False
#
# FLIP_SIGN only negates -- it never changes the spread -- so it can correct a
# polarity but never rescue a diverging trace. On 4404 the "positionc" pickup has
# std ~9.5 m with the flag either way; that is a calibration failure, not a sign.
POSC_FLIP_SIGN = {"positionc": False, "filament": False}
# --- default shot list -------------------------------------------------------
# Used when the script is run with NO command-line argument. Command-line shots
# always win, so this is a convenience for IDE "Run" buttons and repeated local
# runs, not a hidden default that overrides an explicit request.
#
# Keep it small and keep it to shots whose data/<shot>/ you actually have: a
# missing directory raises part-way through the loop, after earlier shots have
# already been plotted. Leave the list EMPTY to force a shot to be given.
DEFAULT_SHOTS = ["1641","1643","2400","2766","3616","3970","4047","4048","4049","4052","4398","4404"]
# --- Biot-Savart NLSQ options ---
BS_FOLLOW_FILAMENT = True  # True = fit whichever probes the filament method used.
                            # With FIL_PROBES == "adaptive" that is its per-sample
                            # choice; with a fixed FIL_PROBES it is FIL_PROBES[0],
                            # the set load_filament plots. FOR COMPARISON ONLY: it
                            # makes the two curves differ by the inversion alone,
                            # so a remaining gap is the (u,v) proxy and the Phi map
                            # rather than probe selection. Overrides BS_PROBES and
                            # reporting each override it applies. Amplitude
                            # treatment comes from FIT_IP and is shared, so only
                            # probes are overridden here.
BS_PROBES = None            # None = all 12 | list of probe numbers, e.g. [3,4,9,10]
BS_WEIGHTS = "auto"         # "auto" | "last" | None (uniform)
# fit_ip is set once, at the top, by FIT_IP.
BS_SEARCH = "grid"          # "grid" = exhaustive 1 mm lattice over the chamber,
                            # every local minimum refined, lowest kept: the
                            # global minimum inside the 0.20 m limiter radius,
                            # subject to the lattice resolving every basin.
                            # "phi"  = descend from the filament method's answer
                            # only. Register "biot_savart_phi" as a method to
                            # plot both and see where they part company.
BS_FORWARD = "internal"     # "internal" (vectorised) | "cal_signal"
BS_PLOT_PHI_START = False   # also plot a second Biot-Savart curve seeded from the
                            # filament answer instead of searched globally. Where
                            # the two coincide the Phi answer is in the global
                            # basin; where they part, it is not. Doubles the
                            # Biot-Savart runtime.
PLOT_BAND = True            # shade a per-sample uncertainty band behind the
                            # magnetic curves (filament and Biot-Savart). It is
                            # derived from each method's own fit residual at its
                            # own answer, converted to millimetres through the
                            # Jacobian of the forward model. A CONDITIONING
                            # measure -- how tightly the probes pin the position
                            # given how badly the model fits -- not a confidence
                            # interval: the residual is dominated by model error,
                            # which is systematic rather than random.
BAND_SIGMAS = 1.0           # width of the shaded band, in standard deviations.
                            # 1.0 by default: "95%" would invite a probability
                            # reading the quantity does not support.
BS_HIDE_AT_WALL = False     # blank samples whose best fit lies on the chamber
                            # wall. They are real fits, not solver failures, so
                            # the default keeps them and marks them instead.

# --- AI camera options ---
# The AI .txt (named "<shot>_pred.txt", e.g. 1641_pred.txt, inside data/<shot>/)
# has columns FRAME,TIME(ms),...,txc,tyc,tr,conf. txc/tyc are the ABSOLUTE image-
# pixel coordinates of the detected plasma centre; tr is the radius (px); conf in
# (0,1). Displacement = (pixel - centre_pixel) * scale.
AI_MMPERPX = 250.0 / 396.0        # mm per pixel (396 px = 250 mm minor radius)
AI_CONF_MIN = 0.5                 # drop frames with conf below this
# Vessel-centre reference pixel. Leave BOTH as None to use the shot-mean of the
# confident frames (dR/dZ are then deviations from the mean position). To use a
# fixed calibrated vessel centre instead, set BOTH to pixel numbers. <-- CENTRE.
AI_CENTRE_TXC = 1155
AI_CENTRE_TYC = 525
# Axis orientation (fixed): right of image = outboard = +R; bottom-left pixel
# origin so tyc increases upward = +Z. Both are plain +(pixel - centre)*scale,
# built into load_ai_camera(); no sign switch is exposed. If dZ comes out
# inverted vs the filament, negate the dZ_mm line in load_ai_camera().
# -------------------------------------------------------------------------


def load_filament(shot):
    """Run the filament method. Returns dict with t_ms, dR_m, dZ_m (np arrays).

    With FIL_PROBES == "adaptive" this is adaptive selection (see the config block
    above). It never raises on a difficult shot: samples no probe set can represent
    come back as NaN, and `coverage` reports how much of the shot was recovered.
    """
    if FIL_PROBES == "adaptive":
        if FIL_ADAPTIVE_WEIGHTS not in ("auto", "last"):
            raise ValueError(
                f"FIL_ADAPTIVE_WEIGHTS must be 'auto' or 'last', got "
                f"{FIL_ADAPTIVE_WEIGHTS!r}.")
        from adaptive_select import adaptive_selection
        r = adaptive_selection(shot, weights_source=FIL_ADAPTIVE_WEIGHTS,
                               fit_ip=FIT_IP)
        cov = r["coverage"]
        label = f"filament [adaptive, cov={cov:.0%}, sw={r['n_switch']}]"
        if cov < 1.0:
            # Not an error. Reported so a partially-recovered shot is never mistaken
            # for a complete one just because the curve looks continuous on the plot.
            print(f"[{shot}] filament: coverage {cov:.1%} -- "
                  f"{int(round((1-cov)*len(r['t_ms'])))} sample(s) had no valid "
                  f"probe set and are NaN.")
        out = {"t_ms": r["t_ms"], "dR_m": r["dR_m"], "dZ_m": r["dZ_m"],
               "label": label}
        if PLOT_BAND:
            sets = [[int(x) for x in nm.split()] for nm in r["order"]]
            _t, sR, sZ = _magnetic_band(
                shot, r["t_ms"], r["dR_m"], r["dZ_m"],
                weights_source=FIL_ADAPTIVE_WEIGHTS, fit_ip=FIT_IP,
                chosen=r["chosen"], probe_sets=sets)
            out["sigma_R_m"], out["sigma_Z_m"] = sR, sZ
        return out

    if not (isinstance(FIL_PROBES, (list, tuple)) and len(FIL_PROBES) > 0
            and all(isinstance(s_, (list, tuple)) for s_ in FIL_PROBES)):
        raise ValueError(
            "FIL_PROBES must be 'adaptive' or a non-empty list of probe-set "
            f"lists, e.g. [[1,2,3,4]]. Got: {FIL_PROBES!r}")
    if not FIL_USE_MPROBE and any(len(s_) != 4 for s_ in FIL_PROBES):
        raise ValueError(
            "FIL_USE_MPROBE=False supports only 4-probe antipodal sets; "
            f"FIL_PROBES has a set of another length: {FIL_PROBES!r}.")

    from methods_script.toroidal_filament.TFM import TFM_main
    from methods_script.toroidal_filament.parameters import probe_lst_to_str

    shot_dir = os.path.join(DATA_DIR, str(shot))
    probe_str = [probe_lst_to_str(x) for x in FIL_PROBES]
    mprobe_cfg = ({"weights": FIL_WEIGHTS, "fit_ip": FIT_IP, "gains": FIL_GAINS}
                  if FIL_USE_MPROBE else None)
    df = TFM_main(shot_path=shot_dir, use_probe_set=probe_str, mprobe=mprobe_cfg)
    t = df["Time (ms)"].to_numpy()
    key = probe_str[0]
    out = {"t_ms": t, "dR_m": df[key + " R"].to_numpy(),
           "dZ_m": df[key + " Z"].to_numpy(), "label": f"filament [{key}]"}
    if PLOT_BAND and FIL_USE_MPROBE:
        _t, sR, sZ = _magnetic_band(shot, out["t_ms"], out["dR_m"], out["dZ_m"],
                                    probes=list(FIL_PROBES[0]),
                                    weights_source=(FIL_WEIGHTS if isinstance(
                                        FIL_WEIGHTS, str) else "auto"),
                                    fit_ip=FIT_IP)
        out["sigma_R_m"], out["sigma_Z_m"] = sR, sZ
    return out


def load_position_c(shot):
    """Run the position.c port. Returns dict with t_ms, dR_m (dZ = None)."""
    import position_c_displacement as pc
    pc.PICKUP_SOURCE = POSC_PICKUP_SOURCE
    if isinstance(POSC_FLIP_SIGN, dict):
        pc.FLIP_SIGN = POSC_FLIP_SIGN.get(POSC_PICKUP_SOURCE, False)
    else:                       # tolerate the old single-bool form
        pc.FLIP_SIGN = bool(POSC_FLIP_SIGN)
    shot_dir = os.path.join(DATA_DIR, str(shot))
    t, displace_mm, displace_plot_mm, Ip = pc.position_c_displacement(shot_dir)
    return {"t_ms": t, "dR_m": displace_plot_mm / 1e3, "dZ_m": None,
            "label": f"position.c (2-probe, pickup={POSC_PICKUP_SOURCE})"}


def report_probe_health(shot):
    """Print probes 1 & 7 pre-plasma sigma from curation, to correlate the
    position.c error source with the health of the two probes it uses."""
    try:
        from methods_script.toroidal_filament.curation import compute_weights
        shot_dir = os.path.join(DATA_DIR, str(shot))
        w, sig, val = compute_weights(shot_dir, list(range(1, 13)))
        order = sorted(range(1, 13), key=lambda p: sig[p])
        rank = {p: i + 1 for i, p in enumerate(order)}   # 1 = cleanest
        print(f"[{shot}] position.c probes: "
              f"GBP1T sigma={sig[1]:.2e} (rank {rank[1]}/12, "
              f"{'valid' if val[1] else 'GATED'}); "
              f"GBP7T sigma={sig[7]:.2e} (rank {rank[7]}/12, "
              f"{'valid' if val[7] else 'GATED'})")
    except Exception as e:
        print(f"[{shot}] probe-health report unavailable ({type(e).__name__}: {e})")


# window (ms) over which the shape/amplitude decomposition is computed. This is a
# DIAGNOSTIC-only alignment: the reference method is interpolated onto the other's
# timestamps purely to compute correlation/slope; the plotted curves are never
# interpolated. Kept off the plot on purpose.
DECOMP_WINDOW_MS = (340.0, 448.0)


def shape_amplitude_decomposition(results, ref="filament"):
    """Print, for each non-reference method, how it relates to the reference:
      corr  - Pearson correlation of the two traces (SHAPE agreement, scale-free)
      slope - best-fit gain  other = slope*ref + offset  (AMPLITUDE match; 1.0 ideal)
      ampR  - std(other)/std(ref)  (amplitude ratio; 1.0 ideal)
    This separates the two error axes: pickup CALIBRATION mainly moves corr;
    the 2-probe FORMULA mainly limits slope/ampR. A high corr with a small slope
    means 'tracks where the plasma moves, but under-reads how far'.

    NOTE: for this diagnostic the reference is interpolated onto the other
    method's timestamps. That is acceptable HERE (both magnetic methods share the
    50 kHz grid); for a sparse method (AI video) treat corr/slope with care and
    prefer nearest-sample matching. The plotted curves use no interpolation.
    """
    if ref not in results:
        return
    a, b = DECOMP_WINDOW_MS
    tr = np.asarray(results[ref]["t_ms"])
    yr = np.asarray(results[ref]["dR_m"]) * 1e3
    printed_header = False
    for name, r in results.items():
        if name == ref or r.get("dR_m") is None:
            continue
        to = np.asarray(r["t_ms"]); yo = np.asarray(r["dR_m"]) * 1e3
        yr_on_o = np.interp(to, tr, yr, left=np.nan, right=np.nan)
        ok = np.isfinite(yr_on_o) & np.isfinite(yo) & (to >= a) & (to <= b)
        if ok.sum() < 10:
            continue
        R, P = yr_on_o[ok], yo[ok]
        corr = np.corrcoef(R, P)[0, 1]
        A = np.column_stack([R, np.ones(len(R))])
        slope, offset = np.linalg.lstsq(A, P, rcond=None)[0]
        ampR = np.std(P) / np.std(R) if np.std(R) > 0 else np.nan
        if not printed_header:
            print(f"[shape/amplitude vs {ref}, {a}-{b} ms; corr=shape, slope&ampR=amplitude]")
            printed_header = True
        print(f"    {name:12s}: corr {corr:+.3f}  slope {slope:+.3f}  "
              f"offset {offset:+6.1f} mm  ampR {ampR:.3f}  (n={ok.sum()})")


def load_ai_camera(shot):
    """Load AI CCD centre-detection and convert to displacement (dR, dZ) in the
    filament frame. Returns dict with t_ms, dR_m, dZ_m, or None if no file.

    File: data/<shot>/<shot>_pred.txt, comma-separated with a header line
    'FRAME,TIME,xc,yc,w,h,r,txc,tyc,tr,conf'. We use TIME (ms), txc, tyc (pixel
    centre), and conf. Steps:
      1. keep frames with conf >= AI_CONF_MIN,
      2. centre pixel = shot-mean of confident frames (or a fixed calibrated
         pixel if AI_CENTRE_TXC/TYC are set),
      3. dR = (txc - txc_ref) * mm/px   (right of image = outboard = +R),
         dZ = (tyc - tyc_ref) * mm/px   (bottom-left origin: tyc increases up = +Z),
      4. return metres, on the AI's own (sparse) timestamps - no resampling.
    Centre treated as the vessel centre (== filament R0/Z0); change the reference
    in the AI config block if that assumption changes.
    """
    path = os.path.join(DATA_DIR, str(shot), f"{shot}_pred.txt")
    if not os.path.exists(path):
        return None

    frames = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.upper().startswith("FRAME"):
                continue
            parts = line.split(",")
            if len(parts) < 11:
                continue
            try:
                t_ms = float(parts[1]); txc = float(parts[7])
                tyc = float(parts[8]);  conf = float(parts[10])
            except ValueError:
                continue
            frames.append((t_ms, txc, tyc, conf))
    if not frames:
        return None

    arr = np.array(frames)
    t_ms, txc, tyc, conf = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    keep = conf >= AI_CONF_MIN
    # clip to the Ip-gated plasma window: the camera often runs longer than the
    # plasma (pre-breakdown / post-decay frames), where the detector reports
    # spurious centres and there is no magnetic data to compare against. Keep only
    # AI frames whose time falls inside |Ip| > IP gate, read from the shot's IP1.
    try:
        from position_c_displacement import IP_GATE
        ip_df = pd.read_csv(os.path.join(DATA_DIR, str(shot), "IP1.txt"),
                            sep=r"\s+", skiprows=8, header=None,
                            names=["t", "v"])
        t_ip, ip = ip_df["t"].to_numpy(), ip_df["v"].to_numpy()
        gated = np.abs(ip) > IP_GATE
        if gated.any():
            tmin, tmax = t_ip[gated].min(), t_ip[gated].max()
            # PROVENANCE GUARD. Every shot folder once carried a COPY of shot
            # 1641's prediction file. The copies were valid detections -- of the
            # wrong discharge -- so nothing crashed and nothing looked malformed;
            # the comparison silently measured magnetics against another shot's
            # plasma. It produced a 115 mm "filament error" on 2766 that was
            # entirely an artefact, and a fictitious dZ investigation on 2400.
            # The cheapest reliable tell is the TIME span: this shot's file must
            # overlap this shot's plasma window. 1641 spans 331-452 ms, which
            # does not sit inside 4404's 334-341 ms. Warn, do not raise: an
            # oddly-gated shot is a data question, not a code defect.
            # Test CONTAINMENT, not overlap. The 1641 copies span 285-452 ms,
            # which CONTAINS every other shot's window, so an overlap test
            # passes them silently -- that is exactly how this survived. A file
            # far wider than the plasma it claims to describe is the signature.
            span = t_ms.max() - t_ms.min()
            plasma = tmax - tmin
            outside = ((t_ms < tmin) | (t_ms > tmax)).mean()
            # Thresholds set from the real files: a correct file runs a
            # little before and after the plasma (2400: 24 ms span vs a 13 ms
            # window, 51% of frames outside) so those alone are NOT suspicious.
            # The 1641 copies are 166 ms against 13 ms with 90% outside. 4x span
            # and 75% outside separates them with room to spare.
            if span > 4.0 * plasma and outside > 0.75:
                print(f"[{shot}] *** WARNING: {shot}_pred.txt spans "
                      f"{t_ms.min():.1f}-{t_ms.max():.1f} ms ({span:.0f} ms) "
                      f"against a plasma window of {tmin:.1f}-{tmax:.1f} ms "
                      f"({plasma:.0f} ms); {100*outside:.0f}% of its frames "
                      f"fall outside the plasma. This file may belong to a "
                      f"DIFFERENT SHOT -- check before trusting any camera "
                      f"comparison from it. ***")
            keep = keep & (t_ms >= tmin) & (t_ms <= tmax)
        else:
            print(f"[{shot}] WARNING: no IP1 samples above IP_GATE={IP_GATE:.0f} A; "
                  f"AI camera NOT time-gated.")
    # Narrow on purpose. A missing or malformed IP1.txt is a DATA problem the
    # unclipped fallback should absorb; a NameError/AttributeError is a CODE
    # defect and must crash loudly. A bare `except Exception` would hide a
    # missing `import pandas as pd` here, silently disabling the gate on every
    # shot -- which is exactly the failure this narrowing prevents recurring.
    except (OSError, ValueError, KeyError, ImportError) as e:
        print(f"[{shot}] WARNING: AI camera Ip-gating failed "
              f"({type(e).__name__}: {e}); showing ALL frames, unclipped.")
    t_ms, txc, tyc = t_ms[keep], txc[keep], tyc[keep]
    if len(t_ms) < 2:
        return None

    if AI_CENTRE_TXC is None or AI_CENTRE_TYC is None:
        txc_ref, tyc_ref = float(np.mean(txc)), float(np.mean(tyc))
        ref_note = "shot-mean"
    else:
        txc_ref, tyc_ref = float(AI_CENTRE_TXC), float(AI_CENTRE_TYC)
        ref_note = "fixed"

    dR_mm = (txc - txc_ref) * AI_MMPERPX      # right of image = outboard = +R
    dZ_mm = -(tyc - tyc_ref) * AI_MMPERPX     # top-left origin: tyc increases downward, so +Z = -(tyc-ref)      # bottom-left origin: tyc up = +Z
    print(f"[{shot}] ai_camera: {len(t_ms)} frames (conf>={AI_CONF_MIN}), "
          f"centre {ref_note} px=({txc_ref:.1f},{tyc_ref:.1f})")
    return {"t_ms": t_ms, "dR_m": dR_mm / 1e3, "dZ_m": dZ_mm / 1e3,
            "label": f"AI camera (centre={ref_note})"}


def _magnetic_band(shot, t_ms, dR_m, dZ_m, probes=None, weights_source="auto",
                   fit_ip=False, chosen=None, probe_sets=None):
    """Per-sample (sigma_dR, sigma_dZ) in metres for a magnetic method.

    Works for any method that reports a filament position: the residual is
    evaluated at whatever answer is passed in, so the filament and Biot-Savart
    bands are computed the same way and are directly comparable. A method whose
    answer fits the probes worse gets a wider band, which is the point.
    """
    from methods_script.biot_savart import adapter, invert

    geom = adapter.load_geometry(strict=True)
    fwd = adapter.load_forward(BS_FORWARD, geom)
    fwd_many = adapter.load_forward_many(geom)
    t, ip, B = adapter.load_shot(shot, data_dir=DATA_DIR)
    w = adapter.load_weights(shot, weights_source)

    # Positions may be sampled differently from the shot; put them on t.
    if len(t_ms) != len(t) or not np.allclose(t_ms, t):
        dR_m = np.interp(t, t_ms, dR_m, left=np.nan, right=np.nan)
        dZ_m = np.interp(t, t_ms, dZ_m, left=np.nan, right=np.nan)

    sR, sZ = invert.sigma_shot(fwd, fwd_many, t, ip, B, dR_m, dZ_m,
                               probes=probes, weights=w, fit_ip=fit_ip,
                               I_ref=geom.I_ref, chosen=chosen,
                               probe_sets=probe_sets)
    return t, sR, sZ


def load_biot_savart(shot, search=None):
    """Run the Biot-Savart NLSQ method. Returns dict with t_ms, dR_m, dZ_m.

    Samples whose fit sat on the solver bound are blanked to NaN when
    BS_HIDE_AT_BOUND is set: the bound is where the gradient need not vanish, so
    such a sample is a failure to fit rather than a position.

    With BS_FOLLOW_FILAMENT both the probe set and fit_ip are taken from the
    filament method, so the two curves differ by the inversion alone and a
    remaining gap is the (u,v) proxy and the Phi map rather than configuration.
    The probes are its per-sample adaptive choice when FIL_PROBES == "adaptive",
    otherwise the fixed FIL_PROBES[0] that load_filament plots. Each override is
    reported, so the run never differs silently from the config block.

    That is a comparison aid, not a way to run the method. In the adaptive case
    the selection is made by the filament path's criteria on its own Phi maps,
    and nothing here validates it.
    """
    from methods_script.biot_savart import adapter, invert

    geom = adapter.load_geometry(strict=True)
    fwd = adapter.load_forward(BS_FORWARD, geom)
    fwd_many = adapter.load_forward_many(geom)
    t, ip, B = adapter.load_shot(shot, data_dir=DATA_DIR)
    w = adapter.load_weights(shot, BS_WEIGHTS)

    search = BS_SEARCH if search is None else search
    phi_xy = None                # filled in below when search == "phi"
    fit_ip = FIT_IP              # single flag; may be overridden below when
                                 # following the filament, though with one flag
                                 # the two already agree
    follow_fixed = None          # probe list when following a fixed FIL_PROBES
    if BS_FOLLOW_FILAMENT:
        if BS_PROBES is not None:
            print(f"[{shot}] biot_savart: BS_FOLLOW_FILAMENT overrides "
                  f"BS_PROBES = {BS_PROBES}.")

        # With a single FIT_IP flag both methods already use the same amplitude
        # treatment, so there is nothing to reconcile here except the one case
        # the flag cannot reach: the legacy non-mprobe filament path has no
        # fit_ip at all, so a comparison against it is not like-for-like.
        if FIL_PROBES != "adaptive" and not FIL_USE_MPROBE:
            print(f"[{shot}] biot_savart: the filament path is not using mprobe, "
                  f"so it has no fit_ip; the comparison is not like-for-like on "
                  f"amplitude.")

    if BS_FOLLOW_FILAMENT and FIL_PROBES == "adaptive":
        from adaptive_select import adaptive_selection
        sel = adaptive_selection(shot, weights_source=FIL_ADAPTIVE_WEIGHTS)
        if search == "phi":
            phi_xy = np.column_stack([sel["dR_m"], sel["dZ_m"]])
        if len(sel["t_ms"]) != len(t):
            raise ValueError(
                f"[{shot}] adaptive_selection returned {len(sel['t_ms'])} samples "
                f"but load_shot returned {len(t)}; cannot align the per-sample "
                f"probe choice.")
        probe_sets = [[int(x) for x in name.split()] for name in sel["order"]]
        r = invert.invert_shot_varying(
            fwd, fwd_many, t, ip, B, probe_sets, sel["chosen"],
            search=search, phi_xy=phi_xy,
            weights=w, I_ref=geom.I_ref, fit_ip=fit_ip)
        used = r["n_probes"][r["n_probes"] > 0]
        n_sw = int(np.sum(np.diff(sel["chosen"][sel["chosen"] >= 0]) != 0))
        print(f"[{shot}] biot_savart: following filament probes, adaptive "
              f"({n_sw} switch(es), {used.min() if used.size else 0}-"
              f"{used.max() if used.size else 0} probes per sample).")
    else:
        if search == "phi" and phi_xy is None:
            fil = load_filament(shot)
            phi_xy = np.column_stack([
                np.interp(t, fil["t_ms"], fil["dR_m"]),
                np.interp(t, fil["t_ms"], fil["dZ_m"])])
        if BS_FOLLOW_FILAMENT:
            # Fixed-set filament path. load_filament plots FIL_PROBES[0], so that
            # is the set to follow; any further sets it computed are not shown.
            if not (isinstance(FIL_PROBES, (list, tuple)) and len(FIL_PROBES)
                    and all(isinstance(x, (list, tuple)) for x in FIL_PROBES)):
                raise ValueError(
                    "BS_FOLLOW_FILAMENT needs FIL_PROBES to be 'adaptive' or a "
                    f"non-empty list of probe-set lists. Got: {FIL_PROBES!r}")
            follow_fixed = [int(x) for x in FIL_PROBES[0]]
            if len(FIL_PROBES) > 1:
                print(f"[{shot}] biot_savart: FIL_PROBES lists "
                      f"{len(FIL_PROBES)} sets; following {follow_fixed}, the "
                      f"one load_filament plots.")
            else:
                print(f"[{shot}] biot_savart: following filament probes, fixed "
                      f"{follow_fixed}.")
        r = invert.invert_shot(
            fwd, fwd_many, t, ip, B,
            probes=(follow_fixed if follow_fixed is not None else BS_PROBES),
            weights=w, search=search, phi_xy=phi_xy,
            fit_ip=fit_ip, I_ref=geom.I_ref)

    dR, dZ = r["dR_m"].copy(), r["dZ_m"].copy()
    n_wall = int(r["at_wall"].sum())
    if BS_HIDE_AT_WALL and n_wall:
        dR[r["at_wall"]] = np.nan
        dZ[r["at_wall"]] = np.nan
    solved = int(np.isfinite(dR).sum())
    if n_wall:
        print(f"[{shot}] biot_savart: {n_wall} sample(s) best-fit on the "
              f"chamber wall ({'blanked' if BS_HIDE_AT_WALL else 'kept'}).")
    n_amb = int((r["n_minima"] > 1).sum())
    if n_amb:
        print(f"[{shot}] biot_savart: {n_amb} ambiguous sample(s) -- more than "
              f"one position reproduces the signals.")
    if BS_FOLLOW_FILAMENT:
        probe_lbl = ("follows filament" if follow_fixed is None
                     else f"follows filament {follow_fixed}")
    else:
        probe_lbl = "all 12" if BS_PROBES is None else str(BS_PROBES)
    label = (f"Biot-Savart NLSQ [{probe_lbl}, {search}, "
             f"fit_ip={fit_ip}, cov={solved / max(len(dR), 1):.0%}]")
    out = {"t_ms": t, "dR_m": dR, "dZ_m": dZ, "label": label}
    if PLOT_BAND:
        out["sigma_R_m"] = r["sigma_R_m"]
        out["sigma_Z_m"] = r["sigma_Z_m"]
    return out


def load_biot_savart_phi(shot):
    """Biot-Savart seeded from the filament answer instead of searched globally.

    Plotted alongside "biot_savart" it separates two reasons the two methods can
    disagree. Where the curves coincide, the Phi answer is in the same basin as
    the global minimum and the whole gap to the filament curve is proxy error.
    Where they part, the Phi proxy landed in a different basin: compare their
    residuals to tell a genuine degeneracy from a Phi answer that simply fits
    worse.
    """
    return load_biot_savart(shot, search="phi")


METHODS = {
    "filament": load_filament,
    "biot_savart": load_biot_savart,
    "biot_savart_phi": load_biot_savart_phi,
    "position_c": load_position_c,
    "ai_camera": load_ai_camera,
}
if not BS_PLOT_PHI_START:
    METHODS.pop("biot_savart_phi", None)

# PLOT_METHODS is the master switch: drop any method turned off, so its loader is
# never called and its config block has no effect. biot_savart_phi follows the
# biot_savart switch, since it is the same method under a different search.
for _m, _on in PLOT_METHODS.items():
    if not _on:
        METHODS.pop(_m, None)
        if _m == "biot_savart":
            METHODS.pop("biot_savart_phi", None)
_unknown = set(PLOT_METHODS) - {"filament", "biot_savart", "position_c", "ai_camera"}
if _unknown:
    raise ValueError(f"PLOT_METHODS has unknown method(s): {sorted(_unknown)}")

STYLE = {
    "filament":  {"color": "C2", "lw": 1.6, "ls": "-"},
    "biot_savart": {"color": "C4", "lw": 1.4, "ls": (0, (5, 1))},
    "biot_savart_phi": {"color": "C1", "lw": 1.2, "ls": (0, (1, 1))},
    "position_c": {"color": "C3", "lw": 1.3, "ls": "--"},
    "ai_camera": {"color": "C0", "lw": 1.6, "ls": "-."},
}


def compare_shot(shot):
    results = {}
    for name, loader in METHODS.items():
        try:
            r = loader(shot)
        except FileNotFoundError as e:
            print(f"[{shot}] {name}: missing file ({e}); skipped")
            continue
        except Exception as e:
            print(f"[{shot}] {name}: error ({type(e).__name__}: {e}); skipped")
            continue
        if r is not None:
            results[name] = r
    if not results:
        print(f"[{shot}] no methods produced output")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    have_dZ = any(r.get("dZ_m") is not None for r in results.values())
    nrows = 2 if have_dZ else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 4 * nrows), sharex=True, squeeze=False)
    axR = axes[0, 0]
    
    def _band(ax, r, st, key_pos, key_sig):
        """Shade +-BAND_SIGMAS around a curve. Sizes are aligned by the loader,
        which interpolates its positions onto the shot's time base before
        computing the band."""
        sig = r.get(key_sig)
        if sig is None:
            return
        y = np.asarray(r[key_pos], float) * 1e3
        s_mm = np.asarray(sig, float) * 1e3
        m = np.isfinite(y) & np.isfinite(s_mm)
        if not m.any():
            return
        ax.fill_between(np.asarray(r["t_ms"])[m],
                        (y - BAND_SIGMAS * s_mm)[m],
                        (y + BAND_SIGMAS * s_mm)[m],
                        color=st.get("color", "0.5"), alpha=0.15, lw=0)

    for name, r in results.items():
        st = STYLE.get(name, {})
        _band(axR, r, st, "dR_m", "sigma_R_m")
        axR.plot(r["t_ms"], np.asarray(r["dR_m"]) * 1e3, label=r["label"], **st)
        
    
    axR.axhline(0, color="0.6", ls=":", lw=0.8)
    axR.set_ylabel("$\\Delta_R$ [mm]")
    axR.set_title(f"Shot {shot}: horizontal displacement by method")
    axR.legend(fontsize=9); axR.grid(alpha=0.3)
    axR.set_ylim(-200, 200)   # clamp to +-200 mm (limited minor radius); position.c can exceed this
    

    if have_dZ:
        axZ = axes[1, 0]
        for name, r in results.items():
            if r.get("dZ_m") is None:
                continue
            st = STYLE.get(name, {})
            _band(axZ, r, st, "dZ_m", "sigma_Z_m")
            axZ.plot(r["t_ms"], np.asarray(r["dZ_m"]) * 1e3, label=r["label"], **st)
        axZ.axhline(0, color="0.6", ls=":", lw=0.8)
        axZ.set_ylabel("$\\Delta_Z$ [mm]")
        axZ.legend(fontsize=9); axZ.grid(alpha=0.3)
        axZ.set_ylim(-200, 200)   # clamp to +-200 mm (limited minor radius)
        axZ.set_xlabel("time [ms]")
    else:
        axR.set_xlabel("time [ms]")

    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{shot}_compare.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{shot}] saved {png}  ({', '.join(results)})")
    if "position_c" in results:
        report_probe_health(shot)
    shape_amplitude_decomposition(results, ref="filament")


if __name__ == "__main__":
    shots = sys.argv[1:] or DEFAULT_SHOTS
    if not shots:
        sys.exit("usage: python compare_methods.py <shot> [<shot> ...]\n"
                 "       e.g. python compare_methods.py 1643 2766\n"
                 "       (or set DEFAULT_SHOTS at the top of this file)")
    if not sys.argv[1:]:
        print(f"[compare_methods] no shot given; using DEFAULT_SHOTS = "
              f"{' '.join(shots)}")
    for s in shots:
        compare_shot(s)
