"""
position_c_displacement.py
==========================
Standalone reimplementation of the TT-1 real-time feedback controller's plasma
horizontal-position calculation (position.c), for OFFLINE use on the shot data
in the repo's data/<shot>/ folder. Drop this file next to main.py and run:

    python position_c_displacement.py 1641
    python position_c_displacement.py 1641 1643        (multiple shots)

For each shot it reads data/<shot>/, computes the position.c displacement, saves
    result_plot/position_c/<shot>_positionc.png      (displacement vs time)
    result_plot/position_c/<shot>_positionc.txt      (time_ms, displace_mm)
and prints a short summary.

--------------------------------------------------------------------------
WHAT position.c COMPUTES  (ProcessThreadData, lines ~325-329)
--------------------------------------------------------------------------
Two antipodal probes GBP1T (f2) and GBP7T (f3). After subtracting coil pickup
(toroidal IT1, ohmic IOH1, vertical IV2):

    displace = (f2 - f3) / (f2 + f3) * 29        [CENTIMETRES]   when Ip > 10 kA
    displace = 0                                                 otherwise

  * RADIAL (horizontal) displacement only - position.c has no vertical channel.
  * UNITS: position.c's native '* 29' output is in CENTIMETRES - the 29 is the
    probe radius P = 0.29 m written in cm. (Verified against data: flat-top ratio
    ~0.22 x 0.29 m ~ 63 mm, matching the filament amplitude; reading 29 as mm
    gives ~6 mm, ~10x too small.) THIS PORT scales by 290 instead of 29 so its
    output is in MILLIMETRES. Reference centre = major radius 0.65 m == the
    filament code's R0, so NO frame reconciliation is needed; divide by 1000 for
    metres.

--------------------------------------------------------------------------
UNITS NOTE (important, and why this is faithful rather than literal)
--------------------------------------------------------------------------
The repo's GBP*T.txt are the INTEGRATED probes, already in TESLA. position.c
instead works from raw ADC codes in its own internal volt-scale, and its pickup
constants (Kv, Koh, Kt) are tuned to THAT scale - they are NOT correct for
Tesla-valued signals. So we do NOT re-apply position.c's raw scaling or its
K-constants here. Instead we perform the SAME physical operation position.c
performs - subtract TF/OH/VF pickup from the two probes, then take the antipodal
ratio - using the pickup calibration that is correct for these Tesla signals,
namely the repo's own calibration_coeff (k1*, k7*). Because the final ratio
(f2-f3)/(f2+f3) is scale-invariant, the *29 -> mm scaling and the displacement
value are unaffected by the Tesla-vs-volt scale; only the pickup subtraction
must be in matching units, which this guarantees.

This reproduces position.c's ALGORITHM on correctly-calibrated data. The one
thing left convention-dependent is the overall SIGN (position.c flips all probe
signs internally and has a commented '/-10'); FLIP_SIGN below exposes it. Verify
against a known excursion or against the filament trace before trusting polarity.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- configuration -----------------------------
DATA_DIR = "data"                      # relative to this script / main.py
OUT_DIR = os.path.join("result_plot", "position_c")
IP_GATE = 10000.0                      # A; displace defined only when Ip > this
# Displacement scale = probe-circle radius. position.c uses '* 29', where 29 is
# the radius P = 0.29 m expressed in CENTIMETRES, so position.c's native output
# is in CENTIMETRES (verified against data: the dimensionless ratio ~0.22 in the
# flat-top x 0.29 m gives ~63 mm, matching the filament amplitude; reading '29'
# as mm would give ~6 mm, ~10x too small). We scale by 290 instead so THIS port
# outputs MILLIMETRES directly (0.29 m = 290 mm), consistent with the rest of the
# comparison. (Equivalent to position.c's *29 in cm, then x10 to mm.)
DISPLACE_SCALE = 290.0                 # mm  (= P=0.29 m as mm; position.c's *29 was cm)
# Sign convention. NOTE: the correct value depends on PICKUP_SOURCE below -- the
# two constant sets carry OPPOSITE polarity conventions, and it also varies by
# shot (see compare_methods.POSC_FLIP_SIGN for the measured truth table).
# FLIP_SIGN only negates; it never changes the spread, so it can correct a
# polarity but cannot rescue a diverging trace.
FLIP_SIGN = False                      # set True if the sign is inverted vs truth

# --- pickup-calibration source (the error-separation switch) ----------------
# The displacement is a small difference of large pickup-subtracted signals, so
# it is very sensitive to the pickup constants. This switch lets you separate
# the two possible error sources when comparing to the filament method:
#   "positionc" -> position.c's OWN constants (faithful controller reproduction)
#   "filament"  -> the filament repo's calibration_coeff for probes 1 and 7
#                  (the better-motivated, project-derived constants)
# If the two methods disagree with "positionc" but AGREE with "filament", the
# disagreement is dominated by the (stale) pickup CALIBRATION, not by the
# 2-probe displacement FORMULA. If they still disagree with "filament", the
# formula / probe-1&7 quality is the dominant source on that shot.
PICKUP_SOURCE = "positionc"    # "positionc" | "filament"

# position.c's OWN pickup constants, verbatim from position.c main() line 827.
# f2 -= Kv1*I2 + Koh1*f1 + Kt1*f0 ;  f3 -= Kv2*I2 + Koh2*f1 + Kt2*f0
_POSC = dict(KV1=-9.8951e-6, KV2=1.8448e-5,
             KOH1=-1.7987e-7, KOH2=5.7942e-7,
             KT1=4.1981e-8, KT2=6.7139e-8)

# filament repo calibration_coeff for probes 1 (f2) and 7 (f3), Tesla-consistent.
# Note the pickup model there is  f = GBP - kt*IT1 - koh*IOH1 - kv*IV2  applied to
# the UN-flipped, positive .txt signals; the wrapper below applies whichever set
# consistently. These are k1t/k1oh/k1v and k7t/k7oh/k7v from parameters.py.
_FIL = dict(KV1=9.82938e-6, KV2=-1.79673e-5,
            KOH1=-1.47234e-7, KOH2=7.16693e-7,
            KT1=1.30033e-7, KT2=-6.88199e-7)

def _pickup_constants():
    return _FIL if PICKUP_SOURCE == "filament" else _POSC

# position.c's I2 (Iv) carries a -500 scale that includes a sign; the repo's
# IV2.txt is already the physical current. This sign applies to the position.c
# constant set. (The filament calibration was derived against +IV2 directly, so
# when PICKUP_SOURCE="filament" the IV sign is folded into k*v already.)
IV_SIGN = -1.0
# -------------------------------------------------------------------------


def read_signal(path):
    """Read a TT-1 signal .txt (8-line header, then 'time_ms value' rows).
    Returns (t_ms, value) as float arrays."""
    t, v = [], []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # header lines contain '=' or start with a letter; data lines are numeric
            if "=" in line or (line[0].isalpha()):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    t.append(float(parts[0])); v.append(float(parts[1]))
                except ValueError:
                    continue
    return np.asarray(t), np.asarray(v)


def position_c_displacement(shot_dir):
    """Compute the position.c radial displacement (mm) for one shot directory.
    Faithful reproduction using position.c's own constants and sign conventions.
    Returns (t_ms, displace_mm, displace_plot_with_nan, Ip)."""
    def load(name):
        return read_signal(os.path.join(shot_dir, name + ".txt"))

    # DELIBERATELY NOT using methods_script/.../current_channels.py here.
    # This module is a FAITHFUL port of the real-time position.c controller,
    # which reads these specific channels off the DAQ. Substituting a healthy
    # redundant channel (as the filament path does) would make this port stop
    # reproducing what the actual controller computes, which is the whole point
    # of having it as a comparison baseline. If IV2 is dead on a shot, position.c
    # would ALSO have been wrong in real time -- and that is a result worth
    # seeing, not correcting away.
    t, Ip_txt = load("IP1")          # IP1.txt: physical plasma current (A), positive
    _, gbp1 = load("GBP1T")          # Tesla (== position.c f2 magnitude)
    _, gbp7 = load("GBP7T")          # Tesla (== position.c f3)
    _, it1 = load("IT1")             # == position.c f0
    _, ioh1 = load("IOH1")           # == position.c f1
    _, iv2 = load("IV2")             # == position.c I2 magnitude (Iv)

    n = min(len(t), len(Ip_txt), len(gbp1), len(gbp7), len(it1), len(ioh1), len(iv2))
    t, Ip = t[:n], Ip_txt[:n]
    gbp1, gbp7, it1, ioh1, iv2 = gbp1[:n], gbp7[:n], it1[:n], ioh1[:n], iv2[:n]

    K = _pickup_constants()

    if PICKUP_SOURCE == "filament":
        # filament convention: pickup subtracted from the POSITIVE .txt signals as
        #   f = GBP - kt*IT1 - koh*IOH1 - kv*IV2   (signs folded into k*)
        # then the antipodal ratio. No line-307 flip, no separate IV sign (the
        # filament k*v already multiplies +IV2). This matches how the filament
        # pipeline removes pickup, so agreement here isolates the FORMULA.
        f2 = gbp1 - K["KT1"] * it1 - K["KOH1"] * ioh1 - K["KV1"] * iv2
        f3 = gbp7 - K["KT2"] * it1 - K["KOH2"] * ioh1 - K["KV2"] * iv2
    else:
        # position.c convention: line-307 flip of f0,f1,f2,f3, IV_SIGN on Iv,
        # then subtract with position.c's own constants (faithful controller).
        f0 = -it1; f1 = -ioh1; f2 = -gbp1; f3 = -gbp7
        I2 = IV_SIGN * iv2
        f2 = f2 - K["KV1"] * I2 - K["KOH1"] * f1 - K["KT1"] * f0
        f3 = f3 - K["KV2"] * I2 - K["KOH2"] * f1 - K["KT2"] * f0

    with np.errstate(divide="ignore", invalid="ignore"):
        d = (f2 - f3) / (f2 + f3) * DISPLACE_SCALE
    displace = np.where(Ip > IP_GATE, d, 0.0)
    if FLIP_SIGN:
        displace = -displace
    displace_plot = np.where(Ip > IP_GATE, displace, np.nan)
    return t, displace, displace_plot, Ip


def process_shot(shot):
    shot_dir = os.path.join(DATA_DIR, str(shot))
    if not os.path.isdir(shot_dir):
        print(f"[skip] {shot_dir} not found")
        return
    t, displace, displace_plot, Ip = position_c_displacement(shot_dir)

    os.makedirs(OUT_DIR, exist_ok=True)
    # save text: time_ms, displace_mm (NaN where Ip <= gate)
    txt_path = os.path.join(OUT_DIR, f"{shot}_positionc.txt")
    np.savetxt(txt_path, np.column_stack([t, displace_plot]),
               fmt="%.6f", header="time_ms  displace_mm (position.c method; NaN where Ip<=10kA)")

    # plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, displace_plot, "C3-", lw=1.2, label="position.c $\\Delta_R$ (2-probe)")
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("time [ms]"); ax.set_ylabel("$\\Delta_R$ [mm]")
    ax.set_title(f"Shot {shot}: position.c horizontal displacement")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, f"{shot}_positionc.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    valid = np.isfinite(displace_plot)
    if valid.any():
        print(f"[{shot}] {valid.sum()} in-plasma samples | "
              f"displace range [{np.nanmin(displace_plot):.2f}, {np.nanmax(displace_plot):.2f}] mm | "
              f"median {np.nanmedian(displace_plot):.2f} mm")
    print(f"[{shot}] saved {png_path} and {txt_path}")


if __name__ == "__main__":
    shots = sys.argv[1:] or ["1641"]
    for s in shots:
        process_shot(s)
