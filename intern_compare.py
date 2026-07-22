"""
intern_compare.py - controlled accuracy comparison of the 1D (paper) vs
2D (new) displacement methods.

Idea: on a real shot the true plasma position is unknown, so we cannot measure
accuracy directly. Instead we INVENT a known position trajectory, compute the
probe signals it would produce using the exact magnetic-field model, then ask
each method to recover the position from those signals. The method whose
recovered position is closer to the invented truth is more accurate.

Both methods get identical simulated signals, so the comparison is fair; the
only thing that differs is how each one inverts signals -> position.

Run:  python intern_compare.py
Output: an error table in the terminal + overlay plots in intern_compare_plots/
See INTERN_GUIDE.md, Part 3, for how to read the results.
"""
import sys, os, types
try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    _t = types.ModuleType("tqdm")
    _t.tqdm = lambda it=None, **kw: it if it is not None else (lambda x: x)
    sys.modules["tqdm"] = _t

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from methods_script.toroidal_filament.parameters import coil_angle_dict, R0, R, shift_domain
from methods_script.toroidal_filament.signal_strength import coil_signal
from methods_script.toroidal_filament.DxDz import cal_newton_DxDz
from methods_script.toroidal_filament.plasma_shift import cal_shift, cal_shift_1d

# ---- config the intern may edit ----
PROBES = [1, 4, 7, 10]     # if you change this, run: python build_all_phi.py "<set>"
OUT = "intern_compare_plots"
# ------------------------------------

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
PROBE_STR = " ".join(map(str, PROBES))


def simulate_signals(dR_all, dZ_all, noise=0.0):
    """Known (dR,dZ) -> probe signals, via the exact field model. Optional noise."""
    sig = np.empty((len(dR_all), len(PROBES)))
    for t, (dR, dZ) in enumerate(zip(dR_all, dZ_all)):
        for j, p in enumerate(PROBES):
            phi = coil_angle_dict[p]
            r_probe = R0 + R * np.cos(phi)
            z_probe = R * np.sin(phi) - dZ
            a_f = R0 + dR
            sig[t, j] = coil_signal(phi, r_probe, z_probe, a_f)
    if noise > 0:
        sig += noise * np.abs(sig).mean() * rng.standard_normal(sig.shape)
    return sig


def recover_1d(signals):
    """Paper method, WITH its previous-step recurrence."""
    dR = np.empty(len(signals)); dZ = np.empty(len(signals))
    eR = eZ = 0.0
    for i, s in enumerate(signals):
        eR = float(np.clip(eR, -shift_domain, shift_domain))
        eZ = float(np.clip(eZ, -shift_domain, shift_domain))
        ((r, _), (z, _)) = cal_shift_1d(cal_newton_DxDz, 3, list(s), eR, eZ, PROBES)
        dR[i], dZ[i] = r, z
        eR, eZ = r, z
    return dR, dZ


def recover_2d(signals):
    """New method, 2D map, current measurement only."""
    dR = np.empty(len(signals)); dZ = np.empty(len(signals))
    for i, s in enumerate(signals):
        ((r, _), (z, _)) = cal_shift(cal_newton_DxDz, 3, list(s), 0.0, 0.0, PROBES)
        dR[i], dZ[i] = r, z
    return dR, dZ


def make_trajectories(n=200):
    t = np.linspace(0, 1, n)
    trajs = {}
    trajs["fast_wiggle"] = (0.08 * np.sin(2 * np.pi * 3 * t),
                            0.06 * np.cos(2 * np.pi * 2 * t))
    rB = np.zeros(n); rB[n // 3:] = 0.06; rB[2 * n // 3:] = -0.04
    zB = np.zeros(n); zB[n // 2:] = 0.05
    trajs["step"] = (rB, zB)
    trajs["slow_ramp"] = (np.linspace(-0.05, 0.05, n),
                          np.linspace(0.03, -0.03, n))
    return t, trajs


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def run_case(noise, tag):
    t, trajs = make_trajectories()
    print(f"\n=== {tag}  (probe set {PROBE_STR}, noise={noise}) ===")
    print(f"{'trajectory':12s} {'axis':4s} {'1D RMS mm':>10s} {'2D RMS mm':>10s} "
          f"{'1D max mm':>10s} {'2D max mm':>10s}")
    fig, axes = plt.subplots(len(trajs), 2, figsize=(12, 3 * len(trajs)), sharex=True)
    for row, (name, (dR_true, dZ_true)) in enumerate(trajs.items()):
        sig = simulate_signals(dR_true, dZ_true, noise=noise)
        r1, z1 = recover_1d(sig)
        r2, z2 = recover_2d(sig)
        for axis, true, v1, v2 in [("dR", dR_true, r1, r2), ("dZ", dZ_true, z1, z2)]:
            e1 = (v1 - true) * 1e3; e2 = (v2 - true) * 1e3
            print(f"{name:12s} {axis:4s} {rms(e1):10.3f} {rms(e2):10.3f} "
                  f"{np.abs(e1).max():10.3f} {np.abs(e2).max():10.3f}")
        for col, (true, v1, v2, lab) in enumerate(
                [(dR_true, r1, r2, "dR"), (dZ_true, z1, z2, "dZ")]):
            ax = axes[row, col]
            ax.plot(t, true * 1e3, "k-", lw=3, alpha=0.4, label="truth")
            ax.plot(t, v1 * 1e3, "C1--", lw=1.3, label="1D (paper)")
            ax.plot(t, v2 * 1e3, "C0-", lw=1.0, label="2D (new)")
            if row == 0:
                ax.set_title(lab)
            if col == 0:
                ax.set_ylabel(f"{name}\n[mm]", fontsize=9)
            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"truth vs recovered  ({tag}, noise={noise})")
    fig.tight_layout()
    p = os.path.join(OUT, f"compare_{tag}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


if __name__ == "__main__":
    run_case(0.0, "clean")
    run_case(0.01, "noise_1pct")
    run_case(0.03, "noise_3pct")
    print("\nSee INTERN_GUIDE.md Part 3 for how to interpret these.")
