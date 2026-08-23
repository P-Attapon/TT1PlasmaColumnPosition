"""Checks that must pass before the package is used on a shot.

    python -m methods_script.biot_savart.selftest

Stage 1  synthetic recovery: prescribe (dR, dZ), generate synthetic signals,
         add injected noise, invert, measure the error. Runs without the
         repository.
Stage 2  convention check: compare field.py against cal_signal over a grid of
         displacements and report which sign convention matches and whether the
         two agree at all. Needs the repository.
Stage 3  fold probe: invert noise-free synthetic signals from a grid of true
         positions and report where a different position fits equally well.
         Needs the repository.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from . import adapter, field, invert  # noqa: E402

# All 12 probes first, then four of the antipodal 4-probe sets.
FOLD_SETS = (None, [2, 3, 8, 9], [3, 4, 9, 10], [1, 4, 7, 10], [12, 3, 6, 9])


def _hdr(s):
    print("\n" + s + "\n" + "-" * len(s))


def stage1_synthetic(seed=0, noise_frac=0.01):
    _hdr("Stage 1 -- synthetic recovery")
    geom = adapter.load_geometry(strict=False)
    fwd = adapter.load_forward("internal", geom)
    fwd_many = adapter.load_forward_many(geom)
    rng = np.random.default_rng(seed)

    truth = np.array([(0.00, 0.00), (0.03, -0.02), (-0.05, 0.04),
                      (0.09, 0.06), (-0.12, -0.03), (0.15, 0.00)])
    sqrtw = np.ones(geom.angles.size)

    print(f"geometry: {geom.source}  R0={geom.R0} m  probe circle={geom.probe_radius} m")
    print(f"injected noise: {noise_frac:.1%} of the median |B|, gaussian, per probe\n")
    print("   dR true   dZ true |   dR fit    dZ fit |  err mm | resid | minima")

    worst = 0.0
    for dR, dZ in truth:
        clean = fwd(dR, dZ)
        scale = float(np.median(np.abs(clean)))
        meas = clean + rng.normal(0.0, noise_frac * scale, clean.size)
        x, amp, rn, nmin, spread, _, _ = invert.invert_sample(
            fwd, fwd_many, meas, sqrtw, scale, np.array([0.0, 0.0]),
            multistart=True)
        err = 1e3 * float(np.hypot(x[0] - dR, x[1] - dZ))
        worst = max(worst, err)
        print(f"  {dR:8.3f} {dZ:9.3f} | {x[0]:8.3f} {x[1]:9.3f} |"
              f" {err:7.2f} | {rn:5.3f} | {nmin}")

    ok = worst < 5.0
    print(f"\n  worst recovery error: {worst:.2f} mm  "
          f"-> {'PASS' if ok else 'FAIL'} (tolerance 5 mm at "
          f"{noise_frac:.1%} injected noise)")
    return ok


def stage2_conventions(n=9, span=0.10):
    _hdr("Stage 2 -- field.py vs cal_signal")
    try:
        geom = adapter.load_geometry(strict=True)
        cal = adapter.load_forward("cal_signal", geom)
    except Exception as exc:
        print(f"  SKIPPED -- repository not importable: {exc}")
        return None

    grid = [(a, b) for a in np.linspace(-span, span, n)
            for b in np.linspace(-span, span, n)]
    ref = np.array([cal(a, b) for a, b in grid])

    best = None
    print("  dz_sign  tan_sign |  median ratio  |  max shape dev.")
    for dz_sign in (-1.0, +1.0):
        for tan_sign in (-1.0, +1.0):
            old = (field.DZ_SIGN, field.TANGENTIAL_SIGN)
            field.DZ_SIGN, field.TANGENTIAL_SIGN = dz_sign, tan_sign
            mine = np.array([field.probe_signals(a, b, geom, I=geom.I_ref)
                             for a, b in grid])
            field.DZ_SIGN, field.TANGENTIAL_SIGN = old

            m = np.isfinite(ref) & np.isfinite(mine) & (np.abs(ref) > 0)
            if not m.any():
                continue
            ratio = mine[m] / ref[m]
            med = float(np.median(ratio))
            dev = float(np.max(np.abs(ratio / med - 1.0))) if med != 0 else np.inf
            print(f"  {dz_sign:+7.0f}  {tan_sign:+8.0f} | {med:14.6g} | {dev:14.3e}")
            # A global sign flip leaves the shape deviation unchanged, so break
            # the tie on the sign of the ratio.
            key = (dev, 0 if med > 0 else 1)
            if best is None or key < best[0]:
                best = (key, dz_sign, tan_sign, med)

    (dev, _), dz_sign, tan_sign, med = best
    print(f"\n  best match: DZ_SIGN={dz_sign:+.0f}  TANGENTIAL_SIGN={tan_sign:+.0f}")
    print(f"  ratio {med:.9f} at I_ref = {geom.I_ref:.4g} A; expected 1")
    print(f"  residual shape deviation: {dev:.3e}")
    if dev < 1e-6:
        print("  -> The two implementations agree.")
        if (dz_sign, tan_sign) != (field.DZ_SIGN, field.TANGENTIAL_SIGN):
            print("  -> ACTION: update the constants at the top of field.py.")
        return True
    print("  -> They disagree in shape, not just scale, so one has a geometry "
          "error. Check probe angle ordering and probe_radius first.")
    return False


def stage3_fold(span=0.20, n=11, probes=None):
    _hdr(f"Stage 3 -- fold probe, probes = {'all 12' if probes is None else probes}")
    try:
        geom = adapter.load_geometry(strict=True)
        full = adapter.load_forward("cal_signal", geom)
        full_many = adapter.load_forward_many(geom)
    except Exception as exc:
        print(f"  SKIPPED -- repository not importable: {exc}")
        return None

    if probes is None:
        fwd, fwd_many, m = full, full_many, geom.angles.size
    else:
        idx = np.asarray(probes, int) - 1
        fwd = lambda a, b: full(a, b)[idx]
        fwd_many = lambda a, b: full_many(a, b)[:, idx]
        m = idx.size
    sqrtw = np.ones(m)

    rows = []
    for dR in np.linspace(-span, span, n):
        for dZ in np.linspace(-span, span, n):
            if np.hypot(dR, dZ) > invert.DISPLACEMENT_BOUND:
                continue
            clean = fwd(dR, dZ)
            scale = float(np.median(np.abs(clean))) or 1.0
            x, amp, rn, nmin, spread, _, _ = invert.invert_sample(
                fwd, fwd_many, clean, sqrtw, scale, np.array([0.0, 0.0]),
                multistart=True)
            err = 1e3 * float(np.hypot(x[0] - dR, x[1] - dZ))
            # A recovered point that fits as well as the truth at a different
            # position is a degeneracy of the forward map. One that fits worse
            # is a solver failure. n_minima cannot separate these, because every
            # start can land in the same wrong root.
            degenerate = err > 1.0 and rn <= 1e-6
            rows.append((dR, dZ, nmin, err, float(rn), float(degenerate)))

    rows = np.array(rows)
    print(f"  probed {len(rows)} noise-free positions out to "
          f"|d| = {invert.DISPLACEMENT_BOUND} m, {m} probes")

    deg = rows[rows[:, 5] > 0]
    amb = rows[rows[:, 2] > 1]
    fail = rows[(rows[:, 3] > 1.0) & (rows[:, 5] == 0)]
    print(f"  degenerate (a different position fits equally well): {len(deg)}")
    print(f"  multi-start disagreement (n_minima > 1):             {len(amb)}")
    print(f"  solver failure (recovered fit is worse than truth):  {len(fail)}")

    hits = deg if len(deg) else amb
    if len(hits):
        r = np.hypot(hits[:, 0], hits[:, 1])
        print(f"\n  smallest |d| with an ambiguity: {r.min():.3f} m")
    else:
        print("\n  -> No ambiguity for this set.")
    return True


if __name__ == "__main__":
    ok1 = stage1_synthetic()
    ok2 = stage2_conventions()
    ok3 = None
    for _set in FOLD_SETS:
        r = stage3_fold(probes=_set)
        if r is None:
            break
        ok3 = r
    print("\n" + "=" * 60)
    print(f"stage 1 synthetic recovery : {'PASS' if ok1 else 'FAIL'}")
    print(f"stage 2 convention check   : "
          f"{'SKIPPED' if ok2 is None else ('PASS' if ok2 else 'FAIL')}")
    print(f"stage 3 fold probe         : {'SKIPPED' if ok3 is None else 'DONE'}")
