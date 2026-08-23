"""Exact Biot-Savart field of a displaced circular filament.

Numerically identical to `signal_strength.cal_signal` (verified by
`selftest.stage2_conventions`) but vectorised over displacement points, so the
solver can evaluate a whole finite-difference stencil in one call.

Units are SI throughout: metres, Tesla, Amperes.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

MU0 = 4.0e-7 * np.pi

# Vertical separation between a probe at height Z_p and a filament at height dZ
# is Z_p + DZ_SIGN * dZ. DZ_SIGN = -1 matches cal_signal, which places the probe
# at R*sin(phi) - dZ.
DZ_SIGN = -1.0

# A tangential probe at poloidal angle theta measures the field along
# (-sin theta, +cos theta) in the (R, Z) plane. TANGENTIAL_SIGN = +1 matches
# cal_signal's -Br*sin(phi) + Bz*cos(phi) projection.
TANGENTIAL_SIGN = +1.0


def loop_field(a_f, R, Z, I=1.0):
    """(Br, Bz) at (R, Z) from a circular loop of radius `a_f` in the plane Z=0.

    All arguments broadcast. `R` must be positive and no point may lie on the
    wire (a_f == R and Z == 0), where the field diverges.
    """
    a_f = np.asarray(a_f, float)
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)

    sum_sq = (a_f + R) ** 2 + Z ** 2
    dif_sq = (a_f - R) ** 2 + Z ** 2
    m = 4.0 * a_f * R / sum_sq

    # scipy's ellipk/ellipe take the parameter m = k^2, not the modulus k.
    K = ellipk(m)
    E = ellipe(m)

    pref = MU0 * I / (2.0 * np.pi)
    root = np.sqrt(sum_sq)

    Br = pref * (Z / (R * root)) * (-K + (a_f ** 2 + R ** 2 + Z ** 2) / dif_sq * E)
    Bz = pref * (1.0 / root) * (K + (a_f ** 2 - R ** 2 - Z ** 2) / dif_sq * E)
    return Br, Bz


def probe_signals(dR, dZ, geom, I=1.0):
    """Tangential field at every probe for a filament displaced by (dR, dZ).

    Scalar (dR, dZ) returns an (M,) array; length-N arrays return (N, M).
    `geom` is an adapter.Geometry.
    """
    th = np.asarray(geom.angles, float)
    R_p = geom.R0 + geom.probe_radius * np.cos(th)
    Z_p = geom.probe_radius * np.sin(th)

    dR = np.asarray(dR, float)
    dZ = np.asarray(dZ, float)
    scalar = (dR.ndim == 0)

    a_f = (geom.R0 + dR).reshape(-1, 1)
    zeta = Z_p[None, :] + DZ_SIGN * dZ.reshape(-1, 1)

    Br, Bz = loop_field(a_f, R_p[None, :], zeta, I=I)
    out = TANGENTIAL_SIGN * (-Br * np.sin(th)[None, :] + Bz * np.cos(th)[None, :])
    return out[0] if scalar else out
