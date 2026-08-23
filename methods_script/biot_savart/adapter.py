"""The only module that reads the rest of the repository.

Everything this package needs from `parameters.py`, `signal_strength.py` and
`adaptive_select.py` is resolved here, so an upstream API change breaks one file
with one message.

What is used, and how:

* `parameters.R0` (major radius), `parameters.R` (probe circle radius),
  `parameters.coil_angle_dict` ({probe -> poloidal angle, rad}),
  `parameters.I` (the current cal_signal is evaluated at).
* `signal_strength.cal_signal(dR, dZ, coil_angle)` -> list of signals, one per
  angle, for a filament carrying `parameters.I`.
* `adaptive_select._load_shot(shot_dir)` -> (t, ip, B) with B a dict
  {probe -> array}, calibrated and gated at Ip > 2500 A.
* `adaptive_select.shot_weights(shot, source)` -> {probe -> weight}.

Measured signals relate to the forward model through the same normalisation
`adaptive_select._proxy` uses: a filament at measured current `ip` produces
`cal_signal(...) * ip / I_ref`.
"""

from __future__ import annotations

import importlib
import os
import warnings
from dataclasses import dataclass, field as _dc_field

import numpy as np

PROBES = tuple(range(1, 13))


@dataclass(frozen=True)
class Geometry:
    R0: float                    # major radius, m
    probe_radius: float          # probe circle radius, m
    angles: np.ndarray           # poloidal angle per probe, rad, ordered by PROBES
    I_ref: float                 # current cal_signal is evaluated at, A
    probes: tuple = _dc_field(default=PROBES)
    source: str = ""


# Used only when parameters.py is unavailable, which happens when the package is
# run outside the repository (tests.py stage 1). Not authoritative.
_FALLBACK = dict(R0=0.65, probe_radius=0.321, I_ref=1.0e5,
                 angles=np.deg2rad(np.arange(12) * 30.0))


def _params(strict=True):
    for name in ("methods_script.toroidal_filament.parameters", "parameters"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    if strict:
        raise ImportError(
            "biot_savart: cannot import parameters.py. Run from the repository "
            "root, e.g.  python biot_savart_main.py")
    return None


def load_geometry(strict=True):
    """Machine geometry from parameters.py.

    `strict=False` falls back to placeholder values with a warning.
    """
    P = _params(strict)
    if P is None:
        warnings.warn("biot_savart: parameters.py not found; using placeholder "
                      "geometry, which is meaningless against real data.",
                      stacklevel=2)
        return Geometry(source="fallback", **_FALLBACK)

    missing = [n for n in ("R0", "R", "coil_angle_dict", "I") if not hasattr(P, n)]
    if missing:
        raise AttributeError(
            "biot_savart/adapter.py: parameters.py is missing "
            + ", ".join(missing) + ". Update load_geometry(). Available: "
            + ", ".join(sorted(n for n in dir(P) if not n.startswith("_"))))

    d = P.coil_angle_dict
    absent = [p for p in PROBES if p not in d]
    if absent:
        raise KeyError(f"biot_savart: coil_angle_dict has no entry for probes "
                       f"{absent}; expected 1..12.")

    return Geometry(R0=float(P.R0), probe_radius=float(P.R),
                    angles=np.array([float(d[p]) for p in PROBES]),
                    I_ref=float(P.I), probes=PROBES,
                    source="parameters.py (R0, R, coil_angle_dict, I)")


def load_forward(kind="cal_signal", geom=None):
    """Return `f(dR, dZ) -> ndarray(12,)`, predicted signals at the model current.

    kind="cal_signal" calls the repository's forward model directly.
    kind="internal"   calls field.py, which is vectorised and much faster.
    Both are checked against each other by selftest stage 2.
    """
    if geom is None:
        geom = load_geometry(strict=(kind != "internal"))

    if kind == "internal":
        from . import field
        return lambda dR, dZ: np.asarray(
            field.probe_signals(dR, dZ, geom, I=geom.I_ref), float)

    if kind != "cal_signal":
        raise ValueError(f"unknown forward model {kind!r}")

    S = None
    for name in ("methods_script.toroidal_filament.signal_strength",
                 "signal_strength"):
        try:
            S = importlib.import_module(name)
            break
        except Exception:
            continue
    if S is None or not hasattr(S, "cal_signal"):
        raise ImportError("biot_savart/adapter.py: cannot import "
                          "signal_strength.cal_signal. Run from the repo root.")

    fn, ang = S.cal_signal, geom.angles

    # Smoke-test the signature once, so a change is caught here rather than
    # thousands of samples into a run.
    try:
        probe = np.asarray(fn(0.0, 0.0, ang), float)
    except Exception as exc:
        raise TypeError(
            "biot_savart/adapter.py: cal_signal(0.0, 0.0, angles) failed. "
            f"Update load_forward().\n({exc})") from exc
    if probe.shape != ang.shape:
        raise ValueError(f"biot_savart: cal_signal returned shape "
                         f"{probe.shape}, expected {ang.shape}.")

    return lambda dR, dZ: np.asarray(fn(float(dR), float(dZ), ang), float)


def load_forward_many(geom=None):
    """Return `f(dR[], dZ[]) -> ndarray(N, 12)`, the batched forward model.

    Always field.py, since cal_signal takes scalars only.
    """
    from . import field
    if geom is None:
        geom = load_geometry(strict=True)
    return lambda dR, dZ: np.atleast_2d(
        field.probe_signals(dR, dZ, geom, I=geom.I_ref))


def load_shot(shot, data_dir="data"):
    """(t, ip, B) for a shot, with B an ndarray (T, 12) ordered by PROBES.

    Reads through `adaptive_select._load_shot`, so the calibration and pickup
    subtraction are the same ones the filament method uses. That function
    already gates at Ip > 2500 A.
    """
    try:
        A = importlib.import_module("adaptive_select")
    except Exception as exc:
        raise ImportError("biot_savart: cannot import adaptive_select. Run "
                          f"from the repository root.\n({exc})") from exc
    if not hasattr(A, "_load_shot"):
        raise AttributeError("biot_savart/adapter.py: adaptive_select has no "
                             "_load_shot. Update load_shot().")

    t, ip, B = A._load_shot(os.path.join(data_dir, str(shot)))
    if isinstance(B, dict):
        absent = [p for p in PROBES if p not in B]
        if absent:
            raise KeyError(f"biot_savart: shot {shot} has no field for probes "
                           f"{absent}.")
        B = np.column_stack([np.asarray(B[p], float) for p in PROBES])
    return np.asarray(t, float), np.asarray(ip, float), np.asarray(B, float)


def load_weights(shot, source="auto"):
    """Per-probe weights as an array over PROBES, or None for uniform.

    Probes absent from the returned dict default to 1.0, matching
    `adaptive_select._estimator`.
    """
    if source is None:
        return None
    try:
        A = importlib.import_module("adaptive_select")
        wdict = A.shot_weights(shot, source=source)
    except Exception as exc:
        warnings.warn(f"biot_savart: shot_weights({shot!r}, {source!r}) failed "
                      f"({exc}); using uniform probe weights.", stacklevel=2)
        return None
    if wdict is None:
        return None
    w = np.array([float(wdict.get(p, 1.0)) for p in PROBES])
    if not np.all(np.isfinite(w)) or not np.any(w > 0):
        warnings.warn("biot_savart: weights not usable; using uniform.",
                      stacklevel=2)
        return None
    return w
