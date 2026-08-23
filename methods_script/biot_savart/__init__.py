"""Biot-Savart nonlinear least-squares displacement for TT-1.

Fits (dR, dZ) directly to the probe signals against the exact filament field,
with no linear proxy, polynomial, Phi table or validity domain.

The method shares the probe calibration, the vacuum-field subtraction and the
single-filament ansatz with the Phi path; see README.md for what that means for
interpreting agreement between the two.

    from methods_script.biot_savart import run
    res = run(1641)          # dict of arrays, SI units
"""

from .cli import run, summarise                                   # noqa: F401
from .invert import invert_sample, invert_shot                    # noqa: F401

__all__ = ["run", "summarise", "invert_sample", "invert_shot"]
