"""
Build and load the 2D inverse map  Phi: (Dx, Dz) -> (horizontal, vertical) shift.

=============================================================================
ADDED FILE (not in the original Attapon et al. repository).
This is the 2D-interpolation counterpart of coefficient.py. Where
coefficient.py fits sliced 1D Taylor polynomials (alpha(Dz | est_R),
beta(Dx | est_Z)) that must be indexed by the PREVIOUS timestep's shift,
this module builds a single 2D map indexed by the CURRENT (Dx, Dz) only.
That removes the timestep-to-timestep recurrence and its error propagation.

Method (offline, accuracy-prioritised):
  1. sweep prescribed (horizontal, vertical) shift over the physical domain
     [-shift_domain, shift_domain]^2 on the same 1 mm grid the paper uses;
  2. forward-map each node with the EXACT model already in the repo
     (cal_signal -> cal_newton_DxDz) to get the (Dx, Dz) it produces;
  3. resample the scattered inverse samples onto a regular (Dx, Dz) grid;
  4. store as Phi_<probe_set>.npz. Runtime uses a bicubic spline over it.

The forward model, calibration and DxDz inversion are reused verbatim from
the original modules; only the inverse representation changes.
=============================================================================
"""
from .parameters import coil_angle_dict as angle_dict
from .parameters import shift_domain
from .DxDz import cal_newton_DxDz as cal_DxDz
from .cache_keys import forward_model_key, describe, digest
from .signal_strength import cal_signal
import numpy as np
import os

current_dir = os.path.dirname(__file__)
PHI_DIR = os.path.join(current_dir, "phi_tables")

# grid controls. PHYS_STEP matches the paper's 1 mm coefficient grid.
PHYS_STEP = 0.001          # m
UV_N = 401                 # regular (Dx,Dz) grid resolution per axis


def _grid_key():
    """Grid component of the legacy cache key.

    The legacy filename was Phi_<probe_set>.npz and encoded NOTHING else, so
    changing PHYS_STEP or UV_N silently reused a map built at the old resolution.
    The grid goes in the FILENAME (not the stamp) because, like the M-probe grid,
    it is a resolution choice someone may reasonably want to hold two of at once
    while comparing them.
    """
    return digest(f"step:{PHYS_STEP:.12g}|uvn:{UV_N:d}", 6)


def _phi_path(probe_set_str: str) -> str:
    return os.path.join(PHI_DIR,
                        f"Phi_{'_'.join(probe_set_str.split())}_{_grid_key()}.npz")


def build_phi(probe_set_str: str, uv_n: int = UV_N) -> str:
    """Build and save the 2D inverse map for one probe set. Returns the file path."""
    from scipy.interpolate import griddata

    probes = [int(p) for p in probe_set_str.split()]
    coil_angle = [angle_dict[p] for p in probes]

    grid = np.arange(-shift_domain, shift_domain + PHYS_STEP / 2, PHYS_STEP)
    n = len(grid)
    RR, ZZ = np.meshgrid(grid, grid, indexing="ij")
    DX = np.empty_like(RR)
    DZ = np.empty_like(RR)
    for i in range(n):
        for j in range(n):
            Dx, Dz = cal_DxDz(cal_signal(RR[i, j], ZZ[i, j], coil_angle), coil_angle)
            DX[i, j], DZ[i, j] = Dx, Dz

    # scattered inverse samples: (Dx, Dz) -> (R, Z)
    pts = np.column_stack([DX.ravel(), DZ.ravel()])
    valR = RR.ravel()
    valZ = ZZ.ravel()

    xg = np.linspace(DX.min(), DX.max(), uv_n)
    zg = np.linspace(DZ.min(), DZ.max(), uv_n)
    XG, ZG = np.meshgrid(xg, zg, indexing="ij")
    q = np.column_stack([XG.ravel(), ZG.ravel()])

    tabR = griddata(pts, valR, q, method="cubic")
    tabZ = griddata(pts, valZ, q, method="cubic")
    outside = np.isnan(tabR)
    if outside.any():   # nearest-fill beyond the invertible image (clamped region only)
        tabR[outside] = griddata(pts, valR, q[outside], method="nearest")
        tabZ[outside] = griddata(pts, valZ, q[outside], method="nearest")

    os.makedirs(PHI_DIR, exist_ok=True)
    out = _phi_path(probe_set_str)
    np.savez_compressed(out, xg=xg, zg=zg,
                        tabR=tabR.reshape(uv_n, uv_n),
                        tabZ=tabZ.reshape(uv_n, uv_n),
                        probe_set=probe_set_str,
                        fm_key=forward_model_key(),
                        model=describe())
    return out


class PhiMap:
    """Loaded 2D map with bicubic-spline evaluation. Built lazily on first use."""

    def __init__(self, probe_set_str: str):
        from scipy.interpolate import RectBivariateSpline
        path = _phi_path(probe_set_str)
        if not os.path.exists(path):
            build_phi(probe_set_str)
        d = np.load(path, allow_pickle=True)
        # Same provenance gate as mprobe._load_or_build_phi -- see cache_keys.py.
        stamp = str(d["fm_key"]) if "fm_key" in d.files else None
        if stamp is None:
            raise RuntimeError(
                f"Legacy Phi map {os.path.basename(path)} has no provenance "
                f"stamp (it predates cache stamping).\n"
                f"  If it WAS built with the current parameters ({describe()}), "
                f"run:\n"
                f"      python stamp_model_caches.py --apply\n"
                f"  Otherwise delete {PHI_DIR}/ and let the maps rebuild.")
        if stamp != forward_model_key():
            raise RuntimeError(
                f"Legacy Phi map {os.path.basename(path)} was built under a "
                f"DIFFERENT forward model.\n"
                f"  stamp   : {stamp}\n"
                f"  current : {forward_model_key()}  ({describe()})\n"
                f"Delete {PHI_DIR}/ and let the maps rebuild.")
        self.xg, self.zg = d["xg"], d["zg"]
        # bicubic (kx=ky=3): map is smooth and offline, so accuracy is free
        self._sR = RectBivariateSpline(self.xg, self.zg, d["tabR"], kx=3, ky=3)
        self._sZ = RectBivariateSpline(self.xg, self.zg, d["tabZ"], kx=3, ky=3)

    def evaluate(self, Dx: float, Dz: float):
        """(Dx, Dz) -> (horizontal_shift, vertical_shift).

        BOUNDARY POLICY: FLAG, PER AXIS. R depends mainly on Dx and Z mainly on
        Dz, so each output is flagged independently: R -> NaN only when Dx is
        outside the table's Dx range; Z -> NaN only when Dz is outside the Dz
        range. This is deliberately NOT all-or-nothing: flagging both whenever
        EITHER axis is out of box would wrongly discard a perfectly good R
        (in-range Dx) just because Dz left its range - which for some probe sets
        happens every timestep and empties the whole plot.

        The spline still needs a finite second coordinate to evaluate, so for the
        axis that is out of range we pass the clamped coordinate into the spline
        for the OTHER axis's output (minor cross-term distortion) but return NaN
        for the out-of-range axis itself.

        Bounds are the table's own axes (self.xg, self.zg); the Dx proxy range is
        offset (not centred on zero) - do not compare against +/-0.1.
        """
        x_lo, x_hi = float(self.xg[0]), float(self.xg[-1])
        z_lo, z_hi = float(self.zg[0]), float(self.zg[-1])
        x_in = x_lo <= Dx <= x_hi
        z_in = z_lo <= Dz <= z_hi

        # clamped coords, used only to keep the spline evaluable for the in-range axis
        xc = min(max(Dx, x_lo), x_hi)
        zc = min(max(Dz, z_lo), z_hi)

        R = float(self._sR(xc, zc)[0, 0]) if x_in else float("nan")
        Z = float(self._sZ(xc, zc)[0, 0]) if z_in else float("nan")
        return R, Z


# cache one PhiMap per probe set (built on demand, reused across timesteps)
_phi_cache: dict = {}


def get_phi(probe_set_str: str) -> "PhiMap":
    if probe_set_str not in _phi_cache:
        _phi_cache[probe_set_str] = PhiMap(probe_set_str)
    return _phi_cache[probe_set_str]
