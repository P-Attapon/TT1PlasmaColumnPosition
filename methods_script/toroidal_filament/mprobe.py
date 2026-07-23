"""
M-probe weighted displacement estimator with optional plasma-current fitting.

=============================================================================
ADDED FILE (not in the original Attapon et al. repository, nor in the first
2D modification). Generalizes the displacement calculation from the fixed
4-probe antipodal-pair inversion to any M >= 2 probes with per-probe weights.

Model (first order around the centred plasma, derived NUMERICALLY from the
repo's own exact forward model cal_signal, so all sign / orientation
conventions are inherited automatically):

    S_i(dR, dZ) ~ S0_i + hU_i * dU + hV_i * dV        (model current I_param)

where S0_i = cal_signal(0,0) at probe i, and hU_i, hV_i are central finite
differences of cal_signal wrt the two displacements. (dU, dV) is the LINEAR
proxy displacement; the 2D map Phi corrects it to the true (dR, dZ), exactly
as in the paper / 2D method.

Measured fields B_i at plasma current Ip relate to the model by the current
ratio kappa = Ip / I_param:   B_i ~ kappa * (S0_i + hU_i dU + hV_i dV).

Two modes (the I0 switch):
  fit_ip = False  (measured current): kappa is known from the measured Ip.
      Solve the weighted 2-unknown least squares
          y_i = B_i/kappa - S0_i  =  hU_i dU + hV_i dV
      via the precomputed 2xM pseudo-inverse P = (H^T W H)^-1 H^T W.
  fit_ip = True   (current as 3rd unknown, for cross-checking):
      Solve the weighted 3-unknown least squares for x = (kappa, kappa dU,
      kappa dV):  B_i = S0_i x1 + hU_i x2 + hV_i x3, then dU = x2/x1,
      dV = x3/x1, and fitted Ip = x1 * I_param (returned for cross-check).

Weights w_i are fixed per shot (curation input). A probe with w_i = 0 is
excluded. The per-shot condition number of the normal matrix is reported as
a health check. The estimate covariance (H^T W H)^-1 is available.

The Phi map for a given (probes, weights, mode) configuration is built on
first use by sweeping the physical domain through cal_signal and the SAME
estimator, then resampled and spline-fitted exactly like phi_map.py. Cached
to methods_script/toroidal_filament/phi_tables/PhiM_<hash>.npz.
Boundary policy: per-axis FLAG (NaN), matching phi_map.PhiMap.evaluate.
=============================================================================
"""

"""
Within the first order of Taylor series, according to Attapon et al., the magnetic field at each
magnetic probes can be written as: 

B_theta \approx mu * I0 / (2*pi*ap) * [1+1/a * (cos(theta) * dU + sin(theta)* dV)]. (Equation 4 in paper)

Here you see that I0 is not linearly independent of dU and dV. I don't understand how you manange to solve
for dU, dV, I0, with linear least square. Please give proper derivation.
"""
import os
import hashlib
import numpy as np

from .parameters import coil_angle_dict, shift_domain, I as I_PARAM
from .signal_strength import cal_signal

_HERE = os.path.dirname(__file__)
PHI_DIR = os.path.join(_HERE, "phi_tables")

FD_STEP = 1e-4       # m, finite-difference step for the linear model
PHYS_STEP = 0.001    # m, physical grid step for Phi construction (paper convention)
UV_N = 401           # regular proxy-grid resolution for Phi


class MProbeEstimator:
    """Weighted M-probe linear estimator + 2D correction map.

    probes  : list of probe numbers (any M >= 2), e.g. [1,2,3,...,12]finite
    weights : list of per-probe weights (same length), or None -> all 1.0
    fit_ip  : False = use measured plasma current (2 unknowns),
              True  = fit current as 3rd unknown (cross-check mode)
    """

    def __init__(self, probes, weights=None, fit_ip=False, gains=None):
        self.probes = list(probes)
        M = len(self.probes)
        if M < 2:
            raise ValueError("need at least 2 probes")
        self.weights = np.ones(M) if weights is None else np.asarray(weights, float)
        if len(self.weights) != M:
            raise ValueError("weights length must match probes length")
        if np.count_nonzero(self.weights) < (3 if fit_ip else 2):
            raise ValueError("not enough non-zero-weight probes for the chosen mode")
        self.fit_ip = bool(fit_ip)
        # ADDED gains: per-probe multiplicative calibration factors g_i such that
        # B_measured_i = g_i * B_physical_i. Measured signals are divfiniteided by g_i
        # before use (a negative g_i corrects a polarity-flipped probe). None ->
        # all 1.0. This is the curation hook for absolute-gain calibration, which
        # the antipodal-ratio method does not need but an absolute-field method does.
        self.gains = np.ones(M) if gains is None else np.asarray(gains, float)
        if len(self.gains) != M or np.any(self.gains == 0):
            raise ValueError("gains must match probes length and be non-zero")
        self.angles = [coil_angle_dict[p] for p in self.probes]

        # ---- linear model from the repo's own forward model (numerical) ----
        #
        #NOTE: How does this relate to Ip?
        self.S0 = np.asarray(cal_signal(0.0, 0.0, self.angles), float)

        #NOTE: Why does this not match the analytical result (Equation 4 in paper)
        hU = (np.asarray(cal_signal(+FD_STEP, 0.0, self.angles)) -
              np.asarray(cal_signal(-FD_STEP, 0.0, self.angles))) / (2 * FD_STEP)
        hV = (np.asarray(cal_signal(0.0, +FD_STEP, self.angles)) -
              np.asarray(cal_signal(0.0, -FD_STEP, self.angles))) / (2 * FD_STEP)
        self.H2 = np.column_stack([hU, hV])                    # M x 2 (measured-Ip mode)
        self.H3 = np.column_stack([self.S0, hU, hV])           # M x 3 (fit-Ip mode)

        W = np.diag(self.weights)
        if self.fit_ip:
            A = self.H3.T @ W @ self.H3                        # 3x3 normal matrix
            self.P = np.linalg.solve(A, self.H3.T @ W)         # 3xM pseudo-inverse
        else:
            A = self.H2.T @ W @ self.H2                        # 2x2 normal matrix
            self.P = np.linalg.solve(A, self.H2.T @ W)         # 2xM pseudo-inverse
        self.cond = float(np.linalg.cond(A))                   # per-shot health check
        self.cov = np.linalg.inv(A)                            # estimate covariance

        self._load_or_build_phi()

    # ------------------------------------------------------------------ Phi
    def _config_hash(self):
        s = ("M:" + " ".join(map(str, self.probes))
             + "|w:" + " ".join(f"{w:.6g}" for w in self.weights)
             + "|fit_ip:" + str(self.fit_ip)
             + "|g:" + " ".join(f"{g:.6g}" for g in self.gains))
        return hashlib.md5(s.encode()).hexdigest()[:10]

    def _linear_estimate_model(self, sig):
        """Proxy (dU, dV) for MODEL-current signals (kappa = 1). Used for Phi build
        and, after current normalization, at runtime."""
        if self.fit_ip:
            x = self.P @ sig
            return x[1] / x[0], x[2] / x[0]
        y = sig - self.S0
        d = self.P @ y
        return d[0], d[1]

    def _load_or_build_phi(self):
        from scipy.interpolate import RectBivariateSpline
        os.makedirs(PHI_DIR, exist_ok=True)
        path = os.path.join(PHI_DIR, f"PhiM_{self._config_hash()}.npz")
        if not os.path.exists(path):
            grid = np.arange(-shift_domain, shift_domain + PHYS_STEP / 2, PHYS_STEP)
            n = len(grid)
            UU = np.empty((n, n)); VV = np.empty((n, n))
            for i, dR in enumerate(grid):
                for j, dZ in enumerate(grid):
                    sig = np.asarray(cal_signal(dR, dZ, self.angles), float)
                    UU[i, j], VV[i, j] = self._linear_estimate_model(sig)
            from scipy.interpolate import griddata
            pts = np.column_stack([UU.ravel(), VV.ravel()])
            RR, ZZ = np.meshgrid(grid, grid, indexing="ij")
            ug = np.linspace(UU.min(), UU.max(), UV_N)
            vg = np.linspace(VV.min(), VV.max(), UV_N)
            UG, VG = np.meshgrid(ug, vg, indexing="ij")
            q = np.column_stack([UG.ravel(), VG.ravel()])
            tabR = griddata(pts, RR.ravel(), q, method="cubic")
            tabZ = griddata(pts, ZZ.ravel(), q, method="cubic")
            out = np.isnan(tabR)
            if out.any():
                tabR[out] = griddata(pts, RR.ravel(), q[out], method="nearest")
                tabZ[out] = griddata(pts, ZZ.ravel(), q[out], method="nearest")
            np.savez_compressed(path, ug=ug, vg=vg,
                                tabR=tabR.reshape(UV_N, UV_N),
                                tabZ=tabZ.reshape(UV_N, UV_N))
        d = np.load(path)
        self.ug, self.vg = d["ug"], d["vg"]
        self._sR = RectBivariateSpline(self.ug, self.vg, d["tabR"], kx=3, ky=3)
        self._sZ = RectBivariateSpline(self.ug, self.vg, d["tabZ"], kx=3, ky=3)

    def _phi(self, dU, dV):
        """Per-axis flagged Phi evaluation (same policy as phi_map.PhiMap)."""
        u_in = self.ug[0] <= dU <= self.ug[-1]
        v_in = self.vg[0] <= dV <= self.vg[-1]
        uc = min(max(dU, self.ug[0]), self.ug[-1])
        vc = min(max(dV, self.vg[0]), self.vg[-1])
        R = float(self._sR(uc, vc)[0, 0]) if u_in else float("nan")
        Z = float(self._sZ(uc, vc)[0, 0]) if v_in else float("nan")
        return R, Z

    # -------------------------------------------------------------- runtime
    def shift(self, signal, Ip):
        """One timestep.

        signal : calibrated fields B_i (Tesla) for self.probes, in order
        Ip     : measured plasma current (A) at this timestep

        Returns (dR, dZ, Ip_used):
          dR, dZ  : corrected displacement (NaN where proxy leaves Phi domain)
          Ip_used : the measured Ip (fit_ip=False) or the FITTED current
                    (fit_ip=True) - the cross-check output.
        """
        sig = np.asarray(signal, float) / self.gains   # ADDED: gain/polarity correction
        if self.fit_ip:
            x = self.P @ sig
            dU, dV = x[1] / x[0], x[2] / x[0]
            Ip_used = float(x[0] * I_PARAM)
        else:
            kappa = Ip / I_PARAM
            y = sig / kappa - self.S0
            d = self.P @ y
            dU, dV = float(d[0]), float(d[1])
            Ip_used = float(Ip)
        dR, dZ = self._phi(dU, dV)
        return dR, dZ, Ip_used
