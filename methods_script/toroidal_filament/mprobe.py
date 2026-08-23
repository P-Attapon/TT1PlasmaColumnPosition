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
import os
import hashlib
import numpy as np

from .parameters import coil_angle_dict, shift_domain, I as I_PARAM, R, mu
from .cache_keys import forward_model_key, describe
from .signal_strength import cal_signal

_HERE = os.path.dirname(__file__)
PHI_DIR = os.path.join(_HERE, "phi_tables")

FD_STEP = 1e-4       # m, (unused since coefficients are analytic; kept for reference)
PHYS_STEP = 0.0005   # m, physical grid step for Phi construction. SINGLE SOURCE OF
                     # TRUTH for the grid: main.py, compare_methods.py and
                     # adaptive_select all use THIS value (they do not expose
                     # their own copy). The Phi cache is keyed on the grid, so this
                     # value defines which cached maps are valid -- change it only
                     # deliberately (it invalidates every cached PhiM_*.npz).
                     # This single parameter sets BOTH grids: the (dR,dZ) sweep is
                     # regular at PHYS_STEP, and the (dU,dV) lookup grid is derived
                     # from it (see UV_N below) so the two resolutions stay matched.
                     # (0.5 mm; the paper convention is 1 mm, but real-data accuracy
                     # is flat vs this since grid error << noise floor.)
# The regular (dU,dV) lookup grid is built by resampling the scattered forward-
# mapped points, of which there are exactly (domain / PHYS_STEP + 1) per axis.
# UV_OVERSAMPLE nudges the lookup grid finer than that source count for the
# unevenly-sampled edge regions.
UV_OVERSAMPLE = 2.0
def _uv_n(phys_step, uv_oversample=None):
    ovs = UV_OVERSAMPLE if uv_oversample is None else float(uv_oversample)
    n_sweep = int(round(2 * shift_domain / phys_step)) + 1   # per-axis sweep count
    return max(11, int(round(n_sweep * ovs)))


_ALL_PROBES = list(range(1, 13))
_FWD_CACHE = {}


def _forward_table(phys_step):
    """cal_signal over the Phi grid for ALL 12 probes: (n*n, 12).

    Shot-independent, weights-independent, probe-set-independent -- it is a
    property of the forward model alone. Keyed on forward_model_key() so a
    recalibration or a shift_domain change invalidates it, and memoised per
    process. This is the expensive part of a Phi build, so sharing it is what
    keeps a per-shot rebuild of every candidate set cheap.
    """
    key = (forward_model_key(), float(phys_step))
    if key in _FWD_CACHE:
        return _FWD_CACHE[key]
    os.makedirs(PHI_DIR, exist_ok=True)
    h = hashlib.md5(("fwd|" + "|".join(map(str, key))).encode()).hexdigest()[:10]
    path = os.path.join(PHI_DIR, f"FwdTab_{h}.npz")
    if os.path.exists(path):
        d = np.load(path)
        if str(d["fm_key"]) == forward_model_key():
            TAB = d["tab"]
            _FWD_CACHE[key] = TAB
            return TAB
    ang = [coil_angle_dict[p] for p in _ALL_PROBES]
    grid = np.arange(-shift_domain, shift_domain + phys_step / 2, phys_step)
    TAB = np.empty((len(grid) * len(grid), 12))
    k = 0
    for dR in grid:
        for dZ in grid:
            TAB[k] = np.asarray(cal_signal(dR, dZ, ang), float)
            k += 1
    np.savez_compressed(path, tab=TAB, fm_key=forward_model_key())
    _FWD_CACHE[key] = TAB
    return TAB


class MProbeEstimator:
    """Weighted M-probe linear estimator + 2D correction map.

    probes  : list of probe numbers (any M >= 2), e.g. [1,2,3,...,12]
    weights : list of per-probe weights (same length), or None -> all 1.0
    fit_ip  : False = use measured plasma current (2 unknowns),
              True  = fit current as 3rd unknown (cross-check mode)
    """

    def __init__(self, probes, weights=None, fit_ip=False, gains=None,
                 phys_step=None, uv_oversample=None):
        self.probes = list(probes)
        M = len(self.probes)
        if M < 2:
            raise ValueError("need at least 2 probes")
        # grid resolution for the Phi build (None -> module defaults). Kept per
        # instance so they can be set from main.py rather than edited here.
        self.phys_step = PHYS_STEP if phys_step is None else float(phys_step)
        self.uv_oversample = UV_OVERSAMPLE if uv_oversample is None else float(uv_oversample)
        self.weights = np.ones(M) if weights is None else np.asarray(weights, float)
        if len(self.weights) != M:
            raise ValueError("weights length must match probes length")
        if np.count_nonzero(self.weights) < (3 if fit_ip else 2):
            raise ValueError("not enough non-zero-weight probes for the chosen mode")
        self.fit_ip = bool(fit_ip)
        # ADDED gains: per-probe multiplicative calibration factors g_i such that
        # B_measured_i = g_i * B_physical_i. Measured signals are divided by g_i
        # before use (a negative g_i corrects a polarity-flipped probe). None ->
        # all 1.0. This is the curation hook for absolute-gain calibration, which
        # the antipodal-ratio method does not need but an absolute-field method does.
        self.gains = np.ones(M) if gains is None else np.asarray(gains, float)
        if len(self.gains) != M or np.any(self.gains == 0):
            raise ValueError("gains must match probes length and be non-zero")
        self.angles = [coil_angle_dict[p] for p in self.probes]

        # ---- linear model: analytic first-order coefficients (Eq. 4 of the paper) ----
        # B_theta ~= (mu*I)/(2*pi*a_p) * [ 1 + (cos(theta)*dU + sin(theta)*dV)/a_p ],
        # with the probe circle radius a_p = R. Reading off per probe i:
        #   S0_i = mu*I/(2*pi*R)                     centred-plasma field
        #   hU_i = mu*I*cos(theta_i)/(2*pi*R^2)      sensitivity to cylindrical dU
        #   hV_i = mu*I*sin(theta_i)/(2*pi*R^2)      sensitivity to cylindrical dV
        # Closed form, so no finite-difference truncation error. These are the
        # straight-line (infinite-R0) cylinder coefficients; (dU, dV) is therefore
        # the cylindrical proxy displacement, and Phi maps it to the true (dR, dZ).
        #
        # Current handling: all three coefficients carry the model current I
        # (=I_PARAM). They are NOT re-derived per timestep. The time-varying measured
        # current enters only through kappa = Ip/I_PARAM at runtime, which rescales the
        # measurement onto the model-current footing before S0 is subtracted. This is
        # why S0/hU/hV can be built once even though Ip = Ip(t).
        #
        # Why I0 can be fit by *linear* least squares despite Eq. 4 being bilinear in
        # (I0, dU, dV): we do not solve for (I0, dU, dV). We solve the linear system
        # for the products x = (I0, I0*dU, I0*dV) [columns S0, hU, hV], which is linear,
        # then recover dU = x2/x1, dV = x3/x1, I0 = x1 by division afterwards.
        # Sign convention: cal_signal (the forward model everything else uses)
        # returns a NEGATIVE tangential field for positive plasma current under
        # its -b_r*sin+ b_z*cos projection. The bare Eq. 4 prefactor mu*I/(2*pi*R)
        # is positive, which would make the fitted I0 come out with the wrong
        # (negative) sign while leaving dR,dZ correct (they are ratios in which the
        # sign cancels). We therefore match cal_signal's convention with a leading
        # minus so that a positive Ip yields a positive fitted I0.
        theta = np.asarray(self.angles, float)
        pref = -mu * I_PARAM / (2.0 * np.pi * R)
        self.S0 = pref * np.ones(len(theta))
        hU = pref * np.cos(theta) / R
        hV = pref * np.sin(theta) / R
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

        # Phi is built LAZILY. P, S0, cond, cov and the convex hull are all
        # closed form in the probe angles and weights -- no forward model needed
        # -- so anything that only wants those (set screening, hull tests, cache
        # inspection) pays nothing here. Phi is built on first use; call
        # ensure_phi() to force it.
        self._phi_ready = False

    # ------------------------------------------------------------------ Phi
    def _config_hash(self):
        """Cache key for this estimator's Phi map -- ESTIMATOR configuration only.

        Probes, weights, fit_ip and gains all enter the pseudo-inverse P, and the
        sweep calls _linear_estimate_model (hence P) at every grid point;
        phys_step/uv_oversample set the grid itself. All of it belongs in the
        FILENAME, because you legitimately hold maps for several probe sets at
        once and want them side by side.

        FORWARD-MODEL configuration (shift_domain, R0, R, mu, I, angles) is NOT
        here. It is enforced instead by the stamp written into the .npz and
        checked on every load in _load_or_build_phi -- see cache_keys.py for why
        the two groups use different mechanisms. Before either existed, changing
        shift_domain from 0.14 m to 0.10 m produced an identical hash and silently
        reloaded the 0.14 m map.

        Coil calibration coefficients (kt/koh/kv) are covered by NEITHER, on
        purpose: they correct the MEASURED field before it reaches the estimator
        and never enter cal_signal, so they cannot change this map. They ARE keyed
        on the weights cache, which does depend on them.
        """
        # Probes are SORTED here even though self.probes keeps its given order.
        # A probe set is unordered: "12 4 6 10" and "4 6 10 12" are the same set
        # and produce the same Phi, since P and sig permute together and so
        # P @ (sig - S0) is permutation-invariant. Sorting makes the two share
        # one cache entry instead of building the identical map twice.
        order = np.argsort(self.probes)
        s = ("M:" + " ".join(str(self.probes[i]) for i in order)
             + "|w:" + " ".join(f"{self.weights[i]:.6g}" for i in order)
             + "|fit_ip:" + str(self.fit_ip)
             + "|g:" + " ".join(f"{self.gains[i]:.6g}" for i in order)
             + f"|grid:{self.phys_step:.6g}:{self.uv_oversample:.4g}")
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

    def ensure_phi(self):
        """Build or load this estimator's Phi map if it is not loaded yet."""
        if not self._phi_ready:
            self._load_or_build_phi()      # sets self._phi, the interpolator
            self._phi_ready = True
        return self

    def _load_or_build_phi(self):
        from scipy.interpolate import RectBivariateSpline
        os.makedirs(PHI_DIR, exist_ok=True)
        path = os.path.join(PHI_DIR, f"PhiM_{self._config_hash()}.npz")
        if not os.path.exists(path):
            grid = np.arange(-shift_domain, shift_domain + self.phys_step / 2, self.phys_step)
            n = len(grid)
            # cal_signal over the physical grid depends on the FORWARD MODEL and
            # the probe angles only -- not on the probe subset, not on the
            # weights. Everything set-specific is the 2xM projection below, so
            # the sweep is computed once for all 12 probes and shared. A new
            # shot changes the weights, hence P, hence every Phi map, but not a
            # single one of these field values.
            TAB = _forward_table(self.phys_step)          # (n*n, 12), cached
            cols = [_ALL_PROBES.index(p) for p in self.probes]
            sig = TAB[:, cols]                            # (n*n, M)
            if self.fit_ip:
                x = sig @ self.P.T                        # (n*n, 3)
                UU = (x[:, 1] / x[:, 0]).reshape(n, n)
                VV = (x[:, 2] / x[:, 0]).reshape(n, n)
            else:
                d = (sig - self.S0) @ self.P.T            # (n*n, 2)
                UU = d[:, 0].reshape(n, n); VV = d[:, 1].reshape(n, n)
            from scipy.interpolate import (CloughTocher2DInterpolator,
                                            NearestNDInterpolator)
            from scipy.spatial import Delaunay
            uv_n = _uv_n(self.phys_step, self.uv_oversample)   # lookup grid tied to sweep
            pts = np.column_stack([UU.ravel(), VV.ravel()])
            RR, ZZ = np.meshgrid(grid, grid, indexing="ij")
            ug = np.linspace(UU.min(), UU.max(), uv_n)
            vg = np.linspace(VV.min(), VV.max(), uv_n)
            UG, VG = np.meshgrid(ug, vg, indexing="ij")
            q = np.column_stack([UG.ravel(), VG.ravel()])
            # ONE triangulation, two interpolants. The triangulation is the
            # expensive half and does not depend on which value array is being
            # interpolated, so it is built once and handed to two CloughTocher
            # interpolators -- which is what griddata(cubic) does internally,
            # one triangulation per call.
            tri = Delaunay(pts)
            tabR = CloughTocher2DInterpolator(tri, RR.ravel())(q)
            tabZ = CloughTocher2DInterpolator(tri, ZZ.ravel())(q)
            out = np.isnan(tabR)
            if out.any():
                # Same argument for the fallback: one KD-tree, not two.
                near = NearestNDInterpolator(pts, np.column_stack(
                    [RR.ravel(), ZZ.ravel()]))
                fill = near(q[out])
                tabR[out] = fill[:, 0]
                tabZ[out] = fill[:, 1]
            np.savez_compressed(path, ug=ug, vg=vg,
                                tabR=tabR.reshape(uv_n, uv_n),
                                tabZ=tabZ.reshape(uv_n, uv_n),
                                fm_key=forward_model_key(),
                                cfg_hash=self._config_hash(),
                                model=describe())
        d = np.load(path)
        # Verify provenance BEFORE using the map. An UNSTAMPED file is refused as
        # firmly as a mismatched one: a map of unknown provenance offers exactly
        # the guarantee this mechanism exists to remove. The fix is one command,
        # and it forces the user to ASSERT (rather than the code to assume) that
        # the file matches the current parameters.
        stamp = str(d["fm_key"]) if "fm_key" in d.files else None
        if stamp is None:
            raise RuntimeError(
                f"Phi map {os.path.basename(path)} has no provenance stamp (it "
                f"predates cache stamping), so it cannot be verified against the "
                f"current forward model.\n"
                f"  If it WAS built with the current parameters ({describe()}), "
                f"run:\n"
                f"      python stamp_model_caches.py --apply\n"
                f"  Otherwise delete {PHI_DIR}/ and let the maps rebuild.")
        if stamp != forward_model_key():
            raise RuntimeError(
                f"Phi map {os.path.basename(path)} was built under a DIFFERENT "
                f"forward model.\n"
                f"  stamp   : {stamp}\n"
                f"  current : {forward_model_key()}  ({describe()})\n"
                f"Delete {PHI_DIR}/ and let the maps rebuild. Using this file "
                f"would produce silently wrong displacements.")
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
