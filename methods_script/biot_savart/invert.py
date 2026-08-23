"""Per-sample nonlinear least-squares inversion for (dR, dZ).

The estimator fits the filament displacement directly to the probe signals
against the exact forward model: no linear proxy, no polynomial, no Phi table
and no interpolation of the answer.

Two search methods are available.

  "grid"  Evaluate the residual at every point of a 1 mm lattice covering the
          chamber, take every local minimum of that surface, and refine each one
          with a continuous least-squares solve. The lowest wins. Subject to the
          lattice being fine enough to place a point in every basin, this finds
          the GLOBAL minimum inside the chamber rather than whichever minimum a
          starting guess happens to reach. Default.

  "phi"   Descend from the Phi method's answer for that sample, and only from
          there. Answers a different question: what the Phi answer refines to,
          which is not necessarily the global minimum.

Running both separates two contributions to a Phi-versus-Biot-Savart gap: the
approximation error within a basin, and Phi having landed in a different basin
altogether.

Displacements are in metres everywhere in this module.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

# Search domain: the TT-1 limiter sits 0.20 m from the vessel centre, so a
# filament further out than this has no physical meaning. The probe circle is at
# 0.321 m, where the field diverges, so the domain stays well clear of it.
CHAMBER_RADIUS = 0.20

# Lattice step for the grid search. The cost is a single matrix-vector product
# per sample, so this is set for confidence rather than speed: refine it and the
# set of minima found should not change.
GRID_STEP = 0.001

# Two minima closer together than this count as the same minimum.
_MINIMA_TOL_M = 1e-3

# At most this many basins are refined per sample, lowest grid residual first.
_MAX_BASINS = 8

# Finite-difference step for the Jacobian, metres.
_JAC_STEP = 1e-7


# ---------------------------------------------------------------- templates

class Templates:
    """Forward model evaluated once on the lattice, for one probe set.

    The predicted signal at a lattice point does not depend on the sample, only
    the measurement does, so the whole lattice is evaluated once per shot and
    reused for every sample. That is what makes an exhaustive search cost one
    matrix-vector product per sample.
    """

    def __init__(self, forward_many, idx, sqrtw,
                 radius=CHAMBER_RADIUS, step=GRID_STEP):
        n = int(np.floor(radius / step))
        ax = np.arange(-n, n + 1) * step
        RR, ZZ = np.meshgrid(ax, ax, indexing="ij")
        inside = (RR ** 2 + ZZ ** 2) <= radius ** 2

        self.radius, self.step, self.shape = radius, step, RR.shape
        self.inside = inside
        self.pts = np.column_stack([RR[inside], ZZ[inside]])

        P = np.asarray(forward_many(self.pts[:, 0], self.pts[:, 1]), float)
        # float32 halves the memory traffic of the per-sample matrix-vector
        # product, which is bandwidth-bound. It costs about seven significant
        # figures in the residual SURFACE, which only has to rank basins; the
        # position itself comes from the double-precision refinement that
        # follows, so the answer is unaffected.
        self.P = np.ascontiguousarray((P[:, idx] * sqrtw[None, :]),
                                      dtype=np.float32)
        self.pp = np.einsum("ij,ij->i", self.P, self.P).astype(np.float64)
        self.pp_safe = np.where(self.pp > 0, self.pp, np.inf)
        # Buffers reused on every sample, so the per-sample cost is arithmetic
        # rather than allocation. float32 halves the memory traffic of the
        # neighbour scan, which is what dominates it.
        self._buf = np.full(self.shape, np.inf, dtype=np.float32)
        self._nb = np.empty((self.shape[0] - 2, self.shape[1] - 2),
                            dtype=np.float32)
        self._inside_c = np.ascontiguousarray(self.inside[1:-1, 1:-1])

    def residual_surface(self, m_all, alpha_fixed=None):
        """Squared weighted residual at every lattice point, as a 2D array.

        With the amplitude profiled out the minimiser of the residual is the
        maximiser of (m.p)^2 / (p.p), i.e. a normalised cross-correlation
        between the measurement and each template.
        """
        mp = (self.P @ m_all.astype(np.float32)).astype(np.float64)
        mm = float(m_all @ m_all)
        if alpha_fixed is None:
            r2 = mm - mp ** 2 / self.pp_safe
        else:
            a = float(alpha_fixed)
            r2 = mm - 2.0 * a * mp + a * a * self.pp
        self._buf[self.inside] = np.maximum(r2, 0.0)
        return self._buf

    def basins(self, m_all, alpha_fixed=None, max_basins=_MAX_BASINS):
        """Lattice-local minima of the residual surface, best first.

        A plateau reports every point on it; the clustering below keeps one.
        """
        r2 = self.residual_surface(m_all, alpha_fixed)
        # 8-neighbour test by slicing. Equivalent to a 3x3 minimum filter on the
        # interior, and several times cheaper; the lattice border lies outside
        # the disc, so nothing is lost by skipping it.
        h, k = self.shape[0] - 2, self.shape[1] - 2
        c = r2[1:-1, 1:-1]
        nb = self._nb
        nb.fill(np.inf)
        for di in (0, 1, 2):
            for dj in (0, 1, 2):
                if di == 1 and dj == 1:
                    continue
                np.minimum(nb, r2[di:di + h, dj:dj + k], out=nb)
        loc = (c <= nb) & self._inside_c
        idx = np.flatnonzero(loc.ravel())
        # indices are into the interior block; lift them to the full lattice
        ii, jj = np.unravel_index(idx, (h, k))
        idx = (ii + 1) * self.shape[1] + (jj + 1)
        if idx.size == 0:
            return np.empty((0, 2)), np.empty(0)

        vals = r2.ravel()[idx]
        order = np.argsort(vals)
        ii, jj = np.unravel_index(idx[order], self.shape)
        n = (self.shape[0] - 1) // 2
        pts = np.column_stack([(ii - n) * self.step, (jj - n) * self.step])

        keep_p, keep_v = [], []
        for p, v in zip(pts, vals[order]):
            if any(np.linalg.norm(p - q) < max(2 * self.step, _MINIMA_TOL_M)
                   for q in keep_p):
                continue
            keep_p.append(p)
            keep_v.append(v)
            if len(keep_p) >= max_basins:
                break
        return np.array(keep_p), np.array(keep_v)


# ----------------------------------------------------------------- residual

def _residual_factory(forward, forward_many, meas, sqrtw, alpha_fixed=None):
    """Build the residual, its Jacobian, and the amplitude reader for one sample.

    The residual is

        r_i = sqrt(w_i) * (m_i - alpha * p_i(dR, dZ)) / local

    where `local` is the weighted RMS of this sample's own measurement, which
    makes the solver's convergence tolerances independent of signal amplitude.

    With `alpha_fixed=None` (fit_ip=True) the amplitude is eliminated
    analytically at each (dR, dZ):

        alpha* = sum_i w_i m_i p_i / sum_i w_i p_i^2

    so only the ratios between probes enter the fit, and `alpha` becomes an
    output. With `alpha_fixed` set (fit_ip=False), that value is used instead
    and the absolute signal level enters the fit.
    """
    m_all = sqrtw * meas
    local = float(np.sqrt(np.mean(m_all ** 2)))
    if not np.isfinite(local) or local <= 0:
        local = 1.0

    def _alpha(p):
        if alpha_fixed is not None:
            return float(alpha_fixed)
        denom = float(p @ p)
        return float(m_all @ p) / denom if denom > 0 else 0.0

    def resid(x):
        p = sqrtw * forward(x[0], x[1])
        return (m_all - _alpha(p) * p) / local

    def jac(x):
        # One batched forward call covers both finite-difference points.
        pts = np.array([[x[0] + _JAC_STEP, x[1]], [x[0], x[1] + _JAC_STEP]])
        P = sqrtw[None, :] * forward_many(pts[:, 0], pts[:, 1])
        r0 = resid(x)
        J = np.empty((m_all.size, 2))
        for c in range(2):
            J[:, c] = ((m_all - _alpha(P[c]) * P[c]) / local - r0) / _JAC_STEP
        return J

    def amplitude(x):
        return _alpha(sqrtw * forward(x[0], x[1]))

    return resid, jac, amplitude, local, m_all


# Gauss-Newton refinement stops when the step falls below this, metres.
_GN_TOL = 1e-9
_GN_MAX_IT = 12


def _polish(resid, jac, x0, radius):
    """Refine one basin.

    The grid search hands over a point within half a lattice step of the
    minimum, so a short Gauss-Newton iteration converges in a few steps and
    avoids the setup cost of a general-purpose solver. Only two parameters, so
    the normal equations are a 2x2 solve. If it fails to converge -- a flat or
    ill-conditioned basin -- it falls back to `least_squares`, which is slower
    but more robust.

    A solution that leaves the disc is re-solved on the boundary circle, where
    the constraint is a single angle.
    """
    x = np.asarray(x0, float).copy()
    nfev, ok = 0, False
    for _ in range(_GN_MAX_IT):
        r = resid(x)
        J = jac(x)
        nfev += 1
        try:
            step = np.linalg.solve(J.T @ J, -(J.T @ r))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        # Never leave the search domain mid-iteration: the forward model
        # diverges at the probe circle well outside it.
        nx = x + step
        n = np.hypot(nx[0], nx[1])
        if n > radius:
            nx = nx * (radius / n)
        x = nx
        if np.linalg.norm(step) < _GN_TOL:
            ok = True
            break

    if not ok:
        rr = least_squares(
            resid, np.asarray(x0, float), jac=jac,
            bounds=([-radius, -radius], [radius, radius]),
            method="trf", xtol=1e-12, ftol=1e-14, gtol=1e-12, max_nfev=200)
        x, nfev = rr.x, nfev + int(rr.nfev)

    at_wall = False

    if np.hypot(x[0], x[1]) > radius:
        def f(th):
            return float(np.sum(resid(np.array(
                [radius * np.cos(th), radius * np.sin(th)])) ** 2))
        th0 = np.arctan2(x[1], x[0])
        best = min((f(t), t) for t in th0 + np.linspace(-0.4, 0.4, 41))
        try:
            m = minimize_scalar(f, bracket=None,
                                bounds=(best[1] - 0.05, best[1] + 0.05),
                                method="bounded")
            th = float(m.x)
        except Exception:
            th = best[1]
        x = np.array([radius * np.cos(th), radius * np.sin(th)])
        at_wall = True

    return x, nfev, at_wall


def invert_sample(forward, forward_many, meas, sqrtw, scale, templates=None,
                  x0=None, radius=CHAMBER_RADIUS, alpha_fixed=None):
    """Invert one sample.

    Supply `templates` for the grid search, or `x0` for a single descent from a
    given starting point. Returns
    (x, amp, resid_norm, n_minima, spread_m, nfev, at_wall), with `resid_norm`
    expressed against the shot-wide `scale` so it is comparable between samples.

    `n_minima` > 1 means more than one filament position reproduces these
    signals, so the sample is ambiguous. From the grid search that count is a
    property of the residual surface; from a single descent it is always 1 and
    says nothing.

    `at_wall` means the best fit inside the chamber lies on the 0.20 m limiter
    radius. That is a statement about the shot, not about the solver.
    """
    resid, jac, amplitude, local, m_all = _residual_factory(
        forward, forward_many, meas, sqrtw, alpha_fixed=alpha_fixed)
    to_global = local / scale if scale > 0 else 1.0

    if templates is not None:
        starts, _ = templates.basins(m_all, alpha_fixed)
        if starts.size == 0:
            starts = np.zeros((1, 2))
    elif x0 is not None:
        starts = np.asarray(x0, float).reshape(1, 2)
    else:
        raise ValueError("invert_sample needs either templates or x0")

    sols, nfev = [], 0
    for s in starts:
        try:
            x, nf, wall = _polish(resid, jac, s, radius)
        except Exception:
            continue
        nfev += nf
        rms = float(np.sqrt(np.mean(resid(x) ** 2))) * to_global
        sols.append((rms, x, wall))

    if not sols:
        return np.array([np.nan, np.nan]), np.nan, np.inf, 0, np.nan, nfev, False

    sols.sort(key=lambda s: s[0])
    best_cost, best_x, best_wall = sols[0]

    near = [x for c, x, _ in sols if c <= best_cost * 1.05 + 1e-15]
    clusters = []
    for x in near:
        if not any(np.linalg.norm(x - c) < _MINIMA_TOL_M for c in clusters):
            clusters.append(x)
    spread = (max(np.linalg.norm(a - b) for a in clusters for b in clusters)
              if len(clusters) > 1 else 0.0)

    return (best_x, amplitude(best_x), best_cost, len(clusters), spread,
            nfev, best_wall)


# -------------------------------------------------------------- whole shot

def _prepare(B, probes, weights):
    idx = (np.arange(B.shape[1]) if probes is None
           else np.asarray(probes, int) - 1)
    if idx.size and (idx.min() < 0 or idx.max() >= B.shape[1]):
        raise IndexError(f"probes {probes} out of range for B with "
                         f"{B.shape[1]} columns (probes are 1-based)")
    w = (np.ones(idx.size) if weights is None
         else np.asarray(weights, float).ravel()[idx])
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("probe weights must be finite and non-negative")
    if w.sum() <= 0:
        raise ValueError("all selected probes have zero weight")
    return idx, np.sqrt(w / w.mean())


def _empty(T, t, ip):
    return dict(
        t=t, ip=ip,
        dR_m=np.full(T, np.nan), dZ_m=np.full(T, np.nan),
        amp=np.full(T, np.nan), resid_norm=np.full(T, np.nan),
        n_minima=np.zeros(T, int), spread_m=np.full(T, np.nan),
        nfev=np.zeros(T, int), gated=np.zeros(T, bool),
        at_wall=np.zeros(T, bool), n_probes=np.zeros(T, int),
        sigma_R_m=np.full(T, np.nan), sigma_Z_m=np.full(T, np.nan),
    )


def invert_shot(forward, forward_many, t, ip, B, probes=None, weights=None,
                ip_min=None, radius=CHAMBER_RADIUS, step=GRID_STEP,
                search="grid", phi_xy=None, fit_ip=True, I_ref=None,
                progress=None):
    """Invert a whole shot, sample by sample.

    forward       f(dR, dZ) -> (M,) predicted signals at the model current.
    forward_many  f(dR[], dZ[]) -> (N, M), the same model batched.
    probes        1-based probe numbers to use; None means every column of B.
    weights       per-probe w_i over all 12 probes, or None for uniform. The
                  selected entries are rescaled to mean 1 before use, so
                  `resid_norm` does not depend on the absolute weight scale.
    ip_min        plasma-current gate in the units of `ip`; None disables it.
    search        "grid" exhaustive lattice search then refinement, or
                  "phi" a single descent from `phi_xy`.
    phi_xy        (T, 2) array of Phi-method displacements in metres, required
                  for search="phi". A sample whose entry is not finite returns
                  NaN rather than falling back to another start: with this
                  search method the Phi answer IS the question.
    fit_ip        True eliminates the overall amplitude analytically, so only
                  the ratios between probes enter the fit and the recovered
                  amplitude becomes an output. False fixes the amplitude at
                  ip / I_ref, so the absolute signal level enters the fit.
                  Same meaning as mprobe.MProbeEstimator's fit_ip.

    Returns a dict of equal-length arrays; displacements are in metres.
    """
    t = np.asarray(t, float)
    ip = np.asarray(ip, float)
    B = np.asarray(B, float)
    T = t.size

    if search not in ("grid", "phi"):
        raise ValueError(f"search must be 'grid' or 'phi', got {search!r}")
    if search == "phi":
        if phi_xy is None:
            raise ValueError("search='phi' needs phi_xy")
        phi_xy = np.asarray(phi_xy, float)
        if phi_xy.shape != (T, 2):
            raise ValueError(f"phi_xy has shape {phi_xy.shape}, expected {(T, 2)}")
    if not isinstance(fit_ip, bool):
        raise TypeError(f"fit_ip must be a bool, got {fit_ip!r}")
    if not fit_ip and I_ref is None:
        raise ValueError("fit_ip=False needs I_ref")

    idx, sqrtw = _prepare(B, probes, weights)

    def fwd(dR, dZ):
        return np.asarray(forward(dR, dZ), float)[idx]

    def fwd_many(dR, dZ):
        return np.asarray(forward_many(dR, dZ), float)[:, idx]

    templates = (Templates(forward_many, idx, sqrtw, radius, step)
                 if search == "grid" else None)

    sel = B[:, idx]
    finite = np.isfinite(sel)
    scale = float(np.median(np.abs(sel[finite]))) if finite.any() else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    out = _empty(T, t, ip)
    for i in range(T):
        if ip_min is not None and not (np.abs(ip[i]) >= ip_min):
            out["gated"][i] = True
            continue
        meas = sel[i]
        if not np.all(np.isfinite(meas)):
            continue
        x0 = None
        if search == "phi":
            x0 = phi_xy[i]
            if not np.all(np.isfinite(x0)):
                continue

        af = None if fit_ip else float(ip[i]) / float(I_ref)
        res = invert_sample(fwd, fwd_many, meas, sqrtw, scale,
                            templates=templates, x0=x0,
                            radius=radius, alpha_fixed=af)
        (out["dR_m"][i], out["dZ_m"][i]) = res[0]
        out["amp"][i], out["resid_norm"][i] = res[1], res[2]
        out["n_minima"][i], out["spread_m"][i] = res[3], res[4]
        out["nfev"][i], out["at_wall"][i] = res[5], res[6]
        out["n_probes"][i] = idx.size
        out["sigma_R_m"][i], out["sigma_Z_m"][i] = sigma_from_residual(
            fwd, fwd_many, meas, sqrtw, res[0], alpha_fixed=af)

        if progress is not None and (i % 250 == 0 or i == T - 1):
            progress(i + 1, T)

    return out


def invert_shot_varying(forward, forward_many, t, ip, B, probe_sets, chosen,
                        weights=None, ip_min=None, radius=CHAMBER_RADIUS,
                        step=GRID_STEP, search="grid", phi_xy=None,
                        fit_ip=True, I_ref=None, progress=None):
    """Invert a shot with a probe set that changes from sample to sample.

    probe_sets  list of probe-number lists, indexed by `chosen`.
    chosen      per-sample index into `probe_sets`; a negative entry means no
                set applies and the sample returns NaN.

    Weights are re-selected and renormalised for whichever set a sample uses.
    `scale` is taken once over all columns of B, so `resid_norm` stays
    comparable between samples that used different sets. Templates are built
    lazily, one lattice per set actually used.
    """
    t = np.asarray(t, float)
    ip = np.asarray(ip, float)
    B = np.asarray(B, float)
    chosen = np.asarray(chosen, int)
    T = t.size
    if chosen.size != T:
        raise ValueError(f"chosen has {chosen.size} entries, expected {T}")
    if search not in ("grid", "phi"):
        raise ValueError(f"search must be 'grid' or 'phi', got {search!r}")
    if search == "phi":
        if phi_xy is None:
            raise ValueError("search='phi' needs phi_xy")
        phi_xy = np.asarray(phi_xy, float)
    if not fit_ip and I_ref is None:
        raise ValueError("fit_ip=False needs I_ref")

    prepared = [_prepare(B, s, weights) for s in probe_sets]
    templates = [None] * len(probe_sets)

    finite = np.isfinite(B)
    scale = float(np.median(np.abs(B[finite]))) if finite.any() else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    out = _empty(T, t, ip)
    for i in range(T):
        k = int(chosen[i])
        if k < 0 or k >= len(prepared):
            continue
        if ip_min is not None and not (np.abs(ip[i]) >= ip_min):
            out["gated"][i] = True
            continue
        idx, sqrtw = prepared[k]
        meas = B[i, idx]
        if not np.all(np.isfinite(meas)):
            continue
        x0 = None
        if search == "phi":
            x0 = phi_xy[i]
            if not np.all(np.isfinite(x0)):
                continue

        if search == "grid" and templates[k] is None:
            templates[k] = Templates(forward_many, idx, sqrtw, radius, step)

        fwd = lambda a, b, I=idx: np.asarray(forward(a, b), float)[I]
        fwd_many = lambda a, b, I=idx: np.asarray(forward_many(a, b), float)[:, I]
        af = None if fit_ip else float(ip[i]) / float(I_ref)

        res = invert_sample(fwd, fwd_many, meas, sqrtw, scale,
                            templates=templates[k], x0=x0,
                            radius=radius, alpha_fixed=af)
        (out["dR_m"][i], out["dZ_m"][i]) = res[0]
        out["amp"][i], out["resid_norm"][i] = res[1], res[2]
        out["n_minima"][i], out["spread_m"][i] = res[3], res[4]
        out["nfev"][i], out["at_wall"][i] = res[5], res[6]
        out["n_probes"][i] = idx.size
        out["sigma_R_m"][i], out["sigma_Z_m"][i] = sigma_from_residual(
            fwd, fwd_many, meas, sqrtw, res[0], alpha_fixed=af)

        if progress is not None and (i % 250 == 0 or i == T - 1):
            progress(i + 1, T)

    return out


# ------------------------------------------------------------- uncertainty

def sigma_from_residual(forward, forward_many, meas, sqrtw, x, alpha_fixed=None):
    """Per-sample (sigma_dR, sigma_dZ) in metres, from the fit residual.

        C = s^2 (J^T W J)^-1 ,   s^2 = |r|^2 / (M - p)

    with J the Jacobian of the weighted residual at `x` and p the number of
    fitted quantities: 3 when the amplitude is profiled out, 2 when it is fixed.
    The Jacobian carries the units, converting a residual in Tesla into a
    displacement in metres.

    This is a CONDITIONING measure, not a confidence interval. The formula
    assumes the residual is independent zero-mean noise of common variance; the
    residual here is dominated by model error, which is systematic and
    correlated between probes. Read it as how tightly these probes pin the
    position given how badly the model fits, and not as a probability that the
    true position lies inside it.

    `x` need not be a minimum of this objective. Evaluating at another method's
    answer gives that method's band on the same footing, with the larger
    residual there widening it, which is the intended behaviour.
    """
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        return np.nan, np.nan
    resid, jac, _amp, _local, m_all = _residual_factory(
        forward, forward_many, meas, sqrtw, alpha_fixed=alpha_fixed)

    M = m_all.size
    p = 2 if alpha_fixed is not None else 3
    if M <= p:
        return np.nan, np.nan          # no degrees of freedom left

    try:
        r = resid(x)
        J = jac(x)
        # resid() is divided by `local`; multiply it back so s^2 and J are in
        # the same units and the ratio comes out in metres^2.
        s2 = float(r @ r) * _local ** 2 / (M - p)
        JTJ = J.T @ J * _local ** 2
        C = s2 * np.linalg.inv(JTJ)
        d = np.diag(C)
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            return np.nan, np.nan
        return float(np.sqrt(d[0])), float(np.sqrt(d[1]))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    except Exception:
        return np.nan, np.nan


def sigma_shot(forward, forward_many, t, ip, B, dR_m, dZ_m, probes=None,
               weights=None, fit_ip=True, I_ref=None, chosen=None,
               probe_sets=None):
    """(sigma_dR, sigma_dZ) arrays for a whole shot at given positions.

    Positions come in as `dR_m`, `dZ_m`, so this works for any method: pass the
    Phi answer to get the Phi band, or the Biot-Savart answer to get its own.
    Supply `chosen` and `probe_sets` instead of `probes` when the probe set
    varies from sample to sample.
    """
    B = np.asarray(B, float)
    ip = np.asarray(ip, float)
    dR_m = np.asarray(dR_m, float)
    dZ_m = np.asarray(dZ_m, float)
    T = dR_m.size
    if not fit_ip and I_ref is None:
        raise ValueError("fit_ip=False needs I_ref")

    varying = chosen is not None
    if varying:
        if probe_sets is None:
            raise ValueError("chosen needs probe_sets")
        prepared = [_prepare(B, s, weights) for s in probe_sets]
        chosen = np.asarray(chosen, int)
    else:
        prepared = [_prepare(B, probes, weights)]

    sR = np.full(T, np.nan)
    sZ = np.full(T, np.nan)
    for i in range(T):
        if not (np.isfinite(dR_m[i]) and np.isfinite(dZ_m[i])):
            continue
        k = int(chosen[i]) if varying else 0
        if k < 0 or k >= len(prepared):
            continue
        idx, sqrtw = prepared[k]
        meas = B[i, idx]
        if not np.all(np.isfinite(meas)):
            continue
        fwd = lambda a, b, I=idx: np.asarray(forward(a, b), float)[I]
        fwd_many = lambda a, b, I=idx: np.asarray(forward_many(a, b), float)[:, I]
        af = None if fit_ip else float(ip[i]) / float(I_ref)
        sR[i], sZ[i] = sigma_from_residual(
            fwd, fwd_many, meas, sqrtw, (dR_m[i], dZ_m[i]), alpha_fixed=af)
    return sR, sZ
