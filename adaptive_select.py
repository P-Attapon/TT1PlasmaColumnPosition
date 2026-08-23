"""
adaptive_select.py -- per-timestep probe-set switching for the filament method,
driven by a cached, shot-independent round-trip field.

WHAT IT DOES
------------
A single probe set has ONE reachable region in the linear (dU,dV) plane (the
image of the +-0.14 m physical domain under that set's estimator P). Where the
plasma trajectory leaves that region Phi cannot invert it and the output is NaN
or railed. Different sets have differently-shaped regions, so a point
unreachable for one set is often reachable for another. Adaptive selection walks
a FIXED order and, at each timestep, uses the first set that is locally
acceptable at that sample -- falling back only where the current one is not.
Output is REAL calibrated (dR, dZ) via each chosen set's Phi
(MProbeEstimator.shift()), not the raw proxy.

There is exactly ONE selection scheme in this module: adaptive_selection().

TWO PHASES
----------
Phase A (preparation, before the first sample):
    weights -> Phi map per set -> convex hull per set -> rt(u,v) field per set
    -> static order. Every one of these is cached. Offline, weights come from
    this shot's own pre-shot window ("auto"); in real time they are inherited
    from a previous shot ("last"), because the pre-shot window is too short to
    compute them live. Nothing else differs between the two.

Phase B (per sample, identical offline and in real time):
    proxy (u,v) -> is the current set still acceptable (inside its hull AND
    rt(u,v) <= RT_GOOD)? -> if yes keep it (hysteresis); if no, walk the static
    order and take the first set that qualifies; if none qualifies, output NaN.

THE ACCEPTANCE TEST -- LOCAL, NOT PER-SET
-----------------------------------------
rt(u,v) is the round trip: invert a point through Phi to (dR,dZ), push that back
through the EXACT forward model to (u',v'), and measure ||(u,v)-(u',v')||. It is
~0 where Phi is faithful and grows toward the domain boundary where cubic
(Clough-Tocher) interpolation degrades -- a MEASURED interpolation error, not a
geometric proxy for one. This is the standard LOOCV/error-estimation remedy from
the scattered-data interpolation literature (Rippa 1999 and successors),
specialised to this interpolator.

Evaluated as a FIELD on a fixed grid, rt is a property of (Phi, estimator,
forward model) alone -- no shot data enters it. So it is computed once per probe
set, cached, and read in microseconds; the shot only decides which (u,v) get
visited. The question asked per sample is therefore LOCAL: is rt acceptable at
THIS point, for the set being considered.

Containment in the hull is still required, but implicitly: outside the hull the
inversion is extrapolation, which shows up as a large or infinite rt. Both tests
are kept anyway -- the hull test is one cheap matmul and makes the failure
reason legible.

The AI camera is never used to select. It is an INDEPENDENT cross-check;
selecting the set that best matches it would make it a ground truth by the back
door, which is circular.

ORDERING
--------
Sets are ordered by good-frac (the share of a set's own IN-HULL nodes meeting
RT_GOOD -- how much of its reachable region can be trusted), ties broken by the
median rt where it IS good. Ordering only: it changes WHICH acceptable set is
picked, never WHETHER a sample is accepted. Both quantities come from the cached
field, so the order is shot-independent and known before the first sample
arrives -- there is no ranking pass on a live shot.

Do NOT read good-frac as a set-quality score. It counts how much of a region is
trustworthy, not how bad the rest gets: sets [12 2 6 8] (0.776) and [11 1 5 7]
(0.738) score well against round trips of 179 mm. The per-sample test is what
catches them.

WEIGHTS AND THE PHI CACHE
-------------------------
Preshot weights w_i = 1/sigma_i^2 depend only on the shot's pre-plasma noise,
not on the set. The Phi map is keyed on (probes, weights, gains, grid) -- NOT on
shot -- so shots with equal weights share a Phi file. prebuild_shot() computes a
shot's weights once and builds every candidate's Phi map, hull and rt field, so
a later run reads instead of computing; a per-shot manifest records the hashes.
Cheaper than storing Phi per shot: identical maps are never duplicated. Note
that the weights enter Phi, so changing weights_source invalidates the caches
and pays the build again.

COST
----
Prebuild (from empty): a few tens of seconds per set for the expensive ones,
under a second for the rest; ~6 min and ~170 MB for the 16 default candidates
(15 antipodal 4-probe sets plus the full 12-probe array). Cached
thereafter, and an adaptive run then takes seconds. Switching adds ~42x a single
set's per-timestep arithmetic (~24 -> ~1000 flops), negligible absolutely.

REAL-TIME NOTE
--------------
The selection logic is causal: the Phase B loop body reads only sample i and the
current set. adaptive_selection() is nonetheless BATCH-SHAPED for offline use --
proxies and rt lookups are vectorised over the whole shot before the loop. That
is a Python optimisation, not part of the algorithm. To port, move the proxy and
rt lookup inside the loop and replace _load_shot with the acquisition feed; no
logic changes. Measured single-threaded: 0.18 us/sample batched vs 25.5
us/sample one at a time -- a 139x gap that is pure per-call overhead, NOT
parallelism (it survives OMP_NUM_THREADS=1). The single-sample Python path costs
~40 us against a 20 us period at 50 kHz; that gap is interpreter overhead, not
arithmetic.

To run online: (1) every candidate's Phi, hull and rt field precomputed, which
means fixed weights (use weights_source="last"; unit weights are NOT valid --
they shift dR ~75 mm and can zero a set's output), and (2) the switch logic
outside Python.
"""
import os
import io
import json
import time
import hashlib
import tempfile
import contextlib

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from methods_script.toroidal_filament.parameters import (
    shift_domain, R, I as I_PARAM, mu, all_arrays, probe_lst_to_str)
from methods_script.toroidal_filament.signal_strength import cal_signal
from methods_script.toroidal_filament.process_probe_data import magnetic_field_calibration
from methods_script.toroidal_filament import mprobe as _mprobe_mod
from methods_script.toroidal_filament.mprobe import MProbeEstimator, PHI_DIR
from methods_script.toroidal_filament.curation import compute_weights, calibration_coeff
from methods_script.toroidal_filament import weights_cache
from methods_script.toroidal_filament.cache_keys import forward_model_key

_IP_THRESH = 2500.0


def _rd(f):
    return pd.read_csv(f, sep=r"\s+", skiprows=8, header=None, names=["t", "v"])


def _load_shot(shot_dir):
    """Read one shot's Ip and calibrated per-probe field, gated to the discharge.
    Uses current_channels.resolve_all for redundant-channel health checking."""
    from methods_script.toroidal_filament.current_channels import resolve_all as _resolve_all
    IP = _rd(os.path.join(shot_dir, "IP1.txt"))
    t   = IP["t"].to_numpy()
    ip  = IP["v"].to_numpy()
    n   = len(t)

    ch, _ = _resolve_all(shot_dir)
    It  = ch["IT"][:n]
    Ioh = ch["IOH"][:n]
    Iv  = ch["IV"][:n]

    B = {}
    for p in range(1, 13):
        g = _rd(os.path.join(shot_dir, f"GBP{p}T.txt"))["v"].to_numpy()[:n]
        B[p] = magnetic_field_calibration(g, calibration_coeff[f"k{p}t"], It,
                                          calibration_coeff[f"k{p}oh"], Ioh,
                                          calibration_coeff[f"k{p}v"], Iv)
    m = ip > _IP_THRESH
    return t[m], ip[m], {p: B[p][m] for p in B}

# The 15 canonical antipodal 4-probe sets from parameters.all_arrays, PLUS the
# full 12-probe array. The list is deliberately not uniform: the 12-probe set is
# not antipodal and not 4-probe, and it often ranks first. Selection places no
# constraint on set size or geometry -- adaptive_selection() accepts any list of
# probe sets, and DEFAULT_CANDIDATES is a default, not a restriction.
DEFAULT_CANDIDATES = [probe_lst_to_str(s) for s in all_arrays] + \
                     ["1 2 3 4 5 6 7 8 9 10 11 12"]

# Phi grid: use mprobe's module constants as the single source of truth, so the
# cached maps are shared with main.py / compare_methods (which also defer to
# mprobe). Do NOT hardcode a separate value here.
PHYS_STEP = _mprobe_mod.PHYS_STEP
UV_OVERSAMPLE = _mprobe_mod.UV_OVERSAMPLE
# Round-trip (Phi self-consistency) ceiling, in METRES: the acceptance threshold
# for rt(u,v). RT_GOOD below is deliberately the same value -- the same criterion
# evaluated pointwise rather than as a per-set average.
#
# WHERE 1.5 mm COMES FROM: on shot 2766 the round-trip values separate into a
# clean group (<= 0.9 mm) and a self-inconsistent one (2.3, 9, 115, 412 mm ...),
# and 1.5 mm sits in that natural gap. The set [2 4 8 10] is the case that fixed
# it: it covered more of the discharge than any other set, yet its round trip was
# 2.3 mm and it produced ~44 mm/sample jitter in dR over 350-355 ms from a single
# set -- not switching noise. Coverage alone would have admitted it.
ROUNDTRIP_MAX = 1.5e-3

_MANIFEST_DIR = os.path.join(os.path.dirname(__file__),
                             "methods_script", "toroidal_filament", "phi_manifest")


# --------------------------------------------------------------------------- #
#  weights
# --------------------------------------------------------------------------- #
def shot_weights(shot, source="auto"):
    """{probe -> weight} for a shot. 'auto' computes+persists; 'last' reuses the
    most recently stored vector (real-time proxy)."""
    if source not in ("auto", "last"):
        raise ValueError(f"weights source must be 'auto' or 'last', got {source!r}")
    sd = os.path.join("data", str(shot))
    if source == "last":
        loaded = weights_cache.load_latest()
        if loaded is None:
            raise RuntimeError("weights='last' but nothing stored; run an 'auto' shot first")
        return loaded[0]
    with contextlib.redirect_stdout(io.StringIO()):
        wdict, _, _ = compute_weights(sd, list(range(1, 13)),
                                      power=2.0, struct_ratio=6.0,
                                      rail_frac=0.01, min_samples=50)
    weights_cache.save_weights(sd, wdict)
    return wdict


def _estimator(probe_set, wdict, fit_ip=False):
    probes = list(map(int, probe_set.split()))
    w = [float(wdict.get(p, 1.0)) for p in probes]
    with contextlib.redirect_stdout(io.StringIO()):
        return MProbeEstimator(probes, weights=w, fit_ip=fit_ip,
                               phys_step=PHYS_STEP, uv_oversample=UV_OVERSAMPLE)


# --------------------------------------------------------------------------- #
#  linear proxy and the convex hull of a set's reachable region
# --------------------------------------------------------------------------- #
def _model_proxy(est, sig):
    """Proxy (u, v) for MODEL-current signals, vectorised over rows of `sig`.

    The batched twin of MProbeEstimator._linear_estimate_model, and it must stay
    that way: the hull, the rt field and the runtime proxy all have to agree on
    what (u, v) means, or a set is scored in one space and used in another.
    """
    sig = np.atleast_2d(np.asarray(sig, float))
    if getattr(est, "fit_ip", False):
        x = sig @ est.P.T                       # (N, 3): I0, I0*dU, I0*dV
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.column_stack([x[:, 1] / x[:, 0], x[:, 2] / x[:, 0]])
    return (sig - est.S0[None, :]) @ est.P.T


def _proxy(est, ip, B):
    """Proxy (u, v) for every sample, in the estimator's own convention.

    With fit_ip=False the measured signal is normalised by the measured current
    and S0 subtracted, leaving two components. With fit_ip=True the amplitude is
    a third unknown, so P is 3xM and the proxy is the pair of ratios that
    divides it out -- the same algebra as MProbeEstimator._linear_estimate_model,
    vectorised over samples. Getting this branch wrong is what previously forced
    the adaptive path to fit_ip=False: everything else (Phi, hull, rt) already
    goes through _linear_estimate_model and handles both.
    """
    sig = np.column_stack([B[p] for p in est.probes])
    if getattr(est, "fit_ip", False):
        # The ratios are invariant to the overall scale, so no current
        # normalisation is needed -- and none must be applied, or the proxy would
        # no longer match the one the hull and rt field were built in.
        return _model_proxy(est, sig)
    return _model_proxy(est, sig / (ip / I_PARAM)[:, None])




HULL_N = 25
# Per-axis sample count for the hull sweep, so n*n forward-model evaluations.
#
# VALIDATED, not inherited. Swept on 1643 across n = 15/25/51/101/201/401: health
# is converged by 51 and n=25 is already within 0.003 of the limit for EVERY set
# (largest movement anywhere: set [11 4 5 10], 0.0324 -> 0.0302). Priority order,
# survivor count (7), coverage (1.0000) and switch count (6) are IDENTICAL across
# that whole 27x range. The reason is geometric: ConvexHull keeps only extreme
# points and the domain image is near-convex, so refining the grid adds vertices
# along nearly-straight edges without moving the enclosed region.
#
# Raising it is not free even with caching, so 25 stays:
#   * face count grows ~linearly (16 faces at n=15 -> 382 at n=401) and enters the
#     PER-TIMESTEP containment test proxy @ A.T, which is T x 2 x n_faces at every
#     sample of every shot;
#   * build cost is quadratic (625 cal_signal calls at 25, 160801 at 401 -- 27 ms
#     vs 6.4 s per set) and is paid on every cache miss, i.e. every time
#     forward_model_key() changes: every recalibration, every domain revisit.
# CAVEAT: swept on ONE shot (1643), which has coverage 1.0000 and every sample
# above the 2500 A threshold. A shot with partial coverage could be more
# hull-shape-sensitive; re-sweep before relying on this elsewhere.

_HULL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "methods_script", "toroidal_filament", "hull_tables")
_HULL_MEM = {}          # in-process memo, cleared with the process


def _hull_key(est, n):
    """Cache key for a hull: everything its shape depends on, and nothing else.

    Keyed on the NUMERICAL CONTENT that determines the hull (P, S0, angles)
    rather than on the inputs that produced it (probe list, weights source). Two
    estimators with the same P and S0 have the same hull by construction, however
    they were configured, so this is both exact and robust to the weights being
    derived differently. forward_model_key() covers shift_domain and the geometry
    -- the hull is swept over [-shift_domain, shift_domain] and built from
    cal_signal, so it is stale the moment either moves.

    Deliberately NOT keyed on the shot: the hull is shot-INDEPENDENT. Only
    _proxy() depends on ip/B. That is the whole reason caching pays here -- one
    hull serves every shot at a fixed configuration.
    """
    h = hashlib.md5()
    h.update(np.ascontiguousarray(est.P, float).tobytes())
    h.update(np.ascontiguousarray(est.S0, float).tobytes())
    h.update(np.ascontiguousarray(est.angles, float).tobytes())
    h.update(f"|n:{int(n)}|fm:{forward_model_key()}".encode())
    return h.hexdigest()[:16]


def _hull_faces(est, n=None, use_cache=True):
    """Convex-hull faces of the mapped domain, as unit-normal half-spaces
    {x : a.x + b <= 0 inside}. Returns (A, b) for membership by half-space test
    (in-hull iff all a.x + b <= 0), which avoids a separate Delaunay.

    CACHED in two tiers: this is a shot-independent quantity costing 625
    cal_signal evaluations plus a ConvexHull, ~27 ms per set.

      1. in-process dict -- removes repeats within one run, costs nothing;
      2. small .npz in hull_tables/ -- survives process restart, which is exactly
         the realtime case (one process per shot).

    Disk caching is worth it HERE but is not automatically worth it: reading a
    file can easily cost more than recomputing when the computation is a few
    arithmetic ops on data already in memory. Measured for this one: 27 ms to
    compute vs 0.32 ms to np.load, a 74x margin on a 1810-byte payload. The
    payload is small enough to stay in the OS page cache, so that 0.32 ms is
    mostly npz container overhead rather than I/O.

    Any cache problem falls through to computing -- a hull is cheap enough that
    correctness beats saving 27 ms, and a corrupt file must never be fatal.
    """
    n = HULL_N if n is None else int(n)
    if not use_cache:
        return _build_hull_faces(est, n)

    key = _hull_key(est, n)
    if key in _HULL_MEM:
        return _HULL_MEM[key]

    path = os.path.join(_HULL_DIR, f"hull_{key}.npz")
    if os.path.exists(path):
        try:
            with np.load(path) as d:
                A, b = d["A"], d["b"]
            _HULL_MEM[key] = (A, b)
            return A, b
        except Exception as e:
            print(f"[hull_cache] ignoring unreadable {os.path.basename(path)} "
                  f"({type(e).__name__}: {e}); recomputing.")

    A, b = _build_hull_faces(est, n)
    _HULL_MEM[key] = (A, b)
    try:
        os.makedirs(_HULL_DIR, exist_ok=True)
        # Write-then-rename: a run interrupted mid-write cannot leave a truncated
        # file that a later run would read as a valid hull.
        fd, tmp = tempfile.mkstemp(suffix=".npz", dir=_HULL_DIR)
        os.close(fd)
        np.savez(tmp, A=A, b=b)
        os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, path)
    except Exception as e:
        print(f"[hull_cache] could not write {os.path.basename(path)} "
              f"({type(e).__name__}: {e}); continuing without disk cache.")
    return A, b


_ALL12 = list(range(1, 13))
_DISC_CACHE = {}


def _disc_table(n):
    """cal_signal over the hull disc for ALL 12 probes: (n_pts, 12).

    Same argument as mprobe._forward_table, on the hull grid instead of the Phi
    grid. Memoised per process and per n; cheap enough not to need a file.
    """
    key = (forward_model_key(), int(n))
    if key in _DISC_CACHE:
        return _DISC_CACHE[key]
    from methods_script.toroidal_filament.parameters import coil_angle_dict
    ang = [coil_angle_dict[p] for p in _ALL12]
    g = np.linspace(-shift_domain, shift_domain, n)
    rows = []
    for a_ in g:
        for b_ in g:
            if np.hypot(a_, b_) > shift_domain:
                continue
            try:
                rows.append(cal_signal(a_, b_, ang))
            except ValueError:
                continue
    TAB = np.asarray(rows, float)
    _DISC_CACHE[key] = TAB
    return TAB


def _build_hull_faces(est, n):
    """Uncached hull construction -- the original computation, unchanged."""
    # DISC, not the square grid. shift_domain is a RADIUS: the operating region is
    # the disc r <= shift_domain. Sampling the full square also feeds in the four
    # corners, which reach r = shift_domain*sqrt(2) = 0.198 m at the default 0.14 --
    # past the ~0.16-0.18 m fold radius, inside the outer shell this project already
    # treats as a resampling artefact rather than trustworthy forward model.
    # Including them inflates every hull, so the guard admitted proxy points that no
    # physically reachable plasma position can produce. Restricting to the disc makes
    # the hull mean what its name says: the image of the OPERATING REGION.
    # cal_signal is evaluated per probe and does not depend on the subset, so the
    # 12-probe table over this disc is the SAME for every candidate set: compute
    # it once and project through each set's P. The ValueError skip depends only
    # on (a_, b_), so the point set is identical for every subset too.
    TAB = _disc_table(n)
    cols = [_ALL12.index(p) for p in est.probes]
    pts = _model_proxy(est, TAB[:, cols])
    h = ConvexHull(pts)
    A = h.equations[:, :2]
    b = h.equations[:, 2]
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    return A / nrm, b / nrm.ravel()







# --------------------------------------------------------------------------- #
#  per-shot Phi prebuild
# --------------------------------------------------------------------------- #
def prebuild_shot(shot, candidates=None, weights_source="auto", verbose=True):
    """Build & cache Phase A for every candidate set: Phi map, convex hull and
    rt(u,v) field. Records a manifest of the hashes and per-set build times.

    All three are keyed on the estimator config, not on the shot, so this is
    shared work: a later shot with the same weights reads instead of building.
    From empty this costs ~6 min and ~170 MB for the 16 default candidates.
    """
    cands = candidates or DEFAULT_CANDIDATES
    wdict = shot_weights(shot, weights_source)
    os.makedirs(_MANIFEST_DIR, exist_ok=True)
    manifest = {"shot": str(shot), "weights_source": weights_source,
                "t_built": time.time(),
                "weights": {str(p): float(wdict.get(p, 1.0)) for p in range(1, 13)},
                "sets": {}}
    t0 = time.time()
    for s in cands:
        est = _estimator(s, wdict, fit_ip=fit_ip)
        h = est._config_hash()
        was_cached = os.path.exists(os.path.join(PHI_DIR, f"PhiM_{h}.npz"))
        # BUILD, don't just hash. Constructing an MProbeEstimator does NOT build
        # its Phi map -- mprobe loads or builds it lazily on first use -- so an
        # earlier version of this function reported every set as uncached and
        # left the whole cost to the first real run, which is exactly what
        # --prebuild exists to avoid. All three Phase A caches are filled here.
        t_set = time.time()
        est.ensure_phi()
        _hull_faces(est)
        rt_field(est)
        manifest["sets"][s] = {"hash": h, "cond": est.cond,
                               "was_cached": was_cached,
                               "seconds": round(time.time() - t_set, 1)}
        if verbose:
            print(f"  [{s:28s}] hash {h}  cond {est.cond:.3g}  "
                  f"{'read' if was_cached else 'built'} "
                  f"in {time.time() - t_set:.1f} s")
    manifest["build_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(_MANIFEST_DIR, f"{shot}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    if verbose:
        print(f"  prebuilt {len(cands)} sets for shot {shot} "
              f"in {manifest['build_seconds']} s")
    return manifest








# =========================================================================== #
#  ROUND-TRIP FIELD SELECTION
# =========================================================================== #
# WHY THIS EXISTS
# ---------------
# The ranked scheme above asks a per-SET question ("is this set good?") and then
# a per-SAMPLE question ("does its hull contain this point?"). Those are answered
# by two different quantities, and only the second is local. Round-trip -- the
# quantity that actually predicts accuracy -- enters only as a scalar per set.
#
# That scalar is an AVERAGE of a spatial field over wherever the plasma happened
# to go, and averaging is what makes it shot-dependent. Measured across three
# shots, the per-set scalar transfers poorly (cross-shot Spearman +0.41 to +0.69;
# set [12 3 6 9] scores 645 mm on 1643 and 0.98 mm on 2766 -- same geometry).
# Realtime inherits an admission decision built from that scalar, so it inherits
# the instability too.
#
# The FIELD is shot-independent. rt(u,v) is a property of (Phi, estimator,
# forward model) alone; the shot only decides which (u,v) get visited. So:
#   * precompute rt(u,v) per set ONCE, offline, cache it beside the hull;
#   * at runtime ask the LOCAL question -- is rt at THIS (u,v) acceptable? --
#     instead of trusting a shot-averaged summary.
#
# The ranking is NOT load-bearing for correctness: it breaks ties between sets
# that both answer "yes" locally, and sets the order in which they are consulted
# so the first hit is usually the best available. Correctness rests on the
# per-sample test, which is why a shot-dependent order would not be fatal.
#
# HYSTERESIS: a set is kept while it remains locally acceptable, even if a
# higher-ranked set would also serve. Switching is not free -- sets disagree by
# 6-12 mm at the median on 1643 -- so each avoided switch is an avoided step in
# dR. Sets are consulted only when the current one fails.

RT_GRID_N = 129           # nodes per axis of the cached rt(u,v) field
RT_GOOD = ROUNDTRIP_MAX   # per-POINT acceptance ceiling [m], same units/value as
                          # the per-set ceiling it localises. Deliberately equal:
                          # this is the same criterion evaluated pointwise rather
                          # than a new free parameter.
_RTF_MEM = {}
_RTF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "methods_script", "toroidal_filament", "rt_fields")


def _rt_field_key(est):
    """Cache key: the Phi map's identity plus the forward model.

    Keyed on est.P/S0/angles (which fix the proxy map) AND the Phi lookup grid
    ug/vg (which fix the interpolant whose fidelity is being measured). Two
    estimators agreeing on all of these produce the same field.
    """
    est.ensure_phi()        # ug/vg come from the Phi map, which is lazy now
    h = hashlib.md5()
    for arr in (est.P, est.S0, est.angles, est.ug, est.vg):
        h.update(np.ascontiguousarray(arr, float).tobytes())
    h.update(f"|n:{RT_GRID_N}|fm:{forward_model_key()}|v:2".encode())
    return h.hexdigest()[:16]


def _build_rt_field(est, n):
    """rt(u,v) on a regular grid spanning the Phi lookup domain.

    For each node: invert through Phi to (dR,dZ), push back through the EXACT
    forward model, and record ||(u,v) - (u',v')||. The same round-trip quantity
    the module docstring describes, evaluated on a fixed grid rather than on the
    samples of one shot -- which is what makes it shot-independent.

    Nodes whose inversion leaves the physical vessel (cal_signal raises) or
    returns non-finite are marked +inf: unusable is unusable, and +inf makes any
    downstream comparison reject them without a special case.
    """
    ug = np.linspace(est.ug[0], est.ug[-1], n)
    vg = np.linspace(est.vg[0], est.vg[-1], n)
    U, V = np.meshgrid(ug, vg, indexing="ij")
    u = U.ravel(); v = V.ravel()
    R = est._sR(u, v, grid=False)          # vectorised: ~4 ms for the whole grid
    Z = est._sZ(u, v, grid=False)
    ang = est.angles
    out = np.full(u.shape, np.inf)
    outR = np.full(u.shape, np.inf)
    ok = np.isfinite(R) & np.isfinite(Z)
    for k in np.where(ok)[0]:
        try:
            back = _model_proxy(
                est, np.asarray(cal_signal(R[k], Z[k], ang), float))[0]
        except ValueError:
            continue                        # inversion left the vessel -> +inf
        out[k] = np.hypot(back[0] - u[k], back[1] - v[k])
        # RADIAL component alone. dU is the radial proxy (hU carries cos(theta)),
        # so |u - u'| is the round-trip error in the direction the feedback coils
        # can actually act on. Used for ORDERING only -- acceptance stays on the
        # isotropic norm above, because acceptance is about whether Phi inverts
        # at all, which is not a per-axis question.
        outR[k] = abs(back[0] - u[k])
    return ug, vg, out.reshape(n, n), outR.reshape(n, n)


def rt_field(est, use_cache=True):
    """Cached rt(u,v) field. Returns (ug, vg, F, FR) -- isotropic and radial.\n    Same two-tier scheme as the
    hulls, and for the same reason: a shot-independent quantity that is far more
    expensive to compute (~0.4 s) than to read (~0.3 ms)."""
    est.ensure_phi()        # the field is measured THROUGH Phi, so it needs it
    if not use_cache:
        return _build_rt_field(est, RT_GRID_N)
    key = _rt_field_key(est)
    if key in _RTF_MEM:
        return _RTF_MEM[key]
    path = os.path.join(_RTF_DIR, f"rtf_{key}.npz")
    if os.path.exists(path):
        try:
            with np.load(path) as d:
                res = (d["ug"], d["vg"], d["F"], d["FR"])
            _RTF_MEM[key] = res
            return res
        except Exception as e:
            print(f"[rt_field] ignoring unreadable {os.path.basename(path)} "
                  f"({type(e).__name__}: {e}); rebuilding.")
    res = _build_rt_field(est, RT_GRID_N)
    _RTF_MEM[key] = res
    try:
        os.makedirs(_RTF_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".npz", dir=_RTF_DIR)
        os.close(fd)
        np.savez_compressed(tmp, ug=res[0], vg=res[1], F=res[2], FR=res[3])
        os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, path)
    except Exception as e:
        print(f"[rt_field] could not write {os.path.basename(path)} "
              f"({type(e).__name__}: {e}); continuing without disk cache.")
    return res


def rt_lookup(field, pts):
    """Nearest-node rt for each (u,v) in pts. Outside the grid -> +inf.

    Nearest-node, not bilinear: the field is not smooth (it is near-zero over most
    of the domain with a thin high-error tail), so interpolating would smear the
    bad region's edge into the good side. Rounding to the nearest node keeps a
    bad node bad.
    """
    ug, vg, F = field[0], field[1], field[2]
    u = np.asarray(pts)[:, 0]; v = np.asarray(pts)[:, 1]
    du = ug[1] - ug[0]; dv = vg[1] - vg[0]
    i = np.rint((u - ug[0]) / du).astype(int)
    j = np.rint((v - vg[0]) / dv).astype(int)
    inside = (i >= 0) & (i < len(ug)) & (j >= 0) & (j < len(vg))
    out = np.full(len(u), np.inf)
    out[inside] = F[i[inside], j[inside]]
    return out


def rt_field_score(est):
    """Static per-set quality from the FIELD, for ordering only.

    Returns (good_fraction, median_RADIAL_rt_over_good). good_fraction is the share of
    IN-HULL nodes meeting RT_GOOD -- how much of its own reachable region a set can
    be trusted over. Shot-independent by construction, which is the property the
    per-set scalar lacked.

    RESTRICTED TO THE HULL, deliberately. The field is built on the rectangular
    ug/vg box, but the reachable region inside it is curved and fills the box by a
    DIFFERENT fraction for each set. Scoring over the whole box therefore mixes
    "how faithful is this set" with "how rectangular is its reachable region", and
    the second term is meaningless. Measured on 1641 against the AI cross-check,
    the unrestricted score ranked [11 3 5 9] in the bottom third despite a
    round-trip of 0.03 mm -- a box-shape artefact, not a fidelity judgement.

    This matters more than "tiebreak" suggests: ~9 sets qualify at the median
    sample, so the ORDER decides which set's inversion is used on ~100% of samples,
    and sets disagree by 6-12 mm at the median.
    """
    ug, vg, F, FR = rt_field(est)
    A, b = _hull_faces(est)
    U, V = np.meshgrid(ug, vg, indexing="ij")
    pts = np.column_stack([U.ravel(), V.ravel()])
    inside = np.all((pts @ A.T) + b <= 0, axis=1).reshape(F.shape)
    fin = np.isfinite(F) & inside
    if not fin.any():
        return 0.0, np.inf
    good = fin & (F <= RT_GOOD)
    frac = float(good.sum()) / float(fin.sum())
    # Tiebreak on the RADIAL round trip, not the isotropic one: the feedback
    # coils cannot move the plasma vertically, so dR is the axis that has to be
    # right and dZ is diagnostic. Acceptance stays isotropic -- good_frac and
    # RT_GOOD use F, because whether Phi inverts at all is not a per-axis
    # question.
    med = float(np.median(FR[good])) if good.any() else np.inf
    return frac, med


def adaptive_selection(shot, candidates=None, weights_source="last",
                       rt_good=None, verbose=False, fit_ip=False):
    """Displacement by LOCAL round-trip acceptance with hysteresis.

    The order is static and shot-independent, read from the cached rt field. A
    sample goes to a set only if that set's LOCAL rt at this (u,v) meets
    rt_good, and the current set is retained while it still qualifies.

    Containment is still required, but implicitly: outside the hull the inversion
    is extrapolation, which shows up as a large or infinite rt, so the rt test
    subsumes the hull test rather than duplicating it. Both are kept anyway --
    the hull test is one cheap matmul and makes the failure reason legible.

    weights_source defaults to "last" (inherited), because the point of this path
    is that NOTHING needs to be ranked on the live shot. Pass "auto" to use this
    shot's own preshot weights.

    C PORT NOTE
    -----------
    This function is BATCH-SHAPED for offline use: proxies and rt lookups are
    computed vectorised over the whole shot before the loop. That is a Python
    optimisation, not part of the algorithm -- numpy vectorisation amortises
    interpreter overhead (measured single-threaded: 0.18 us/sample batched vs
    25.5 us/sample one at a time, a 139x gap that is pure per-call overhead, NOT
    parallelism; it survives OMP_NUM_THREADS=1).

    The SELECTION LOGIC is causal: the loop body reads only sample i and `cur`.
    To port, move the proxy and rt lookup inside the loop and replace _load_shot
    with the acquisition feed; no logic changes. In C the per-call overhead that
    forces the batching here disappears, so the per-sample work is what it looks
    like -- one 2xM matvec, one half-space test, one grid index, one bicubic
    evaluation. For reference, the Python single-sample path costs ~40 us against
    a 20 us period at 50 kHz; that gap is interpreter overhead, not arithmetic.
    """
    cands = candidates or DEFAULT_CANDIDATES
    rt_good = RT_GOOD if rt_good is None else rt_good
    t, ip, B = _load_shot(os.path.join("data", str(shot)))
    w = shot_weights(shot, weights_source)
    T = len(ip)

    info = {}
    dropped = []
    for s in cands:
        # A set must have at least one degree of freedom left after the fit, or
        # its score means nothing: with as many live probes as unknowns the fit
        # is exact by construction, so it round-trips perfectly and ranks FIRST.
        # That is not hypothetical -- on 1643, where curation zeroes probes 11
        # and 12, [11 12 5 6] has two live probes against two unknowns and takes
        # the top of the priority order with fit_ip=False.
        #
        # The scoring path cannot catch this by itself: good_frac and rt are
        # computed from the forward MODEL, and curation enters as a weight rather
        # than as membership, so nothing downstream knows a probe is dead.
        n_unknown = 3 if fit_ip else 2
        live = sum(1 for p in map(int, s.split()) if w.get(p, 1.0) > 0)
        if live <= n_unknown:
            dropped.append((s, f"{live} live probe(s) against {n_unknown} "
                               f"unknowns -- no degrees of freedom"))
            continue
        try:
            e = _estimator(s, w, fit_ip=fit_ip)
        except ValueError as exc:
            dropped.append((s, str(exc)))
            continue
        A, b = _hull_faces(e)
        pr = _proxy(e, ip, B)
        frac, med = rt_field_score(e)
        info[s] = {"est": e, "faces": (A, b), "proxy": pr,
                   "rt": rt_lookup(rt_field(e), pr),
                   "inside": np.all((pr @ A.T) + b <= 0, axis=1),
                   "score": frac, "med": med}

    # Static order: most of its region trustworthy first; ties broken by how good
    # it is where it IS good. Ordering only -- it changes which acceptable set is
    # picked, never whether a sample is accepted.
    cands = [s for s in cands if s in info]
    if dropped:
        print(f"[{shot}] adaptive: dropped {len(dropped)} candidate set(s) with "
              f"too few live probes for fit_ip={fit_ip}:")
        for nm, why in dropped:
            print(f"[{shot}]   [{nm}] -- {why}")
    if not cands:
        raise RuntimeError(
            f"adaptive_selection(shot={shot}, fit_ip={fit_ip}): no candidate set "
            "has enough non-zero-weight probes. With fit_ip=True a set needs "
            "three; check the curation weights for this shot.")
    order = sorted(cands, key=lambda s: (-info[s]["score"], info[s]["med"]))

    ok = {s: (info[s]["inside"] & (info[s]["rt"] <= rt_good)) for s in cands}
    dR = np.full(T, np.nan); dZ = np.full(T, np.nan)
    Ip_used = np.full(T, np.nan)
    chosen = np.full(T, -1, int)
    cur = None
    for i in range(T):
        # hysteresis: keep the current set while it remains locally acceptable
        pick = cur if (cur is not None and ok[cur][i]) else None
        if pick is None:
            for s in order:
                if ok[s][i]:
                    pick = s
                    break
        if pick is None:
            continue
        e = info[pick]["est"].ensure_phi()
        sig = np.array([B[p][i] for p in e.probes], float)
        r, z, ipu = e.shift(sig, ip[i])
        dR[i] = r; dZ[i] = z; Ip_used[i] = ipu
        chosen[i] = order.index(pick)
        cur = pick

    # NO RUNTIME COVERAGE GATE, deliberately.
    # Phase A is shot-independent: no order is inherited from another shot, so
    # there is nothing for a coverage gate to validate. Such a gate would also
    # never be actionable live, because coverage is only knowable after the last
    # sample and so cannot inform any decision during the shot.
    #
    # Nothing is lost in safety. A sample no set accepts stays NaN in dR/dZ, so
    # the failure is visible in the OUTPUT rather than in a summary statistic.
    # coverage/n_switch below are OFFLINE DIAGNOSTICS only -- reported, never
    # acted on -- and compare_methods.py reads both for its plot label.
    #
    # C PORT: everything above this line is per-sample and causal; this block is
    # the only part that touches the whole array, and it is not part of the
    # control path. Omit it entirely in the C implementation.
    n_switch = int(np.sum(np.diff(chosen[chosen >= 0]) != 0))
    coverage = float(np.mean(chosen >= 0))
    if verbose:
        for rank, s in enumerate(order):
            print(f"    rank {rank} [{s}] good-frac {info[s]['score']:.3f} "
                  f"med {info[s]['med']:.2e}  used {int(np.sum(chosen == rank))}")
    # "chosen" is the per-sample index into "order" (-1 = no set qualified).
    # A DIAGNOSTIC, like coverage and n_switch: never acted on by the selection
    # logic, but it is what makes a set-attributed plot possible.
    # Ip_used is the measured current when fit_ip=False and the FITTED current
    # when fit_ip=True, which is what makes a four-way current comparison
    # possible (IP1, IP2, filament-fitted, Biot-Savart-fitted).
    return {"t_ms": t, "dR_m": dR, "dZ_m": dZ, "coverage": coverage,
            "n_switch": n_switch, "order": order, "chosen": chosen,
            "Ip_used_A": Ip_used, "fit_ip": fit_ip,
            "provenance": f"adaptive:{weights_source}:fit_ip={fit_ip}"}



if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    do_prebuild = "--prebuild" in args
    weights = "last" if "--last" in args else ("auto" if "--auto" in args else None)
    shots = [a for a in args if not a.startswith("--")]
    if not shots:
        sys.exit("usage: python adaptive_select.py [--prebuild] [--auto|--last] "
                 "<shot> [<shot> ...]\n"
                 "  --prebuild   build and cache Phase A (Phi, hull, rt field) "
                 "for every candidate set\n"
                 "  --auto       weights from THIS shot's pre-shot window "
                 "(offline default)\n"
                 "  --last       weights inherited from a previous shot "
                 "(what real time must use)\n"
                 "e.g. python adaptive_select.py --prebuild --auto 1643")
    for shot in shots:
        if do_prebuild:
            print(f"=== prebuild shot {shot} ===")
            prebuild_shot(shot, weights_source=weights or "auto")
        ws = weights or "auto"
        r = adaptive_selection(shot, weights_source=ws, verbose=True)
        T = len(r["t_ms"])
        print(f"\n=== shot {shot}: adaptive_selection "
              f"(provenance={r['provenance']}) ===")
        print(f"  order[0]: {r['order'][0]}")
        print(f"  coverage {r['coverage']:.1%}  switches {r['n_switch']}")
        fin = np.isfinite(r["dR_m"])
        print(f"  dR: {fin.sum()}/{T} finite, range "
              f"{np.nanmin(r['dR_m'])*1e3:.0f}..{np.nanmax(r['dR_m'])*1e3:.0f} mm")
