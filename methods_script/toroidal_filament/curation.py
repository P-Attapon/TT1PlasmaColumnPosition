"""
Layer-1 curation: per-probe weights from the pre-plasma residual.

=============================================================================
ADDED FILE. Implements the curation-workstream deliverable consumed by the
M-probe estimator:

    residual  r_i(t) = B_i^meas(t) - B_i^pred(t)      (pre-plasma window)
    B_i^pred          = k_i^t It + k_i^oh Ioh + k_i^v Iv   (repo coefficients)
    sigma_i           = std( detrend( r_i ) )         (scatter, not RMS -
                                                       a benign offset/slope
                                                       must not inflate it)
    w_i               = 1 / sigma_i^2                 (the fit weight)

q collapses to a BOOLEAN VALIDITY GATE (not a graded score): a probe is
dropped (w_i = 0) if it fails a data-integrity check that sigma cannot see -
railed/saturated samples, NaNs/dropouts, no usable pre-plasma window, or a
residual that a linear detrend cannot reduce to stationary noise (structured
/ non-stationary). Otherwise the probe is kept and sigma_i does all the
graded down-weighting through 1/sigma^2.

Design follows the curation workstream conclusion:
  "sigma_i = std of the detrended pre-plasma residual is the quality measure;
   w_i = 1/sigma_i^2; and q collapses to a boolean validity mask for
   data-integrity failures, not a separate graded score."

Weights are computed ONCE per shot (pre-plasma window only) and are then
fixed for the whole discharge, matching the estimator's per-shot-fixed-weight
assumption. Zero real-time cost.
=============================================================================
"""
import os
import numpy as np

from .parameters import calibration_coeff
from .process_probe_data import read_txt

# ---- gate thresholds (data-integrity, not quality grading) ----
RAIL_FRAC = 0.01        # >1% of pre-plasma samples at the extreme value -> railed
STRUCT_RATIO = 3.0      # half-window scatter ratio beyond this -> non-stationary
MIN_SAMPLES = 50        # need at least this many pre-plasma samples


def _detrend(x):
    """Remove mean + linear slope; return the residual scatter (std)."""
    n = len(x)
    t = np.arange(n)
    A = np.column_stack([t, np.ones(n)])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def _valid_gate(B_pre, resid_detrended):
    """Boolean validity: True = keep probe, False = drop (data-integrity fail)."""
    if len(B_pre) < MIN_SAMPLES:
        return False
    if not np.all(np.isfinite(B_pre)):
        return False
    # railed: too many samples pinned at the min or max value
    vmax, vmin = np.nanmax(B_pre), np.nanmin(B_pre)
    if vmax != vmin:
        pinned = max(np.mean(B_pre == vmax), np.mean(B_pre == vmin))
        if pinned > RAIL_FRAC:
            return False
    # structured / non-stationary: scatter of the two halves differs wildly
    h = len(resid_detrended) // 2
    s1 = np.std(resid_detrended[:h]); s2 = np.std(resid_detrended[h:])
    if s1 > 0 and s2 > 0 and max(s1 / s2, s2 / s1) > STRUCT_RATIO:
        return False
    return True


def compute_weights(shot_path, probes, discharge_current=2500,
                    pre_start_ms=0.5, pre_guard_ms=0.2):
    """Compute per-probe weights w_i = 1/sigma_i^2 from the pre-plasma window.

    shot_path         : directory with IP1/IT1/IOH1/IV2/GBP*T .txt files
    probes            : list of probe numbers to weight
    discharge_current : |Ip| threshold (A) marking discharge onset
    pre_start_ms      : skip the initial trigger transient before this time
    pre_guard_ms      : end the pre-plasma window this far before onset

    Returns dict {probe_number: weight}. Dropped probes get weight 0.0.
    Probes whose sigma is zero/degenerate but valid get the max finite weight.
    """
    def load(name):
        d = read_txt(os.path.join(shot_path, f"{name}.txt"), ["t", "v"])
        return d["t"].to_numpy(), d["v"].to_numpy()

    t, Ip = load("IP1")
    _, It = load("IT1"); _, Ioh = load("IOH1"); _, Iv = load("IV2")
    onset_idx = np.where(np.abs(Ip) > discharge_current)[0]
    t_onset = t[onset_idx[0]] if len(onset_idx) else t[-1]
    pre = (t > pre_start_ms) & (t < t_onset - pre_guard_ms)

    sigmas = {}
    valid = {}
    for p in probes:
        _, B = load(f"GBP{p}T")
        m = min(len(B), len(It), len(Ioh), len(Iv), len(t))
        prem = pre[:m]
        kt = calibration_coeff[f"k{p}t"]; koh = calibration_coeff[f"k{p}oh"]; kv = calibration_coeff[f"k{p}v"]
        resid = B[:m] - (kt * It[:m] + koh * Ioh[:m] + kv * Iv[:m])
        r_pre = resid[prem]
        B_pre = B[:m][prem]
        if len(r_pre) < MIN_SAMPLES:
            valid[p] = False; sigmas[p] = np.nan; continue
        rd = _detrend(r_pre)
        valid[p] = _valid_gate(B_pre, rd)
        sigmas[p] = float(np.std(rd))

    # convert to weights: w = 1/sigma^2 for valid probes, 0 for gated-out.
    finite_sig = [sigmas[p] for p in probes if valid[p] and sigmas[p] > 0]
    sig_floor = min(finite_sig) if finite_sig else 1.0   # cap weight for sigma->0
    weights = {}
    for p in probes:
        if not valid[p]:
            weights[p] = 0.0
        else:
            s = sigmas[p] if sigmas[p] > 0 else sig_floor
            weights[p] = 1.0 / (s * s)
    return weights, sigmas, valid
