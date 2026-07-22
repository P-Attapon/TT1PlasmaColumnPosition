# Changes: M-probe generalization (weighted least squares, any M >= 2)

Applied on top of the 2D-map version (see CHANGES_2D.md). Adds a displacement
method using any number of probes M >= 2 with per-probe weights, with the
plasma current either taken from measurement (2 unknowns) or fitted as a third
unknown (cross-check mode), plus a per-probe gain/polarity calibration hook.
The original 4-probe antipodal path is untouched and remains the default-off
switch away.

## Files ADDED

- `methods_script/toroidal_filament/curation.py`
  Layer-1 curation: computes per-probe weights from the pre-plasma residual,
  once per shot, at zero real-time cost.
    residual r_i = B_i^meas - (k_i^t It + k_i^oh Ioh + k_i^v Iv)  (pre-plasma)
    sigma_i = std( detrend(r_i) )      (scatter; detrended so a benign
                                        offset/slope does not inflate it)
    w_i     = 1 / sigma_i^2
  q is a BOOLEAN VALIDITY GATE (not a graded score): a probe is dropped
  (w_i = 0) on a data-integrity failure sigma cannot see - too few pre-plasma
  samples, non-finite values, railed/pinned samples, or a residual whose two
  halves differ in scatter by more than STRUCT_RATIO (non-stationary "drift
  that increases the noise"). Enable via mprobe_weights = "auto" in main.py.
  Matches the curation-workstream conclusion: sigma_i grades, q gates,
  w_i = 1/sigma_i^2.

- `methods_script/toroidal_filament/mprobe.py`
  The M-probe estimator. Linear model (dU, dV proxy) derived NUMERICALLY from
  the repo's own exact forward model cal_signal (finite differences at the
  centred position), so all sign/orientation conventions are inherited.
  Weighted least squares via a precomputed pseudo-inverse P (2xM or 3xM);
  per-shot condition number and estimate covariance available. Its own 2D
  correction map Phi (same construction and per-axis NaN flag policy as
  phi_map.py) built and cached per (probes, weights, gains, mode)
  configuration in phi_tables/PhiM_<hash>.npz.
  Modes:
    fit_ip=False : measured Ip scales the signals; solve 2 unknowns (dU, dV).
    fit_ip=True  : current fitted as 3rd unknown; fitted Ip returned for
                   cross-checking. Also absorbs any COMMON gain error.
  Gains: per-probe factors g_p (measured = g_p * physical); signals divided
  by g_p; negative g_p corrects polarity-flipped probes.

## Files EDITED

- `methods_script/toroidal_filament/TFM.py`
  1. TFM_main signature: added optional `mprobe: dict = None` parameter
     (docstring documents the dict). None -> behaviour identical to before.
  2. After probe-set setup: builds one MProbeEstimator per probe set when
     mprobe is given (once per shot; prints condition number per set). When
     mprobe["weights"] == "auto", first calls curation.compute_weights on the
     shot to get w_i = 1/sigma_i^2 (prints sigma/weight/gate per probe).
  3. In the per-line loop: branch - when mprobe is enabled, displacement
     comes from est.shift(signal, measured IP1); otherwise the original
     cal_shift call runs unchanged.

- `main.py`
  Added a config block next to use_probes: `use_mprobe`, `mprobe_weights`
  (dict, None, or "auto" for curation weights), `mprobe_fit_ip`,
  `mprobe_gains` (all VS Code-editable), passed to TFM_main. Default set to
  mprobe_weights = "auto".

## Files UNCHANGED

Everything else, including phi_map.py, plasma_shift.py (cal_shift /
cal_shift_1d), coefficient.py, DxDz.py, signal_strength.py, parameters.py,
process_probe_data.py, OFIT/*, simulation/*.

## Recorded decisions

- Linear model by finite differences of cal_signal rather than the analytic
  cylinder formula: inherits every convention of the repo's forward model and
  makes the proxy first-order exact for the torus (smaller Phi correction;
  proxy ~ true displacement near centre).
- Weights fixed per shot (curation input); P and Phi built once per shot.
- Normal-equations solve (2x2 / 3x3): adequate conditioning for this problem;
  condition number printed per set as the health check.
- Uncertainty: estimate covariance (H^T W H)^-1 is computed and stored on the
  estimator; not yet written into the output DataFrame (kept identical shape).

## Validation performed (container; re-run on your machine)

- Round trip (known displacement -> cal_signal -> estimator): exact to
  <0.0001 mm for M=12 and M=4, both modes; fitted Ip recovered to 0.13%.
- Full pipeline (main.py / TFM_main) runs end-to-end on shot 1641.
- REAL-DATA FINDING: raw M=12 with unit gains disagrees strongly with the
  antipodal reference. Per-probe measured/model comparison at flat-top shows
  probes 11 and 12 POLARITY-FLIPPED (ratios ~ -1.2 and -0.4) and per-probe
  gain spread ~0.77-1.52 elsewhere. The antipodal-ratio method cancels common
  gain per pair and so partially hides this; an absolute-field method exposes
  it and REQUIRES per-probe gain calibration (a curation deliverable).
  With demonstration gains (derived circularly from shot 1641's flat-top
  against the antipodal solution - NOT a calibration) M=12 measured-Ip gives
  91% valid samples and ~9-11 mm median agreement with the antipodal method.
  fit_ip=True absorbs the common gain without any gains input (M=4 agreement
  improves 55 mm -> 13 mm with no gains supplied).

## Recommended usage until gains are calibrated

- fit_ip=True (immune to common gain), probe sets excluding 11 and 12, or
- supply mprobe_gains from a proper calibration when curation produces one.

## Curation weights - validation on shot 1641 (container)

mprobe_weights = "auto" computed w_i = 1/sigma_i^2 from the pre-plasma window:
  clean probes (GBP1,3,4,10) sigma ~ 1-2e-4 T -> large weight;
  GBP6 (bad on all shots) sigma 7.2e-3 and GBP11 sigma 2.9e-3 -> tiny weight;
  GBP2, GBP5 GATED OUT (half-window scatter ratio 3.5-3.6 > threshold 3.0:
  genuinely non-stationary, verified).
With curation weights, M=12 fit_ip=True gives 100% valid samples (vs 5%/14%
with unit weights) - down-weighting the noisy probes recovers the trace,
consistent with the noise-limited-accuracy finding.

NOTE: the gate thresholds in curation.py (RAIL_FRAC, STRUCT_RATIO, MIN_SAMPLES)
are heuristic defaults. The curation workstream specifies thresholds should be
set from the good-shot population, not hardcoded - tune these against
Workstream-1's shot classification before production use.
