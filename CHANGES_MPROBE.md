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
  `mprobe_gains`, plus curation tuning `mprobe_weight_power`,
  `mprobe_struct_ratio`, `mprobe_rail_frac`, `mprobe_min_samples`
  (all VS Code-editable), passed to TFM_main. Default mprobe_weights = "auto".

## Later revisions (after first validation)

1. Stationarity gate made ONE-SIDED and threshold raised 3.0 -> 6.0.
   Rationale: only GROWING noise is a fault. The pre-plasma window ends just
   before breakdown, so a probe whose noise is rising is degrading into the
   phase that matters; noise that SHRINKS is a settling transient already over
   before plasma starts, and should not be penalised. The old symmetric test
   at 3.0 dropped GBP2 (ratio 3.52) and GBP5 (3.64) - but the whole 12-probe
   population on shot 1641 spans only 1.29-3.64, so 3.0 was cutting into the
   normal population rather than isolating faults. At 6.0 all 12 probes pass
   and GBP2 (the quietest channel) is restored, improving the synthetic
   accuracy gain of ALL12 over best-4 from 1.20x to 1.85x in R.

2. Weight exponent made adjustable (WEIGHT_POWER / mprobe_weight_power),
   DEFAULT KEPT AT 2.0. w = 1/sigma^2 is the maximum-likelihood weighting for
   independent zero-mean Gaussian errors and the only exponent for which
   Cov = (H^T W H)^-1 is a genuine position covariance. Documented guidance:
   leave at 2.0 unless there is strong evidence the dominant error is
   non-random. Power sweep on shot 1641 (1.0/1.5/2.0/2.5): pure-noise accuracy
   is flat and best at 2.0 (0.436 mm R); an injected per-probe bias favoured
   1.5 (0.570 vs 0.742 mm R); real-data cross-set spread improved monotonically
   with power (66 -> 33 mm) but that is partly sets converging on shared
   probes, not evidence of accuracy. A leave-one-out test found 1.5 was NOT
   more robust (worst-probe sensitivity 25.7 mm vs 22.0 mm at power 2), so the
   theoretical case for a lower exponent is not backed by measurement.

3. Gain hypothesis RETRACTED. The GBP*T channels are the INTEGRATED Mirnov
   probes, reported in Tesla, so per-probe gain is already applied in the data.
   (A separate non-integrated set is reported in Volts and would need gains;
   this workstream does not use it.) The `mprobe_gains` hook remains but should
   be left at None for this data. The probe-to-probe inconsistency therefore
   has no gain explanation and remains unattributed - candidates are vessel
   eddy currents, unmodelled feedback-coil pickup, and filament-model
   inadequacy, none yet distinguished.

## Later revision: analytic Eq. 4 coefficients (replaces finite differences)

The linear-model coefficients S0, hU, hV in mprobe.py are now the closed-form
Eq. 4 (paper) cylinder coefficients:
    S0_i = mu*I/(2*pi*R),  hU_i = mu*I*cos(theta_i)/(2*pi*R^2),
    hV_i = mu*I*sin(theta_i)/(2*pi*R^2)
replacing the previous central finite differences of cal_signal. Motivation:
remove finite-difference truncation error and use the paper's analytic form.

Note this is a genuinely different (cylindrical / infinite-R0) linear model,
not merely an exact form of the previous one - the FD version differentiated
the exact toroidal forward model, which does not equal Eq. 4 probe-by-probe.
Consequence measured: end-to-end accuracy is essentially unchanged because the
Phi map absorbs the coordinate choice. Synthetic accuracy at measured noise
(all-12 vs best-4):
    FD coeffs : all-12 R RMS 0.436 mm, Z 0.500 mm  (improvement 1.85x / 1.40x)
    Eq4 coeffs: all-12 R RMS 0.464 mm, Z 0.519 mm  (improvement 1.92x / 1.35x)
Round-trip recovery is exact for both (<1e-4 m). Eq4 has a slightly higher
condition number (4.2 vs 3.5) and ~6% worse noise-limited accuracy, both
negligible against the systematic error that dominates real shots.

Clarifying comments were also added in mprobe.py explaining (a) how S0/hU/hV
relate to Ip via kappa = Ip/I_PARAM, and (b) why I0 can be fit by linear least
squares despite Eq. 4 being bilinear (solve for the products (I0, I0*dU,
I0*dV), then divide).

## Open finding: systematic inconsistency dominates

Measured on shot 1641, flat-top, with the final configuration:
  - in-plasma fit residual is 6-24x the pre-plasma noise for every probe set;
  - the three curation-valid 4-probe sets disagree by ~42 mm (R) and ~76 mm (Z),
    against ~1 mm predicted from noise alone - a factor ~40;
  - leave-one-out on the 12-probe fit moves the answer by a median 3 mm and up
    to 22-26 mm, where noise alone predicts ~0.5 mm;
  - sensitivity does not track weight (GBP7 carries 3% of the weight but moves
    the answer 15-22 mm), so it is geometry and probe disagreement, not
    weighting, that dominates.
Consequence: the M-probe weighted method improves the NOISE-limited part of the
error (1.4-1.85x better than best-4 in synthetic tests at measured sigma), but
that part is currently ~2% of the total. The systematic part is untouched by
any weighting scheme and is the limiting factor on real shots.

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
