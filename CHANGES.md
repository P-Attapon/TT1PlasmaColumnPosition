# Changes

This is the combined changelog for this fork of P-Attapon/TT1PlasmaColumnPosition.
It documents, in order, three layers of change on top of the paper's original
1D method: the 2D interpolation map, the M-probe weighted estimator, and the
current-channel / adaptive-selection work that followed. Each section was
written at the time the change was made; later sections sometimes revise
conclusions from earlier ones (e.g. the weight-exponent and gain-hypothesis
findings), and that revision history is kept rather than silently edited out.

For a guide to *using* the code as it stands today, see `README.md`. This file
is a record of *why* it works this way.

---

# Part 1 — 2D interpolation (replaces the paper's 1D recurrence)

Fork of P-Attapon/TT1PlasmaColumnPosition. The plasma displacement calculation
is changed from the paper's sliced-1D Taylor-polynomial correction (indexed by
the previous timestep's shift) to a single 2D inverse map Phi:(Dx,Dz)->(R,Z)
indexed by the current (Dx,Dz) only. This removes the timestep recurrence and
its error propagation. Everything else (forward model, calibration, DxDz
inversion, IO, plotting) is untouched.

Structure, filenames, and data layout are identical to the original (data
folder excluded — drop the original `data/` back in to run).

## Files ADDED

- `methods_script/toroidal_filament/phi_map.py`
  New. Builds and loads the 2D map. `build_phi(probe_set)` sweeps the physical
  displacement grid (±shift_domain, 1 mm), forward-maps each node with the
  repo's own `cal_signal` + `cal_newton_DxDz`, resamples the scattered inverse
  onto a regular (Dx,Dz) grid, saves `phi_tables/Phi_<set>.npz`. `PhiMap`
  evaluates it with a bicubic spline (kx=ky=3); `get_phi(set)` caches one per
  probe set. Boundary policy: CLAMP to the tabulated box.

- `phi_tables/Phi_1_4_7_10.npz`
  Prebuilt map for the default probe set so the repo runs out of the box.
  Other sets are built automatically on first use.

- `build_all_phi.py` (repo root)
  Convenience: build maps for one or all probe sets ahead of time. NOTE (added
  later): this targets the LEGACY 4-probe path only — see Part 2 for the
  M-probe equivalent.

## Files EDITED

- `methods_script/toroidal_filament/plasma_shift.py`
  1. Added import: `from .phi_map import get_phi`.
  2. `cal_shift(...)` — body replaced with 2D-map evaluation. Signature and
     return shape `[[R, R_unc],[Z, Z_unc]]` UNCHANGED. Changes inside:
       - `est_horizontal_shift`, `est_vertical_shift` are now UNUSED (kept for
         signature compatibility; the recurrence is gone);
       - shift = `get_phi(probe_key).evaluate(Dx, Dz)`;
       - uncertainty slots return `0.0` (the 2D map has no per-point covariance;
         the paper's sigma_f came from the 1D fit covariance). Downstream reads
         only shift[0][0] / shift[1][0], so this is safe.
       - `DxDz_method` defaults to `cal_newton_DxDz` if None is passed.
  3. Original function preserved verbatim as `cal_shift_1d(...)` for comparison
     / running the paper method. Not used by the 2D pipeline.
  4. `toroidal_filament_shift_progression(...)` — one comment added noting the
     previous-step estimate is now inert. No logic change (it calls cal_shift,
     which is now 2D).

- `methods_script/toroidal_filament/TFM.py`
  Comments only. Marked `dR_prev`/`dZ_prev` as inert (still computed and passed
  for signature compatibility; no longer affect the result). No logic change.

## Files UNCHANGED (physics / IO / rest)

parameters.py, signal_strength.py, DxDz.py, process_probe_data.py,
coefficient.py, shift_domain.py, coefficient_nested_dict.pkl, TFM.py logic,
main.py, OFIT/*, simulation/*, plotting.ipynb, requirements.txt.

## Recorded design decisions

- Physical grid ±shift_domain at 1 mm: matches the paper's coefficient grid, so
  results are directly comparable.
- Boundary: FLAG (NaN). Out-of-domain / low-Ip samples (Dx,Dz outside the
  tabulated box) return NaN instead of saturating to the domain edge, so plots
  show gaps and batch stats skip them (use np.nanmedian / np.nanmean). Bounds
  are the table's own axes self.xg/self.zg (the Dx proxy range is offset, NOT
  centred on zero) - do not compare against +/-0.1. (Earlier versions clamped;
  changed to flag for honest out-of-domain handling and comparable plots.)
- Bicubic spline: the map is built offline, so higher-order interpolation costs
  nothing at runtime and accuracy is prioritised.
- Uncertainty returned as 0.0: placeholder. If uncertainty is needed later it
  can be derived offline (e.g. local Jacobian of Phi, or residual of the
  resampling) and stored alongside the map.

## Validation done here (before shipping)

- Round-trip: known (R,Z) -> cal_signal -> 2D cal_shift recovers to ~1e-4 m
  across the domain.
- Paper (cal_shift_1d) vs 2D (cal_shift) on shots 1641/1643 gave median
  |difference| ~0.6 mm in R, ~0.05 mm in Z (the paper method's recurrence/slice
  artifact).

---

# Part 2 — M-probe generalization (weighted least squares, any M >= 2)

Applied on top of Part 1. Adds a displacement method using any number of
probes M >= 2 with per-probe weights, with the plasma current either taken
from measurement (2 unknowns) or fitted as a third unknown (cross-check mode),
plus a per-probe gain/polarity calibration hook. The original 4-probe
antipodal path is untouched and remains the default-off switch away.

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
  (all editable at the top of main.py), passed to TFM_main.
  Default mprobe_weights = "auto".
  (NOTE: the curation-tuning knobs listed here were later removed from
  main.py's config surface — see Part 3, "config consolidation".)

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
   has no gain explanation and remains unattributed - candidates were vessel
   eddy currents, unmodelled feedback-coil pickup, and filament-model
   inadequacy. (Later resolved for one large contributor — see Part 3: a dead
   redundant current channel silently corrupting the pickup subtraction.)

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

## Later revision: single grid-resolution parameter

The (dU,dV) lookup-grid resolution is no longer an independent constant (was
UV_N = 401). It is now DERIVED from PHYS_STEP via _uv_n(), so one parameter
sets both grids and they stay consistent. Rationale: the scattered (dU,dV)
points number exactly (2*shift_domain/PHYS_STEP + 1) per axis; a lookup grid
finer than that adds nodes but no information (it only interpolates between the
same points) and measurably slows the runtime spline evaluation (median eval
~4.0 us at 101 -> ~6.6 us at 1601 per axis, from cache pressure on the larger
coefficient table). So UV_N is tied to the sweep count; UV_OVERSAMPLE allows a
small nudge if the unevenly-sampled edge regions ever need it.
PHYS_STEP is now part of the Phi cache hash, so changing resolution rebuilds
the table instead of reusing a stale one. Grid-resolution sweep showed no
accuracy change to 4 decimals (synthetic RMS 0.4639/0.5189 mm across all
resolutions - grid error is ~1e-4 mm, far below the ~0.5 mm noise floor).
FD_STEP is retained as an unused reference constant (coefficients are analytic).
(NOTE: the operating values of PHYS_STEP/UV_OVERSAMPLE were changed again in
Part 3 — see "config consolidation" — to 0.0005 m / 2.0, the values actually
used by every entry point.)

## Bugfix: error-bar overlay crashed on M-probe sets

main.py error_line_overlay looked up error_dict[probe_set + direction], but
error_dict only has the 15 antipodal 4-probe keys, so any M-probe set raised
KeyError after the (successful) displacement computation. Fixed with a
dict.get() guard that skips the static error band for sets not in error_dict.
The paper's tabulated per-set error does not apply to M-probe sets anyway; a
genuine M-probe uncertainty (estimator covariance) lives in proxy space and
would need mapping through Phi to plot as an R/Z band - noted, not implemented.

## Bugfix: fitted I0 sign (fit_ip=True)

The analytic Eq. 4 coefficients were coded with the paper's written sign
(prefactor +mu*I/(2*pi*R)), but cal_signal - the forward model the rest of the
code uses - returns a NEGATIVE tangential field for positive plasma current
(its -b_r*sin + b_z*cos projection). The mismatch made the fitted I0 come out
with the wrong sign: a clean mirror image of measured Ip (correct magnitude,
negative sign). Displacement (dR,dZ) was unaffected because it comes from the
ratios x2/x1, x3/x1, in which the sign cancels - which is why only the reported
current was upside down. Fixed by negating the coefficient prefactor to match
cal_signal's convention. Verified: a known +Ip synthetic case now returns
I0/Ip = +1.000 with dR,dZ exact; real shot 1641 (12 probes) gives fitted/
measured median +1.076 (positive, right-side up). The FD-coefficient version
never had this because it differentiated cal_signal directly and inherited its
sign automatically. NOTE: the previously reported ~8-18% magnitude offset
between fitted I0 and measured Ip stood at the time as a real systematic,
separate from this sign fix — see Part 3 for a resolved contributor.

## Open finding (at the time): systematic inconsistency dominates

Measured on shot 1641, flat-top, with the then-current configuration:
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
that part was ~2% of the total at the time. The systematic part was untouched
by any weighting scheme. (Part 3 identifies and fixes one concrete, large
source of this systematic: a dead redundant current channel, found via a large
filament/AI-camera disagreement on shot 3970.)

## Files UNCHANGED (at this stage)

phi_map.py, plasma_shift.py (cal_shift / cal_shift_1d), coefficient.py,
DxDz.py, signal_strength.py, parameters.py, process_probe_data.py, OFIT/*,
simulation/*.

## Recorded decisions

- Linear model by finite differences of cal_signal rather than the analytic
  cylinder formula: inherits every convention of the repo's forward model and
  makes the proxy first-order exact for the torus (smaller Phi correction;
  proxy ~ true displacement near centre). (Superseded — see the analytic Eq.4
  revision above, which is what ships today.)
- Weights fixed per shot (curation input); P and Phi built once per shot.
- Normal-equations solve (2x2 / 3x3): adequate conditioning for this problem;
  condition number printed per set as the health check.
- Uncertainty: estimate covariance (H^T W H)^-1 is computed and stored on the
  estimator; not yet written into the output DataFrame (kept identical shape).

## Validation performed (before shipping this stage)

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

## Recommended usage until gains are calibrated (at the time)

- fit_ip=True (immune to common gain), probe sets excluding 11 and 12, or
- supply mprobe_gains from a proper calibration when curation produces one.

## Curation weights - validation on shot 1641

mprobe_weights = "auto" computed w_i = 1/sigma_i^2 from the pre-plasma window:
  clean probes (GBP1,3,4,10) sigma ~ 1-2e-4 T -> large weight;
  GBP6 (bad on all shots) sigma 7.2e-3 and GBP11 sigma 2.9e-3 -> tiny weight;
  GBP2, GBP5 GATED OUT (half-window scatter ratio 3.5-3.6 > threshold 3.0:
  genuinely non-stationary, verified).
With curation weights, M=12 fit_ip=True gives 100% valid samples (vs 5%/14%
with unit weights) - down-weighting the noisy probes recovers the trace,
consistent with the noise-limited-accuracy finding.

NOTE: the gate thresholds in curation.py (RAIL_FRAC, STRUCT_RATIO, MIN_SAMPLES)
are heuristic defaults tuned against the shots examined so far, not derived
from a full population study. Revisit against a larger shot sample before
production/batch use.

---

# Part 3 — Current-channel curation and adaptive probe-set selection

Applied on top of Parts 1-2. Two problems drove this work: (1) a single fixed
probe set can be geometrically unable to represent some shots (the linear
proxy leaves the set's valid domain and the Phi lookup returns NaN/rails), and
(2) a large, unexplained filament/AI-camera disagreement on shot 3970 turned
out to trace back to a dead current-diagnostic channel, not a probe or
algorithm fault.

## Files ADDED

- `methods_script/toroidal_filament/current_channels.py`
  Every current diagnostic on TT-1 has two channels (IP1/IP2, IT1/IT2,
  IOH1/IOH2, IV1/IV2). The code previously hardcoded one of each
  (IP1, IT1, IOH1, IV2 — the PRIMARY channels the calibration coefficients
  kt/koh/kv were fit to). On shot 3970 the IV2 integrator was dead (std ~5 A
  in-discharge vs ~1000 A on IV1), so the pickup subtraction
  `B_corrected = raw - kt*It - koh*Ioh - kv*Iv` received Iv ~ 0 and left a
  large uncorrected vertical-field pickup baked into every probe. The filament
  method faithfully fit that corrupted data and reported dR ~ +127 mm; the AI
  camera (and simple physical plausibility: that position would put the
  plasma outside the vessel wall) indicated dR ~ -35 mm. Per-probe curation
  (Part 2) cannot see this failure because it corrupts every probe identically
  rather than one probe differently from its neighbours.

  `resolve(shot_dir, base, average=True)` and `resolve_all(shot_dir)` implement,
  per channel pair, once per shot (preshot-cost only):
    1. both healthy + agree -> average (IT, IOH, IV only; NOT IP -- see below);
    2. primary dead, secondary healthy -> sign-corrected secondary;
    3. both dead -> raise RuntimeError (refuse rather than guess);
    4. both healthy but disagree badly -> WARN and use the primary.
  The sign relationship between each pair (IP1≈+IP2, IT1≈-IT2, IOH1≈-IOH2,
  IV2≈-IV1) is a fixed wiring property, declared as a table, not re-derived
  per shot. IV's sign is inferred from the fit-quality result on shot 3970
  (the only shot seen so far where IV2 was dead) and is flagged as such in the
  module docstring pending confirmation from a shot where both IV channels are
  live.
  IP is never averaged, even when both channels are healthy: IP feeds
  kappa = Ip/I_PARAM, the current normalisation, and averaging it would change
  that normalisation on every shot including ones that already work correctly.
  It falls back to the live channel only if the primary is dead.

- `methods_script/toroidal_filament/weights_cache.py`, `ranking_cache.py`
  Persist per-shot preshot curation weights and VALIDATED probe-set priority
  orders to disk, so they can be reused without recomputation. See
  `adaptive_select.py`'s module docstring for the offline/realtime design these
  support.

- `adaptive_select.py` (repo root)
  Per-timestep probe-set switching. A single probe set has one reachable
  region in the linear (dU,dV) proxy plane; where the plasma trajectory leaves
  it, Phi returns NaN/rails. Adaptive selection ranks all candidate sets by
  hull-health (fraction of the discharge each set's proxy leaves its own
  reachable region — cross-shot-validated Spearman +0.92 against full-sweep
  health) and, among close-health sets, tie-breaks by Phi self-consistency
  ("round-trip" error: invert a proxy point through Phi, push the result back
  through the exact forward model, and measure the residual — large near a
  set's domain boundary, the standard interpolation-error signature; see the
  module docstring for the literature basis and why an earlier geometric-margin
  tiebreak was replaced by this measured one). At each timestep the
  highest-ranked set whose region contains that sample is used, with a
  round-trip CEILING that excludes sets whose self-consistency error is too
  large to trust even if their coverage looks good (found via shot 2766, where
  a set with the best raw coverage produced ~44 mm/sample noise because it was
  operating near its own domain edge for a stretch of the discharge).

  offline/realtime modes: `mode="offline"` ranks and validates on the CURRENT
  shot (best possible reference = itself) and refuses to run if the ranking
  fails a safety gate (too few surviving sets, or resulting coverage too low).
  `mode="realtime"` reuses the latest VALIDATED ranking from a prior shot
  (there is no current-shot data available before the shot happens) and
  refuses if the inherited order covers too little of the live shot (the
  reference was too dissimilar) rather than silently emitting a bad result.

  IMPORTANT, EXPLICITLY REJECTED DESIGN: the AI camera is used only as an
  AFTER-THE-FACT sanity check on results in this changelog and in code
  comments, never as a selection signal. Choosing whichever probe set best
  matches the AI camera would make the AI a ground truth by the back door
  (circular), and the AI is treated throughout as an independent cross-check,
  not ground truth. A magnetics-internal tiebreak (fitted I0 vs the Rogowski
  channel) was tried and rejected for the same reason it might seem
  attractive: it looked good on the one shot it was developed against but
  showed no real signal (pooled Spearman +0.12) once tested on a shot with
  more candidate sets — see the module docstring's "TIEBREAK SEARCH" section
  for the full negative result, kept so it is not silently retried.

## Files EDITED

- `methods_script/toroidal_filament/TFM.py`
  Current-channel reads now go through `current_channels.resolve_all()` once
  per shot instead of hardcoding IT1/IOH1/IV2; the chosen provenance per
  channel is printed. `required_files` no longer hardcodes only the primary
  channels — the `*2` variants are tolerated-if-absent. Dead code from an
  intermediate version of this change (`required_files` shadow variable,
  unused `_ip1_t`) was removed.

- `methods_script/toroidal_filament/curation.py`
  Its three hardcoded loads (IT1, IOH1, IV2) replaced by the same
  `resolve_all()` call, so curation's pre-plasma weights are computed from the
  same resolved channels the displacement calculation uses.

- `methods_script/toroidal_filament/process_probe_data.py`
  Unrelated latent bug fixed while auditing this area: `sep = "\s+"` (an
  invalid escape sequence, `SyntaxWarning` in current Python and a future
  `SyntaxError`) corrected to `sep = r"\s+"`.

- `main.py`, `compare_methods.py`
  Both gained `mode`/`weights_source` plumbing for the adaptive path
  (`use_probes = "adaptive"` / `FIL_PROBES = "adaptive"`), and config-guard
  functions that raise on contradictory settings (e.g. `use_probes="adaptive"`
  with `use_mprobe=False`) rather than silently doing something unintended.
  `main.py`'s fitted-current plot (`mprobe_fit_ip=True`) now overlays the
  fitted I0 against BOTH raw Ip channels (IP1 and IP2), not just IP1 — a
  disagreement between the fit and only one of the two raw channels localises
  the fault to a Rogowski integrator rather than the magnetic fit itself.

- `position_c_displacement.py`
  DELIBERATELY left hardcoded on IT1/IOH1/IV2, unlike the filament path. This
  module is a faithful port of the real-time position.c controller, which
  reads those specific DAQ channels; if IV2 was dead on a shot, the real
  controller was also wrong in real time on that shot, and that is a result
  worth preserving for comparison, not correcting away.

## config consolidation (main.py / compare_methods.py / adaptive_select.py)

Several parameters that had settled on one fixed, documented value were
removed from the top-level config surface and now live only at their source
of truth, so main.py and compare_methods.py cannot silently drift apart:
  - `weight_power` (curation.py, WEIGHT_POWER=2.0 — the maximum-likelihood
    exponent; changing it invalidates the covariance-as-position-variance
    interpretation, see Part 2),
  - `struct_ratio`, `rail_frac`, `min_samples` (curation.py gate thresholds),
  - `phys_step`, `uv_oversample` (mprobe.py grid resolution — previously
    inconsistent between main.py, compare_methods.py, and adaptive_select.py;
    now a single source of truth at 0.0005 m / 2.0 that all three import).
`FIL_PROBES = "adaptive"` in compare_methods.py and `use_probes = "adaptive"`
in main.py now route through the same `adaptive_select.py` machinery.

## Validation

- Round-trip unit test of the M-probe estimator (known displacement ->
  cal_signal -> estimator) remains exact to <0.0001 m — confirms the
  inversion machinery itself was never the source of the 3970 discrepancy.
- Shot 3970: filament dR moved from the corrupted +127 mm (median, 12-probe
  set) to -59 mm after the current-channel fix, consistent with the AI
  camera's independent ~-35 mm and with a constrained best-fit of the
  well-behaved probes alone (-60 mm) done as an intermediate diagnostic.
- Shot 2766: the round-trip ceiling reduced adaptive dR jitter in a 5 ms
  window from ~44 mm/sample (single set operating near its domain edge,
  mistakenly top-ranked by coverage alone) to ~3 mm/sample, matching the
  smooth 12-probe reference in that window.
- The current-channel resolver's dead/healthy classification and the
  IT/IOH/IV sign table were checked against two shots (2766, 3970) where the
  "which channel is alive" answer happens to be reversed between them —
  confirms the classification is not accidentally tied to one shot's specific
  channel identities.

## Open items

- IV's sign convention (IV2 ≈ -IV1) rests on one shot's fit-quality evidence,
  not a shot with both IV channels simultaneously live. Confirm when such a
  shot is available.
- current_channels.py's DEAD_STD_FLOOR / DEAD_REL_FLOOR / DISAGREE_MAX
  thresholds are tuned on two shots (2766, 3970), same caveat as the Part 2
  curation thresholds — revisit against a larger population.
- adaptive_select.py's hull-health and round-trip-ceiling thresholds
  (HULL_MAX, ROUNDTRIP_MAX) are likewise tuned on a small number of shots.
- Real-time (in-loop) use of the adaptive selector needs the switch logic
  ported outside Python and the priority order precomputed with fixed
  (non-"auto") weights; see adaptive_select.py's module docstring,
  "REAL-TIME NOTE".

---

# Part 4 — Repository clean-up for publication

Housekeeping pass before making the repository public. No physics, no numerical
results, and no default behaviour of the filament model changed. Recorded here
so the diff is not mistaken for a method change.

**Dependency list was wrong.** `requirement.txt` was a full `pip freeze` of one
developer machine — it listed desktop packages (`PyGObject`, `cupshelpers`,
`udiskie`, `VapourSynth`, `gps`, `nftables`) and omitted **every** package the
code imports. `pip install -r requirement.txt` could not produce a working
environment. Replaced by `requirements.txt` listing what is actually imported,
split into a core group and an OFIT-only group.

**`lsq-ellipse` was never a dependency.** `from ellipse import LsqEllipse` in
`OFIT/transformation.py` was never called anywhere in the repository. Removing
that one line drops the package, the install instruction and a troubleshooting
entry. `AgglomerativeClustering` was also unused in that module — but note it
was reaching `simulation/simulation_OFIT.py` through `import *`, so that file
now imports it directly rather than relying on the leak.

**`main.py` no longer imports OFIT at module load.** It pulled in
`opencv-python` and `scikit-learn` for every run, including filament-only runs
that never touch OFIT, and most of the imported names were unused. The import
now sits inside the branch guarded by `use_calibration_plane_transformation`.

**`compare_methods.py` had a broken default shot list.** With no argument it ran
`["2766", "1641", "1643", "2766", "3616", "3970", "4405", "4047", "4048",
"4049", "4052", "4398"]` — `2766` twice, and `4405`, which is not a shot in this
project (`4404` is). It now prints usage and exits.

**`POSC_FLIP_SIGN` was one flag for two conventions.** The `"positionc"` and
`"filament"` pickup sources come out with opposite polarity against the filament
trace on every shot tested, and 4404 inverts both relative to 1643 and 2766. One
shared boolean cannot express that, so it silently mis-signed whichever source
it had not been tuned for. It is now a dict keyed by pickup source; the old
boolean form is still accepted. The measured truth table is in the comment
beside it. Note this flag only negates — it never changes the spread — so it
cannot rescue a diverging trace: on 4404 the `"positionc"` pickup has std ~9.5 m
either way, which is a calibration failure, not a sign error.

**Dead imports removed** from `phi_map.py`, `process_probe_data.py`,
`shift_domain.py`, `OFIT.py`, `local_image.py`, `transformation.py`,
`adaptive_select.py`, `simulation_OFIT.py` and `simulation_toroidal_filament.py`.

**`.idea/` removed from version control** and added to `.gitignore`. It was
committed, and `misc.xml` leaked an absolute path containing a developer's
Windows username.

**`.gitignore` fixed.** It ignored `result_plot/` while twelve comparison PNGs
under `result_plot/comparison/` were tracked — a contradiction that made
`git status` misleading. The rule now ignores generated output but keeps the
tracked reference figures.

**README corrections.** It listed `intern_compare.py`, which does not exist in
this repository, and omitted `stamp_model_caches.py`, which does. It claimed the
filament configuration block in `compare_methods.py` "mirrors `main.py`'s" —
they set up **different selectors** (see below). Install instructions rewritten.

**Hull health retired; one selection scheme remains.** `adaptive_select.py` had
grown two selectors, reached through the same `"adaptive"` setting: `main.py`
called `adaptive_displacement()` (hull-health ranking) and `compare_methods.py`
called `adaptive_selection()` (the `rt(u,v)`-field scheme). The same
configuration word therefore produced different traces depending on which script
was run.

`main.py` is now ported to `adaptive_selection()`, and the hull-health path is
**removed, not deprecated**: `adaptive_displacement`, `rank_sets`,
`health_and_margin`, `_admit`, `commit_ranking`, `resolve_ranking`,
`_roundtrip_error`, the constants `HULL_MAX` and `_REALTIME_MIN_COVERAGE`, the
`"offline"`/`"realtime"` mode switch, and `ranking_cache.py` with its
`ranking_store/` cache. 338 lines out of `adaptive_select.py`.

Note what was NOT retired: the convex hull itself stays and is still tested per
sample. What went is hull *health* — the per-shot aggregate "what fraction of
this discharge leaves this set's hull" — as a ranking key.

The reason is that the per-set round-trip scalar the old path ranked on is an
average of the rt field over wherever the plasma happened to go, and averaging
is what made it shot-dependent (cross-shot Spearman +0.41 to +0.69; set
[12 3 6 9] scores 645 mm on 1643 and 0.98 mm on 2766 with identical geometry).
Real time inherited an admission decision built from that scalar, so it
inherited the instability. Its aggregate gate was also binary and destructive:
on 4404 the best set sat outside its hull for 20.9% of the shot against a 20.0%
ceiling, so the entire shot was refused — discarding the ~79% of samples that
inverted perfectly. The field-based test asks the same question locally, per
sample, and cannot refuse a shot: an unrepresentable sample is a NaN at that
sample.

Verified numerically identical on shot 4404 before and after removal: coverage
0.9246, 6 switches, same top-ranked set.

The module docstring, the `adaptive_select.py` CLI, `main.py`'s configuration
comments, `compare_methods.py`, `cache_keys.py`, `stamp_model_caches.py` and
README Section 4 were all rewritten to match.

`rtfield_displacement`, the name this path carried while it was experimental,
is also removed rather than kept as an alias. It was bound to the same function
object, so it cost nothing to keep — but it emitted no `DeprecationWarning` and
nothing in the repository called it, so "deprecated" would have been a claim the
code did not back up. One public name, `adaptive_selection()`. Anything outside
this repository that used the old name (including the project context document)
needs the one-word rename.

**`prebuild_shot()` reported its cache state wrongly.** It evaluated
`cached = os.path.exists(PhiM_<hash>.npz)` *after* calling `_estimator()`, which
at the time built the Phi map inside `__init__` -- so the file always existed by
then and every set was reported as cached whether or not it had just been built.
It now builds all three Phase A artefacts explicitly (Phi map, convex hull, rt
field), records per-set build times, and reports `was_cached` from a check taken
before the build.

*(An earlier revision of this entry claimed the function "did not prebuild
anything". That was wrong: `MProbeEstimator.__init__` called
`_load_or_build_phi()`, so the build did happen as a side effect of constructing
the estimator. Only the reporting was broken. Part 5 makes the build explicit
rather than incidental.)*

**Dead imports removed** from `phi_map.py`, `process_probe_data.py`,
`shift_domain.py`, `OFIT.py`, `local_image.py`, `transformation.py`,
`adaptive_select.py`, `simulation_OFIT.py` and `simulation_toroidal_filament.py`.

**`.idea/` removed from version control** and added to `.gitignore`. It was
committed, and `misc.xml` leaked an absolute path containing a developer's
Windows username.

**`.gitignore` fixed.** It ignored `result_plot/` while twelve comparison PNGs
under `result_plot/comparison/` were tracked — a contradiction that made
`git status` misleading. The rule now ignores generated output but keeps the
tracked reference figures.

**README corrections.** It listed `intern_compare.py`, which does not exist in
this repository, and omitted `stamp_model_caches.py`, which does. It claimed the
filament configuration block in `compare_methods.py` "mirrors `main.py`'s" —
they set up **different selectors** (see below). Install instructions rewritten.

**Hull health retired; one selection scheme remains.** `adaptive_select.py` had
grown two selectors, reached through the same `"adaptive"` setting: `main.py`
called `adaptive_displacement()` (hull-health ranking) and `compare_methods.py`
called `adaptive_selection()` (the `rt(u,v)`-field scheme). The same
configuration word therefore produced different traces depending on which script
was run.

`main.py` is now ported to `adaptive_selection()`, and the hull-health path is
**removed, not deprecated**: `adaptive_displacement`, `rank_sets`,
`health_and_margin`, `_admit`, `commit_ranking`, `resolve_ranking`,
`_roundtrip_error`, the constants `HULL_MAX` and `_REALTIME_MIN_COVERAGE`, the
`"offline"`/`"realtime"` mode switch, and `ranking_cache.py` with its
`ranking_store/` cache. 338 lines out of `adaptive_select.py`.

Note what was NOT retired: the convex hull itself stays and is still tested per
sample. What went is hull *health* — the per-shot aggregate "what fraction of
this discharge leaves this set's hull" — as a ranking key.

The reason is that the per-set round-trip scalar the old path ranked on is an
average of the rt field over wherever the plasma happened to go, and averaging
is what made it shot-dependent (cross-shot Spearman +0.41 to +0.69; set
[12 3 6 9] scores 645 mm on 1643 and 0.98 mm on 2766 with identical geometry).
Real time inherited an admission decision built from that scalar, so it
inherited the instability. Its aggregate gate was also binary and destructive:
on 4404 the best set sat outside its hull for 20.9% of the shot against a 20.0%
ceiling, so the entire shot was refused — discarding the ~79% of samples that
inverted perfectly. The field-based test asks the same question locally, per
sample, and cannot refuse a shot: an unrepresentable sample is a NaN at that
sample.

Verified numerically identical on shot 4404 before and after removal: coverage
0.9246, 6 switches, same top-ranked set.

The module docstring, the `adaptive_select.py` CLI, `main.py`'s configuration
comments, `compare_methods.py`, `cache_keys.py`, `stamp_model_caches.py` and
README Section 4 were all rewritten to match.

`rtfield_displacement`, the name this path carried while it was experimental,
is also removed rather than kept as an alias. It was bound to the same function
object, so it cost nothing to keep — but it emitted no `DeprecationWarning` and
nothing in the repository called it, so "deprecated" would have been a claim the
code did not back up. One public name, `adaptive_selection()`. Anything outside
this repository that used the old name (including the project context document)
needs the one-word rename.

**`prebuild_shot()` did not prebuild anything.** It constructed each candidate's
estimator, hashed the configuration, and reported whether a cached Phi map
existed — but constructing an `MProbeEstimator` does not build its Phi map,
which `mprobe.py` loads or builds lazily on first use. So `--prebuild` reported
every set as uncached and left the entire cost to the first real run, which is
precisely what the flag exists to avoid. It now builds all three Phase A
artefacts (Phi map, convex hull, rt field) and records per-set build times in
the manifest.

**Open, deliberately not changed:**

- No `LICENSE` file. This repository is a fork, so the upstream licence
  constrains the choice — see README Section 8.
- The 7.8 MB thesis PDF is redistributed in the repository root; confirm that
  is permitted or replace it with a citation.
- `main.py` ships `mprobe_fit_ip = True`, while the comment beside it marks
  `False` (use measured `Ip`) as recommended. Left as-is because changing a
  default changes results; worth a deliberate decision.
- `methods_script/OFIT/*` relies on `from .parameters import *` for `np` and the
  `TT1_*` constants. It works, but the names are invisible to any static
  checker.
- `TFM.py` imports `pytest` inside its `__main__` self-test, so running that
  file directly requires pytest even though it is not a runtime dependency.


---

# Part 5 — Ordering weakness, lazy Phi, and shared forward tables

Driven by an investigation into shot 2400, where the adaptive trace disagreed
with the AI camera by ~58 mm in dZ over 343.17-348.05 ms.

**What the investigation established, before any code changed.** The disagreement
is not an acceptance-threshold problem: the chosen set's `rt` is 0.0000 mm on all
245 samples it served, and sweeping `RT_GOOD` from 1.5 mm down to 0.01 mm changes
nothing. Only two sets are admissible across that window, and they trade off
against each other -- `[12 3 6 9]` gives 10.8 mm in dR and 57.9 mm in dZ,
`[12 4 6 10]` gives 46.0 mm and 15.4 mm. An exhaustive screen of **all 4017
probe subsets of sizes 3-12** found 39 that pass the hull test on >=90% of the
window; every one tested then fails the `rt` test. Note the direction of that
result: the low-conditioning sets (cond ~2.6, versus 72.0 for the incumbent) are
precisely the ones whose map folds. Conditioning is not a proxy for usability
here, and selecting on it would have picked the worst available sets.

**good_frac's tiebreak is now radial.** The feedback coils cannot move the plasma
vertically, so dR is the axis that has to be right and dZ is diagnostic. The
ordering previously broke ties on the median isotropic round trip
`||(u,v)-(u',v')||`, which weights an axis the machine cannot act on equally with
the one it can. `_build_rt_field` now also records `|u - u'|` -- the radial
component, `dU` being the radial proxy since `hU` carries `cos(theta)` -- and
`rt_field_score` breaks ties on that instead.

**Acceptance is deliberately unchanged.** `good_frac` and `RT_GOOD` still use the
isotropic norm, because whether Phi inverts at all is not a per-axis question.
Only the ordering, which decides *which* acceptable set is used, is now aware
that the two axes are not equally important.

Measured on 2400: coverage 0.9879 and 3 switches, both unchanged; the order
changes; agreement with the camera holds at 10.8 mm dR in the disputed window and
12.3 mm over the whole shot. The change is a correction of principle rather than
a rescue of this shot -- on 2400 the old ordering already happened to pick the
better set for dR, by accident, since the isotropic tiebreak knew nothing about
either axis. The rt-field cache key is bumped to `v:2`; old two-array caches are
rebuilt rather than misread.

**Phi is now lazy.** `MProbeEstimator.__init__` called `_load_or_build_phi()`
unconditionally, so obtaining `P`, `S0`, `cond`, `cov` or a convex hull -- all
closed form in the probe angles and weights, none of which need a forward model
-- forced a full Phi build. This is why the subset screen cost 2-30 s per set.
Phi is built on first use; `ensure_phi()` forces it. Measured after the change:
constructing an estimator 0.000 s (was 0.4-60 s), a hull 0.00 s.

**Forward tables are shared instead of recomputed per set.** `cal_signal` is
evaluated per probe and depends on the forward model alone -- not on the probe
subset, not on the weights. The 12-probe table over a grid is therefore the same
for every candidate set, and everything set-specific is the 2xM projection
through `P`. Two places recomputed it per set:

- `mprobe._forward_table()` now computes the Phi-grid sweep once, caches it to
  `FwdTab_<hash>.npz` keyed on `forward_model_key()`, and each Phi build becomes
  a matmul plus the interpolation. This directly attacks the per-shot
  bottleneck: a new shot changes the weights, hence `P`, hence every Phi map --
  but not one value in this table.
- `adaptive_select._disc_table()` does the same on the hull grid. Screening all
  4017 subsets went from an estimated 2-33 hours to **2.0 s**, verified to
  reproduce `_build_hull_faces` exactly (`P`, `S0` and `cond` identical to
  machine precision, identical face counts, identical sample classification).

**Probe sets are hashed as sets, not sequences.** `_config_hash` joined
`self.probes` in its given order, so `"12 4 6 10"` and `"4 6 10 12"` -- the same
set, producing the same Phi, since `P` and the signal vector permute together --
hashed differently and built and stored the identical map twice. The hash now
sorts. `self.probes` keeps its given order, which callers rely on for indexing.

**One triangulation per Phi build, not two.** The build called
`griddata(..., method="cubic")` twice over the same scattered points -- once for
R, once for Z -- and `griddata` builds a fresh Delaunay triangulation on each
call. The triangulation is the expensive half, and it does not depend on which
value array is being interpolated. It is now built once and handed to two
`CloughTocher2DInterpolator`s, which is what `griddata` does internally; the
`nearest` fallback likewise builds one `NearestNDInterpolator` for both arrays
instead of two KD-trees. Verified **bit-identical** to the previous path:
`max|diff| = 0.000e+00` for both R and Z, with identical NaN masks.

**Measured cost of a Phi build** (314,721 grid points, 12 probes):

| step | before | after |
|---|---|---|
| `cal_signal` sweep | 27.7 s per set | 27.7 s once, shared |
| projection through `P` | -- | 0.11 s per set |
| cubic interpolation | 8.7 s per set | 4.2 s per set |
| **16 sets, new shot** | **~600 s** | **172 s** (measured end to end) |

So a new shot's full Phase A rebuild is about 3.5x faster. Note what does NOT
share: the interpolation is over the (u,v) plane, whose geometry is
set-specific and weight-specific, so every set needs its own. That 4.2 s x 16 is
the floor unless the interpolation itself changes.

**Known and not addressed:** Phi build cost still varies between sets with no
documented cause; and 34 of the 39 hull survivors above were never put through
the `rt` test (they are near-duplicates of the four that were, and all four
failed).


---

# Part 6 — The camera prediction files were the wrong shot's

A data-provenance failure, found while investigating an apparent filament defect
on shot 2766. Recorded in full because several conclusions in this file and in
the project context document were derived from the bad comparison, and because
the failure mode is one that no amount of code review would have caught.

**What happened.** Every shot folder carried a byte-identical copy of shot
**1641's** `_pred.txt` (md5 `682c28ce...`). The copies were valid detections --
correctly formatted, sensible confidences, real plasma positions -- just of a
different discharge. So nothing crashed, nothing looked malformed, and
`load_ai_camera` returned a plausible trace for every shot. Confirmed by
comparing the shared file against a correctly-exported 1641 file: `txc` identical
on all 237 overlapping frames, `tyc`/`tr` differing on one row by 0.5 px.

**What it cost.** Shot 2766 appeared to disagree with the camera by 116 mm
radially and was treated as a broken outlier for an extended investigation; with
its own detections the figure is **23.6 mm**, among the best in the set. A
detector was nearly written from scratch to explain a discrepancy that did not
exist. Every camera-scored number produced before the fix is void.

**Guard added** to `load_ai_camera`. The test is CONTAINMENT, not overlap: the
1641 copies span 285-452 ms, which contains every other shot's plasma window, so
an overlap test passes them silently -- that is precisely how this survived. A
prediction file more than 4x wider than the plasma with over 75% of its frames
outside it now prints a loud warning. Thresholds were set against the real files:
a correct file legitimately runs a little either side of the plasma (2400: 24 ms
span against a 13 ms window, 51% outside), while the 1641 copies are 166 ms
against 13 ms with 90% outside. Verified to fire on the bad file and stay silent
on all four correct ones.

**Rule adopted:** a cross-check instrument's data must be verified to belong to
the shot it is being compared against. An independent reference that is silently
the wrong shot is worse than no reference: it manufactures false defects in the
method under test and consumes the effort spent chasing them.

## Re-measured results

Adaptive vs camera, median |method - camera| in mm, `shift_domain` scan:

| shot | 0.10 m dR | 0.14 m dR | 0.16 m dR |
|---|---|---|---|
| 1643 | **23.0** | 31.8 | 31.8 |
| 2766 | **23.6** | **23.6** | 28.2 |
| 4404 | 27.9 | 15.8 | **13.1** |
| 2400 | 87.3 | **43.1** | 87.3 |
| mean | 40.5 | **28.6** | 40.1 |

`shift_domain = 0.14` is retained: best mean radial error, and the only setting
never badly wrong. But no domain wins on every shot -- 1643 and 2766 prefer 0.10,
4404 prefers 0.16 -- so 0.14 is the best COMPROMISE, not a physical optimum, and
the median across shots is nearly flat (25.8 / 27.7 / 30.0).

`RT_GOOD = 1.5e-3` is retained and is demonstrably insensitive: swept over 2000x
(0.05 to 100 mm) at every domain, coverage and switch counts do not move. The rt
distribution is bimodal -- essentially 0 or hundreds of mm -- so any threshold in
the void between behaves identically. The comment describing 1.5 mm as read off
"a natural gap" understates this: it is an arbitrary point in a very wide gap,
which is a robustness property worth knowing.

**`AI_CONF_MIN` stays at 0.5.** A proposal to raise it to 0.6 was withdrawn: it
rested on two "demonstrable misfits" at 342.0 and 343.0 ms on 2400 which were
1641 frames. On 2400's real file those times carry confidences of 0.848 and
0.892, among the best in the shot.

**Retracted: "only antipodal 4-probe sets survive both tests."** Re-running the
4017-subset search with correct data found `4 6 10 11 12` -- five probes,
non-antipodal, outside the current candidate list -- admissible on 100% of the
2400 disputed window, halving the dZ error (54.6 -> 27.8 mm) for ~15 mm of dR
(29.9 -> 44.6 mm). A real compromise candidate the earlier search missed. Note
this was a sampling error on my part, not a consequence of the bad camera data.

**Still true after re-measurement:** the dZ/dR trade-off that motivated the
radial tiebreak (§3.16 of the context document) -- on correct data
`12 3 6 9` gives 29.9/54.6 and `12 4 6 10` gives 70.6/21.1, the same structure;
the conditioning inversion (every cond ~2.6 set is admissible on 0% of the
window); and every internal measurement -- coverage, the rt=0.0000 result on
samples recovered past 0.14, the domain acting as a pure mask (0.000 mm shift on
samples solved at both domains), and all Phi build timings.

**One caveat resolved.** 0.10 and 0.16 give identical 87.3 mm medians on 2400
from *different* frame sets -- a coincidence of the median, not identical
results. The mechanism is real: `12 4 6 10` carries 107 mm median radial error on
this shot, and 0.14 is better only because it hands 10 frames to `12 3 6 9`
(29.9 mm) while 0.10 and 0.16 give nearly everything to the bad set. 0.14's
advantage on 2400 is therefore narrow and set-driven, not intrinsic to the
domain.


---

# Part 7 — Comment policy, and the candidate-list description

**Python comments no longer carry history.** Comments and docstrings in `.py`
files now describe how the code works as it stands, not what it used to do or
what was tried before. The `RETIRED` section of `adaptive_select.py`'s module
docstring, the "why there is no longer a per-shot ranking mode" block in
`compare_methods.py`, and retrospective framing in `main.py`, `mprobe.py`,
`TFM.py`, `plasma_shift.py`, `stamp_model_caches.py` and `simulation_OFIT.py`
were removed. The reasoning they contained is preserved here and in the project
context document, which is where history belongs. Nothing executable changed.

**The candidate list was being described inaccurately.** `DEFAULT_CANDIDATES` is
the 15 canonical antipodal 4-probe sets from `parameters.all_arrays` **plus the
full 12-probe array** — 16 entries, but not 16 of a kind. Comments and README
text that implied a uniform family of antipodal quads were corrected, and a note
was added at the definition stating that selection places no constraint on set
size or geometry: `adaptive_selection()` accepts any list of probe sets.

This also sharpens the retraction in Part 6. There was never an "antipodal
design" being tested — the 12-probe set is in the default list and frequently
ranks first (it is `order[0]` on shot 2400).

**`top=` removed from the compare_methods legend.** It reported `order[0]`, the
first entry of the static priority order, which is easily misread as "the set
that produced this curve". Under adaptive selection many sets contribute and the
top-ranked one may serve few samples or none. The legend now reports coverage
and switch count only; per-sample attribution is available from the `chosen`
field returned by `adaptive_selection()`.
