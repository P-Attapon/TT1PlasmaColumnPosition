# TT1PlasmaColumnPosition

Python implementation for calculating the plasma column position in Thailand
Tokamak-1 (TT-1), as a foundation for future real-time negative-feedback
control. Two independent methods are implemented:

- **Toroidal Filament Model (TFM)** — fits the plasma's magnetic signature
  (from the Mirnov probes, `GBP*T.txt`) to a 2D displacement `(dR, dZ)`, using
  a precomputed inverse map. Linear time complexity; the method with the
  actual real-time-control potential.
- **OFIT (Optical Boundary Reconstruction)** — detects the plasma edge from
  camera video and reconstructs a position from image geometry.

Two further, independent cross-checks are included for validating the
filament method against:

- **position.c** — an offline reproduction of the real-time controller's
  simple 2-probe formula (`position_c_displacement.py`).
- **AI camera** — a machine-learning plasma-centre detector's output
  (`<shot>_pred.txt`), used only as an independent sanity check, never as a
  ground truth (see `compare_methods.py` and `CHANGES.md` for why).

Theoretical background: `ANALYSIS OF PLASMA POSITION IN THAILAND TOKAMAK-1
USING TOROIDAL FILAMENT MODEL.pdf` (also in MUIC's library database).
**For the history of changes made on top of the original paper/repo, and the
reasoning behind them, see `CHANGES.md`.** This README covers how to *use*
the code as it stands today.

---

## 1. Install

Python 3.10+ recommended. Required packages:

```
pip install -r requirements.txt
```

For the filament model, position.c and adaptive selection alone, only the
first block of `requirements.txt` is needed:

```
pip install numpy scipy pandas matplotlib tqdm
```

`opencv-python` and `scikit-learn` are needed only for OFIT. `main.py` imports
OFIT lazily, inside the branch guarded by
`use_calibration_plane_transformation`, so a filament-only run does not need
them.

Checkpoint:
```
python -c "import numpy, scipy, pandas, matplotlib, tqdm"
```
should print nothing.

---

## 2. Data preparation

Create `data/<shot_number>/` for each shot you want to process (e.g.
`data/1641/`). Inside, the following files must exist:

| file | contents |
|---|---|
| `IP1.txt` | plasma current (A) |
| `IT1.txt` | toroidal-field coil current (A) |
| `IOH1.txt` | ohmic-heating coil current (A) |
| `IV2.txt` | vertical-field coil current (A) |
| `GBP1T.txt` … `GBP12T.txt` | the 12 Mirnov probe signals (T) |
| `<shot_number>.avi` | camera video (only needed for OFIT) |

These are the **primary** channels — the calibration coefficients
(`kt`, `koh`, `kv` in `parameters.py`) were derived against them.

**Redundant channels (optional but strongly recommended).** Every current
diagnostic above has a second channel: `IP2.txt`, `IT2.txt`, `IOH2.txt`,
`IV1.txt`. If present, `methods_script/toroidal_filament/current_channels.py`
uses them to:
- detect a dead/broken integrator on the primary channel (this happened on a
  real shot — see `CHANGES.md`, Part 3 — where a dead `IV2` silently corrupted
  every probe's pickup correction) and substitute the healthy channel, and
- average both channels when they're healthy and agree, halving integrator
  noise.

Without the secondary files the code still runs on the primary channels
alone, but a dead primary channel then goes **undetected** and silently
corrupts every probe (per-probe curation cannot see a fault that hits all
probes identically). If a shot's current channels are missing entirely,
displacement computation raises rather than proceeding on invalid data.

For the AI-camera cross-check, also place `<shot_number>_pred.txt` (columns
`FRAME, TIME(ms), ..., txc, tyc, tr, conf`) in the shot directory.

If a subdirectory called `imgs` does not exist inside the shot folder, all
frames from the `.avi` are extracted into it automatically on first OFIT run.

See `data/1641` (if present) for the format of working input files. **Time
steps within `.txt` files must be consistent** — check this if a shot behaves
unexpectedly.

---

## 3. Quickstart

### 3a. Run the filament model + OFIT on one or more shots

Edit the "Parameter setup" block near the top of `main.py`:

```python
shot_lst = [1641]
use_toroidal_filament_model = True
use_calibration_plane_transformation = False   # turn OFIT on/off
use_probes = [[1,2,3,4,5,6,7,8,9,10,11,12]]     # a fixed 12-probe set
```

Then:
```
python main.py
```

This saves a figure to `result_plot/calculation_result/<shot>.png` with two
panels — ΔR (radial) and ΔZ (vertical) vs time.

**Sanity-checking the output** (works for any run, not just a first one):
1. Displacement should sit within roughly ±140 mm (the model's physical
   domain). A trace pinned flat at the domain edge for long stretches means
   the fit is out of range for that probe set on that shot — try a different
   set or `use_probes = "adaptive"` (below).
2. It should move smoothly during the flat-top (steady, high plasma current)
   — sample-to-sample jaggedness there is a red flag, not noise to ignore
   (see `CHANGES.md`, Part 3, for a real case of this and its fix).
3. Large jumps right at breakdown/termination (very low plasma current) are
   expected and can be ignored.
4. No unexplained NaN gaps or flat straight-line stretches outside #1.

### 3b. Compare methods on a shot

```
python compare_methods.py 1641
python compare_methods.py 1641 1643
```

A shot number is required; run with no argument and the script prints usage and
exits rather than attempting a hard-coded list of shots you may not have.

Overlays the filament, position.c, and (if `<shot>_pred.txt` exists) AI-camera
traces on shared axes, saved to `result_plot/comparison/<shot>_compare.png`.
Each method is plotted on its own native time samples — no interpolation, so
gaps are visible rather than smoothed over. The filament configuration block at
the top of the file mirrors `main.py`'s, and under `"adaptive"` both entry
points call the same `adaptive_selection()`.

### 3c. Run position.c alone

```
python position_c_displacement.py 1641
```

---

## 4. Choosing a probe set: fixed vs. adaptive

### Fixed set (default, simplest)

```python
use_probes = [[1,2,3,4,5,6,7,8,9,10,11,12]]     # main.py
FIL_PROBES = [[1,2,3,4,5,6,7,8,9,10,11,12]]      # compare_methods.py
```

Any list of probe-number lists. With `use_mprobe = False`, sets must be one
of the 15 canonical 4-probe antipodal combinations (`print(all_arrays)` to
see them). With `use_mprobe = True` (default), any set of M ≥ 2 probes works.

A single fixed set can be geometrically unable to represent some shots — the
linear proxy leaves that set's valid domain and the lookup returns NaN or
rails at the domain edge. This is a real, observed failure mode (see
`CHANGES.md`, shot 3970/2766 cases), not a hypothetical one.

### Adaptive (per-timestep set switching)

```python
use_probes = "adaptive"      # main.py
FIL_PROBES = "adaptive"      # compare_methods.py
```

The default candidate list is the 15 canonical antipodal 4-probe sets **plus the
full 12-probe array** — deliberately not a uniform family, and the 12-probe set
often ranks first. Selection places no constraint on set size or geometry;
`adaptive_selection()` accepts any list of probe sets, so `DEFAULT_CANDIDATES` is
a default rather than a restriction.

At each timestep the estimator uses a probe set whose valid region contains
that sample, falling back to another set only where the preferred one cannot
represent the point. **The AI camera is never used to select** — see
`adaptive_select.py` for why that would be circular. It is a cross-check only,
and `load_ai_camera` now warns if a shot's prediction file looks like it belongs
to a different discharge.

Selection has two phases. **Phase A** runs before the first sample: compute the
weights, then build or load each set's Phi map, convex hull and `rt(u,v)` field,
and order the sets. **Phase B** runs per sample and is identical offline and in
real time: keep the current set while it is still inside its own hull and its
local `rt(u,v) <= RT_GOOD`; otherwise walk the static order and take the first
set that qualifies; if none qualifies, emit NaN at that sample.

The order comes from `good_frac` — the share of a set's own in-hull grid nodes
meeting `RT_GOOD` — with the median **radial** round trip `|u − u′|` where it is
good as tiebreak. The radial component is used because the feedback coils cannot
move the plasma vertically: `dR` is the axis that must be right, `dZ` is
diagnostic, and an isotropic tiebreak weights them equally. Acceptance is
deliberately *not* radial — `good_frac` and `RT_GOOD` use the isotropic norm,
because whether Φ inverts at all is not a per-axis question. Both are read from
the cached field, which contains no shot data, so **the order is known before the
shot starts and nothing is ranked on a live shot**.

Do not read `good_frac` as a set-quality score. It measures how much of a
region is trustworthy, not how bad the rest gets: two sets score 0.776 and
0.738 against round trips of 179 mm. The per-sample test is what rejects them.

**Weights are the only thing that differs between offline and real time:**

```python
FIL_ADAPTIVE_WEIGHTS = "auto"   # this shot's own pre-shot window (offline)
FIL_ADAPTIVE_WEIGHTS = "last"   # inherited from a previous shot (real time)
```

Real time must use `"last"`: the pre-shot window is too short to compute weights
live. `"last"` requires a previous run to have stored weights — see
`weights_cache.py` — and raises if the store is empty.

There is **one** selection scheme, `adaptive_selection()`. An earlier per-shot
ranking path (`adaptive_displacement`, hull-health ranking, and the
`"offline"`/`"realtime"` mode switch with a commit/validate gate) was removed;
`CHANGES.md` Part 4 records why. The experimental-era name
`rtfield_displacement` was removed with it — call `adaptive_selection()`.

To pay the one-time Phase A build cost ahead of a run, `--prebuild` builds and
caches every candidate set's Phi map, hull and rt field. Cached afterwards — the
weights enter Φ, so changing `FIL_ADAPTIVE_WEIGHTS` still invalidates the Φ maps
and rt fields and pays those again.

The expensive part of a Φ build, the `cal_signal` sweep over the grid, is shared
across every probe set and every weight choice (`FwdTab_<hash>.npz`, keyed on the
forward model alone), so a rebuild after a weights change is a projection plus an
interpolation rather than a fresh forward sweep. Measured: a full 16-set Phase A
rebuild for a new shot takes **172 s**, against ~600 s before. The interpolation
does not share — it is over the (u,v) plane, which is set- and weight-specific —
so ~4 s per set is the floor.

Note also that Φ is built lazily: constructing an `MProbeEstimator` gives you
`P`, `S0`, `cond`, `cov` and a convex hull at no cost, and only `shift()` and the
rt field force the map.

Invocations:
```
python adaptive_select.py --auto 1641             # run adaptively, print per-set diagnostics
python adaptive_select.py --last 1641             # inherited weights (the real-time case)
python adaptive_select.py --prebuild --auto 1641  # build Phase A first, then run
```
Run with no shot number to print usage.

---

## 5. The M-probe method's other settings

These apply to the **fixed-set** path only (ignored under `"adaptive"`):

```python
use_mprobe = True            # False = original 4-probe antipodal path (paper method)
mprobe_weights = "auto"      # "auto" (curation weights) | dict | None (unit weights)
mprobe_fit_ip = True         # False = use measured Ip; True = fit Ip as a 3rd unknown (cross-check)
mprobe_gains = None          # per-probe gain/polarity correction, if calibrated
```

`mprobe_weights = "auto"` computes `w_i = 1/sigma_i^2` per probe from the
pre-plasma noise (Layer-1 curation), gating out probes that fail a
data-integrity check. This is the recommended setting; see `CHANGES.md` Part 2
for the validation behind it.

`mprobe_fit_ip = True` also produces a diagnostic plot overlaying the fitted
current against **both** raw plasma-current channels (`IP1`, `IP2`) — a
mismatch with only one of the two channels points at a Rogowski-integrator
fault rather than a magnetic-fit problem.

**Not exposed at this level** (settled values, single source of truth so the
different entry points can't drift apart — see them there if you need to
change one):
- curation gate thresholds → `methods_script/toroidal_filament/curation.py`
  (`WEIGHT_POWER=2.0`, `STRUCT_RATIO=6.0`, `RAIL_FRAC=0.01`, `MIN_SAMPLES=50`)
- Phi-map grid resolution → `methods_script/toroidal_filament/mprobe.py`
  (`PHYS_STEP=0.0005`, `UV_OVERSAMPLE=2.0`)
- current-channel health thresholds →
  `methods_script/toroidal_filament/current_channels.py`
  (`DEAD_STD_FLOOR`, `DEAD_REL_FLOOR`, `DISAGREE_MAX`)

---

## 6. Directory structure

```
main.py                     run TFM + OFIT on a list of shots
compare_methods.py          overlay filament / position.c / AI-camera on one shot
position_c_displacement.py  standalone position.c reproduction
adaptive_select.py          adaptive probe-set selection (rt field, prebuild)
build_all_phi.py            prebuild Phi maps for the LEGACY 4-probe path (see its docstring;
                             not the same cache as the M-probe/adaptive path)
time_cal_shift.py           timestamp calibration utility
stamp_model_caches.py       stamp/inspect the cache keys of the model caches

data/<shot>/                per-shot experimental data (see Section 2)
result_plot/                all output figures
  calculation_result/       main.py output
  comparison/                compare_methods.py output
  edge_detection/           OFIT per-frame edge-detection images (if enabled)

methods_script/
  toroidal_filament/
    TFM.py                  orchestrates a full shot: reads files, dispatches to
                             cal_shift (legacy) or the M-probe estimator, writes output
    parameters.py           physical/geometric constants, probe angles, calibration coeffs
    process_probe_data.py   file IO and signal calibration helpers
    curation.py             Layer-1 per-probe weight computation (pre-plasma noise)
    current_channels.py     redundant current-channel health check / resolution
    mprobe.py                the M-probe weighted-least-squares estimator + its Phi map
    phi_map.py               legacy 2D inverse map (4-probe antipodal path)
    plasma_shift.py          cal_shift (2D map) / cal_shift_1d (original paper method)
    weights_cache.py         per-shot probe-weight persistence
    coefficient.py, DxDz.py, signal_strength.py, shift_domain.py   forward-model /
                             geometry support
  OFIT/
    OFIT.py                  orchestrates the optical boundary reconstruction
    parameters.py             ROIs and other OFIT-specific constants
    transformation.py, local_image.py, extract_frames.py, detection_projection.py

simulation/
  simulation_toroidal_filament.py, simulation_OFIT.py   runtime/error analysis

plotting.ipynb              exploratory plotting notebook
CHANGES.md                  full changelog and design-decision history
```

---

## 7. Possible improvements

**Toroidal Filament Model.** Currently relies purely on plasma current `I_p`.
The codebase in `methods_script/toroidal_filament/TFM.py` could be extended
to incorporate the plasma density profile (`I_p = ∫ j dA`). Magnetic-field
contribution from vessel eddy currents induced by the plasma is also
unmodelled and worth investigating — plausibly part of the systematic
disagreement documented in `CHANGES.md`, Part 2/3.

**Optical Boundary Reconstruction.** Currently runs edge detection on
full-HD (1920×1080) frames, which is slow; reducing resolution would need a
correspondingly adjusted pixel-to-world conversion factor. Also worth
exploring: incorporating real camera calibration parameters once known,
alternative edge-fitting algorithms (GradientBoosting, Hough transform), and
PCA for dimensionality reduction before fitting.

**Current-channel curation** (`current_channels.py`). The IV sign convention
and the health thresholds are validated on only two shots; both should be
revisited against a larger shot population. See `CHANGES.md`, Part 3,
"Open items" for the full list.

**Adaptive selection real-time readiness.** The switching logic itself is
causal and cheap, but running it inside the actual feedback loop needs the
Phi maps precomputed with fixed (non-`"auto"`) weights and the selection logic
ported outside Python. See `adaptive_select.py`'s module docstring,
"REAL-TIME NOTE".

---

## 8. Licence and provenance

**No licence file is present in this repository yet.** Without one, GitHub's
default applies: all rights reserved, and nobody may reuse the code. Add a
`LICENSE` before publishing.

Choosing one is not free here — this repository is a **fork of
P-Attapon/TT1PlasmaColumnPosition** (see `CHANGES.md`), so the upstream
licence constrains what may be applied to the derived work. Check the upstream
repository's terms first; if it carries no licence either, the original
author's permission is needed before publication.

`ANALYSIS OF PLASMA POSITION IN THAILAND TOKAMAK-1 USING TOROIDAL FILAMENT
MODEL.pdf` (7.8 MB) is redistributed here for convenience and is also in
MUIC's library database. Confirm redistribution is permitted, or replace the
file with a citation and a link.

The experimental data under `data/` is not distributed with the code, and
`.gitignore` excludes it. Confirm the shot data is cleared for release before
adding any of it.

---

## 9. Troubleshooting

- `*** WARNING: <shot>_pred.txt spans ... may belong to a DIFFERENT SHOT ***` →
  the AI camera prediction file in that shot's folder does not match the shot's
  plasma window. Every folder once carried a copy of shot 1641's file; the copies
  are well-formed and plausible, so nothing else detects them. Do not trust any
  camera comparison until the correct file is in place (see `CHANGES.md` Part 6).
- `ModuleNotFoundError: No module named 'cv2'` / `'sklearn'` → these are OFIT
  dependencies: `pip install opencv-python scikit-learn`. They are not needed
  for the filament model, position.c or adaptive selection.
- `FileNotFoundError: ...Phi_...npz` (legacy path) → run
  `python build_all_phi.py "<probe set>"`.
- `RuntimeError: ... BOTH channels dead` → both the primary and secondary
  reading of a current channel are at noise level for that shot; check the
  raw `.txt` files, or that shot's DAQ log.
- `KeyError` on a probe set in `main.py`'s error-band overlay → the paper's
  tabulated per-set error only covers the 15 canonical 4-probe antipodal sets;
  M-probe/adaptive sets skip that overlay by design (see `CHANGES.md`, Part 2,
  "Bugfix: error-bar overlay crashed on M-probe sets").
- Adaptive coverage well below 100%, or NaN gaps in the trace → no candidate set
  was locally acceptable at those samples. This is the designed behaviour, not a
  failure: adaptive selection never refuses a whole shot. Run with `verbose=True`
  to see each set's `good-frac` and how many samples it served.
- `RuntimeError: weights='last' but nothing stored` → `"last"` inherits weights
  from a previous run and the store is empty. Run once with `"auto"` first, or
  pass `"auto"` for offline work.
- Plot looks empty / all-NaN → check the shot's data actually exists in
  `data/<shot>/` and that current channels aren't both dead for that shot
  (see the `[current_channels]` log lines printed during the run).
