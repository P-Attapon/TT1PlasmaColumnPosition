# Intern guide: check the 2D displacement code and compare it to the paper's 1D method

You've been given a modified copy of the TT-1 plasma-position code. The only
change is *how* the displacement is calculated: the paper's method fits sliced
1D polynomials that are indexed by the **previous** time step's position (a
recurrence); the new version replaces that with a single **2D map** indexed by
the **current** measurement. Your job is to (1) get the new code running,
(2) sanity-check by eye that it produces sensible displacement, and
(3) compare its accuracy to the old method in a controlled way.

You do not need to understand the physics to do this. Follow the steps.

---

## Part 0 — Setup (once)

1. Install Python (Anaconda is fine) and these packages:
   `numpy scipy pandas matplotlib opencv-python tqdm`.
2. Put the shot data in place. Inside the repo there is a `data/` folder with a
   placeholder file. Copy the real shot folders (`1641`, `1643`, `2766`) into
   `data/` so you have `data/1641/IP1.txt`, etc. Ask the team for the data if
   you don't have it.
3. Open a terminal in the repo's top folder (the one containing `main.py`).

**Checkpoint:** `python -c "import numpy, scipy, pandas, matplotlib, cv2, tqdm"`
prints nothing (no error). If it errors, install the missing package.

---

## Part 1 — Run the 2D code on one shot

The 2D map has to be built once per probe set before first use. It builds
automatically the first time, but build it explicitly so you see it work:

```
python build_all_phi.py "1 4 7 10"
```

This should print `building Phi for [1 4 7 10] ...` and then a path ending in
`Phi_1_4_7_10.npz`. That file is the 2D map. (A prebuilt copy already ships in
the repo; running this just regenerates it.)

Now run the main pipeline. Open `main.py` and set these lines near the top:

```python
shot_lst = [1641]
use_toroidal_filament_model = True          # was False — turn the TFM on
use_calibration_plane_transformation = False # turn the camera method off for now
use_probes = [[1,4,7,10]]
```

Then:

```
python main.py
```

**Checkpoint:** it prints the peak plasma current and saves a figure to
`result_plot/calculation_result/1641.png`. Open that image.

---

## Part 2 — Eyeball check: does the output look sensible?

Look at the saved `1641.png`. It has two panels: ΔR (radial) on top, ΔZ
(vertical) below, versus time. Ask yourself these questions — you're checking
for *obvious wrongness*, not precision:

1. **Is the displacement in a physical range?** It should sit within about
   ±0.1 m (±100 mm). If the curve pins to a flat line at ±0.1 for long
   stretches, that's the model hitting its domain edge (the "clamp") — a little
   is fine, mostly-flat is suspicious.
2. **Is it smooth in the flat-top?** During the steady middle part of the
   discharge (where plasma current is high and roughly constant), the position
   should move smoothly, not jump around wildly sample-to-sample.
3. **Does it start near zero and evolve?** Big instantaneous jumps at the very
   start/end (where plasma current is tiny) are expected and can be ignored —
   the measurement is unreliable when there's almost no plasma.
4. **No NaNs / gaps / straight vertical lines.**

If all four look OK, the new code is running correctly. If something looks
badly wrong, note *which* question failed and show the team before continuing.

**Cross-check against the camera (optional but recommended).** Re-run with
`use_calibration_plane_transformation = True`. This overlays an independent
position estimate from the camera edge-detection (black points). The TFM curve
should roughly follow the camera points in the flat-top — they won't match
perfectly (different methods), but they shouldn't disagree wildly.

---

## Part 3 — Systematic accuracy comparison (2D vs 1D)

Eyeballing tells you the code *runs*; it does not tell you the 2D method is
*more accurate*. For that you need a case where the true position is known.
On real shots the true position is unknown, so we use a **synthetic test**:
invent a known position trajectory, compute what the probes *would* read for it
(using the exact field model), then feed those readings to each method and see
how well each recovers the position we started from. Lower recovery error =
more accurate method.

A script `intern_compare.py` is included that does exactly this. Read its top
comment, then run:

```
python intern_compare.py
```

It will:
- create three known trajectories (a fast wiggle, a sudden step, a slow ramp);
- simulate the probe signals for each;
- recover position with **both** the 1D (paper) and 2D (new) methods;
- print a table of recovery error (RMS and worst-case, in mm) for each;
- save overlay plots (`intern_compare_plots/`) showing truth vs both methods.

**How to read the result:**

- Look at the **clean (no-noise)** table first. This isolates the *method's own*
  error. The 2D column should be near zero (well under 0.1 mm). The 1D column
  will be larger (tenths of a mm up to a few mm), worst on the **step**
  trajectory — that's the recurrence lagging when the position jumps. Seeing
  2D ≈ 0 while 1D has visible error, on identical data, is the core evidence
  that the 2D method removes a real error source.
- Open `intern_compare_plots/compare_clean.png`. On the step panel, the 2D line
  should sit on the black truth line while the 1D line overshoots at each jump.
  That single picture is the clearest demonstration.
- Then look at the **noisy** tables. With measurement noise added, the two
  methods get much closer — noise, not the method, now dominates. The honest
  conclusion is: **2D is exactly right when signals are clean and never worse
  when they're noisy.** Do not claim more than that.

**What to report back:** the clean-case error table, the step plot, and one
sentence: whether 2D's clean-case error is negligible and whether it is ever
worse than 1D in the noisy cases. If 2D is ever meaningfully worse, flag it.

---

## Important honesty notes (read before writing anything up)

- This synthetic test simulates signals with the same exact-field model the 2D
  map is built from, so the clean-case "≈0" means "no self-inconsistency," not
  "0 mm on a real plasma." It does **not** include real-world effects (vessel
  eddy currents, coil pickup, the plasma not being a single filament). Those
  hit **both** methods equally and are not what this test measures.
- The comparison between the two methods is fair: both use the same forward
  model, same probe set, same data. What differs is only the inversion.
- If you change the probe set in `intern_compare.py`, build that set's 2D map
  first (`python build_all_phi.py "2 5 8 11"`).

---

## If something breaks

- `FileNotFoundError: ...Phi_...npz` → run `python build_all_phi.py "<set>"`.
- `KeyError` with a probe set → the set isn't one of the 15 valid ones; use the
  default `1 4 7 10` or ask the team.
- Path error mentioning `/home/piti-archlinux/...` → a leftover hardcoded path
  in an unrelated script; ignore unless it's in the script you're running.
- Plot looks empty → check the shot data actually copied into `data/<shot>/`.
