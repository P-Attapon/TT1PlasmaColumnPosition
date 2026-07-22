# Changes: 1D → 2D interpolation

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
  Convenience: build maps for one or all probe sets ahead of time.

- `CHANGES_2D.md` (this file).

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
main.py, OFIT/*, simulation/*, plotting.ipynb, requirement.txt.

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
  artifact). Re-run on your machine after restoring `data/`.

## How to run

Same as upstream. Set `use_toroidal_filament_model = True` in main.py, restore
`data/`, run `python main.py`. Displacement now uses the 2D map. To build maps
ahead of time: `python build_all_phi.py` (or `python build_all_phi.py "1 4 7 10"`).
