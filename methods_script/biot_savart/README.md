# Biot–Savart NLSQ displacement

Fits `(dR, dZ)` directly to the measured probe signals by nonlinear least
squares against the exact circular-filament field. No `(dU, dV)` proxy, no
third-order polynomial, no Φ table, no interpolation grid, no hull, no validity
domain.

```bash
python -m methods_script.biot_savart.selftest    # do this first
python biot_savart_main.py                       # edit parameters in the file
python -m methods_script.biot_savart.cli 1641    # or pass them as flags
python compare_methods.py 1641                   # overlay against the other methods
```

## What this is, and what it is not

**A reference implementation of the filament method with the linearisation
removed.** Not a fourth independent measurement, and it must not be presented as
one.

Removed — four sources of approximation error, all internal to our pipeline:

| removed | what it was |
|---|---|
| Eq. 5 proxy | the cylindrical `R₀ → ∞` linearisation to `(ΔU, ΔV)` |
| Eq. 6 polynomial | the third-order fit mapping the proxy back to `(ΔR, ΔZ)` |
| Φ tabulation | 0.5 mm grid, bicubic interpolation, and the `shift_domain` limit that comes with it |

Shared, and therefore invisible to this method:

- the probe calibration `k_t, k_oh, k_v` and the vacuum-field subtraction
- the single-filament ansatz itself — no current-density profile
- eddy currents and localised field perturbations
- probe positions, orientations, and the curation weights

**Not removed, because Φ removed it already.** The paper's Eq. 6 retrieves its
polynomial coefficients using *tₙ₋₁'s answer*, so error propagates sample to
sample. The 2D Φ map replaced that with a joint inversion of `(u, v) → (ΔR, ΔZ)`
— the simultaneous fit Eq. 6 called computationally infeasible, made feasible by
tabulating it — so `mprobe.shift()` is already stateless. This method is stateless
too, by per-sample optimisation rather than by tabulation. The two are equal here;
neither carries Eq. 6's propagation. Only the legacy 1D path
(`coefficient.py` / `plasma_shift.py`) still has that structure.

There is now no warm start at all: the grid search solves each sample from
scratch, so statelessness is structural rather than something to verify. T6
checks the related property that the exhaustive search reaches the same minimum
a perfectly seeded descent would.

So agreement with the filament method means *the linearisation is not the
problem*. It cannot vouch for anything the two share.

## Sensitivity to bad probes

The Φ path projects the measurement onto two or three linear combinations,
`P @ y`, which discards most per-probe error. This method compares absolute field
levels probe by probe, so a probe that disagrees with the model enters the fit
directly. That is a structural difference between the two, independent of what
causes any individual probe to disagree.

Measured on shot 1641 at mid-discharge, at the fitted position:

| probe | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| meas/pred | 1.08 | 1.03 | 0.93 | 0.97 | 1.09 | **1.71** | 1.03 | **0.83** | **0.67** | 1.00 | **−1.85** | **−0.28** |
| curation weight | 0.91 | 6.47 | 2.62 | 0.81 | 0.27 | **0.001** | 0.19 | **0.023** | **0.014** | 0.68 | **0.003** | **0.018** |

Probes 11 and 12 read the opposite sign to the model. The discrepancy is real
and is not a displacement artefact.

**Curation already handles it.** The five worst-fitting probes (6, 8, 9, 11, 12)
are the five lowest-weighted, by two to three orders of magnitude. Curation
derives those weights from pre-plasma variance, with no reference to any forward
model, and lands on the same set — two independent routes agreeing these probes
are bad. That is why the fit survives without any per-probe correction.

**The cause is not established.** Candidates include wiring polarity, a failed or
disconnected probe, and gain. Integrated Mirnov gain is taken to be 1, so a gain
explanation would require that to be wrong for these channels specifically; a
simple probe failure requires nothing. No per-probe correction factor is applied
or exposed by this package.

## Reading the output

Everything is SI. Millimetres appear only in printed summaries, CSV and plots.

| field | meaning |
|---|---|
| `dR_m`, `dZ_m` | displacement, metres |
| `resid_norm` | weighted RMS **fit residual** ÷ the shot's median \|B\|. Dimensionless and comparable across samples and shots. Weights are rescaled to mean 1 first, so the absolute weight scale does not enter. A model-versus-measurement mismatch, not a noise level — see below. |
| `at_wall` | the best fit inside the chamber lies on the 0.20 m limiter radius. A statement about the shot, not a solver failure. |
| `n_minima` | distinct minima found. From the grid search this is a property of the residual surface, so > 1 genuinely means more than one filament position reproduces the signals. From the phi search it is always 1 and says nothing. |
| `spread_m` | how far apart those minima are |
| `sigma_R_m`, `sigma_Z_m` | per-sample uncertainty in metres, from the fit residual — see below |
| `amp` | fitted overall amplitude (see below) |
| `gated` | excluded by the Ip gate, not a failed fit |

**Nothing is dropped for fitting badly.** A bad fit is reported as a bad fit;
acceptance thresholds belong to the caller.

### The amplitude: `fit_ip`

`cal_signal` is evaluated at `parameters.I = 1e5 A`, and
`adaptive_select._proxy` normalises measured signals as `sig / (ip / I)`. So a
filament at measured current `ip` produces `cal_signal(...) · ip / I_ref`.

`fit_ip` has the same meaning here as in `mprobe.MProbeEstimator`. With
`fit_ip=True` the overall amplitude is eliminated analytically at every
`(dR, dZ)`:

```
alpha* = Σ wᵢ mᵢ pᵢ / Σ wᵢ pᵢ²
```

so only the *ratios* between probes enter the fit. `alpha` then becomes an
output that should equal `ip / I_ref` exactly — not merely correlate with it.

**It does not.** On 1641/1643/2400 the ratio comes out at 1.156 / 1.176 / 1.147
with IQR 0.04–0.09. A consistent ~15% excess across three shots with that little
scatter is a calibration-scale effect, not noise. Unexplained; recorded here
because this method is the only one that measures it.

`fit_ip=False` fixes the amplitude at `ip / I_ref` instead, so the absolute level
enters the fit and a calibration error surfaces as a *position* error. Running
both localises a problem to the amplitude or to the geometry.

**Note the comparison caveat.** `adaptive_select._estimator` hardcodes
`fit_ip=False`, so the adaptive Φ path runs in fixed-amplitude mode while this
package defaults to `True`. A `|ΔR_BS − ΔR_Φ|` figure taken with mismatched
`fit_ip` is not like-for-like, and given the ~15% amplitude excess the difference
is not negligible — on 1641 all-12 moves from ΔR +10 mm to +31 mm between the two
modes. Set `fit_ip = False` before quoting that comparison. (`main.py` uses
`mprobe_fit_ip = True` for its fixed-set path while `compare_methods.py` uses
`FIL_FIT_IP = False`, so the repository is already inconsistent here.)

**Degrees of freedom.** With M probes, `fit_ip=True` leaves M − 3 and
`fit_ip=False` leaves M − 2. On a 4-probe set that is 1 against 2, which matters:
see below.

## Comparing against the filament method

`compare_methods.py` has `BS_FOLLOW_FILAMENT`. With it set, this method takes
both the probe set and `fit_ip` from the filament method, so the two curves
differ by the **inversion alone**. Any remaining gap is then the `(u, v)` proxy
and the Φ map, not configuration — which is the whole point of running it.

| `FIL_PROBES` | what is followed |
|---|---|
| `"adaptive"` | the per-sample choice, switching set when the filament path switches |
| a list of sets | `FIL_PROBES[0]`, the one `load_filament` plots (a message names it when more are listed) |

This is a comparison aid, not a way to run the method. In the adaptive case the
selection is made by the filament path's criteria on its own Φ maps; nothing
here validates it, and following it does not make the sets right.

**`BS_FIT_IP` is overridden, not just checked.** Matching the probes alone would
leave the amplitude treatment as a second difference, so the flag copies
`fit_ip` as well: `False` when the filament path is adaptive (`adaptive_select`
hardcodes it), `FIL_FIT_IP` when it is fixed and goes through mprobe, and left
alone on the legacy path, where there is no `fit_ip` to copy. Every override is
printed, so a run never differs silently from the config block.

This matters rather than being tidiness: a 4-probe set with `fit_ip=True` has one
degree of freedom left, and most samples then hit the bound.

How much this matters, on 1643 (where the filament path uses `2 3 8 9` for 3583
of 3606 samples):

| Biot–Savart configuration | ΔR vs filament, median | corr | slope |
|---|---|---|---|
| all 12, `fit_ip=True` | 31.7 mm | 0.990 | — |
| follows filament, `fit_ip=False` | **2.6 mm** | 0.990 | 0.996 |

Most of what looked like Φ approximation error on that shot was configuration
mismatch. Note the corollary: an unmatched comparison **overstates** the
approximation error, so the earlier figures were pessimistic, not optimistic.

## Search: finding the global minimum

The forward model at a lattice point does not depend on which sample is being
solved — only the measurement does. So the model is evaluated once per shot on a
1 mm lattice covering the chamber, and the residual at every lattice point for a
given sample is one matrix–vector product. With the amplitude profiled out,
minimising the residual is the same as maximising `(m·p)²/(p·p)`, a normalised
cross-correlation against each template.

Every lattice-local minimum is then refined by a continuous least-squares solve,
so the lattice chooses which basins to examine and never limits the accuracy of
the answer. Subject to it being fine enough to place a point in every basin, the
result is the **global** minimum inside the chamber. Refine the step and confirm
the set of minima does not change — that is the test, and it has not been run at
several resolutions yet.

Cost is ~4 ms/sample on all 12 probes, with a ~1 s, 6 MB template build per shot
per probe set — roughly 30 s for a 5,900-sample shot. Three things keep it there:
the templates are stored as float32 and the lattice buffers are reused, so the
per-sample cost is arithmetic rather than allocation; the local-minimum test is
an eight-way `np.minimum` reduction over shifted slices rather than a general
filter; and the refinement is a short Gauss–Newton iteration rather than a
general-purpose solver, since the grid already supplies a point within half a
lattice step. Gauss–Newton falls back to `least_squares` if it fails to converge.

float32 costs about seven significant figures in the residual *surface*, which
only has to rank basins; the position comes from the double-precision
refinement, so the answer is unaffected. Checked directly: over 400 real samples
the Gauss–Newton and `least_squares` refinements of the same basin agree to a
median of 3e-7 mm and a maximum of 2e-5 mm.

**There is no warm start.** Every sample is solved independently of every other,
so no question of drift or start order arises.

### `search = "phi"`

Refines a single point: the filament method's own answer, with no fallback to
the grid — a poor fit from that start is the finding, not something to retry
away. `compare_methods.py` reaches it through `BS_PLOT_PHI_START`, which plots
both curves.

Running both separates a Φ-versus-Biot–Savart gap into two parts:

| grid vs phi-seeded | reading |
|---|---|
| same position | Φ is in the global basin; the whole gap to the filament curve is proxy error |
| differ, phi residual **higher** | Φ's proxy landed in a non-global basin; part of the gap is branch mismatch, not proxy magnitude |
| differ, residual **equal** | genuine degeneracy — the data does not distinguish them |

Measured so far:

- **1641**, all 12, `fit_ip=False`: identical on all 985 sampled points, 0 worse-basin cases. Φ is in the global basin throughout, so the whole gap is proxy error.
- **2400**, following the filament's 4-probe sets: 731/736 identical, **5 samples where the Φ start sits in a strictly worse basin**, separated by up to 398 mm; 0 equal-residual degeneracies.

So branch mismatch is real but rare, and large when it happens.

## The uncertainty band

`sigma_R_m` and `sigma_Z_m` come from the fit residual at the reported position:

```
C = s² (JᵀWJ)⁻¹ ,   s² = ‖r‖² / (M − p)
```

with `p = 3` when the amplitude is profiled out and `2` when it is fixed. The
Jacobian carries the units, turning a residual in Tesla into a displacement in
metres.

`sigma_shot()` evaluates this at *any* position, not only at a minimum, so the
filament method gets its band the same way: its answer is pushed through the
same residual and the same Jacobian. A method whose answer fits the probes worse
gets a wider band, which is the intended behaviour. `compare_methods.py` shades
±`BAND_SIGMAS` (default 1σ) behind both magnetic curves when `PLOT_BAND` is set.

**It is a conditioning measure, not a confidence interval.** The formula assumes
the residual is independent zero-mean noise of common variance; ours is
dominated by model error, which is systematic and correlated between probes.
Read it as *how tightly these probes pin the position given how badly the model
fits*, and not as a probability that the truth lies inside it. It also cannot
see anything both methods share — calibration, the filament ansatz — so it is a
lower bound on total error.

Measured on 1641, all 12 probes, `fit_ip=False`:

| quantity | median | p95 |
|---|---|---|
| Biot–Savart σ_ΔR | 16.4 mm | 34.3 mm |
| Φ σ_ΔR | 16.4 mm | 35.2 mm |
| **\|ΔR_BS − ΔR_Φ\|** | **1.6 mm** | — |

The two methods differ by about a tenth of what the residual says either one is
pinned to. So on this shot the linear proxy is not the limiting factor: it
contributes roughly 1.6 mm where the probes themselves constrain the position to
around 16 mm.

## Files

| file | role |
|---|---|
| `adapter.py` | the only module that reads the rest of the repo; every upstream dependency is resolved here |
| `field.py` | Biot–Savart field, vectorised over displacement points |
| `invert.py` | the solver: chamber-lattice templates, exhaustive grid search, weighted NLSQ refinement, profiled amplitude, minima and wall detection |
| `cli.py` | flag-driven runner |
| `selftest.py` | convention check against `cal_signal`, synthetic recovery, fold probe |
| `tests.py` | 17-check battery, no repo or shot data needed |

`biot_savart_main.py` at the repository root is the parameters-in-the-file entry
point, matching `main.py`'s convention.

### Why two implementations of the same equations

`field.py` duplicates what `cal_signal` computes. `cal_signal` takes scalars, so
a solver that calls it pays for one probe-vector evaluation per residual;
`field.py` broadcasts over displacement points, letting the whole
finite-difference stencil go in one call. It is also a genuine cross-check:
selftest stage 2 compares the two over a grid and reports the sign convention
that matches. They currently agree to **2e-15**, with `DZ_SIGN = -1` and
`TANGENTIAL_SIGN = +1`.

Set `forward_model = "cal_signal"` to route through the repository's own
implementation instead; results are identical to ~1e-11 and it runs ~2.4×
slower per call.

## Verification status

`tests.py` — 17/17. T1 checks the on-axis field against the elementary closed
form (agreement 1.8e-14) and the far field against a dipole; T2 checks ∇·B and
∇×B off the wire; T3 checks mirror symmetry. These three are the only ones that
can catch an error in the field formula itself, since the rest generate and
invert with the same code.

### Two quantities called "noise"

**Fit residual** — the mismatch between the measured probe signals and what the
model predicts at the fitted position. This is `resid_norm`. It exists on every
real shot and contains everything the model does not capture: instrument noise,
calibration error, eddy currents, and the failure of a single filament to
describe a real current profile. It is not a noise level and cannot be converted
into one.

**Injected noise** — a gaussian perturbation added deliberately to *synthetic*
signals in `tests.py` and `selftest.py` stage 1, as a fraction of the median
|B|. It never touches real data.

Quoting an injected-noise figure as though it described a real shot, and reading
`resid_norm` as a noise level, are both mistakes.

### Injected-noise budget — synthetic signals only

Every row is measured on signals generated by `field.py` at a known (ΔR, ΔZ) and
then perturbed by a gaussian of the stated size. No shot data is involved.
Because the generator and the inverter share the same forward model, this
measures the **solver's sensitivity** to a given perturbation and nothing else.
It is not an accuracy figure for a real discharge, where calibration error, eddy
currents and the single-filament approximation all contribute and none of them
resembles gaussian noise.

12 probes, cold start, true positions out to 0.14 m:

| injected noise | median err | p95 | max |
|---|---|---|---|
| 0.1% | 0.07 mm | 0.18 | 0.20 |
| 1% | 0.78 | 2.02 | 2.18 |
| 3% | 3.09 | 6.70 | 9.87 |
| 10% | 7.48 | 25.56 | 35.15 |

Roughly 0.8 mm of displacement error per 1% of injected noise, at the median.

## Results so far

| shot | solved | at bound | ΔR median | resid median | \|ΔR_BS − ΔR_Φ\| median |
|---|---|---|---|---|---|
| 1641 | 5865/5907 | 42 | +10.5 mm | 0.071 | 18.9 mm (p95 29.4) |
| 1643 | 3575/3606 | 31 | −7.8 mm | 0.045 | 31.7 mm (p95 48.6) |
| 2400 | 728/745 | 17 | −164.2 mm | 0.178 | 100.6 mm (p95 166) |

**The Φ-gap figures are not purely linearisation error.** The adaptive Φ path
switches probe sets per sample while these runs used all 12, so probe-set choice
is mixed in. Isolating the linearisation needs a comparison against a *fixed*
12-probe Φ run.

**Shot 2400 is not believable.** ΔR at −164 mm median lies outside the 250 mm
minor radius, with four times the residual of 1643. Either the shot breaks the
filament ansatz or something upstream is wrong for it. Not diagnosed.

## The ~15% amplitude excess — investigated, not resolved

With `fit_ip=True` the fitted amplitude should equal `Ip/I_ref` exactly. It comes
out ~1.15× that on all three shots examined. What the investigation established:

- **Two components.** `ratio = c + b/Ip` fits with R² = 0.79–0.97: a
  multiplicative `c ≈ 1.12` and an additive `b` worth 1.6–2.5 kA of equivalent
  plasma current. The `1/Ip` shape means `b` is a field that does not scale with
  the plasma current.
- **Uniform across probes.** Flat-top medians over well-behaved probes: 1.159 /
  1.145 / 1.142, spread 0.06–0.09. A global scale, not spatial structure.
- **Not the geometry.** `R0` has zero effect. A probe radius of 0.280 m would
  zero the ratio, but it makes the *shape* fit monotonically worse (1641
  residual 0.0770 → 0.0823), so 0.321 m is right and the excess carries no
  geometric content.
- **Not the filament ansatz.** Ampère's law round the probe circle — no model at
  all — gives |I_probes/I_Rogowski| = 1.04–1.14 on well-behaved probes, the same
  magnitude.
- **Not eddy currents, for the multiplicative part.** Eddy currents must reverse
  sign between ramp-up and ramp-down. At matched plasma current the two phases
  differ by −0.002 (1641) and −0.001 (2400); corr(ratio, dIp/dt) ≈ 0.09; and the
  excess is fully present at flat-top. 1643 is the exception at −0.038. The
  additive term `b` remains a plausible eddy signature, and the probes at
  0.321 m sit *outside* the 0.25 m vessel, so the Ampère contour does enclose
  vessel currents.

**Remaining candidates, degenerate under every magnetic test:** the probe
calibration under-reads by ~13%, or the Rogowski over-reads by ~13%. Separating
them needs a known current — an in-vessel calibration loop, or a vacuum shot with
a known primary current.

Note this is invisible to the pre-plasma calibration check, because `k_t, k_oh,
k_v` are *fitted* on that window and absorb a probe-gain error.

**A loose end worth pulling.** `data/<shot>/IP2.txt` exists and nothing in the
repository reads it — `_load_shot` takes IP1 unconditionally, while
`current_channels.resolve_all` does redundant-channel health checking for IT,
IOH and IV. The two Rogowskis agree to 1.7% on 1641 and 1643 but differ by
**9.4%** on 2400, where the probe-fitted current agrees with IP2 to 3% and with
IP1 to 13%.

**Practical consequence.** Use `fit_ip=True` on 12-probe runs: the amplitude is
profiled out, so a uniform scale error cancels in the inter-probe ratios and
never reaches the position. With `fit_ip=False` it lands in ΔR — +10.2 mm versus
+31.3 mm on 1641, all 12 probes. That 21 mm is an order of magnitude larger than
the 1.6 mm linear-proxy error.

## Open

- The ~15% amplitude excess above.
- Whether the Φ fold past ~0.16 m survives without the linearisation. Selftest
  stage 3 finds no degeneracy on real geometry for any probe set tested, but a
  97-point noise-free grid cannot see a flat valley, so this is **not** evidence
  that the fold is a linearisation artefact. A Jacobian condition-number map
  would settle it.
- The cause of the probes 11/12 sign discrepancy, as above. Curation suppresses
  it, so it is not blocking, but it is unexplained.
