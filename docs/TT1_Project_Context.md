# TT-1 Displacement Workstream — Project Context

*Replaces the previous `TT1_Project_Context.md`. Written for a reader new to the
project. Records decisions **with** their justification; where this document
diverges from `Proposal_V8.md`, the divergence and its reason are stated.*

Last updated: 2026-08-16 (rev 8).

---

## 1. What this workstream does

Reconstruct the TT-1 plasma column's horizontal and vertical displacement
`(dR, dZ)` from magnetic probe signals, as an input to position feedback control
and as a data source for the wider ML discharge-dynamics effort.

Four methods are maintained and compared on the same axes:

| method | source | outputs | role |
|---|---|---|---|
| **filament** | toroidal-filament model via the Φ map, this repo | `dR`, `dZ`, metres | primary |
| **biot_savart** | same physics, inverted by nonlinear least squares instead of through Φ | `dR`, `dZ`, metres | reference implementation (§3.19) |
| **position_c** | port of the real-time controller's 2-probe `position.c` | `dR` only | baseline |
| **ai_camera** | CCD-image AI centre detection (`<shot>_pred.txt`) | `dR`, `dZ` | independent cross-check |

The AI camera is **never** used to select probe sets or tune the filament
method. It is the only measurement independent of the magnetics, and it keeps
that value only if nothing in the magnetic pipeline is fitted to it.

**Biot–Savart is not a fourth independent measurement**, and must not be
presented as one. It shares the probe calibration, the vacuum-field subtraction
and the single-filament ansatz with the filament method, and differs only in how
the inverse problem is solved. Its purpose is to measure what the Φ
approximation costs, and that is what it has now done (§3.19).

---

## 2. Repository layout (changed files marked ✎, new ★)

```
TT1PlasmaColumnPosition-2D/
├── main.py                        entry point, fixed probe set
├── biot_savart_main.py          ★ Biot–Savart entry point, parameters in-file
├── ip_compare.py                ★ four-way plasma-current comparison (§3.21)
├── compare_methods.py           ✎ four-method overlay, uncertainty bands
├── adaptive_select.py           ✎ probe-set selection; fit_ip now a parameter
├── position_c_displacement.py     position.c port
├── stamp_model_caches.py        ★ one-off cache migration tool
├── docs/                        ★ method document (.md and .docx)
├── methods_script/biot_savart/  ★ adapter, field, invert, cli, selftest, tests
└── methods_script/toroidal_filament/
    ├── cache_keys.py            ★ model fingerprints
    ├── mprobe.py                ✎ M-probe estimator + Phi maps
    ├── phi_map.py               ✎ legacy 4-probe map
    ├── weights_cache.py         ✎ per-shot probe weights
    ├── coefficient.py           ✎ 1D Taylor coefficients (legacy)
    ├── plasma_shift.py          ✎ 1D coefficient consumer (legacy)
    ├── curation.py                probe weights from pre-plasma noise
    ├── signal_strength.py         `cal_signal` forward model
    └── parameters.py              geometry, `shift_domain`, calibration coeffs
```

Cache directories (all regenerable, none in version control):
`phi_tables/` (~174 MB), `hull_tables/` (~200 kB), `rt_fields/` (~4 MB),
`weights_store/`. (`ranking_store/` went with the ranking path — §3.15.)

---

## 3. Decisions recorded this cycle

### 3.1 AI camera Ip gate — fixed

**Symptom.** The AI trace extended far beyond the plasma on every shot
(1643: 285.5–452.0 ms against a plasma window of 334.6–405.3 ms).

> **Rev 6 note.** That 285.5–452.0 ms span is now known to be shot **1641's**
> prediction file, which had been copied into every shot folder (§3.17). The
> code defect fixed here — missing pandas import, bare `except` — was real and
> the fix stands. But the symptom that revealed it was partly the wrong file,
> and the reported result below gated 1641's frames inside a 1643 comparison.
> 1643's own file spans 332–406 ms, so the corrected numbers land close by
> coincidence: both are long shots.

**Cause.** `compare_methods.py` never imported pandas, so the gating block
raised `NameError` on its first line. A bare `except Exception` swallowed it.
Two earlier fixes changed the Ip *source* and never touched the defect.

**Fix.** Import pandas; read `IP1.txt` directly (the gate needs a time window,
not a curated Ip); narrow the handler to `(OSError, ValueError, KeyError,
ImportError)`.

**Rule adopted.** *Data* problems may be absorbed by a fallback; *code* defects
must crash. A bare `except Exception` around a computation is how a defect
survives review — it converts a crash into a silent wrong answer.

Result: 335.0–405.0 ms, 134 frames.

### 3.2 Cache provenance — every artefact self-invalidates

**Symptom found by test.** Changing `shift_domain` from 0.14 m to 0.10 m
produced an **identical** Phi cache key and silently reloaded the 0.14 m map:

```
shift_domain = 0.14 -> hash f19e61db70
shift_domain = 0.10 -> hash f19e61db70      <-- same file, wrong domain
```

`shift_domain` is the parameter this project's own record flags as revisited.

**Cause.** `mprobe._config_hash()` covered *estimator* configuration
(probes/weights/`fit_ip`/gains/grid) but not *forward-model* configuration
(`shift_domain`, `R0`, `R`, `mu`, `I`, angles).

**Design: two mechanisms, chosen deliberately.**

- *Estimator* config → **filename**. It varies routinely; you legitimately hold
  maps for several probe sets side by side.
- *Forward-model* config → **stamp inside the artefact, verified on load**. It is
  global; nobody wants two domains at once. A mismatch should **halt with an
  explanation**, not silently trigger a ~170 MB rebuild that looks like a hang.

Rejected alternative: putting the domain in the filename. Correct, but it
orphans every existing map and forces a multi-minute silent rebuild.

**Failure behaviour differs per artefact, on purpose:**

| artefact | on mismatch | why |
|---|---|---|
| Phi maps | `RuntimeError` | expensive; a wrong map silently corrupts displacements |
| rankings | refuse, return `None` | realtime inherits these blind |
| weights | warn, return `None` | cheap to recompute; `"auto"` just reruns curation |
| coefficient `.pkl` | warn only | imported by paths that never read it; raising would break unrelated runs |

Unstamped artefacts are refused as firmly as mismatched ones — "cannot be
verified" and "known stale" deserve the same answer.

**Two keys, not one.** `forward_model_key()` covers what `cal_signal` computes
and over what region. `curation_key()` covers `calibration_coeff` and the
curation gates. Verified: changing `k1t` moves `curation_key` and leaves
`forward_model_key` **unchanged** — Phi maps genuinely do not depend on the coil
calibration, which corrects the *measured* field on the far side of the
estimator. This is the coupling that will fire when the outstanding `k_if`
feedback-coil term is added: cached weights become stale, Phi maps do not.

**Migration.** `python stamp_model_caches.py --apply` stamps existing caches
without rebuilding. Stamping **asserts**, it does not verify — there is no way to
recover `shift_domain` from a finished map. Dry run is the default.

### 3.3 Duplicate Phase-1 pass — removed  *(historical: this code is gone)*

> Superseded by §3.15. `commit_ranking()`, `rank_sets()` and
> `adaptive_displacement()` were removed from the repository in the
> publication clean-up, and with them this fix and the `--commit` CLI.
> Retained because the failure mode it records — a cache-shaped return value
> discarding work its caller then redoes — is not specific to that path.

`commit_ranking()` returned only a JSON-serializable record, discarding the
`info` and `(t, ip, B)` that `rank_sets()` had already produced. So
`adaptive_displacement()` called `rank_sets()` a second time.

Fixed with `return_phase1=True` threaded through `resolve_ranking`. The
`--commit` CLI keeps the plain record return.

Also: when a `priority` is supplied the ranking's order is discarded, so
`tiebreak="margin"` is now passed and `_roundtrip_error` never runs there.

Results bit-identical; 1643: offline 31.7 s → 10.5 s, realtime 10.0 s → 6.6 s.

### 3.4 Hull caching (and why `HULL_N = 25` needs re-validating)

Hulls are **shot-independent** — only `_proxy()` depends on `ip`/`B` — and were
recomputed on every call (23 ms/set, 378 ms per `rank_sets`).

Two-tier cache keyed on `(P, S0, angles, n, forward_model_key())`. Keying on
numerical content rather than on probe list + weights source makes it exact and
robust to weights being derived differently.

| | |
|---|---|
| uncached | 13.4 ms (was 23.0 ms before the disc fix, §3.11) |
| memory hit | 0.063 ms |
| disk hit | 0.33 ms |

Disk caching is **not** automatically worth it — reading can cost more than
recomputing when the computation is trivial. It is worth it here by 74×.

**`n = 25` was validated on the SQUARE grid — the sweep needs redoing.** Swept
15/25/51/101/201/401 on 1643: health converged by 51, `n=25` within 0.003 of the
limit for every set; order, survivors (7), coverage (1.0000), switches (6)
identical across the range. **That sweep predates the disc fix (§3.11)**, which
cut the face count from 26 to 8 for `[1 4 7 10]`. The conclusion probably holds —
fewer, better-placed samples, and convergence was comfortable — but it is no
longer measured. **Re-run before quoting.** Raising it is not free even when cached — face count grows ~linearly
(16→382) and enters the **per-timestep** containment test forever, while build
cost is quadratic (27 ms → 6.4 s per set). Caveat: swept on one shot.

`n` is **not** derived from `PHYS_STEP`. The earlier `PHYS_STEP` convergence work
validated the Phi grids and says nothing about the hull grid.

### 3.5 Map folding — reconciled with the earlier test

A fold (Jacobian determinant sign change in `(dR,dZ) → (u,v)`) makes the map
non-injective: two positions share a proxy, so inversion is ambiguous.

An earlier session tested this on the **12-probe** set and located the fold at
~0.16–0.18 m, outside `shift_domain = 0.14`. That result is **confirmed**, not
overturned:

| set | fold frac r<0.10 | r<0.14 | r<0.16 | r<0.20 |
|---|---|---|---|---|
| all 12 | 0.000 | **0.000** | 0.001 | 0.037 |
| `2 3 8 9` | 0.000 | **0.000** | 0.001 | 0.049 |
| `1 4 7 10` | 0.000 | **0.000** | 0.000 | 0.077 |
| `1 2 7 8` | 0.000 | 0.059 | 0.089 | 0.138 |
| `3 4 9 10` | 0.016 | 0.103 | 0.138 | 0.189 |

**Correction to a wrong intermediate result.** An earlier measurement in this
session sampled the **square** `[-0.14, 0.14]²`, whose corners reach r = 0.198 m —
past the fold radius — and reported in-domain folds that do not exist. Restricted
to the disc they are 0.000. *Sample the operating region, not its bounding box.*

**New, and not contradicted by prior work:** fold behaviour is **set-dependent**,
and two 4-probe sets fold inside the domain where all-12 does not. Only all-12
had ever been tested.

**This does not affect the 2400/2766 diagnostic.** That concerns proxies leaving
the hull (*hull exit*); folding is the image covering itself. Different
mechanisms, and all-12 has zero in-domain fold. The presentation's claims stand.

### 3.6 Selection criteria — what does and does not predict accuracy

| criterion | verdict | evidence |
|---|---|---|
| hull containment | necessary, not sufficient | sets that both contain a sample disagree by 34–48 mm at p95 |
| hull **depth** (margin) | **rejected** | corr with per-sample error −0.09 to −0.13; earlier Spearman −0.01 vs AI |
| local `\|detJ\|` | **rejected** | corr −0.08 to −0.26; `2 4 8 10` at 0.003 is good, `12 2 6 8` at 0.0024 is catastrophic |
| fold fraction | **rejected as sole criterion** | 7/16 sets are exactly 0.000 yet span 0.001–0.604 mm round-trip; `12 3 6 9` at fold 0.002 has 645 mm round-trip |
| conditioning | rejected | `11 12 5 6`: cond 1084, round-trip 0.172 mm |
| **round-trip** | **adopted** | directly measures inversion fidelity; Spearman **−0.797** vs AI on 1641, reproduced this cycle (§3.10) |

Why the geometric proxies all fail: for an admitted set, Φ is exact to ~1e−10
almost everywhere inside the hull. There is no graded "bad region" to detect —
a flat floor and a thin tail — and the dominant in-domain error is
**resampling/interpolation degradation, not folding**, which is exactly what the
geometric measures are blind to. This is also why the earlier det-floor and hull
guards inside `mprobe` were built and dropped.

### 3.7 Round-trip is shot-dependent — hence the field

The per-set scalar averages a spatial field over wherever the plasma went:

| shot | `12 3 6 9` health | shot-based rt |
|---|---|---|
| 1643 | 0.653 | 645 mm |
| 2766 | 0.182 | 0.98 mm |
| 3970 | 0.815 | 52 mm |

Cross-shot Spearman for the scalar: +0.41 to +0.69. Realtime inherited an
admission decision built from it.

**The field `rt(u,v)` is shot-independent** — a property of (Φ, estimator,
forward model); the shot only decides which `(u,v)` are visited. Averaging is
what introduced the shot dependence. Synthetic cross-shot Spearman: +0.53 to
**+0.89**; within-shot agreement with the scalar +0.36 to +0.77.

### 3.8 `adaptive_selection()` — the selection path

Added alongside `adaptive_displacement()`. **Superseded that path entirely in
the August 2026 publication clean-up: the hull-health scheme is now removed
from the repository and `main.py` calls this function.** The experimental-era
name `rtfield_displacement` is gone with it — the only public name is
`adaptive_selection()`. See §3.15.

**Phase A (once per configuration, no shot):** cached hulls, cached `rt(u,v)`
fields (129×129, ~0.4 s build / 0.3 ms read), static order
`sorted(−good_frac, med)` with `good_frac` scored **inside the hull** (§3.10). No health, no per-set round-trip, no admission gate,
no validation, no `committed_latest.json`. Offline and realtime are now
**identical** here — the reason the old table had two different columns is gone.

**Phase B (per sample, causal):**

```
accept(s, i) = inside_hull[s][i] and rt_at_uv[s][i] <= RT_GOOD
pick = current set if accept(current, i) else first s in order with accept(s, i)
```

Accuracy is asked **locally, per sample** rather than inherited as a shot-averaged
scalar. The ranking is demoted to tiebreak and consultation order — it decides
*which* acceptable set is used, never *whether* a sample is accepted. Hysteresis
retains the current set while it stays acceptable; most samples test one set.

| shot | scheme | coverage | n_switch | top set |
|---|---|---|---|---|
| 1643 | rtfield | 1.0000 | **3** | `11 12 5 6` |
| | ranked | 1.0000 | 6 | `2 3 8 9` |
| 2766 | rtfield | **0.9721** | **2** | `11 12 5 6` |
| | ranked | 0.9392 | 9 | `1 4 7 10` |
| 3970 | rtfield | **0.9796** | **2** | `2 3 8 9` |
| | ranked | 0.9769 | 6 | all 12 |

In this table "rtfield" is the scheme now called simply `adaptive_selection()`,
and "ranked" is the hull-health path retired in §3.15. The row labels are kept as
measured so the comparison stays readable as evidence; neither is a function name
in the repository any more.

Switching matters because sets disagree by 6–12 mm at the median, so each
avoided switch is an avoided step in `dR`.

### 3.9 No runtime coverage gate

Removed deliberately, not overlooked. The ranked path raised if realtime coverage
fell below 0.60. That guarded **cross-shot order transfer**; this path inherits no
order, so the thing it guarded does not exist. It was never actionable live
either — coverage is only knowable after the last sample.

Nothing is lost: an unaccepted sample stays `NaN` in `dR`/`dZ`, so the failure
appears in the **output**, not in a summary statistic. `coverage` and `n_switch`
remain as offline diagnostics (`compare_methods.py` reads both for its label) and
are never acted upon.

A rolling consecutive-rejection counter was considered as a live health signal
and **rejected** on the user's judgement.

### 3.10 AI cross-check validation, and the `good-frac` hull fix

> **Rev 6, checked and CONFIRMED.** §3.17 records that every shot folder carried
> a copy of **1641's** prediction file. This section is the one place that is
> *unaffected*, because it was deliberately and correctly done on 1641 itself —
> the file was genuine for this shot and this shot only. Spearman −0.797, the
> hull-fix table below, and the criterion comparison all stand. This matters:
> `good_frac` underpins the entire ordering, and its validation was never
> measured against the wrong discharge. The line below that "2766 and 3970 have
> no AI output at all" was correct when written — their folders held 1641's file,
> so a careful check found no genuine output for them.

**Only 1641 can run this test.** It needs a `<shot>_pred.txt` *and* spread in
per-set AI agreement. 1643 has the file but no spread — all 16 sets score between
−0.09 and +0.21, so there is nothing to correlate against and any Spearman
computed there is meaningless. 2766 and 3970 have no AI output at all. 1641 spans
−0.227 to +0.928, which is why the original analysis chose it.

**Correction to an earlier mis-citation.** The figure "Spearman +0.79 vs AI" was
being quoted in this project as a general property of round-trip. It is specific
to 1641. Reproduced independently this cycle: **−0.797** (negative because lower
round-trip is better). The earlier analysis is confirmed and the current pipeline
agrees with it — but it is a one-shot result and must be stated as such.

**The `good-frac` box artefact, found and fixed.** `rt_field_score` originally
scored over the whole rectangular `ug/vg` box. The reachable region inside that
box is curved and fills it by a different fraction for each set, so the score
mixed "how faithful is this set" with "how rectangular is its region". Nodes
outside the hull are extrapolation no sample can ever query, since Phase B tests
containment first.

Fix: mask the score to the hull interior, using the already-cached hull faces.

| | box-scored | hull-masked |
|---|---|---|
| Spearman vs internal round-trip (1641) | −0.715 | **−0.741** |
| Spearman vs AI cross-check (1641) | +0.413 | **+0.727** |

Measured again on 1643 after the disc fix (§3.11): Spearman(good-frac, round-trip)
= **−0.888**, the strongest agreement recorded for any criterion. The 1641 AI
figures above predate the disc fix and have not been re-run.

The artefact was concrete: `11 3 5 9` scored 0.416 (bottom third) despite a
round-trip of **0.03 mm**, the second-best on the shot by the internal measure.
Hull-masked it scores 0.792. `2 3 8 9` → 0.988 and all-12 → 0.993, matching their
~0.00 mm round-trips. Scores compress upward overall (0.6–0.99 rather than
0.27–0.72) because out-of-hull nodes were depressing every set.

`good-frac` now roughly matches round-trip's own AI agreement, so the static
ordering is validated to the same standard as the criterion it replaces — on the
same single shot, with the same caveat.

**The ordering is load-bearing, not a tiebreak.** Measured on 1641: a median of
**9 sets qualify per sample** (mean 8.6, min 2, max 10), and more than one
qualifies on **100%** of samples. So the order decides which set's inversion is
used essentially everywhere, and sets disagree by 6–12 mm at the median. Calling
it a tiebreak understates it — acceptance is local and per-sample, but among
accepted sets the ordering is what picks the answer.

**Methodological note.** The AI camera is a cross-check, never ground truth. It is
legitimate to ask whether a criterion's ranking *correlates* with AI agreement; it
is not legitimate to call a set "good" because the AI likes it. Internal measures
(round-trip) judge sets; the AI only tests whether those judgements track an
independent instrument.

### 3.11 The hull was sampled over the square, not the disc — fixed

`_hull_faces` swept `[-shift_domain, shift_domain]²` on a full square grid.
`shift_domain` is a RADIUS, so the corners reach `0.14·√2 = 0.198 m` — past the
~0.16–0.18 m fold radius, inside the outer shell this project already treats as a
resampling artefact rather than trustworthy forward model.

Consequence: every hull was inflated by samples from positions the plasma cannot
occupy, and the guard admitted proxy points with no physically reachable preimage.
Found while drawing the coverage figure — samples the guard called "inside" plotted
outside the disc-based hull.

Fix: restrict the sweep to `hypot(dR,dZ) ≤ shift_domain`, and skip positions where
`cal_signal` reports a non-physical geometry.

| quantity (shot 1643) | square | disc |
|---|---|---|
| faces, `[1 4 7 10]` | 26 | **8** |
| outside-hull fraction, `[12 2 6 8]` | 0.36 | **0.64** |
| outside-hull fraction, `[2 3 8 9]` | 0.000 | 0.006 |
| hull build cost | 23.0 ms | **13.4 ms** |

`adaptive_selection` on 1643 after the fix: coverage 0.9981, 3 switches, top set
`[11 12 5 6]`. Coverage is no longer exactly 1.0 — correctly, since the earlier
1.0000 partly came from accepting unreachable points.

**The cache key does not cover this.** `n` is in the key but the square-vs-disc
choice was never a parameter, so old `hull_tables/` entries are silently stale.
Delete the directory when installing this change. (This is the same class of defect
as §3.2 and a reminder that the rule there — *a cache key must cover every input* —
extends to inputs that are hard-coded rather than named.)

### 3.12 What the hull is still for, and what it is not

Terminology, because these were being conflated:

* **Hull** — the convex region a set can reach in its own proxy space.
* **Health** — the fraction of a shot a given set cannot reach. A number *derived
  from* the hull, not the hull itself.

**Health is gone.** `adaptive_selection` never computes it. Its two old jobs are
both done better: ranking → `good-frac`; admission (`HULL_MAX`, excluding whole
sets in advance) → deleted, replaced by per-sample acceptance, so no set is
pre-judged.

**The hull is kept, for two jobs:** the per-sample containment test, and the mask
for `good-frac` (§3.10).

**Hull volume rejected as a ranking.** Spearman(hull area, round-trip) = **+0.521**
on 1643 — the wrong sign. A folded map lays the domain over itself and inflates its
own image, so area rewards exactly the pathology it should penalise: `[12 1 6 7]`
has the largest area (786,212 mm²) and 9720 mm round-trip; `[1 4 7 10]` is among
the smallest (59,251 mm²) at 0.02 mm.

**How much the containment test actually earns.** Dropping it changes the chosen set
on **0.55%** of samples (20/3606, shot 1643), and always by promoting a
higher-ranked set on the strength of extrapolation. Coverage rises 0.9981 → 1.0000
without it — fake coverage, bought by accepting points with no preimage. So it is a
cheap safety margin (one matmul against ~8–16 faces), not a primary mechanism: the
rt field does nearly all the work. An earlier note in this document cited 1.31%,
which was pooled over (set, sample) pairs and overstated the operational effect.

---

### 3.13 `position.c` — why its calibration is stale, and its sign conventions

**The coefficients are outdated, and there is now a reason rather than an
assertion.** TT-1 staff confirmed that the position calibration coefficients were
computed while the tokamak was still in China, and that several coils — the Ohmic
coil in particular — were repositioned after the move. The pickup constants in
`position.c` therefore describe a machine geometry that no longer exists. This
explains the shape disagreement recorded in §3.x (correlation with the filament
0.28 under `position.c`'s own constants, 0.98 under the filament's), without
needing to invoke any defect in the 2-probe formula itself.

**Consequence for scope.** There is no forward use for the `position.c` pickup
calibration. Keeping it as a selectable `PICKUP_SOURCE` is still useful for
*separating* error sources — it isolates calibration error from formula error —
but it is not a candidate method. Presentations should frame it as *why we
stopped using it*, not as a live alternative.

**Sign conventions are measured, not derived.** `POSC_FLIP_SIGN` only negates; it
never changes `std`, so it cannot repair a scale problem. The flag was chosen per
shot by matching the filament's sign:

| shot | `positionc` pickup | `filament` pickup |
|---|---|---|
| 1643 | False | True |
| 2766 | False | True |
| 4404 | **True** | **False** |

Two things follow. The two pickup sources carry **opposite polarity
conventions**, and shot 4404 inverts *both* relative to 1643 and 2766. And
`POSC_FLIP_SIGN` is currently a single flag applied to both sources, which is
wrong — it cannot express the table above. An earlier attempt to plot 4404
produced an off-scale curve for exactly this reason. **Fixing the flag into two
independent flags is outstanding.**

**Unresolved: a ~165 mm offset on 1643.** With the filament pickup, the
`position.c` trace sits about 165 mm from the reference while carrying the *best*
`std` of any curve on that shot (22.2 mm against 34.5 mm). A pure sign error
cannot produce that pattern; a missing centre subtraction can. **This is not
diagnosed** — it is recorded here so it is not mistaken for settled.

---

### 3.14 Shot 4404 — the two failures separated and measured

4404 is the shot that shows *both* `position.c` failure modes at once, which is
why it, not 1643, carries the comparison slide. Measured on the full plasma
window (333.8–340.9 ms, 358 samples):

| pickup source | `FLIP_SIGN` | range of ΔR (mm) | std (mm) |
|---|---|---|---|
| `positionc` | False | −125,948 … +37,908 | 9,518 |
| `positionc` | **True** | −37,908 … +125,948 | 9,518 |
| `filament` | **False** | −340 … −70 | 44 |
| `filament` | True | +70 … +340 | 44 |

**The two failures are independent, and the table separates them.**

- *Calibration.* With `position.c`'s own constants the trace spans ±10⁵ mm —
  three orders of magnitude outside a machine whose minor radius is 250 mm. This
  is the antipodal denominator `f₂+f₃` crossing zero, which is
  calibration-dependent (§ on the 4404 blow-up): the filament constants keep
  `f₂+f₃` around 1e-2 T, `position.c`'s own drive it to ~8e-5 T. The cause is
  documented in §3.13 — coefficients computed before the machine moved from
  China, coils repositioned since.
- *Formula.* Feed the same 2-probe formula a correct pickup and it stops
  diverging, but it still sits at −70 to −340 mm and drifts past the vessel wall,
  while the filament method and the AI camera track each other near zero. So the
  formula is wrong even when the calibration is right. This is the half of the
  argument 1643 could not make.

**`FLIP_SIGN` cannot be implicated in either.** It only negates: the `positionc`
std is 9,518 mm with the flag either way. No sign convention rescues that trace,
which is worth stating because "the sign must be wrong" is the natural first
guess at a curve that large.

**Filament result on 4404** (`adaptive_selection`): coverage **92.5%**, 6 set
switches, top-ranked set the full 12-probe array. The 7.5% of samples with no
valid probe set are NaN, as designed — the shot is not refused.

**Weights source: `"auto"`, not the `"last"` default.** `"last"` raises
`RuntimeError: weights='last' but nothing stored` in a clean checkout, and no
shot consecutive with 4404 is available to seed the store; the only candidates
are 1643 and 2766. Seeding from an arbitrary non-adjacent donor would put an
unjustifiable choice inside a figure, so 4404's own pre-shot window was used.
This is defensible because the slide is an **offline method comparison**, not a
real-time demonstration. **Divergence from the repo default is deliberate and
should be restated if the figure is ever regenerated.**

**Cache cost, for planning.** Building all 16 candidate Φ maps from empty took
about 6 minutes and ~170 MB; a handful of sets dominate (25–65 s each) while most
build in under a second. Once cached, `adaptive_selection` on 4404 runs in 15 s.
Because the weights enter Φ, changing `weights_source` invalidates the maps and
pays the build again.

---

### 3.15 Publication clean-up — hull health retired, one selection path

Preparing the repository for public release forced a decision that had been
deferred: `adaptive_select.py` contained **two** selectors reached through the
same `use_probes = "adaptive"` setting. `main.py` called `adaptive_displacement()`
(hull-health ranking); `compare_methods.py` called `adaptive_selection()`. The
same configuration word produced different traces depending on which script ran,
and the README documented only the older one.

**Resolved by retiring hull health, not by documenting the split.** `main.py` now
calls `adaptive_selection()`. Removed from the repository: `adaptive_displacement`,
`rank_sets`, `health_and_margin`, `_admit`, `commit_ranking`, `resolve_ranking`,
`_roundtrip_error`, the constants `HULL_MAX` and `_REALTIME_MIN_COVERAGE`, the
`"offline"`/`"realtime"` mode switch, and `ranking_cache.py` with its
`ranking_store/` cache — 338 lines.

**The hull itself is untouched.** It is still built, still cached, still tested
per sample. What went is hull *health*: the per-shot aggregate "what fraction of
this discharge leaves this set's hull", used as a ranking key. §3.14's naming
lesson applies — the two are easy to conflate and only one was retired.

**Justification** is the §3.8/§3.10 argument, now acted on rather than recorded:
the per-set scalar the old path ranked on is an average of the rt field over
wherever the plasma happened to go, so it is shot-dependent (cross-shot Spearman
+0.41 to +0.69; `[12 3 6 9]` scores 645 mm on 1643 and 0.98 mm on 2766 with
identical geometry). Its admission gate was also aggregate and binary: on 4404
the best set sat outside its hull for 20.9% against a 20.0% ceiling, so the whole
shot was refused, discarding the ~79% of samples that inverted perfectly.

**Verified numerically unchanged.** `adaptive_selection(4404, "auto")` before and
after the removal: coverage 0.9246, 6 switches, same top-ranked set. This was
dead-code excision, not a behaviour change.

**Correction (rev 5): `prebuild_shot()` did prebuild.** Rev 4 recorded that it
did not. That was wrong — `MProbeEstimator.__init__` called
`_load_or_build_phi()`, so the build happened as a side effect of constructing
the estimator. The real defect was narrower: `cached` was evaluated *after*
`_estimator()` had already created the file, so every set reported as cached.
The rev 4 code change (build all three Phase A artefacts explicitly, record
per-set times) stands and is now correct rather than accidental — see §3.16,
which makes Φ lazy and so makes the explicit build necessary.

**Other repository fixes**, none affecting physics or numbers: `requirement.txt`
was a `pip freeze` of a developer desktop that omitted every package the code
imports, replaced by a real `requirements.txt`; `lsq-ellipse` was never a
dependency (`LsqEllipse` was imported and never called), so it is gone along with
its install instruction; `main.py` no longer imports OFIT at module load;
`compare_methods.py`'s default shot list contained `4405` (not a shot in this
project) and `2766` twice; `.idea/` was committed and leaked a developer's
Windows username. Full list in `CHANGES.md` Part 4.

**Still open for publication:** the repository has **no `LICENSE`**, and it is a
fork of P-Attapon/TT1PlasmaColumnPosition, so the upstream terms constrain the
choice. The 7.8 MB thesis PDF in the repository root needs its redistribution
confirmed.

### 3.19 The Biot–Savart method, and what the Φ proxy actually costs

`methods_script/biot_savart/` fits `(dR, dZ)` directly to the probe signals by
nonlinear least squares against the exact circular-filament field. It removes
three approximations and **only** three: the cylindrical `R₀ → ∞` linearisation
to `(dU, dV)`, the third-order polynomial that maps the proxy back, and the Φ
tabulation with its `shift_domain` limit. It shares everything else with the
filament path.

A fourth item that was initially listed as removed — Eq. 6's time-stepping — was
**wrong and has been retracted**. Eq. 6 propagates error because its
coefficients are indexed by `t(n-1)`'s answer; the 2D Φ map replaced that with a
joint inversion, so `mprobe.shift()` is already stateless. Only the legacy 1D
path (`coefficient.py` / `plasma_shift.py`) still carries that structure.

**Search.** The forward model at a lattice point does not depend on the sample,
so it is evaluated once per shot on a 1 mm lattice covering the chamber
(`CHAMBER_RADIUS = 0.20 m`, the limiter radius), and the residual everywhere for
a given sample is one matrix–vector product. Every lattice-local minimum is then
refined by a continuous solve, so the result is the **global** minimum inside the
chamber, subject to the lattice resolving every basin. ~4 ms/sample. There is no
warm start; samples are independent of each other by construction.

**The headline number.** With probe set and `fit_ip` matched to the filament
path, on shot 1641, all 12 probes:

| quantity | median |
|---|---|
| \|dR_BS − dR_Φ\| | **1.6 mm** |
| residual-implied uncertainty on either | **16.4 mm** |

On 1643, following the filament's own per-sample probe choice: gap 2.6 mm,
slope 0.996, amplitude ratio 1.006 against the filament curve.

**So the linear proxy is not the limiting error.** It contributes ~1.6 mm where
the probes constrain the position to ~16 mm. Earlier figures of 18.9 and 31.7 mm
were configuration mismatch — different probe sets and different `fit_ip` — not
proxy error. An unmatched comparison **overstates** the approximation error.

**Branch mismatch is real but rare.** Seeding the Biot–Savart solve from Φ's
answer instead of searching globally gives an identical result on 985/985 sampled
points of 1641. On 2400, following the filament's 4-probe sets, 731/736 are
identical and **5 samples** have Φ sitting in a strictly worse basin, up to
398 mm away. `compare_methods.BS_PLOT_PHI_START` plots both curves so this is
visible per sample.

### 3.20 Uncertainty bands — conditioning, not confidence

Both magnetic curves now carry a per-sample band from their own fit residual:
`C = s²(JᵀWJ)⁻¹`, `s² = ‖r‖²/(M − p)`, with `p = 3` when the amplitude is
profiled out and 2 when fixed. The Jacobian converts a residual in Tesla into a
displacement in metres. `invert.sigma_shot()` evaluates this at *any* position,
so the filament band is computed by pushing its answer through the same residual
and the same Jacobian — same footing, directly comparable.

**Read it as conditioning, not probability.** The formula assumes independent
zero-mean noise of common variance; the residual is dominated by model error,
which is systematic and correlated between probes. It also cannot see anything
the two methods share. It is a lower bound on total error. `compare_methods.py`
shades ±1σ by default (`BAND_SIGMAS`), deliberately not 1.96σ, because "95%"
invites a reading the quantity does not support.

### 3.21 The ~15% amplitude excess — investigated, not resolved

With `fit_ip=True` the fitted amplitude should equal `Ip/I_ref` exactly, because
`adaptive_select._proxy` normalises signals by that factor. It comes out ~1.15×
that on all three shots examined (1641, 1643, 2400). What was established:

- **Two components.** `ratio = c + b/Ip` fits with R² = 0.79–0.97: multiplicative
  `c ≈ 1.12` plus an additive `b` worth 1.6–2.5 kA of equivalent current. The
  `1/Ip` shape means `b` does not scale with plasma current.
- **Uniform across probes** — flat-top medians 1.159 / 1.145 / 1.142 over
  well-behaved probes, spread 0.06–0.09. A global scale, not spatial structure.
- **Not the geometry.** `R0` has zero effect. A probe radius of 0.280 m would
  zero the ratio, but makes the *shape* fit monotonically worse (1641 residual
  0.0770 → 0.0823), so `parameters.R = 0.321 m` is right.
- **Not the filament ansatz.** Ampère's law round the probe circle — no model at
  all — gives |I_probes / I_Rogowski| = 1.04–1.14, the same magnitude.
- **Not eddy currents, for the multiplicative part.** Eddy currents must reverse
  sign between ramp-up and ramp-down. At matched plasma current the two phases
  differ by −0.002 (1641) and −0.001 (2400); corr(ratio, dIp/dt) ≈ 0.09; and the
  excess is fully present at flat-top. 1643 is the exception at −0.038, so the
  additive term `b` remains a plausible eddy signature — and note the probes at
  0.321 m sit **outside** the 0.25 m vessel, so the Ampère contour does enclose
  vessel currents.

**Two candidates remain, degenerate under every magnetic test:** the probe
calibration under-reads by ~13%, or the Rogowski over-reads by ~13%. Separating
them needs a known current — an in-vessel calibration loop, or a vacuum shot with
a known primary current. This is invisible to the pre-plasma calibration check,
because `k_t, k_oh, k_v` are *fitted* on that window and absorb a probe-gain
error.

**A loose end.** `data/<shot>/IP2.txt` exists in every shot folder and nothing in
the repository reads it — `_load_shot` takes IP1 unconditionally, while
`current_channels.resolve_all` does redundant-channel health checking for IT,
IOH and IV. The two Rogowskis agree to 1.7% on 1641 and 1643 but differ by
**9.4%** on 2400, where the Biot–Savart fitted current agrees with IP2 to 3% and
with IP1 to 13%. `ip_compare.py` plots IP1, IP2, and the two fitted currents
together.

**Practical consequence.** With `fit_ip=True` the amplitude is profiled out, so a
uniform scale error cancels in the inter-probe ratios and never reaches the
position. With `fit_ip=False` it lands in `dR`: +10.2 mm versus +31.3 mm on 1641,
all 12 probes. That 21 mm is an order of magnitude larger than the 1.6 mm proxy
error of §3.19.

### 3.22 `fit_ip` — four places, three positions, no decision

| location | value |
|---|---|
| `main.py` | `mprobe_fit_ip = True` (comment beside it marks `False` as recommended) |
| `compare_methods.py` | `FIL_FIT_IP = False` |
| `adaptive_select._estimator` | was hardcoded `False`, now a parameter |
| `CHANGES.md`, "recommended usage until gains are calibrated" | `fit_ip=True` (immune to common gain) |

`adaptive_select.adaptive_selection()` now takes `fit_ip=` instead of hardcoding
it, and `_proxy()` handles the 3-unknown case — its inline copy of the
2-unknown formula was the only thing blocking it, since Φ, the hull and the rt
field all route through `MProbeEstimator._linear_estimate_model`, which already
handles both.

**The conflict was three copies of one formula.** `adaptive_select` computed the
proxy inline in three places — the runtime proxy, the hull sweep, and the rt
field — each hardcoding the `fit_ip=False` algebra `(sig − S0) @ P.T`. With
`fit_ip=True` the estimator's `P` is 3×M and the proxy is the pair of ratios that
divides out the fitted amplitude, so all three computed nonsense: every set
scored 0.000, rt came out 0.15–0.19 m, and the hull rejected every sample. Φ
itself was never at fault — a direct round trip through it is exact to 0.00 mm in
both modes.

Fixed by giving the three sites one shared helper, `_model_proxy()`, the batched
twin of `MProbeEstimator._linear_estimate_model`. They must agree on what
`(u, v)` means or a set is scored in one space and used in another, which is
exactly what was happening. `fit_ip=True` now reaches 0.96–1.00 coverage on
1641, 1643 and 2400.

Two things this surfaced. **Stale caches survive a bug fix**: `hull_tables/` and
`rt_fields/` are keyed on `P`, `S0` and the angles, none of which changed when
the formula was corrected, so the broken artefacts were silently reused until
they were deleted by hand. The keys cover the estimator configuration but not the
code that consumes it. **And `fit_ip=True` needs three live probes**, not two,
because the amplitude is a third unknown; curation zeroes bad probes, so a
4-probe candidate with two dead members cannot be built. Those sets are now
dropped with a message rather than crashing the run.

**What it changes.** The two modes disagree substantially — on 1641,
`dR` median +32.8 mm against +15.3 mm — and the fitted current runs 1.08 (1641),
1.42 (1643) and 0.70 (2400) times IP1. The 1643 and 2400 figures are far from
both 1.0 and the Biot–Savart value of ~1.15, so the filament fitted current is
not yet a trustworthy fourth trace; §3.21's amplitude question is not settled by
having enabled this.

### 3.23 Probes 11 and 12 — two separate problems, previously conflated

There are **two** distinct issues with these probes, and an earlier revision of
this section ran them together. They are unrelated and have different remedies.

**(a) Polarity, on every shot.** Measured from the Biot–Savart residual, probes
11 and 12 read the opposite sign to the model — ratios ≈ −1.85 and −0.28 at the
fitted position on 1641 — and probes 6 and 9 are also poor. `CHANGES.md` had
already recorded this ("probes 11 and 12 POLARITY-FLIPPED, ratios ~ −1.2 and
−0.4"), so this is corroboration by an independent route, not a new finding.
Curation down-weights them by two to three orders of magnitude, which is why the
12-probe fit survives.

The structural point is worth keeping: the Φ path projects onto two or three
linear combinations and discards most per-probe error, while an absolute-field
fit sees all of it — so a bad probe is visible in Biot–Savart and largely
invisible in Φ. **No per-probe gain correction is applied anywhere.** A `gains`
hook was built and then removed, because attributing the discrepancy to gain was
an unsupported causal claim when polarity, a failed probe, and gain all fit the
evidence equally.

**(b) A data-integrity failure, on 1643 only.** Separate from (a), and covered in
§3.25.

**Retracted.** A previous revision stated that the 1643 drops were the polarity
issue crossing a threshold. That is wrong on the mechanism (a binary gate, not a
threshold on weight), wrong on probe 11's failure mode, and was asserted without
being checked. The measured ratio of −21.4 for probe 11 on 1643 was read as a
worse polarity flip; a ratio of that magnitude is not a sign error at all, and
the magnitude should have been noticed rather than only the sign.

### 3.24 A set with no degrees of freedom ranked first

On 1643 curation drops probes 11 and 12 outright. Not a weight sliding past a
cut-off — the Layer-1 **validity gate** fails them, for two different reasons,
and both are specific to that shot (§3.25). That exposed a scoring hole.

With `fit_ip=False` the top of the priority order on 1643 was **`[11 12 5 6]`** —
both dead members, **two live probes against two unknowns**. Such a fit is exact
by construction, so it round-trips perfectly, scores perfectly, and ranks first.

The mechanism: the candidate list is a fixed geometric family built *before*
curation runs, and `good_frac` and `rt` are computed from the forward **model**,
not from the data. Curation enters as a weight, never as membership, so nothing
in the scoring path knows a probe is dead. A set can therefore be scored on
twelve-probe-quality physics while running on two.

This is a sharper instance of blocking item 4: there a bad set scored well; here
a set that *cannot fail* scored best. It did not bite on 1643 — the per-sample
hull/rt gate rejected it on real data and `[2 3 8 9]` took 3583 of 3606 samples —
but it was one shot away from being selected.

**Fixed at the source.** `adaptive_selection` now requires at least one degree of
freedom, dropping any candidate whose live-probe count does not exceed the number
of unknowns (2 with `fit_ip=False`, 3 with `fit_ip=True`), and names each dropped
set and why. On 1643 that removes `[11 12 5 6]` in both modes and eight more
3-live sets under `fit_ip=True`; `order[0]` becomes `[11 3 5 9]` and all-12
respectively, and coverage is unchanged at 0.998. 1641 and 2400 are untouched:
no set there loses enough probes to trip it.

**The general lesson**, worth applying beyond this case: a score computed from
the model cannot see a data problem. Any quantity used to *rank* sets needs a
check that the set can still be falsified by the data.

### 3.26 One `FIT_IP` flag, not two

`compare_methods.py` had `FIL_FIT_IP` and `BS_FIT_IP` as independent settings.
They could disagree, which quietly made the filament-vs-Biot–Savart gap a
function of configuration rather than of method — the trap §3.19 already had to
untangle once. They are now a single `FIT_IP`, applied to both magnetic methods
and to the uncertainty-band computation, so the two are always compared on the
same footing. `BS_FOLLOW_FILAMENT` still exists, but with one flag it only has to
reconcile probe selection; the amplitude treatment is already shared.

`ip_compare.py` likewise now has one `FIT_IP` driving both fitted currents. They
must match or the two "fitted" traces are measured against different definitions
and cannot be compared, which is the whole point of that plot. With `FIT_IP=False`
neither current is fitted (both collapse to `Ip/I_ref`), and the script says so.

Also added: `PLOT_METHODS`, a per-method master switch at the top of
`compare_methods.py` that includes or excludes each method and overrides its
configuration block when off.

### 3.25 What the curation validity gate actually does, on 1643

`curation._valid_gate` is a data-integrity check, separate from the `1/sigma^2`
weighting. It returns a boolean, so a probe is either kept or dropped; it has
four conditions — too few pre-plasma samples, non-finite values, railed, and
non-stationary. Traced per probe:

| shot | probe | pinned | s2/s1 | outcome |
|---|---|---|---|---|
| 1643 | 11 | **59.8%** | 0.82 | **DROP — railed** (threshold 1%) |
| 1643 | 12 | 0.0% | **6.85** | **DROP — non-stationary** (threshold 6.0) |
| 1641 | 11 / 12 | 0.0% / 0.0% | 2.11 / 2.13 | keep |
| 2400 | 11 / 12 | 0.0% / 0.0% | 2.18 / 2.26 | keep |

**Probe 11 is railed:** 59.8% of its pre-plasma samples sit pinned at a single
extreme value. That is a saturated or stuck channel, not a signal, and it
explains both the +0.468 flat-top mean (~25× any other probe, and ~25× its own
value on other shots) and the −21.4 measured/predicted ratio.

**Probe 12 is non-stationary:** levels are fine, but the scatter in the second
half of the pre-plasma window is 6.85× the first half. Whole-window means and
standard deviations look unremarkable, which is why an earlier inspection of
those statistics found nothing — the gate tests how the scatter *evolves within*
the window.

Both are 1643-specific. On 1641 and 2400 the same probes sit comfortably inside
both thresholds.

**Worth knowing, not worth acting on yet.** Probe 12's 6.85 is only 14% past a
threshold of 6.0, and other probes on other shots reach 3.3–3.6, so the
distribution is not far below the cut. `STRUCT_RATIO = 6.0` alone separates keep
from drop here, and dropping probe 12 is what removed eight candidate sets under
`fit_ip=True`. Check the threshold's provenance before leaning on the 1643
results. Do **not** move it because one probe landed just outside — that is how a
validity cut-off gets fitted to the data in front of you.

**The gate's reason is computed and then discarded.** `_valid_gate` returns only
a boolean, so recovering *which* condition fired needed a traced wrapper.
Deliberately not changed: `valid=False` already surfaces the drop, and this
diagnosis is wanted rarely enough not to justify altering a function every method
depends on. The procedure is recorded here instead.

---

## 4. Realtime and the C port

"Realtime" = a physically live shot: samples arrive one at a time, nothing after
*t* is available. The selection logic is **causal** — the loop body reads only
sample *i* and the current set.

`adaptive_selection()` is nonetheless **batch-shaped**: proxies and rt lookups
are vectorised over the whole shot before the loop. That is a Python
optimisation, not part of the algorithm.

**Vectorisation here is not parallelism.** It amortises interpreter overhead and
uses SIMD within one core. Measured with `OMP_NUM_THREADS=1`:

| | |
|---|---|
| batched, 3606 samples | 0.18 µs/sample |
| one sample at a time | 25.5 µs/sample (139×) |
| Φ inversion `e.shift()` | 14.6 µs/sample |
| **total, live path (Python)** | **~40 µs** vs a 20 µs period at 50 kHz |

The 139× gap survives single-threading, so it is per-call overhead, not
multi-core. **On a live shot batch size is 1, so none of it applies.** The Python
live path is ~2× over budget — an interpreter cost, not an arithmetic one. The
per-sample work is one 2×M matvec, one half-space test, one grid index and one
bicubic evaluation, all trivial in C.

**To port:** move the proxy and rt lookup inside the loop, replace `_load_shot`
with the acquisition feed, omit the coverage/n_switch block entirely. No logic
changes.

---

## 5. Open items

### Settled since the last revision — recorded, not deleted

- **Feedback-coil pickup (`k_if`).** Previously the leading unexplained
  systematic in the filament method. It was tested and found unimportant; it is
  no longer an open question and no longer worth presenting. `curation_key()`
  will still invalidate cached weights automatically if a coefficient is ever
  added, so the machinery costs nothing to leave in place.
- **`position.c` calibration.** The coefficients are stale for a documented
  reason — coils repositioned after the machine moved from China (§3.13). This
  closes "are the coefficients wrong?" as a question and closes `position.c` as a
  forward candidate.
- **Weight transfer between consecutive shots.** Weights almost certainly
  transfer; the ranking they feed is only a tie-breaker, so the consequence of
  being wrong is small. Not worth further testing or presentation time.
- **IV1 / IV2 channel difference.** Does not materially affect the displacement
  result. Being checked separately under the per-channel validity workstream.

### Blocking for the presentation

1. **Method disagreement on 1643 is unexplained.** The disagreement is common to
   the whole magnetic pipeline rather than specific to a probe set — which is why
   1643 cannot validate any selection criterion (§3.10).

   *Narrowed this cycle.* Biot–Savart, configured to match the filament path
   exactly (its per-sample probe choice, `fit_ip=False`), agrees with it to
   **2.6 mm** median with slope 0.996 — so the Φ linearisation is not the source.
   Combined with §3.21, whatever remains is upstream of both: calibration, the
   probe polarity of §3.23, or the single-filament ansatz. The wording "all three
   methods disagree" was carried forward without re-verification and should be
   re-derived from current figures before it is presented.
2. **Shot 2400 changed its story.** The discovery slide now shows the fixed
   12-probe set outside its hull for **79%** of shot 2400, not the ~10% the
   earlier version implied. Computed with inherited weights (`"last"`), since
   2400's own preshot weights would force fresh Φ builds. **Check against the
   original 2400 analysis before presenting** — it is the number most likely to
   draw a question.
3. **`adaptive_selection` is validated on one shot.** §3.10 ran on 1641 only —
   the only shot with both a `_pred.txt` and spread in per-set AI agreement.
4. **Two bad sets still score well.** `[12 2 6 8]` 0.776 and `[11 1 5 7]` 0.738
   against round-trips of 179 mm. `good_frac` counts *how much* of the region is
   trustworthy, not how bad the rest gets. The per-sample test catches them, so
   this is an ordering weakness — **do not present `good_frac` as a set-quality
   score.** (Pre-disc-fix figures; §3.11.) It is also validated on one shot
   (1641) only.
5. **Shot 4404 carries one argument only.** Slide 5 is now rebuilt on it
   (§3.14). But the shot is ~7 ms long (358 samples, 333.8–340.9 ms), so it
   cannot support any other claim, and it makes the `DECOMP_WINDOW_MS`
   shape/amplitude diagnostics meaningless (n ≈ 30). Do not quote correlation or
   amplitude-ratio numbers from 4404.

### Open questions the software mockup is meant to explore

These replace the three questions that previously stood here. Source: *Software
Mockup Plan for Displacement Feedback Control*.

6. **Independent position reference.** Feeding the magnetic method signals
   generated from its own forward model tests only self-consistency, not
   accuracy. The loop needs an anchor outside the method before any stability or
   bandwidth number it produces means anything. This is the question that governs
   whether the rest of the mockup's output is meaningful.
7. **Loop rate versus sensor rate**, and whether the displacement computation
   fits inside one control period. The loop need not run at the sensor rate, and
   on the real controller it does not. Measured: ~40 µs per sample in Python
   against a 20 µs period at 50 kHz — interpreter overhead, not arithmetic
   (§4), but it is the number to quote until a C port exists.
8. **Supply bandwidth and slew rate.** The main deliverable of the mockup, and
   explicitly *conditional* on the placeholder plant model until that plant is
   anchored to something independent (item 6).
9. **Sign and unit conventions between the two displacement methods.** No longer
   abstract — see the measured flip-sign table and the undiagnosed ~165 mm offset
   in §3.13.

### New this cycle

14. **The ~15% amplitude excess (§3.21).** Narrowed to two candidates —
    probe calibration under-reading or the Rogowski over-reading, both ~13% —
    which no magnetic measurement can separate. Needs a known current. Until
    then it biases `fit_ip=False` positions by ~21 mm on 1641, which is larger
    than every other error this workstream has quantified.
15. **`fit_ip` has no agreed value (§3.22).** Four places, three positions, and
    `CHANGES.md` recommends the one the primary path does not use. This is now a
    decision to make, not a bug to find.
16. **The filament fitted current is not yet trustworthy (§3.22).** `fit_ip=True`
    now runs, but `I_fit/IP1` comes out 1.08 / 1.42 / 0.70 on 1641 / 1643 / 2400
    — inconsistent with each other and with Biot–Savart's ~1.15. Until that is
    understood, do not read the filament trace in `ip_compare.py` as a
    measurement of the plasma current.
19. **Model-derived scores cannot see data problems (§3.24).** Fixed for the
    degrees-of-freedom case, but `good_frac` and `rt` are still computed from the
    forward model while curation acts only through weights. Blocking item 4 is
    the same root cause and is not fixed by this.
20. **`STRUCT_RATIO = 6.0` provenance unchecked (§3.25).** Probe 12 on 1643
    fails it at 6.85 — 14% past the line — and that single drop removes eight
    candidate sets under `fit_ip=True`. Find where 6.0 came from before quoting
    1643 results that depend on it.
21. **Cache keys do not cover the consuming code (§3.22).** `hull_tables/` and
    `rt_fields/` are keyed on `P`, `S0` and angles, so artefacts built by a buggy
    proxy survived the fix and had to be deleted by hand. Per §6's own
    convention, this key does not cover every input.
17. **`IP2.txt` is unread (§3.21).** Present in every shot folder, ignored by
    `_load_shot`, and 9.4% away from IP1 on 2400. One channel is wrong there and
    nothing currently notices. `current_channels.resolve_all` already does this
    kind of health check for IT/IOH/IV.
18. **The lattice-resolution claim is untested.** §3.19 asserts the grid search
    finds the global minimum "subject to the lattice resolving every basin".
    Refine to 0.5 mm on one shot and confirm the set of minima is unchanged.
    Expected to pass; currently taken on faith.

### Not blocking

10. **`POSC_FLIP_SIGN` should be two flags, not one.** One flag cannot express
    the per-source polarity table in §3.13. Cosmetic today because each figure is
    produced with one source at a time, but it is a trap for anyone plotting both.
11. Legacy `phi_map.py` filenames now include a grid key, orphaning old
    `Phi_<set>.npz`. That path is superseded; rebuild cost accepted.
12. `HULL_N`: the `n`-sweep predates the disc fix and needs re-running (§3.4,
    §3.11); it was also only ever done on 1643.
13. Hulls could also be keyed and shared more aggressively; low value.

---

## 6. Conventions worth keeping

- **Cache keys must cover every input.** A key that covers some inputs is worse
  than none: it creates confidence without correctness.
- **Match the configuration before quoting a method-to-method gap.** Probe set
  and `fit_ip` must both match, or the gap measures the configuration rather than
  the method. This cost most of a cycle: figures of 18.9 and 31.7 mm collapsed to
  1.6 and 2.6 mm once the configurations were aligned (§3.19).
- **Name what a number measures, not what you hope it means.** The residual is a
  model-versus-measurement mismatch, not a noise level; the band is conditioning,
  not confidence; injected noise in the tests is not shot noise. Each of these
  was conflated at some point in this cycle and each conflation produced a wrong
  claim.
- **Code comments describe how the code works.** Project history, rationale for a
  decision, and current results belong in this document or in a README, because
  they go stale in the source and nobody re-reads them there.
- **Absorb data problems; crash on code defects.**
- **Refuse unstamped artefacts.** Unverifiable earns the same answer as stale.
- **Measure before caching.** Reading can cost more than recomputing.
- **Sample the operating region, not its bounding box.** Cost us twice: once as a
  wrong intermediate result (§3.5), once as a real defect shipped in the guard
  (§3.11).
- **Name the thing precisely.** "Hull" and "health" were used interchangeably for
  a while; they are a region and a number derived from it, and only one survived.
- **The AI camera is never used to select or tune.**
- **Prefer simple choices when the cost of being wrong is small — but verify
  that, don't assume it.** `n=25` looked like a free default and needed a sweep
  to become a decision.


---

### 3.16 The dR asymmetry, and what the 4017-subset search found

**The control asymmetry is now recorded in the code.** The feedback coils cannot
move the plasma vertically. dR is the axis that must be right; dZ is a useful
diagnostic but not actionable. Nothing in the selection logic knew this: the
ordering tiebreak used the isotropic round trip `||(u,v)-(u',v')||`, weighting an
axis the machine cannot act on equally with the one it can.

**Fixed by making the tiebreak radial**, not by touching acceptance. `_build_rt_field`
now also records `|u - u'|` (dU is the radial proxy — `hU` carries `cos(theta)`),
and `rt_field_score` breaks ties on the median of that over the good region.
`good_frac` and `RT_GOOD` still use the isotropic norm, because whether Φ inverts
at all is not a per-axis question. Only *which* acceptable set gets used is now
axis-aware. rt-field cache key bumped to `v:2`.

**Measured on 2400** (figures corrected in rev 6 — the originals were scored
against 1641's file): coverage 0.9879 and 3 switches, unchanged; the order
changes; camera agreement 29.9 mm dR in the disputed window, 43.1 mm over the
whole shot. This is a correction of principle, not a rescue: the old
ordering already picked the better set for dR on this shot, by accident.

**What prompted it.** The adaptive trace disagrees with the camera by ~58 mm in
dZ over 343.17-348.05 ms on 2400. Ruled out in order: not an acceptance
threshold (the chosen set's rt is 0.0000 mm on all 245 samples it served, and
sweeping RT_GOOD 1.5 → 0.01 mm changes nothing); and not a better probe set.

> **Rev 6 correction.** A third argument stood here — that the camera was not at
> fault because its confidence in that window was the lowest of the shot (median
> 0.718 against 0.835) with two demonstrable misfits at 342.0 and 343.0 ms, plus
> an eyeball of the video. **All of that was 1641's data** (§3.17). On 2400's own
> file the window's confidence is **0.828 against 0.782 for the shot** — the
> window is *better* than average, not worse — and the two "misfits" carry 0.848
> and 0.892, among the best frames in the shot. The sub-argument is struck, and
> the `AI_CONF_MIN = 0.6` proposal that came from it is withdrawn; it stays 0.5.
>
> **The conclusion survives re-measurement on correct data.** The trade-off is
> unchanged in structure: `12 3 6 9` gives 29.9 mm dR / 54.6 mm dZ and
> `12 4 6 10` gives 70.6 / 21.1, both admissible on 100% of the window. The
> radial tiebreak remains justified.

**The exhaustive search.** All 4017 subsets of sizes 3-12 were screened. 39 pass
the hull test on >=90% of the window; every one tested then fails the rt test.
The survivors cluster at cond ~2.6 against the incumbent's 72.0 — **the
best-conditioned sets are precisely the ones whose map folds.** Conditioning is
not a proxy for usability here, and choosing sets on it would have picked the
worst available.

> **Rev 6 retraction.** This section originally added "only antipodal 4-probe
> sets survive both tests, which vindicates that design empirically". That is
> **wrong**, and it was a sampling error rather than a consequence of the camera
> data: only 12 of the 39 hull survivors were ever tested. Re-running with
> correct detections found `4 6 10 11 12` — five probes, non-antipodal, outside
> the current candidate list — admissible on **100%** of the window, halving the
> dZ error (54.6 → 27.8 mm) at a cost of ~15 mm in dR (29.9 → 44.6 mm). A
> genuine compromise candidate. The conditioning inversion is unaffected: every
> cond ≈ 2.6 set is still admissible on 0% of the window.

**Open:** 34 of the 39 hull survivors were never rt-tested (near-duplicates of
the four that failed); this is one window of one shot; and RT_GOOD was only ever
tightened, never loosened, which is the direction that would admit more of the 39.

**Three engineering findings recorded in CHANGES.md Part 5.** Φ was being built
inside `__init__`, making every cheap question expensive. The `cal_signal` sweep
is shared across all probe sets and all weights, so recomputing it per set was
pure waste — the subset screen went from an estimated 2-33 hours to 2.0 s on the
strength of that alone. And the Φ build ran `griddata(cubic)` twice over the same
scattered points, rebuilding the same Delaunay triangulation each time.

**Measured, per Φ build over 314,721 grid points:** the `cal_signal` sweep is
27.7 s and now runs once for all sets; the projection through `P` is 0.11 s; the
cubic interpolation was 8.7 s and is 4.2 s with the triangulation shared
(bit-identical output, `max|diff| = 0.000e+00`). A new shot's full 16-set Phase A
rebuild: **~600 s → 172 s**, measured end to end.

**What does not share, and why it is the floor.** The interpolation is over the
(u,v) plane, whose geometry depends on the probe set and on the weights. Only the
forward model is common. So ~4 s per set stands unless the interpolation itself
is replaced — the earlier hope that this would reduce to "seconds" was wrong, and
was stated before it was measured.


---

### 3.17 The camera prediction files were the wrong shot's

**Every shot folder carried a byte-identical copy of shot 1641's `_pred.txt`**
(md5 `682c28ce…`). The copies were valid detections — correct format, sensible
confidences, real plasma positions — of a *different discharge*. Nothing crashed,
nothing looked malformed, and `load_ai_camera` returned a plausible trace for
every shot. Confirmed by comparison against a correctly-exported 1641 file:
`txc` identical on all 237 overlapping frames, `tyc`/`tr` differing on one row by
0.5 px.

**Cost.** Shot 2766 appeared to disagree with the camera by 116 mm radially and
was treated as a broken outlier through an extended investigation; on its own
detections it is **23.6 mm**, among the best-agreeing shots in the set. An
independent detector was nearly written from scratch to explain a discrepancy
that did not exist.

**Why it survived.** The failure is invisible to every check that was in place.
The data is well-formed, so no parser complains. The positions are physical, so
no plausibility check fires. And 1641's span (285–452 ms) *contains* every other
shot's plasma window, so a naive overlap test passes it silently. Only a
containment test separates them.

**Guard added** to `load_ai_camera`: a prediction file more than 4× wider than
the plasma window with over 75% of its frames outside it prints a loud warning.
Thresholds set from the real files — a correct file legitimately runs a little
either side (2400: 24 ms span against a 13 ms window, 51% outside), the 1641
copies are 166 ms against 13 ms with 90% outside. Verified to fire on the bad
file and stay silent on all four correct ones.

**Corrected camera agreement** (adaptive, median |method − camera|, mm):

| shot | dR before | dR after | dZ before | dZ after |
|---|---|---|---|---|
| 2766 | 115.7 | **23.6** | 14.2 | 8.6 |
| 1643 | 45.0 | **31.8** | 12.6 | 21.4 |
| 2400 | 12.3 | **43.1** | 25.6 | 33.9 |
| 4404 | 26.5 | **15.8** | 87.5 | 128.1 |

**`shift_domain` scan, re-measured** (dR, mm): 1643 23.0/**31.8**/31.8,
2766 **23.6**/**23.6**/28.2, 4404 27.9/15.8/**13.1**, 2400 87.3/**43.1**/87.3 at
0.10/0.14/0.16. Mean 40.5 / **28.6** / 40.1.

**0.14 is retained** — best mean radial error and the only setting never badly
wrong — but **no domain wins on every shot**: 1643 and 2766 prefer 0.10, 4404
prefers 0.16. It is the best compromise, not a physical optimum, and the median
across shots is nearly flat (25.8 / 27.7 / 30.0). Coverage argues the other way
throughout: 0.16 is monotonically better (97.7% mean vs 96.2%). Choosing 0.14
prefers radial accuracy over sample count, backed by the structural point that
samples recovered past 0.14 pass `rt` at 0.0000 mm and so cannot be policed.

**2400's advantage at 0.14 is narrow and set-driven, not intrinsic.** The 87.3 mm
figure at both 0.10 and 0.16 is a coincidence of the median, not identical
results — the frame sets differ. The mechanism: `12 4 6 10` carries 107 mm median
radial error on this shot, and 0.14 wins only because it hands 10 frames to
`12 3 6 9` (29.9 mm) while 0.10 and 0.16 give nearly everything to the bad set.

**Unresolved.** 4404's dZ is 94–128 mm at every domain, the worst in the set, and
it worsens as dR improves. Only 13 confident frames on a 7 ms shot. Not
investigated.

**Rule adopted.** A cross-check instrument's data must be verified to belong to
the shot it is compared against. An independent reference that is silently the
wrong shot is worse than no reference: it manufactures false defects in the
method under test, and consumes the effort spent chasing them.


---

### 3.18 The candidate list is not a uniform family

`DEFAULT_CANDIDATES` is the 15 canonical antipodal 4-probe sets from
`parameters.all_arrays` **plus the full 12-probe array** — 16 entries, but not 16
of a kind. The 12-probe set is neither 4-probe nor antipodal, and it frequently
ranks first: it is `order[0]` on shot 2400.

This was being described loosely as "16 candidate sets" in a way that implied a
uniform antipodal-quad family, and that phrasing has been corrected in the code
comments and README. It also sharpens the retraction in §3.16: there was never an
"antipodal design" under test, because the default list was never restricted to
antipodal sets.

**Selection places no constraint on set size or geometry.**
`adaptive_selection()` accepts any list of probe sets; `DEFAULT_CANDIDATES` is a
default, not a restriction. That is what makes the exhaustive-subset result
actionable — `4 6 10 11 12` (five probes, non-antipodal) could simply be added to
the list, with no change to the selection machinery.

**Comment policy adopted (rev 7).** `.py` comments and docstrings describe the
code as it stands, not its history. Rationale for past changes lives in this
document and in `CHANGES.md`. The `RETIRED` docstring section and similar
retrospective blocks were removed from the source; their content is preserved
in §3.15 and `CHANGES.md` Parts 4-7.
