"""
cache_keys.py -- fingerprints that make every persisted artefact self-invalidating.

=============================================================================
ADDED FILE (not in the original Attapon et al. repository).

THE BUG THIS EXISTS TO PREVENT
------------------------------
Several artefacts in this repo are EXPENSIVE, DERIVED, and CACHED ON DISK:

    phi_tables/PhiM_<hash>.npz      M-probe inverse map        (mprobe.py)
    phi_tables/Phi_<set>.npz        legacy 4-probe map         (phi_map.py)
    coefficient_nested_dict.pkl     paper 1D Taylor coeffs     (coefficient.py)
    weights_store/<shot>.json       per-shot probe weights     (weights_cache.py)
    hull_tables/hull_<hash>.npz     reachable-region hull      (adaptive_select.py)
    rt_fields/rtf_<hash>.npz        round-trip field rt(u,v)   (adaptive_select.py)

Every one of them is a function of configuration held elsewhere in the package.
If that configuration changes and the cache key does not, the next run silently
loads an artefact built under the OLD configuration and produces wrong numbers
with no error, no warning, and no visible symptom.

This was not hypothetical. Before this module existed, mprobe's cache key covered
the ESTIMATOR configuration (probes / weights / fit_ip / gains / grid) but not the
FORWARD-MODEL configuration. Changing `shift_domain` from 0.14 m to 0.10 m --
exactly the parameter the project's own record flags as a live, revisited
decision -- produced an IDENTICAL key and reloaded the 0.14 m map:

    shift_domain = 0.14 -> hash f19e61db70
    shift_domain = 0.10 -> hash f19e61db70      <-- same file, wrong domain

THE RULE
--------
A cache key must cover EVERY input the artefact depends on. Two groups matter:

  (a) FORWARD-MODEL config -- what cal_signal() computes and over what region:
      shift_domain, R0, R, mu, I, and the probe angles.   forward_model_key()

  (b) MEASUREMENT-CORRECTION config -- what turns raw probe volts into a
      calibrated field, and the thresholds that grade the result:
      calibration_coeff (kt/koh/kv) and the curation gates.   curation_key()

Which group applies depends on the artefact, and the two are NOT
interchangeable. The Phi maps depend on (a) only: cal_signal models the PLASMA's
field from geometry alone and never sees kt/koh/kv, which correct the MEASURED
field on the other side of the estimator. The stored weights depend on (b) only:
they come from the pre-plasma residual B - (kt*It + koh*Ioh + kv*Iv). Rankings
depend on BOTH. Keying an artefact on the wrong group is as bad as not keying it
at all, so each call site states which group it uses and why.

WHAT IS DELIBERATELY EXCLUDED
-----------------------------
Nothing about a particular SHOT (its data, its Ip, its noise) belongs in these
keys. Shot identity is already the store filename where it matters. These keys
describe the MODEL, so one fingerprint stays comparable across all shots.

FILENAME KEY vs PROVENANCE STAMP
--------------------------------
Two mechanisms are used, and the choice between them is deliberate:

  * ESTIMATOR config goes in the FILENAME. It varies routinely -- you legitimately
    hold maps for several probe sets at once -- so one file per configuration,
    side by side, is what you want.

  * FORWARD-MODEL config goes in a STAMP written INSIDE the artefact and checked
    on every load. It is a global property of the model; nobody wants to hold maps
    for two different domains simultaneously. Changing it should HALT with an
    explanation, not silently trigger a ~170 MB rebuild that just looks like a
    hang. So a mismatch REFUSES rather than rebuilds.

USAGE
-----
    from .cache_keys import forward_model_key, curation_key, describe

    key = forward_model_key()    # whole forward model, probe-independent
    key = curation_key()         # calibration coeffs + curation thresholds

describe() returns a human-readable expansion for error messages -- when a cache
is rejected the user needs to know WHICH parameter moved, and a bare md5 digest
cannot tell them that.
=============================================================================
"""
import hashlib

from .parameters import (shift_domain, R0, R, mu, I, coil_angle_dict,
                         calibration_coeff)

# Bump when the KEY CONSTRUCTION itself changes (not when a parameter value
# changes). Incrementing invalidates every cache, which is the point: it
# guarantees artefacts built under an older keying scheme can never be mistaken
# for current ones.
KEY_FORMAT = 1

_ALL_PROBES = tuple(sorted(coil_angle_dict))


def digest(s, n=10):
    """Short stable hex digest of a key string. 10 chars is ample: these keys
    discriminate between a handful of configurations, not adversarial input."""
    return hashlib.md5(s.encode()).hexdigest()[:n]


def forward_model_key():
    """Fingerprint of everything cal_signal() and the swept domain depend on.

    Deliberately PROBE-INDEPENDENT -- it covers all 12 angles regardless of which
    probes a given artefact uses. That over-invalidates slightly (correcting probe
    3's angle invalidates a map built from probes 1/4/7/10) but it errs on the safe
    side, and it buys something worth more: ONE fingerprint describes the whole
    model, so a single stamp can be verified against every cached artefact in the
    package instead of each needing its own probe-aware key.

    Formatting is fixed-precision on purpose. %.12g round-trips a double exactly
    while staying insensitive to repr differences across Python versions, so the
    same configuration yields the same key on any machine.
    """
    parts = [
        f"fmt:{KEY_FORMAT}",
        f"dom:{shift_domain:.12g}",     # swept region half-width [m]
        f"R0:{R0:.12g}",                # major radius [m]
        f"R:{R:.12g}",                  # probe-circle radius [m]
        f"mu:{mu:.12g}",                # permeability
        f"I:{I:.12g}",                  # model current [A]
        "ang:" + " ".join(f"{p}={coil_angle_dict[p]:.12g}" for p in _ALL_PROBES),
    ]
    return digest("|".join(parts))


def curation_key():
    """Fingerprint of everything the per-probe WEIGHTS depend on.

    Two inputs, both able to change a weight silently:

      * calibration_coeff -- the pre-plasma residual is B - (kt*It + koh*Ioh +
        kv*Iv), so any coefficient edit changes sigma and hence w = 1/sigma^power.
        This is the coupling that will fire when the outstanding k_if
        (feedback-coil) term is added: weights cached before that change are stale.
      * the curation gate thresholds, which decide whether a probe is dropped
        outright (w = 0) rather than merely down-weighted.

    curation is imported lazily so this module stays free of import cycles:
    curation.py imports parameters.py, and nothing should force an ordering
    between the two.
    """
    from . import curation
    coeffs = "|".join(f"{k}={calibration_coeff[k]:.12g}"
                      for k in sorted(calibration_coeff))
    parts = [
        f"fmt:{KEY_FORMAT}",
        f"power:{curation.WEIGHT_POWER:.12g}",
        f"struct:{curation.STRUCT_RATIO:.12g}",
        f"rail:{curation.RAIL_FRAC:.12g}",
        f"minsamp:{curation.MIN_SAMPLES:d}",
        "coeff:" + coeffs,
    ]
    return digest("|".join(parts))


def describe():
    """Human-readable expansion of the current forward-model configuration.

    Used in cache-rejection messages. A bare hex digest tells the user THAT
    something changed; this tells them WHAT to go and look at.
    """
    return (f"shift_domain={shift_domain:g} m, R0={R0:g} m, R={R:g} m, "
            f"mu={mu:.6g}, I={I:g} A, angles={{"
            + ", ".join(f"{p}:{coil_angle_dict[p]:.6f}" for p in _ALL_PROBES) + "}")
