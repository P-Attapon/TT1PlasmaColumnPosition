"""
weights_cache.py -- persist and retrieve per-shot preshot probe weights.

The preshot weights w_i = 1/sigma_i^2 depend ONLY on each shot's pre-plasma
noise, not on the probe set. This module stores them so that:

  * OFFLINE: "auto" computes weights once per shot (in TFM_main) and, as a side
    effect, saves them here keyed by shot. All 16 default candidate sets on that shot
    then share the identical weight vector, so their Phi caches are consistent
    and no weight is recomputed per set.

  * REAL-TIME: weights="last" loads the most recently stored weights instead of
    running curation on the current shot. This lets the preshot window use a
    fixed weight vector (hence precomputable Phi maps) at the cost of assuming
    the previous shot's noise is representative. That assumption -- that weights
    change little between consecutive shots -- is UNTESTED here and should be
    checked before relying on it (low priority).

Storage: one JSON file per shot under weights_store/, plus a pointer file
'latest.json' updated on every save so "last" is O(1). Weights are keyed by
probe number (int).

PROVENANCE
----------
A stored weight vector is a function of the CALIBRATION COEFFICIENTS (the
pre-plasma residual is B - (kt*It + koh*Ioh + kv*Iv), so sigma and hence
w = 1/sigma^power move with them) and of the CURATION GATE THRESHOLDS. Neither
appears in the filename -- the filename carries only the shot number -- so every
record is stamped with curation_key() and the stamp is checked on load.

A mismatch WARNS and returns None rather than raising: unlike a Phi map, weights
are cheap to recompute, and "auto" simply reruns curation. Refusing to load is
therefore a graceful degradation, not a halt. The case this exists for is the
outstanding k_if (feedback-coil) term: adding it changes every sigma, and without
this stamp weights cached beforehand would keep being served to weights="last".
"""
import os
import json
import glob
import time

from .cache_keys import curation_key

_STORE_DIR = os.path.join(os.path.dirname(__file__), "weights_store")
_LATEST = os.path.join(_STORE_DIR, "latest.json")


def _ensure_dir():
    os.makedirs(_STORE_DIR, exist_ok=True)


def _shot_id(shot_path):
    """Stable id from the shot directory name (e.g. 'data/2400' -> '2400')."""
    return os.path.basename(os.path.normpath(shot_path))


def _stamp_ok(payload, what):
    """True if a stored record was produced under the current curation config.

    Warns and returns False otherwise, including for records with no stamp at all
    (written before stamping existed) -- an unverifiable weight vector gets no
    more trust than a known-stale one, and recomputing costs one curation pass.
    """
    stamp = payload.get("curation_key")
    if stamp == curation_key():
        return True
    if stamp is None:
        print(f"[weights_cache] ignoring {what}: no curation stamp "
              f"(written before cache stamping); weights will be recomputed.")
    else:
        print(f"[weights_cache] ignoring {what}: calibration/curation config "
              f"changed (stamp {stamp} != current {curation_key()}); "
              f"weights will be recomputed.")
    return False


def save_weights(shot_path, weights):
    """Persist a {probe:int -> weight:float} dict for one shot, and mark latest.

    Called by TFM_main after an "auto" curation run. Silent no-op safety: any IO
    failure is swallowed so it can never break a displacement calculation.
    """
    try:
        _ensure_dir()
        shot = _shot_id(shot_path)
        payload = {"shot": shot, "t_saved": time.time(),
                   "curation_key": curation_key(),
                   "weights": {str(int(p)): float(w) for p, w in weights.items()}}
        path = os.path.join(_STORE_DIR, f"{shot}.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        with open(_LATEST, "w") as fh:
            json.dump(payload, fh, indent=2)
        return path
    except Exception as e:
        print(f"[weights_cache] save skipped ({type(e).__name__}: {e})")
        return None


def load_latest():
    """Return the most recently saved {probe:int -> weight:float} dict, or None.

    Used for weights='last' (real-time). Falls back to the newest per-shot file
    if the latest pointer is missing.
    """
    try:
        if os.path.exists(_LATEST):
            with open(_LATEST) as fh:
                payload = json.load(fh)
        else:
            files = [f for f in glob.glob(os.path.join(_STORE_DIR, "*.json"))
                     if not f.endswith("latest.json")]
            if not files:
                return None
            newest = max(files, key=os.path.getmtime)
            with open(newest) as fh:
                payload = json.load(fh)
        if not _stamp_ok(payload, "latest"):
            return None
        return {int(p): float(w) for p, w in payload["weights"].items()}, payload.get("shot")
    except Exception as e:
        print(f"[weights_cache] load_latest failed ({type(e).__name__}: {e})")
        return None


def load_shot(shot_path):
    """Return the stored weights for a specific shot, or None if not stored."""
    try:
        path = os.path.join(_STORE_DIR, f"{_shot_id(shot_path)}.json")
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            payload = json.load(fh)
        if not _stamp_ok(payload, _shot_id(shot_path)):
            return None
        return {int(p): float(w) for p, w in payload["weights"].items()}
    except Exception as e:
        print(f"[weights_cache] load_shot failed ({type(e).__name__}: {e})")
        return None
