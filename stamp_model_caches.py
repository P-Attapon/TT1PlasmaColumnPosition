"""
stamp_model_caches.py -- one-off migration for caches built before stamping.

=============================================================================
ADDED FILE (not in the original Attapon et al. repository).

WHY THIS EXISTS
---------------
cache_keys.py made every persisted artefact carry a fingerprint of the model
configuration that produced it, and the loaders now refuse anything unstamped.
That is correct for artefacts built from here on, but it strands the ones that
already exist: a phi_tables/ directory is ~170 MB and takes minutes per probe set
to regenerate, purely to add a field that says what it already is.

This tool writes that field, without rebuilding anything.

THE ASSERTION YOU ARE MAKING
----------------------------
Stamping does NOT verify. There is no way to look at a finished Phi map and
recover the shift_domain it was swept over. Running --apply is YOU asserting that
the existing caches were built with the parameters currently in parameters.py.

So the tool refuses to do it quietly. It prints the full current configuration and
requires the explicit --apply flag; the default action is a dry run that changes
nothing. If you are not sure the caches match, the safe move is to delete
phi_tables/ and let them rebuild -- slow, but it cannot be wrong.

Once run, this file has no further use: everything written afterwards is stamped
at creation. Keep it for the next person who restores an old phi_tables/ from a
backup.

USAGE
-----
    python stamp_model_caches.py            # dry run: list what would change
    python stamp_model_caches.py --apply    # write the stamps
=============================================================================
"""
import os
import sys
import json
import glob
import shutil
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from methods_script.toroidal_filament.cache_keys import (   # noqa: E402
    forward_model_key, curation_key, describe)

_TF = os.path.join(_HERE, "methods_script", "toroidal_filament")
PHI_DIR = os.path.join(_TF, "phi_tables")
PKL = os.path.join(_TF, "coefficient_nested_dict.pkl")
WEIGHTS_DIR = os.path.join(_TF, "weights_store")


def stamp_npz(path, apply):
    """Add fm_key/model to a .npz that lacks them.

    np.savez cannot append, so the archive is rewritten. It is written to a
    temporary file in the same directory and moved into place, so an interrupted
    run cannot leave a half-written map where a good one used to be.
    """
    with np.load(path, allow_pickle=True) as d:
        if "fm_key" in d.files:
            return "already stamped" if str(d["fm_key"]) == forward_model_key() \
                else f"STAMPED WITH A DIFFERENT MODEL ({str(d['fm_key'])}) - left alone"
        arrays = {k: d[k] for k in d.files}
    if not apply:
        return "would stamp"
    arrays["fm_key"] = forward_model_key()
    arrays["model"] = describe()
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=os.path.dirname(path))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        shutil.move(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, path)
    finally:
        for leftover in (tmp, tmp + ".npz"):
            if os.path.exists(leftover):
                os.remove(leftover)
    return "stamped"


def stamp_json(path, apply, keys):
    """Add missing stamp fields to a JSON store record."""
    with open(path) as fh:
        rec = json.load(fh)
    want = {"fm_key": forward_model_key(), "curation_key": curation_key()}
    want = {k: v for k, v in want.items() if k in keys}
    if all(rec.get(k) == v for k, v in want.items()):
        return "already stamped"
    if any(rec.get(k) not in (None, v) for k, v in want.items()):
        return "STAMPED WITH A DIFFERENT MODEL - left alone"
    if not apply:
        return "would stamp"
    rec.update(want)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    return "stamped"


def main():
    apply = "--apply" in sys.argv
    print("Current forward model:")
    print(f"  {describe()}")
    print(f"  forward_model_key = {forward_model_key()}")
    print(f"  curation_key      = {curation_key()}")
    print()
    if not apply:
        print("DRY RUN - nothing will be written. Re-run with --apply to stamp.")
        print("Only do that if the caches below WERE built with the model above.")
        print()

    n = 0
    for path in sorted(glob.glob(os.path.join(PHI_DIR, "*.npz"))):
        print(f"  {os.path.basename(path):40s} {stamp_npz(path, apply)}")
        n += 1

    if os.path.exists(PKL):
        meta = PKL + ".meta.json"
        if os.path.exists(meta):
            cur = json.load(open(meta)).get("fm_key")
            status = "already stamped" if cur == forward_model_key() \
                else f"STAMPED WITH A DIFFERENT MODEL ({cur}) - left alone"
        elif apply:
            # taylor_order/decimal_precision are unknowable from the .pkl alone;
            # recorded as null rather than guessed, so a later reader can tell the
            # difference between "order 3" and "nobody knows".
            json.dump({"fm_key": forward_model_key(), "model": describe(),
                       "taylor_order": None, "decimal_precision": None,
                       "note": "stamped retroactively by stamp_model_caches.py; "
                               "order/precision unknown"},
                      open(meta, "w"), indent=2)
            status = "stamped (order/precision recorded as unknown)"
        else:
            status = "would stamp"
        print(f"  {os.path.basename(PKL):40s} {status}")
        n += 1

    for d, keys in ((WEIGHTS_DIR, {"curation_key"}),):
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            print(f"  {os.path.join(os.path.basename(d), os.path.basename(path)):40s} "
                  f"{stamp_json(path, apply, keys)}")
            n += 1

    if n == 0:
        print("  (no cached artefacts found - nothing to migrate)")
    print()
    print("Done." if apply else "Dry run complete.")


if __name__ == "__main__":
    main()
