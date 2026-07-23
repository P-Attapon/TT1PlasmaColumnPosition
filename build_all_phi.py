"""Prebuild 2D inverse maps (Phi) for probe sets.

ADDED (not in the original repo). Convenience wrapper around
methods_script.toroidal_filament.phi_map.build_phi.

Usage (VS Code Run button uses the default; CLI overrides):
    python build_all_phi.py                 # build the default set below
    python build_all_phi.py "1 4 7 10"      # build one set
    python build_all_phi.py all             # build every set in all_arrays
"""
import sys, types
try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    _t = types.ModuleType("tqdm")
    _t.tqdm = lambda it=None, **kw: it if it is not None else (lambda x: x)
    sys.modules["tqdm"] = _t

from methods_script.toroidal_filament.phi_map import build_phi
from methods_script.toroidal_filament.parameters import all_arrays, probe_lst_to_str

DEFAULT_SET = "1 4 7 10"

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SET
    if arg == "all":
        sets = [probe_lst_to_str(s) for s in all_arrays]
    else:
        sets = [arg]
    for s in sets:
        print(f"building Phi for [{s}] ...")
        print("  ->", build_phi(s))
    print("done.")
