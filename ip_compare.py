"""Compare every available estimate of the plasma current for a shot.

    python ip_compare.py 1641 1643 2400

Four traces, and they should agree:

  IP1          the Rogowski channel every method in this repository uses
  IP2          the second Rogowski channel, which nothing else reads
  filament     the current fitted by mprobe via adaptive_selection
  Biot-Savart  amp * I_ref from the Biot-Savart fit

Both fitted currents use the same amplitude treatment, set by FIT_IP, so they are
measured against one definition. With FIT_IP=False neither is fitted -- both are
just Ip/I_ref -- so the comparison is only meaningful with FIT_IP=True.

The Biot-Savart amplitude should equal Ip/I_ref exactly, because
adaptive_select._proxy normalises signals by that factor. It comes out about
15% high on every shot examined, which is why this comparison exists. The
filament fitted current is less consistent still (1.08 / 1.42 / 0.70 x IP1 on
1641 / 1643 / 2400) and is not yet a trustworthy measurement -- see
methods_script/biot_savart/README.md and the project context doc.
"""

import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from methods_script.biot_savart import adapter, invert

DATA_DIR = "data"
OUT_DIR = os.path.join("result_plot", "ip_compare")
WEIGHTS = "auto"
FLAT_TOP_FRAC = 0.6      # define flat-top as Ip above this fraction of its peak

# Amplitude treatment for BOTH fitted currents. They must match, or the two
# fitted traces are measured against different definitions and cannot be
# compared -- which is the whole purpose of this plot. One flag drives both.
FIT_IP = True


def _read_channel(path):
    d = pd.read_csv(path, sep=r"\s+", skiprows=8, header=None, names=["t", "v"])
    return d["t"].to_numpy(), d["v"].to_numpy()


def compare(shot):
    geom = adapter.load_geometry(strict=True)
    fwd = adapter.load_forward("internal", geom)
    fwd_many = adapter.load_forward_many(geom)
    t, ip1, B = adapter.load_shot(shot, data_dir=DATA_DIR)
    w = adapter.load_weights(shot, WEIGHTS)

    if not FIT_IP:
        print(f"[{shot}] note: FIT_IP=False, so the two 'fitted' traces are just "
              f"Ip/I_ref and carry no independent information.")
    traces = {"IP1 (used by every method)": (t, ip1)}

    p2 = os.path.join(DATA_DIR, str(shot), "IP2.txt")
    if os.path.exists(p2):
        t2, ip2 = _read_channel(p2)
        traces["IP2 (read by nothing else)"] = (t, np.interp(t, t2, ip2))

    r = invert.invert_shot(fwd, fwd_many, t, ip1, B, weights=w, fit_ip=FIT_IP)
    traces["Biot-Savart, fitted"] = (t, r["amp"] * geom.I_ref)

    try:
        from adaptive_select import adaptive_selection
        sel = adaptive_selection(shot, weights_source=WEIGHTS, fit_ip=FIT_IP)
        traces["filament, fitted"] = (sel["t_ms"], sel["Ip_used_A"])
    except Exception as exc:
        print(f"[{shot}] filament fitted current unavailable: {exc}")

    hi = ip1 > FLAT_TOP_FRAC * np.nanmax(ip1)
    print(f"[{shot}] flat-top ratios against IP1:")
    for name, (tt, v) in traces.items():
        if name.startswith("IP1"):
            continue
        vv = np.interp(t, tt, v) if len(tt) != len(t) else v
        m = hi & np.isfinite(vv) & (ip1 != 0)
        if m.any():
            print(f"    {name:32s} {np.median(vv[m] / ip1[m]):.3f}")

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    # Fix one colour per source, so the top panel and the ratio panel agree even
    # though the ratio panel omits IP1 (whose ratio to itself is 1 by
    # definition). Interpolate every trace onto the shot time base t once.
    colors = {name: f"C{i}" for i, name in enumerate(traces)}
    for name, (tt, v) in traces.items():
        vv = np.interp(t, tt, v) if len(tt) != len(t) else np.asarray(v)
        ax.plot(t, vv / 1e3, lw=1.4, color=colors[name], label=name)
        if not name.startswith("IP1"):
            with np.errstate(divide="ignore", invalid="ignore"):
                axr.plot(t, np.where(ip1 != 0, vv / ip1, np.nan), lw=1.2,
                         color=colors[name], label=name)
    axr.axhline(1.0, ls="--", color="gray", alpha=0.6)
    ax.set_ylabel("plasma current [kA]")
    axr.set_ylabel("ratio to IP1")
    axr.set_xlabel("time [ms]")
    axr.set_ylim(0.5, 1.6)
    ax.legend(fontsize="small", frameon=False)
    ax.set_title(f"Shot {shot} — plasma current by source")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{shot}_ip_compare.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[{shot}] wrote {path}")


if __name__ == "__main__":
    shots = sys.argv[1:]
    if not shots:
        sys.exit("usage: python ip_compare.py <shot> [<shot> ...]")
    for sh in shots:
        compare(sh)
