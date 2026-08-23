"""
current_channels.py -- Redundant current-channel resolver with health gate.

BACKGROUND
----------
Every TT-1 current diagnostic has two channels (e.g. IP1/IP2, IT1/IT2, IOH1/IOH2,
IV1/IV2). The code historically hardcoded one channel per signal. On shot 3970 the
IV2 integrator was broken (std 5.3 A in-discharge vs 502 A on IV1), so the pickup
subtraction   B_corrected = raw - kt*It - koh*Ioh - kv*Iv   received Iv ≈ 0, leaving
~22% of each probe's signal uncorrected and the displacement was estimated as +127 mm
instead of the AI-indicated ~-35 mm.

PRIMARY CHANNEL AND SIGN CONVENTION
-------------------------------------
The calibration coefficients (kt, koh, kv) were derived assuming specific channel
readings. The original code hardcoded:

    IP  -> IP1   (plasma current; Rogowski, positive)
    IT  -> IT1   (toroidal-field coil current)
    IOH -> IOH1  (ohmic-heating coil current)
    IV  -> IV2   (vertical-field coil current)

These are the PRIMARY channels. The resolver always outputs a value in the primary
channel's convention. The secondary channels have the same physical meaning but
different integrator polarity; the measured sign relationship is:

    IP1 ≈  +IP2   (same sign)
    IT1 ≈  -IT2   (opposite sign)
    IOH1 ≈ -IOH2  (opposite sign)
    IV2  ≈ -IV1   (opposite sign; IV2 is primary, IV1 is secondary)

CHANNEL_SIGN[base] = sign such that PRIMARY ≈ SIGN * SECONDARY.
When secondary substitutes for dead primary: output = SIGN * secondary.
When primary substitutes for dead secondary: output = primary as-is.

NOTE: IV sign is inferred from a displacement fit on shot 3970 (the only shot
where IV2 was dead). Not yet confirmed from a shot where both IV channels are live.

IP IS NOT AVERAGED
------------------
IP is used as kappa = Ip / I_PARAM. Always uses IP1 when healthy; falls back to
sign*IP2. Does NOT average even when both healthy (avoids changing kappa for working
shots).

DECISION LOGIC (per pair, per shot, once in the preshot window)
---------------------------------------------------------------
1. Compute in-discharge std of each channel.
2. Dead = std < DEAD_STD_FLOOR (absolute) OR std < DEAD_REL_FLOOR * max(std pair).
3. Decision:
   a. Primary healthy, secondary healthy, agreement OK -> average (primary + sign*secondary)/2.
      IP exception: always use primary only.
   b. Primary healthy, secondary healthy, but disagree badly -> WARN, use primary.
   c. Primary dead, secondary healthy -> sign * secondary (matches primary convention).
   d. Primary healthy, secondary dead -> primary as-is.
   e. Both dead -> RAISE RuntimeError.
"""
import os
import numpy as np
import pandas as pd

# Primary channel per signal (what the calibration coefficients were fit to).
PRIMARY = {"IP": 1, "IT": 1, "IOH": 1, "IV": 2}

# Sign: primary ≈ SIGN * secondary (measured from data).
CHANNEL_SIGN = {"IP": +1, "IT": -1, "IOH": -1, "IV": -1}

# Health thresholds (tuned on shots 3970 and 2766).
DEAD_STD_FLOOR = 10.0    # A; absolute std below this = dead regardless
DEAD_REL_FLOOR = 0.02    # fractional; std < this fraction of the pair's max = dead
DISAGREE_MAX   = 0.15    # median |primary - sign*secondary| / peak > this = warn


def _rd(path):
    return pd.read_csv(path, sep=r"\s+", skiprows=8, header=None, names=["t", "v"])


def _discharge_window(shot_dir, discharge_current=2500.0):
    ip = _rd(os.path.join(shot_dir, "IP1.txt"))
    t  = ip["t"].to_numpy(); v = ip["v"].to_numpy()
    m  = np.abs(v) > discharge_current
    if not m.any():
        return np.ones(len(t), dtype=bool)
    i0, i1 = np.where(m)[0][[0, -1]]
    mask = np.zeros(len(t), dtype=bool)
    mask[i0:i1 + 1] = True
    return mask


def _is_dead(v, peer_std):
    s = float(np.std(v))
    return s < DEAD_STD_FLOOR or (peer_std > 0 and s < DEAD_REL_FLOOR * peer_std)


def resolve(shot_dir, base, discharge_current=2500.0, average=True):
    """Return (values: np.ndarray, provenance: str) in the primary channel's
    convention (suitable for direct use with calibration coefficients).

    Raises RuntimeError if both channels are dead.
    """
    pnum = PRIMARY[base]       # 1 or 2
    snum = 3 - pnum            # the other
    sign = CHANNEL_SIGN[base]

    p_path = os.path.join(shot_dir, f"{base}{pnum}.txt")
    s_path = os.path.join(shot_dir, f"{base}{snum}.txt")

    p_arr = _rd(p_path)["v"].to_numpy()

    # Secondary file may be absent on older shots; treat as dead in that case.
    try:
        s_arr = _rd(s_path)["v"].to_numpy()
    except FileNotFoundError:
        s_arr = np.zeros_like(p_arr)

    n = min(len(p_arr), len(s_arr))
    p_arr, s_arr = p_arr[:n], s_arr[:n]

    try:
        win = _discharge_window(shot_dir, discharge_current)[:n]
        if win.sum() < 5:
            win = np.ones(n, dtype=bool)
    except Exception:
        win = np.ones(n, dtype=bool)

    p_win, s_win = p_arr[win], s_arr[win]
    sp = float(np.std(p_win)); ss = float(np.std(s_win))
    peer = max(sp, ss)

    dead_p = _is_dead(p_win, peer)
    dead_s = _is_dead(s_win, peer)

    if dead_p and dead_s:
        raise RuntimeError(
            f"[current_channels] {base}: BOTH channels dead "
            f"(std{pnum}={sp:.1f} A, std{snum}={ss:.1f} A < floors "
            f"abs={DEAD_STD_FLOOR}, rel={DEAD_REL_FLOOR}). "
            f"Cannot produce a reliable {base} signal.")

    if dead_p:
        result = sign * s_arr
        prov = f"{base}{pnum} dead -> using sign*{base}{snum}"
        print(f"[current_channels] {prov} "
              f"(std{pnum}={sp:.1f} A, std{snum}={ss:.1f} A)")
        return result, prov

    if dead_s:
        prov = f"{base}{snum} dead -> using {base}{pnum}"
        print(f"[current_channels] {prov} "
              f"(std{pnum}={sp:.1f} A, std{snum}={ss:.1f} A)")
        return p_arr, prov

    # Both healthy -- check agreement.
    diff = np.abs(p_win - sign * s_win)
    peak = max(np.percentile(np.abs(p_win), 95), 1.0)
    disagreement = float(np.median(diff) / peak)

    if disagreement > DISAGREE_MAX:
        print(
            f"[current_channels] WARNING: {base} channels both live but disagree "
            f"(median rel |{base}{pnum} - sign*{base}{snum}| = {disagreement:.2f} "
            f"> {DISAGREE_MAX}). Using primary {base}{pnum}. "
            f"Something may be unmodelled (check DAQ/calibration).")
        return p_arr, f"{base}{pnum} (both live, disagreement={disagreement:.2f})"

    if not average:
        return p_arr, f"{base}{pnum} (both healthy; not averaged per policy)"

    # Average in primary's convention: (primary + sign*secondary) / 2.
    result = 0.5 * (p_arr + sign * s_arr)
    prov = f"mean({base}{pnum}, sign*{base}{snum})"
    return result, prov


def resolve_all(shot_dir, discharge_current=2500.0):
    """Resolve all four channel pairs.

    Returns:
        signals    : {"IP": arr, "IT": arr, "IOH": arr, "IV": arr}
        provenances: {"IP": str, "IT": str, "IOH": str, "IV": str}
    """
    signals, provenances = {}, {}
    for base in ["IP", "IT", "IOH", "IV"]:
        avg = (base != "IP")
        signals[base], provenances[base] = resolve(
            shot_dir, base, discharge_current=discharge_current, average=avg)
    return signals, provenances
