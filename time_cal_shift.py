"""Non-invasive wall-clock timing of cal_shift inside the real TFM pipeline.

Times every cal_shift call during TFM_main on a real shot, without modifying
any repo file: the function is wrapped at import time and re-bound in TFM's
namespace (TFM.py imported it with `from .plasma_shift import cal_shift`,
so the name to patch lives in the TFM module, not in plasma_shift).

Run it either way:
  - Any IDE / plain `python time_cal_shift.py`: edit the CONFIG
    constants below to change shot / probe set / n_runs. Working directory
    does not matter (the script chdirs to its own repo).
  - Command line (optional overrides): python3 time_cal_shift.py [shot_dir]
    [probe_set] [n_runs], e.g. python3 time_cal_shift.py data/1641 "1 4 7 10" 10

n_runs repeats the full pipeline and pools timings. Medians stabilize by
~1e3 samples; p99 needs >=1e4; p99.9 >=1e5; max never converges.
"""
import sys, os, time, types, functools

# --- CONFIG: edit these when running the file with no arguments --------------
SHOT_DIR  = "data/1641"     # shot folder, relative to the repo root
PROBE_SET = "1 4 7 10"      # probe numbers, space-separated, one string
N_RUNS    = 10               # pooled repetitions. samples ~= discharge_len x N_RUNS.
                            # p99 wants >=1e4 total, p99.9 >=1e5. A big shot (~25k
                            # timesteps) already gives 1e5 at N_RUNS=4; do NOT set
                            # this to thousands.

# --- make repo importable from any launch directory (IDE run buttons often
# --- start elsewhere); stub tqdm if absent (progress bars only) --------------
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)              # so relative paths like data/1641 always resolve
# --- suppress the repo's OWN internal tqdm bars (per-line/per-probe loops) ----
# TFM_main calls tqdm internally; those are the duplicate bars. We replace the
# tqdm module with a no-op stub BEFORE importing TFM, so its bars never appear.
# Our single pooled bar uses tqdm imported separately below (real one if present).
try:
    import tqdm as _real_tqdm            # keep a handle to the real one for OUR bar
    HAVE_TQDM = True
except ModuleNotFoundError:
    _real_tqdm = None
    HAVE_TQDM = False

_stub = types.ModuleType("tqdm")         # no-op stub the repo will import
_stub.tqdm = lambda it=None, **kw: it if it is not None else (lambda x: x)
sys.modules["tqdm"] = _stub              # repo's `from tqdm import tqdm` -> no-op

import numpy as np
import methods_script.toroidal_filament.TFM as TFM

# --- config (CLI overrides the CONFIG constants when given) ------------------
shot = sys.argv[1] if len(sys.argv) > 1 else SHOT_DIR
probe_set = [sys.argv[2]] if len(sys.argv) > 2 else [PROBE_SET]
n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else N_RUNS
print(f"config: shot={shot}  probe_set={probe_set[0]!r}  n_runs={n_runs}")

# --- pre-count timesteps for a determinate total bar -------------------------
# one cal_shift call per discharge timestep per run; count the discharge window
# once using the repo's own logic so the bar total is exact.
def count_timesteps(shot_dir):
    from methods_script.toroidal_filament.process_probe_data import read_txt, discharge_duration
    df = read_txt(f"{shot_dir}/IP1.txt", ["t", "v"])
    t, ip = df["t"].to_numpy(), df["v"].to_numpy()
    t0, t1 = discharge_duration(t, ip)
    return int(((t >= t0) & (t <= t1)).sum())

try:
    total_calls = count_timesteps(shot) * n_runs
except Exception as e:                     # fall back to indeterminate if it fails
    print(f"(could not pre-count timesteps: {e}; using indeterminate bar)")
    total_calls = None

# --- one pooled progress bar, advanced from inside the wrapper ---------------
if HAVE_TQDM:
    bar = _real_tqdm.tqdm(total=total_calls, unit="call", desc="cal_shift", smoothing=0.05)
    def _tick():
        bar.update(1)
        n = len(call_times_ns)
        if n % 500 == 0:                   # recompute stats a few times/sec, not every call
            a = np.asarray(call_times_ns) / 1e3
            bar.set_postfix(mean=f"{a.mean():.1f}us", p99=f"{np.percentile(a,99):.1f}us")
    def _close():
        bar.close()
else:                                      # no tqdm: lightweight in-place counter
    _t_start = time.perf_counter()
    def _tick():
        n = len(call_times_ns)
        if n % 500 == 0:                   # throttle terminal writes + stats
            el = time.perf_counter() - _t_start
            tot = f"/{total_calls}" if total_calls else ""
            a = np.asarray(call_times_ns) / 1e3
            sys.stdout.write(f"\rcal_shift {n}{tot}  {el:5.1f}s  "
                             f"mean {a.mean():.1f}us  p99 {np.percentile(a,99):.1f}us")
            sys.stdout.flush()
    def _close():
        sys.stdout.write("\n")

# --- wrap and re-bind --------------------------------------------------------
call_times_ns = []
_orig = TFM.cal_shift                      # the reference TFM_main actually calls

@functools.wraps(_orig)
def _timed(*args, **kwargs):
    t0 = time.perf_counter_ns()
    out = _orig(*args, **kwargs)
    call_times_ns.append(time.perf_counter_ns() - t0)
    _tick()
    return out

TFM.cal_shift = _timed                     # patch TFM's namespace, not plasma_shift's

# --- run the real pipeline, pooled over n_runs -------------------------------
cold_idx = []                              # index of first call of each run
for r in range(n_runs):
    cold_idx.append(len(call_times_ns))
    TFM.TFM_main(shot, probe_set)
_close()
print(f"pooled {len(call_times_ns)} samples over {n_runs} run(s)")

# --- report distribution (µs); worst case matters for real-time --------------
ts = np.asarray(call_times_ns, dtype=float) / 1e3
warm = np.delete(ts, cold_idx)             # drop each run's cold-start call
cold = ts[cold_idx]
print(f"\ncal_shift wall-clock: {len(warm)} warm calls "
      f"({n_runs} cold starts excluded, median cold {np.median(cold):.1f} µs)")
print(f"  min {warm.min():8.2f}  median {np.median(warm):8.2f}  "
      f"mean {warm.mean():8.2f}  p90 {np.percentile(warm, 90):8.2f}  "
      f"p99 {np.percentile(warm, 99):8.2f}", end="")
if len(warm) >= 1e5:
    print(f"  p99.9 {np.percentile(warm, 99.9):8.2f}", end="")
print(f"  max {warm.max():10.2f}  µs")
if len(warm) < 1e4:
    print("  note: p99 unreliable below ~1e4 samples; increase n_runs")
budget = 20.0
print(f"  fraction of calls over {budget:.0f} µs (ADC period): "
      f"{(warm > budget).mean() * 100:.2f}%")
