"""v25 context data: full-grid KMA rn_day NCs -> a 600 km x 600 km window at 1 km, one npz per day.

The window is centred on the model's 100 x 100 Seoul crop (raw grid rows 1396..1495, cols 998..1097), so any
smaller centred window (256 / 400 / 512 km) can be cut from it at load time. Orientation is the raw NC's
(row index grows northward), the same as the 100 x 100 training npz files.

Conversion is the provider's Method4C (causal cumulative -> 10-min interval, weak-rain removal, 3x3
mass-conserving smoothing), the same one that built data_10m, applied on the 500 m window and then averaged
2 x 2 over valid pixels. Output per day, <out>/<split dir>/<YYYYMMDD>.npz:
  rain   uint16 [144, 600, 600]   mm / 10 min x 100 (0.01 mm steps), frame i = 00:00 + 10 min * i
  valid  uint8  [600, 600]        share of valid 500 m sub-pixels (0..255), averaged over the day

usage (server7):
  python3 preprocess_context.py --profile debug_data --procs 16
  python3 preprocess_context.py --verify 20210723      # centre crop must reproduce data_10m for that day
"""
import argparse
import datetime as dt
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RN_DIR = "/datasets/rn_day"
MANIFEST = "/workspace/project/experiments/model_arch_test/configs/data_manifest_fixed.yaml"
OUT_DIR = "/workspace/project/data/context_600km_1km"

CROP_Y0, CROP_X0, CROP = 1396, 998, 100
CENTER_Y, CENTER_X = CROP_Y0 + CROP // 2, CROP_X0 + CROP // 2
HALF_PX = 600                                  # 500 m px each side -> 600 km window
WIN_Y0, WIN_X0 = CENTER_Y - HALF_PX, CENTER_X - HALF_PX
STEPS = 144
MISSING_LIMIT, SCALE = 9999.0, 10.0
RAIN_UNIT = 100.0                              # stored value = mm * 100

# ---- Method4C: verbatim logic of exefolder3/src/nc_preprocess.py (the conversion that built data_10m) ----
SMOOTH_KERNEL_3X3 = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float64) / 16.0


try:
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover
    _ndimage = None


def weighted_filter_3x3_mass_conserving(field):
    values_in = np.asarray(field, dtype=np.float64)
    valid = np.isfinite(values_in)
    values = np.where(valid, values_in, 0.0)
    if _ndimage is not None:
        # same zero-padded 3x3 weighted sums, in C (4x faster on the 1200x1200 window; identical output)
        numerator = _ndimage.correlate(values, SMOOTH_KERNEL_3X3, mode="constant", cval=0.0)
        denominator = _ndimage.correlate(valid.astype(float), SMOOTH_KERNEL_3X3, mode="constant", cval=0.0)
        smoothed = np.where(denominator > 0.0, numerator / np.maximum(denominator, 1e-12), np.nan)
        smoothed = np.where(valid, smoothed, np.nan)
        before = float(np.nansum(values))
        after = float(np.nansum(smoothed))
        if after > 0.0:
            smoothed *= before / after
        return smoothed.astype(np.float32)
    padded_values = np.pad(values, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    padded_valid = np.pad(valid.astype(float), ((1, 1), (1, 1)), mode="constant", constant_values=0)
    height, width = values_in.shape
    numerator = np.zeros_like(values)
    denominator = np.zeros_like(values)
    for dy in range(3):
        for dx in range(3):
            weight = SMOOTH_KERNEL_3X3[dy, dx]
            numerator += weight * padded_values[dy:dy + height, dx:dx + width]
            denominator += weight * padded_valid[dy:dy + height, dx:dx + width]
    smoothed = np.where(denominator > 0.0, numerator / np.maximum(denominator, 1e-12), np.nan)
    smoothed = np.where(valid, smoothed, np.nan)
    before = float(np.nansum(values))
    after = float(np.nansum(smoothed))
    if after > 0.0:
        smoothed *= before / after
    return smoothed.astype(np.float32)


class Method4CCausal:
    def __init__(self, min_rain_10m_mm=0.1):
        self.min_rain_10m_mm = float(min_rain_10m_mm)
        self.previous = None
        self.carried_decrease = None

    def update(self, cumulative, daily_start=False):
        current = np.asarray(cumulative, dtype=np.float64)
        valid_current = np.isfinite(current) & (current >= 0.0)
        current = np.where(valid_current, current, np.nan)
        if daily_start or self.previous is None:
            self.previous = current.copy()
            self.carried_decrease = np.zeros_like(current, dtype=np.float64)
            rain = np.where(valid_current, 0.0, np.nan)
            return weighted_filter_3x3_mass_conserving(rain)
        valid = valid_current & np.isfinite(self.previous)
        step = np.where(valid, current - self.previous, np.nan)
        interval = np.full_like(current, np.nan)
        negative = valid & (step < 0.0)
        positive = valid & (step > 0.0)
        unchanged = valid & (step == 0.0)
        self.carried_decrease[negative] += -step[negative]
        interval[negative] = 0.0
        offset = np.zeros_like(current)
        offset[positive] = np.minimum(step[positive], self.carried_decrease[positive])
        interval[positive] = np.maximum(step[positive] - offset[positive], 0.0)
        self.carried_decrease[positive] -= offset[positive]
        interval[unchanged] = 0.0
        reinitialized = valid_current & ~np.isfinite(self.previous)
        interval[reinitialized] = np.nan
        self.carried_decrease[reinitialized] = 0.0
        self.previous[valid_current] = current[valid_current]
        weak = np.isfinite(interval) & (interval > 0.0) & (interval <= self.min_rain_10m_mm + 1e-7)
        interval[weak] = 0.0
        return weighted_filter_3x3_mass_conserving(interval)
# ---------------------------------------------------------------------------------------------------------


def nc_path(nc_dir, t):
    return os.path.join(nc_dir, f"{t:%Y%m%d%H%M}.nc")


def read_window(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        a = ds["data"][WIN_Y0:WIN_Y0 + 2 * HALF_PX, WIN_X0:WIN_X0 + 2 * HALF_PX].astype(np.float32)
    return np.where((a >= 0) & (a < MISSING_LIMIT), a / SCALE, np.nan)


def pool2_nanmean(field):
    """[2H, 2W] with NaN -> ([H, W] mean of the finite sub-pixels (0 where none), [H, W] finite share)."""
    h, w = field.shape[0] // 2, field.shape[1] // 2
    blocks = field.reshape(h, 2, w, 2)
    ok = np.isfinite(blocks)
    n = ok.sum((1, 3))
    s = np.where(ok, blocks, 0.0).sum((1, 3))
    return np.where(n > 0, s / np.maximum(n, 1), 0.0), n / 4.0


def build_day(nc_dir, day, keep_fullres=False):
    """-> rain [144, 600, 600] float32 mm, valid [600, 600] float32 (0..1), optional full-res [144,1200,1200]."""
    proc = Method4CCausal(min_rain_10m_mm=0.1)
    rain = np.zeros((STEPS, HALF_PX, HALF_PX), np.float32)
    valid = np.zeros((HALF_PX, HALF_PX), np.float64)
    full = np.zeros((STEPS, 2 * HALF_PX, 2 * HALF_PX), np.float32) if keep_fullres else None
    for i in range(STEPS):
        t = day + dt.timedelta(minutes=10 * i)
        frame = proc.update(read_window(nc_path(nc_dir, t)), daily_start=(t.hour == 0 and t.minute == 0))
        frame = np.where(frame > 500.0, np.nan, frame)          # same outlier cap as the eval conversion
        if full is not None:
            full[i] = np.nan_to_num(frame, nan=0.0)
        r, v = pool2_nanmean(frame)
        rain[i] = r
        valid += v
    return rain, (valid / STEPS).astype(np.float32), full


def save_day(path, rain, valid):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp,
        rain=np.clip(np.round(rain * RAIN_UNIT), 0, 65535).astype(np.uint16),
        valid=np.clip(np.round(valid * 255), 0, 255).astype(np.uint8),
    )
    os.replace(tmp, path)


def _job(args):
    nc_dir, day_str, out_path = args
    if Path(out_path).exists():
        return day_str, "exists", 0.0
    t0 = time.time()
    day = dt.datetime.strptime(day_str, "%Y%m%d")
    missing = [i for i in range(STEPS) if not os.path.exists(nc_path(nc_dir, day + dt.timedelta(minutes=10 * i)))]
    if missing:
        return day_str, f"missing {len(missing)} files", 0.0
    try:
        rain, valid, _ = build_day(nc_dir, day)
        save_day(Path(out_path), rain, valid)
    except Exception as e:  # noqa: BLE001
        return day_str, f"error {e!r}", 0.0
    return day_str, "ok", time.time() - t0


def manifest_days(manifest_path, profile):
    import yaml
    m = yaml.safe_load(open(manifest_path))
    files = {int(r["id"]): r for r in m["files"]}
    out = []
    for key in ("train_ids", "val_ids", "test_ids"):
        for i in m["splits"][profile][key]:
            rec = files[int(i)]
            rel = Path(rec["path"])                      # e.g. train/19970515.npz
            out.append((rec["date"].replace("-", ""), rel.parent.name))
    return out


def verify(nc_dir, day_str, data_root):
    """The 100 x 100 centre of the 500 m window must equal data_10m away from the crop border
    (Method4C's 3x3 smoothing sees different neighbours at the crop edge)."""
    day = dt.datetime.strptime(day_str, "%Y%m%d")
    _, _, full = build_day(nc_dir, day, keep_fullres=True)
    c = HALF_PX - CROP // 2
    centre = full[:, c:c + CROP, c:c + CROP]
    hits = list(Path(data_root).glob(f"*/{day_str}.npz"))
    ref = np.load(hits[0])["rain"][:, 0] if np.load(hits[0])["rain"].ndim == 4 else np.load(hits[0])["rain"]
    inner = (slice(None), slice(1, -1), slice(1, -1))
    diff = np.abs(centre[inner] - ref[inner])
    print(f"{day_str}: data_10m {hits[0]}  max|diff| interior {diff.max():.2e}  "
          f"rain total ctx {centre[inner].sum():.1f} vs data_10m {ref[inner].sum():.1f}")
    return float(diff.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc-dir", default=RN_DIR)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--profile", default="debug_data")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N days (smoke test)")
    ap.add_argument("--verify", default="", help="YYYYMMDD: compare the window centre with data_10m and exit")
    ap.add_argument("--data-root", default="/workspace/project/data/data_10m")
    a = ap.parse_args()

    if a.verify:
        verify(a.nc_dir, a.verify, a.data_root)
        return

    days = manifest_days(a.manifest, a.profile)
    if a.limit:
        days = days[:a.limit]
    jobs = list(dict.fromkeys((a.nc_dir, d, str(Path(a.out) / split / f"{d}.npz")) for d, split in days))
    print(f"{len(jobs)} days ({a.profile}) -> {a.out}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    status = {}
    t0 = time.time()
    with Pool(a.procs) as pool:
        for n, (d, s, sec) in enumerate(pool.imap_unordered(_job, jobs), 1):
            status[d] = s
            if s not in ("ok", "exists"):
                print(f"  {d}: {s}", flush=True)
            if n % 100 == 0 or n == len(jobs):
                done = sum(v == "ok" for v in status.values())
                print(f"  {n}/{len(jobs)}  new {done}  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    summary = {k: sum(v == k for v in status.values()) for k in ("ok", "exists")}
    summary["not_done"] = sorted(d for d, v in status.items() if v not in ("ok", "exists"))
    json.dump({"profile": a.profile, "status": status}, open(Path(a.out) / f"status_{a.profile}.json", "w"), indent=0)
    print(f"done: {summary['ok']} new, {summary['exists']} already there, {len(summary['not_done'])} not done")


if __name__ == "__main__":
    main()
