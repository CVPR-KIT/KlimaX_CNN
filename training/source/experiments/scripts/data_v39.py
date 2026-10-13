"""V39 dataset: raw fine (500 m / 50 km) + raw wide/context (2 km / 400 km) windows
from the corrected reset_v2 corpus, with per-frame coverage, known absolute-time clock,
and FULL-resolution 18-lead targets (INTERFACE.md).

Contract returned per window (dict of tensors, batch dim added by the default collate):
  fine       [H,1,100,100]   raw mm/10min, 500 m, observed history (NOT log1p'd here)
  context    [H,1,200,200]   raw mm/10min, 2 km, observed history, SAME H timestamps
  valid      [H,1,200,200]   coverage 0..1, PER FRAME (not averaged into one channel)
  clock      [19,4]          known cyclical time features for the anchor frame r and the
                              18 future lead times r+1..r+18; carries no rain information
  truth      [18,1,100,100]  raw mm/10min fine targets, r+1..r+18 (future, never an input)
  wide_truth [18,1,200,200]  raw mm/10min wide targets at native 2 km (NOT pooled to any
                              coarser "memory grid" -- see module note below)
  wide_valid [18,1,200,200]  coverage 0..1 for wide_truth, per lead, native resolution
  fine_valid [18,1,100,100]  OPTIONAL, conservative projection (see below); ONLY masks
                              which fine target pixels may be scored, never fed to a model

Why not the old ResetV2Windows pooling
---------------------------------------
climax/regional_forecast/reset_v2_datamodule.py (v36-v38) pools the 18-lead context
target to a 50x50 "memory grid" (`avg_pool2d(..., pool)`) because its decoder only had
25x25 memory tokens to supervise, and averages the six input coverage frames down to one
channel. V39's contract wants the native 200x200 wide target/valid for all 18 leads, with
per-cell partial coverage preserved as a real mask (not blurred away by pooling), so
BOTH of those shortcuts are dropped here. The day-file reading/caching pattern is reused
because it is the only way to avoid re-decompressing the same NPZ per window (see
`_DayCache` below), not because the pooled target shape is reused.

Source NPZ per day (experiments/scripts/preprocess_v27.py, VERSION
'rn_day_method4c_reset_v2_400km', read from --root/<split>/<YYYYMMDD>.npz):
  rain    [144,100,100] float32  raw mm/10min, 500 m, centred 50 km crop
  context [144,200,200] float16  raw mm/10min, 2 km, the SAME underlying 800x800
                                  native-pixel (500 m) crop, averaged in 4x4 blocks
  valid   [144,200,200] uint8    coverage fraction (0..255 == 0..1) of that SAME crop

Frame 0 of every day is the daily-reset instant (a structural zero, not an observed dry
field) and is excluded by construction: the earliest INPUT frame is r-history+1, and
`assert_no_frame_zero` enforces r >= history so frame 0 is never read as an input, and
targets start at r+1 so frame 0 can never be a target either. Inputs only ever read
frames <= r; targets only ever read frames > r -- there is no future leak by construction,
and the 18-lead FUTURE clock entries carry only the (known in advance) time-of-day/
day-of-year phase, never rain.

Native fine coverage does not exist (see preprocess_v27.py: `valid` covers the shared
800x800 crop at 2 km, `rain` at 500 m has no coverage array of its own). `fine_valid`
below is therefore a CONSERVATIVE NEAREST-CELL projection of the 2 km context validity
onto the 500 m grid: the fine crop is `rain[..., 350:450, 350:450]` of the same 800x800
crop that `context` block-averages in 4x4 native-pixel groups (`FINE_NATIVE_OFFSET`,
`CONTEXT_BLOCK` below), so fine pixel i's nearest context cell is
`(FINE_NATIVE_OFFSET + i) // CONTEXT_BLOCK`. A fine cell is marked valid ONLY where its
containing context cell has FULL coverage; partial coverage is treated as unknown. This
is explicitly NOT a recovered native measurement -- it is a conservative label carried
through with that limitation, never claimed as ground truth.
"""
import json
import math
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

HISTORY = 6
STEPS = 18
FINE = 100
CONTEXT = 200
FRAMES_PER_DAY = 144
FINE_RES_KM = 0.5
CONTEXT_RES_KM = 2.0
HRS_PER_STEP = 1 / 6.0

# preprocess_v27.py `build_day`: fine is `rain[350:450, 350:450]` of a shared 800x800
# native-pixel (500 m) crop; context is that SAME 800x800 crop averaged in 4x4-pixel
# blocks to 200x200 (2 km). Both constants are fixed by that script, not measured here.
FINE_NATIVE_OFFSET = 350
CONTEXT_BLOCK = 4


def validate_manifest_contract(audit, history, steps):
    """Legacy six-frame manifests remain supported; longer inputs require new QC."""
    if audit.get('audit_pass') is not True:
        raise ValueError('a passed dataset audit is required')
    protocol = audit.get('protocol', {})
    expected_history = int(protocol.get('history', HISTORY))
    expected_steps = int(protocol.get('horizon', STEPS))
    if (int(history), int(steps)) != (expected_history, expected_steps):
        raise ValueError(f'manifest history/horizon {expected_history}/{expected_steps} '
                         f'does not match requested {history}/{steps}; '
                         'derive an audited manifest for the longer input first')
    if int(history) == 18:
        derivation = audit.get('derivation', {})
        if (protocol.get('manifest_derivation') != 'v39_history18_parent_window_union_v1'
                or derivation.get('parent_audit_pass') is not True
                or not protocol.get('parent_manifest_sha256')):
            raise ValueError('history18 requires parent-QC union derivation evidence')


def _fine_to_context_index(n=FINE, offset=FINE_NATIVE_OFFSET, block=CONTEXT_BLOCK, context=CONTEXT):
    idx = (offset + np.arange(n)) // block
    return np.clip(idx, 0, context - 1)


_FINE_ROW_IN_CONTEXT = _fine_to_context_index()
_FINE_COL_IN_CONTEXT = _fine_to_context_index()


def project_fine_valid(context_valid_frame):
    """Conservative nearest-cell projection of one [200,200] context validity frame
    (0..1) onto the [100,100] fine grid. A fine cell is 1.0 only if its containing 2 km
    context cell is FULLY covered (>= 1 - 1e-6); any partial coverage becomes 0
    (unknown), which is the conservative direction. NOT a native fine measurement --
    see the module docstring."""
    cell = context_valid_frame[np.ix_(_FINE_ROW_IN_CONTEXT, _FINE_COL_IN_CONTEXT)]
    return (cell >= 1.0 - 1e-6).astype(np.float32)


def assert_no_frame_zero(reference_indices, history=HISTORY):
    """Frame 0 is the daily-reset instant, a structural zero the validity mask does
    NOT flag as invalid, not an observed dry field. Inputs read r-history+1 .. r, so the
    smallest reference index must be >= history or frame 0 would be fed as an
    observation."""
    smallest = int(np.min(reference_indices))
    if smallest - history + 1 < 1:
        raise ValueError(
            f'reference index {smallest} would read frame {smallest - history + 1}; '
            'frame 0 is the unavailable daily-reset frame, not an observed dry field')


def timestamp_clock_features(date, step_idx, hrs_each_step=HRS_PER_STEP):
    """Known absolute-time cyclical features [sin(doy),cos(doy),sin(hod),cos(hod)] for
    one frame index of one day. Matches the convention in
    `climax.pretrain.dataset._timestamp_features_from_step` (same trig/definitions) so
    clock semantics stay identical across model generations; reimplemented locally
    (pure stdlib/numpy) so this dataset does not import the ClimaX architecture tree."""
    base = datetime.strptime(date, '%Y%m%d')
    ts = base + timedelta(hours=float(step_idx) * float(hrs_each_step))
    doy = ts.timetuple().tm_yday - 1
    sec = ts.hour * 3600 + ts.minute * 60 + ts.second
    leap = ts.year % 4 == 0 and (ts.year % 100 != 0 or ts.year % 400 == 0)
    days_in_year = 366 if leap else 365
    doy_phase = (doy + sec / 86400.0) / float(days_in_year)
    hod_phase = sec / 86400.0
    return np.asarray([
        math.sin(2.0 * math.pi * doy_phase), math.cos(2.0 * math.pi * doy_phase),
        math.sin(2.0 * math.pi * hod_phase), math.cos(2.0 * math.pi * hod_phase),
    ], dtype=np.float32)


class _DayCache:
    """Bounded LRU cache of decompressed (rain, context, valid) triplets, one entry per
    day-file. NPZ decompression is the slow part of random per-window access; this cache
    is only effective when consecutive `get()` calls tend to hit the same day, which is
    what `DayBlockShuffleSampler` below is for."""

    def __init__(self, limit=8):
        self.limit = int(limit)
        self._store = OrderedDict()

    def get(self, path):
        hit = self._store.get(path)
        if hit is not None:
            self._store.move_to_end(path)
            return hit
        with np.load(path) as z:
            rain = np.asarray(z['rain'], dtype=np.float32)
            context = np.asarray(z['context'], dtype=np.float32)
            valid = np.asarray(z['valid'], dtype=np.float32) / 255.0
        if rain.shape != (FRAMES_PER_DAY, FINE, FINE):
            raise ValueError(f'{path}: unexpected rain shape {rain.shape}')
        if context.shape != (FRAMES_PER_DAY, CONTEXT, CONTEXT) or valid.shape != context.shape:
            raise ValueError(f'{path}: unexpected context/valid shape {context.shape}/{valid.shape}')
        hit = (rain, context, valid)
        self._store[path] = hit
        if len(self._store) > self.limit:
            self._store.popitem(last=False)
        return hit


class ResetV2WindowsV39(Dataset):
    """One item per selected (day, reference index) window; map-style so a shuffling
    sampler visits every selected window exactly once per epoch."""

    def __init__(self, root, manifest, split, cache_days=8, history=HISTORY, steps=STEPS,
                 hrs_each_step=HRS_PER_STEP, fixed_eval=False, return_meta=False):
        self.root = Path(root)
        self.split = str(split)
        self.history, self.steps = int(history), int(steps)
        self.hrs_each_step = float(hrs_each_step)
        self.return_meta = bool(return_meta)
        with open(manifest) as fh:
            audit = json.load(fh)
        validate_manifest_contract(audit, self.history, self.steps)
        key = 'eval_indices' if fixed_eval else 'train_indices'
        self.refs = []
        for row in audit['days']:
            if row.get('split') != self.split:
                continue
            path = self.root / self.split / f"{row['date']}.npz"
            if not path.exists():
                continue
            for r in row.get(key, []):
                self.refs.append((str(path), row['date'], int(r)))
        if not self.refs:
            raise ValueError(f'no {key} windows for split {self.split!r} under {root}')
        assert_no_frame_zero([r for _, _, r in self.refs], self.history)
        max_r = max(r for _, _, r in self.refs)
        if max_r + self.steps > FRAMES_PER_DAY - 1:
            raise ValueError(f'reference {max_r} + {self.steps} leads overflows a {FRAMES_PER_DAY}-frame day')
        self.days = [d for _, d, _ in self.refs]  # parallel to self.refs; locality key
        self._cache = _DayCache(cache_days)

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        path, date, r = self.refs[i]
        rain, context, valid = self._cache.get(path)
        lo, hi = r - self.history + 1, r + 1
        fine = torch.from_numpy(np.clip(rain[lo:hi], 0, None)).unsqueeze(1)          # H,1,100,100
        ctx = torch.from_numpy(np.clip(context[lo:hi], 0, None)).unsqueeze(1)        # H,1,200,200
        cov = torch.from_numpy(valid[lo:hi]).unsqueeze(1)                            # H,1,200,200
        clock = np.stack([timestamp_clock_features(date, r + j, self.hrs_each_step)
                          for j in range(self.steps + 1)], axis=0)                   # 19,4
        truth = torch.from_numpy(np.clip(rain[r + 1:r + 1 + self.steps], 0, None)).unsqueeze(1)
        wide_truth = torch.from_numpy(np.clip(context[r + 1:r + 1 + self.steps], 0, None)).unsqueeze(1)
        wide_valid = torch.from_numpy(valid[r + 1:r + 1 + self.steps]).unsqueeze(1)
        fine_valid = torch.from_numpy(np.stack(
            [project_fine_valid(valid[r + 1 + j]) for j in range(self.steps)], axis=0)).unsqueeze(1)
        item = dict(fine=fine, context=ctx, valid=cov, clock=torch.from_numpy(clock),
                   truth=truth, wide_truth=wide_truth, wide_valid=wide_valid, fine_valid=fine_valid)
        if self.return_meta:
            item['date'], item['ref'] = date, r
        return item


class DayBlockShuffleSampler(Sampler):
    """Shuffle days in cache-sized blocks, then mix all windows within each block.
    Every index occurs once per epoch. Eight active days reduce temporal batch
    correlation while retaining bounded decompression/cache costs with workers=0.
    """

    def __init__(self, days, seed, block_days=8):
        self.block_days = int(block_days)
        self.days = list(days)
        self.seed = int(seed)
        self.epoch = 0
        by_day = OrderedDict()
        for idx, d in enumerate(self.days):
            by_day.setdefault(d, []).append(idx)
        self._by_day = by_day
        self._day_keys = list(by_day.keys())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.days)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(len(self._day_keys))
        result = []
        for start in range(0, len(order), self.block_days):
            idxs = [i for k in order[start:start + self.block_days]
                    for i in self._by_day[self._day_keys[k]]]
            rng.shuffle(idxs)
            result.extend(idxs)
        return iter(result)


class DataModuleV39:
    """Minimal datamodule: full passes over the fixed v34 manifest windows, no
    `limit_train_batches` slicing. Train/val date disjointness is verified explicitly
    here (not just assumed from the manifest's own split field)."""

    def __init__(self, root, manifest, batch_size=16, num_workers=0, seed=3901,
                 cache_days=8, history=HISTORY, steps=STEPS, pin_memory=None):
        self.root, self.manifest = str(root), str(manifest)
        self.batch_size, self.num_workers, self.seed = int(batch_size), int(num_workers), int(seed)
        self.pin_memory = bool(pin_memory) if pin_memory is not None else torch.cuda.is_available()
        self.train = ResetV2WindowsV39(root, manifest, 'train', cache_days, history, steps, fixed_eval=False)
        self.val = ResetV2WindowsV39(root, manifest, 'val', cache_days, history, steps, fixed_eval=True)
        overlap = set(self.train.days) & set(self.val.days)
        if overlap:
            raise ValueError(f'train/val date overlap ({len(overlap)} dates): {sorted(overlap)[:5]}')
        self.sampler = DayBlockShuffleSampler(self.train.days, self.seed, block_days=cache_days)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, sampler=self.sampler, drop_last=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.num_workers > 0)

    def summary(self):
        return dict(train_windows=len(self.train), val_windows=len(self.val),
                    history=self.train.history, steps=self.train.steps,
                    train_days=len(set(self.train.days)), val_days=len(set(self.val.days)),
                    batch_size=self.batch_size, updates_per_epoch=len(self.train) // self.batch_size,
                    root=self.root, manifest=self.manifest, sampler_seed=self.seed, sampler_block_days=self.sampler.block_days)
