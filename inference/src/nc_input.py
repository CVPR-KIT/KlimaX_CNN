"""Causal raw daily cumulative NC -> exact V45 H6 inference inputs.

No future frames or future truth are read. All cumulative frames since the
daily reset are required because Method4C carries negative-difference debt.
"""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from preprocess_v27 import ResetSafeMethod4C, VERSION
from raw_contract import QC_VERSION, read_raw_strict, mask_interval

Y0, X0, SIZE = 1046, 648, 800


def clock_features(timestamp):
    sec = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    year = timestamp.year
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    dayphase = (timestamp.timetuple().tm_yday - 1 + sec / 86400.) / (366 if leap else 365)
    hourphase = sec / 86400.
    return np.asarray([math.sin(2 * math.pi * dayphase), math.cos(2 * math.pi * dayphase),
                       math.sin(2 * math.pi * hourphase), math.cos(2 * math.pi * hourphase)], np.float32)


def _source_path(root, timestamp):
    name = timestamp.strftime('%Y%m%d%H%M.nc')
    candidates = [root / timestamp.strftime('%Y%m%d') / name, root / name,
                  root / 'nc' / timestamp.strftime('%Y%m%d') / name, root / 'nc' / name]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        raise FileNotFoundError('Missing daily-reset input: %s (searched %s)' % (name, root))
    if len(existing) > 1:
        # A chosen input directory must identify one source, not silently select
        # between different duplicate datasets.
        raise ValueError('Ambiguous raw NC location: ' + ', '.join(map(str, existing)))
    return existing[0]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_raw(path):
    import netCDF4
    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        if 'data' not in nc.variables:
            raise ValueError('Raw NC must contain data: ' + str(path))
        variable = nc['data']
        if len(variable.shape) != 2 or variable.shape[0] < Y0 + SIZE or variable.shape[1] < X0 + SIZE:
            raise ValueError('Expected full native 500m cumulative NC grid, got %s: %s' % (variable.shape, path))
        raw = variable[Y0:Y0 + SIZE, X0:X0 + SIZE].astype(np.float32)
    return np.where((raw >= 0) & (raw < 9999), raw / 10, np.nan)


def _convert(rain):
    rain = np.where(rain <= 500, rain, np.nan)
    fine = np.nan_to_num(rain[350:450, 350:450], nan=0.)
    blocks = rain.reshape(200, 4, 200, 4)
    count = np.isfinite(blocks).sum((1, 3))
    context = (np.nan_to_num(blocks, nan=0.).sum((1, 3)) / np.maximum(count, 1)).astype(np.float16)
    valid = np.round(count * 255 / 16).astype(np.uint8)
    return fine.astype(np.float32), context, valid


def _save_nc(path, timestamp, fine, context, valid, version=VERSION):
    import netCDF4
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
    with netCDF4.Dataset(temporary, 'w', format='NETCDF4') as nc:
        nc.createDimension('y', 100)
        nc.createDimension('x', 100)
        nc.createDimension('context_y', 200)
        nc.createDimension('context_x', 200)
        data = nc.createVariable('data', 'f4', ('y', 'x'), zlib=True)
        data[:] = fine
        data.units = 'mm/10min'
        wide = nc.createVariable('context', 'f4', ('context_y', 'context_x'), zlib=True)
        wide[:] = context.astype(np.float32)
        wide.units = 'mm/10min'
        coverage = nc.createVariable('valid', 'u1', ('context_y', 'context_x'), zlib=True)
        coverage[:] = valid
        coverage.units = 'valid native pixel fraction times 255'
        nc.timestamp = timestamp.isoformat()
        nc.preprocessing = version
        nc.orientation = 'raw NC row increases northward; no transpose or flip'
    os.replace(temporary, path)


def prepare_input(raw_root, first_forecast, converted_nc_dir, converted_npz_dir, progress=None, strict_qc=False):
    """Return channel-first numpy fine/context/valid H6 and clock19 arrays.

    ``first_forecast`` is the first OUTPUT timestamp, not the observation anchor.
    Outputs are loaded back from the saved NPZ before inference.
    """
    if not isinstance(first_forecast, dt.datetime):
        raise TypeError('first_forecast must be datetime.datetime')
    if first_forecast.minute % 10 or first_forecast.second or first_forecast.microsecond:
        raise ValueError('Forecast timestamp must be on a 10-minute boundary')
    anchor = first_forecast - dt.timedelta(minutes=10)
    day = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
    reference = (anchor.hour * 60 + anchor.minute) // 10
    if reference < 6:
        raise ValueError('Six valid within-day inputs require anchor >=01:00; earliest forecast 01:10')
    root = Path(raw_root)
    # Resolve the complete required prefix BEFORE writing any converted output.
    sources = [(day + dt.timedelta(minutes=10 * i), _source_path(root, day + dt.timedelta(minutes=10 * i)))
               for i in range(1, reference + 1)]
    processor = ResetSafeMethod4C()
    previous_finite = None
    version = QC_VERSION if strict_qc else VERSION
    fine_frames, context_frames, valid_frames, input_times = [], [], [], []
    provenance = []
    input_qc = []
    for index, (timestamp, path) in enumerate(sources, 1):
        source_hash = _sha256(path)
        if strict_qc:
            cumulative, finite = read_raw_strict(path, timestamp)
        else:
            cumulative = _read_raw(path)
        rain = processor.update(cumulative, timestamp)
        if strict_qc:
            rain = mask_interval(rain, finite, previous_finite)
            previous_finite = finite
        provenance.append(dict(timestamp=timestamp.isoformat(), filename=path.name, sha256=source_hash))
        if index >= reference - 5:
            raw_fine = np.where(rain <= 500, rain, np.nan)[350:450, 350:450]
            input_qc.append(dict(timestamp=timestamp.isoformat(),
                                 invalid_pixels=int((~np.isfinite(raw_fine)).sum())))
            fine, context, valid = _convert(rain)
            fine_frames.append(fine)
            context_frames.append(context)
            valid_frames.append(valid)
            input_times.append(timestamp)
            _save_nc(Path(converted_nc_dir) / timestamp.strftime('%Y%m%d%H%M.nc'), timestamp, fine, context, valid, version=version)
        if progress:
            progress(index, reference, timestamp)
    metadata = dict(version=version, raw_unit='0.1mm cumulative since daily reset', unit='mm/10min',
                    anchor=anchor.isoformat(), first_forecast=first_forecast.isoformat(),
                    frame0='unavailable; not a valid model input', consumed_files=provenance,
                    window_native=[Y0, X0, SIZE, SIZE], fine_native=[350, 450, 350, 450],
                    context_pool=[4, 4], context_storage='float16', valid_storage='uint8 fraction*255')
    output = Path(converted_npz_dir) / ('%s_through%s.npz' % (anchor.strftime('%Y%m%d'), anchor.strftime('%H%M')))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.%s.tmp.npz' % os.getpid())
    clock = np.stack([clock_features(anchor + dt.timedelta(minutes=10 * i)) for i in range(19)])
    np.savez_compressed(temporary, rain=np.stack(fine_frames), context=np.stack(context_frames),
                        valid=np.stack(valid_frames), clock=clock,
                        input_timestamps=np.asarray([t.isoformat() for t in input_times]),
                        metadata_json=json.dumps(metadata, ensure_ascii=False))
    os.replace(temporary, output)
    with np.load(output, allow_pickle=False) as saved:
        fine = saved['rain'].copy()[:, None]
        context = saved['context'].astype(np.float32)[:, None]
        valid = saved['valid'].astype(np.float32)[:, None] / 255.
        loaded_clock = saved['clock'].copy()
        loaded_metadata = json.loads(str(saved['metadata_json']))
    assert fine.shape == (6, 1, 100, 100) and context.shape == valid.shape == (6, 1, 200, 200)
    if not all(np.isfinite(a).all() for a in (fine, context, valid, loaded_clock)):
        raise ValueError('Non-finite inference input after conversion')
    return dict(fine=fine, context=context, valid=valid, clock=loaded_clock, anchor=anchor,
                npz_path=output, provenance=loaded_metadata, input_timestamps=input_times,
                input_qc=input_qc)
