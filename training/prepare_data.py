"""Audit raw KMA rn_day files and build a data-only V45 H6/L18 corpus.

Uses the inference Method4C implementation. Unknown pixels retain coverage masks;
zeros in stored rainfall at unknown pixels never establish valid observations.
Generated LCC coordinates are header-derived, not the official lat/lon product.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

INFERENCE = Path(__file__).resolve().parents[1] / 'inference' / 'src'
sys.path.insert(0, str(INFERENCE))
from preprocess_v27 import ResetSafeMethod4C
from raw_contract import GEOMETRY, validate_header, read_raw_strict, mask_interval, QC_VERSION as VERSION
from nc_input import Y0, X0, SIZE, _convert, _source_path




def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_dates(value):
    dates = [x.strip() for x in value.split(',') if x.strip()]
    if not dates or len(set(dates)) != len(dates):
        raise ValueError('Dates must be nonempty and unique')
    for date in dates:
        if datetime.strptime(date, '%Y%m%d').strftime('%Y%m%d') != date:
            raise ValueError('Dates must use YYYYMMDD')
    return sorted(dates)


def build_grid(geometry):
    """Inverse KMA spherical LCC (30/60 parallels, radius6371.00877km).

    Uses integer native pixel coordinates, without an invented half-cell shift.
    Official sfc_grid_latlon.nc may differ by subpixel offsets; those exact grids
    are not recoverable from the rn_day header and this output states its basis.
    """
    rad = np.pi / 180
    s1, s2 = 30*rad, 60*rad
    n = np.log(np.cos(s1)/np.cos(s2))/np.log(np.tan(np.pi/4+s2/2)/np.tan(np.pi/4+s1/2))
    factor = np.tan(np.pi/4+s1/2)**n*np.cos(s1)/n
    radius = 6371.00877/geometry['grid_size']
    rho0 = radius*factor/np.tan(np.pi/4+geometry['map_slat']*rad/2)**n
    xx, yy = np.meshgrid(np.arange(X0+350, X0+450)-geometry['map_sx'],
                         np.arange(Y0+350, Y0+450)-geometry['map_sy'])
    dy = rho0-yy
    rho = np.hypot(xx, dy)
    lat = (2*np.arctan((radius*factor/rho)**(1/n))-np.pi/2)/rad
    lon = geometry['map_slon']+np.arctan2(xx, dy)/(n*rad)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or not (np.diff(lat, axis=0)>0).all():
        raise ValueError('Generated coordinates failed orientation/finite checks')
    return lat.astype(np.float32), lon.astype(np.float32)


def audit_windows(rain, context, valid, native_valid, min_fine=.95, min_context=.50):
    """Accept windows only using predeclared actual coverage and finite QC."""
    if not 0 < min_fine <= 1 or not 0 < min_context <= 1:
        raise ValueError('Coverage thresholds must be in (0,1]')
    indices = (350+np.arange(100))//4
    conservative = valid[:, indices[:, None], indices[None, :]] == 255
    fine_fraction = np.minimum(native_valid.mean((1,2)), conservative.mean((1,2)))
    context_fraction = valid.astype(np.float64).mean((1,2))/255
    finite = np.isfinite(rain).all((1,2)) & np.isfinite(context).all((1,2))
    finite &= (rain>=0).all((1,2)) & (context>=0).all((1,2))
    accepted, excluded = [], []
    for ref in range(6, 126):
        window = slice(ref-5, ref+19)
        good = finite[window] & (fine_fraction[window]>=min_fine) & (context_fraction[window]>=min_context)
        if good.all():
            accepted.append(ref)
        else:
            excluded.append(dict(ref=ref, bad_frame_indices=(np.flatnonzero(~good)+ref-5).tolist()))
    return accepted, dict(excluded_windows=excluded,
                         fine_known_fraction=fine_fraction.tolist(),
                         context_known_fraction=context_fraction.tolist())


def build_day(date, sources):
    import netCDF4
    processor = ResetSafeMethod4C()
    rain = np.zeros((144,100,100), np.float32)
    context = np.zeros((144,200,200), np.float16)
    valid = np.zeros((144,200,200), np.uint8)
    native_valid = np.zeros((144,100,100), bool)
    provenance, previous_finite = [], None
    for frame, (timestamp, path) in enumerate(sources, 1):
        cumulative, finite = read_raw_strict(path, timestamp)
        converted = mask_interval(processor.update(cumulative, timestamp), finite, previous_finite)
        previous_finite = finite
        native_valid[frame] = np.isfinite(converted[350:450,350:450])
        rain[frame], context[frame], valid[frame] = _convert(converted)
        provenance.append(dict(timestamp=timestamp.isoformat(), filename=path.name,
                               sha256=sha256(path), invalid_native_count=int((~finite).sum()),
                               invalid_interval_count=int((~np.isfinite(converted)).sum())))
    return rain, context, valid, native_valid, provenance


def prepare(raw_dir, out, train_dates, val_dates, min_fine=.95, min_context=.50, grid_source_dir=None):
    import netCDF4
    raw_dir, out = Path(raw_dir).resolve(), Path(out).resolve()
    train_dates, val_dates = parse_dates(train_dates), parse_dates(val_dates)
    if not 0 < min_fine <= 1 or not 0 < min_context <= 1:
        raise ValueError('Coverage thresholds must be in (0,1]')
    if set(train_dates)&set(val_dates):
        raise ValueError('Training and validation dates must be disjoint')
    if out.exists():
        raise FileExistsError('Refusing to overwrite existing output directory: '+str(out))
    planned = []
    geometry = None
    # Resolve and validate every required header before creating any output.
    for split, dates in [('train',train_dates), ('val',val_dates)]:
        for date in dates:
            day = datetime.strptime(date,'%Y%m%d')
            sources = [(day+timedelta(minutes=10*i), _source_path(raw_dir,day+timedelta(minutes=10*i))) for i in range(1,144)]
            for timestamp,path in sources:
                with netCDF4.Dataset(path) as dataset:
                    current = validate_header(dataset,timestamp)
                if geometry is not None and current != geometry:
                    raise ValueError('Inconsistent raw geometry')
                geometry = current
            planned.append((split,date,sources))
    lat,lon = build_grid(geometry)
    grid_basis='header-derived KMA spherical LCC, standard parallels30/60, radius6371.00877km; not official sfc_grid_latlon.nc'
    grid_sources=[]
    if grid_source_dir is not None:
        source_root=Path(grid_source_dir).resolve()
        supplied=[]
        for name, expected in [('lat.npy',lat),('lon.npy',lon)]:
            path=source_root/name
            values=np.load(path,allow_pickle=False)
            if values.shape!=(100,100) or not np.isfinite(values).all():
                raise ValueError('Supplied grid must be finite100x100: '+name)
            if np.max(np.abs(values-expected))>.02:
                raise ValueError('Supplied coordinates disagree with native LCC region: '+name)
            supplied.append(values.copy())
            grid_sources.append(dict(path=str(path),sha256=sha256(path)))
        if not (np.diff(supplied[0],axis=0)>0).all() or not (np.diff(supplied[1],axis=1)>0).all():
            raise ValueError('Supplied grid must increase northward by row and eastward by column')
        lat,lon=supplied
        grid_basis='supplied lat/lon product preserved exactly; dimensions/orientation/LCC region checked, source paths and SHA256 recorded'
    out.mkdir(parents=True)
    (out/'grid').mkdir()
    np.save(out/'grid/lat.npy',lat)
    np.save(out/'grid/lon.npy',lon)
    days, sampling, artifacts = [], [], []
    for split,date,sources in planned:
        rain,context,valid,native_valid,provenance = build_day(date,sources)
        refs,qc = audit_windows(rain,context,valid,native_valid,min_fine,min_context)
        if not refs:
            raise ValueError('No audited H6/L18 windows for '+split+'/'+date+'; partial output is not a passed dataset')
        path=out/split/(date+'.npz')
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,rain=rain,context=context,valid=valid,native_valid=native_valid.astype(np.uint8),
                            version=VERSION,date=date,frame_minutes=np.arange(144)*10,unit='mm/10min',context_km=400)
        daily_hash=sha256(path)
        days.append(dict(date=date,split=split,sha256=daily_hash,version=VERSION,unit='mm/10min',
                         train_indices=refs if split=='train' else [],
                         eval_indices=refs if split=='val' else [],qc=qc,raw_provenance=provenance))
        # Neutral weights are transparent and reproducible; no invented birth
        # classifier or claim to recreate the historical weighted index.
        sampling.extend(dict(split=split,date=date,ref=ref,weight=1.) for ref in refs if split=='train')
        artifacts.append(dict(path=path.relative_to(out).as_posix(),sha256=daily_hash,windows=len(refs)))
        print(f'{split}/{date}: {len(refs)}/120 audited windows',flush=True)
    source_paths=[Path(__file__),INFERENCE/'preprocess_v27.py',INFERENCE/'preprocess_context.py',INFERENCE/'nc_input.py',INFERENCE/'raw_contract.py']
    identity={p.name:sha256(p) for p in source_paths}
    manifest=dict(audit_pass=True,protocol=dict(history=6,horizon=18,version=VERSION,
                  min_fine_known_fraction=min_fine,min_context_known_fraction=min_context,
                  frame0='structural unavailable, excluded from inputs and targets',
                  fine_valid_basis='full containing 2km context cell AND native interval coverage',
                  missing_policy='current and previous raw interval coverage; unknown stored zero remains masked'),
                  days=days,artifacts=artifacts,pipeline_sha256=identity,
                  counts=dict(train_windows=len(sampling),val_windows=sum(len(d['eval_indices']) for d in days)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    (out/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sampling),encoding='utf8')
    info=dict(version=VERSION,unit='mm/10min',raw_dir=str(raw_dir),grid_dir=str(out/'grid'),
              coordinate_basis=grid_basis,grid_source_provenance=grid_sources,
              raw_unit='integer tenths of mm daily accumulation; historical mm/h attribute explicitly whitelisted',
              grid_geometry=geometry,pipeline_sha256=identity,sampling='uniform weight1 audited train refs; not historical birth index',
              manifest_sha256=sha256(out/'manifest.json'),index_sha256=sha256(out/'index.jsonl'),
              grid_sha256={name:sha256(out/'grid'/name) for name in ('lat.npy','lon.npy')})
    (out/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--train-dates',required=True)
    parser.add_argument('--val-dates',required=True)
    parser.add_argument('--grid-source-dir',type=Path,help='Preserve supplied lat.npy/lon.npy rather than approximating from LCC header')
    parser.add_argument('--min-fine-coverage',type=float,default=.95)
    parser.add_argument('--min-context-coverage',type=float,default=.50,
                        help='Minimum known native area fraction in400km context (default half; outside observation domain stays masked)')
    args=parser.parse_args()
    result=prepare(args.raw_dir,args.out,args.train_dates,args.val_dates,args.min_fine_coverage,args.min_context_coverage,args.grid_source_dir)
    print(json.dumps(dict(audit_pass=result['audit_pass'],counts=result['counts'])))


if __name__=='__main__':
    main()
