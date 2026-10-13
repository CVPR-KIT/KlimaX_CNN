"""TC9: evaluate saved 1 km ASCII predictions against matching raw NC cells."""
import csv
from datetime import timedelta
from pathlib import Path

import numpy as np
from pyproj import Geod, Transformer

from src.nc_input import _read_raw, _source_path, _sha256


def read_ascii(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        header = {}
        for _ in range(6):
            key, value = stream.readline().split()
            header[key.lower()] = float(value)
        data = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    required = ('ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata_value')
    if any(key not in header for key in required):
        raise ValueError('ASCII requires corner-based six-line ESRI header')
    if header['cellsize'] != 1000 or data.shape != (int(header['nrows']), int(header['ncols'])):
        raise ValueError('ASCII must be the saved 1000 m prediction grid')
    if not all(np.isfinite(header[key]) for key in required):
        raise ValueError('Invalid ASCII header')
    return header, np.where(data == header['nodata_value'], np.nan, data) / 6.


def tc9_row(timestamp, truth, prediction, reason=''):
    """Inclusive hybrid tolerance; excluded rows never become dry observations."""
    valid = np.isfinite(truth) and np.isfinite(prediction) and truth >= 0 and prediction >= 0
    if not valid:
        return dict(timestamp=timestamp, truth_mm10=float(truth) if np.isfinite(truth) else None,
                    prediction_mm10=float(prediction) if np.isfinite(prediction) else None,
                    absolute_error_mm10=None, absolute_error_percent=None, threshold_mm10=None,
                    passed=None, excluded=True, reason=reason or 'Invalid truth or prediction')
    error = abs(float(prediction) - float(truth))
    threshold = max(1., float(truth)*.5)
    return dict(timestamp=timestamp, truth_mm10=float(truth), prediction_mm10=float(prediction),
                absolute_error_mm10=error, absolute_error_percent=error/truth*100 if truth > 0 else None,
                threshold_mm10=threshold, passed=bool(error <= threshold), excluded=False, reason='')


def _geometry(header, lat, lon, stations):
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if lat.ndim == lon.ndim == 2:
        lat1, lon1 = lat[:, 0], lon[0, :]
    elif lat.ndim == lon.ndim == 1:
        lat1, lon1 = lat, lon
    else:
        raise ValueError('Coordinate dimensions must match')
    if lat1.shape != (100,) or lon1.shape != (100,) or not np.isfinite(lat1).all() or not np.isfinite(lon1).all():
        raise ValueError('Coordinates must describe the native 100x100 fine grid')
    forward = Transformer.from_crs(4326, 3857, always_xy=True)
    inverse = Transformer.from_crs(3857, 4326, always_xy=True)
    geod = Geod(ellps='WGS84')
    longitude, latitude = np.meshgrid(lon1, lat1)
    x, y = forward.transform(longitude, latitude)
    native_lat, native_lon = (lat,lon) if lat.ndim == 2 else (latitude,longitude)
    if native_lat.shape != (100,100) or native_lon.shape != (100,100) or not np.isfinite(native_lat).all() or not np.isfinite(native_lon).all():
        raise ValueError('Actual native station coordinates must be finite 100x100 grids')
    native_x,native_y = forward.transform(native_lon,native_lat)
    nr, nc, size = int(header['nrows']), int(header['ncols']), header['cellsize']
    xll, yll = header['xllcorner'], header['yllcorner']
    dx = float(np.nanmedian(np.abs(np.diff(x[50, :]))))
    dy = float(np.nanmedian(np.abs(np.diff(y[:, 50]))))
    expected_xll, expected_yll = float(np.min(x)-dx/2), float(np.min(y)-dy/2)
    expected_nc = int(np.ceil((float(np.max(x)+dx/2)-expected_xll)/size))
    expected_nr = int(np.ceil((float(np.max(y)+dy/2)-expected_yll)/size))
    if not np.isclose(xll, expected_xll, atol=1e-6, rtol=0) or not np.isclose(yll, expected_yll, atol=1e-6, rtol=0) or (nr, nc) != (expected_nr, expected_nc):
        raise ValueError('Saved ASCII geometry differs from native-grid mean aggregation')
    cols = np.clip(np.floor((x-xll)/size).astype(int), 0, nc-1)
    bottom_rows = np.clip(np.floor((y-yll)/size).astype(int), 0, nr-1)
    cell_ids = (nr-1-bottom_rows)*nc+cols
    count = np.bincount(cell_ids.ravel(), minlength=nr*nc)
    mapping = []
    if len(stations) != 3 or len({str(s['station_id']) for s in stations}) != 3:
        raise ValueError('TC9 requires three distinct stations')
    # Nearest output cell center; empty cells stay excluded, never shift station.
    occupied = np.arange(nr*nc)
    centers_x = xll+(occupied % nc+.5)*size
    centers_y = yll+(nr-1-occupied//nc+.5)*size
    for station in stations:
        sx, sy = forward.transform(float(station['lon']), float(station['lat']))
        distance = np.hypot(centers_x-sx, centers_y-sy)
        nearest = int(np.argmin(distance))
        cell = int(occupied[nearest])
        cx, cy = float(centers_x[nearest]), float(centers_y[nearest])
        center_lon, center_lat = inverse.transform(cx, cy)
        native = np.unravel_index(int(np.argmin((native_x-sx)**2+(native_y-sy)**2)), x.shape)
        _,_,center_distance = geod.inv(float(station['lon']),float(station['lat']),center_lon,center_lat)
        _,_,native_distance = geod.inv(float(station['lon']),float(station['lat']),float(native_lon[native]),float(native_lat[native]))
        mapping.append(dict(station_id=str(station['station_id']), name=station.get('name', ''),
                            station_lat=float(station['lat']), station_lon=float(station['lon']),
                            asc_row=cell//nc, asc_col=cell % nc, center_lat=float(center_lat),
                            center_lon=float(center_lon), distance_m=float(center_distance),
                            distance_basis='geodesic WGS84; nearest selected EPSG3857',
                            native_row=int(native[0]), native_col=int(native[1]),
                            native_lat=float(native_lat[native]), native_lon=float(native_lon[native]),
                            native_distance_m=float(native_distance),
                            native_contribution_count=int(count[cell])))
    return cell_ids, count, mapping


def evaluate_tc9(raw_root, first, ascii_dir, lat, lon, stations):
    result = dict(status='unavailable', criterion='abs_error <= max(1.0 mm/10min, truth*0.5)',
                  unit='mm/10min', prediction_source='Saved prediction ASC (mm/h divided by 6)',
                  truth_source='Unsmoothed raw NC intervals, mean over matching 1 km native contributions',
                  future_observation_used_for_inference=False)
    try:
        timestamps = [first+timedelta(minutes=10*i) for i in range(18)]
        # Resolve the full series before publishing any score; 00:10 resets need no previous NC.
        required = sorted(set(t for stamp in timestamps for t in
                              ([stamp] if stamp.hour == 0 and stamp.minute in (0,10) else [stamp-timedelta(minutes=10),stamp])))
        sources = {stamp: _source_path(Path(raw_root), stamp) for stamp in required}
        paths = [Path(ascii_dir)/('pred_physical_t%04d_EPSG3857_1km.asc' % (i+6)) for i in range(18)]
        predictions = []
        header = None
        for path in paths:
            current_header, values = read_ascii(path)
            if header is not None and current_header != header:
                raise ValueError('Prediction ASCII geometries do not match across leads')
            header = current_header
            predictions.append(values)
        ids, counts, mapping = _geometry(header, lat, lon, stations)
        cumulative = {stamp: _read_raw(path)[350:450,350:450] for stamp,path in sources.items()}
        provenance = [dict(timestamp=stamp.isoformat(), filename=path.name,sha256=_sha256(path))
                      for stamp,path in sources.items()]
        records = [dict(station_id=m['station_id'],name=m['name'],rows=[]) for m in mapping]
        truth_qc = []
        for i, stamp in enumerate(timestamps):
            current = cumulative[stamp]
            if stamp.hour == 0 and stamp.minute == 0:
                delta = np.full_like(current, np.nan)
                midnight = True
            else:
                delta = current if stamp.hour == 0 and stamp.minute == 10 else current-cumulative[stamp-timedelta(minutes=10)]
                midnight = False
            valid = np.isfinite(delta) & (delta >= 0)
            finite_count = np.bincount(ids[valid], minlength=counts.size)
            sums = np.bincount(ids[valid], weights=delta[valid], minlength=counts.size)
            truth = np.full(counts.size,np.nan)
            complete = (counts>0)&(finite_count==counts)
            truth[complete] = sums[complete]/counts[complete]
            truth_qc.append(dict(timestamp=stamp.isoformat(), negative_native_pixels=int((delta<0).sum()),
                                 nonfinite_native_pixels=int((~np.isfinite(delta)).sum()),
                                 complete_truth_cells=int(complete.sum())))
            for record,m in zip(records,mapping):
                cell=m['asc_row']*int(header['ncols'])+m['asc_col']
                observed=truth[cell]; prediction=predictions[i][m['asc_row'],m['asc_col']]
                reason = 'Unavailable midnight reset interval' if midnight else 'Incomplete native truth coverage or negative NC increment'
                if np.isfinite(observed) and (not np.isfinite(prediction) or prediction<0):
                    reason='Prediction ASCII NODATA/nonfinite/negative value'
                record['rows'].append(tc9_row(stamp.isoformat(),observed,prediction,reason))
        all_rows=[dict(station_id=s['station_id'],name=s['name'],**row) for s in records for row in s['rows']]
        valid_rows=[row for row in all_rows if not row['excluded']]
        exclusions=[row for row in all_rows if row['excluded']]
        summaries=[]
        for station in records:
            rows=[row for row in station['rows'] if not row['excluded']]
            hits=sum(row['passed'] for row in rows)
            summaries.append(dict(station_id=station['station_id'],name=station['name'],hits=hits,
                                  valid_pairs=len(rows),excluded_count=18-len(rows),
                                  accuracy_percent=hits/len(rows)*100 if rows else None,
                                  mae_mm10=float(np.mean([row['absolute_error_mm10'] for row in rows])) if rows else None))
        hits=sum(row['passed'] for row in valid_rows)
        result.update(status='available',rows=records,mapping=mapping,station_summary=summaries,
                      exclusions=exclusions,hits=hits,passes=hits,pairs=54,valid_pairs=len(valid_rows),
                      excluded_count=len(exclusions),accuracy_percent=hits/len(valid_rows)*100 if valid_rows else None,
                      mae_mm10=float(np.mean([r['absolute_error_mm10'] for r in valid_rows])) if valid_rows else None,
                      dry_both_count=sum(r['truth_mm10']==0 and r['prediction_mm10']==0 for r in valid_rows),
                      truth_qc=truth_qc,truth_provenance=provenance,
                      prediction_provenance=[dict(filename=p.name,sha256=_sha256(p)) for p in paths])
    except (OSError,ValueError,RuntimeError) as error:
        result['reason']=str(error)
    return result


def write_tc9_tables(result,outroot):
    folder=Path(outroot)/'evaluation'
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise FileExistsError('Refusing to overwrite TC9 tables: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
    row_fields=['station_id','name','timestamp','truth_mm10','prediction_mm10','absolute_error_mm10',
                'absolute_error_percent','threshold_mm10','passed','excluded','reason']
    rows=[dict(station_id=s['station_id'],name=s.get('name',''),**row)
          for s in result.get('rows',[]) for row in s['rows']]
    tables=[('accuracy_table.csv',row_fields,rows),
            ('station_summary.csv',['station_id','name','hits','valid_pairs','excluded_count','accuracy_percent','mae_mm10'],result.get('station_summary',[])),
            ('grid_mapping.csv',['station_id','name','station_lat','station_lon','asc_row','asc_col','center_lat','center_lon','distance_m','distance_basis','native_row','native_col','native_lat','native_lon','native_distance_m','native_contribution_count'],result.get('mapping',[])),
            ('excluded.csv',row_fields,result.get('exclusions',[]))]
    for name,fields,records in tables:
        with (folder/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(records)
    return folder
