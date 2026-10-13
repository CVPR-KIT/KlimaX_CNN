"""Prediction-only legacy-compatible EPSG:3857 export. Input unit: mm/10min."""
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

def _infer_lat_lon_1d(lat, lon):
    lat, lon = np.asarray(lat), np.asarray(lon)
    if lat.ndim == 1 and lon.ndim == 1:
        return lat.astype(float), lon.astype(float)
    if lat.ndim == 2 and lon.ndim == 2:
        return lat[:, 0].astype(float), lon[0, :].astype(float)
    raise ValueError(f"lat/lon dimensions do not match: lat {lat.shape}, lon {lon.shape}")

def _snap_down(x, origin, step):
    return origin + np.floor((x - origin) / step) * step

def _snap_up(x, origin, step):
    return origin + np.ceil((x - origin) / step) * step

def _write_prj_epsg(epsg, asc_path):
    from pyproj import CRS

    Path(asc_path).with_suffix(".prj").write_text(CRS.from_epsg(epsg).to_wkt(), encoding="utf-8")

def _agg_to_3857(grid2d, lat, lon, out_cell_m=1000.0, nodata=-9999.0, agg="mean", fmt="%.4f", out_path=None):
    from pyproj import Transformer

    lat1d, lon1d = _infer_lat_lon_1d(lat, lon)
    height, width = grid2d.shape

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lon2, lat2 = np.meshgrid(lon1d, lat1d)
    xc, yc = transformer.transform(lon2, lat2)

    dx_src = float(np.nanmedian(np.abs(np.diff(xc[height // 2, :]))))
    dy_src = float(np.nanmedian(np.abs(np.diff(yc[:, width // 2]))))

    x0 = float(np.nanmin(xc) - dx_src / 2)
    y0 = float(np.nanmin(yc) - dy_src / 2)
    x1 = float(np.nanmax(xc) + dx_src / 2)
    y1 = float(np.nanmax(yc) + dy_src / 2)

    xll = _snap_down(x0, x0, out_cell_m)
    yll = _snap_down(y0, y0, out_cell_m)
    xur = _snap_up(x1, x0, out_cell_m)
    yur = _snap_up(y1, y0, out_cell_m)

    ncols = int(round((xur - xll) / out_cell_m))
    nrows = int(round((yur - yll) / out_cell_m))

    col = np.clip(np.floor((xc - xll) / out_cell_m).astype(int), 0, ncols - 1)
    row = np.clip(np.floor((yc - yll) / out_cell_m).astype(int), 0, nrows - 1)

    valid = np.isfinite(grid2d)
    rr, cc, vv = row[valid].ravel(), col[valid].ravel(), grid2d[valid].ravel()

    if agg == "max":
        out_grid = np.full((nrows, ncols), -np.inf)
        np.maximum.at(out_grid, (rr, cc), vv)
        out_grid = np.where(np.isfinite(out_grid), out_grid, nodata)
    else:
        sums = np.zeros((nrows, ncols))
        counts = np.zeros((nrows, ncols), dtype=np.int64)
        np.add.at(sums, (rr, cc), vv)
        np.add.at(counts, (rr, cc), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out_grid = np.where(counts > 0, sums / counts, nodata)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            f"ncols         {ncols}\n"
            f"nrows         {nrows}\n"
            f"xllcorner     {xll:.10f}\n"
            f"yllcorner     {yll:.10f}\n"
            f"cellsize      {out_cell_m:.10f}\n"
            f"NODATA_value  {nodata}\n"
        )
        np.savetxt(f, np.flipud(out_grid), fmt=fmt, delimiter=" ")

def export_predictions(corrected_mm10, lat, lon, outroot, first_forecast):
    """Write 18 legacy ASC/PRJ pairs; convert mm/10min to mm/h exactly once.

    ``outroot`` is the caller-selected output_YYMMDD_HHMM directory, where
    HHMM denotes first_forecast. Existing results are never cleared.
    """
    values = np.asarray(corrected_mm10, dtype=np.float32)
    if values.shape != (18, 100, 100):
        raise ValueError("Expected 18 predictions of 100x100, got %r" % (values.shape,))
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Predictions must be finite, nonnegative mm/10min")
    if not isinstance(first_forecast, datetime):
        raise TypeError("first_forecast must be datetime")
    lat1d, lon1d = _infer_lat_lon_1d(lat, lon)
    if lat1d.shape != (100,) or lon1d.shape != (100,):
        raise ValueError("Latitude and longitude must describe the 100x100 fine grid")
    if not np.isfinite(lat1d).all() or not np.isfinite(lon1d).all():
        raise ValueError("Coordinates must be finite")
    root = Path(outroot)
    ascii_dir = root / "ascii_epsg3857"
    alignment = root / "time_alignment.txt"
    if ascii_dir.exists() and (not ascii_dir.is_dir() or any(ascii_dir.iterdir())):
        raise FileExistsError("Refusing to overwrite existing ASCII results: %s" % ascii_dir)
    if alignment.exists():
        raise FileExistsError("Refusing to overwrite existing time alignment: %s" % alignment)
    ascii_dir.mkdir(parents=True, exist_ok=True)
    for index, grid in enumerate(values * np.float32(6.0)):
        path = ascii_dir / ("pred_physical_t%04d_EPSG3857_1km.asc" % (index + 6))
        _agg_to_3857(grid, lat, lon, out_cell_m=1000.0, out_path=path)
        _write_prj_epsg(3857, path)
    lines = [
        "Forecast output: prediction only; no GT or frames_gt",
        "Grid: EPSG:3857, 1000 m; legacy mean aggregation",
        "Model/correction unit: mm/10min; ASCII unit: mm/h (multiplied by 6 once)",
        "No additional smoothing; forecast interval: 10 minutes",
        "Filename steps are window-relative: inputs t0000..t0005; predictions t0006..t0023",
        "First forecast: " + first_forecast.isoformat(timespec="minutes"),
        "INPUT_TIMES",
    ]
    for index in range(6):
        stamp = first_forecast + timedelta(minutes=10 * (index - 6))
        lines.append("t%04d %s" % (index, stamp.isoformat(timespec="minutes")))
    lines.append("FORECAST_TIMES")
    for index in range(18):
        stamp = first_forecast + timedelta(minutes=10 * index)
        lines.append("pred_physical_t%04d_EPSG3857_1km.asc %s lead_min=%d" % (
            index + 6, stamp.isoformat(timespec="minutes"), 10 * (index + 1)))
    alignment.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ascii_dir
