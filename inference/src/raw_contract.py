"""Shared strict raw contract for new QC-versioned training and inference."""
import numpy as np

QC_VERSION = "rn_day_method4c_reset_v3_qc_400km"
Y0, X0, SIZE = 1046, 648, 800

GEOMETRY = dict(map_sx=880., map_sy=1540., map_slon=126., map_slat=38.,
                grid_size=.5, grid_nx=2049., grid_ny=2049., map_nx=2048., map_ny=2048.)


def read_raw_strict(path, timestamp):
    """Header-validated integer accumulation with explicit sentinel coverage."""
    import netCDF4
    with netCDF4.Dataset(path) as dataset:
        validate_header(dataset, timestamp)
        dataset.set_auto_maskandscale(False)
        variable = dataset['data']
        raw = np.asarray(variable[Y0:Y0+SIZE, X0:X0+SIZE], dtype=np.float32)
        finite = np.isfinite(raw) & (raw >= 0) & (raw < 9999)
        for key in ('_FillValue', 'missing_value'):
            if key in variable.ncattrs():
                finite &= ~np.isin(raw, np.asarray(variable.getncattr(key)))
    return np.where(finite, raw / 10., np.nan), finite


def mask_interval(rain, finite, previous_finite):
    """A gap and the next increment stay unknown, rather than false dry/rain."""
    known = finite if previous_finite is None else finite & previous_finite
    return np.where(known & np.isfinite(rain) & (rain <= 500), rain, np.nan)

def validate_header(dataset, timestamp):
    """Accept only the known native daily accumulation contract, never rn_10m."""
    if str(getattr(dataset, 'product_type', '')).strip() != 'SFC_GRID_rn_day_':
        raise ValueError('Expected raw cumulative SFC_GRID_rn_day_ product')
    if tuple(map(int, np.asarray(getattr(dataset, 'time', []))[:6])) != (
            timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, 0):
        raise ValueError('NC time disagrees with requested filename timestamp')
    if str(getattr(dataset, 'map_pro', '')).strip() != 'Lambert Conformal Conic Projection':
        raise ValueError('Unsupported projection')
    geometry = {}
    for key, expected in GEOMETRY.items():
        value = float(getattr(dataset, key, float('nan')))
        if not np.isclose(value, expected, rtol=0, atol=1e-8):
            raise ValueError('Unsupported native geometry: ' + key)
        geometry[key] = value
    variable = dataset.variables.get('data')
    if variable is None or variable.shape != (2049, 2049) or not np.issubdtype(variable.dtype, np.integer):
        raise ValueError('Expected native integer data[2049,2049]')
    if float(getattr(variable, 'data_scale', 0)) != 10.:
        raise ValueError('Raw cumulative data_scale must be 10')
    if any(key in variable.ncattrs() for key in ('scale_factor', 'add_offset')):
        raise ValueError('Auto-scaled NC is not the supported raw integer contract')
    # Legacy KMA rn_day says mm/h despite being daily accumulation. Explicitly
    # whitelist that known combination; never infer accumulation from units alone.
    unit = str(getattr(variable, 'unit', '')).strip()
    description = str(variable.getncattr('name')).strip() if 'name' in variable.ncattrs() else ''
    if unit not in ('mm/h', 'mm') or description != 'day rain':
        raise ValueError('Unsupported cumulative NC unit/name')
    if str(getattr(variable, 'transform', '')).replace(' ', '') != 'value=data/data_scale':
        raise ValueError('Unsupported raw-value transform')
    return geometry
