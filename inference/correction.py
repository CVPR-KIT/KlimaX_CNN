"""Explicit, fixed V45 fine-rain correction profiles; no date-based selection.

Call after extracting nonnegative linear rain in mm, before plotting or scores.
The lead axis must contain all 18 ten-minute leads in chronological order.
Output stays float64 to reproduce the saved corrected-map artifacts exactly.
"""
from pathlib import Path
import json
import operator

import numpy as np


def load_profiles(path=None):
    """Load bundled profiles (or an explicitly supplied JSON path)."""
    source = Path(path) if path is not None else Path(__file__).with_name("presets.json")
    with source.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported correction preset schema")
    if data.get("lead_minutes") != list(range(10, 181, 10)):
        raise ValueError("Profiles require leads +10 through +180 minutes")
    return data["profiles"]


def apply_correction(prediction, profile="raw", *, lead_axis, presets_path=None):
    """Return a corrected copy with explicit lead axis; never modify the input.

    Arrays may have batch/quantile dimensions, e.g. [batch, 18, H, W] or
    [batch, 18, H, W, quantile]. All dimensions and their order are preserved.
    """
    values = np.asarray(prediction)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("Prediction must contain real numeric rain amounts")
    try:
        axis = operator.index(lead_axis)
    except TypeError as exc:
        raise ValueError("lead_axis must be an integer") from exc
    if values.ndim < 3 or not -values.ndim <= axis < values.ndim:
        raise ValueError("Prediction requires a lead axis and two spatial axes")
    axis %= values.ndim
    if values.shape[axis] != 18:
        raise ValueError("Lead axis must contain exactly 18 chronological leads")
    if values.size == 0:
        raise ValueError("Prediction must not have empty dimensions")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Prediction must be finite and nonnegative")
    profiles = load_profiles(presets_path)
    if profile not in profiles:
        raise ValueError(f"Unknown correction profile: {profile}")
    configuration = profiles[profile]
    result = values.astype(np.float64, copy=True)
    if configuration["method"] == "raw":
        return result
    if configuration["method"] != "three_block_power":
        raise ValueError("Unsupported correction method")
    blocks = configuration.get("blocks", [])
    if len(blocks) != 3:
        raise ValueError("Three-block correction requires exactly three blocks")
    output = np.moveaxis(result, axis, 0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for index, block in enumerate(blocks):
                gain, gamma = float(block["gain"]), float(block["gamma"])
                if not np.isfinite([gain, gamma]).all() or gain <= 0 or gamma <= 0:
                    raise ValueError("Correction gain and gamma must be finite and positive")
                selection = slice(index * 6, index * 6 + 6)
                output[selection] = gain * output[selection] ** gamma
        except FloatingPointError as exc:
            raise ValueError("Correction produced an out-of-range rain amount") from exc
    if not np.isfinite(result).all():
        raise ValueError("Correction produced a nonfinite rain amount")
    return result
