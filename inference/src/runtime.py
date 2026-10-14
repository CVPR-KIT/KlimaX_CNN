"""Strict, Lightning-free inference for the frozen V45 H6 checkpoint."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pathlib
import pickle
import types

import numpy as np
import torch

from motion_core_arch import MotionCoreNowcaster

CHECKPOINT_SHA256 = "fb7e43c65c38493efdec4f4a4af9649ec7f8149cf0a80f0e9cd552777faae755"
CHECKPOINT_STEP = 85140
QUANTILES = (0.5, 0.6, 0.75, 0.9, 0.95)


class _PortableUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Linux training metadata contains PosixPath. Map only that metadata
        # type on Windows, without monkey-patching pathlib globally.
        if os.name == "nt" and module == "pathlib" and name == "PosixPath":
            return pathlib.WindowsPath
        return super().find_class(module, name)


_PORTABLE_PICKLE = types.ModuleType("portable_pickle")
_PORTABLE_PICKLE.__dict__.update(pickle.__dict__)
_PORTABLE_PICKLE.Unpickler = _PortableUnpickler


def load_checkpoint(path):
    return torch.load(str(path), map_location="cpu", weights_only=False,
                      pickle_module=_PORTABLE_PICKLE)


@dataclass
class Runtime:
    model: MotionCoreNowcaster
    device: torch.device
    checkpoint_path: Path
    checkpoint_sha256: str
    global_step: int


def read_model_contract(path):
    """Explicit opt-in to a locally trained V45 artifact, never an unchecked override."""
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = dict(schema="klimax_v45_model_contract_v1", history_steps=6,
                    forecast_steps=18, quantiles=list(QUANTILES), hidden=48,
                    fine_hidden=32, fine_birth_enabled=True,
                    fine_correction_cap=20.0, fine_readout_floor=0.05,
                    preprocess_version="rn_day_method4c_reset_v3_qc_400km")
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError("Unsupported model contract field: " + key)
    digest = contract.get("checkpoint_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid model contract checkpoint_sha256")
    step = contract.get("global_step")
    if type(step) is not int or step < 1:
        raise ValueError("Invalid model contract global_step")
    grids = contract.get("grid_sha256", {})
    if not isinstance(grids, dict) or set(grids) != {"lat.npy", "lon.npy"}:
        raise ValueError("Model contract requires both coordinate hashes")
    for digest in grids.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid model contract coordinate hash")
    return contract


def validate_contract_grid(contract_path, grid_dir):
    contract = read_model_contract(contract_path)
    for name, expected in contract["grid_sha256"].items():
        actual = hashlib.sha256((Path(grid_dir) / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Model contract coordinate SHA256 mismatch: " + name)


def load_runtime(path, device="cpu", contract_path=None):
    """Default frozen checkpoint or an explicitly supplied, hash-bound local artifact."""
    contract = read_model_contract(contract_path) if contract_path else None
    expected_sha = contract["checkpoint_sha256"] if contract else CHECKPOINT_SHA256
    expected_step = contract["global_step"] if contract else CHECKPOINT_STEP
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    if checksum != expected_sha:
        raise ValueError("V45 checkpoint SHA256 mismatch: " + checksum)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "cpu":
        torch.set_num_threads(2)
    checkpoint = load_checkpoint(path)
    step = checkpoint.get("global_step", -1)
    if type(step) is not int:
        raise ValueError("Checkpoint global_step must be an integer")
    if step != expected_step:
        raise ValueError("Unexpected V45 checkpoint step: " + str(step))
    state = checkpoint["state_dict"]
    if not state or any(not key.startswith("net.") for key in state):
        raise ValueError("Checkpoint must contain exclusively net.* model state")
    model = MotionCoreNowcaster(
        history_steps=6, forecast_steps=18, quantiles=QUANTILES,
        hidden=48, fine_hidden=32, fine_birth_enabled=True,
        fine_correction_cap=20.0, fine_readout_floor=0.05,
    )
    model.load_state_dict({key[4:]: value for key, value in state.items()}, strict=True)
    model.eval().to(device=device, dtype=torch.float32)
    return Runtime(model, device, path.resolve(), checksum, step)


def _array(value, shape, name, nonnegative=False, mask=False):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape:
        raise ValueError("%s shape must be %s, got %s" % (name, shape, value.shape))
    if not np.isfinite(value).all():
        raise ValueError(name + " contains non-finite values")
    if nonnegative and np.any(value < 0):
        raise ValueError(name + " must be nonnegative")
    if mask and np.any((value < 0) | (value > 1)):
        raise ValueError(name + " must be in [0, 1]")
    return value


def predict(runtime, fine, context, valid, clock):
    """Return native q0.6 mm/10min, shape [18,100,100], without correction.

    Inputs contain only six observed frames and 19 known calendar vectors
    (anchor plus future times). No target or saved forecast is consumed.
    """
    arrays = [
        _array(fine, (6, 1, 100, 100), "fine", nonnegative=True),
        _array(context, (6, 1, 200, 200), "context", nonnegative=True),
        _array(valid, (6, 1, 200, 200), "valid", mask=True),
        _array(clock, (19, 4), "clock"),
    ]
    tensors = [torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(runtime.device)
               for value in arrays]
    with torch.inference_mode():
        prediction = runtime.model(*tensors)["fine"][0, :, QUANTILES.index(0.6)]
    result = prediction.float().cpu().numpy().copy()
    if result.shape != (18, 100, 100) or not np.isfinite(result).all() or np.any(result < 0):
        raise RuntimeError("V45 produced invalid native rainfall")
    return result
