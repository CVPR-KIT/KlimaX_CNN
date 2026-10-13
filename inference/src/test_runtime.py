"""Contract tests and an optional real-checkpoint direct-module comparison.

Run with the KlimaX EXE Python environment:
  python test_runtime.py --checkpoint PATH [--inputs INPUTS.npz]
INPUTS must contain fine, context, valid, clock (no forecast/target required).
"""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from motion_core_arch import MotionCoreNowcaster
from runtime import CHECKPOINT_STEP, QUANTILES, load_checkpoint, load_runtime, predict


def sample_inputs():
    fine = np.zeros((6, 1, 100, 100), np.float32)
    context = np.zeros((6, 1, 200, 200), np.float32)
    for step in range(6):
        fine[step, 0, 42:55, 42 + step:52 + step] = 0.4 + step * 0.1
        context[step, 0, 93:103, 90 + step:100 + step] = 0.3 + step * 0.05
    valid = np.ones_like(context)
    # Known calendar features can be negative; rain cannot.
    phase = np.linspace(2.0, 2.4, 19, dtype=np.float32)
    clock = np.stack([np.sin(phase), np.cos(phase),
                      np.sin(phase / 365), np.cos(phase / 365)], axis=1)
    return fine, context, valid, clock


class ContractTests(unittest.TestCase):
    def test_checkpoint_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.ckpt"
            path.write_bytes(b"not the approved checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_runtime(path)

    def test_invalid_inputs_fail_before_inference(self):
        arrays = sample_inputs()
        for index, mutation, message in [
            (0, lambda a: a[:-1], "shape"),
            (1, lambda a: a * np.nan, "non-finite"),
            (0, lambda a: a - 2, "nonnegative"),
            (2, lambda a: a + 1, r"\[0, 1\]"),
            (3, lambda a: a[:-1], "shape"),
        ]:
            with self.subTest(index=index, message=message):
                changed = list(arrays)
                changed[index] = mutation(changed[index])
                with self.assertRaisesRegex(ValueError, message):
                    predict(None, *changed)


def verify_checkpoint(checkpoint_path, inputs_path=None):
    runtime = load_runtime(checkpoint_path)
    if inputs_path:
        with np.load(inputs_path, allow_pickle=False) as archive:
            arrays = tuple(archive[key] for key in ("fine", "context", "valid", "clock"))
            saved_golden = archive["prediction"].copy() if "prediction" in archive.files else None
    else:
        arrays = sample_inputs()
        saved_golden = None
    actual = predict(runtime, *arrays)
    # Independent construction/state loading bypasses the runtime's loader and
    # output-selection path. The reviewed network is the golden implementation.
    checkpoint = load_checkpoint(checkpoint_path)
    reference = MotionCoreNowcaster(history_steps=6, forecast_steps=18,
                                   quantiles=QUANTILES, hidden=48, fine_hidden=32,
                                   fine_birth_enabled=True, fine_correction_cap=20.,
                                   fine_readout_floor=.05).eval()
    reference.load_state_dict({key.removeprefix("net."): value
                               for key, value in checkpoint["state_dict"].items()}, strict=True)
    tensors = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0)
               for a in arrays]
    with torch.inference_mode():
        full = reference(*tensors)["fine"].numpy()[0]
    golden = full[:, 1]
    np.testing.assert_allclose(actual, golden, rtol=1e-5, atol=1e-6)
    assert np.all(np.diff(full, axis=1) >= -1e-6), "Unordered quantiles"
    assert actual.dtype == np.float32 and actual.shape == (18, 100, 100)
    assert runtime.global_step == CHECKPOINT_STEP
    report = {"checkpoint_sha256": runtime.checkpoint_sha256,
            "step": runtime.global_step, "input_source": str(inputs_path or "synthetic observed-only"),
            "direct_module_max_abs_error": float(np.max(np.abs(actual - golden))),
            "output_shape": list(actual.shape), "output_min": float(actual.min()),
            "output_max": float(actual.max()), "ordered_quantiles": True,
            "rtol": 1e-5, "atol": 1e-6}
    if saved_golden is not None:
        # Cross-version PyTorch kernels need a separately documented tolerance.
        np.testing.assert_allclose(actual, saved_golden, rtol=1e-4, atol=1e-4)
        report.update(remote_golden_max_abs_error=float(np.max(np.abs(actual - saved_golden))),
                      remote_golden_mean_abs_error=float(np.mean(np.abs(actual - saved_golden))),
                      remote_rtol=1e-4, remote_atol=1e-4)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--inputs")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.checkpoint:
        print(json.dumps(verify_checkpoint(args.checkpoint, args.inputs), indent=2))
