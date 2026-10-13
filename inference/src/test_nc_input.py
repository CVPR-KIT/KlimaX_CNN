"""Meaningful causal/debt/persisted-input tests, no model or future truth needed."""
import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import nc_input
from preprocess_v27 import ResetSafeMethod4C


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.day = dt.datetime(2022, 8, 8)
        for i in range(1, 13):
            (self.root / (self.day + dt.timedelta(minutes=10 * i)).strftime('%Y%m%d%H%M.nc')).write_bytes(b'raw-placeholder')
        self.seen = []

    def tearDown(self):
        self.temp.cleanup()

    def raw(self, path):
        index = int(path.stem[-4:-2]) * 6 + int(path.stem[-2:]) // 10
        self.seen.append(index)
        # Debt arose BEFORE the six-frame window. It must offset later growth.
        levels = [0, 10, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 999]
        a = np.full((800, 800), levels[index], np.float32)
        a[351:353, 351:353] = np.nan  # exercises frame coverage quantization
        return a

    def run_input(self):
        with patch.object(nc_input, '_read_raw', side_effect=self.raw):
            return nc_input.prepare_input(self.root, self.day + dt.timedelta(minutes=120),
                                          self.root / 'converted_nc', self.root / 'converted_npz')

    def test_daily_prefix_debt_and_reloaded_inputs(self):
        result = self.run_input()  # anchor01:50, input01:00..01:50
        self.assertEqual(self.seen, list(range(1, 12)))
        proc = ResetSafeMethod4C()
        expected = []
        for index in range(1, 12):
            timestamp = self.day + dt.timedelta(minutes=10 * index)
            field = proc.update(self.raw(Path(timestamp.strftime('%Y%m%d%H%M.nc'))), timestamp)
            if index >= 6:
                # Independently execute frozen training conversion contract.
                field = np.where(field <= 500, field, np.nan)
                fine = np.nan_to_num(field[350:450, 350:450], nan=0.)
                blocks = field.reshape(200, 4, 200, 4)
                count = np.isfinite(blocks).sum((1, 3))
                context = (np.nan_to_num(blocks, nan=0.).sum((1, 3)) / np.maximum(count, 1)).astype(np.float16)
                valid = np.round(count * 255 / 16).astype(np.uint8)
                expected.append((fine, context, valid))
        np.testing.assert_array_equal(result['fine'][:, 0], np.stack([e[0] for e in expected]))
        np.testing.assert_array_equal(result['context'][:, 0], np.stack([e[1] for e in expected]).astype(np.float32))
        np.testing.assert_array_equal(result['valid'][:, 0], np.stack([e[2] for e in expected]).astype(np.float32) / 255.)
        # Starting at window01:00 would make its cumulative8 a false8mm step.
        self.assertEqual(float(result['fine'][0, 0, 50, 50]), 0.)
        with np.load(result['npz_path']) as saved:
            np.testing.assert_array_equal(saved['rain'], result['fine'][:, 0])
        self.assertEqual(len(list((self.root / 'converted_nc').glob('*.nc'))), 6)

    def test_future_file_changes_cannot_change_inputs(self):
        first = self.run_input()
        future = self.root / '202208080200.nc'
        future.write_bytes(b'not-a-netcdf-and-drastically-different-future')
        self.seen.clear()
        second = self.run_input()
        self.assertEqual(self.seen, list(range(1, 12)))
        for key in ('fine', 'context', 'valid', 'clock'):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first['provenance'], second['provenance'])

    def test_missing_early_debt_history_rejected_before_output(self):
        # Rename within temporary test directory; do not touch project raw NC.
        (self.root / '202208080010.nc').rename(self.root / 'early_missing.nc')
        with self.assertRaises(FileNotFoundError):
            self.run_input()
        self.assertFalse((self.root / 'converted_npz').exists())

    def test_unavailable_midnight_frame_rejected(self):
        with self.assertRaises(ValueError):
            nc_input.prepare_input(self.root, self.day + dt.timedelta(minutes=60),
                                   self.root / 'converted_nc', self.root / 'converted_npz')

    def test_known_clock_wraps_across_year(self):
        clock = nc_input.clock_features(dt.datetime(2025, 1, 1))
        np.testing.assert_array_equal(clock, np.asarray([0, 1, 0, 1], np.float32))


if __name__ == '__main__':
    unittest.main()
