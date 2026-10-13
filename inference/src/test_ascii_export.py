"""Data-free projection, aggregation, units and time preservation checks."""
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pyproj import CRS, Transformer
from ascii_export import export_predictions

class ExportTests(unittest.TestCase):
    def setUp(self):
        # Synthetic projected pixel centres; no private coordinate files.
        inverse = Transformer.from_crs(3857, 4326, always_xy=True)
        self.lon, _ = inverse.transform(14000000 + np.arange(100)*499.3, np.full(100,4500000))
        _, self.lat = inverse.transform(np.full(100,14000000), 4500000 + np.arange(100)*499.3)
        self.first = datetime(2022, 8, 8, 23, 50)

    def test_numeric_projection_units_prj_and_time(self):
        yy, xx = np.indices((100, 100))
        field = ((xx + yy * 3) % 17).astype(np.float32) / 10
        values = np.stack([field + np.float32(i / 10) for i in range(18)])
        values[0] = 2  # exact unit check: 2 mm/10min -> 12 mm/h
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "output_220808_2350"
            out = export_predictions(values, self.lat, self.lon, root, self.first)
            asc = sorted(out.glob("*.asc"))
            self.assertEqual(len(asc), 18)
            self.assertEqual(len(list(out.glob("*.prj"))), 18)
            self.assertEqual(asc[0].name, "pred_physical_t0006_EPSG3857_1km.asc")
            self.assertEqual(asc[-1].name, "pred_physical_t0023_EPSG3857_1km.asc")
            for i, path in enumerate(asc):
                header = dict(line.split() for line in path.read_text().splitlines()[:6])
                nx,ny=int(header['ncols']),int(header['nrows'])
                actual=np.loadtxt(path,skiprows=6)
                projection=Transformer.from_crs(4326,3857,always_xy=True)
                buckets={}
                for row in range(100):
                    for col in range(100):
                        x,y=projection.transform(self.lon[col],self.lat[row])
                        cx=min(nx-1,max(0,int((x-float(header['xllcorner']))/1000)))
                        cy=min(ny-1,max(0,int((y-float(header['yllcorner']))/1000)))
                        buckets.setdefault((ny-1-cy,cx),[]).append(float(values[i,row,col]*np.float32(6)))
                expected=np.full((ny,nx),-9999.)
                for (row,col),samples in buckets.items(): expected[row,col]=np.mean(samples)
                np.testing.assert_allclose(actual,expected,rtol=0,atol=0.000051)
                self.assertEqual(CRS.from_wkt(path.with_suffix(".prj").read_text()).to_epsg(), 3857)
            data = np.loadtxt(asc[0], skiprows=6)
            self.assertEqual(data.shape, (50, 50))
            np.testing.assert_array_equal(data[data != -9999], 12)
            header = asc[0].read_text().splitlines()[:6]
            self.assertIn("cellsize      1000.0000000000", header)
            text = (root / "time_alignment.txt").read_text()
            forecast = text.split("FORECAST_TIMES\n")[1].splitlines()
            inputs = text.split("INPUT_TIMES\n")[1].split("FORECAST_TIMES")[0].splitlines()
            self.assertEqual(len(forecast), 18)
            self.assertEqual(len(inputs), 6)
            for i, line in enumerate(forecast):
                self.assertIn((self.first + timedelta(minutes=10*i)).isoformat(timespec="minutes"), line)
                self.assertIn("lead_min=%d" % (10*(i+1)), line)
            self.assertIn("2022-08-08T22:50", inputs[0])
            self.assertIn("2022-08-08T23:40", inputs[-1])
            self.assertIn("2022-08-09T02:40", forecast[-1])
            self.assertFalse((root / "frames_gt").exists())
            self.assertFalse(any(out.glob("true*")))
            before = asc[0].read_bytes()
            with self.assertRaises(FileExistsError):
                export_predictions(values, self.lat, self.lon, root, self.first)
            self.assertEqual(asc[0].read_bytes(), before)

    def test_bad_shapes_and_nonfinite_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "invalid"
            for values in (np.zeros((17,100,100)), np.full((18,100,100), np.nan), np.full((18,100,100), -1)):
                with self.assertRaises(ValueError):
                    export_predictions(values, self.lat, self.lon, root, self.first)
                self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
