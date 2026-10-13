import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from correction import apply_correction

PRESETS=Path(__file__).resolve().parents[1]/'presets.json'
class CorrectionTests(unittest.TestCase):
    def test_common_profile_all_axes_cap_and_no_mutation(self):
        raw=np.linspace(0,25,18*2*3).reshape(2,18,3)
        before=raw.copy()
        params=json.loads(PRESETS.read_text(encoding='utf-8'))['profiles']['shared_three_block_power_cap12']
        actual=apply_correction(raw,'shared_three_block_power_cap12',lead_axis=1,presets_path=PRESETS)
        expected=raw.astype(np.float64).copy()
        for i,p in enumerate(params['blocks']): expected[:,i*6:i*6+6]=p['gain']*expected[:,i*6:i*6+6]**p['gamma']
        np.minimum(expected,12,out=expected)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(raw,before)
        self.assertEqual(actual.max(),12)
    def test_invalid_cap_rejected_and_legacy_profiles_unchanged(self):
        source=json.loads(PRESETS.read_text(encoding='utf-8'))
        raw=np.ones((18,2,2))
        for bad in (0,-1,float('nan'),True):
            source['profiles']['shared_three_block_power_cap12']['cap_mm10']=bad
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'presets.json';path.write_text(json.dumps(source),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'cap'):
                    apply_correction(raw,'shared_three_block_power_cap12',lead_axis=0,presets_path=path)
        result=apply_correction(raw,'recommended_2022_three_block_power',lead_axis=0,presets_path=PRESETS)
        self.assertAlmostEqual(result[0,0,0],2.4060500721642337)
        self.assertAlmostEqual(result[17,0,0],3.819366415641665)
