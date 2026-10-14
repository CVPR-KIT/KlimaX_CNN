"""New artifacts must opt in; weights, step, geometry and architecture stay bound."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from motion_core_arch import MotionCoreNowcaster
from runtime import load_runtime, predict, read_model_contract, validate_contract_grid
from test_runtime import sample_inputs


class ArtifactTests(unittest.TestCase):
    def test_new_weights_require_explicit_matching_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.manual_seed(7)
            model = MotionCoreNowcaster(history_steps=6, forecast_steps=18,
                quantiles=(.5,.6,.75,.9,.95), hidden=48, fine_hidden=32,
                fine_birth_enabled=True, fine_correction_cap=20., fine_readout_floor=.05)
            weights = root/'new.ckpt'
            torch.save({'global_step':1,'state_dict':{'net.'+k:v for k,v in model.state_dict().items()}},weights)
            for name in ('lat.npy','lon.npy'): np.save(root/name,np.zeros((100,100)))
            contract = dict(schema='klimax_v45_model_contract_v1',
                checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                global_step=1,history_steps=6,forecast_steps=18,quantiles=[.5,.6,.75,.9,.95],
                hidden=48,fine_hidden=32,fine_birth_enabled=True,fine_correction_cap=20.,
                fine_readout_floor=.05,preprocess_version='rn_day_method4c_reset_v3_qc_400km',
                grid_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('lat.npy','lon.npy')})
            path = root/'model_contract.json'
            path.write_text(json.dumps(contract),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'): load_runtime(weights)
            runtime = load_runtime(weights,contract_path=path)
            self.assertEqual(runtime.global_step,1)
            result = predict(runtime,*sample_inputs())
            self.assertEqual(result.shape,(18,100,100))
            self.assertTrue(np.isfinite(result).all())
            validate_contract_grid(path,root)
            np.save(root/'lat.npy',np.ones((100,100)))
            with self.assertRaisesRegex(ValueError,'coordinate SHA256 mismatch'): validate_contract_grid(path,root)
            for key,value,message in [('global_step',2,'Unexpected'),('history_steps',18,'Unsupported')]:
                changed=dict(contract,**{key:value})
                path.write_text(json.dumps(changed),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,message): load_runtime(weights,contract_path=path)
            path.write_text(json.dumps(dict(contract,checkpoint_sha256='0'*64)),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'): load_runtime(weights,contract_path=path)


if __name__ == '__main__': unittest.main()
