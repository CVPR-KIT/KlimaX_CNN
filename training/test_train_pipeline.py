"""End-to-end synthetic day-file training, including a real CPU update/reload."""
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
import torch

from train_pipeline import Module, PREPROCESS_VERSION, parser, sha256, train, validate_data


def fixture(root):
    for split,date,amount in [('train','20000101',.5),('val','20000102',.7)]:
        (root/split).mkdir()
        np.savez_compressed(root/split/(date+'.npz'),rain=np.full((144,100,100),amount,np.float32),
                            context=np.full((144,200,200),amount,np.float16),
                            valid=np.full((144,200,200),255,np.uint8),version=PREPROCESS_VERSION,unit='mm/10min')
    manifest=dict(audit_pass=True,protocol=dict(history=6,horizon=18),
                  days=[dict(date='20000101',split='train',train_indices=[6]),
                        dict(date='20000102',split='val',eval_indices=[6])])
    for row in manifest['days']:row['sha256']=sha256(root/row['split']/(row['date']+'.npz'))
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (root/'grid').mkdir()
    lon,lat=np.meshgrid(np.linspace(126.7,127.2,100),np.linspace(37.3,37.8,100))
    np.save(root/'grid/lat.npy',lat);np.save(root/'grid/lon.npy',lon)
    (root/'dataset_info.json').write_text(json.dumps(dict(raw_dir=str(root/'original_nc'),manifest_sha256=sha256(root/'manifest.json'),grid_sha256={name:sha256(root/'grid'/name) for name in ('lat.npy','lon.npy')})),encoding='utf-8')
    return manifest


class PipelineTests(unittest.TestCase):
    def test_smoke_validation_limit_requires_max_steps_and_positive_limit(self):
        for options in [['--smoke-val-windows','1'],['--max-steps','1','--smoke-val-windows','0']]:
            args=parser().parse_args(['--data-dir','unused','--out','unused']+options)
            with self.assertRaisesRegex(ValueError,'requires max-steps'):train(args)
    def test_manifest_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=fixture(root)
            manifest['days'][1]['date']='20000101'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'disjoint'):
                validate_data(root,root/'manifest.json')

    def test_missing_manifest_day_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            (root/'val/20000102.npz').unlink()
            with self.assertRaises(FileNotFoundError):validate_data(root,root/'manifest.json')

    def test_substituted_npz_rejected_by_manifest_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            with (root/'train/20000101.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):validate_data(root,root/'manifest.json')

    def test_substituted_grid_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            lat=np.load(root/'grid/lat.npy');np.save(root/'grid/lat.npy',lat+.01)
            args=parser().parse_args(['--data-dir',str(root),'--out',str(root/'out')])
            with self.assertRaisesRegex(ValueError,'grid SHA256 mismatch'):train(args)

    def test_one_cpu_update_export_and_strict_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root);out=root/'trained'
            args=parser().parse_args(['--data-dir',str(root),'--out',str(out),'--device','cpu','--max-steps','1','--smoke-val-windows','1'])
            result=train(args)
            self.assertEqual(result['completed_steps'],1)
            self.assertTrue(result['nonzero_gradients'])
            self.assertFalse(result['independent_predictive_quality_verified'])
            self.assertEqual(result['selection_scope'],'limited_smoke_only')
            self.assertFalse(result['full_validation_completed'])
            saved=torch.load(out/'best.ckpt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['global_step'],1)
            self.assertTrue(all(key.startswith('net.') for key in saved['state_dict']))
            restored=Module(**saved['hyper_parameters']).eval()
            restored.load_state_dict(saved['state_dict'],strict=True)
            torch.manual_seed(args.seed)
            initial=Module(history_steps=6,lr=args.lr,warmup_steps=args.warmup_steps,
                           birth_supervision=True,fine_birth_enabled=True)
            self.assertTrue(any(not torch.equal(saved['state_dict'][key],value) for key,value in initial.state_dict().items()))
            from data_v39 import ResetV2WindowsV39
            sample=ResetV2WindowsV39(root,root/'manifest.json','val',history=6,fixed_eval=True)[0]
            with torch.inference_mode():
                prediction=restored.net(*[sample[key].unsqueeze(0) for key in ['fine','context','valid','clock']])['fine']
            self.assertEqual(tuple(prediction.shape),(1,18,5,100,100))
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertTrue((prediction>=0).all())
            self.assertTrue((prediction[:,:,1:]-prediction[:,:,:-1]>=-1e-6).all())
            contract=json.loads((out/'model_contract.json').read_text())
            self.assertEqual(contract['checkpoint_sha256'],sha256(out/'best.ckpt'))
            self.assertEqual(contract['grid_sha256']['lat.npy'],sha256(root/'grid/lat.npy'))
            self.assertEqual(contract['global_step'],result['selected_step'])
            self.assertEqual(contract['schema'],'klimax_v45_model_contract_v1')
            import yaml
            cfg=yaml.safe_load((out/'inference_config.yaml').read_text())
            self.assertEqual(cfg['inference']['correction'],'raw')
            self.assertEqual(cfg['paths']['model_contract'],str(out/'model_contract.json'))
            self.assertEqual(cfg['paths']['input_dir'],str(root/'original_nc'))
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'inference/src'))
            from runtime import load_runtime,predict,validate_contract_grid
            validate_contract_grid(out/'model_contract.json',root/'grid')
            runtime=load_runtime(out/'best.ckpt',device='cpu',contract_path=out/'model_contract.json')
            actual=predict(runtime,*[sample[key].numpy() for key in ['fine','context','valid','clock']])
            np.testing.assert_allclose(actual,prediction[0,:,1].numpy(),rtol=1e-5,atol=1e-6)
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                load_runtime(out/'best.ckpt',device='cpu')
            with self.assertRaises(FileExistsError):train(args)


if __name__=='__main__':unittest.main()
