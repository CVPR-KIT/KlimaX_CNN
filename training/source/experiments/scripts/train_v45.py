"""Coherent fine readout, scratch H6 training with operational smoke evidence."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ClimaX/src'))
sys.path.insert(0, str(ROOT / 'experiments/scripts'))
import torch
import pytorch_lightning as pl
import train_v39 as base
from train_v44_module import Module
from data_v39 import DataModuleV39
from birth_sampling import BirthSampler

EVIDENCE_FILES = ('DONE.json', 'config.json', 'STRICT_RELOAD.json',
                  'gradient_report.json', 'first_batch_verified.json',
                  'SMOKE_SHAPES.json', 'occurrence_gradient_report.json')


def source_hashes():
    return {str(p.relative_to(ROOT)): base.sha256_file(p) for p in sorted(ROOT.rglob('*.py'))}


def check_smoke(smoke, manifest, index, seed=4401):
    smoke = Path(smoke)
    evidence = {name: json.loads((smoke / name).read_text()) for name in EVIDENCE_FILES}
    done, config = evidence['DONE.json'], evidence['config.json']
    reload = evidence['STRICT_RELOAD.json']
    gradients = evidence['gradient_report.json']
    inputs, shapes = evidence['first_batch_verified.json'], evidence['SMOKE_SHAPES.json']
    occurrence = evidence['occurrence_gradient_report.json']
    checks = {
        'smoke_budget': done.get('mode') == 'smoke' and done.get('steps') == 8,
        'same_source': config.get('source_hashes') == source_hashes(),
        'same_seed': config.get('seed') == seed,
        'same_inputs': config.get('manifest_sha256') == base.sha256_file(manifest)
            and config.get('index_sha256') == base.sha256_file(index),
        'strict_reload': all(reload.get(k) is True for k in ('strict_reload', 'flags_match', 'state_hash_match')),
        'checkpoint_bound': base.sha256_file(smoke / 'last.ckpt') == done.get('checkpoint_sha256')
            == reload.get('checkpoint_sha256'),
        'gradients': gradients.get('optimizer_steps') == 8 and bool(gradients.get('expected_branches'))
            and all(gradients.get('branch_abs_grad_sum', {}).get(k, 0) > 0 for k in gradients.get('expected_branches', [])),
        'occurrence_gradient': occurrence.get('optimizer_steps') == 8
            and occurrence.get('fine_occurrence_abs_grad_sum', 0) > 0,
        'exact_first_batch': inputs.get('exact_tensor_match') is True and inputs.get('history_steps') == 6,
        'restored_forward': shapes.get('history_steps') == 6 and shapes.get('horizon_steps') == 18
            and all(shapes.get(k) is True for k in ('finite_outputs', 'ordered_quantiles', 'fine_direct_addition_absent')),
    }
    if not all(checks.values()):
        raise RuntimeError('V45 smoke readiness failed: ' + str(checks))
    return checks


def build_readiness(out, smoke, manifest, index, epochs=20, seed=4401):
    if epochs != 20 or seed != 4401:
        raise RuntimeError('audited contract requires 20 epochs and seed 4401')
    checks = check_smoke(smoke, manifest, index, seed)
    result = dict(ready=True, quality_accepted=False, quality_gate_passed=False,
                  scope='operational smoke only; no predictive-quality acceptance',
                  history=6, epochs=epochs, seed=seed, checks=checks,
                  source_hashes=source_hashes(), manifest_sha256=base.sha256_file(manifest),
                  index_sha256=base.sha256_file(index), smoke=str(Path(smoke).resolve()),
                  evidence={name: dict(path=str((Path(smoke)/name).resolve()),
                                      sha256=base.sha256_file(Path(smoke)/name)) for name in EVIDENCE_FILES})
    base.write_json(Path(out), result)
    return result


class OccurrenceEvidence(pl.Callback):
    def __init__(self, out):
        self.out, self.steps, self.grad = Path(out), 0, 0.0

    def on_before_optimizer_step(self, trainer, module, optimizer, optimizer_idx=0):
        self.steps += 1
        head = module.net.fine_decoder[-1]
        row = module.net.n_quantiles
        self.grad += sum(float(p.grad[row].detach().float().abs().sum())
                         for p in (head.weight, head.bias) if p.grad is not None)
        base.write_json(self.out/'occurrence_gradient_report.json', dict(
            optimizer_steps=self.steps, fine_occurrence_abs_grad_sum=self.grad,
            expected_amount_gradient=False, fine_direct_addition=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--root', required=True, help='Local corrected reset_v2 NPZ corpus directory')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--mode', choices=('smoke', 'train', 'prepare'), required=True)
    ap.add_argument('--smoke', '--smoke-out', dest='smoke', type=Path)
    ap.add_argument('--readiness', type=Path)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=4401)
    a = ap.parse_args()
    if a.mode == 'prepare':
        if not a.smoke:
            raise RuntimeError('prepare requires --smoke')
        build_readiness(a.out, a.smoke, a.manifest, a.index, a.epochs, a.seed)
        return
    if a.epochs != 20 or a.seed != 4401:
        raise RuntimeError('V45 fixed contract: scratch H6, 20 epochs, seed 4401')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    readiness = None
    if a.mode == 'train':
        if not a.readiness:
            raise RuntimeError('train requires new V45 operational readiness')
        readiness = json.loads(a.readiness.read_text())
        if readiness.get('ready') is not True or readiness.get('quality_accepted') is not False:
            raise RuntimeError('invalid V45 operational readiness or unsupported quality claim')
        if readiness.get('source_hashes') != source_hashes() or readiness.get('seed') != a.seed:
            raise RuntimeError('readiness source/seed changed')
        if readiness.get('history') != 6 or readiness.get('epochs') != a.epochs:
            raise RuntimeError('readiness history/budget differs')
        if readiness.get('manifest_sha256') != base.sha256_file(a.manifest) or readiness.get('index_sha256') != base.sha256_file(a.index):
            raise RuntimeError('readiness inputs changed')
        if set(readiness.get('evidence', {})) != set(EVIDENCE_FILES):
            raise RuntimeError('readiness evidence incomplete')
        for item in readiness['evidence'].values():
            if base.sha256_file(item['path']) != item['sha256']:
                raise RuntimeError('readiness evidence changed')
        check_smoke(readiness['smoke'], a.manifest, a.index, a.seed)
    a.out.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(a.seed, workers=True)
    torch.set_num_threads(4)
    dm = DataModuleV39(a.root, a.manifest, batch_size=8, num_workers=0, seed=a.seed, history=6)
    dm.sampler = BirthSampler(dm.train, a.index, seed=a.seed, block_days=8)
    if (len(dm.train), len(dm.val), len(dm.sampler)) != (90823, 815, 136235) or dm.sampler.indexed_windows != len(dm.train):
        raise RuntimeError('corpus/sampler differs from audited H6 contract')
    per_epoch = len(dm.sampler)//16
    total = 8 if a.mode == 'smoke' else a.epochs*per_epoch
    epochs = math.ceil(total/per_epoch)
    model = Module(history_steps=6, lr=2e-4, warmup_steps=1000,
                   birth_supervision=True, fine_birth_enabled=True)
    base.EXPECTED_BRANCHES = tuple(sorted({n.split('.',1)[0] for n,p in model.net.named_parameters() if p.requires_grad}))
    meta = dict(mode=a.mode, version='v45_coherent_fine_h6', pid=os.getpid(), seed=a.seed,
                initialization='scratch', init_state_sha256=base.state_hash(model), hparams=dict(model.hparams),
                parameters=sum(p.numel() for p in model.parameters()), history=6, horizon=18,
                batch_size=8, accumulate_grad_batches=2, effective_batch_size=16,
                epochs=epochs, planned_epochs=20, total_updates=total, updates_per_epoch=per_epoch,
                sampler_samples=len(dm.sampler), manifest=a.manifest, manifest_sha256=base.sha256_file(a.manifest),
                index=a.index, index_sha256=base.sha256_file(a.index), data=dm.summary(),
                gpu=torch.cuda.get_device_name(), source_hashes=source_hashes(),
                checkpoint_monitor='val/selection_score', quality_accepted=False, quality_gate_passed=False,
                start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if readiness:
        meta.update(readiness_path=str(a.readiness), readiness_sha256=base.sha256_file(a.readiness))
    base.write_json(a.out/'config.json', meta)
    trainer, best = base.build_trainer(a.out, total, min(per_epoch,total), 1000 if a.mode=='train' else total,
        limit_val_batches=2 if a.mode=='smoke' else 1.0, smoke=a.mode=='smoke',
        extra_callbacks=[base.Evidence(a.out,a.mode=='smoke',8), OccurrenceEvidence(a.out),
                         base.EpochSampler(dm), base.HistoryInputEvidence(a.out,dm,6), base.SignalStop(a.out)],
        accumulate_grad_batches=2, limit_train_batches=per_epoch*2, max_epochs=epochs,
        checkpoint_monitor='val/selection_score')
    trainer.fit(model,train_dataloaders=dm.train_dataloader(),val_dataloaders=dm.val_dataloader())
    if trainer.global_step != total:
        raise RuntimeError('declared update budget not completed')
    trainer.save_checkpoint(str(a.out/'last.ckpt'))
    restored = Module.load_from_checkpoint(str(a.out/'last.ckpt'),map_location='cpu',strict=True)
    if base.state_hash(restored) != base.state_hash(model):
        raise RuntimeError('strict reload state mismatch')
    flags = bool(restored.net.fine_birth_enabled) and bool(restored.hparams.birth_supervision)
    if not flags or int(restored.hparams.history_steps) != 6 or restored.net.history_steps != 6:
        raise RuntimeError('strict reload flags/history mismatch')
    checksum = base.sha256_file(a.out/'last.ckpt')
    base.write_json(a.out/'STRICT_RELOAD.json',dict(strict_reload=True,state_hash_match=True,flags_match=True,
        checkpoint_sha256=checksum,state_sha256=base.state_hash(restored),mode=a.mode,steps=int(trainer.global_step),history=6))
    if a.mode == 'smoke':
        restored.eval()
        batch = next(iter(dm.val_dataloader()))
        sample = {k:v[:1] if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.no_grad():
            output = restored(sample['fine'],sample['context'],sample['valid'],sample['clock'])
        finite = all(bool(torch.isfinite(v).all()) for v in output.values() if torch.is_tensor(v))
        ordered = bool((output['fine'][:,:,1:] >= output['fine'][:,:,:-1]).all())
        no_add = bool((output['fine_new_rain'] == 0).all())
        if not finite or not ordered or not no_add or output['fine'].shape != (1,18,5,100,100) or output['wide'].shape != (1,18,1,200,200):
            raise RuntimeError('restored forward finite/order/shape/direct-addition failure')
        base.write_json(a.out/'SMOKE_SHAPES.json',dict(history_steps=6,horizon_steps=18,finite_outputs=finite,
            ordered_quantiles=ordered,fine_direct_addition_absent=no_add,strict_restored_forward=True,
            fine_input_shape=list(sample['fine'].shape),wide_input_shape=list(sample['context'].shape),
            fine_output_shape=list(output['fine'].shape),wide_output_shape=list(output['wide'].shape)))
        occurrence = json.loads((a.out/'occurrence_gradient_report.json').read_text())
        if occurrence['fine_occurrence_abs_grad_sum'] <= 0:
            raise RuntimeError('fine occurrence auxiliary disconnected')
    base.write_json(a.out/'DONE.json',dict(steps=int(trainer.global_step),mode=a.mode,checkpoint_sha256=checksum,
        quality_accepted=False,quality_gate_passed=False,best_path=best.best_model_path,
        best_selection_score=float(best.best_model_score) if best.best_model_score is not None else None))


if __name__ == '__main__':
    main()
