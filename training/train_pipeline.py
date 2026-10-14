"""Generic scratch V45 training on audited H6 reset-v2 day files.

This runner uses the historical V45 model and loss without Lightning Trainer.
It supports arbitrary audited dataset sizes; it does not resume or fine-tune.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source/experiments/scripts'))
from data_v39 import ResetV2WindowsV39
from train_v44_module import Module

PREPROCESS_VERSION = 'rn_day_method4c_reset_v3_qc_400km'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_data(data_dir, manifest):
    audit = json.loads(Path(manifest).read_text(encoding='utf-8'))
    days = audit.get('days', [])
    dates = [row['date'] for row in days]
    if len(set(dates)) != len(dates):
        raise ValueError('Manifest dates must be unique and train/val days disjoint')
    provenance = []
    artifacts = {item['path']:item['sha256'] for item in audit.get('artifacts',[])}
    for row in days:
        if not isinstance(row['date'],str) or len(row['date']) != 8 or not row['date'].isdigit():
            raise ValueError('Manifest dates must use YYYYMMDD')
        datetime.strptime(row['date'],'%Y%m%d')
        if row['split'] not in ('train', 'val'):
            raise ValueError('Manifest supports only train and val splits')
        key = 'train_indices' if row['split'] == 'train' else 'eval_indices'
        indices = row.get(key, [])
        if len(set(indices)) != len(indices) or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
            raise ValueError('Window reference indices must be unique integers')
        path = Path(data_dir) / row['split'] / (row['date'] + '.npz')
        if not path.is_file():
            raise FileNotFoundError('Missing manifest day: ' + str(path))
        with np.load(path,allow_pickle=False) as saved:
            if 'version' not in saved or str(saved['version']) != PREPROCESS_VERSION:
                raise ValueError('Training requires explicitly versioned reset-v2 day files: '+str(path))
            if 'unit' not in saved or str(saved['unit']) != 'mm/10min':
                raise ValueError('Training day files require unit mm/10min')
            for name,shape,dtype in [('rain',(144,100,100),np.float32),('context',(144,200,200),np.float16),('valid',(144,200,200),np.uint8)]:
                values=saved[name]
                if values.shape!=shape or values.dtype!=np.dtype(dtype) or not np.isfinite(values).all() or (values<0).any():
                    raise ValueError('Invalid training array shape/dtype/finite/nonnegative values: '+name)
                if name=='valid' and (values>255).any():raise ValueError('Coverage must be 0..255')
        checksum=sha256(path)
        claimed=[row[key] for key in ('sha256','npz_sha256') if key in row]
        relative=row['split']+'/'+row['date']+'.npz'
        if 'artifacts' in audit:
            if relative not in artifacts:raise ValueError('Manifest artifact hash missing: '+relative)
            claimed.append(artifacts[relative])
        if any(value!=checksum for value in claimed):raise ValueError('Manifest NPZ SHA256 mismatch: '+str(path))
        provenance.append(dict(date=row['date'], split=row['split'], path=str(path.resolve()), sha256=checksum))
    return provenance


def validation_score(model, loader, device):
    """Pooled fine q0.6 CSI .1/.5 and lead +120..180 CSI .5."""
    model.eval()
    counts = np.zeros((3, 3), dtype=np.int64)
    losses = []
    with torch.inference_mode():
        for batch_index,batch in enumerate(loader):
            if batch_index==0 or (batch_index+1)%20==0 or batch_index+1==len(loader):
                print(json.dumps(dict(stage='validation',batch=batch_index+1,total_batches=len(loader))),flush=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model.net(batch['fine'], batch['context'], batch['valid'], batch['clock'])
            loss, *_ = model._compute_losses(batch, out)
            if not torch.isfinite(loss):raise ValueError('Nonfinite validation loss')
            losses.append(float(loss))
            prediction = out['fine'][:, :, 1:2]
            truth = batch['truth']
            valid = batch['fine_valid'] > .5
            if prediction.shape != truth.shape or valid.shape != truth.shape:
                raise ValueError('Validation prediction, truth and mask shapes must match exactly')
            for j, (threshold, first_lead) in enumerate(((.1, 0), (.5, 0), (.5, 11))):
                p = prediction[:, first_lead:] >= threshold
                t = truth[:, first_lead:] >= threshold
                mask = valid[:, first_lead:]
                counts[j] += np.asarray([int((event & mask).sum()) for event in (p & t, p & ~t, ~p & t)])
    if not losses:
        raise ValueError('Validation loader must contain at least one window')
    if not counts.sum():
        raise ValueError('Validation has no rain events or predicted events at the selection thresholds; CSI cannot select a model')
    denominator = counts.sum(axis=1)
    csi = np.divide(counts[:, 0], denominator, out=np.zeros(3, dtype=float), where=denominator > 0)
    return dict(selection_score=float(csi.mean()), csi_01=float(csi[0]), csi_05=float(csi[1]),
                csi_05_long=float(csi[2]), counts=counts.tolist(), mean_batch_loss=float(np.mean(losses)),
                no_event_components=[int(i) for i in np.flatnonzero(denominator == 0)],
                no_event_policy='Absent events in a component score zero; all three absent rejects selection')


def make_contract(checkpoint, step, model, coordinate_dir):
    return dict(schema='klimax_v45_model_contract_v1', checkpoint_sha256=sha256(checkpoint),
                global_step=int(step), history_steps=6, forecast_steps=18,
                quantiles=list(model.hparams.quantiles), hidden=int(model.hparams.hidden),
                fine_hidden=int(model.hparams.fine_hidden), fine_birth_enabled=bool(model.hparams.fine_birth_enabled),
                fine_correction_cap=float(model.hparams.fine_correction_cap),
                fine_readout_floor=float(model.hparams.fine_readout_floor), preprocess_version=PREPROCESS_VERSION,
                grid_sha256={name: sha256(Path(coordinate_dir)/name) for name in ('lat.npy', 'lon.npy')})


def save_checkpoint(path, model, step, epoch, evidence):
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if any(not key.startswith('net.') for key in state):
        raise ValueError('Portable checkpoint must contain exclusively net.* state')
    torch.save(dict(state_dict=state, global_step=int(step), epoch=int(epoch),
                    hyper_parameters=dict(model.hparams), training_metadata=evidence), path)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError('Epochs, batch size, threads must be positive; workers nonnegative')
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('max_steps must be positive')
    if args.smoke_val_windows is not None and (args.smoke_val_windows<1 or args.max_steps is None):
        raise ValueError('smoke-val-windows must be positive and requires max-steps')
    if not math.isfinite(args.lr) or args.lr <= 0 or args.warmup_steps < 0:
        raise ValueError('Learning rate must be finite positive; warmup nonnegative')
    data_dir = Path(args.data_dir).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else data_dir/'manifest.json'
    grid = Path(args.coordinate_dir).resolve() if args.coordinate_dir else data_dir/'grid'
    out = Path(args.out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Use a new or empty training output directory')
    provenance = validate_data(data_dir, manifest)
    train_set = ResetV2WindowsV39(data_dir, manifest, 'train', history=6, fixed_eval=False)
    val_set = ResetV2WindowsV39(data_dir, manifest, 'val', history=6, fixed_eval=True)
    for name in ('lat.npy', 'lon.npy'):
        values = np.load(grid/name, allow_pickle=False)
        if values.shape != (100,100) or not np.isfinite(values).all():
            raise ValueError('Training grid must provide finite 100x100 coordinates: ' + name)
    metadata_path=data_dir/'dataset_info.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
    if 'manifest_sha256' in metadata and metadata['manifest_sha256']!=sha256(manifest):
        raise ValueError('Dataset metadata manifest SHA256 mismatch')
    for name,claimed in metadata.get('grid_sha256',{}).items():
        if name not in ('lat.npy','lon.npy') or sha256(grid/name)!=claimed:
            raise ValueError('Dataset metadata grid SHA256 mismatch')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda:0' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    if device.type not in ('cpu','cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested CPU/CUDA device is unavailable')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
                              num_workers=args.workers, drop_last=False)
    validation_set=Subset(val_set,range(min(len(val_set),args.smoke_val_windows))) if args.smoke_val_windows is not None else val_set
    selection_scope='limited_smoke_only' if args.smoke_val_windows is not None else 'all_audited_validation_windows'
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = Module(history_steps=6, lr=args.lr, warmup_steps=args.warmup_steps,
                   birth_supervision=True, fine_birth_enabled=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=float(model.hparams.weight_decay))
    planned = len(train_loader)*args.epochs
    budget = min(planned,args.max_steps) if args.max_steps is not None else planned
    out.mkdir(parents=True,exist_ok=True)
    evidence = dict(schema='klimax_v45_training_run_v1', initialization='scratch', resume=False,
                    data_dir=str(data_dir), manifest=str(manifest), manifest_sha256=sha256(manifest),
                    coordinate_dir=str(grid), data_files=provenance, arguments=vars(args),
                    train_windows=len(train_set), val_windows=len(val_set), planned_steps=planned,
                    evaluated_val_windows=len(validation_set),selection_scope=selection_scope,
                    requested_step_budget=budget, history=6, horizon=18,
                    sampler='All audited train windows, shuffled each epoch, no weighted duplication',
                    historical_training_reproduction=False,
                    selection='mean pooled fine CSI0.1, CSI0.5, +120..180min CSI0.5',
                    device=str(device), torch_version=str(torch.__version__),
                    source_sha256={str(p.relative_to(HERE)):sha256(p) for p in sorted(list((HERE/'source').rglob('*.py'))+[Path(__file__).resolve()]+([HERE/'prepare_data.py'] if (HERE/'prepare_data.py').is_file() else []))})
    (out/'training_config.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    history=[]
    best_score=-float('inf')
    best_step=0
    step=0
    nonzero_gradients=False
    for epoch in range(args.epochs):
        model.train()
        losses=[]
        for batch in train_loader:
            batch = {key:value.to(device) for key,value in batch.items()}
            warmup=min(1.,(step+1)/max(1,args.warmup_steps))
            phase=max(0.,(step-args.warmup_steps)/max(1,budget-args.warmup_steps))
            lr=max(float(model.hparams.eta_min),args.lr*warmup*(.5*(1+math.cos(math.pi*min(phase,1.)))))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            output=model.net(batch['fine'],batch['context'],batch['valid'],batch['clock'])
            loss,*_=model._compute_losses(batch,output)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError('Absent or nonfinite training gradients')
            nonzero_gradients |= any(bool((g!=0).any()) for g in gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            step+=1
            losses.append(float(loss.detach()))
            del output,loss,batch
            if step>=budget:break
        validation=validation_score(model,val_loader,device)
        row=dict(epoch=epoch,global_step=step,partial_epoch=len(losses)<len(train_loader),train_mean_loss=float(np.mean(losses)),learning_rate=lr,validation=validation)
        history.append(row)
        if validation['selection_score']>best_score:
            best_score=validation['selection_score'];best_step=step
            save_checkpoint(out/'best.ckpt',model,step,epoch,evidence)
        save_checkpoint(out/'last.ckpt',model,step,epoch,evidence)
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
        if step>=budget:break
    if not nonzero_gradients:
        raise ValueError('No nonzero gradients observed; reject training artifact')
    contract=make_contract(out/'best.ckpt',best_step,model,grid)
    (out/'model_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    raw_dir=Path(metadata.get('raw_dir',data_dir/'raw_nc'))
    if not raw_dir.is_absolute():raw_dir=data_dir/raw_dir
    reference=val_set.refs[0][2]
    forecast_minutes=(reference+1)*10
    first_clock=f'{forecast_minutes//60:02d}{forecast_minutes%60:02d}'
    configuration=dict(paths=dict(checkpoint=str(out/'best.ckpt'),model_contract=str(out/'model_contract.json'),input_dir=str(raw_dir.resolve()),
                                  grid_dir=str(grid),output_dir=str(out/'predictions')),
                       inference=dict(date=val_set.days[0],time=first_clock,device='cpu',correction='raw'))
    (out/'inference_config.yaml').write_text(yaml.safe_dump(configuration,allow_unicode=True,sort_keys=False),encoding='utf-8')
    result=dict(status='PASS',completed_steps=step,requested_step_budget=budget,selected_step=best_step,
                val_windows=len(val_set),evaluated_val_windows=len(validation_set),selection_scope=selection_scope,
                full_validation_completed=args.smoke_val_windows is None,
                best_selection_score=best_score,nonzero_gradients=True,checkpoint=str(out/'best.ckpt'),
                contract=str(out/'model_contract.json'),inference_config=str(out/'inference_config.yaml'),
                independent_predictive_quality_verified=False)
    (out/'TRAINING_RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--manifest')
    p.add_argument('--coordinate-dir')
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--seed',type=int,default=4401)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--warmup-steps',type=int,default=0)
    p.add_argument('--max-steps',type=int)
    p.add_argument('--smoke-val-windows','--max-val-windows',dest='smoke_val_windows',type=int,
                   help='Smoke only: evaluate deterministic first N audited validation windows; requires --max-steps')
    return p


if __name__=='__main__':
    train(parser().parse_args())
