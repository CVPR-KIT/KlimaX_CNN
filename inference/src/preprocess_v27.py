"""Versioned, reset-safe fine + 400-km context data; never overwrite v25 data.

Frame zero is deliberately unavailable (zero), matching the daily-file contract.
00:10 starts from zero accumulated rainfall, not yesterday's 00:00 total.
Context validity is per-frame, not a future-dependent daily mean. Both detail and
context have the SAME six input timestamps; all 18 future fine frames are targets.
"""
import argparse
import datetime as dt
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from preprocess_context import Method4CCausal

VERSION = 'rn_day_method4c_reset_v2_400km'
EXCLUDED = {'20220807', '20220808', '20220809', '20250813'}


class ResetSafeMethod4C:
    def __init__(self, threshold=.1):
        self.processor = Method4CCausal(threshold)
        self.day = None
        self.last_timestamp = None

    def update(self, cumulative, timestamp):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('timestamps must increase')
        if timestamp.date() != self.day:
            if timestamp.hour != 0 or timestamp.minute not in (0, 10):
                raise ValueError('a daily series must start at 00:00 or 00:10')
            self.day = timestamp.date()
            zero = np.where(np.isfinite(cumulative), 0., np.nan)
            self.processor.update(zero, daily_start=True)
        self.last_timestamp = timestamp
        if timestamp.hour == 0 and timestamp.minute == 0:
            return self.processor.update(np.where(np.isfinite(cumulative),0.,np.nan), daily_start=True)
        return self.processor.update(cumulative)


def build_day(raw_root, date):
    import netCDF4
    day = dt.datetime.strptime(date, '%Y%m%d')
    proc = ResetSafeMethod4C()
    detail = np.zeros((144,100,100),np.float32)
    context = np.zeros((144,200,200),np.float16)
    valid = np.zeros((144,200,200),np.uint8)
    # 400 km at native 500m, centered on Seoul crop. +y is north.
    y0,x0 = 1446-400,1048-400
    for i in range(144):
        timestamp=day+dt.timedelta(minutes=10*i)
        with netCDF4.Dataset(Path(raw_root)/f'{timestamp:%Y%m%d%H%M}.nc') as nc:
            nc.set_auto_mask(False)
            a=nc['data'][y0:y0+800,x0:x0+800].astype(np.float32)
        a=np.where((a>=0)&(a<9999),a/10,np.nan)
        rain=proc.update(a,timestamp)
        rain=np.where(rain<=500,rain,np.nan)
        detail[i]=np.nan_to_num(rain[350:450,350:450],nan=0.)
        blocks=rain.reshape(200,4,200,4)
        finite=np.isfinite(blocks);count=finite.sum((1,3))
        context[i]=(np.nan_to_num(blocks,nan=0.).sum((1,3))/np.maximum(count,1)).astype(np.float16)
        valid[i]=np.round(count*255/16).astype(np.uint8)
    return detail,context,valid


def job(args):
    root,out,split,date=args
    target=Path(out)/split/(date+'.npz')
    if target.exists():
        with np.load(target) as z:
            if str(z['version'])!=VERSION:raise ValueError(f'wrong existing version: {target}')
        return date,'exists'
    detail,context,valid=build_day(root,date)
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix(f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp,rain=detail,context=context,valid=valid,version=VERSION,date=date,
                        frame_minutes=np.arange(144)*10,unit='mm/10min',context_km=400)
    os.replace(tmp,target)
    return date,'ok'


def choose(manifest, ntrain, nval):
    import yaml
    m=yaml.safe_load(Path(manifest).read_text());files={int(r['id']):r for r in m['files']}
    rng=np.random.default_rng(27019);chosen={}
    for split,n in [('train',ntrain),('val',nval)]:
        candidates=[files[int(i)] for i in m['splits']['debug_data'][split+'_ids']]
        candidates=[r for r in candidates if r['date'].replace('-','') not in EXCLUDED and r.get('rain_max',0)>=1]
        candidates=sorted(candidates,key=lambda r:r['date'])
        ids=rng.choice(len(candidates),size=min(n,len(candidates)),replace=False)
        chosen[split]=sorted(candidates[int(i)]['date'].replace('-','') for i in ids)
    return chosen


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-root',default='/datasets/rn_day')
    ap.add_argument('--manifest',default='/workspace/project/experiments/model_arch_test/configs/data_manifest_fixed.yaml')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--train-days',type=int,default=48)
    ap.add_argument('--val-days',type=int,default=12);ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args();chosen=choose(a.manifest,a.train_days,a.val_days)
    a.out.mkdir(parents=True,exist_ok=True);selection=a.out/'selection.json'
    metadata=dict(version=VERSION,seed=27019,selection=chosen,criterion_dates_excluded=sorted(EXCLUDED))
    if selection.exists() and json.loads(selection.read_text())!=metadata:
        raise ValueError('selection already exists with different settings; use a new output root')
    selection.write_text(json.dumps(metadata,indent=2))
    jobs=[(a.raw_root,str(a.out),split,date) for split,dates in chosen.items() for date in dates]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(job,jobs)):
            print(i+1,'/',len(jobs),result,flush=True)


if __name__=='__main__':main()
