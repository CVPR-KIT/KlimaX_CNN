"""Measured runtime/device details and validated TC8 outputs; no future NC reads."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
from pyproj import CRS
import torch

from src.nc_input import _source_path
from src.tc9_evaluation import read_ascii

KST = timezone(timedelta(hours=9))
PROTOCOL = 'TC8-daily-prefix-ASC-PNG-v2.0.0'


def now_kst():
    return datetime.now(KST)


def clock_text(stamp):
    return stamp.strftime('%H:%M:%S.')+f'{stamp.microsecond//1000:03d}'


def resolve_device(requested):
    if requested == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda':
        return 'cuda:0'
    return requested


def device_details(selected):
    result = dict(selected=selected,pytorch=torch.__version__,gpu_name=None,
                  total_gib=None,used_gib=None,utilization_percent=None,driver=None)
    smi = shutil.which('nvidia-smi')
    if smi:
        try:
            command=[smi,'--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version',
                     '--format=csv,noheader,nounits']
            answer=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=4)
            rows=list(csv.reader(answer.stdout.decode('utf-8',errors='replace').splitlines()))
            index=int(selected.split(':')[1]) if selected.startswith('cuda:') else 0
            if answer.returncode == 0 and rows:
                row=rows[min(index,len(rows)-1)]
                result.update(gpu_name=row[1].strip(),total_gib=float(row[2])/1024,
                              used_gib=float(row[3])/1024,utilization_percent=float(row[4]),driver=row[5].strip())
        except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):
            pass
    if result['gpu_name'] is None and torch.cuda.is_available():
        index=int(selected.split(':')[1]) if selected.startswith('cuda:') else 0
        props=torch.cuda.get_device_properties(index)
        result.update(gpu_name=props.name,total_gib=props.total_memory/1024**3)
    return result


def device_line(info):
    gpu=info['gpu_name'] or '정보 없음/조회 불가'
    total=f"{info['total_gib']:.1f}" if info['total_gib'] is not None else 'N/A'
    used=f"{info['used_gib']:.1f}" if info['used_gib'] is not None else 'N/A'
    usage=f"{info['utilization_percent']:.0f}%" if info['utilization_percent'] is not None else 'N/A'
    return (f"[장비확인] GPU: {gpu} ({total} GiB) | 사용장치: {info['selected']} | "
            f"사용률: {usage} | 메모리: {used}/{total} GiB | 드라이버: {info['driver'] or 'N/A'} | "
            f"PyTorch: {info['pytorch']} (조회 시점 스냅샷)")


def inspect_observations(root,first):
    anchor=first-timedelta(minutes=10)
    count=(anchor.hour*60+anchor.minute)//10
    if count < 6:
        raise ValueError('최소 첫 예측 시각은 01:10입니다.')
    day=anchor.replace(hour=0,minute=0,second=0,microsecond=0)
    sources=[_source_path(root,day+timedelta(minutes=10*i)) for i in range(1,count+1)]
    return dict(prefix_count=len(sources),window_start=anchor-timedelta(minutes=60),
                window_nc_count=min(7,count),virtual_midnight_baseline=count==6)


def validate_outputs(ascii_dir,png_dir,first):
    expected_asc=[f'pred_physical_t{i+6:04d}_EPSG3857_1km.asc' for i in range(18)]
    expected_prj=[name.replace('.asc','.prj') for name in expected_asc]
    expected_png=[f"pred_corrected_{first+timedelta(minutes=10*i):%Y%m%d_%H%M}_lead{10*(i+1):03d}min.png" for i in range(18)]
    for folder,pattern,expected in [(ascii_dir,'*.asc',expected_asc),(ascii_dir,'*.prj',expected_prj),(png_dir,'*.png',expected_png)]:
        if sorted(path.name for path in folder.glob(pattern)) != sorted(expected):
            raise ValueError('필수 출력 누락/중복/예상 외 이름: '+str(folder)+' '+pattern)
    alignment=(ascii_dir.parent/'time_alignment.txt').read_text(encoding='utf-8').splitlines()
    header0=None
    for index,(asc,prj,png) in enumerate(zip(expected_asc,expected_prj,expected_png)):
        stamp=(first+timedelta(minutes=10*index)).isoformat(timespec='minutes')
        if alignment.count(f'{asc} {stamp} lead_min={10*(index+1)}') != 1:
            raise ValueError('출력 시각/순서/중복 검수 실패')
        header,data=read_ascii(ascii_dir/asc)
        if not np.isfinite(np.loadtxt(ascii_dir/asc,skiprows=6)).all():
            raise ValueError('ASC에 비정상 NaN/Inf 수치가 있습니다.')
        if header0 is not None and header!=header0:
            raise ValueError('ASC 시계열 격자 불일치')
        header0=header
        if np.isinf(data).any() or (data[np.isfinite(data)] < 0).any():
            raise ValueError('ASC 예측값 검수 실패')
        if CRS.from_wkt((ascii_dir/prj).read_text(encoding='utf-8')).to_epsg()!=3857:
            raise ValueError('PRJ 좌표계 검수 실패')
        with Image.open(png_dir/png) as image:
            if image.format!='PNG' or image.size!=(940,830):
                raise ValueError('PNG 형식/크기 검수 실패')
            image.verify()
    return dict(asc_count=18,prj_count=18,png_count=18,missing=0,duplicates=0,
                chronological=True,readable=True,generation_percent=100.,valid_frames=18)


def record_tc8(output,signature,seconds,start,end):
    key=hashlib.sha256(json.dumps(signature,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    record=dict(protocol=PROTOCOL,signature=signature,signature_sha256=key,seconds=seconds,
                started_at=start.isoformat(),ended_at=end.isoformat(),output=str(output))
    (output/'TC8.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    eligible=[]
    for folder in output.parent.glob(signature.get('mode','v2')+'_*'):
        path=folder/'TC8.json'
        if not path.is_file():
            continue
        try:
            prior=json.loads(path.read_text(encoding='utf-8'))
            if prior.get('signature_sha256')==key and prior.get('protocol')==PROTOCOL and isinstance(prior.get('seconds'),(int,float)) and np.isfinite(prior['seconds']) and prior['seconds']>=0:
                eligible.append(prior)
        except (OSError,ValueError,KeyError):
            pass
    eligible.sort(key=lambda item:item['ended_at'])
    recent=eligible[-3:]
    complete=len(recent)==3
    summary=dict(eligible_count=len(eligible),used_count=len(recent),required_runs=3,complete=complete,
                 mean_seconds=sum(r['seconds'] for r in recent)/3 if complete else None,
                 threshold_seconds=300.,passed=sum(r['seconds'] for r in recent)/3<=300 if complete else None,
                 runs=[dict(output=r['output'],seconds=r['seconds'],ended_at=r['ended_at']) for r in recent])
    (output/'TC8_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary
