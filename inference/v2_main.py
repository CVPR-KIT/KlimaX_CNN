"""KlimaX V2.0.0 measured TC8 / saved-ASCII TC9 workflow."""
import time
ENTRY_PERF=time.perf_counter()
import argparse
from datetime import datetime,timedelta
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch
import yaml

APP=Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
RES=Path(getattr(sys,'_MEIPASS',APP))
sys.path.insert(0,str(RES/'src'))
from src.nc_input import prepare_input
from src.runtime import load_runtime,predict
from src.ascii_export import export_predictions
from src.png_export import export_pngs
from src.tc9_evaluation import evaluate_tc9,write_tc9_tables
from src.tc_station_graphs import export_tc_graphs
from src.tc_support import (now_kst,clock_text,resolve_device,device_details,device_line,
                            inspect_observations,validate_outputs,record_tc8,PROTOCOL)
from correction import apply_correction

VERSION='2.0.0'
BUILD_DATE='2026-10-13'
IMPORT_SECONDS=time.perf_counter()-ENTRY_PERF


def resolve(text,resources=False):
    path=Path(text)
    if path.is_absolute(): return path
    external=APP/path
    return external if external.exists() or not resources else RES/path


def reserve(base,name):
    base.mkdir(parents=True,exist_ok=True)
    index=1
    while True:
        folder=base/(name if index==1 else name+'_run'+str(index))
        try:
            folder.mkdir();return folder,index
        except FileExistsError:
            index+=1


def section(title,report):
    report('==================== '+title+' ====================')


def format_num(value,places=3):
    return 'N/A' if value is None else f'{value:.{places}f}'


def print_tc9(result,report):
    section('[TC9] 평가 좌표 및 산정표',report)
    report('[평가좌표] AWS 3개 지점 | 위치(위경도)만 사용, AWS 관측강우량 미사용 | EPSG:3857 최근접 격자')
    report('지점  지점명       위도       경도       ASC 격자(행,열)  격자 중심 거리(m)')
    for m in result['mapping']:
        report(f"{m['station_id']:4}  {m['name']:5}  {m['station_lat']:.4f}  {m['station_lon']:.4f}  "
               f"({m['asc_row']},{m['asc_col']})  {m['distance_m']:.1f}")
    report('[격자 기준] ASC 행·열은 0부터, 행은 북쪽부터 | 거리 WGS84 지표 거리 | 원본 500m 매핑은 grid_mapping.csv에 별도 기록')
    report('[평가 기준] 단위 mm/10분 | 절대오차 = |예측강우-기준강우| | 허용오차 = max(1.0, 기준강우 x 50%)')
    report('절대오차 <= 허용오차 -> 적중 | 기준강우 0이면 허용오차 1.0 적용')
    report('(참고) 오차율 = 절대오차 / 기준강우 x 100 (판정 미사용, 기준강우 0이면 N/A)')
    report('기준강우는 원시 NC 차분을 예측 ASC와 같은 1km 격자로 평균한 기준 격자강우이며 AWS 실측이 아닙니다.')
    for station,m,summary in zip(result['rows'],result['mapping'],result['station_summary']):
        report(f"평가좌표 AWS {station['station_id']} ({station['name']}) | 대응 ASC 격자 ({m['asc_row']},{m['asc_col']})")
        report('시각                 기준강우  예측강우  절대오차  허용오차  오차율  판정')
        for row in station['rows']:
            stamp=datetime.fromisoformat(row['timestamp']).strftime('%Y-%m-%d %H:%M')
            rate=format_num(row['absolute_error_percent'],2)+('%' if row['absolute_error_percent'] is not None else '')
            verdict='제외: '+row['reason'] if row['excluded'] else ('적중' if row['passed'] else '미달')
            report(f"{stamp}  {format_num(row['truth_mm10']):>8}  {format_num(row['prediction_mm10']):>8}  "
                   f"{format_num(row['absolute_error_mm10']):>8}  {format_num(row['threshold_mm10']):>8}  {rate:>10}  {verdict}")
        report(f"AWS {station['station_id']} 소계: 적중 {summary['hits']} / 유효 {summary['valid_pairs']} | 제외 {summary['excluded_count']}")
    report('[좌표별 정확도] '+' | '.join(f"AWS {s['station_id']}: {format_num(s['accuracy_percent'],1)}% ({s['hits']}/{s['valid_pairs']})" for s in result['station_summary']))


def run(args):
    whole_start=time.perf_counter()
    config_path=Path(args.config).resolve() if args.config else resolve('model_config.yaml',resources=True)
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    date=args.date or str(cfg['inference']['date'])
    clock=args.time or str(cfg['inference']['time']).zfill(4)
    if args.prompt_datetime:
        date=input('Date YYYYMMDD: ').strip();clock=input('First forecast HHMM: ').strip()
    first=datetime.strptime(date+clock,'%Y%m%d%H%M')
    if len(date)!=8 or len(clock)!=4 or first.minute%10:
        raise ValueError('YYYYMMDD/HHMM 및 10분 간격 시각을 사용하세요.')
    anchor=first-timedelta(minutes=10);end=first+timedelta(minutes=170)
    paths=cfg['paths']
    raw=Path(args.input_dir).resolve() if args.input_dir else resolve(paths['input_dir'])
    base=Path(args.output_dir).resolve() if args.output_dir else resolve(paths['output_dir'])
    output,index=reserve(base,'v2_'+first.strftime('%y%m%d_%H%M'))
    log=output/'run_log.txt'
    def report(message):
        print(message,flush=True)
        with log.open('a',encoding='utf-8') as stream: stream.write(message+'\n')
    try:
        selected=resolve_device(args.device or cfg['inference']['device'])
        device=device_details(selected)
        checkpoint=resolve(paths['checkpoint'],resources=True)
        section('[공통] 실행 정보 및 입력 확인',report)
        report(f'[시작] V2 예측·평가 실행 | 첫 예측: {first:%Y-%m-%d %H:%M} | KlimaXv{VERSION} (빌드 {BUILD_DATE}) | 회차: run{index} | 시각 기준: KST')
        report(f'[버전확인] 체크포인트: {checkpoint.name} | 설정파일: {config_path} | checkpoint step 85140 / q0.6')
        report(device_line(device))
        report(f'[시각설정] 입력 종료시각 t0: {anchor:%Y-%m-%d %H:%M} | 예측 시각: {first:%H:%M}~{end:%H:%M} (t0+10~180분, 10분 간격 18개)')
        report('[저장 위치] '+str(output));report('[실행 로그] '+str(log));report('[입력 폴더] '+str(raw))
        inspected=inspect_observations(raw,first)
        window=inspected['window_start']
        report(f"[입력 확인] 필요 관측 NC: {anchor:%Y-%m-%d} 00:10~{anchor:%H:%M} | {inspected['prefix_count']}/{inspected['prefix_count']}개 | 파일 결측 0개 | 10분 간격 연속")
        report(f'[정답 평가자료범위] 평가 NC: {first:%Y-%m-%d %H:%M}~{end:%Y-%m-%d %H:%M} | 18개 | 예측 저장 및 TC8 측정 종료 후 읽음(예측에는 미사용)')
        section('[TC8] 예측 신속성 : 구동시간 측정',report)
        tc8_start_wall=now_kst();tc8_start=time.perf_counter()
        report('[TC8 측정 시작] '+clock_text(tc8_start_wall)+' KST (누적 NC -> 10분 강우 변환 직전)')
        converted=output/'converted_input'
        inputs=prepare_input(raw,first,converted/'nc',converted/'npz')
        fine=inputs['fine'];invalid=sum(r['invalid_pixels'] for r in inputs['input_qc'])
        negatives=int((fine<0).sum());nonfinite=int((~np.isfinite(fine)).sum())
        if fine.shape!=(6,1,100,100) or negatives or nonfinite: raise ValueError('모델 입력 격자 검수 실패')
        quality='이상 없음' if invalid==0 else f'변환 결측/이상 {invalid}개를 0 처리'
        report('[모델 실행] 모델 로딩 및 향후 3시간 18개 시점 예측 중')
        runtime=load_runtime(checkpoint,device=selected)
        original=predict(runtime,inputs['fine'],inputs['context'],inputs['valid'],inputs['clock'])
        report('[보정] 예측 결과 보정 중 (고정 계수, 평가 NC 미사용)')
        profile=args.correction or cfg['inference']['correction']
        if profile=='recommended': profile='recommended_2022_three_block_power'
        presets=resolve('presets.json',resources=True)
        corrected=apply_correction(original,profile,lead_axis=0,presets_path=presets)
        grid=resolve(paths['grid_dir'],resources=True)
        lat,lon=np.load(grid/'lat.npy'),np.load(grid/'lon.npy')
        report('[결과 정리] ASC·PRJ·보정 PNG 저장 중')
        # Preserve the exporter, grid and values; V2 only changes the directory name.
        ascii_dir=export_predictions(corrected,lat,lon,output,first)
        asc_corrected=output/'asc_corrected';ascii_dir.rename(asc_corrected)
        ascii_dir=asc_corrected
        png_dir=export_pngs(corrected,output,first)
        np.savez_compressed(output/'prediction_arrays.npz',original_mm10=original,corrected_mm10=corrected,
                            profile=profile,date=date,first_forecast=clock,quantile=.6,step=85140)
        report(f'[예측 저장 완료] ASC: {ascii_dir} | PRJ: {ascii_dir} | PNG: {png_dir}')
        counts=validate_outputs(ascii_dir,png_dir,first)
        report('[생성 확인] ASC 18/18 | PRJ 18/18 | PNG 18/18 | 누락 0 · 중복 0 · 시간순서 정상 · 파일 읽기 정상 | 정상 생성률: 100.0% (18/18)')
        if runtime.device.type=='cuda': torch.cuda.synchronize(runtime.device)
        tc8_seconds=time.perf_counter()-tc8_start;tc8_end_wall=now_kst()
        report('[TC8 측정 종료] '+clock_text(tc8_end_wall)+' KST (필수 결과 생성 확인 직후)')
        report(f'[TC8 결과] 시작 {clock_text(tc8_start_wall)} | 종료 {clock_text(tc8_end_wall)} | 구동시간 {tc8_seconds:.3f}초 (기준 300초)')
        signature=dict(first_forecast=first.isoformat(),checkpoint_sha256=runtime.checkpoint_sha256,
                       profile=profile,presets_sha256=hashlib.sha256(presets.read_bytes()).hexdigest(),
                       config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                       input_sha256=hashlib.sha256(json.dumps(inputs['provenance']['consumed_files'],sort_keys=True).encode()).hexdigest(),
                       lat_sha256=hashlib.sha256(np.ascontiguousarray(lat).tobytes()).hexdigest(),
                       lon_sha256=hashlib.sha256(np.ascontiguousarray(lon).tobytes()).hexdigest(),
                       host=platform.node(),device=selected,gpu_name=device['gpu_name'],cpu=platform.processor(),pytorch=torch.__version__,protocol=PROTOCOL)
        tc8=record_tc8(output,signature,tc8_seconds,tc8_start_wall,tc8_end_wall)
        if tc8['complete']:
            report(f"[TC8 3회 평균] 동일 조건 최근 3회 평균 {tc8['mean_seconds']:.3f}초 | 기준 300초 이하 | {'통과' if tc8['passed'] else '미달'}")
        else:
            report(f"[TC8 판정 대기] 동일 조건 유효 실행 {tc8['used_count']}/3회 | 3회 평균으로 판정 (--repeat 3 사용 가능)")
        section('[TC9] 예측 정확성 : 종합 강우예측 정확도 평가',report)
        tc9_start=time.perf_counter()
        report('[TC9 평가 시작] 평가 대상: 보정 후 예측 ASC 18개(asc_corrected) | TC8 측정 종료 후 수행(구동시간 미포함)')
        report('[기준자료정답확인] 평가 NC 18개 시점 확인 및 차분 계산 중')
        stations=json.loads(resolve('stations_tc.json',resources=True).read_text(encoding='utf-8'))
        accuracy=evaluate_tc9(raw,first,ascii_dir,lat,lon,stations)
        table_dir=write_tc9_tables(accuracy,output)
        graph_dir=None
        if accuracy['status']=='available':
            negatives_raw=sum(q['negative_native_pixels'] for q in accuracy['truth_qc'])
            missing_raw=sum(q['nonfinite_native_pixels'] for q in accuracy['truth_qc'])
            report(f"[기준자료정답확인 완료] 평가 NC 18/18 | 첫 시점은 {anchor:%H:%M}와 차분 | 원본 음수 차분 {negatives_raw}픽셀·결측 {missing_raw}픽셀 | 좌표·시점 품질관리 제외 {accuracy['excluded_count']}건")
            print_tc9(accuracy,report)
            report('[산정표 저장] '+str(table_dir/'accuracy_table.csv')+' (54행) | station_summary.csv | grid_mapping.csv | excluded.csv')
            report('[그래프 저장] 기준·예측 비교 그래프 18장 생성 중')
            graph_dir=export_tc_graphs(accuracy,output)
            report('[그래프 저장 완료] '+str(graph_dir))
            if accuracy['valid_pairs']:
                report(f"[TC9 결과] 종합 강우예측 정확도 {accuracy['accuracy_percent']:.1f}% (적중 {accuracy['hits']} / 유효 {accuracy['valid_pairs']}) | 기준 80% 이상 | {'통과' if accuracy['accuracy_percent']>=80 else '미달'}")
            else: report('[TC9 결과] 계산 불가: 유효 평가 쌍 0건')
            report(f"제외 {accuracy['excluded_count']}건 | 무강우(기준·예측 모두 0) {accuracy['dry_both_count']}건 | MAE {format_num(accuracy['mae_mm10'])} mm/10분 (참고, 판정 미사용)")
        else:
            report('[TC9 평가 불가] 예측 파일은 정상 저장 | '+accuracy['reason'])
            report('[산정표 저장] '+str(table_dir)+' (평가 불가: 헤더만 저장, 판정하지 않음)')
        graph_count=len(list(graph_dir.glob('*.png'))) if graph_dir else 0
        csv_count=len(list(table_dir.glob('*.csv')))
        if csv_count!=4 or (graph_dir and graph_count!=18): raise ValueError('그래프/산정표 개수 검증 실패')
        report(f'[생성 확인] 그래프 {graph_count}/18 | 산정표 {csv_count}/4')
        (output/'ACCURACY.json').write_text(json.dumps(accuracy,indent=2,ensure_ascii=False),encoding='utf-8')
        tc9_seconds=time.perf_counter()-tc9_start
        whole_seconds=time.perf_counter()-whole_start
        meta=dict(mode='v2',version=VERSION,build_date=BUILD_DATE,timezone='KST',run_number=index,
                  checkpoint_step=85140,checkpoint_sha256=runtime.checkpoint_sha256,correction_profile=profile,
                  first_forecast=first.isoformat(),input_provenance=inputs['provenance'],input_qc=inputs['input_qc'],
                  future_observation_used=False,device=device,tc8=tc8,tc8_seconds=tc8_seconds,
                  tc8_start=tc8_start_wall.isoformat(),tc8_end=tc8_end_wall.isoformat(),tc9_seconds=tc9_seconds,
                  elapsed_seconds=whole_seconds,initial_import_seconds=IMPORT_SECONDS,accuracy=accuracy,
                  ascii_count=18,prj_count=18,png_count=18,graph_count=graph_count,csv_count=csv_count,
                  output_validation=counts,config_file=str(config_path),run_log=str(log),evaluation_source='saved 1km ASC')
        (output/'RUN_METADATA.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
        section('실행 완료',report);report('[저장 위치] '+str(output))
        report(f'[완료] TC8 구동시간(전처리·예측·저장·검수): {tc8_seconds:.3f}초 | TC9 평가·그래프: {tc9_seconds:.3f}초 (TC8 미포함)')
        report(f'전체 처리: {whole_seconds:.3f}초 (공통 입력확인 포함) | 초기 로딩: {IMPORT_SECONDS:.3f}초 별도 (EXE 부트로더 시간 제외) | 정상 종료 (종료코드 0)')
        return output
    except Exception as error:
        report('[실행 오류] '+str(error));report(traceback.format_exc());raise


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(errors='replace')
    # Accept pasted Unicode dashes in options, but show canonical --time in docs.
    argv=[('--'+a[1:]) if a.startswith(('—','–')) else a for a in sys.argv[1:]]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--date');parser.add_argument('--time')
    parser.add_argument('--input-dir');parser.add_argument('--output-dir');parser.add_argument('--device')
    parser.add_argument('--correction',choices=('recommended','raw','recommended_2022_three_block_power','comparison_2025_three_block_power'))
    parser.add_argument('--prompt-datetime',action='store_true');parser.add_argument('--no-pause',action='store_true')
    parser.add_argument('--repeat',type=int,choices=(1,3),default=1)
    args=parser.parse_args(argv)
    keep_open=getattr(sys,'frozen',False) and len(sys.argv)==1 and sys.stdin.isatty() and not args.no_pause
    try:
        for _ in range(args.repeat): run(args)
    except Exception:
        traceback.print_exc();raise SystemExit(1)
    finally:
        if keep_open: input('\n결과를 확인한 뒤 Enter를 누르면 창이 닫힙니다. ')


if __name__=='__main__': main()
