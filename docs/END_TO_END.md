# 원본 데이터에서 학습·추론까지

이 저장소의 코드만 내려받고 **지원 형식의 원본 NC 데이터만 별도로 준비하면**, 데이터 전처리·검사, 새 모델 학습, 체크포인트 선정, V1 예측과 V2 평가까지 실행할 수 있습니다. 아래 명령은 저장소 루트에서 실행합니다. 데이터·생성 NPZ·학습 가중치·결과는 Git에 올리지 않습니다.

새 범용 경로는 `prepare_data.py` → `train_pipeline.py` → 추론 코드입니다. 과거 `train_v45.py`의 고정 규모 학습 경로도 보존합니다. 새 경로는 동일 모델 구조와 V45 손실을 사용하지만, 표본 추출·학습 예산·checkpoint 선정 집계가 과거 실험의 완전한 재현을 뜻하지 않습니다. 짧은 실행 검증은 예측 성능 검증이 아닙니다.

## 1. 설치

새 경로의 실행 검증 환경은 Windows, Python 3.11, PyTorch 2.13.0+cpu입니다. 학습은 CPU와 CUDA를 지원하며 큰 데이터 본학습에는 CUDA GPU를 권장합니다. CUDA 경로는 이번 CPU 실행 검증과 별도로 확인해야 합니다. 격리한 가상환경을 사용합니다.

```powershell
git clone https://github.com/CVPR-KIT/KlimaX_CNN.git
cd KlimaX_CNN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-pipeline.txt
python pipeline.py --help
```

Linux에서는 가상환경 활성화만 `source .venv/bin/activate`로 바꿉니다. CUDA 사용 시 CPU wheel 대신 장비에 맞는 CUDA PyTorch wheel을 먼저 설치합니다. 과거 고정 학습 runner의 환경은 [학습 README](../training/README.md)에 별도로 기록했습니다.

## 2. 원본 NC 준비

```text
data/raw_nc/
  20220802/
    202208020010.nc
    ...
    202208022350.nc
  20220809/
    202208090010.nc
    ...
    202208092350.nc
```

하루에 **00:10~23:50의 143개 누적 NC**가 필요합니다. 00:00은 읽지 않으며 모델에 사용하지 않는 구조적 0 프레임을 생성합니다. 평면 폴더와 날짜 하위 폴더를 지원하지만 같은 시각 파일이 중복되면 오류입니다.

지원 자료는 KMA `SFC_GRID_rn_day_` 일누적 정수 `data[2049,2049]`, 500m LCC 원본 격자입니다. 내부 시각·제품명·격자 속성·`data_scale=10`·transform을 검사합니다. 기존 제품의 `mm/h` 표기는 일누적 제품 메타데이터와 함께 명시적으로 허용하며, 데이터를 10으로 나눠 누적 mm로 해석합니다. 일반적인 mm/h NC, 다른 격자, 이미 차분된 NC를 넣을 수는 없습니다.

학습 날짜와 검증 날짜는 겹치면 안 됩니다. 아래 두 날짜는 **명령 형식과 실행 검증 예시**입니다. 실제 본학습에는 별도의 학습/검증/최종 평가 날짜를 정하고 충분한 강우 사례를 준비합니다.

## 3. 전처리와 데이터 검사

```powershell
python pipeline.py prepare --raw-dir data/raw_nc --out work/dataset --train-dates 20220802 --val-dates 20220809
```

여러 날짜는 `--train-dates 20220801,20220802,20220803`처럼 쉼표로 지정합니다. 출력 경로는 새 경로여야 합니다. 파일 확인부터 실패하면 학습 가능한 자료로 인정하지 않고, 전처리 도중 실패한 부분 출력도 통과한 자료로 인정하지 않습니다.

처리 순서:

1. 모든 필요 NC의 존재·중복·내부 시각·제품·격자를 검사합니다.
2. 날짜별 누적 초기화, 감소 이월, 약한 비 제거, 총량 보존 공간 처리를 하는 Method4C를 적용합니다. 미래 자료를 과거 입력 보정에 사용하지 않습니다.
3. 상세 100×100과 광역 200×200 강우 및 시점별 유효도를 만듭니다. 결측 뒤 여러 구간의 누적 증가를 한 구간의 유효한 10분 강우로 바꾸지 않도록 그 구간도 마스킹합니다.
4. 6개 입력+18개 정답의 window를 검사합니다. 프레임 0과 자정 경계를 넘는 window는 제외합니다. 기본 최소 상세 유효 비율 95%, 광역 관측 비율 50%를 요구합니다. 광역의 구조적 영역 밖 결측은 마스크로 유지하며, 모든 픽셀이 관측됐다고 취급하지 않습니다.
5. 실제 검사 결과·제외 window·원본 hash를 기록한 `manifest.json`과 균일 가중치의 `index.jsonl`을 생성합니다. 과거 birth sampling index를 재현했다고 주장하지 않습니다.

상세 target 손실 유효도는 기존 loader에서 **해당 2km 셀이 완전히 관측된 경우만** 허용하는 보수적 마스크를 사용합니다. 저장 배열에서 결측을 0으로 채우는 것과 관측된 무강우는 다릅니다. 설정을 변경하면 검사 기준도 manifest에 기록됩니다.

```text
work/dataset/
  train/20220802.npz      rain/context/valid/native_valid + 단위·버전
  val/20220809.npz
  grid/lat.npy
  grid/lon.npy
  manifest.json          통과·제외 window, 품질관리 기준, 원본/NPZ hash
  index.jsonl            학습 window별 균일 가중치
  dataset_info.json      원본 경로, 좌표 기준, 소스와 생성 자료 hash
```

### 좌표를 준비하는 방법

원본 헤더에서 KMA 구면 LCC 좌표를 생성하므로 원본 NC만으로 전체 경로를 실행할 수 있습니다. 이 좌표는 헤더 기반 계산이며 공식 `sfc_grid_latlon.nc`에서 얻은 좌표와 미세한 차이가 있을 수 있습니다. **기존 배포 ASC 위치를 정확히 재현하려면 기존과 동일한 좌표를 제공**합니다.

```powershell
python pipeline.py prepare --raw-dir data/raw_nc --out work/dataset_official --train-dates 20220802 --val-dates 20220809 --grid-source-dir data/official_grid
```

`data/official_grid`에는 `lat.npy`와 `lon.npy`의 실제 상세 격자 좌표를 넣습니다. 100×100 크기·유한 값·방향·지원 지역 일치를 검사하며 사용한 파일 hash를 보존합니다. 임의의 좌표 파일로 다른 지역 예측을 대신할 수는 없습니다.

## 4. 짧은 학습 실행 확인

```powershell
python pipeline.py train --data-dir work/dataset --out work/smoke_model --epochs 1 --batch-size 1 --max-steps 1 --smoke-val-windows 2 --device cpu
```

가중치 갱신 1회 후 검증 split의 **앞 2개 승인 window만** 평가하고 checkpoint를 저장합니다. `--smoke-val-windows`는 `--max-steps`를 지정한 짧은 실행 확인에서만 허용하며 제한된 검증 범위를 결과에 기록합니다. 이 옵션을 생략하면 모든 검증 window를 평가합니다. 비어 있는 검증 또는 모든 CSI 항목의 강우·예측 사건이 없는 경우 모델 선정 불가로 실패합니다. 이 명령은 코드·데이터 연결 확인용이며 모델 성능을 주장하기 위한 학습이 아닙니다.

## 5. 본학습

```powershell
python pipeline.py train --data-dir work/dataset --out work/model_run1 --epochs 20 --batch-size 8 --device cuda:0 --warmup-steps 1000
```

CPU에서도 `--device cpu`로 실행할 수 있습니다. 메모리가 부족하면 batch size를 줄입니다. batch·epoch·seed·학습률·warmup·worker를 설정할 수 있으며, 과거 90,823개 학습 window 검사에 묶이지 않습니다. 현재 새 runner는 **scratch 학습**입니다. 중단 학습 재개와 기존 모델 미세조정은 아직 제공하지 않습니다.

AdamW와 gradient clipping, warmup/cosine 학습률을 사용합니다. 승인된 모든 학습 window를 epoch마다 섞으며 가중 반복 표본은 사용하지 않습니다. validation은 섞지 않습니다. 새 runner의 checkpoint 선정 지표는 상세 q0.6의 pooled CSI 0.1·CSI 0.5·+120~180분 CSI 0.5 평균입니다. **3개 지점 TC9 적중률과 다른 지표**입니다.

```text
work/model_run1/
  best.ckpt                 검증 지표로 선정한 가중치
  last.ckpt                 마지막 가중치
  model_contract.json       가중치 hash·step·구조·전처리·좌표 hash
  inference_config.yaml     이 모델과 원본 데이터로 추론하는 설정
  training_config.json      데이터·소스 hash, 설정, window 수
  history.json              epoch별 손실·검증 지표
  TRAINING_RESULT.json      총 갱신 수와 선택된 step 구분
```

`best.ckpt`와 `last.ckpt`는 optimizer 상태를 포함한 재개 파일이 아니라 **추론용 가중치 checkpoint**입니다. `TRAINING_RESULT.json`의 PASS는 실행·저장 성공이며 독립 예측 품질의 통과 판정이 아닙니다.

## 6. 새 모델로 추론

```powershell
python pipeline.py predict --config work/model_run1/inference_config.yaml --date 20220809 --time 1600
python pipeline.py evaluate --config work/model_run1/inference_config.yaml --date 20220809 --time 1600
```

`--time 1600`은 **첫 예측 시각**입니다. 입력은 15:00~15:50, 예측은 16:00~18:50입니다. 누적 감소 이력을 반영하려면 입력 종료까지 당일 00:10부터의 NC가 필요합니다. V1은 미래 NC를 읽지 않고 예측만 저장하며 V2는 예측 저장·TC8 종료 후 미래 NC를 평가에 사용합니다.

생성 설정에는 새 checkpoint와 model contract의 절대 경로가 연결됩니다. 다른 컴퓨터로 이동하면 이 경로들을 수정하거나 다음 명시 옵션을 사용합니다.

```powershell
python pipeline.py predict --checkpoint work/model_run1/best.ckpt --model-contract work/model_run1/model_contract.json --grid-dir work/dataset/grid --input-dir data/raw_nc --output-dir work/predictions --date 20220809 --time 1600 --correction raw
```

새 모델은 **보정 없음(`raw`)이 기본**입니다. 기존 두 사례에 맞춘 공통 power 계수가 새 가중치에도 적합하다고 가정하지 않습니다. 입력 Method4C는 유지됩니다. 보정이 필요하면 별도의 보정 선정 자료와 최종 평가 자료로 검증합니다.

모델 계약은 hash·step·지원 구조·전처리 버전·좌표 hash를 검사합니다. 기본 배포 모델의 고정 hash 검사는 유지되며 계약 없이 다른 checkpoint만 넣으면 거부됩니다. 새 계약은 사용자가 학습한 파일을 명시적으로 선택하기 위한 기능이며 정확도를 인증하는 기능은 아닙니다.

## 7. 출력과 최종 평가

- V1: ASC 18개·PRJ 18개·예측 PNG 18개, 실행 로그와 메타데이터.
- V2: V1 결과 + 3지점 기준/예측 그래프 18장, 평가 CSV 4개와 정확도 기록.
- 내부·학습 단위는 mm/10분, 저장 ASC·지도 표시 단위는 mm/h입니다.
- TC9 정답은 AWS 실측이 아니라 원시 누적 NC 차분을 같은 ASC 격자로 평균한 값입니다. 다른 지점 자동 평가는 아직 고정 3지점 코드 확장이 필요합니다.
- 학습·검증에 사용하지 않은 날짜를 최종 평가로 사용하고 무강우 제외·강우 강도별 성능·지도 탐지 성능을 함께 확인합니다. 이 문서의 두 날짜 예시는 최종 성능 검증용 분할이 아닙니다.

이번 변경은 **Git 소스 경로**에 적용됐습니다. 기존 배포 ZIP의 EXE에는 새 학습·계약 선택 기능이 추가되지 않았습니다. EXE가 필요하면 [README 빌드 안내](../README.md)에 따라 최신 소스로 재빌드합니다.

## 8. 재현 검사

```powershell
python training/test_prepare_data.py
python training/test_train_pipeline.py
cd inference
python -m unittest discover -s src -p "test_*.py" -v
```

전처리 검사는 결측·중복·시간·단위·감소 이월과 마스크를 확인하고, 학습 검사는 실제 역전파·가중치 변화·검증 분리·저장/복원·새 모델 추론을 확인합니다. 실제 NC 실행 증거는 [PIPELINE_VALIDATION.json](PIPELINE_VALIDATION.json)에 요약합니다. 원본 자료와 생성 가중치는 기록에 포함하지 않습니다.
