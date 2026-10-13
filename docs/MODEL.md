# V45 모델 구조와 시간 조건

V45는 과거 1시간의 강우 관측으로 미래 3시간을 예측하는 `MotionCoreNowcaster`입니다. CNN으로 공간 특징과 이동을 추정하고, ConvGRU로 광역 강우와 강우 중심 상태를 전개합니다. 이 구현은 사전학습 ClimaX Transformer를 미세조정한 모델이 아니며, 처음부터 학습한 강우 예측 모델입니다. 실행 프로그램의 정확한 생성 설정에서 파라미터 수는 **446,047개**입니다.

구조의 구현은 [추론용 motion_core_arch.py](../inference/src/motion_core_arch.py), 체크포인트 생성·검증과 출력 선택은 [runtime.py](../inference/src/runtime.py)에 있습니다. [학습용 구조 파일](../training/source/ClimaX/src/climax/regional_forecast/motion_core_arch.py)도 같은 모델 연산을 구현합니다.

## 관측 입력과 공간 범위

`B`는 배치 크기입니다. 모든 강우량 입력과 모델 출력의 물리 단위는 **mm/10분**입니다.

| 입력 | shape | 해상도와 범위 | 역할 |
|---|---|---|---|
| `fine` | `[B,6,1,100,100]` | 500m, 50×50km | 상세 강우 관측 |
| `context` | `[B,6,1,200,200]` | 2km, 400×400km | 상세 영역 주변의 광역 강우 관측 |
| `valid` | `[B,6,1,200,200]` | 광역 격자와 동일 | 관측 유효 픽셀 비율, 0~1 |
| `clock` | `[B,19,4]` | 기준 시각 1개 + 미래 18개 | 알려진 달력·시각 조건 |

기준 시각을 `t0`라 하면 관측은 `t0-50분`부터 `t0`까지 10분 간격 6장이고, 예측은 `t0+10분`부터 `t0+180분`까지 18장입니다. 예를 들어 첫 예측이 20:10이면 입력 시각은 19:10~20:00, 예측 시각은 20:10~23:00입니다.

광역 격자는 native 500m의 800×800 영역을 4×4 평균하여 구성하며, 상세 영역은 그 native 영역의 `[350:450,350:450]`입니다. 원시 NC의 행 인덱스는 북쪽으로 증가합니다. 원시 누적 NC의 입력 준비는 [nc_input.py](../inference/src/nc_input.py)와 [preprocess_v27.py](../inference/src/preprocess_v27.py)에 있습니다. **모델이 사용하는 관측은 마지막 6장**이지만, 현재 Method4C 전처리는 음수 차분의 이월 상태를 재현하기 위해 당일 00:10부터 관측 종료까지의 누적 NC를 읽습니다. 마지막 6개의 누적 파일만으로 같은 입력을 만드는 계약은 아닙니다.

## 절대 시간 조건

`clock_features()`는 윤년을 반영한 연중 위상과 하루 내 시각 위상에 각각 sin/cos를 적용합니다. 네 값의 순서는 연중 sin, 연중 cos, 하루 내 sin, 하루 내 cos입니다. 시간대 보정이나 과거 시간 차이의 학습된 추정이 아니라 주어진 timestamp의 달력 정보를 사용합니다.

`clock[:,0]`은 `t0`, `clock[:,1:19]`는 미래 18시각입니다. 각 rollout 단계 `j`의 조건은 다음과 같습니다.

```text
조건[j] = lead_embed[j] + clock_proj(clock[:,j+1])
```

단계별 학습 임베딩과 해당 미래 시각의 절대 시간 특성을 결합하여 광역·강우 중심 recurrent 상태와 상세 decoder에 전달합니다. `clock`의 기준 시각 행은 배열 계약에 포함되지만, 현재 `forward()`의 단계 조건 계산은 미래 행 `j+1`을 직접 사용합니다. 미래 강우량이나 미래 정답 NC는 모델 입력에 포함하지 않습니다.

## 관측 정렬과 미래 상태 전개

1. **관측 검사와 이동 추정:** `_observe()`는 광역 coverage가 0.999 이상인 관측을 사용합니다. 상세 유효성은 해당 광역 셀의 coverage에서 보수적으로 전달하므로 native 상세 결측 마스크를 독립적으로 복구한 값은 아닙니다. 광역 강우의 `log1p`와 유효성을 이용해 인접 관측 쌍의 정방향·역방향 이동을 CNN으로 추정합니다. 기준 시각 좌표로 정렬한 여러 쌍의 이동 추정에서 성분별 중앙값을 이동 prior로 구성합니다.
2. **공간 특징과 recurrent 상태:** 광역 encoder는 200×200 관측을 50×50 특징으로 줄이고, 상세 encoder는 100×100 특징을 유지합니다. 광역 ConvGRU와 강우 중심 ConvGRU는 서로의 특징과 시간 조건을 받아 18단계 동안 상태를 갱신합니다. 광역 hidden 채널은 48, 상세 hidden 채널은 32입니다.
3. **강우 이동과 증감:** prior, 단계별 이동 보정, 지역 조건으로 미래 이동장을 구성합니다. midpoint 방식의 기본 3개 substep으로 위치와 역방향 source map을 전개합니다. 관측 강우는 source/sink에 따른 multiplicative amplitude로 성장·감쇠하며, 발생 이후 생성 강우는 별도의 한 구간 이동 상태를 따라 이동합니다.
4. **상세 기준 강우:** 전개한 광역 강우를 상세 영역에서 읽고, 이동한 native 상세 잔차를 더해 기준 강우 `anchor`를 만듭니다. 상세 잔차에도 관측 광역 강우와 같은 누적 성장·감쇠를 적용합니다. 유효 source support 밖의 잔차는 억제합니다.
5. **상세 출력 보정:** 상세 decoder가 기준 강우, 광역 특징, 강우 중심 확률·강도, 상세 특징, 시간 조건을 받아 분위수 출력을 보정합니다.

모델의 이동장은 **관측 강우 영상의 이동 추정과 학습된 보정**입니다. 이를 실제 대기 바람 벡터로 해석하면 안 됩니다. 현재 입력에는 별도의 풍향·풍속 관측장이 없습니다.

## 순서가 유지되는 5개 분위수

모델은 0.5, 0.6, 0.75, 0.9, 0.95의 다섯 분위수를 상세 격자에서 출력합니다. 전체 `fine` 출력 shape은 `[B,18,5,100,100]`이며, 배포 추론은 **q0.6**을 선택하여 `[18,100,100]` 배열로 저장합니다.

`_fine_quantile_readout()`은 기준 강우를 inverse-softplus 좌표로 변환하고, 하나의 부호 있는 보정과 양수 분위수 간격을 적용합니다.

```text
보정 = 20 × tanh(raw_correction / 20)
분위수 간격 = softplus(raw_gaps)
간격 누적값을 q0.6 중심으로 정렬
최종 강우 = max(anchor + softplus 좌표에서의 변화량, 0)
```

양수 간격의 누적값으로 분위수 순서를 유지하고 최종 결과는 비음수로 제한합니다. `fine_readout_floor=0.05`는 좌표 변환의 안정화 값이며, 결과 강우에 공통 바닥 강우량으로 더하지 않습니다. 보정이 0일 때 q0.6은 원래 `anchor`를 유지하며, 기준 강우가 0인 경우도 포함합니다.

V45는 상세 발생량을 최종 강우에 직접 더하던 독립 합산 경로를 제거했습니다. 상세 decoder의 마지막 채널인 occurrence logit은 해당 미래 시각의 강우 발생 여부를 보조 학습하지만, 그 확률이나 별도 발생량을 최종 분위수 강우에 직접 합산하지 않습니다. 광역 생성 강우의 상태 전개는 유지합니다. 이 구분은 [train_v44_module.py](../training/source/experiments/scripts/train_v44_module.py)의 V45 손실 구현에서도 확인할 수 있습니다. 파일 이름에는 V44가 남아 있지만 해당 클래스의 문서 문자열과 구현은 V45입니다.

## 학습 실행과 선택 체크포인트

실제 학습 설정은 [train_v45.py](../training/source/experiments/scripts/train_v45.py), 기본 학습·손실 코드는 [train_v39.py](../training/source/experiments/scripts/train_v39.py), 데이터 window 구성은 [data_v39.py](../training/source/experiments/scripts/data_v39.py)에 있습니다.

| 항목 | 실제 설정 |
|---|---|
| 초기화 / seed | scratch / 4401 |
| 관측 / 예측 | 6장 / 18장 |
| 배치 / gradient accumulation | 8 / 2, 실효 배치 16 |
| 최적화 | AdamW, lr 0.0002, weight decay 0.00001 |
| 학습률 | 1,000 update warmup, cosine decay, 최소 lr 0.00000001 |
| 원래 학습 window / 고정 검증 window | 90,823 / 815 |
| 반복 포함 sampler 표본 수 | 136,235 |
| 전체 학습 예산 | 20 epoch, epoch당 8,514 update, 총 170,280 update |
| 선택한 체크포인트 | step 85,140, 체크포인트 내부 epoch 9 (0부터 시작) |

전체 실행의 학습 완료량과 배포용 best 체크포인트의 저장 시점은 다릅니다. 최종 학습 완료 기록은 170,280 update이며, 배포용 파일은 검증 선택 점수가 가장 좋았던 step 85,140입니다. 검증 선택 점수는 CSI 0.1mm, CSI 0.5mm, +120~180분 CSI 0.5mm의 평균입니다. 체크포인트가 더 이른 단계에서 선택됐다는 이유로 전체 학습 실행이 그 단계에서 중단됐다고 해석하지 않습니다.

배포 체크포인트의 SHA256은 다음과 같습니다. `runtime.py`는 이 해시, `global_step`, `net.*` state key와 구조의 strict load를 검증합니다.

```text
fb7e43c65c38493efdec4f4a4af9649ec7f8149cf0a80f0e9cd552777faae755
```

`train_v44_module.Module`의 일반 생성자 기본 history 값과 달리, 실제 V45 runner는 `history_steps=6`을 명시합니다. 재현에는 runner와 저장된 하이퍼파라미터를 기준으로 사용해야 합니다.

## 모델 출력과 실행 프로그램 후처리의 구분

모델의 분위수 decoder 보정은 학습된 구조의 일부입니다. 실행 프로그램의 [correction.py](../inference/correction.py)는 그 뒤에 별도의 고정 수학 계수를 적용합니다. 이 후처리는 모델 재학습이나 미래 정답으로 계수를 재추정하는 연산이 아닙니다. 후처리 효과와 모델 자체 성능을 구분하여 평가해야 합니다.

원본·보정 NPZ 배열은 mm/10분입니다. [ascii_export.py](../inference/src/ascii_export.py)는 보정 배열을 한 번만 6배 하여 mm/h로 바꾸고 EPSG:3857 1km 격자로 평균하여 ASC/PRJ를 저장합니다. PNG도 표시 단위가 mm/h이지만 native 500m 상세 격자를 유지합니다. 모델의 native 상세 출력, ASC의 1km 평균 출력, 지점 평가 결과는 서로 다른 공간 표현이므로 동일한 수치로 간주하지 않습니다.
