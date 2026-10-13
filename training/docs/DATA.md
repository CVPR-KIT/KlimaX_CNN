# 데이터 계약

데이터를 배포하지 않습니다. 전처리 전체 파이프라인과 원래 감사 manifest/index의 생성 도구도 이 최소 패키지에는 포함하지 않습니다. 아래는 loader가 요구하는 형식이며, 새 데이터의 감사를 대신하지 않습니다.

## 일별 NPZ

```text
corpus/
  train/YYYYMMDD.npz
  val/YYYYMMDD.npz
```

| key | shape | 값 |
|---|---|---|
| rain | [144,100,100] | 상세 500m, mm/10min |
| context | [144,200,200] | 광역 2km, mm/10min |
| valid | [144,200,200] | uint8 coverage 0~255 |

프레임 간격은 10분입니다. 프레임 0은 일별 reset으로 생긴 구조적 0이므로 관측 입력과 target에서 제외합니다. 상세 crop은 광역 800×800 native crop의 [350:450,350:450]에 해당하고 광역은 4×4 block 평균입니다. 다른 기하에 같은 loader를 적용하면 안 됩니다.

## 감사 manifest

loader가 사용하는 최소 구조:

```json
{
  "audit_pass": true,
  "protocol": {"history": 6, "horizon": 18},
  "days": [
    {"date": "20000101", "split": "train", "train_indices": [6, 7]},
    {"date": "20000102", "split": "val", "eval_indices": [6]}
  ]
}
```

이 예시는 형식 설명만을 위한 것이며 실행 가능한 원래 manifest가 아닙니다. `audit_pass`를 임의로 true로 설정해도 데이터 검증이 수행되는 것은 아닙니다. index r에서 입력은 r-5~r, target은 r+1~r+18이며 6 ≤ r ≤ 125입니다. Train/val 날짜는 겹치면 안 됩니다.

## sampling index (JSONL)

```json
{"split":"train","date":"20000101","ref":6,"weight":2.0}
```

학습 window마다 (date,ref)로 하나의 row가 필요합니다. 중복 row는 오류입니다. sampler는 weight를 [1,4]로 제한합니다. 원래 V45 runner는 모든 90,823개 학습 window가 index에 있음을 확인하고, 반복 포함 sampler 길이 136,235와 검증 window 815개를 요구합니다.

## batch tensor

배치 차원을 제외하면 `fine=[6,1,100,100]`, `context/valid=[6,1,200,200]`, `clock=[19,4]`, `truth/fine_valid=[18,1,100,100]`, `wide_truth/wide_valid=[18,1,200,200]`입니다. target은 raw 강우량이며 loader에서 log1p로 바꾸지 않습니다. clock은 기준 시점과 미래의 알려진 시간 특성만 포함합니다.
