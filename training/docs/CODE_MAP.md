# V45 코드 안내

GitHub 저장소의 루트로 사용할 폴더는 `klimax-v45/`입니다. 상위 폴더의 ZIP과 전달 기록은 업로드 대상에 포함할 필요가 없습니다.

## 폴더 구조

```text
klimax-v45/
├── README.md                     실행 안내와 실험 상태
├── requirements.txt              직접 실행 의존성
├── LICENSE.climax                상속한 코드의 MIT 고지
├── .gitignore                    데이터·가중치·캐시 제외
├── configs/
│   └── v45_contract.json          원래 학습 설정 기록
├── docs/
│   ├── CODE_MAP.md               코드 위치와 읽는 순서
│   ├── METHOD.md                 모델·손실·평가 방식
│   ├── DATA.md                   NPZ·manifest·sampling index 계약
│   ├── interim_evidence.json     원래 실행의 중간 검증 결과
│   └── source_provenance.json    원본 hash와 공개용 수정 내역
├── source/
│   ├── ClimaX/src/climax/
│   │   ├── regional_forecast/motion_core_arch.py
│   │   └── utils/lr_scheduler.py
│   └── experiments/scripts/
│       ├── train_v45.py
│       ├── train_v44_module.py
│       ├── train_v39.py
│       ├── data_v39.py
│       ├── birth_sampling.py
│       ├── birth_objective.py
│       └── test_loss.py
├── test_model.py                 데이터 없는 CPU 모델 검증
├── test_loss.py                  CPU 손실 검증 진입점
├── verify_package.py             패키지 검사와 검증 기록 생성
└── PACKAGE_VALIDATION.json       공개용 복사본의 검증 결과
```

## 핵심 파일

| 역할 | 파일 | 확인할 내용 |
|---|---|---|
| 모델 | `source/ClimaX/src/climax/regional_forecast/motion_core_arch.py` | `MotionCoreNowcaster`, 이동·상태 전개, `_fine_quantile_readout` |
| V45 학습 진입점 | `source/experiments/scripts/train_v45.py` | 고정 H6 학습 계약, smoke/prepare/train, 복원 검사 |
| V45 손실 통합 | `source/experiments/scripts/train_v44_module.py` | `Module`, 상세 occurrence와 광역 발생 감독 |
| 공통 학습 기반 | `source/experiments/scripts/train_v39.py` | 기본 손실, CSI/FSS, optimizer, callback, 체크포인트 |
| 데이터 | `source/experiments/scripts/data_v39.py` | NPZ 읽기, 입력/target 분리, clock, coverage |
| 표본 추출 | `source/experiments/scripts/birth_sampling.py` | `BirthSampler`, 가중 반복, 날짜 블록 |
| 감독 target | `source/experiments/scripts/birth_objective.py` | 발생 target, balanced location, amount/negative 손실 |
| lr 스케줄 | `source/ClimaX/src/climax/utils/lr_scheduler.py` | warmup + cosine |

읽는 순서는 `README → METHOD → motion_core_arch → train_v44_module → train_v45 → DATA`를 권합니다. 파일 이름의 V39/V44는 상속과 체크포인트 클래스 경로를 유지하기 위한 이름이며, 이 폴더의 모델·손실 코드는 V45 실험 스냅샷입니다. 실행은 `train_v45.py`를 사용합니다.

## 보존 및 제외 범위

원래 실험 폴더에는 운영 launcher, 감사 기록과 동결 소스를 보존합니다. 공개용 폴더에는 V45의 실행 의존성과 문서만 모았습니다. 데이터·체크포인트·과거 실험 전체·서버 연결 파일·Python 캐시는 포함하지 않습니다. 새 데이터 전처리나 임의 데이터셋을 위한 범용 학습 도구는 포함하지 않으며, 원래 감사 데이터 계약을 전제로 합니다.
