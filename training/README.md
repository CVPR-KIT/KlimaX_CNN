# V45 학습 소스 — 1시간 입력으로 3시간 강우 예측

이 디렉터리는 V45의 모델·학습 runner·손실·데이터 loader 코드입니다. 체크포인트, 실제 데이터셋 및 예측 산출물은 포함하지 않습니다. 명령은 저장소 루트에서 `cd training` 후 실행합니다. 배포 EXE와 예측 후 강우량 보정은 상위 디렉터리의 추론 패키지에서 다룹니다.

최근 V45 학습 방식의 소스와 실행 절차입니다. 10분 간격 강우 6개 프레임을 입력해 +10~180분의 18개 시점을 예측합니다. 사전학습 가중치 없이 처음부터 학습하는 `MotionCoreNowcaster` 기반 실험이며, ClimaX Transformer를 직접 미세조정하는 방식은 아닙니다.

## V45에서 바뀐 점

- 상세 강우에 별도의 발생량을 직접 더하던 경로를 제거했습니다.
- 상세 강우는 하나의 보정 디코더에서 순서가 유지되는 비음수 분위수로 출력합니다. 보정은 강우 생성과 감소를 모두 표현하고, 보정값이 0이면 q0.6이 기준 강우를 유지합니다.
- 상세 발생 헤드는 각 예측 시점의 강우 유무를 보조 감독합니다. 이 헤드의 출력은 상세 강우에 직접 합산하지 않습니다.
- 관측 상세 잔차에도 광역 관측 상태의 누적 성장·감쇠를 적용합니다.
- 상세 발생량·appearance·negative-final·dry-window의 중복 손실은 제거하고, 광역 상태와 광역 생성 감독은 유지합니다.

세부 구조와 손실은 [METHOD.md](docs/METHOD.md), 입력 형식은 [DATA.md](docs/DATA.md)에 설명합니다.
모델·학습·데이터·검증 코드의 위치와 역할은 [CODE_MAP.md](docs/CODE_MAP.md)에서 확인할 수 있습니다.

## 학습 설정

| 항목 | 설정 |
|---|---|
| 입력 / 출력 | 6개 / 18개 프레임, 10분 간격 |
| 상세 / 광역 | 100×100, 500m / 200×200, 2km |
| 분위수 | 0.5, 0.6, 0.75, 0.9, 0.95 |
| 초기화 / seed | scratch / 4401 |
| 배치 | 8 × gradient accumulation 2 = 16 |
| 최적화 | AdamW, lr 2e-4, weight decay 1e-5 |
| 스케줄 | 1,000 update warmup → cosine decay, 최소 lr 1e-8 |
| 예산 | 20 epoch, 8,514 update/epoch, 총 170,280 update |
| 표본 추출 | 원본 표본 + 50% 가중 반복, 8일 블록 |
| 검증 선택 | mean(CSI 0.1, CSI 0.5, +120~180분 CSI 0.5) 최대화 |

전체 수치는 [configs/v45_contract.json](configs/v45_contract.json)에 있습니다. JSON은 설명용 기록이며 실행 코드가 읽는 설정 파일은 아닙니다.

## 실행

원래 실행 환경은 Linux, Python 3.10.12, PyTorch 2.0.1+cu118, Lightning 1.9.5, L40S GPU입니다. Lightning 2.x는 이 코드의 기존 callback API와 호환 대상으로 검증하지 않았습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python test_model.py
python test_loss.py
python verify_package.py
```

데이터 없이 두 CPU 검증을 실행할 수 있습니다. 본학습은 CUDA GPU와 별도로 준비한 **동일한 감사 완료 데이터·manifest·sampling index**가 필요합니다. 데이터셋은 이 저장소에 포함하지 않습니다. 현재 runner는 학습 90,823개, 검증 815개, sampler 136,235개라는 원래 실험의 수를 엄격하게 검사합니다. 작은 예제나 다른 데이터셋으로 바로 학습하는 범용 runner는 아닙니다.

아래 경로를 실제 로컬 경로로 바꿉니다. 각 출력 디렉터리는 새 경로여야 합니다.

```bash
python source/experiments/scripts/train_v45.py --mode smoke --root /path/to/corpus --manifest /path/to/history6_manifest.json --index /path/to/birth_index.jsonl --out runs/smoke8 --epochs 20 --seed 4401
python source/experiments/scripts/train_v45.py --mode prepare --root /path/to/corpus --manifest /path/to/history6_manifest.json --index /path/to/birth_index.jsonl --smoke-out runs/smoke8 --out runs/READY.json --epochs 20 --seed 4401
python source/experiments/scripts/train_v45.py --mode train --root /path/to/corpus --manifest /path/to/history6_manifest.json --index /path/to/birth_index.jsonl --readiness runs/READY.json --out runs/train20 --epochs 20 --seed 4401
```

8-update smoke에서 기울기, 실제 입력 배치, 복원 출력, 분위수 순서, 상세 직접 합산 제거를 검사합니다. `prepare`는 소스·입력·체크포인트·검증 기록의 hash를 묶고, 본학습 직전에 다시 확인합니다. 공개용 파일 구성과 경로가 원래 실행과 달라졌으므로 **이 복사본에서 새 smoke 기록을 생성해야 합니다**. Smoke 후에는 소스나 evidence 파일을 변경하거나 이동하지 않습니다.

학습은 `progress.json`, `validation.jsonl`, `checkpoints/`, `best/`를 생성합니다. `best/`는 모든 검증 시점 체크포인트를 보존하므로 폴더 이름만으로 최선의 모델을 결정하면 안 됩니다. 완료 시 `DONE.json`의 `best_path`와 `best_selection_score`를 확인합니다. runner에는 체크포인트에서 중단 학습을 재개하는 CLI 옵션이 없습니다.

## 검증 상태

원래 V45 실행은 독립 소스 검수, CPU 구조·손실 검사, GPU 8-update smoke와 엄격한 체크포인트 복원을 통과했습니다. 2026-10-08의 3개 epoch 검증 결과는 [interim_evidence.json](docs/interim_evidence.json)에 정리했습니다. 해당 JSON의 진행 상태는 **2026-10-08 당시의 역사적 기록**입니다. 이후 20 epoch / 170,280 update 학습이 완료됐으며 전역 validation으로 선택한 step 85,140 체크포인트는 `../inference/saved_model.ckpt`에 있습니다. 최신 상태와 배포 평가 결과는 [저장소 README](../README.md) 및 [검증 기록](../docs/VALIDATION.md)을 기준으로 합니다.

`python verify_package.py`를 실행하면 현재 환경의 별도 검증 결과를 `PACKAGE_VALIDATION.json`에 생성합니다. 해당 파일은 실행 결과이므로 소스 배포에는 포함하지 않습니다. 원래 실행과의 차이는 [source_provenance.json](docs/source_provenance.json)에서 확인할 수 있습니다.

## 파일 구성과 라이선스

`source/`에는 V45 실행에 필요한 코드만 포함하고, 과거 이름의 `train_v39.py`와 `train_v44_module.py`는 상속·체크포인트 클래스 경로 호환성을 위해 유지했습니다. 데이터, 가중치, 서버 접속 정보, 원래 운영 launcher는 제외했습니다.

Microsoft에서 유래한 코드의 MIT 고지는 [LICENSE.climax](LICENSE.climax)에 보존했습니다. 새 프로젝트 코드 전체에 적용할 라이선스는 아직 지정하지 않았습니다.
