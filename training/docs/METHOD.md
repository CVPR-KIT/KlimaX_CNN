# V45 학습 방식

## 입력과 상태 전개

상세 500m 강우와 광역 2km 강우, 광역 유효 관측 마스크를 각각 6개 프레임 입력합니다. clock은 기준 시점과 미래 18개 시점의 연중·하루 내 위상을 sin/cos로 표현합니다. 미래 강우는 입력하지 않습니다.

광역 순환 상태는 이동, 관측 상태의 성장·감쇠, 생성 상태를 전개합니다. 상세 영역에서는 마지막 관측의 광역 성분과 상세 잔차를 이동시킵니다. 관측 상세 잔차는 광역 관측 상태와 같은 누적 multiplicative amplitude를 적용받습니다. 상세 디코더는 이동된 기준 강우를 보정합니다.

## 상세 분위수 출력

`motion_core_arch.py::_fine_quantile_readout`는 역 softplus 좌표에서 하나의 signed correction을 적용합니다. correction은 `20 * tanh(raw / 20)`으로 제한하고, softplus 분위수 간격의 누적값을 q0.6 중심에 맞춥니다. `fine_readout_floor=0.05`는 좌표 변환의 안정화에 쓰이며 결과 강우에 고정 바닥값으로 더하지 않습니다. 출력은 0 이상이고 분위수 순서가 유지됩니다.

발생 로짓은 디코더의 마지막 채널이며 각 시점의 truth ≥ 0.1mm를 보조 감독합니다. 발생 로짓만 바꾸면 강우 분위수는 변하지 않습니다. 기준 강우가 0이어도 공유 보정 경로에는 생성 가능한 기울기가 있습니다.

## 손실

| 손실 | 가중치 |
|---|---:|
| 상세 pinball | 1.0 |
| 광역 q0.6 pinball | 0.3 |
| core BCE / core auxiliary | 0.5 / 0.1 |
| 구조 손실 | 0.2 |
| motion photo / forward-backward | 0.1 / 0.05 |
| motion smooth / temporal | 0.02 / 0.02 |
| 광역 birth location / amount | 0.2 / 0.2 |
| 광역 appearance / negative-final | 0.3 / 0.3 |
| 상세 시점별 occurrence | 0.2 |

상세 분위수별 pinball 가중치는 1, 1, 0.75, 0.5, 0.25입니다. 상세 발생량, appearance, negative-final과 dry-window 관련 추가 손실은 0으로 고정하고, 이를 0이 아닌 값으로 설정하면 오류를 냅니다. 광역 negative-final 손실은 약한 실제 강우도 0으로 강제하지 않고 실제 truth를 초과하는 양을 대상으로 합니다.

광역 손실은 full coverage ≥ 0.999를 요구하고, core label은 주변 invalid 영역의 영향을 피하도록 마스크를 침식합니다. 상세 유효 마스크는 광역 coverage를 상세 픽셀로 보수적으로 투영한 값이며 native 상세 관측 마스크를 복구한 값은 아닙니다.

## 표본 추출과 평가

원래 학습 window를 한 번씩 포함하고, sampling index의 weight를 [1,4]로 제한해 원본 수의 50%를 반복 추출합니다. 미래에서 도출한 분류는 학습 표본 선택에만 쓰고 validation은 변경하지 않습니다. seed와 epoch로 표본 순서를 결정하고, 8일 단위 블록 안에서 섞습니다. 마지막 불완전 배치는 버립니다.

Validation은 고정 815개 window 전체를 사용합니다. threshold 0.1mm CSI, threshold 0.5mm CSI, +120~180분 threshold 0.5mm CSI의 평균을 최선 체크포인트 선택에 씁니다. CSI는 해당 범위의 TP/FP/FN을 합산해 계산하며 시점별 CSI의 단순 평균은 아닙니다. 구조 FSS의 실제 커널은 광역 해상도에서 3/5/9/17 픽셀입니다.

원래 실행 검증과 공개용 복사본의 검증은 별개입니다. 중간 CSI 수치만으로 이전 버전 대비 개선이나 최종 품질을 판정하지 않습니다.
