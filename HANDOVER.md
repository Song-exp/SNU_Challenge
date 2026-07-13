# 작업 인수인계서 — SNU AI Challenge (비디오 프레임 순서 예측)

> 최종 갱신: **2026-07-14** | 범위: 초기 세팅 → zero-shot 비교 → **파인튜닝 실험 1라운드(진행 중)**
> 대회 규정·평가 기준은 `PROJECT_SETUP.md`, 실험 상세 설계는 `EXPERIMENTS.md` 참조. 이 문서는 "지금까지 한 것, 지금 상태, 다음에 할 것"만 다룬다.

---

## 1. 한 줄 현황

**첫 파인튜닝(exp01: Qwen3-VL-2B QLoRA)이 돌아가는 중 (7/14 오전 완료 예상). 종료되면 어댑터 평가 → 프롬프트 스크리닝 2종까지 자동 실행되어, 아침에 `experiments.csv` 한 표로 "파인튜닝 효과 + CoT 프롬프트 가능성"을 판정한다.**

## 2. 완료된 작업

| 작업 | 산출물 |
|---|---|
| 로컬 GPU 환경 구축 (RTX 5060 8GB, torch cu128) | `.venv` (Python 3.14, torch 2.11+cu128, transformers 5.13, bitsandbytes, peft) |
| 더미 제출 파일 (팀 구성용) | `outputs/submission.csv` |
| 고정 평가셋 (train에서 시드 42, 300개) | `splits/holdout_300.csv` |
| 모델 다운로드 인프라 (불안정 네트워크 대응) | `scripts/download_models.py`, `models/` 5개 41.7GB |
| 5개 모델 zero-shot 비교 | `outputs/zero_shot_report.md`, `outputs/experiments.csv` |
| QLoRA 학습 엔진 + 실험 조종석 | `scripts/train.py`, `Train_Experiments.ipynb`, `EXPERIMENTS.md` |
| 프롬프트 레지스트리 (학습·평가 공유) | `scripts/prompts.py` — v1_list / v2_temporal / v3_cot(CoT) |
| 학습 종료 후 자동 평가 큐 | `scripts/prompt_screen_queue.py` (+ `outputs/prompt_queue.log`) |
| 결과 파일 보존 체계 (7/14 정비) | 전체 예측 `outputs/preds/`, 오답 `outputs/errors_*.csv` — 모델 출력 전문 포함 |

## 3. 핵심 결과 요약 (7/13 zero-shot, 상세: `outputs/zero_shot_report.md`)

- 5개 모델 전부 **섞인 샘플 정확도 0.8~4.4% = 무작위 수준** → zero-shot으로는 태스크 불가, 점수는 파인튜닝에서 나와야 함
- 전체 점수 차이는 identity(`[1,2,3,4]`) 응답 빈도 착시 → **모든 판정은 `acc_shuffled` 기준** (무작위 4.2%, ±4%p는 노이즈)
- 선정: 레시피 개발 = Qwen3-VL-2B (0.69초/샘플) → 최종 후보 = Qwen3-VL-4B

## 4. 실험 체계 (7/14 기준 — 이 절이 현재 세팅의 본체)

### 4.1 실험 사이클

모든 실험은 같은 사이클: **`train.py` 학습 → `eval_zero_shot.py --adapter` 평가 → `experiments.csv` 비교**.
실험 정의는 `Train_Experiments.ipynb` ① 레지스트리에 한 줄 추가 (기본값과 다른 것만 명시):

- 등록된 실험: exp01(기준점) / exp02(aug 4배) / exp03(lr 5e-5) / exp04(LoRA r32) / exp05(v2 프롬프트 세트) / (보류) exp10(4B 스케일업)
- 실행 순서: 스모크(3분, 코드 변경 시 필수) → 본학습(백그라운드) → 게이지 → 평가 → 오답 비교

### 4.2 비용 사다리 — 실험 예산 운영 원칙

| 단계 | 비용 | 용도 |
|---|---|---|
| 추론 스크리닝 | 분 단위 | zero-shot 프롬프트 1차 필터, 어댑터 교차 평가 |
| 미니 학습 (`--max-samples 1000 --max-steps 300`) | ~1.5시간 | 프롬프트/데이터/하이퍼파라미터 후보 비교의 **본편** |
| 본학습 (전체 데이터) | ~10시간 | 미니 학습 승자의 결승전 (밤 배치 1회 = 1실험) |

### 4.3 프롬프트 실험의 원칙 (오해 주의)

- **프롬프트는 학습 데이터의 일부다**: 학습 샘플 = 이미지 4장 + 프롬프트 + 정답. 따라서 어댑터는 학습 프롬프트에 조율됨
- **학습·평가에 반드시 같은 프롬프트** (`--prompt <이름>` 세트 적용) — 어댑터에 다른 프롬프트를 꽂은 결과는 우열 근거가 아님
- 어댑터 없는 zero-shot에서만 "추론만으로" 프롬프트 비교 가능 → 그래서 스크리닝은 zero-shot, 본 비교는 프롬프트별 미니 학습
- CoT 계열(v3_cot)은 평가 시 `--max-new-tokens 256` 필수 (기본 32면 잘려서 전부 파싱 실패)

### 4.4 지금 돌아가는 것 (7/14 00시 기준)

1. **exp01 본학습**: Qwen3-VL-2B fp16 + LoRA(trainable 0.3%), 재셔플 증강 2배 = 18,080항목, lr 1e-4, 1 epoch, v1_list. 약 3.9초/항목, VRAM ~7.0GB, **7/14 오전 10시대 완료 예상**. holdout 300 + `eda/stratified_valid.csv` 494개는 학습에서 제외됨
2. **종료 후 자동 체인**: 노트북 ⑥ exp01 어댑터 평가 → 큐(`prompt_screen_queue.py`)가 2분 양보 후 v2_temporal → v3_cot(256) zero-shot 평가 (GPU 경합은 VRAM 대기로 자동 직렬화)
3. **결과 파일**: 성능 = `experiments.csv` | 전체 예측(출력 전문 포함) = `outputs/preds/<모델_프롬프트>.csv` | 오답만 = `outputs/errors_*.csv` | 큐 진행 = `outputs/prompt_queue.log`

### 4.5 아침에 확인할 것 (판정 기준 포함)

1. `Train_Experiments.ipynb` ⑧ 비교표 실행 → `acc_shuffled` 정렬
2. **exp01 vs zero-shot(0.8%)**: 크게 오르면 파인튜닝 레시피 유효 → exp02~04 밤 배치 개시 / 안 오르면 학습 로그·오답부터 진단
3. **v3_cot vs 4.2%**: 유의미하게 넘으면 CoT 프롬프트 세트 학습(exp06)을 레지스트리에 추가할 가치
4. 싼 추가 실험: exp01 어댑터 + v3_cot 교차 평가 (5분) → "파인튜닝 후에도 프롬프트가 중요한가"의 직접 측정

## 5. 저장소 구조

```
SNU_AI_Challenge/
├── PROJECT_SETUP.md            # 대회 규정·평가 기준
├── HANDOVER.md                 # 본 문서
├── EXPERIMENTS.md              # 파인튜닝 실험 설계·로드맵 상세
├── Model_Experiments.ipynb     # zero-shot 모델 비교 실험장
├── Train_Experiments.ipynb     # 파인튜닝 조종석 (레지스트리·학습·게이지·평가·오답)
├── SNU_AI_Challenge_Baseline_Code.ipynb   # 원본 베이스라인 (참고용)
├── scripts/
│   ├── download_models.py      # curl 이어받기 다운로더 (동시 1개만)
│   ├── train.py                # QLoRA 학습 엔진 (증강·검증셋 제외·밤샘 안전장치)
│   ├── eval_zero_shot.py       # holdout 평가 (--adapter, --prompt, 전체 예측 저장)
│   ├── prompts.py              # 프롬프트 레지스트리 (train/eval 공유)
│   ├── prompt_screen_queue.py  # 학습 종료 대기 → 순차 평가 큐
│   └── overnight_run.py        # (7/13 사용) 다운로드→평가 오케스트레이터
├── splits/holdout_300.csv      # 고정 평가셋 ⚠️ 학습 제외 필수
├── eda/                        # 팀원 EDA 산출물 + stratified_valid.csv(학습 제외 처리됨)
├── snuaichallenge_data/        # 대회 데이터 (git 제외)
├── models/                     # 모델 5개 41.7GB (git 제외)
└── outputs/                    # (git 제외)
    ├── experiments.csv         # 실험 기록 누적 (acc_shuffled 포함)
    ├── runs/<실험명>/           # 어댑터·학습로그·콘솔로그
    ├── preds/<모델_프롬프트>.csv # 전체 예측 원본 (출력 전문)
    ├── errors_*.csv            # 오답 추출본
    └── prompt_queue.log        # 큐 진행 기록
```

## 6. 실행 방법

```powershell
.venv\Scripts\Activate.ps1

# 파인튜닝 실험: Train_Experiments.ipynb에서 ① 레지스트리 수정 → ② NAME 지정 → 모두 실행
# (스모크 → 본학습 백그라운드 → 게이지 → 평가 → 분석 자동)

# 프롬프트 스크리닝 큐 (학습 종료 후 자동 실행; 지금도 하나 대기 중 — 중복 실행 금지)
python scripts/prompt_screen_queue.py --prompts v2_temporal v3_cot:256

# 개별 평가
python scripts/eval_zero_shot.py --model ./models/Qwen3-VL-2B-Instruct --prompt v3_cot --max-new-tokens 256
python scripts/eval_zero_shot.py --model ./models/Qwen3-VL-2B-Instruct --adapter ./outputs/runs/exp01_aug2_lr1e4/adapter
```

## 7. ⚠️ 함정 목록 (전부 실제로 겪은 것)

1. **대용량 다운로드는 `download_models.py`로만** — HF 자동 다운로드는 이 네트워크에서 몇 시간씩 행업 (노트북 로드는 `local_files_only=True`로 차단됨)
2. **다운로더·큐는 동시에 1개만** — 중복 실행이 락 경합/파일 오염의 원인
3. **venv python은 프로세스가 쌍(런처+본체)으로 보임** — 중복 판단은 ParentProcessId로
4. **PowerShell 1MB=MiB vs HF 십진 MB** — 진행률 오판의 단골 원인
5. **VRAM 실사용 한계 ~7.3GB** (시스템 0.8GB 상시 점유) — eval 대기 기준 7.0GB로 설정됨
6. **절전이 밤샘 작업을 죽임** — 스크립트가 차단하지만 덮개 닫힘은 못 막음. 전원 + 덮개 열기
7. **holdout_300 + stratified_valid의 Id는 학습 제외** — train.py가 자동 제외하지만, 새 학습 코드를 짤 때도 유지할 것
8. **제출은 팀 전체 1일 2회** — 검증은 holdout으로
9. **실행 중인 노트북(.ipynb)을 밖에서 수정 금지** (자동저장과 충돌). 반대로 `scripts/*.py` 수정은 실행 중 프로세스에 영향 없음 (메모리에 이미 로드됨) — 다음 프로세스부터 적용
10. **프롬프트-어댑터 세트 원칙** (§4.3) — 어기면 결과 해석이 무효

## 8. 다음 단계 (우선순위순)

1. (7/14 아침) §4.5 판정 → exp01 결과에 따라 분기
2. 하이퍼파라미터 밤 배치: exp02(aug4) → exp03(lr) → exp04(r32) — 변수 하나씩
3. v3_cot 결과 좋으면 CoT 학습 세트(exp06) 추가, 프롬프트 후보는 미니 학습으로 스크리닝
4. 출력 형식 실험(24-순열 분류/제약 디코딩)은 train.py target_text + 파서 동시 수정 필요 — 별도 브랜치 권장
5. 레시피 확정 → Qwen3-VL-4B 스케일업(exp10, lr 재탐색) → test 추론 → 첫 진짜 제출
6. 검증셋 기준 (합의됨): **holdout_300이 공식 채점용**, stratified_valid는 유형별 보조 분석용, 학습 제외는 두 셋의 합집합 (train.py에 반영돼 있음)

## 9. 팀 분업 참고

- **모델링(GPU 보유자)**: 학습·평가 실행 전담 — "설계 요청 → 밤 배치 → 결과 공유" 사이클
- **텍스트 EDA**: `outputs/errors_*.csv`(이제 출력 전문 포함) 오답 유형 분석, 프롬프트 후보 제안 (`prompts.py`에 추가만 하면 실험 가능)
- **이미지 EDA**: 해상도-정확도 트레이드오프, 프레임 유사도, 후반 재현성 패키징
- 규정: 최종 모델은 단일 모델 (실험 비교는 OK, 결과 조합=앙상블=실격)

## 10. 기타

- git: 7/13 zero-shot 단계까지 푸시됨. **7/14 변경분(train.py 라운드, prompts.py, 큐, eval 저장 체계) 미커밋** — exp01 결과 확인 후 결과 요약과 함께 커밋 권장
- 외부 API 사용액: 0원
- 예선 마감 **7/24** — 남은 열흘 기준, 본학습 슬롯은 최대 8~9회. 미니 학습 스크리닝으로 아껴 쓸 것
