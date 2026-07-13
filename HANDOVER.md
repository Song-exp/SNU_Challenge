# 작업 인수인계서 — SNU AI Challenge (비디오 프레임 순서 예측)

> 최종 갱신: 2026-07-13 | 작성: 초기 세팅~zero-shot 비교 단계 담당
> 대회 규정·평가 기준은 `PROJECT_SETUP.md` 참조. 이 문서는 "지금까지 한 것, 지금 상태, 다음에 할 것"만 다룬다.

---

## 1. 한 줄 현황

**환경 구축과 5개 후보 모델 zero-shot 비교까지 완료. 결론: zero-shot으로는 아무 모델도 못 푼다(섞인 샘플 정확도 = 무작위 수준). 다음 단계는 Qwen3-VL-2B QLoRA 파인튜닝(`train.py` 작성부터).**

## 2. 완료된 작업

| 작업 | 산출물 |
|---|---|
| 로컬 GPU 환경 구축 (RTX 5060 8GB, torch cu128) | `.venv` (Python 3.14, torch 2.11+cu128, transformers 5.13, bitsandbytes 0.49) |
| 더미 제출 파일 (팀 구성용, 전부 `[1,2,3,4]`) | `outputs/submission.csv` |
| 고정 평가셋 구축 (train에서 시드 42로 300개) | `splits/holdout_300.csv` |
| 모델 다운로드 인프라 (불안정 네트워크 대응) | `scripts/download_models.py`, `models/` 5개 모델 41.7GB |
| 모델 비교 실험 노트북 | `Model_Experiments.ipynb` |
| 평가 자동화 | `scripts/eval_zero_shot.py`, `scripts/overnight_run.py` |
| **5개 모델 zero-shot 비교** | `outputs/experiments.csv`, `outputs/zero_shot_report.md`, `outputs/errors_*.csv` |

## 3. 핵심 결과 (상세: `outputs/zero_shot_report.md`)

- 찍기 기준선(항상 `[1,2,3,4]`) = 16.0% (No_ordering 비율)
- 5개 모델 전체 정확도 7.0~16.3%지만, **섞인 샘플만 보면 0.8~4.4% = 무작위(4.2%)와 동급**
- 점수 차이는 전부 "identity 응답 빈도" 차이 → **zero-shot 순위로 모델 잠재력 판단 금지**
- 평가 파이프라인 재현성 검증 완료 (동일 입력 → 소수점까지 동일 결과)

**권고**: 레시피 개발 = Qwen3-VL-2B (0.69초/샘플, fp16 학습 여유) → 최종 = Qwen3-VL-4B QLoRA. 이후 모든 실험의 핵심 지표는 **섞인 샘플 정확도**.

## 4. 저장소 구조

```
SNU_AI_Challenge/
├── PROJECT_SETUP.md            # 대회 규정·평가 기준·환경 가이드
├── HANDOVER.md                 # 본 문서
├── EXPERIMENTS.md              # 파인튜닝 실험 세팅·실행·로드맵 (2026-07-13 추가)
├── SNU_AI_Challenge_Baseline_Code.ipynb   # 원본 베이스라인 (참고용)
├── Model_Experiments.ipynb     # 모델 비교 실험장 (MODEL_ID만 바꿔 Run All)
├── Train_Experiments.ipynb     # 파인튜닝 조종석 (train.py 실행·loss·평가·오답 분석)
├── scripts/
│   ├── download_models.py      # curl 이어받기 다운로더 (※ 동시에 1개만 실행)
│   ├── train.py                # QLoRA 파인튜닝 엔진 (재셔플 증강·검증셋 제외·밤샘 안전장치)
│   ├── eval_zero_shot.py       # holdout 300 평가 (--adapter로 파인튜닝 결과도 평가)
│   └── overnight_run.py        # 다운로드 대기 → 순차 평가 오케스트레이터
├── splits/holdout_300.csv      # 고정 평가셋 ⚠️ 학습 데이터에서 제외 필수
├── snuaichallenge_data/        # 대회 데이터 (git 제외)
├── models/                     # 모델 가중치 5개, 41.7GB (git 제외)
└── outputs/                    # 결과물 (git 제외)
    ├── experiments.csv         # 실험 기록 누적 (정확도/속도/VRAM)
    ├── zero_shot_report.md     # 비교 분석 보고서
    ├── errors_<모델>.csv       # 모델별 오답 상세 (EDA 담당 분석용)
    └── submission.csv          # 더미 제출 파일
```

## 5. 실행 방법

```powershell
# 환경
.venv\Scripts\Activate.ps1

# 모델 추가 다운로드 (스크립트 상단 MODELS 리스트 수정 후)
python scripts/download_models.py

# 평가 (모델 하나)
python scripts/eval_zero_shot.py --model ./models/Qwen3-VL-2B-Instruct
python scripts/eval_zero_shot.py --model ./models/Qwen2.5-VL-7B-Instruct --load-4bit

# 노트북 실험: Model_Experiments.ipynb 설정 셀만 수정 → Run All
# (모델 교체 시 커널 재시작 필수 — VRAM 잔여물 방지)
```

## 6. ⚠️ 함정 목록 (전부 실제로 겪은 것)

1. **네트워크가 대용량 다운로드의 긴 연결을 주기적으로 끊는다** → 반드시 `download_models.py`(curl 이어받기) 사용. HF `from_pretrained`의 자동 다운로드에 맡기면 몇 시간씩 행업함 (노트북 로드는 `local_files_only=True`로 차단해둠)
2. **다운로더는 동시에 1개만** — 중복 실행 시 같은 파일에 동시 쓰기로 오염 위험
3. **venv의 python.exe는 프로세스가 항상 2개(런처+본체)로 보인다** — 중복 실행 판단은 ParentProcessId로
4. **PowerShell 1MB = MiB, HF 크기 = 십진 MB** — "다운로드가 95%에서 멈췄다"는 오판의 원인이었음. 같은 단위로 비교할 것
5. **VRAM 8GB 중 실사용 가능은 ~7.3GB** (시스템 상시 점유 0.8GB) — eval 스크립트의 대기 기준은 7.0GB로 설정돼 있음
6. **밤샘 작업은 절전이 죽인다** — 스크립트들이 절전 차단을 걸지만, 덮개 닫힘은 못 막음. 전원 연결 + 덮개 열기
7. **holdout_300.csv의 Id는 파인튜닝 학습 데이터에서 반드시 제외** — 안 지키면 이후 모든 평가가 오염됨
8. **제출은 1일 2회(팀 전체)** — 실험 검증은 holdout으로, 제출은 확인용으로만

## 7. 다음 단계 (우선순위순)

1. **`scripts/train.py` 작성** — Qwen3-VL-2B QLoRA: 재셔플 증강(샘플당 2~4개 순열), holdout 제외, fp16, gradient checkpointing + batch 1 + accumulation
2. 파인튜닝 모델을 `eval_zero_shot.py --model <출력경로>`로 평가 → **섞인 샘플 정확도**가 오르는지 확인
3. 레시피 확정 후 Qwen3-VL-4B로 스케일 업
4. 출력 형식 실험 (리스트 생성 vs 24-순열 분류 vs 제약 디코딩)
5. 유의미한 모델이 나오면 test 추론 → 제출 (베이스라인 노트북 또는 eval 스크립트 변형)

## 8. 팀 분업 참고 (합의된 구조)

- **모델링(GPU 보유자)**: 학습·평가 실행 전담, GPU 실험은 "설계 요청 → 밤 배치 실행 → 결과 공유" 사이클
- **텍스트 EDA**: `errors_*.csv` 오답 유형 분석, 재셔플 증강 생성기, 프롬프트 후보
- **이미지 EDA**: 해상도-정확도 트레이드오프 설계, 프레임 유사도 분석, 후반 재현성 패키징
- 규정: 최종 모델은 단일 모델 (여러 실험은 OK, 결과 조합=앙상블=실격)

## 9. 기타

- git: 현재 변경사항 미커밋 상태. `.gitignore`에 데이터/모델/출력 제외 설정 완료. 커밋 및 GitHub private 저장소 연결 권장 (팀 코드 공유용)
- 외부 API 사용액: 현재까지 0원
- Kaggle: 더미 제출로 팀 구성 요건 충족 가능 (`outputs/submission.csv` 업로드)
