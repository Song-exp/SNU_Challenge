# 작업 인수인계서 — SNU AI Challenge (비디오 프레임 순서 예측)

> 최종 갱신: **2026-07-15 밤** | 범위: 초기 세팅 → zero-shot → 파인튜닝 1라운드(완료) → **프롬프트 추론 실험·첫 리더보드 제출·CoT 파인튜닝 준비(최신)**
> 대회 규정·평가 기준은 `PROJECT_SETUP.md`, 실험 상세 설계는 `EXPERIMENTS.md` 참조. 이 문서는 "지금까지 한 것, 지금 상태, 다음에 할 것"만 다룬다.
> **최종 모델링 큰 그림 = `VISION_final_modeling.md`** (7/17 수립, 구조 인지형 파이프라인) | 단기 실행 계획 = `PLAN_post_labeling.md`

---

## 1. 한 줄 현황

**주력은 여전히 exp07(aug2, 48.41%, Public 0.766). 7/18: 라벨링 완주(train+test, 실패 0) → 4유형 확정(최약점 sparse_camX 21.6%·train 29%, `REPORT_typing_criteria.md`) → 트랙 2(힌트 주입·gemma CoT) 미니 2종 대폭 기각으로 종료 (§4f) → **exp16(sparse_camX x4 타깃 증강, ~23h) 가동 중, 판정 7/19 밤 vs 48.41%**. gemma 라벨의 가치는 증강 선별·분석으로 확정, 모델 주입은 입력·출력 모두 무익.**

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

## 4b. 7/15 진행 — 실험 2라운드·첫 제출·CoT 준비 (최신 상태, 이 절이 지금의 본체)

### 완료된 것

| 작업 | 결과 | 상세 문서 |
|---|---|---|
| exp06/exp07 완주 (7/14~15) | aug1 42.06% / **aug2 48.41%** (shuffled) — 증강2 유효, **exp07이 주력** | `EXPERIMENTS.md` |
| 전처리 파이프라인 검토 + 실측 | Partition·ai_score가 exp07 정확도를 강하게 예측 (Type-1 17.6% vs Type-2 58.5%; ai_score 단조) — 타깃팅 설계 실증 | `PLAN_prompt_and_preprocessing.md` (기법 4종 검토 포함) |
| 프롬프트 추론 실험 15종 (R0~R5) | **어떤 변형도 채택 기준(fixed−broken≥+10) 미달 → v1 유지**. 수확: r1_reorder(+6, 학습 후보), CoT는 학습으로만(풀CoT가 취약 세그먼트 붕괴), CLIP 힌트 무학습 주입 무효 | `PLAN_prompt_experiments.md` §실측 결과 |
| **첫 리더보드 제출 2건 (7/15 20시)** | baseline **0.76265** / combo **0.76614** — 격차 2문제=동률. holdout과 노이즈 내 정합 → **holdout 스크리닝 계속 유효** | 〃 §제출 결과 |
| test 분포 발견 | test는 평균 42단어(train 24)·26단어+ 80%·Type-2 80% → 우리 강점 구간. **Type-2·장문 개선이 최우선 레버리지, Type-1 구제(exp10)는 우선순위 하락** | 〃 |
| CoT 파인튜닝 인프라 (검증 완료) | `train_cot.py`(train.py 무변경), `<ANSWER>` 파서(단위테스트 6/6), target 기계생성(미리보기 검수), 전용 노트북 | `PLAN_cot_finetune.md` |

### 신규 파일 (7/15)

- `scripts/prompt_lab.py` — 프롬프트 추론 실험 헬퍼 (모델 1회 로드, 라우팅, paired 비교, **test 제출 생성 `make_submission`**)
- `scripts/train_cot.py` — CoT SFT 전용 학습 스크립트 (`--preview N`으로 target 눈검사 가능)
- `Prompt_Experiments.ipynb` / `Train_CoT_Experiments.ipynb` — 각 트랙 조종석
- `PLAN_prompt_and_preprocessing.md` / `PLAN_prompt_experiments.md` / `PLAN_cot_finetune.md` — 설계·실측 기록
- `prompts.py`에 v4_story(직답)·v4_story_cot(팀 고도화 원문) 추가, `eval_zero_shot.py` 파서에 `<ANSWER>` 우선 추가(하위호환)
- `outputs/prompt_experiments.csv`(추론 실험 기록), `outputs/submissions/`(제출 파일 + raw 감사본)

### 재부팅 후 CoT 실행 절차 (다음 담당자는 여기부터)

```
1. 재부팅 (학습 시작 전이므로 안전 — 오히려 GPU/커널이 깨끗해짐)
2. VS Code → Train_CoT_Experiments.ipynb 열기
3. ①(레지스트리) → ②(NAME=exp12_v4cot_aug1) → ⓪(미리보기) → ③(스모크 3분)
   ⚠️ 스모크 종료 로그의 peak VRAM 확인 — 7.5GB 초과 시 중단하고 상의
4. ④ 본학습 시작(백그라운드, ~11-13h) → ⑤ 게이지 → 전원 연결 + 덮개 열기 유지
   ⚠️ 학습 시작 후에는 컴퓨터를 끄면 안 됨 (커널이 죽어도 학습은 살지만, OS 종료엔 죽음)
5. 다음날 ⑥ 평가 (--max-new-tokens 512 내장, 40~75분) → ⑦⑧ 분석
6. 판정: vs exp06 42.06% 기준 +4%p↑ = 승(aug2 확전) / 이내 = 보류 / 하락 = 기각
```

## 4c. 7/16 새벽 — exp12 판정 완료(기각), 미니 3종 체인 진행 중

**exp12 결과 (03:02 평가 완료): shuffled 37.7% vs exp06 42.06% = -4.4%p → CoT SFT 기각.**

- 세그먼트 분해: Type-1 14.7%(exp06과 동일 — 취약 구제 실패), Type-2 43.9%(vs 50.0% **하락**), Type-3 33.3%(vs 35.2%). 이득 구간이 없고, test의 80%인 Type-2에서 가장 크게 잃음
- CoT 형식 자체는 완벽 학습됨 (파싱 실패 0/300, 이벤트 분해·매핑 형식 정확) — 즉 형식이 아니라 **내용이 무익**: target의 매핑에 근거(because)가 없어 "정답을 길게 말하는 법"만 배운 것. 추론 11.5초/샘플(v1의 13배)로 비용만 큼
- 결론: exp13(CoT aug2) 폐기. [Visual Evidence]를 채울 수 없는 한 CoT SFT 재도전 근거 없음 → 자원은 미니 승자·exp08(aug4)·4B로
- 노트북 ⑦이 자동 실행돼 큐 선행 평가와 중복 (experiments.csv 02:48/03:02 두 행, greedy라 결과 동일 — 한 행 무시)

**진행 중인 체인** (`train_queue.py` PID 26564, 독립 프로세스): mini_v1_aug2(기준점) → mini_reorder_aug2(`v5_reorder` 신규 승격) → mini_v4story_aug2 — 각 1000샘플·aug2 미니 학습+평가, 완료 예상 아침 6~7시. 진행: `outputs/train_queue.log` | 결과: `experiments.csv`

**7/16 아침 — 미니 3종 최종 (08:34 체인 완료, 전 항목 성공)**

| 미니 (1000샘플·aug2 동일 예산) | shuffled | vs 기준 |
|---|---|---|
| mini_v1_aug2 (기준점) | 15.5% | — |
| **mini_reorder_aug2 (v5_reorder)** | **21.8%** | **+6.4%p 승** (추론 스크리닝 +6과 정합 — 독립 신호 2개) |
| mini_v4story_aug2 (v4_story) | 15.9% | +0.4%p = 노이즈. 팀 제안 문구는 학습 이득 없음 → 직답판도 기각 |

v4 계열 최종 정리: 문구(v4_story) 무효 + CoT 형식(exp12) 역효과 — **v4 트랙 전체 종료**. reorder(구조 변형)만 생존.
**exp14_reorder_aug2 (exp07 레시피 + v5_reorder, ~10h) 낮 배치가 2번째 큐(PID 17392)로 자동 시작** — 저녁 ~19시 완주+평가. 판정: vs exp07 48.41% (같은 레시피, 프롬프트 구조만 다름 → full끼리 직접 비교 유효). 이기면 주력 교체 후보 + 익일 제출 검증. 밤 슬롯 후보: exp08(aug4) 또는 4B 미니 스크리닝.

**7/17 00:22 — exp14 판정: 총점 동률(48.02% vs exp07 48.41%), 주력 교체 없음. 단, 세그먼트 구성이 크게 다름:**

| | exp14(reorder) | exp07(v1) |
|---|---|---|
| Type-1 (test ~9%) | **35.3%** | 17.6% |
| Type-2 (test ~80%) | 54.3% | **58.5%** |
| Type-3 | 37.0% | 37.0% |

- reorder는 **Type-1을 2배로 구제**하지만 Type-2에서 그만큼 잃음 — test 분포(Type-2 80%)로 가중하면 exp07이 ~1.8%p 우세 → **제출 없음, exp07 유지**
- 미니의 +6.4%p는 full 규모에서 총점으론 소멸 (미니 예산에선 reorder가 빨리 배우는 것뿐, 수렴하면 동률) — **미니 스크리닝의 한계 사례로 기록**
- 두 어댑터를 유형별로 라우팅하면 좋겠지만 **규정상 단일 모델 위반(실격)** — 불가
- 학습 소요 15.6h (3.0초/항목, 낮 시간 스로틀링 추정 — 밤 배치 대비 1.7배 느림, 슬롯 계획 시 참고)

## 4d. 7/17 새벽 — 카메라 표현 발견 + gemma 라벨링 밤샘 가동 (최신)

**핵심 발견 (7/17 01시대, 정규식 실측)**: 문장의 **카메라/촬영 표현 유무가 최강 정확도 인자** — exp07 기준 카메라O 62.5% vs 카메라X 35.6% (**27%p**, 문장 길이 통제 후에도 +16.6%p). test는 카메라O가 43.8%뿐 → **과반이 약점 구간**. 상세: `scratchpad` 분석 + 본 절.

**exp12 기각 사유 정정**: 손실은 spacy 분해가 깨진 문장(전체 48%)에 집중(-7.4%p), 깨끗한 분해에선 동률(-1.5%p) → "분해 품질" 주범, 단 깨끗해도 이득은 없었음 → CoT 재도전은 gemma 라벨 + 미니 게이트 전제.

**gemma 라벨링 완료 (02:15~09:35, 900/900 실패 0)** — 전 과정·발견·활용 계획은 **`REPORT_gemma_labels.md`** 참조. 요점:
- holdout 300 + train 미니 풀(시드42) 앞 600 | `outputs/gemma_labels/labels.jsonl` | 이어받기: `python scripts/gemma_label_sentences.py --train-count 1000` (나머지 400, ~3.3h)
- **camera 축 확정** (+27.3%p, 정규식과 92.1% 일치 → train 전량 가중은 정규식으로) | **viewer는 역방향**(-6.8%p, 병합 금지) | **n_events 계단식** (1개 16.7% → 5개 74.5%) — 이벤트 1~2개(33%)가 진짜 약점
- ⚠️ WSL ollama :11435 + `think: false` 필수 (상세: 보고서 §2)

**남은 할 일**: ① exp15 구현: 카메라X 문장 aug4 / 카메라O aug2 (~26k 항목, 정규식 축) → 스모크 → 본학습 (~20h, 낮 시작이면 익일 아침 판정) ② exp14 제출 여부 결정 (`submission_exp14_reorder_0717_0110.csv` 생성돼 있음 — holdout 근거로는 exp07 우선)

**아침 판정 가이드**
- 미니 3종은 **서로끼리만** 비교 (mini_v1 기준, full 결과와 비교 금지): reorder 또는 v4_story가 mini_v1 대비 +4%p↑ 이기면 오늘 밤 배치 = 그 승자의 aug2 완주(~10h)
- 미니 전패 시 밤 배치 = exp08(aug4, 검증된 축의 연장). 4B 미니 스크리닝(4bit, VRAM 확인)도 낮 슬롯 후보 — 4B 갈 거면 늦어도 7/19-20 시작
- 인프라 변경: `train_queue.py`에 `--pre-eval` 추가(선행 run 어댑터 평가를 큐 맨 앞에) + train_cot.py 감지 버그 수정, `prompts.py`에 `v5_reorder` 추가

## 4e. 7/17 낮 — gemma 전량 확장·구조 파이프라인 구축·규정 정독 (최신)

**gemma 라벨링을 train 전량(9,535 대상)으로 확장** — 300건 단위 `outputs/gemma_labels/parts/part_NNN.jsonl` 롤오버 저장, 기존 labels.jsonl 902건 보존, 이어받기는 두 곳 합산. 순서 = 미니 풀 1000(시드42) 먼저 → 나머지 시드43 셔플 (중간에 끊어도 무작위 표본). **flash attention 재기동으로 8.9초/문장 (어제의 1/3)** — ETA 7/18 오전. ⚠️ 이 작업이 도는 동안 GPU 학습 불가.

**구조 파이프라인 구축 (VISION 실행 인프라, 전부 CPU 작업으로 완료)**:
- `VISION_final_modeling.md`(큰 그림) / `PLAN_post_labeling.md`(단기 실행·게이트) — 문서 계층: VISION → PLAN → HANDOVER
- `Structure_Pipeline.ipynb` — 새 조종석 (라벨 품질 → holdout 약점 기준선 → 레지스트리 → 스모크/본학습/게이지/평가 → **게이트 자동 판정** → 제출)
- `scripts/structure_features.py` — 카메라 정규식(검증본 영구 이식)·gemma 라벨 병합 로더·CLIP 스키마 검증·증강 가중 생성
- `train.py --aug-weights <csv>` — Id별 가변 증강. `outputs/aug_weights_exp15.csv` 생성 완료 (카메라X 4,949→x4 / O 4,286→x2, ~28k 항목)
- CLIP 수령 인터페이스 = `snu_clip_features.csv` 스키마 고정 (이미지 피처는 팀원 담당)

**규정 정독 결과 (`PROJECT_SETUP.md` §4.3) — 중요**:
- **"평가 데이터 특성 분석 후 학습" = 명시적 실격 사유.** 증강 가중·실험 우선순위 근거는 holdout/train 실측만 사용·문서화 (test 분포 인용 금지 — VISION·PLAN·노트북에서 걷어냄)
- 추론 시 전처리 모델(gemma 라벨·CLIP 피처)은 허용 판단 (앙상블 금지 = "추론 결과 조합"의 정의상). gemma4 가중치 공개일(≤2026-05-31) 확인만 남음
- **정정**: `eda/stratified_valid.csv`는 7/14 23:36 커밋에서 삭제됨 — 이후 실험(exp12·미니·exp14)과 gemma 풀은 holdout 300만 제외. §8-5의 "합집합" 서술은 exp06/07 시점에만 유효했던 것

## 4f. 7/18 — 라벨링 완주·유형화 확정·트랙 2 기각·exp16 가동 (최신)

**라벨링 전 과정 완주 (실패 0)**: train 9,535 + test 819. test 파생 = `test_features.csv`(분류)·`test_hints.csv`(추론)·`test_types.csv` — **추론/제출 후 분석 전용** (학습 설계 사용 = 실격).

**문장 유형화 확정** (`REPORT_typing_criteria.md`, `Structure_Typing_EDA.ipynb`): **비카메라 사건 수(≤2/≥3) × camera 4유형** — gemma가 camera를 주어·사건에 포함해 원본 카운트가 camera 축과 교란(camO의 80.8%)됨을 실측, 제거 후 채택. holdout: sparse_camX **21.6%**(train 29%) / sparse_camO 48.7% / dense_camX 52.7% / dense_camO 73.8%. 신규 실측: 무표지 -40.5%p(단일 최강), 다주체는 오히려 쉬움.

**트랙 2 (구조 정보 학습 주입) — 미니 2종 모두 대폭 기각, 트랙 종료**:
- mini_hint(v6, CLIP 유사쌍 입력) 9.9% vs 기준 15.5%: "유사하다" 힌트가 No_ordering 상관의 **identity 지름길을 증폭** (identity 답변 28→46%)
- mini_gemma_cot(v7, gemma events 타깃) 3.2%, identity 81%: 분해 품질을 고쳐도 CoT SFT 무익 — exp12 결론 재확인, CoT 재도전 게이트 완전 종료
- 조합(mini_cot_hint)은 자동 취소. 교훈: **gemma 라벨의 가치는 증강 선별·분석이지 모델 주입이 아님**

**exp16_sparsecam_aug 가동 (7/18 23:28~, ~23h)**: sparse_camX 2,672개 x4 / 나머지 x2 = 23,814항목, v1_list·exp07 레시피 (차이는 `--aug-weights`뿐). 판정 7/19 밤: vs exp07 48.41% ±2%p + sparse_camX 세그먼트 상승 확인. exp15(camX 전체) 원안은 유형 분석 결과로 대체됨.

**운영 사건·수정**: 절전으로 작업 3회 사망 (05:47/08:06 WSL, 17:19 학습) → **독립 프로세스 런처**(절전 차단 내장, 하네스 태스크와 분리)로 전환 + 덮개 설정 변경(사용자). train_queue **자기 감지 데드락**(--script 문자열) 수정 — 함정 목록 참조.

**남은 일정 (마감 7/24)**: exp16 판정(7/19 밤) → 승자 제출 검증. 4B는 exp16이 GPU를 잡는 관계로 7/19 밤 미니 스크리닝이 마지막 진입 기회 — 미실행 시 자동 폐기. 제출 잔여 확인 필요.

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
11. **CoT 계열 평가는 `--max-new-tokens 512` 필수** — 32로 돌리면 잘려서 전부 파싱 실패 (노트북 eval_cmd에 내장됨)
12. **추론 노트북 커널이 GPU를 계속 잡는다** (모델 ~4.7GB) — 학습 시작 전 다른 노트북 커널 종료/재시작 필수. 7/15에만 두 번 겪음 (VRAM 대기 루프의 원인)
13. **백그라운드로 죽인 프로세스가 VRAM을 몇 분 쥐고 있을 수 있음** — nvidia-smi로 확인 후 진행
14. **ollama 러너가 클라이언트 Ctrl-C 직후 먹통(wedge)이 될 수 있음** — 증상: 스크립트가 로그 없이 조용 + `ollama ps`엔 모델이 떠 있는데 직접 curl 생성 요청도 무응답. 해법: ollama 완전 재시작 (`ollama stop`으로는 안 풀림). 7/17 실제 발생
14b. **WSL 안에서 도는 작업은 절전을 못 막는다** — train.py의 keep_awake는 Windows 프로세스에만 유효. WSL 작업(ollama·라벨링)은 절전 시 VM째 죽음 (7/18 새벽 2회 발생). 해법: Windows 쪽에서 `SetThreadExecutionState(0x80000001)` 유지 루프를 별도로 상주시킬 것
15. **pandas에서 `df.gt` 열 접근 금지** — `.gt()`(greater-than) 메서드와 충돌해 KeyError. 반드시 `df["gt"]`
16. **train_queue의 학습 감지는 자기 자신을 제외해야 함** — `--script train.py` 인자 문자열이 큐 프로세스 커맨드라인에 들어가 자기를 학습으로 감지 → 영구 대기 데드락 (7/18 발생, 수정됨). 유사 패턴 감지 코드를 짤 때 주의
18. **특정 샘플이 VRAM 스파이크로 OOM을 낸다** — 7/19 exp16이 시드 고정 순서에서 **같은 540스텝에 2회 재현 실패**. 평소 peak 6.47GB인데 8GB(가용 7.3GB)라 스파이크 한 번에 넘어감. 대응: train.py에 OOM 샘플 자동 스킵 + `expandable_segments` 추가(7/19). **같은 지점 반복 실패 = 외부 요인이 아니라 데이터 요인**이라는 진단 단서로 기억할 것 (1차 실패를 ollama 탓으로 오진했음)
17. **긴 백그라운드 작업은 독립 프로세스로** — 세션(하네스) 태스크는 절전·세션 이벤트에 죽을 수 있음. `Start-Process`로 분리 + 절전 차단 내장 런처 패턴 사용 (scratchpad `run_exp16.ps1` 참조), 진행 감시는 로그 파일 테일로

## 8. 다음 단계 (우선순위순, 7/17 오전 갱신)

1. **exp15 타깃 증강**: 카메라X aug4 / 카메라O aug2 — 구현→스모크→본학습(~20h) → 익일 판정 (vs exp07 48.41%)
2. **4B 스케일업 미니 스크리닝** (4bit, VRAM 확인 포함) — 갈 거면 늦어도 7/19-20 시작해야 마감 내 완주
3. 보류 카드: exp08(균일 aug4, exp15 실패 시) / CoT 재도전(gemma events 미니, `REPORT_gemma_labels.md` §7) / DPO(슬롯 남으면)
4. 제출 운용: exp15 승리 시 1회는 exp15 검증에. exp14 파일은 생성돼 있으나 holdout 근거로는 exp07 우선
4. 팀 병행: Type-1 수동 상한 확인(단, test 레버리지 9%로 하락 반영), test 819 CLIP 피처 생성
5. 검증셋 기준 (합의됨): **holdout_300이 공식 채점용**, stratified_valid는 유형별 보조 분석용, 학습 제외는 두 셋의 합집합 (train.py·train_cot.py 모두 반영)

## 9. 팀 분업 참고

- **모델링(GPU 보유자)**: 학습·평가 실행 전담 — "설계 요청 → 밤 배치 → 결과 공유" 사이클
- **텍스트 EDA**: `outputs/errors_*.csv`(이제 출력 전문 포함) 오답 유형 분석, 프롬프트 후보 제안 (`prompts.py`에 추가만 하면 실험 가능)
- **이미지 EDA**: 해상도-정확도 트레이드오프, 프레임 유사도, 후반 재현성 패키징
- 규정: 최종 모델은 단일 모델 (실험 비교는 OK, 결과 조합=앙상블=실격)

## 10. 기타

- git: 7/14 커밋(cc14dfc)까지. **7/15 변경분 대량 미커밋** — prompt_lab.py, train_cot.py, 노트북 2개, PLAN_*.md 3개, prompts.py/eval_zero_shot.py 수정 등. exp12 스모크 통과 확인 후 커밋 권장
- 외부 API 사용액: 0원
- 예선 마감 **7/24** — 남은 9일 기준 본학습 슬롯 최대 7~8회, 제출 잔여 ~18회 (오늘 2회 소진). 미니 학습 스크리닝으로 아껴 쓸 것
