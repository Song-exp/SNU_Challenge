# 프롬프트 추론 실험 상세 계획 (exp07 고정) — 실행: `Prompt_Experiments.ipynb`

2026-07-15 확정. 배경·기법 검토는 `PLAN_prompt_and_preprocessing.md` §2.5·§3, 실험 규칙은 `EXPERIMENTS.md`.

## 목적·전제
- **목적**: 가중치 업데이트 없이 ① 프롬프트 민감도 곡선 확보, ② 취약 세그먼트 라우팅 이득 탐색,
  ③ exp09(힌트 SFT)·exp11(CoT SFT)에 등록할 문구의 사전 스크리닝.
- **전제**: exp07은 v1_list로 학습 → 대변경은 하락이 기본 기대값. 절대치보다 **상대 비교 + paired**.
- **기준선**: exp07 v1_list, holdout 300, shuffled **48.41%** (greedy라 결정론적 — 기존 예측 파일 재사용).
- **취약 세그먼트**(라우팅 대상): `Type-1 또는 ai_score>0.5` — 실측 acc 16~18% 구간.

## 인프라 (구축 완료)
| 파일 | 역할 |
|---|---|
| `scripts/prompt_lab.py` | 헬퍼 모듈: holdout+피처 로드, 모델 1회 로드, 변형 실행/채점/기록, 라우팅 합성, 세그먼트 분해 |
| `Prompt_Experiments.ipynb` | 실행 조종석 (변형 레지스트리 = ③셀, 실험 추가 = 한 줄) |
| `outputs/prompt_experiments.csv` | 변형별 요약 누적 (acc_shuffled, acc_weak, Type별, fixed/broken, parse_fail) |
| `outputs/preds/prompt_<name>.csv` | 변형별 예측 전문 (raw 포함 — CoT 질적 분석용) |

파싱·채점은 `eval_zero_shot.py`와 동일 로직 재사용(마지막 대괄호 파싱 → CoT 호환).
피처(Partition/ai_score/CLIP 거리)는 노트북 시작 시 메모리에서 계산·병합 (별도 파일 생성 없음).

## 실험 파라미터 레퍼런스

### 모델·추론 설정 (`prompt_lab.load_model` / `_infer_one`)
| 파라미터 | 값 | 설명·주의 |
|---|---|---|
| model | `./models/Qwen3-VL-2B-Instruct` (fp16) | exp07의 베이스. fp16 로드 ≈ 4.7GB VRAM |
| adapter | `./outputs/runs/exp07_aug2_full/adapter` | LoRA r=16. 실험 내내 고정 (가중치 업데이트 없음) |
| `max_pixels` | **640×480 = 307,200** | 프로세서가 이미지를 이 픽셀 수 이하로 리사이즈. vision 토큰 수·VRAM·속도를 결정. **학습(exp07)·기존 평가와 동일값이라야 비교 유효** — 바꾸면 baseline 재현부터 깨짐 |
| `max_new_tokens` | 32 (기본) / 256 (경량 CoT) / 512 (풀 CoT) | 생성 토큰 상한. 리스트 즉답은 32면 충분. CoT에 32를 주면 추론 과정에서 잘려 **전부 파싱 실패** (v3_cot 297/300 전례) |
| `do_sample` | **False (greedy)** | 결정론적 — 같은 입력이면 항상 같은 출력. baseline 재추론 없이 기존 예측과 paired 비교가 가능한 근거. 온도 샘플링 도입 시 이 전제 전부 무효 |
| `wait_for_free_vram` | 5.5GB | 다른 프로세스(학습 등)가 GPU 점유 중이면 확보될 때까지 자동 대기 |

### 데이터·세그먼트 정의 (`prompt_lab.load_holdout`)
| 항목 | 값 | 설명 |
|---|---|---|
| 평가셋 | `splits/holdout_300.csv` (300개) | 학습에서 제외된 내부 검증셋. shuffled 252 + identity(No_ordering=True) 48 |
| `Partition` | Type-1/2/3 | `flag_detector.classify_syntax_spacy` — 절 구조 3분류 (노트북 시작 시 실시간 계산) |
| `ai_score` | 0.0~1.1 | 통사 구조 + N플래그 가감점으로 산출한 모호성 점수. holdout 실측에서 정확도와 단조 관계 |
| `weak` (라우팅 대상) | **Type-1 OR ai_score>0.5** | 41개 (shuffled 34개), baseline acc 22.0% — 실측된 최약 세그먼트. R4는 이 41개만 재추론하고 나머지 259개는 baseline 결과 유지 |
| CLIP 힌트 재료 | `snu_clip_features.csv`의 `dist_12`~`dist_34` | 프레임쌍 6개의 CLIP 코사인 거리. `clip_pairs_text(row, k=2)` = 거리 최소 2쌍을 "Image a & Image b" 문자열로 |

### 채점·파싱 (`eval_zero_shot.parse_model_output` 재사용)
| 항목 | 규칙 |
|---|---|
| 파싱 | 출력에서 **마지막 대괄호 그룹**을 리스트로 해석 → CoT처럼 중간에 대괄호가 섞여도 최종 답만 집음. `<ANSWER>[..]</ANSWER>`도 이 규칙으로 파싱됨 |
| 유효성 | 파싱 결과가 1~4의 순열이 아니면 실패 처리 → 예측을 [1,2,3,4]로 대체하고 `parsed=False` |
| `parse_fail` | 파싱 실패 샘플 수. 급증 = 프롬프트가 출력 형식을 붕괴시켰다는 신호 (정확도보다 먼저 볼 것) |

### 기록 지표 (`outputs/prompt_experiments.csv` 컬럼)
| 컬럼 | 의미 | 판정에서의 역할 |
|---|---|---|
| `accuracy` | 전체 300개 정확도 | 참고용 (identity 48개 포함이라 부풀려짐) |
| `acc_shuffled` | **핵심 지표** — 섞인 252개 정확도 | 무작위 4.2%, baseline 48.41%, ±4%p(≈10문제) 이내는 노이즈 |
| `acc_weak` | 취약 41개 정확도 | 힌트/라우팅이 노리는 표적. n이 작아 등락이 큼 → fixed/broken과 같이 볼 것 |
| `acc_type1/2/3` | Partition별 정확도 | **풍선 효과 감시** — Type-1이 올라도 Type-2(핵심 물량, baseline 58.5%)가 내려가면 총점 손해 |
| `fixed` / `broken` | baseline 대비 **새로 맞춘 / 새로 틀린** 문제 수 (paired, 같은 300개·greedy라 1:1 비교) | **채택 기준: fixed − broken ≥ +10** (전체 +4%p 상당). acc 델타보다 민감하고 노이즈에 강함 |
| `routed_n` | 라우팅 실험에서 실제 재추론한 샘플 수 (전역 실험은 공란) | 41이면 R4/R5 취약 라우팅 |
| `eval_n` | 항상 300 (라우팅도 나머지는 baseline 합성) | 변형 간 동일 분모 보장 |
| `sec_per_sample` | 샘플당 추론 시간 | 제출 비용 환산: test 819개 × 이 값 (v1 ~0.9초=13분, 풀CoT ~8초=2h) |
| `max_new_tokens` | 해당 변형의 생성 상한 | CoT 계열 식별용 |

예측 전문(`outputs/preds/prompt_<name>.csv`)에는 `raw`(모델 출력 원문), `pred`, `correct`,
`parsed`, `Partition`, `ai_score`, `weak`, `Sentence`가 남는다 — CoT가 지시를 따랐는지는
raw를 직접 봐야 안다 (스모크에서는 무시하고 리스트 즉답).

## 라운드 구성

순서 (7/15 확정): **R0 검증 → R1 기본 소변형 → R2 v4 분해 → R3 주입형 힌트 → R4 라우팅 → R5 CoT 베팅**
(전역 변형으로 승자 요소를 먼저 가려낸 뒤, 그 승자를 주입·라우팅에 투입하고, 비용 큰 CoT는 맨 뒤)

### R0 — 파이프라인 검증 (5분)
`r0_v1_baseline`: v1_list 전체 300 재추론 → **48.41%와 일치해야 이후 비교 유효** (assert 내장).

### R1 — 기본 프롬프트(v1) 소변형 민감도 (15분, 3개)
| 변형 | 내용 | 보는 것 |
|---|---|---|
| `r1_causal` | v1 + 인과 단서 유도 1문장 | 취약 세그먼트(단서 빈약 문장) 델타 |
| `r1_visual` | v1 + 시각 상태변화 주목 1문장 | 〃 |
| `r1_reorder` | 문장을 지시문 뒤로 옮긴 구조 변형 | **대조군** — 이탈 크기 측정. 크게 무너지면 이후 변형은 문구 추가 수준으로 제한 |

### R2 — v4(팀 제안 7/15) 기능 분해 (~22분)
팀 제안 "스토리텔러 CoT 프롬프트"에서 **당장 추론에 쓸 수 있는 요소만 분리**해 개별 실측.
여기서 이기는 요소가 R3 주입·R4 라우팅의 재료가 된다.
| 변형 | 분리한 기능 |
|---|---|
| `r2_role` | 역할 부여(expert visual storyteller)만 v1에 추가 |
| `r2_story` | "Storyline" 프레이밍만 (역할·단서 없음) |
| `r2_cues` | 시각 단서 가이드(object states/actions/background)만 v1에 추가 |
| `r2_combo` | 셋 다 결합 = `prompts.py`의 **`v4_story`** (미니 학습 후보) |

### R3 — 현재 exp07에 적용 가능한 주입형 힌트 (~5분+)
- `r3_clip_pairs` (전체 300): v1 + "가장 유사한 이미지쌍 (CLIP dist 최소 2쌍)" 한 줄.
  학습 없이 주입이라 하락 예상 — **형식 간 상대 비교(acc_weak 기준)로 exp09 등록 후보 선별**이 목적.
- R2 승자 요소가 있으면 `v1 + 승자요소 + CLIP 힌트` 조합 변형을 추가.

### R4 — 취약 세그먼트 라우팅 (핵심, 변형당 ~1분)
- 취약 41개만 재추론, 나머지 259개는 baseline 결과 유지 → 하락 위험 국소화.
- 기본 투입: `r4_route_causal`, `r4_route_clip`. R1~R3 승자를 `subset="weak"`으로 추가 투입.
- **채택 기준: fixed − broken ≥ +10문제 (전체 shuffled +4%p 상당)** → submission 파이프라인에 라우팅 채택.

### R5 — CoT 소액 베팅 (취약 41개만, ~7분)
- `r5_cot256_weak`: 경량 단계 유도(256토큰) | `r5_fullcot_weak`: 팀 제안 풀버전 원문(`v4_story_cot`, 512토큰)
- 스모크에서 exp07이 단계 지시 무시(리스트 즉답) 확인 → 기대치 낮음, 정량 기록 목적.
  양수 신호(fixed>broken)면 CoT 학습 실험(아래 "다음 단계") 우선순위 상향 근거.

각 변형의 개선폭은 baseline(v1_list 48.41%) 대비 acc_shuffled 델타 + paired(fixed/broken)로 기록.
결과는 아래 "실측 결과" 섹션에 갱신.

## 판정 규칙
- 핵심 지표 `acc_shuffled` (무작위 4.2%, ±4%p 노이즈). 전 변형에 paired(fixed/broken) 병기.
- 세그먼트 분해(Type-1/2/3, weak/strong)를 모든 판정에 첨부 — **Type-2 하락(풍선 효과) 감시**.
- 취약 세그먼트 단독 판정은 n≈46이라 ±4%p로는 부족 → **fixed−broken 차이**로 판단.
- `parse_fail` 급증 = 형식 붕괴 신호 (특히 CoT).

## 출구
| 결과 | 다음 행동 |
|---|---|
| 라우팅 +4%p↑ | submission 파이프라인에 라우터 포함 (전처리 트랙의 피처 사전계산과 연결) |
| CLIP 힌트가 형식 중 최선 | 해당 문구를 `prompts.py`에 등록 → exp09 미니 학습(`--max-samples 1000 --max-steps 300`) |
| CoT 양수 신호 | exp11 조건 1/3 충족 기록 |
| 전부 무효 | 추론 프롬프트는 v1 확정, 자원을 exp09/10(학습 트랙)에 집중 |

결과는 이 문서와 `EXPERIMENTS.md`에 갱신. GPU 규칙: 학습 배치와 동시 실행 금지
(모델 로드 셀이 VRAM 확보까지 자동 대기).

---

## 다음 단계 (확정, 7/15 지시): CoT 프롬프트 학습 실험 — exp12_v4cot

R1~R5 실측 종료 후 진행. "v4_story_cot 프롬프트 + 단계 응답 target"으로 SFT.
**→ 7/15 구현 완료: 상세 설계·커맨드는 `PLAN_cot_finetune.md`** (train.py 무변경, `scripts/train_cot.py` 신규)
(exp09 CLIP 힌트·exp10 증강과 별개 트랙 — 실험 번호는 레지스트리 등록 시 확정)

### 학습 target 생성 레시피 — 헛소리 없는 기계 생성 원칙
CoT 응답 9천 개를 사람이 못 쓰므로 기계 생성하되, **정답·문장에서 역산 가능한 사실만** 넣는다:
| 섹션 | 생성 방법 | 근거 유무 |
|---|---|---|
| [Story Analysis] | spacy 절 분해로 문장을 이벤트 목록화 | ✅ 텍스트에서 유도 |
| [Chronological Mapping] | 정답 Answer에서 역산 — "1st event is Image k" | ✅ 정답 자체 (단, "because ..."는 넣지 않음) |
| [Visual Evidence] | ❌ 이미지를 안 보고는 못 씀 → **1차 버전에서 제외** (또는 CLIP 유사쌍 문장으로 대체) |
| [Final Answer] | `<ANSWER>[...]</ANSWER>` | ✅ |

### 필요한 코드 수정 (prompts.py 헤더 규칙대로 세트 수정)
1. `train.py`: target_text 생성부 — prompt가 v4_story_cot일 때 단계 응답 생성
2. `eval_zero_shot.py` 파서: `<ANSWER>` 태그 우선 + 마지막 대괄호 폴백 (기존 프롬프트와 호환 유지)
3. 평가 `--max-new-tokens 512`

### 절차·판정
1. 스모크 (~3분): target 형식·VRAM(응답 길이 증가) 확인
2. **미니 학습 스크리닝** (~1.5h): v4_story_cot vs v4_story(직답) vs v1_list — 같은 1000샘플/300스텝
3. 승자만 밤 배치 완주 → holdout 평가 (acc_shuffled + parse_fail + Type별 분해)
4. 비용 항목 명시: CoT는 제출 추론이 819개 × ~8초 ≈ 2h (v1은 13분) — 채택 시 감수 여부 판단

### 실측 결과 (R0~R5, 7/15 실행 완료)

baseline: v1_list **48.41%** (shuffled) / weak 22.0% / Type-2 61.4% / Type-3 43.6%

| 변형 | acc_shuffled | Δ | acc_weak | acc_type2 | fixed−broken | 비고 |
|---|---|---|---|---|---|---|
| r1_reorder | **49.60%** | +1.2 | 24.4% | 61.9% | **+6** | 전체 1위. 문장을 지시문 뒤로 — Type-3 +6.5%p |
| r5_cot256_weak | 49.60% | +1.2 | 29.3% | 61.4% | **+3/−0** | 취약 41개만. 유일하게 broken 0 |
| r4_route_causal | 48.81% | +0.4 | 24.4% | 61.4% | +1 | 미미 |
| r0 / r4_route_clip | 48.41% | 0 | 22.0% | 61.4% | 0 | — |
| r2_cues | 48.02% | −0.4 | 24.4% | 61.4% | 0 | 중립 |
| r5_fullcot_weak | 48.02% | −0.4 | **12.2%** | 61.4% | −4 | 풀 CoT — 취약에서 절반 붕괴, 15초/샘플 |
| r2_story | 47.62% | −0.8 | **34.1%** | 57.4% | −3 | weak 최고(+12.2%p)·Type-2 −4%p 트레이드오프 |
| r1_visual | 47.22% | −1.2 | 26.8% | 60.4% | −2 | 노이즈 |
| r1_causal / r2_role | 46.43% | −2.0 | 24~27% | 58~61% | −2/−5 | 하락 |
| r2_combo (v4 직답판) | 46.03% | −2.4 | 29.3% | 56.9% | −8 | 요소 결합이 오히려 최악권 |
| r3_clip_pairs | 46.03% | −2.4 | 22.0% | 56.4% | −11 | CLIP 힌트 무학습 주입은 무효+해악 |

**판정: 어떤 변형도 채택 기준(fixed−broken ≥ +10) 미달 → 추론 프롬프트는 v1_list 유지 확정.**

수확 (다음 실험의 근거):
1. **r1_reorder를 학습 프롬프트 후보로 등록** — 유일하게 전 세그먼트 무손실 +6.
   v1과 문장 위치만 다르므로 미니 학습 스크리닝 1순위 후보.
2. **CoT는 학습으로만** 확정 — 풀 CoT는 형식(<ANSWER>)은 완벽히 따르지만(parse_fail 0)
   추론 품질이 없어 취약 세그먼트를 절반으로 붕괴시킴. 경량 CoT의 +3/−0은 CoT SFT(exp12) 약한 양성 신호.
3. **CLIP 힌트 무학습 주입은 무효** (weak 개선 0, Type-2 −5%p) — exp09는 학습 세트로만 검증, 기대치 하향.
4. r2_story의 weak +12.2%p는 유일하게 큰 취약 개선 — Type-2 손상 없는 **라우팅 버전(r4_route_story) 1분 확인 가치** 있음.
   스토리 프레이밍도 학습 후보 재료.

### 첫 리더보드 제출 결과 (7/15 20:00, Public 70%)

| 제출 | holdout(전체/shuffled) | **리더보드** |
|---|---|---|
| r0_v1_baseline | 52.3% / 48.4% | **0.76265** |
| r2_combo | 49.7% / 46.0% | **0.76614** |

**해석 (제출 파일 분석 + 분포 비교로 검증):**
- 리더보드 76% >> holdout 52%의 원인은 **분포 차이**: test 문장은 평균 42단어(train 24),
  26단어 이상이 80%(train 54%), Type-2가 80%(train 63%) — 우리 실측 길이-정확도 곡선
  (26-35단어 63.6%, 36+단어 75.0%)에 그대로 부합. identity 비율은 test도 ~16%로 동일
  (제출 예측의 [1,2,3,4] 비율 15~16% = holdout과 일치).
- combo↔baseline 격차 +0.35%p ≈ Public 573개 중 **2문제 = 동률(노이즈)**. holdout의
  −2.4%p도 노이즈 경계. 순서 뒤집힘이 아니라 "둘 다 구분 불가"가 맞는 결론 → **v1 유지**.
- holdout은 스크리닝용으로 계속 유효 (절대값이 아니라 상대 비교 도구).

**전략 레버리지 재계산 (test 분포 기준):**
- Type-1/취약 세그먼트 개선(exp10 증강·라우팅): test의 ~9% → **우선순위 하락**
- **Type-2·장문 개선: test의 80% → 최우선.** CoT의 Story Analysis(다절 분해)는 장문
  Type-2에 정확히 부합 — exp12 가설을 "취약 구제"에서 "장문 다절 강화"로 재조준
- 전역 개선(exp08 aug4, 4B 스케일업)도 고배당 유지
