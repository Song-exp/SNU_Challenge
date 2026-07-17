# 전처리 파이프라인 검토 & 프롬프트 실험 계획 (2026-07-15)

기준 모델: **exp07_aug2_full** (Qwen3-VL-2B + LoRA, v1_list 학습, holdout shuffled **48.4%**)
관련 문서: `pipeline_data_mapping.md`(전처리 설계), `EXPERIMENTS.md`(실험 규칙)

---

## 1. 실측 검증 — 전처리 설계의 타깃팅은 유효하다

holdout 300 × exp07 예측 결과에 `src/features/flag_detector.py` 분류기를 교차한 결과 (shuffled 252개):

### 문장 유형(Partition)별 exp07 정확도
| Partition | n | acc |
|---|---|---|
| Type-1 (단일 절) | 34 | **17.6%** |
| Type-2 (복합 종속) | 164 | 58.5% |
| Type-3 (대등 병렬) | 54 | 37.0% |

### ai_score 구간별 — 정확도와 거의 완벽한 단조 관계 (난이도 지표로 유효)
| ai_score | n | acc |
|---|---|---|
| ≤0.3 | 114 | 63.2% |
| 0.3–0.5 | 105 | 41.9% |
| 0.5–0.7 | 8 | 25.0% |
| >0.7 | 25 | **16.0%** |

### 보조 세그먼트 (참고)
- 시간 키워드 있음 180개 58.9% vs 없음 72개 **22.2%**
- 36단어 이상 75.0% vs 15단어 이하 **22.4%**
- N7_ordinal(서수) 표본 4개 전부 오답 (표본 작음, 관찰 지속)

**결론**: "부족/취약 유형만 증강 + 힌트 주입" 타깃팅의 근거가 실증됨.
학습·추론 양쪽에 동일 힌트를 주입하는 설계는 분포 이탈 함정을 피하는 올바른 구조.
모든 피처가 Sentence·이미지에서만 나오므로 test 적용 가능, 정답 누수 없음.

---

## 2. 검토 권고 4가지

1. **Type-1 증강 전 "정보 상한" 확인 (최우선)** — Type-1 오답 ~28개를 사람이 직접 보고
   "이미지+문장으로 풀 수 있는 문제인가" 확인 (~30분). 풀 수 없는 비율이 높으면
   증강은 노이즈 학습(정렬 불가 샘플 정답 암기)이 됨 → 증강 대신 모호 케이스
   CLIP 유사도 폴백 전략으로 전환.
2. **"실시간 계산" 대신 "사전 계산 + 조회"** — train/test 모두 고정 데이터.
   실시간 계산은 학습·추론 간 환경 스큐 위험만 키움 (실례: 학습 .venv에 spacy 부재했음,
   7/15 설치 완료). `extract_features.py`로 train+test 전체를 1회 계산해 CSV 저장,
   학습·추론 모두 조회. CLIP은 train용(`snu_clip_features.csv`) 존재 — **test 819개용 추가 생성 필요**.
3. **힌트는 한 번에 하나씩, CLIP 유사쌍부터** — 텍스트 플래그는 모델이 문장에서 어느 정도
   스스로 추출 (시간 키워드 세그먼트 이미 58.9%). CLIP 프레임 유사쌍은 모델이 스스로
   계산하기 어려우면서 정렬에 직접적 단서 (인접 프레임 = 시각적 유사). 우선순위:
   ① CLIP sim_pairs 한 줄 → ② 플래그 요약 한 줄. 각각 `prompts.py` 등록 후
   미니 학습(`--max-samples 1000 --max-steps 300`, ~1.5h) 스크리닝 → 승자만 밤 배치.
4. **차등 증강 부작용 감시** — 유형별 차등 증강은 학습 분포를 test 분포(Type-2 63%)와
   다르게 만듦. 모든 평가에 Partition별 분해를 붙여 Type-2 하락을 함께 판정.

### 실험 순서
- **exp09 (힌트 주입)**: CLIP 힌트 프롬프트 미니 스크리닝 → 승자 밤 배치
- **exp10 (차등 증강)**: Type-1 수동 상한 확인 통과 시에만
- 조합은 각각 단독 승리 후

### 소소한 정리 항목
- 문서 경로 불일치: `eda/sentence_type_labels.csv` 미생성, `count_3stage*.py`·`snu_clip_features.csv`는 루트
- 힌트 주입 시 `train.py`/`eval_zero_shot.py`의 `build_user_text(prompt_name, sentence)`가
  피처도 받도록 인터페이스 확장 필요 (소폭 수정 + 스모크 필수)

---

## 2.5 제안 기법 4종 검토 (7/15, 팀 제안)

| 기법 | 판정 | 근거·조건 |
|---|---|---|
| ① 추론 시 Visual CoT 유도 | ⚠️ 소액 베팅만 | zero-shot v3_cot는 shuffled 0%·파싱실패 297/300 전례. exp07은 리스트 즉답만 학습 → 분포 이탈. 샘플당 ~8초(9배). **Round 3에 변형 1개로 실측** — `max_new_tokens=256` 필수, 출력은 `[2,4,1,3]` 형식 유지(파서 호환) |
| ② CoT 응답 Instruction Tuning | ⚠️ 라이트 버전 먼저 | CoT 정답지 9천 개 작성 불가·템플릿 생성은 이미지 안 보고 쓴 추론문이라 헛소리 학습 위험. **exp09(힌트 주입, 응답은 리스트)가 안전한 라이트 버전**. 풀버전은 exp11 조건부: exp09 힌트 효과 + Round 3 CoT 신호 + 근거문 자동생성 방법 3조건 충족 시 |
| ③ Hard Negative Mining | ✅ 채택 | exp10을 이걸로 업그레이드: 정보상한 노이즈 Drop + Type-3(37.0%) 우선 오버샘플링. **전제: train-set 9,035개 오답 마이닝**(holdout은 학습 사용 금지) ~2.5h 1회. train.py 샘플별 가중 복제 지원 수정 필요 |
| ④ DPO | 📋 백로그 | SFT만으로 peak 6.99GB/8GB — chosen·rejected×policy·ref 4배 forward는 TRL PEFT 통합(ref=어댑터 비활성)으로만 가능성 있음. preference pair도 ③의 오답 마이닝 선행 필요. exp09/10 정체 시 카드. 사전에 30분 타당성 체크(TRL+Qwen3-VL+4bit 스텝 구동)만 |

**실행 순서 반영**: 지금(Round 1~3 + CoT 변형) → GPU 빈 시간(오답 마이닝 1회, ③④ 공용)
→ exp09(②라이트) → exp10(③) → exp11(②풀, 조건부) → DPO(백로그)

---

## 3. 전처리 완료 전 프롬프트 추론 실험 (exp07 고정, 병행 트랙)

**전제**: exp07은 v1_list로 학습됨 → 추론 프롬프트 대변경은 분포 이탈로 하락이 기본 기대값.
목적은 ① 민감도 곡선 확보, ② 취약 세그먼트 라우팅 이득 탐색, ③ **힌트 문구 사전 스크리닝**
(여기서 덜 깨지고/취약 세그먼트에서 이득 보는 형식 = exp09 학습 후보).

상세 실행 계획은 **`PLAN_prompt_experiments.md`** 참고 (아래는 요약).

### Round 0 — 인프라 (30분)
- `Prompt_Experiments.ipynb` 신규 (Train_Experiments와 분리 — 모두실행 사고 방지)
- 모델+exp07 어댑터 1회 로드, `eval_zero_shot.py` 파싱/채점 재사용, `do_sample=False` 유지
- v1_list 재현으로 48.4% 확인 (greedy라 정확히 일치해야 정상)
- 기록: `outputs/prompt_experiments.csv` (본 experiments.csv와 분리)
- 소요: 변형당 300개 × ~0.9초 ≈ 5분

### Round 1 — 전역 소변형 민감도 (~15분, 3개)
- `v1_causal`: v1 + "If the sentence has few time cues, infer the order from cause and effect."
- `v1_visual`: v1 + "Pay attention to visual state changes across the images."
- `v1_reorder`: 구조 변형 대조군 (이탈 크기 측정용)

### Round 2 — 케이스 라우팅 (핵심, 변형당 ~2분)
- 라우터: **ai_score > 0.5 또는 Type-1** (holdout 기준 ~40개, 현재 acc ~17%)
- 해당 샘플만 재추론, 나머지는 v1 결과 유지 → 하락 위험이 취약 세그먼트로 국한
- 판정: paired 비교(새로 맞춤 vs 새로 틀림), 전체 shuffled **+4%p(≈10문제) 이상**이면 채택

### Round 3 — CLIP 힌트 형식 스크리닝 + CoT 소액 베팅 (exp09/exp11 준비)
- v1 + `Hint: visually similar pairs: (2,3), (1,4)` 류의 한 줄 힌트를 추론에 주입
- 학습 없이 넣는 것이므로 하락 예상 — **절대치가 아니라 형식 간 상대 비교**로
  exp09에 등록할 힌트 문구 후보를 고른다 (취약 세그먼트 델타 기준)
- **CoT 변형 1개** (§2.5 기법①): 힌트 확인→추론→리스트 순 유도. `max_new_tokens=256` 필수,
  최종 출력은 `[.., ..]` 리스트 형식 유지. 샘플당 ~8초 → 취약 세그먼트 ~40개만 돌려도 판정 가능

### 판정·출구
- 라우팅 성공(+4%p↑) → submission 파이프라인에 라우팅 채택
- 실패해도 Round 3 순위는 exp09 프롬프트 등록의 근거
- 결과는 `EXPERIMENTS.md`와 이 문서에 갱신

---

## 4. 남은 TODO (전처리 트랙)
- [ ] Type-1 오답 수동 상한 확인 (질적 분석, 진행 중)
- [ ] test 819개 CLIP 피처 생성
- [ ] `extract_features.py`로 train+test 통합 피처 CSV 확정
- [ ] `build_user_text` 인터페이스 확장 (피처 주입)
- [ ] test 추론 → submission.csv 생성 스크립트 (아직 없음)
- [ ] **train-set 9,035개 exp07 오답 마이닝** (~2.5h, GPU 빈 시간 1회 — exp10 하드네거티브·DPO 공용 전제)
- [ ] train.py 샘플별 가중 복제(오버샘플링) 지원 수정 + 스모크
- [ ] (백로그 진입 조건 검토용) TRL DPO + Qwen3-VL 2B + 4bit 8GB 구동 타당성 체크 30분
