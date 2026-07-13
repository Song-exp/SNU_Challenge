# 신규 문장 유형 발굴 보고서 (선행연구 기반 Type Discovery Report)

본 문서는 현행 문장 유형 체계(절 개수 기반 3유형: 단일 절 / 다중 절-종속 / 다중 절-병렬, `eda_문장모호성.md` 참조) 바깥에 존재할 수 있는 **신규 유형 후보를 선행연구에 근거해 정의**하고, 각 후보가 실제 train 데이터에 존재하는지 **어휘 프로브(Lexical Probe)로 실측 검증**한 결과를 정리한 보고서입니다.

**파이프라인상 위치**: ① 신규 유형 정의(본 문서) → ② 프로브 어휘로 후보 샘플 발굴(본 문서, 실측 완료) → ③ 질적 분석(블라인드 리뷰) → ④ 학습/추론 → ⑤ 오류 슬라이스 분석으로 유형 재갱신(8장)

---

## 1. 유형 체계 설계 원칙: "파티션 확장"이 아니라 "직교 플래그 추가"

### 1.1 왜 새 유형을 4번째, 5번째 파티션으로 넣으면 안 되는가

현행 3유형은 **통사 구조(절 개수)** 라는 단일 차원의 상호배타적 분할(partition)입니다. 반면 아래에서 발굴한 신규 유형들은 대부분 **담화·의미 차원**이라 절 구조와 직교(orthogonal)합니다. 예를 들어 "The camera pans left, then shifts to a rear view"는 다중 절이면서 동시에 카메라 담화입니다. 이런 유형을 파티션에 추가하면:

- 한 문장이 여러 유형에 걸리는 충돌이 발생하고, 우선순위 규칙이 늘어나며 규칙 사전이 train에 과적합됩니다.
- 유형 수가 늘수록 유형별 샘플이 희소해져 통계적 판정력이 붕괴합니다 (기존 Static-State 19개 문제의 재현).

### 1.2 권장 구조: 1차 파티션 + 직교 플래그 세트 (Multi-label)

```
문장 표현 = [1차 파티션: 절 구조 3유형 (상호배타)] + [플래그 벡터: 신규 유형들 (독립 이진, 중복 허용)]
예: "The athlete transitions from a dark vest to a red vest as the camera shifts focus, then vaults..."
    → 파티션: 다중 절-종속 | 플래그: CAM=1, STATE_CHG=1, PHASE=1, ITER=0, ...
```

- **1차 파티션(절 구조)**: purity 분석(문장유형 × 이미지 씬 구조)으로 유효성 검증 — 기존 계획 유지.
- **플래그(신규 유형)**: 각각 독립적인 정규식 사전으로 탐지하는 이진 피처. 상호배타 분류가 아니므로 충돌 규칙이 불필요하고, 개별 플래그 단위로 채택/폐기 판정이 가능합니다.
- 학습 전략 분기(라우터)는 파티션과 플래그 조합 위에서 **사후적으로, 오류 분석 결과에 근거해** 최소 개수로 결정합니다 (플래그가 8개라고 전략이 8개가 되는 게 아님).

### 1.3 플래그 채택 판정 2조건 (사전 등록)

새 유형(플래그)은 다음 두 조건을 모두 만족할 때만 채택합니다:

1. **식별 가능성**: 추론 시점에 텍스트(정규식) 또는 이미지(픽셀 통계)에서 폐쇄망 제약 하에 안정적으로 탐지 가능한가?
2. **전략 차별성**: 그 플래그가 켜진 그룹에서 모델의 정답률 또는 최적 전략(프롬프트/학습 비중)이 실제로 달라지는가? (④⑤단계에서 검증)

---

## 2. 발굴 방법론

1. **선행연구 기반 후보 정의 (Top-down)**: 시간 관계 주석 표준(TimeML TLINK/ALINK), 담화 관계 체계(PDTB), 서사 스크립트 이론(Narrative Event Chains), 시각적 스토리텔링 과제(Sort Story/VIST)에서 "순서 단서의 유형"을 도출.
2. **어휘 프로브 실측 (Bottom-up 검증)**: 각 후보의 대표 어휘를 정규식으로 컴파일해 train.csv 전체 9,537문장에 대해 빈도와 실제 예문을 추출. 프로브는 발굴용 근사치이며(오탐/미탐 존재), 정확한 유병률은 질적 분석 단계에서 확정.

---

## 3. 신규 유형 후보 카탈로그 (실측 빈도순)

### N1. 카메라/편집 담화 (Cinematographic Discourse) — 실측 44.5% ★최우선

- **정의**: 스토리 속 행위자가 아니라 **영상 촬영·편집 행위 자체**를 서술하는 절이 포함된 문장. 하위 변형: ⓐ 카메라가 문장 주어인 경우(5.1%), ⓑ 명시적 장면 전환 서술 "the scene shifts/cuts to"(4.2%), ⓒ 제시형 수동태 "is shown/is seen"(3.3%).
- **선행연구 근거**: 본 데이터 문장은 dense video captioning 스타일로, [Sort Story (Agrawal et al., 2016)](https://www.researchgate.net/publication/304469686_Sort_Story_Sorting_Jumbled_Images_and_Captions_into_Stories)와 VIST 계열 연구가 구분하는 "서사적 스토리 언어 vs 화면 기술 언어(literal description)" 중 후자에 해당하는 담화 층위입니다. 절-개수 통사 분석은 "The camera pans"를 일반 절로 취급하므로 이 층위를 원리적으로 포착하지 못합니다.
- **프로브 어휘**: `camera, scene, zoom(s/ed/ing), pan(s/ned/ning), shot, close-up, cuts to, transition(s/ed), fade(s), screen, view shifts`
- **데이터 실례**:
  - *"The camera pans left to reveal two people kneeling beside a fishing hole on the ice, then shifts from a frontal to a rear view..."*
  - *"A girl hula hoops indoors before the scene shifts outdoors to a cheering group on rocks..."*
- **순서 단서 성격**: **강력한 순단서(+)**. 특히 "the scene shifts" 절은 이미지 측 씬 컷(pairwise MSE 급증 경계)과 1:1 대응될 가능성이 높아, 이 데이터에서 가장 강한 텍스트-이미지 앵커 후보입니다. "zoom in/out", "pans left" 역시 인접 프레임 간 기하 변환으로 검증 가능한 물리적 단서입니다.
- **분류(탐지) 방법**: 카메라 동사/명사 정규식 사전 (폐쇄망 안전). 하위 변형 ⓐⓑⓒ를 별도 서브 플래그로 관리.
- **전략 가설**: 카메라 절을 씬 경계 앵커로 사용 — "scene shifts" 앞뒤 절을 MSE 기반 씬 클러스터에 정렬하면 절-씬 매핑 문제가 크게 단순화됨. 검증: CAM 플래그 샘플에서 카메라 절 위치와 MSE 컷 위치의 정합률 측정.

### N2. 상적 국면 전이 (Aspectual Phase Transition) — 실측 20.2% ★차순위

- **정의**: 사건의 시작·지속·종결 **국면(phase)** 을 명시하는 상 동사(phase verb)가 포함된 문장. "begins to glide"는 '글라이딩의 시작 국면이 어느 프레임인가'라는 프레임 내부 순서 단서를 제공합니다.
- **선행연구 근거**: [TimeML의 ALINK(Aspectual Link)](https://timeml.github.io/site/publications/timeMLdocs/timeml_1.2.1.html)가 정확히 이 관계를 표준화 — INITIATES("started to read") / CULMINATES("finished assembling") / TERMINATES("stopped talking") / CONTINUES("kept talking") / REINITIATES. 현행 체계의 Vendler Aktionsart(동사 자체의 경계성)와 달리, ALINK는 **상 동사가 다른 사건을 논항으로 취하는 관계**로 별도 차원입니다.
- **프로브 어휘**: `begin(s)/began, start(s/ed), continue(s/d), finish(es/ed), stop(s/ped), resume(s/d), end(s) up/by/with, proceeds to`
- **데이터 실례**:
  - *"The skater begins to glide down the road, then falls and lies on the ground..."*
  - *"...he begins running his fingers over the wood..."*
- **순서 단서 성격**: **순단서(+)**, 특히 단일 씬 비디오에서 유효. "begins X" → X의 개시 자세 프레임이 앞, "finishes X" → 완료 상태 프레임이 뒤.
- **분류 방법**: 상 동사 정규식 + 논항(뒤따르는 동명사/부정사) 추출. ALINK 5관계 중 INITIATES/CULMINATES/TERMINATES 3종으로 단순화한 서브 태그 권장.
- **전략 가설**: 저전환(단일 씬) 비디오 × PHASE 플래그 조합에서 "동작의 개시/완료 국면을 비교하라"는 프롬프트 힌트의 효과 검증. 이미지 EDA의 "미세 행동 변화" 타겟군(1.49%)과 교차 확인.

### N3. 스크립트/절차 지식 의존 (Script-Knowledge Dependent) — 프로브 하한 5.5~6.5%

- **정의**: 시간 접속사가 전혀 없지만, **세계 지식(스크립트)** 이 사건 순서를 결정하는 문장. "Dough is rolled, placed on sheet, and baked"는 언어적 순서 표지가 0개지만 순서가 유일하게 결정됩니다. 현행 체계에서는 다중 절-병렬(구 Implicit-Sequential)에 뭉뚱그려지지만, "나열 순서가 시간 순서"라는 가정이 아니라 **절차 지식**이 근거라는 점에서 성격이 다릅니다.
- **선행연구 근거**: [Chambers & Jurafsky (ACL 2008) Narrative Event Chains](https://www.researchgate.net/publication/220873399_Unsupervised_Learning_of_Narrative_Event_Chains) — 공통 행위자(protagonist)를 공유하는 사건들의 부분 순서를 비지도 학습으로 추출; [proScript](https://arxiv.org/pdf/2104.08251) — 절차의 부분 순서 그래프 생성; [LLM의 스크립트 지식 연구](https://arxiv.org/pdf/2112.13834). VLM의 사전학습 지식이 이 유형의 해결사라는 함의.
- **프로브 어휘**: (요리 도메인 예시) `bake, mix, pour, chop, fry, serve, stir, knead, slice` → 6.5%. 구조 프로브(다중 동사 + 시간 표지 부재) → 5.5%. 도메인이 요리 외(운동 루틴, 공예, 정비 등)로 넓으므로 실제 유병률은 더 높을 것.
- **데이터 실례**:
  - *"Two people measure the floor before one stirs a bucket, then a cake is frosted and lowered from above..."*
- **순서 단서 성격**: **순단서(+)이나 모델 의존적** — 텍스트에 단서가 있는 게 아니라 모델의 상식에 있음.
- **분류 방법**: 정규식만으로는 한계. 2단계 접근 권장 — ① EDA 단계에서 LLM 보조 라벨링 또는 수동 코딩으로 골드 라벨 확보(폐쇄망 제약은 추론 서버에만 적용되므로 EDA는 자유), ② 골드 라벨로 "다중 동사 + 시간 표지 부재 + 도메인 어휘" 규칙을 증류(distill)해 배포용 정규식 확정.
- **전략 가설**: 파인튜닝이 주 해결책(스크립트 지식은 프롬프트로 주입 불가). 이 플래그는 라우팅용이 아니라 **오류 분석의 슬라이스 축**으로 가치가 큼 — 모델이 상식 순서를 아는지 측정하는 리트머스.

### N4. 지시 표현 진행 (Referential Progression) — 프로브 근사 5.0%

- **정의**: 절 간에 지시 표현이 부정관사→대명사/정관사로 진행되는 문장 ("**A man** enters... **he** sits down"). 담화에서 신정보(a X)는 구정보(he/the X)보다 먼저 도입되므로, 절의 서술 순서가 곧 시간 순서라는 보조 단서를 제공합니다.
- **선행연구 근거**: Chambers & Jurafsky의 서사 체인이 공통 행위자의 공지시(coreference)를 순서 학습의 핵심 신호로 사용; 문장 정렬 연구([STaCK, EMNLP 2021](https://github.com/declare-lab/sentence-ordering) 등)에서 개체 연속성은 표준 피처.
- **프로브 어휘**: `a (man|woman|boy|girl|person|player|child|dog|group) ... (he|she|they|his|her|their)` (동일 문장 내 선행)
- **순서 단서 성격**: **약한 순단서(+)**. 다중 절-병렬 파티션의 "나열 순서=시간 순서" 가정을 보강하는 신뢰도 가중치로 활용 가능.
- **분류 방법**: 관사/대명사 시퀀스 정규식. 절 분할 결과와 결합해 "절1에 부정관사 도입 + 절2에 대명사" 패턴 확인.
- **전략 가설**: 단독 라우팅 가치는 낮음. 다중 절-병렬 유형의 순서 신뢰도 서브 스코어로 흡수.

### N5. 외형/상태 변화 앵커 (Appearance & State-Change Anchor) — 실측 4.2%

- **정의**: 인물의 복장·사물의 상태가 "X에서 Y로" 바뀌었다고 명시하는 문장 ("transitions from wearing a dark vest to a red vest", "is now wearing an orange shirt"). 프레임 판별이 색상/외형 매칭이라는 **저수준 시각 과제로 환원**되는 유형.
- **선행연구 근거**: TimeML의 결과 상태(resultative) 개념 및 [NarrativeTime](https://arxiv.org/pdf/1908.11443)의 상태 구간 주석과 상통. 시각 측면에서는 이미지 EDA의 "시각적 상태 전이 분석(Method A)"과 정확히 결합되는 텍스트 대응물.
- **프로브 어휘**: `transitions from ... to, changes into/from/to, switches to/from, now wearing, different (outfit|shirt|jacket)`
- **데이터 실례**:
  - *"The athlete transitions from wearing a dark vest to a red vest as the camera shifts focus..."*
  - *"The same lady is now wearing an orange shirt and is throwing the Frisbee to the dog."*
- **순서 단서 성격**: **매우 강한 순단서(+)** — 텍스트가 프레임 구분 기준(색상)까지 알려주는 최고 명확도 유형.
- **분류 방법**: "from X to Y" 구문 정규식 + 색상/의복 어휘 사전.
- **전략 가설**: 프롬프트에서 별도 처리 불필요할 만큼 쉬울 가능성 — 대신 이 그룹의 정답률이 낮게 나오면 모델의 색상-프레임 바인딩 실패라는 진단 신호. 오류 분석 슬라이스 축으로 사용.

### N6. 반복/순환 동작 (Iterative & Cyclic Action) — 실측 2.6%

- **정의**: 동일 동작의 반복을 서술하는 문장 ("swings again", "back and forth", "repeatedly"). **역단서(−)**: 반복 동작의 프레임들은 상호 교환 가능하므로 텍스트·이미지 모두에서 순서 결정이 원리적으로 불가능해집니다.
- **선행연구 근거**: 상 의미론의 반복상(iterative aspect); TimeML ALINK의 REINITIATES. 현행 5단계 체계의 어떤 유형도 "순서가 존재하지 않는" 케이스를 포착하지 않음 — 모호성의 '최고 단계'는 정의돼 있으나, 반복상은 모호한 게 아니라 **순서 정보가 물리적으로 소실**된 경우라 성격이 다름.
- **프로브 어휘**: `again, repeatedly, multiple times, several times, once more, back and forth, continues to`
- **순서 단서 성격**: **역단서(−)** — 유일한 '난이도 상한' 마커.
- **분류 방법**: 반복 부사 정규식.
- **전략 가설**: 이 그룹은 EM 기대치를 낮춰야 하는 그룹 — 검증셋 성적 해석 시 분리 집계(안 그러면 노이즈가 다른 실험 효과를 가림). No_ordering과의 상관도 확인 가치 있음.

### N7. 서수 열거 (Ordinal Enumeration) — 실측 2.1%

- **정의**: `first, second, finally, eventually` 등 서수/단계 부사로 사건을 열거하는 문장. 구 Explicit-Sequential의 하위 변형이나, 접속사(then)와 달리 **전역 순서 인덱스**를 직접 제공.
- **선행연구 근거**: [PDTB 3.0](https://catalog.ldc.upenn.edu/docs/LDC2019T05/PDTB3-Annotation-Manual.pdf)의 Expansion 계열 담화 관계 중 열거(List) — Temporal과 별도 차원으로 주석됨.
- **프로브 어휘**: `first, initially, at first, second(ly), third, lastly, eventually, in the end, ultimately`
- **분류 방법**: 서수 부사 정규식. **주의**: "the second hand(초침)" 같은 오탐이 실측에서 확인됨 → 명사 수식 위치는 제외하는 부정 조건 필요.
- **전략 가설**: 별도 전략 불필요(가장 쉬운 그룹). 파티션 검증용 보조 플래그.

### N8. 인과 관계 (Causal, PDTB Contingency) — 실측 0.4% (명시적)

- **정의**: 원인→결과 순서를 함의하는 인과 표지 문장 ("causing the camera to tilt"). [PDTB의 4대 최상위 담화 관계](https://link.springer.com/chapter/10.1007/978-94-024-0881-2_45) 중 Contingency는 Temporal과 독립 차원이며, 인과는 시간 접속사 없이도 순서를 결정합니다.
- **판정**: 명시적 인과는 0.4%로 희소 → **독립 플래그로 채택 보류**. 암묵적 인과(fell → lies on the ground)는 N3 스크립트 지식에 흡수. PDTB가 보여주듯 "since" 등은 시간·인과 중의적이므로 기존 시간 접속사 사전이 이미 상당 부분 커버.

---

## 4. 실측 요약표

| 플래그 | 빈도 (train, n=9,537) | 단서 방향 | 탐지 난이도 | 채택 권고 |
|---|---:|:---:|:---:|---|
| N1 카메라/편집 담화 | **44.5%** (서브: 카메라주어 5.1 / scene shift 4.2 / shown·seen 3.3) | + 강 | 하 (정규식) | **채택 — 최우선** |
| N2 상적 국면 전이 | **20.2%** | + 중 | 하 (정규식) | **채택** |
| N3 스크립트 지식 | ≥5.5~6.5% (하한) | + (모델 의존) | 상 (증류 필요) | 채택 — 오류분석 축 |
| N4 지시 진행 | ~5.0% | + 약 | 중 | 보류 — 서브 스코어로 흡수 |
| N5 상태 변화 앵커 | 4.2% | + 최강 | 하 | 채택 — 진단용 |
| N6 반복/순환 | 2.6% | **− (역단서)** | 하 | **채택 — 유일한 난이도 상한 마커** |
| N7 서수 열거 | 2.1% | + 강 | 하 (오탐 주의) | 보조 플래그 |
| N8 명시적 인과 | 0.4% | + | 하 | **기각 (희소)** — N3에 흡수 |

※ 프로브 빈도는 발굴용 근사치(단순 정규식, 오탐/미탐 포함). 확정 유병률은 질적 분석에서 산출.

---

## 5. 신규 유형의 분류(탐지) 아키텍처

절 구조 파티션은 purity 분석으로 검증하는 기존 계획을 유지하고, 신규 유형은 다음 3단 구조로 분류합니다:

1. **탐지기 (배포용, 폐쇄망 안전)**: 플래그별 독립 정규식 사전. 기존 규칙 사전(Rulebook) 인프라를 그대로 확장 — `LEXICON`에 `camera_terms`, `phase_verbs`, `iterative_advs`, `state_change_patterns` 키 추가.
2. **골드 라벨링 (EDA 단계)**: 플래그별 층화 샘플 50~100개를 블라인드 수동 라벨링(N3은 LLM 보조 가능). 정규식 대비 정밀도/재현율 산출 → 정밀도 낮은 플래그는 어휘 보정 후 홀드아웃 재검증(리뷰 샘플 반분 원칙).
3. **이미지 교차 검증 (플래그별 purity의 대응물)**: 각 플래그가 주장하는 시각적 함의를 기존 이미지 EDA 산출물로 검증:
   - N1(scene shift 서브 플래그) ↔ MSE 씬 컷 존재 여부 정합률
   - N2 ↔ 저전환 비디오(유사쌍 3+)에서의 상대 빈도
   - N6 ↔ 프레임 간 고유사도(순수 미세 행동군)와의 상관
   - 정합률이 기저 대비 리프트 없음 → 해당 플래그의 시각 앵커 가설 기각 (플래그 자체는 텍스트 피처로 존치 가능)

---

## 6. 질적 분석 연계 (③단계 입력)

- 플래그별 층화 샘플링: {플래그 on × 절 구조 3유형} 셀에서 무작위 추출, 블라인드 순서(문장→판정→이미지→판정→파서 라벨 대조) 준수.
- 중점 확인 사항: N1 카메라 절과 실제 씬 경계의 대응, N2 국면 동사와 프레임 내 자세 국면의 대응, N6 반복 샘플의 실제 순서 결정 가능 여부(Q2 판정).

## 7. 학습·추론 후 오류 기반 유형 갱신 (⑤단계 예고)

파인튜닝 모델의 train/검증 예측 결과가 나오면, 파티션×플래그 조합별 정답률 분해와 함께 **오류 슬라이스 발견 기법**으로 본 카탈로그에 없는 유형을 추가 탐색합니다: 크로스모달 임베딩 공간에서 오류 응집 군집을 찾는 계열([HiBug2/DebugAgent](https://arxiv.org/html/2501.16751), [영향 임베딩 군집화](https://arxiv.org/html/2312.04712v1), [LADDER](https://pmc.ncbi.nlm.nih.gov/articles/PMC12756946/)). 여기서 나온 응집 오류 군집을 사람이 서술 → 1.3의 채택 2조건 통과 시 신규 플래그로 편입.

---

## 8. 참고문헌

- Agrawal et al., *Sort Story: Sorting Jumbled Images and Captions into Stories*, EMNLP 2016 — https://www.researchgate.net/publication/304469686_Sort_Story_Sorting_Jumbled_Images_and_Captions_into_Stories
- Chambers & Jurafsky, *Unsupervised Learning of Narrative Event Chains*, ACL 2008 — https://www.researchgate.net/publication/220873399_Unsupervised_Learning_of_Narrative_Event_Chains (프로젝트 페이지: https://nlp.stanford.edu/projects/narratives.shtml)
- Sakaguchi et al., *proScript: Partially Ordered Scripts Generation*, 2021 — https://arxiv.org/pdf/2104.08251
- *What do Large Language Models Learn about Scripts?*, 2021 — https://arxiv.org/pdf/2112.13834
- TimeML Specification 1.2.1 (ALINK: INITIATES/CULMINATES/TERMINATES/CONTINUES/REINITIATES) — https://timeml.github.io/site/publications/timeMLdocs/timeml_1.2.1.html
- TimeBank 1.2 Documentation (TLINK 체계) — https://timeml.github.io/site/timebank/documentation-1.2.html
- Webber & Prasad et al., *The Penn Discourse Treebank 3.0 Annotation Manual* — https://catalog.ldc.upenn.edu/docs/LDC2019T05/PDTB3-Annotation-Manual.pdf
- *The Penn Discourse Treebank: An Annotated Corpus of Discourse Relations* — https://link.springer.com/chapter/10.1007/978-94-024-0881-2_45
- Rogers et al., *NarrativeTime: Dense Temporal Annotation on a Timeline* — https://arxiv.org/pdf/1908.11443
- *A Survey on Temporal Reasoning for Temporal Information Extraction* — https://arxiv.org/pdf/2005.06527
- STaCK: *Sentence Ordering with Temporal Commonsense Knowledge*, EMNLP 2021 — https://github.com/declare-lab/sentence-ordering
- *HiBug2/DebugAgent: Efficient and Interpretable Error Slice Discovery* — https://arxiv.org/html/2501.16751
- *Error Discovery By Clustering Influence Embeddings* — https://arxiv.org/html/2312.04712v1
- *LADDER: Language-Driven Slice Discovery and Error Rectification* — https://pmc.ncbi.nlm.nih.gov/articles/PMC12756946/
