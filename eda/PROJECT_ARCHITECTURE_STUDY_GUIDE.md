# 서울대 AI 챌린지 통합 학습 가이드 — [문장·이미지·모델 융합 및 코드 해설]

본 문서는 서울대학교 AI 챌린지 프로젝트의 **문장 분석(NLP), 이미지 전처리(CV), 그리고 모델 학습 및 추론(Deep Learning)**이라는 세 가지 핵심 축이 어떻게 유기적으로 결합하여 최종 코드로 도출되었는지 그 개발 이력과 설계 개념을 상세히 해설합니다. 

사용자님이 이미지 전처리 영역을 넘어 프로젝트의 전반적인 AI 아키텍처와 소스 코드를 심도 있게 이해하고 학습할 수 있도록 작성된 종합 학습 가이드입니다.

---

## 1. 세 축의 융합 과정 (Spatio-Temporal Fusion Timeline)

```
[문장 분석 파트]  ────────┐
(통사 구조 & 모호성 지수)    │
                          ▼
[이미지 전처리 파트] ───► [하이브리드 데이터셋 구성] ───► [LoRA 학습 및 K4 우도 추론]
(CLIP 분할 & 기하 궤적)      (Soft Prompting 내재화)        (Position Bias 소거 및 완주)
                          ▲
[딥러닝 학습/추론] ───────┘
(KV-Cache 복제 & RoPE)
```

### A. 문장 분석 (NLP) 파트의 진화
1. **단어 매칭 (초기)**: 캡션 문장에 단순히 "zoom" 또는 "left"와 같은 특정 단어가 들어있는지만을 정규식으로 판별했습니다.
2. **SpaCy 구문 분석 트리 (중기)**: 단어 매칭만으로는 시간의 전후 관계를 모호하게 서술한 복잡한 문장을 분석하기 어려웠습니다. 이에 SpaCy 라이브러리의 의존성 구문 분석(Dependency Parsing)을 활용하여 문장을 **Type-1 (단일 절)**, **Type-2 (복합 종속)**, **Type-3 (대등 병렬)**로 3분류하고 주어/동사 개수를 정량화했습니다.
3. **다차원 직교 플래그(N1~N7) 및 AI 점수**: 문장의 시간적 왜곡과 모호성을 점수로 계량화하기 위해 **Ambiguity Index (AI)** 공식을 설계하여 `ai_score`로 척도화했습니다.

### B. 이미지 전처리 (CV) 파트의 진화
1. **픽셀 MSE (Adjacent MSE)**: 픽셀 값 변화를 이용해 장면 단절을 구하려 했으나, 카메라 흔들림(Whip Pan)을 장면 전환으로 오판하는 문제가 컸습니다.
2. **CLIP 기반 의미론적 분할**: OpenAI CLIP(ViT-B/32) 이미지 특징 코사인 거리를 활용하여 물리 노이즈에 왜곡되지 않는 **0.20 절대 임계값 기반 장면 분할 알고리즘**을 수립하고, **유니온-파인드(Union-Find)** 자료구조로 이미지 세트의 탐색 공간을 $4! = 24$에서 $4$로 줄였습니다.
3. **OWL-ViT Open-Vocabulary 피사체 트래킹**: 고정 클래스만 잡는 YOLOv8의 한계를 넘어, Gemma가 뽑아낸 특수 도구 명사를 OWL-ViT에 동적 입력하여 피사체 중심 $(x,y)$과 면적비($R_{\text{bbox}}$)를 마스킹 기법과 함께 추적했습니다.

### C. 딥러닝 학습 및 추론 파트의 융합
* **통찰**: "추론 단에서 비전 모델(OWL-ViT, CLIP)과 언어 모델(Gemma, Qwen)을 동시에 구동하면 Kaggle의 T4 GPU가 **VRAM OOM**으로 터지고, 24시간 내에 819문항을 풀지 못해 **시간 초과 실격**이 발생한다."
* **해결 (Soft Prompting 위임)**: 
  * 이미지/문장 분석기의 결과(Z-Score, 씬 경계, 객체 물리 궤적)를 학습 데이터 구성 단계에서 프롬프트에 녹여 미세조정(Fine-Tuning)함으로써, 8B VLM의 Attention 레이어가 시공간 기하학적 정보(Spatial Prior)를 스스로 가중치에 녹여내도록 학습시켰습니다.
  * 추론 시에는 무거운 비전 트래커를 모두 걷어내고, 오직 **Qwen3-VL 8B 단독 모델**에 **우도 K4 TTA** 기법만을 씌워 초고속 고정밀 추론을 달성했습니다.

---

## 2. 최종 학습 코드 (`FINAL_8B_v2.py`) 핵심 개념 및 동작 설명

학습 코드는 Qwen3-VL-8B-Instruct 모델을 효율적으로 파인튜닝하고, 대규모 증강 데이터셋의 학습을 가중치 편중 없이 안정적으로 완주하도록 돕는 안전장치들로 무장되어 있습니다.

### 🔬 핵심 개념 해설

#### 1. 4bit QLoRA 및 BitsAndBytes NF4 양자화
8B 모델(약 16GB VRAM 요구)을 16GB 단일 GPU 환경에서 학습시키려면 일반적인 파라미터 업데이트는 불가능합니다.
* **NF4 (Normal Float 4) 양자화**: 가중치를 4비트로 압축하여 메모리 점유율을 1/4로 줄입니다.
* **`bnb_4bit_compute_dtype=torch.float16`**: 4비트로 저장된 가중치를 연산할 때는 `float16` 정밀도로 올려 계산 오차를 줄이고 속도를 높입니다.
* **LoRA (Low-Rank Adaptation)**: 원본 가중치는 얼리고(Freeze), `q_proj`, `v_proj` 등 핵심 Attention 레이어 옆에 작은 어댑터 가중치(Rank=16)만 덧붙여 이 작은 어댑터만 학습시킵니다.

#### 2. 가중화 증강 (Weighted Augmentation) 및 난이도 분기
```python
mult = augw.get(row["Id"], CFG["aug_mult"])
```
* **동작**: `aug_weights_exp16.csv`에서 각 비디오의 ID별 증강 승수(`aug_mult`)를 가져옵니다. 모델이 기존에 자주 틀렸거나 모호한 문항일수록 증강 횟수를 늘려(예: 3~4배), 배치에서 자주 노출되도록 유동적인 집중 학습(Hard-sample Mining) 구조를 취합니다.

#### 3. 강건 셔플링 (Hard Shuffle) 및 이미지 물리 왜곡 주입
```python
perm = hard_perm(seen, sp, files, tfiles)
```
* **동작**: 무작위 셔플링 대신, CLIP 유사도 상 장면 전환이 일어나지 않는 이웃 이미지들 간의 거리를 인위적으로 역행하거나 뒤흔드는 **가장 난해한 가짜 순열**을 생성하여 학습 샘플로 제공합니다. 이를 통해 모델이 단순 이미지의 겉모습이 아닌, 실제 동작의 세밀한 물리적 선후 관계를 학습하게 강제합니다.

#### 4. 체크포인트 상태 재개 (Resume Auto-Save)
```python
if resume and os.path.exists(optpt):
    st = torch.load(optpt, map_location="cpu")
    opt.load_state_dict(st["o"])
    sched.load_state_dict(st["s"])
```
* **동작**: Kaggle 세션은 12시간이 지나면 강제 종료됩니다. 코드 내부적으로 매 100스텝마다 모델 가중치(`adapter_model.safetensors`), 옵티마이저 상태(`optim.pt`), 현재 스텝 수(`meta.json`)를 동시 저장합니다. 재실행 시 기존의 모든 학습 스텝과 Cosine LR 스케줄러 상태를 그대로 이어받아 학습의 일관성을 유지합니다.

---

## 3. 최종 추론 코드 핵심 개념 및 동작 설명

추론 코드는 Kaggle T4 듀얼 GPU 환경을 100% 활용하면서, VLM의 포지션 편향을 수학적으로 완전히 극복하고, OOM으로 코드가 멈추는 현상을 동적으로 방어하는 완성형 추론 시스템입니다.

### 🔬 핵심 개념 해설

#### 1. KV-Cache 복제 (`_clone_cache`) 및 배칭 가속
VLM 추론 시 가장 연산량이 많은 부분은 **4장의 고해상도 이미지를 비전 엔코더에 넣어 토큰 임베딩으로 변환하는 과정**입니다. 24가지의 순열 후보를 평가할 때마다 매번 이미지를 새로 읽어 인코딩하면 추론 속도가 24배 느려집니다.
```python
def _clone_cache(cache):
    # DynamicCache의 Key-Value 캐시 데이터를 복제
```
* **동작**:
  1. 먼저 이미지와 질문 템플릿(Prompt)을 모델에 통과시키고, 그때 계산된 중간 **KV-Cache(Key-Value Cache)**를 확보하여 메모리에 올립니다.
  2. 24가지 순열 후보의 텍스트 토큰을 평가할 때, 이 원본 KV-Cache를 배치 크기($b$)만큼 빠르게 복제(`batch_repeat_interleave`)하여 사용합니다.
  3. 이미지를 다시 포워딩하지 않고 오직 정답 텍스트에 대한 우도 계산만 병렬로 빠르게 수행하여 **추론 속도를 20배 이상 가속**시킵니다.

#### 2. RoPE Delta 교정 (`rope_deltas`)
Qwen-VL 모델은 입력된 이미지 크기와 텍스트 토큰의 길이에 따라 절대 위치 임베딩 값을 회전시키는 **RoPE (Rotary Position Embedding)** 아키텍처를 채택하고 있습니다.
* **문제**: KV-Cache를 복제하여 재사용할 때, 뒤이어 달라붙는 텍스트 토큰들의 위치 인덱스 오프셋이 내부 RoPE 상대값(`rope_deltas`)과 어긋나면 모델이 텍스트의 선후 관계를 엉뚱하게 오독합니다.
* **해결**: 내부 모델 레이어에서 `rope_deltas`의 텐서 값을 실시간으로 읽어와 position_ids 계산 시 오프셋을 강제로 누적 연산(Correction)함으로써 수치 정밀도 결함을 완벽하게 수정했습니다.
  ```python
  pos = ((torch.arange(plen, plen + L, device=dev) + rd).view(1, 1, -1)...)
  ```

#### 3. 가변 청크 및 dynamic VRAM OOM 방어 로직
24가지 순열에 대해 한 번에 우도를 계산하려고 배치 크기 24를 GPU 메모리에 올리면, 특정 문항의 이미지 해상도가 높을 경우 순간적으로 VRAM이 초과되어 프로그램이 죽게 됩니다.
```python
ch = CHUNK # 초기 안전 청크 6설정
while True:
    try:
        tot = score_perm(enc, plen, amat, L, ch)
        break
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        ch = max(2, ch // 2)
        print(f"⚠️ OOM 감지 → CHUNK {ch}로 줄여서 재시도합니다.")
```
* **동작**: 24개의 후보군을 `CHUNK=6`개씩 쪼개어 순차 배치 연산을 수행합니다. 만약 해상도가 너무 높은 이미지를 만나 OOM(OutOfMemory) 예외가 감지되면, 코드가 멈추지 않고 즉시 GPU 메모리를 청소(`empty_cache`)한 뒤 청크 크기를 절반(예: 3 또는 2)으로 줄여 연산을 끝까지 완주해 냅니다.

#### 4. 우도 K4 TTA (Test-Time Augmentation)
* **포지션 편향**: VLM 모델은 4장의 이미지가 입력될 때, 문장의 의미와 무관하게 앞쪽이나 뒤쪽 특정 물리적 자리에 위치한 이미지를 정답으로 더 강하게 출력하려는 기하학적 편향이 있습니다.
* **TTA 해결책**: 
  1. 이미지의 위치를 원본 배열 순서에서 $[0,1,2,3] \rightarrow [1,2,3,0] \rightarrow [2,3,0,1] \rightarrow [3,0,1,2]$로 회전시킨 **4가지 세트(K4)**를 만듭니다.
  2. 각 세트마다 24가지 순열에 대한 조건부 확률 로그 우도(Conditional Log-Likelihood)를 구합니다.
  3. 4개 세트에서 도출된 동일 정답 순열의 우도 점수를 모두 누적 합산합니다.
  * 이 과정을 거치면 4개 자리에 고르게 한 번씩 이미지가 위치하게 되어 **포지션 편향이 수학적으로 완전히 상쇄(Cancellation)**되고 순수한 이미지-문장 매핑 점수만 남게 됩니다. 결과적으로 리더보드 성적을 **0.830에서 0.888로 수직 상승**시킨 최고 수훈 기법입니다.

---

## 4. 핵심 데이터 매핑 매트릭스 (Data-to-Code Map)

| 처리 단계 | 핵심 구현 모듈/변수 | 기여한 연구 논리 및 해결 과제 |
| :--- | :--- | :--- |
| **학습 데이터 구성** | `aug_weights_exp16.csv` | 취약 문항 학습 빈도 조절 (Hard-sample Mining) |
| **장면 전환 결합** | `load_pairs(CLIP_FEATS)` | 0.20 절대 임계값 필터링 및 씬 경계 그룹화 정보 주입 |
| **모델 경량화** | `BitsAndBytesConfig` (4bit NF4) | 8B VLM의 단일 GPU 학습 및 추론 VRAM 방어 |
| **추론 속도 가속** | `_clone_cache(base)` | 이미지 토큰 인코딩 KV-Cache 재사용 (속도 20배 상승) |
| **위치 왜곡 복구** | `inner.rope_deltas.item()` | KV-Cache 복제에 따른 텍스트 위치 임베딩 오차 교정 |
| **VRAM 안전장치** | `try-except OutOfMemoryError` | 가변 배치 크기(CHUNK) 축소로 추론 시 100% 완주 보장 |
| **최종 순서 결정** | `USE_LIKELIHOOD = True` | 포지션 편향 제거용 K4 우도 앙상블 TTA 작동 |
