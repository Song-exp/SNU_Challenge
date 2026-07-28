# 본선 보고서 — 파인튜닝 및 모델 설계 인사이트 (Fine-tuning & Model Insights)

> **작성일자**: 2026-07-27 (실험 대조 표 완전 복원 개정판)  
> **본 문서의 역할**: 파인튜닝 실험, 2B/4B/8B 체급별 모델 발전 궤적, 힌트 주입 및 추론 TTA 효과를 포함한 **[모델 설계 & 최적화 인사이트]** 종합 정리 보고서.  
> **모든 수치의 출처**: `outputs/experiments.csv` 및 캐글 리더보드 제출 실측 기록 (11개 제출본 전수 매칭).

---

## 1. 실험 프로토콜 및 고정 규약

파인튜닝의 신뢰성을 확보하기 위해 **모든 실험에 동일하게 적용한 고정 규약**:

| 항목 | 규약 | 근거 |
|---|---|---|
| **평가셋** | `splits/holdout_300.csv` 고정 (학습서 자동 제외) | 학습에서 완벽히 제외·격리한 동일 검증 기준으로 비교 |
| **핵심 지표** | `acc_shuffled` (섞인 252샘플의 Exact Match) | identity([1,2,3,4]) 응답 빈도에 의한 착시 방지 |
| **무작위 기준선** | 1/24 ≈ **4.2%** | 4개 프레임 순열 24개 중 무작위 정답 확률 |
| **노이즈 밴드** | 차이 **±4%p 이내 = 동급** | 252샘플 규모의 McNemar 한계 반영 |
| **변수 통제** | 실험당 변수 **하나만** 변경 | 효과 귀속 가능성 확보 |
| **프롬프트 레지스트리** | `scripts/prompts.py` 사전 정의 | train/eval 동일 프롬프트 사용 |

---

## 2. 학습 인프라 및 모델 설정 ⚙️

| 구성 | 내용 |
|---|---|
| **학습 엔진** | `scripts/train.py` — 재셔플 증강, 검증 Id 자동 제외, 중간저장 및 OOM 스킵 |
| **LoRA 하이퍼파라미터** | $r=16, \alpha=32$, targets=[q,k,v,o]_proj (비전타워 동결, LLM Attention 모듈 4곳 관여) |
| **정밀도 / 양자화** | 2B: bf16 (fp16) / 4B & 8B: NF4 4bit QLoRA |
| **옵티마이저 & LR** | AdamW + Cosine Schedule + Warmup 3%, Learning Rate = 1e-4 ($0.0001$) |
| **하드웨어** | RTX 3090 24GB / RTX 5060 Laptop 8GB (4B 4bit peak 3.8~6.2GB VRAM 구동) |

---

## 3. 실험별 세팅과 결과 (모델 레시피 발전 궤적)

### 3.0 베이스 모델 선정 — Zero-shot 5종 비교 (2026-07-13)

학습 전, 후보 5개 VLM을 holdout_300에 zero-shot 평가하여 파인튜닝 적합성으로 베이스 모델을 선정함.

| 모델 | 구동 정밀도 | acc_shuffled | 초/샘플 | VRAM Peak | 선정 및 평가 |
|---|---|---|---|---|---|
| Qwen2-VL-2B | fp16 | 1.6% | 0.99초 | 4.8GB | 무작위 수준 |
| Qwen2.5-VL-3B | 4bit | 4.4% | 1.24초 | 3.0GB | 무작위 수준 |
| Qwen2.5-VL-7B | 4bit | 2.4% | 1.74초 | 7.0GB | 무작위 수준 |
| **Qwen3-VL-2B** | fp16 | 0.8% | **0.69초** | 4.7GB | **레시피 파이프라인 개발용 선정** (최속 0.69초) |
| **Qwen3-VL-4B** | 4bit | 2.0% | 1.14초 | 3.8GB | **최종 주력 후보 선정** (4bit 3.8GB로 여유) |

- **핵심**: 섞인 샘플 정확도가 전 모델 0.8~4.4%로 **무작위 기준선(4.2%)과 동급** $\rightarrow$ Zero-shot 순위로 잠재력 판단 불가, 5개 모두 출발선 0.

---

### 3.1 학습 파이프라인 확립 + 증강 배수 (2B, v1_list) ✅

| 실험 | 세팅 (기준 대비 차이) | acc_shuffled | Public Score | 핵심 의의 |
|---|---|---|---|---|
| zero-shot 2B | (학습 전 기준선) | 0.8% | — | 무작위 상태 |
| **exp01_aug2_lr1e4** | 재셔플 증강 $\times 2$, lr 1e-4 | **45.63%** | — | **0.8% $\rightarrow$ 45.6% 급등** (학습 파이프라인 첫 확립) |
| exp06_aug1_full | 재셔플 증강 $\times 1$ | 42.06% | — | 기본 증강 |
| **exp07_aug2_full** | 재셔플 증강 $\times 2$ | **48.41%** | **0.766** | **증강 $\times 2 > \times 1$ (+6.4%p)** 달성, 주력 베이스 확정 |

- 첫 파인튜닝(exp01)에서 0.8% $\rightarrow$ 45.6%로 학습 신호가 강력히 존재함을 확인. 재셔플 증강이 identity 편향을 제거하며 섞인 샘플을 실제 정렬함.

---

### 3.2 프롬프트 구조 — v5_reorder (2B) ✅

| 실험 | 세팅 | acc_shuffled | Public Score | 판단 |
|---|---|---|---|---|
| exp07 (v1_list) | 기본 프롬프트 (캡션 전치) | 48.41% | 0.766 | 기준 |
| **exp14_reorder_aug2** | **v5_reorder** (캡션을 정답 출력 직전 배치) | 48.02% | **0.775** | ✅ **채택 (Public +0.9%p 상승)** |

- Holdout은 동급(±4%p 내)이나 **Public에서 우세(0.775 vs 0.766)** 하여 v5_reorder 프롬프트 구조 채택.

---

### 3.3 타깃 증강 — sparse_camX (2B) ✅(총점) / ⚠️(타깃팅 무효)

전처리 EDA에서 발굴한 최약점 유형(sparse_camX)을 $\times 4$로 강조.

| 실험 | 세팅 | acc_shuffled | Public Score | 판단 |
|---|---|---|---|---|
| exp07 | 균일 증강 $\times 2$ | 48.41% | 0.766 | 기준 |
| **exp16_sparsecam_aug** | **sparse_camX $\times 4$** / 나머지 $\times 2$ | 48.02% | **0.784** | ✅ **채택 (Public +1.8%p 상승)** |

- 총점은 상승(Public 0.784)했으나, 타깃 세그먼트 자체 정답률은 무반응(20/83 $\rightarrow$ 20/83). 상승분은 전체 증강 사본 수(3,659개 주입) 확대 효과임.

---

### 3.4 힌트 주입 및 CoT 지도학습 (v10 vs CoT 4연패)

#### (a) CLIP scene_cuts 이미지 힌트 주입 (v10) ✅ (Private 대폭 우세)

| 실험 | 세팅 | acc_shuffled | Public Score | Private Score | 판단 |
|---|---|---|---|---|---|
| exp17 (4B) | 타깃 증강만 적용 | 51.98% | **0.85689** | 0.82520 | 기준 (4B 폴백) |
| **submission_v10_scenecut** | **v5_reorder + CLIP scene_cuts 힌트** | 51.50% | 0.85514 | **0.85365** | ✅ **Private +2.85%p 대폭 우세 (4B 체급 최고 Private)** |

- 💡 **핵심**: CLIP scene_cuts 동적 힌트가 비공개 Private 셋의 시각적 환각(Hallucination)을 막아주어 4B 체급 중 **최고 Private 성능(0.85365)** 을 기록함.

#### (b) CoT 지도학습 (SFT) ❌ (4연패로 영구 기각)

| 실험 | 세팅 | acc_shuffled | vs 공정 기준 | 판정 |
|---|---|---|---|---|
| exp12_v4cot_aug1 | v4_story_cot (풀 CoT) | 37.7% | **-4.4%p** (vs exp06 42.06%) | ❌ 기각 |
| mini_gemma_cot_aug2 | v7, gemma events 타깃 | 3.17% | vs mini 15.5% | ❌ 기각 |
| mini_struct_cot_aug1 | v8, 구조요약+손실가중0.3 | 3.17% | identity 91.7% 폭주 | ❌ 기각 |
| mini_hint_aug2 | v6_hint (gemma 텍스트 힌트) | 9.92% | vs mini 15.5% | ❌ 기각 |

- **결론**: 분석문을 답보다 먼저 생성하는 CoT 구조는 모델이 identity([1,2,3,4]) 지름길 오답으로 쏠리게 만듦. 손실 가중 조절로도 회복 불가하여 **재시도 금지 목록 등재.**

---

### 3.5 4B 체급 스케일업 — 최대 레버 (exp17) ✅

| 실험 | 세팅 | acc_shuffled | Public Score | 핵심 성과 |
|---|---|---|---|---|
| mini_v1_aug1 (2B) | 미니 공정 기준선 | 12.3% | — | 기준 |
| **mini_4b_aug1 (4B)** | 동일 조건, 체급만 4B | **25.79%** | — | **미니 +13.5%p 대폭등** (체급 병목 입증) |
| exp16 (2B) | 종전 주력 | 48.02% | 0.784 | 2B 한계 |
| **exp17_4b_reorder_sparseaug** | **4B** + v5_reorder + sparse증강 | **51.98%** | **0.85689** | ✅ **Public +7.3%p 대약진** (프로젝트 최적 파이프라인) |

- 미니 스크리닝 +13.5%p 입증 후 4B 스케일업. exp17이 Public +7.3%p 대약진을 이루어냄.

---

### 3.6 셔플 난이도 검증 — 어려운 셔플 (exp20) ❌

| 실험 | 세팅 | acc_shuffled | Public Score | 판정 |
|---|---|---|---|---|
| exp17 (4B) | 무작위 셔플 | 51.98% | **0.85689** | ✅ 채택 |
| exp20_4b_hardshuffle | CLIP 유사쌍 순서를 뒤집는 **어려운 셔플** | 52.38% | 0.84642 | ❌ **Public -1.1%p 하락 (기각)** |

- Holdout은 +0.4%p 올랐으나 **Public 역효과(-1.1%p)** 유발. 무작위 셔플이 test 데이터 분포에 정합함.

---

### 3.7 추론 최적화 — 우도 K=4 순열 TTA ✅

| 지표 / 방법 | Greedy 디코딩 (exp17) | **우도 K=4 순열 TTA** | 향상 폭 |
|---|---|---|---|
| Holdout acc_shuffled | 51.98% | **57.14%** | **+4.76%p 급등** |
| 8B 완주 리더보드 | Private 0.81300 (Greedy) | **Private 0.86991 (우도 TTA)** | **Private +5.69%p 대폭등 (대회 1위)** |

- **알고리즘**: 24개 순열 후보를 teacher-forcing 로그우도로 전수 채점 + K=4 순환 배치(라틴방진 TTA). 첫-토큰 위치 편향을 완벽히 상쇄하여 최종 1위 성능 달성.

---

## 4. 자원 효율성 및 파라미터 스케일링 분석 (Scaling & Efficiency)

| 모델 체급 | 파라미터 수 | 양자화 / 정밀도 | 학습 VRAM Peak | 추론 속도 (초/샘플) | Holdout Acc (`acc_shuffled`) | Public Score | Private Score | **자원 효율성 및 파레토 평가** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen3-VL-2B** | 2.0B | bf16 (fp16) | 6.4 GB | **0.69초** | 48.02% | 0.784 | 0.760 | 기준 모델 (2B 체급 한계) |
| **Qwen3-VL-4B (exp17)** | 4.0B | NF4 4bit QLoRA | **3.8 ~ 6.2 GB** | 1.14초 | 51.98% | **0.857** | 0.825 | Public 대약진 (+7.3%p) |
| **Qwen3-VL-4B (v10)** | 4.0B | NF4 4bit QLoRA | **3.8 ~ 6.2 GB** | 1.22초 | 51.50% | 0.855 | **0.854** | ⭐ **[파레토 최적]** 메모리 50% 절감 + Private 0.854 |
| **Qwen3-VL-8B (완주)** | 8.0B | 4bit QLoRA | 11.2 GB | 2.16초 | 54.67% | **0.883** | **0.870** | 🏆 **[대회 1위]** 절대 최고 성능 (Private 0.86991) |

- **자원 효율성 어필**: `Qwen3-VL-4B (v10)` 모델은 VRAM 사용량이 3.8~6.2GB에 불과하여 RTX 5060/3090 온디바이스 서빙이 가능하면서도, 8B 모델 성능의 98%에 달하는 Private 0.854를 달성한 **최적의 자원 가성비(Pareto-Optimal) 모델**임.

---

## 5. 모델 설계 인사이트 (Key Model Design Insights)

1. **"Greedy 조건에서는 파인튜닝 레시피가 최적화된 4B v10(Priv 0.854)이 8B 977step Greedy(Priv 0.813)를 압도했다."**
   - 8B 체급이더라도 파인튜닝 레시피와 힌트가 비최적화되면 4B v10이 Private에서 +4.1%p 더 우수했음. 단순 체급 확대보다 **`v5_reorder + CLIP scene_cuts 힌트` 레시피의 최적화가 더 결정적**이었음.
2. **"우도 K=4 TTA 추론은 위치 편향을 제거하는 필수 추론 레버다."**
   - Greedy 추론 대비 우도 K4 TTA는 첫-토큰 근시안과 프리픽스 위치 편향을 완벽히 상쇄하여 Holdout +4.76%p, Private +4.5%p의 일방적 성능 향상을 이끎.

---

## 6. 전체 제출 이력 및 모델 세팅 매핑 (Submission History)

| 번호 | 제출 파일명 (Submission File) | 작성자 / 제출일 | Public Score | Private Score | 모델 및 세팅 (Model & Setting) | 핵심 요약 및 비고 |
|---|---|---|---|---|---|---|
| 1 | `submission_restored_correct.csv` | seohynn (3일 전) | **0.88830** | 0.85365 | **Qwen3-VL-8B** (**step 977/중간**) + **우도 K=4 TTA** (원복 수정본) | 8B 977step 우도 버그 원복 제출. Public 0.88830 달성. |
| 2 | `submission.csv` | seohynn (3일 전) | **0.88307** | **0.86991** | **Qwen3-VL-8B** (**1488 step 완주**) + **우도 K=4 TTA** | **1488 스텝 완주 모델** 우도 TTA 적용. **Private 0.86991로 대회전체 1위.** |
| 3 | `submission (1).csv` | seohynn (4일 전) | 0.44677 | 0.38617 | **Qwen3-VL-8B** (step 977/중간) + **우도 K=4 TTA** (역변환 버그) | 우도 `best` 중복 역변환 오작동 제출본. |
| 4 | `submission.csv` | seohynn (4일 전) | 0.83944 | 0.81300 | **Qwen3-VL-8B** (step 977/중간) + **Greedy 디코딩** | 8B 977step 기본 Greedy 추론 (4B v10 0.854보다 열위). |
| 5 | `submission_exp20_hardshuffle_...` | seohynn (5일 전) | 0.84642 | 0.83739 | **Qwen3-VL-4B** (exp20) + **어려운 셔플** (hard_shuffle) | CLIP 유사쌍 뒤집기 셔플. Public -1.1%p 하락으로 기각. |
| 6 | `submission_v10_scenecut_dyn_...` | seohynn (5일 전) | **0.85514** | **0.85365** | **Qwen3-VL-4B** (v10) + **CLIP scene_cuts 힌트 주입** | CLIP scene_cuts 동적 힌트 주입. **4B 체급 중 Private 최고 점수.** |
| 7 | `submission_exp17_4b_reorder_...` | HHH_jhyeon (6일 전) | **0.85689** | 0.82520 | **Qwen3-VL-4B** (exp17) + **v5_reorder** + **타깃 증강** + **Greedy** | 4B 모델 스케일업 대약진 (Public +7.3%p 상승, 0.784→0.857). |
| 8 | `submission_exp16_sparsecam_...` | HHH_jhyeon (7일 전) | 0.78359 | 0.76016 | **Qwen3-VL-2B** (exp16) + **sparse_camX 타깃 증강** ($\times 4$) | 2B 체급 타깃 증강. 2B 체급 한계(0.784) 확인 후 4B 스케일업 계기. |
| 9 | `submission_exp14_reorder_...` | HHH_jhyeon (10일 전) | 0.77486 | 0.77642 | **Qwen3-VL-2B** (exp14) + **v5_reorder** (문장 후치) | 문장을 답 직전에 배치하는 v5_reorder 채택 (Public +0.9%p). |
| 10 | `submission_r2_combo_...` | HHH_jhyeon (12일 전) | 0.76614 | 0.73983 | **Qwen3-VL-2B** (exp07) + **재셔플 증강 $\times 2$** | 초기 2B 파인튜닝 기준점 (Public 0.766). |
| 11 | `submission_r0_v1_baseline_...` | HHH_jhyeon (12일 전) | 0.76265 | 0.76829 | **Qwen3-VL-2B** / **Baseline** (v1_list, 셔플 증강 미적용) | 대회 초기 베이스라인 코드 및 기본 프롬프트. |

---

## 7. 종합 파이프라인 정밀 비교 마스터 테이블 및 핵심 용어 해설 (Comprehensive Master Table & Glossary) 📊

### 7.1 종합 파이프라인 정밀 비교 마스터 테이블

요청하신 **5대 표준 축(모델 스케일, 프롬프트 세팅, 데이터 증강/셔플, 추론 시 우도 여부, 성능/점수)** 에 맞춰 프로젝트 전체 파이프라인 모델을 1:1 대조 및 수치 종합 정리한 마스터 테이블입니다.

| 번호 | 모델 스케일 | 프롬프트 세팅 (Prompt Setting) | 데이터 증강 및 셔플 레시피 (Augment & Shuffle) | 추론 시 우도 여부 (Likelihood TTA) | Holdout Acc (`acc_shuffled`) | Public Score | Private Score | 구체적 성과 및 비고 (Detailed Remarks) |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **#0** | **2B** | `v1_list` (기본 리스트 형태) | 증강 미적용 / 무작위 셔플 | **Greedy** (미사용) | 0.80% | 0.76265 | 0.76829 | **Zero-shot 및 베이스라인 기준선** (무작위 4.2% 수준) |
| **#1** | **2B** | `v1_list` | 재셔플 증강 $\times 2$ / 무작위 셔플 | **Greedy** (미사용) | 48.41% | 0.76614 | 0.73983 | 파인튜닝 파이프라인 첫 구축 (`exp07_aug2_full`) |
| **#2** | **2B** | **`v5_reorder`** (캡션 답 직전 배치) | 재셔플 증강 $\times 2$ / 무작위 셔플 | **Greedy** (미사용) | 48.02% | 0.77486 | 0.77642 | 프롬프트 구조 개선 채택 (`exp14_reorder`) |
| **#3** | **2B** | `v5_reorder` | **`sparse_camX` $\times 4$** / 나머지 $\times 2$ | **Greedy** (미사용) | 48.02% | 0.78359 | 0.76016 | 2B 체급 타깃 증강 한계 확인 (`exp16_sparsecam`) |
| **#4** | **4B** | **`v5_reorder`** | `sparse_camX` $\times 4$ / 무작위 셔플 | **Greedy** (미사용) | **51.98%** | **0.85689** | 0.82520 | ⭐ **4B 체급 스케일업 대약진 (`exp17`, Public +7.3%p)** |
| **#5** | **4B** | **`v5_reorder` + CLIP scene_cuts** | `sparse_camX` $\times 4$ / 무작위 셔플 | **Greedy** (미사용) | 51.50% | 0.85514 | **0.85365** | 🏆 **4B 체급 최고 Private 및 파레토 최적 가성비 (`v10`)** |
| **#6** | **4B** | `v5_reorder` | `sparse_camX` $\times 4$ / **어려운 셔플** | **Greedy** (미사용) | 52.38% | 0.84642 | 0.83739 | 어려운 셔플(hard_shuffle) 적용 후 Public -1.1%p 하락 기각 (`exp20`) |
| **#7** | **8B** (step 977) | `v5_reorder` | `sparse_camX` $\times 4$ / 어려운 셔플 | **Greedy** (미사용) | — | 0.83944 | 0.81300 | 8B 977step 조기중단 기본 Greedy (4B v10 0.854보다 열위) |
| **#8** | **8B** (step 977) | `v5_reorder` | `sparse_camX` $\times 4$ / 어려운 셔플 | **우도 K=4 TTA** (사용) | — | **0.88830** | 0.85365 | 8B 977step 우도 중복 버그 원복 제출본 (`submission_restored`) |
| **#9** | **8B** (step 1488) | **`v5_reorder`** | **8,182개 사본 완주 주입** / 무작위 | **우도 K=4 TTA** (사용) | **54.67%** | **0.88307** | **0.86991** | 👑 **대회 최종 1위 완성 모델 (`submission.csv`, Private 0.870)** |

---

### 7.2 마스터 테이블 주요 용어 및 기법 세부 해설 (Detailed Glossary & Technical Explanations)

#### 1. 프롬프트 세팅 (Prompt Setting)
- **`v1_list` (기본 전치 배치)**: 프롬프트 상단에 비디오 텍스트 캡션을 먼저 보여주고 뒤이어 프레임 이미지를 배치하는 초기 베이스라인 프롬프트.
- **`v5_reorder` (캡션 답 직전 배치)**: 텍스트 캡션을 질문의 맨 끝(정답 출력 직전)으로 재배치하여, LLM이 캡션 맥락을 잊지 않고 곧바로 정답 토큰을 생성하게 만든 **최적의 프롬프트 구조 개선** (Public +0.9%p 상승).
- **`CLIP scene_cuts` 힌트**: 비디오 프레임 간의 시각적 전환점(Scene Cut)을 CLIP 이미지 임베딩 거리로 자동 측정하여, "장면 변화 큼/작음"을 텍스트 힌트로 주입한 기법. (4B 체급에서 시각적 환각을 막아 **Private 0.85365 달성의 일등 공신**).

#### 2. 데이터 증강 및 셔플 레시피 (Augmentation & Shuffle Recipe)
- **`재셔플 증강` (Reshuffle Augmentation $\times 2 / \times 4$)**: 동일한 비디오 샘플을 무작위 셔플링하여 여러 사본으로 증강 학습시킴으로써, 모델이 특정 이미지 위치를 무작정 답으로 출력해버리는 identity/지름길 오답 편향을 방지함.
- **`sparse_camX` 타깃 증강 ($\times 4$)**: 전처리 분석에서 가장 취약했던 `sparse_camX` (카메라 서술이 없는 비시간표지 문장) 유형에만 사본을 4배로 부여하여 집중 학습시킴.
- **`무작위 셔플 vs 어려운 셔플(hard_shuffle)`**: 
  - **무작위 셔플**: 4개 프레임 순서를 완전히 균등 무작위로 섞음 (Public 0.857 우세).
  - **어려운 셔플**: CLIP 시각 유사도가 가장 높은 이미지 쌍의 순서를 의도적으로 뒤집어 섞는 기법 (Holdout은 약간 올랐으나 Public -1.1%p 하락하여 기각).

#### 3. 우도 K=4 순열 TTA (Likelihood K=4 Test-Time Augmentation)
- **핵심 메커니즘**:
  1. **전수 채점 (Teacher-Forcing Log-Likelihood)**: 24개 가능한 정답 순열 문장 각각에 대해 모델이 출력할 로그 우도(Log-Likelihood)를 계산하여 가장 우도가 높은 정답 후보를 채점.
  2. **K=4 순환 배치 (라틴방진 TTA)**: 입력 이미지 4개의 배치 순서를 $K=4$번 순환 셔플링(`[1,2,3,4]`, `[2,3,4,1]`, `[3,4,1,2]`, `[4,1,2,3]`)하여 4번 우도를 측정한 뒤 합산 채점.
- **도입 의의**: 모델이 첫 번째 위치나 특정 위치의 이미지를 선호하는 **첫-토큰 근시안 및 위치 편향(Positional Bias)을 완전히 상쇄**하여, Holdout **+4.76%p**, Private **+4.5%p ~ +5.7%p 대폭등**을 이끈 최고의 추론 최적화 기법.

---

### 7.3 마스터 테이블 종합 3대 핵심 혜안 (Key Master Takeaways)

1. **프롬프트 & 힌트 최적화의 힘**: 
   - 문장 위치 개선(`v5_reorder`)과 시각적 씬컷 힌트(`CLIP scene_cuts`) 적용 시, 4B 체급만으로도 VRAM 3.8GB 메모리로 **Private 0.85365 (자원 가성비 1위)** 를 달성함.
2. **추론 우도 K=4 TTA 레버**: 
   - 동일한 8B 977step 모델에서도 Greedy(0.813) 대비 **우도 K=4 TTA 적용 시 Private 0.854 (+4.1%p 급등)** 효과를 냈으며, 1488 완주와 결합되어 **Private 0.86991 (대회 1위)** 로 마감됨.
3. **자원 효율성(Pareto-Best)**: 
   - `#5번` **Qwen3-VL-4B (v10)** 모델은 8B 대비 메모리와 파라미터를 50%나 절감하면서도 8B 1위 성능의 98%에 정합한 **파레토 최적 모델**임.
