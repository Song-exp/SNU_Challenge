# 로컬 최종 후보 스펙 — exp17 + 우도 K4

> 로컬(RTX 5060 8GB)에서 만든 최종 제출 후보의 완전한 재현 스펙.
> = 검증된 최고 어댑터(exp17, Public 0.857) + 추론 레버(우도 K4, holdout +4.76%p).
> Kaggle 8B가 실패할 경우의 **안전 폴백**이자, 규정상 완전히 깨끗한 구성.

---

## 요약

| 항목 | 값 |
|---|---|
| **베이스 모델** | Qwen3-VL-4B-Instruct (4bit QLoRA) |
| **어댑터** | exp17_4b_reorder_sparseaug |
| **학습 Public** | 0.85689 (greedy 추론 시) |
| **추론** | 우도 K=4 순열 TTA |
| **holdout (추론기법 적용)** | 52.38% → **57.14%** (+4.76%p) |
| **규정** | 단일 모델(Qwen+LoRA), 외부 주입 없음 → 완전 안전 |

---

## 1. 학습 스펙 (exp17 어댑터)

`outputs/runs/exp17_4b_reorder_sparseaug/run_config.json` 실측:

```yaml
model:        Qwen3-VL-4B-Instruct
load_4bit:    true              # QLoRA nf4
prompt:       v5_reorder        # 문장 후치 구조
aug_mult:     2
aug_weights:  aug_weights_exp16.csv   # sparse_camX ×4 / 나머지 ×2 (타깃 증강)
lr:           1e-4
lora_r:       16
lora_alpha:   32
lora_targets: q_proj,k_proj,v_proj,o_proj   # LLM attention만 (비전타워 제외)
grad_accum:   16
max_pixels:   307200            # 640×480 (원본 해상도 무손실)
epochs:       1
n_items:      23814             # 9235 × 타깃증강
total_opt_steps: 1488
```

**핵심 설계 근거** (상세: `FINAL_MODEL_EDA_RATIONALE.md`):
- `v5_reorder`: 문장을 답 직전에 배치 (프롬프트 실험 유일 양성, +0.9%p)
- `aug_weights`: sparse_camX(최약점 21.6%) 4배 강조 (문장 4유형 EDA)
- **무작위 셔플** (어려운셔플은 Public -1.1%p로 기각)
- max_pixels 307200: 로컬 원본해상도 (8GB에 들어가는 최대)

---

## 2. 추론 스펙 (우도 K=4 순열 TTA)

**스크립트**: `scripts/score_permutations.py`

```bash
python scripts/score_permutations.py \
  --model ./models/Qwen3-VL-4B-Instruct --load-4bit \
  --adapter ./outputs/runs/exp17_4b_reorder_sparseaug/adapter \
  --prompt v5_reorder \
  --k 4 --prior outputs/prior_exp17.csv \
  --split test --submission exp17_likelihood_k4
```

**알고리즘**:
1. 24개 순열 후보를 teacher-forcing 로그우도로 채점 (생성 아님, 읽기전용)
2. 이미지를 **4가지 순환배치**(라틴방진 e0/r1/r2/r3)로 제시하며 각각 채점
3. 후보별 점수를 **원본 좌표로 합산** → argmax
4. prior 차감(문자열 편향 제거)은 선택 — holdout에선 shuffled 무영향/identity↓

**KV 캐시 공유 구현** (M-RoPE): 프리픽스 1회 forward → 캐시 24배 확장 → 답토큰만 채점.
좌표 규약은 `scripts/test_perm_coords.py`로 단위검증 (answer↔출력이 역순열 관계).

**비용**: test 819 × K4 ≈ 7시간 (4B 4bit, 로컬). holdout 300은 ~2.5h.

---

## 3. 검증 근거

| 지표 | greedy (exp17) | 우도 K=4 |
|---|---|---|
| holdout acc_shuffled | 51.98% | **57.14% (+4.76%p)** |
| Public | 0.85689 | (제출로 검증 예정) |

- **greedy 근시안 회수**: 첫 토큰 argmax가 놓치는 순열을 전수 채점으로 회수
- **위치 편향 상쇄**: K=4 순열 TTA가 "먼저 보여준 프레임=먼저" 편향 완화
- holdout→Public이 그동안 더 크게 벌어졌으니 (exp17: holdout +3.96 vs Public +7.3), Public에서 **0.88~0.90 기대**

---

## 4. 재현 절차 (전체)

```bash
# 1) exp17 어댑터는 이미 학습됨: outputs/runs/exp17_4b_reorder_sparseaug/adapter
# 2) prior 테이블도 생성됨: outputs/prior_exp17.csv
# 3) 우도 K4 추론 → 제출 생성:
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe scripts/score_permutations.py \
  --model ./models/Qwen3-VL-4B-Instruct --load-4bit \
  --adapter ./outputs/runs/exp17_4b_reorder_sparseaug/adapter \
  --prompt v5_reorder --k 4 --prior outputs/prior_exp17.csv \
  --split test --submission exp17_likelihood_k4
# ⚠️ 절전차단 필수 (7h+ 소요, 안 걸면 밤샘 중 사망)
# 산출: outputs/submissions/submission_exp17_likelihood_k4_*.csv
```

---

## 5. Kaggle 8B와의 관계

| | 로컬 (이 문서) | Kaggle 8B |
|---|---|---|
| 모델 | 4B | 8B (체급↑) |
| 학습 셔플 | 무작위 | 무작위 |
| 증강 | 타깃(원본해상도) | 타깃(max_pixels 224) |
| 추론 | 우도 K4 | 우도 K4 |
| 역할 | **안전 폴백** | 도약 시도(0.904) |

- **최종 제출** = 로컬 exp17+K4 vs Kaggle 8B+K4 중 Public 높은 것
- 로컬은 **규정 100% 안전**(순수 Qwen+LoRA, 외부주입 0), 마감 리스크 없음
- Kaggle 8B가 시간/OOM으로 실패해도 이 로컬 후보가 든든한 백업

---

## 6. 규정 적합성
- 단일 모델 (Qwen3-VL-4B + LoRA 어댑터 1개), 앙상블 아님
- 우도 K4 = 추론 전략 (§4.3 TTA 명시 허용)
- prior = train 파생(holdout)에서 동결, test 특성 분석 무저촉
- 입력 = test 문장 + 이미지 4장만 (외부 피처 주입 0)
- gemma/CLIP은 타깃증강 CSV 생성(학습 데이터 선별)에만 사용, 추론 파이프라인 미포함
