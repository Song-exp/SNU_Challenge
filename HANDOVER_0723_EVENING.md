# 인수인계 — 2026-07-23 저녁 (마감 7/24 23:59, D-1)

> 목표: 본선 커트라인 **Public 0.904**. 현재 최고 exp17 = 0.85689.
> 세 트랙 병렬 진행 중. **가장 급한 것: 팀원 우도 0.44 → 재배열로 고치기 (30초)**

---

## 🔴 최우선 액션 — 팀원 우도 K4 결과 재배열 (재추론 X)

**증상**: 팀원 8B(step 977) 우도 K4 추론 결과가 **0.44** (무작위 수준).
**원인**: 버그. 우도 모드에서 `best`(이미 정답형식)를 **한 번 더 역순열 변환**해서 답이 뒤집힘.
**해결**: 재추론 불필요. 이미 나온 `submission(2).csv`를 역변환 한 번 더 하면 원복 (검증 24/24).

```python
# Kaggle 새 셀에 붙여넣고 실행 (30초)
import pandas as pd, ast
df = pd.read_csv("/kaggle/working/submission(2).csv")   # 경로는 실제 파일명 확인
def fix(s):
    w = ast.literal_eval(s); f=[0]*4
    for i,n in enumerate(w): f[n-1]=i+1
    return str(f)
df["Answer"] = df["Answer"].apply(fix)
df.to_csv("/kaggle/working/submission_fixed.csv", index=False)
print("✅ 재배열 완료:", len(df))
```
→ `submission_fixed.csv` 다운 → 제출. **greedy 0.839 → 우도 0.88+ 기대**.

**근본 수정** (앞으로 우도 코드 쓸 때): 우도 모드 끝의
```python
best = max(CANDS, key=lambda a: score[tuple(a)])
sub=[0]*4
for i,n in enumerate(best): sub[n-1]=i+1   # ← 이 3줄 삭제
recs.append({"Id":row["Id"],"Answer":str(sub)})
```
→ `recs.append({"Id":row["Id"],"Answer":str(best)})` (best 그대로).
※ 로컬 `score_permutations.py`는 best 그대로 써서 이 버그 없음 (정상).

---

## 📊 세 트랙 현황

### 트랙 1: 로컬 exp17 + 우도 K4 (규정 안전 폴백)
- **모델**: 4B (exp17 어댑터, Public 0.857 검증됨) + 우도 K4
- **상태**: 07:57 시작, 9시간+ 진행 중 (예상보다 느림, 곧 완료)
- **산출**: `outputs/submissions/submission_exp17_likelihood_k4_*.csv`
- **기대**: holdout 57.14% → Public 0.88~0.90
- **규정**: 순수 Qwen+LoRA, 외부주입 0 → 완전 안전. **든든한 백업**
- 재시작 명령: `LOCAL_FINAL_SPEC.md` 참조 (절전차단 필수)

### 트랙 2: Kaggle 팀원 8B (step 977) ← Kaggle 메인
- **모델**: 8B, 팀원 학습 (aug2 + 어려운셔플 + max_pixels 512×384)
- **greedy 제출**: **0.839** (4B 0.857 못 넘음 — 어려운셔플·부분학습 탓 추정)
- **우도 K4**: 버그로 0.44 → **위 재배열로 고치면 0.88+ 기대**
- ⚠️ 팀원 학습이 **어려운셔플(hard_shuffle=True)** 사용 → Public 손해 가능성. 그래도 8B 체급+우도로 만회 기대

### 트랙 3: Kaggle 당신 계정 8B (step 511)
- **모델**: 8B, FINAL_8B_v2 (aug1 + 무작위셔플 + max_pixels 224)
- **상태**: **step 511/1488 (34%)에서 세션종료·저장됨** (11.3h 컷)
- **속도 문제**: 216초/스텝 (T4×2 분산 오버헤드 추정) → 완주까지 ~50h = 마감 불가
- **판단**: 완주 사실상 포기. 팀원 트랙(977)이 더 학습됨 → **우선순위 낮음**
- 살리려면: step511 어댑터로 우도 추론 가능하나 팀원 977보다 열위

---

## 🎯 최종 제출 후보 (마감까지 정할 것)

| 후보 | 예상 Public | 규정 | 상태 |
|---|---|---|---|
| **exp17 + 우도K4 (로컬)** | 0.88~0.90 | 안전 | 곧 완료 |
| **팀원 8B + 우도K4** | 0.88+ (재배열 후) | 8B 안전 | 재배열 대기 |
| exp17 greedy (폴백) | 0.857 | 안전 | 제출·검증됨 |

**전략**: 로컬 exp17+K4 와 팀원 8B+K4(재배열) **둘 다 제출**해서 높은 것 채택.
폴백은 exp17(0.857). 제출 슬롯 하루 2회 확인.

---

## ✅ 확정된 설계 (근거: FINAL_MODEL_EDA_RATIONALE.md)

**최종 레시피** = 8B/4B + v5_reorder + 타깃증강(sparse_camX) + **무작위셔플** + **우도 K4**

- **우도 K4** = 최대 레버 (holdout +4.76%p). 학습X 추론기법, 규정 안전(TTA 허용)
- **어려운셔플 = 제외** (Public -1.1%p 실측). ⚠️ 팀원 트랙은 이걸 써서 손해 가능
- 재시도 금지: CoT(4연패), 힌트주입(v10무효), gemma힌트

---

## 📁 GitHub 파일 (github.com/Song-exp/SNU_Challenge)

| 파일 | 용도 |
|---|---|
| `kaggle/FINAL_8B_v2.py` | 8B 학습+추론 (당신 트랙, 무작위셔플+K4) |
| `kaggle/INFER_ONLY_K4.py` | 추론 전용 (어댑터→submission, ⚠️best 변환 확인) |
| `kaggle/aux_upload/aug_weights_exp16_half.csv` | 항목 절반 가중 |
| `FINAL_MODEL_EDA_RATIONALE.md` | EDA↔세팅 근거 |
| `LOCAL_FINAL_SPEC.md` | 로컬 최종 후보 재현 스펙 |
| `scripts/score_permutations.py` | 로컬 우도 채점 (검증본, best 그대로 사용) |

---

## ⚠️ 함정 정리
- **우도 답 변환**: best는 이미 정답형식 → 추가 변환 금지 (팀원 0.44 원인)
- **어려운셔플**: Public -1.1%p → 최종 제외
- **Kaggle P100**: PyTorch 호환불가 (T4만) / 그냥 Run = 세션끊기면 날아감 (Commit 필수)
- **동일인 다계정 = 팀 실격** (팀원 계정 릴레이만)
- **8B T4 학습**: 216초/스텝 느림(분산). max_pixels·항목 줄여도 마감 빠듯
- 로컬 절전 = 밤샘작업 사망 (절전차단 필수)

---

## ⏭️ 지금 당장 할 일 (우선순위)
1. **팀원 우도 0.44 재배열** (30초) → 제출 → 0.88+ 확인 ← 최급
2. **로컬 exp17+K4 완료 대기** (곧) → 제출
3. 두 결과 비교 → 높은 것 최종 제출
4. 폴백 exp17(0.857) 확보됨
5. 당신 트랙(step511)은 보조 — 여유 시 우도 추론
