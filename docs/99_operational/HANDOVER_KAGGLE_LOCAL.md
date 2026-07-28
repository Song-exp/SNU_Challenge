# 인수인계 — Kaggle & 로컬 현황 (2026-07-23 08:30 기준, 마감 D-1)

> 목표: 본선 커트라인 **Public 0.904**. 현재 최고 **exp17 = 0.85689** (4B).
> 0.904 도약 경로 = **Kaggle 8B** (로컬 8GB는 8B 학습 불가, 채점환경 24GB라 8B 제출은 가능).

---

## 🏆 리더보드 현황 (Public)

| 제출 | 구성 | Public |
|---|---|---|
| exp17 | 4B + v5_reorder + 타깃증강(무작위셔플) | **0.85689 ← 현재 최고·폴백** |
| exp20 | 4B + v5 + **어려운셔플** | 0.84642 (역효과, 셔플 -1.1%p) |
| v10 | 4B + v5 + scene_cuts 힌트 | 0.85514 (힌트 무효) |

**확정 결론:**
- **어려운 셔플 = Public 역효과** → 최종 모델에서 제외, 무작위 셔플 사용
- **이미지 힌트 주입 = 무효** (v10). CoT도 4연패. 전부 재시도 금지
- **우도 K=4 순열 TTA = holdout +4.76%p** (학습X 추론기법, 최대 레버) → 최종에 필수

---

## 🖥️ 로컬 현황 (RTX 5060 8GB)

### 지금 돌아가는 것
- **exp17 + 우도 K4 제출 생성** (07:41 시작, ~14:40 완료 예정)
  - test 819 × K4 ≈ 7시간. 절전차단 내장 독립프로세스 (어젯밤 절전사망 재발방지)
  - 진행 로그: `outputs/exp17_k4.log` | 산출물: `outputs/submissions/submission_exp17_likelihood_k4_*.csv`
  - ⚠️ 어젯밤 exp20 K4가 절전으로 사망(7h 손실) → exp20은 스킵, exp17만 재시작

### 로컬로 짜낸 것 (4B 천장 ~0.86 확인)
- exp17(0.857)이 로컬 최고. exp20(어려운셔플)·우도K1 등은 +0.4%p 수준
- **우도 K4가 유일한 큰 레버**: holdout 52.38%→57.14% (+4.76%p)
- 최종 로컬 후보 = **exp17 + 우도 K4** (완료되면 제출 → 0.88~0.90 기대)

### 로컬 재시작 명령 (죽었을 때)
```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe scripts/score_permutations.py --model ./models/Qwen3-VL-4B-Instruct --load-4bit --adapter ./outputs/runs/exp17_4b_reorder_sparseaug/adapter --prompt v5_reorder --k 4 --prior outputs/prior_exp17.csv --split test --submission exp17_likelihood_k4
# 절전차단 필수 (scratchpad exp17_k4.ps1 참조) - 안 걸면 밤샘 중 사망
```

---

## ☁️ Kaggle 현황 (8B QLoRA — 0.904 도약 승부처)

### 두 트랙 병렬
| 트랙 | 계정 | 코드 | 시작점 | 셔플 |
|---|---|---|---|---|
| **팀원 트랙** | leebyeongcheol | 팀원 V9 원본 | step 511 이어받기 | (V9 설정) |
| **당신(최종) 트랙** | hhhjhyeon | KAGGLE_ALL_IN_ONE.py (최신) | 처음부터 | 무작위(hard_shuffle=False) |

### 당신 최종 트랙 = `final_notebook`
- **T4×2, Commit 실행 중** (백그라운드, 노트북 닫아도 됨)
- 구성: 8B + v5 + 타깃증강(aug_weights 붙어 **23,814항목**) + 무작위셔플 + 추론 우도K4
- ⚠️ **속도 이슈**: step 초반 10초/항목 = 68시간 예상. 워밍업 후 빨라질지 관찰 필요
  - 5초대로 안정되면 팀원과 동급(~34h), 계속 10초면 조정 필요
- ⚠️ **P100 쓰지 말 것** (sm_60, PyTorch 호환 불가). **T4만**

### Kaggle 필수 세팅 (새 노트북/이어받기 공통)
1. **GPU T4×2, Internet On** (P100 금지)
2. Add Input 3개: 대회데이터(`suudata`/`sunaichallenge`) + `aux_upload` + (이어받기면 팀원 체크포인트)
3. 최신 코드: github.com/Song-exp/SNU_Challenge → `kaggle/KAGGLE_ALL_IN_ONE.py`
4. **Commit(Save & Run All)로 실행** (Run 금지 - 세션 끊기면 날아감)

### 첫 셀 실행 시 확인
```
✅ 대회 데이터: .../snuaichallenge_data
aug_weights: .../aug_weights_exp16.csv   ← None이면 aux 재첨부
✅ 학습 항목 23814개                        ← 풀레시피(타깃증강). 9235면 aug_weights 안붙은것
```

### 세션 간 이어달리기 (12h 컷 → 릴레이)
- 8B 풀레시피 = ~26-34h = **2~3세션** 필요 (계정당 주 30h)
- **계정 릴레이**: 세션1 Commit → Output을 다음 계정 Add Input → 코드가 `♻️ 체크포인트 복원` 자동 → 이어감
- ⚠️ **동일인 다계정 금지**(대회 규정=팀 실격). **팀원 각자 계정**으로 릴레이할 것
- 주간 할당 리셋 = 토요일(7/25) = 마감(7/24) 이후라 못 기다림 → 계정 릴레이 필수

### Kaggle 추론 (학습 완주 후)
- 코드 맨 아래 `RUN_INFERENCE = False` → **True**로 → 재실행(Commit)
- 우도 K=4 자동 (test 819 × 4배치 ≈ 5-7h)
- `/kaggle/working/submission.csv` → 다운로드 → 제출
- ⚠️ **추론도 Kaggle에서** (로컬 8GB로는 8B 추론 불가, 67초/샘플+스필)

---

## 🎯 최종 모델 확정 세팅

> **Qwen3-VL-8B + v5_reorder + 타깃증강 + 무작위셔플 (학습)** → **우도 K=4 TTA (추론)**

**넣은 것(검증)**: 8B체급 / v5(+0.9%p) / 타깃증강(0.784레시피) / 우도K4(+4.76%p)
**뺀 것(실측기각)**: 어려운셔플(-1.1%p) / CoT(4연패) / 힌트주입(v10무효) / gemma힌트(미니2패)

---

## ⏭️ 남은 액션 (마감 7/24)

1. **[로컬] exp17+K4 제출** (~14:40 완료) → Public 확인 (0.88~0.90 기대)
2. **[Kaggle] 8B 완주** → 우도K4 추론 → 제출 (0.904 도약 시도)
3. **최종 제출** = 로컬 exp17+K4 vs Kaggle 8B+K4 중 높은 것
4. **폴백** = exp17 (0.857, 이미 제출·검증, 순수 Qwen+LoRA 규정 안전)
5. **미결**: 마감 정확한 시각 미확인 (exp17 K4가 14:40 완료라 마감이 그 전이면 K=2로 단축 필요)

## ⚠️ 함정
- 로컬 절전 = 밤샘작업 사망 (절전차단+독립프로세스 필수)
- Kaggle P100 = PyTorch 호환불가 (T4만)
- Kaggle 그냥 Run = 세션끊기면 날아감 (Commit 필수)
- 동일인 다계정 = 팀 실격 (팀원 계정 릴레이만)
- 어려운셔플·CoT·힌트주입 = 재시도 금지 (실측 기각)
