# Kaggle 8B QLoRA 학습·제출 가이드

> 목적: 로컬 8GB에서 불가한 **Qwen3-VL-8B**를 Kaggle GPU(16GB)에서 파인튜닝 → 0.904 커트라인 도전.
> 전제: 로컬 4B(exp17 0.857)는 안전 폴백으로 계속 유지. Kaggle 8B는 상방 도박.

## 왜 이 레시피인가 (실측 근거)

**넣은 것 (검증됨):**
- 8B 모델 — 4B→8B 체급 도약 (로컬 미니에서 2B→4B가 +13.5%p였던 것처럼)
- v5_reorder 프롬프트 — Public +0.9%p (exp14 vs exp07)
- 타깃 증강 (sparse_camX x4) — Public 0.784 레시피 (exp16)
- 어려운 셔플 — 주 오답(쌍교환) 공략
- max_pixels 상향 (512×384) — 16GB 여유로 시각 정보 ↑ (로컬은 VRAM 부족으로 307200)

**뺀 것 (실측 기각 — 넣으면 점수 하락):**
- CoT (4연패, identity 붕괴) / scene_cuts·OWL-ViT 힌트 주입 (v10 무효)
- gemma 힌트 (미니 2연패) / 역할부여 프롬프트 (노이즈)

## 절차

### 0. 준비 (한 번만)
1. Kaggle 계정 GPU 인증 (전화번호) 확인
2. **대회 데이터 확인**: `/kaggle/input/`에 SNU 데이터셋 attach.
   - 팀원 경로 참고: `snu-ai-challenge-data`. 실제 slug는 대회 Data 탭에서 확인.
   - `kaggle_8b_train.py`의 `CONFIG["data_dir"]`를 실제 경로로 수정.
3. **보조 데이터 업로드**: `kaggle/aux_upload/` 3개 CSV를 Kaggle Dataset으로 (이름: `snu-ai-aux`)
   - aug_weights_exp16.csv, snu_clip_features.csv, holdout_300.csv (총 1.8MB)
   - `CONFIG`의 aug_weights/clip_features/holdout 경로를 그 Dataset 경로로 수정.

### 1. 학습 (kaggle_8b_train.py)
- 새 Notebook → GPU 켜기 (**T4×2 권장** — 16GB×2, device_map=auto가 분산) → 인터넷 ON (모델 다운로드용)
- `kaggle_8b_train.py` 내용을 셀에 붙여넣기 (또는 Utility Script로 import)
- **실행 → 12시간 세션**. 11.3h에서 자동 안전저장 후 종료.
- **진행률 100% 아니면**: 노트북 **재실행** → 체크포인트에서 이어서 학습 (`/kaggle/working/exp_8b/ckpt`).
  ⚠️ working 디렉토리는 세션 간 유지 안 됨 → 세션 끊기면 **ckpt를 Dataset으로 저장 후 재업로드**하거나, "Save Version(Commit)"으로 output 보존.
- 완주하면 `/kaggle/working/exp_8b/adapter` 생성.

### 2. 추론·제출 (kaggle_8b_infer.py)
- 학습 완주 후 같은/새 노트북에서 실행 (adapter 경로 연결)
- `use_likelihood=True` (권장) — 24후보 우도 채점 (로컬에서 캐시 구현·검증한 방식, +정확도)
- test 819 → `/kaggle/working/submission.csv` 생성 → 다운로드 → 리더보드 제출

## 체크포인트/재개 주의 (12h 세션의 핵심)
- 세션이 12h에 끊기면 `/kaggle/working`이 날아갈 수 있음. **대응 2가지**:
  - (간단) 노트북 "Save & Run All (Commit)" → output이 버전에 보존됨 → 다음 세션에서 그 output을 input으로 attach → ckpt 경로 연결
  - (확실) 학습 중 ckpt를 주기적으로 Kaggle Dataset API로 push
- `save_every_steps=100`이라 최대 100스텝(~15분)만 손실.

## 예상 시간
- 8B, 23,814항목, T4×2. 항목당 ~2-3초 추정 → **13-20시간** = 2세션 예상.
- 추론(우도, test 819) → ~1-2시간.
- **마감 7/24 역산**: 7/22 밤 시작 → 7/23 완주·제출이 현실적. 세션 관리가 관건.

## 로컬과의 관계
- 로컬 exp20(4B 어려운셔플)은 별개로 진행 → 폴백/비교군.
- Kaggle 8B가 exp17(0.857) 넘으면 → 그게 최종 제출. 못 넘거나 실패 → 로컬 최고 유지.
- 최종 제출물 규정: 8B라도 3090(24GB)에서 추론 OK, 단일 모델, 공개일 통과 — 문제없음.
