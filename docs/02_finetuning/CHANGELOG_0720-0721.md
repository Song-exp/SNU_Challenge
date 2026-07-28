# 변경사항 기록 — 2026-07-20 ~ 07-21

> 범위: CoT 4연패 종결 → exp16 주력 등극 → **4B 스케일업 대약진(Public 0.857)** → 이미지 힌트 첫 풀 학습(v10) 가동
> 마감 D-3 (예선 7/24). 이전 이력은 HANDOVER.md / RESUME.md 참조.

---

## 1. 리더보드 현황 (Public)

| 제출 | 구성 | Public | 비고 |
|---|---|---|---|
| exp07 (구 베이스) | 2B + v1 + 균일 aug2 | 0.766 | |
| exp14 | 2B + v5_reorder + aug2 | 0.775 | holdout 동률이나 Public 우세 (첫 타이브레이크 교훈) |
| exp16 | 2B + v1 + sparse 타깃증강 | 0.784 | 7/20 주력 |
| **exp17** | **4B + v5_reorder + sparse증강** | **0.85689** | **7/21 신기록, +7.3%p 대약진** |

**핵심 발견**: holdout(51.98%, +3.96%p)보다 Public(+7.3%p)에서 격차가 더 크게 벌어짐 → 4B+v5 조합이 test 분포에서 특히 강함. holdout은 보수적 추정치.

---

## 2. 확정된 실험 결론 (7/20~21)

### 2a. CoT SFT — 4연패로 영구 종결
- exp12 (spaCy CoT, 풀): 37.7% vs exp06 42.1% = **-4.4%p**
- mini_gemma_cot (v7, gemma events): 3.2% vs 기준 15.5%
- **mini_struct_cot (v8, 구조요약 + 손실가중 0.3)**: 3.2%, identity 92% — "그래디언트 희석" 가설까지 실측 기각
- 결론: 분석문 선행 생성 구조 자체가 identity 지름길 유발. 재시도 금지.

### 2b. 4B 스케일업 — 최대 레버 확정
- 4B 미니 게이트: **25.8% vs 2B 12.3% = +13.5%p** (사상 최대 미니 신호)
- exp17 풀 검증: holdout 51.98%, Public 0.857 — 게이트 신호가 풀에서 유지·증폭
- 4B(4bit) 학습 VRAM 6.2~6.9GB (캡 7.3GB 내), 속도 2.23초/항목 (2B의 2.4배)

### 2c. 타깃 증강의 한계 — exp16 실측
- sparse_camX ×4 증강 → 그 세그먼트 20/83 → 20/83 (**무반응**)
- 취약 유형이 "덜 배워서"가 아니라 "정보가 없어서" 약함 → 단순 증강 무효
- 총점 상승(0.784)은 타깃팅이 아니라 증강량 전체 효과

### 2d. 이미지 힌트 재료 평가
- **OWL-ViT 좌표 (v9)**: 검출률 28.5%, 4프레임 전부 9.3% — 부실. "작은 물체 안 잡힘 / 잡히는 배경은 안 움직임" 구조적 미스매치. **폐기**
- **scene_cuts (CLIP)**: 커버리지 100%, holdout 37.8%p 격차 (cuts0 31% → cuts3 69%) — 강력. **v10으로 채택**

---

## 3. 코드 변경

### 3a. 신규 프롬프트 (`scripts/prompts.py`)
- `v8_struct_cot`: gemma 구조요약 CoT (기각됨, 기록용)
- `v9_owlvit_hint`: OWL-ViT 좌표 힌트 (재료 부실로 미투입)
- **`v10_scenecut_hint`**: v5_reorder + scene_cuts 힌트 (가동 중) — exp17과 힌트 한 줄만 차이

### 3b. `scripts/train.py` — 데이터/신호 레벨 옵션 3종 추가
- `--loss-weights <csv>`: Id별 손실 가중 (증강 복제 없이 그래디언트 기여 조절)
- `--hard-shuffle`: CLIP 유사쌍 순서를 뒤집는 순열 우선 (쌍교환 오답 직격, identity 배제는 타깃 기준)
- `--scene-cut-hints`: scene_cuts 힌트 주입 (v10 전용, 최우선 소스)
- 힌트 소스 우선순위: scene_cuts > OWL-ViT > CLIP 유사쌍
- VRAM 위생 (7/20): `set_per_process_memory_fraction(0.85)` 하드캡 — 공유메모리 스필 원천봉쇄 (크롤·540 OOM 근본원인)

### 3c. `scripts/train_cot.py`
- `build_struct_target` / `load_gemma_structs`: gemma 구조요약 타깃
- `--analysis-loss-weight`: 구간별 토큰 손실 가중 (분석 구간 축소)

### 3d. `scripts/structure_features.py` — 힌트 빌더
- `load_scene_cuts` / `build_scene_cut_hint_text`: scene_cuts 힌트 (전체 속성, perm 불변)
- `load_owlvit_frames` / `build_owlvit_hint_text`: OWL-ViT 좌표 힌트 (제시순서 재매핑)

### 3e. `scripts/eval_zero_shot.py`
- v10 프롬프트는 scene_cuts 경로로 분기 (**팀원 누수 코드 = Answer 역산 경로 우회**)
- scene_cuts는 정답 무관 → 누수 원천 없음

### 3f. 신규 스크립트
- `scripts/extract_owlvit_clean.py`: 누수 없는 OWL-ViT 추출 (제시순서 저장, 쿼리 신뢰도 자동선택 + POS 명사필터)
- `scripts/download_owlvit.sh`: OWL-ViT 가중치 이어받기 다운로더
- `scripts/watch_log.py`: GPU 무점유 학습 감시

### 3g. `scripts/prompt_lab.py`
- `load_model(load_4bit=True)`: 4B 어댑터 제출 생성용 4bit 로드

---

## 4. 현재 가동 중 (7/21 07:22~)

**v10_scenecut_4b**: 4B + v5_reorder + scene_cuts 힌트 + exp16 증강 (~15.5h).
- **프로젝트 최초의 이미지 힌트 풀 학습** (힌트 계열은 지금껏 미니에서만 시험됨)
- 판정: exp17 51.98% 대비 → 이기면 이미지 힌트가 풀에서 통한 첫 증거 / 지면 힌트 계열 최종 종결

---

## 5. 준비된 다음 카드 (순차 누적, exp17 베이스 위에 변수 하나씩)

| 카드 | 변수 | 앙상블 리스크 | 기대 |
|---|---|---|---|
| v10 (가동 중) | scene_cuts 힌트 | 있음(CLIP, 주최측 확인 필요) | 첫 이미지 힌트 풀 |
| exp20 | 어려운 셔플 (--hard-shuffle) | 없음 | 쌍교환 오답 직격 |
| exp19 | 손실 가중 (--loss-weights, 학습불가군 876개 w=0.3) | 없음 | 그래디언트 재분배 |

**원칙**: 한 번에 한 변수. 조합은 무엇이 효과인지 해석 불가 + 상호 간섭(어려운셔플 ↔ scene_cuts 힌트 방향 반대).

---

## 6. 미결 사항
- **앙상블 조항 해석**: CLIP/OWL-ViT 피처의 추론 파이프라인 포함 가부 → 주최측 문의 필요. 폴백 = exp17 (제출·검증됨, 순수 Qwen+LoRA)
- **gemma4 공개일**: 2026-06-03 (커트라인 5/31 초과). 제출물엔 미포함이나 확인 권장. 폴백 = gemma3 재라벨
