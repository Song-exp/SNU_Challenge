# 즉시 인수인계 — 2026-07-20 00:20 갱신

> 새 세션 에이전트가 이 문서만 읽고 이어받을 수 있게 쓴 **실행용** 문서.
> 전체 이력 `HANDOVER.md` | 큰 그림 `VISION_final_modeling.md` | 유형 기준 `REPORT_typing_criteria.md`
> **마감 D-4 (예선 7/24). 제출 = 팀 전체 1일 2회. 주력·폴백 = exp07 (holdout 48.41%, Public 0.766)**

## 0. 지금 돌아가는 것 — exp16 3차 (7/20 00:05 시작)

**exp16_sparsecam_aug**: sparse_camX(비카메라 사건≤2 & 카메라X, holdout 21.6% 최약점) 2,672개 x4 / 나머지 x2 = 23,814항목.
exp07과 `--aug-weights` 하나만 다름 → 순수 증강 효과 측정. **새벽 속도(1.1초/항목)면 오전 중 완료**, 큐가 평가까지 자동.

- 진행: `outputs/runs/exp16_sparsecam_aug/train_log.csv` (opt_step/1488) | 큐: `outputs/train_queue.log`
- **판정**: `experiments.csv`의 acc_shuffled vs exp07 0.4841, ±2%p 게이트 + **sparse_camX 세그먼트가 실제 올랐는지** (`Structure_Pipeline.ipynb` ⑨ 참고, 총점 동률·구성만 다른 exp14 전례)
- 1·2차는 **같은 540스텝에서 CUDA OOM으로 사망**. 조사 결과 거대 샘플은 없었고(데이터셋 균일 640×360) **할당자 단편화**로 결론 → train.py에 메모리 위생 5종 적용됨 (텐서 참조 즉시 해제, 50스텝마다 empty_cache, 저장 후 empty_cache, OOM 샘플 스킵, reserved_vram 로깅). **540스텝 통과가 첫 관문** — 로그의 reserved_vram_gb가 계속 치솟으면 단편화 재발 신호
- 재시작 런처(절전 차단 + ollama 워치독 내장): scratchpad `run_exp16.ps1` — 커맨드 원문은 §5

## 1. ⚠️ 프롬프트 힌트 연동 상황 (7/21 업데이트)

**추론 시 외부 파생 피처(Gemma, CLIP, OWL-ViT) 프롬프트 주입 트랙은 폐기가 아닌 "정현이 형 모델 파이프라인 연동 대기 중" 상태임.**

- **연동 조건**: 오프라인 검증 환경(인터넷 차단)에서 CLIP/OWL-ViT 모델을 로컬 디스크 경로에서 안정적으로 로드할 수 있어야 함.
- **적용 계획**: 로컬 가중치 로딩이 완료되면 `v6_hint` 및 `v7_cot_hint` 프롬프트를 통해 학습/추론에 힌트를 일관되게 주입하여 성능 제고 도모.
- **CLIP 임계값 이원화(V2)**: 장면 전환 분할(Global)에는 **`0.20` 거리 임계값**을 유지하고, 피사체 추적 유실 방지(Local Crop)에는 **`0.30` 거리 임계값**을 독자적으로 튜닝하여 적용 (정상 피사체 오판율 0.0% 확보).

## 2. ⚠️ OWL-ViT 정답 누수(Leakage) 버그 해결 완료

7/19 병합된 `build_comprehensive_hints` 코드에서 정답(Answer)을 역산해 힌트 좌표를 매핑하던 치명적인 데이터 누수 버그가 존재했음.

- **V2 해결 완료**: 정답을 참조하지 않고, 화면에 제시된 셔플 순서(Input_1~4)의 좌표 수치 그대로를 캡션 분석 기반 물리 인과성 가이드라인(카메라 패닝/피사체 이동 분리 규칙)과 결합하는 누수 없는 V2 힌트 모듈을 구현 및 동기화 완료함.
- **추출기 용도**: 향후 힌트 주입 실험용 및 약점 분석/데이터 증강 가중치용으로 병행 활용.

## 3. exp16 판정 후 시나리오 (D-4~D-1)

1. **승리 (+2%p↑)**: 오전 제출 1회로 검증 → 주력 교체. 다음 슬롯 = 무표지 축 or 4B
2. **동률/패배**: 다음 슬롯 = **무표지 축 증강** (미사용 최강 카드: n_markers=0 → 17.2% vs 있음 57.7%, −40.5%p. `outputs/aug_weights` 생성은 `structure_features.make_aug_weights`에 `r.n_markers == 0` 조건, 5분 작업)
3. **4B 스케일업**: 4bit 필수 + 스모크 VRAM 확인. 완주 ~40h라 **7/21 밤이 마지막 시작 기회**
4. 미니 스크리닝 기준선 = mini_v1_aug2 15.5% (+4%p 게이트) | full 판정 = vs exp07 48.41%
5. 재시도 금지 목록: 힌트 주입(§1), CoT SFT(3연패: exp12·mini_gemma_cot·미니 힌트 조합), v4 문구, CLIP 무학습 주입

## 4. 데이터 자산과 허용 용도

| 자산 | 내용 | 용도 (§1 제약 하) |
|---|---|---|
| `outputs/gemma_labels/parts/` | train 9,535 문장 구조 라벨 (실패 0) | 증강 선별·유형·분석 |
| `outputs/gemma_labels/test_*` | test 819 라벨 + features/hints/types CSV | **주입 금지.** 제출 후 분석만 |
| `outputs/gemma_labels/train_types.csv` | 4유형(sparse/dense × camO/X) + 태그 | 증강 축의 근거 (holdout 실측) |
| `snu_clip_features.csv` | train CLIP 거리 (팀원) | 분석·증강 축 후보 |
| `outputs/aug_weights_exp16.csv` | 현 실험 증강 가중 | 사용 중 |
| OWL-ViT 계획서 PDF + 추출 코드 | §2 누수 수정 전제 | 약점 식별·보고서 |

## 5. 실행 커맨드

```powershell
# exp16 재시작 (죽었을 때) — 런처 우선, 없으면 직접:
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe scripts/train_queue.py --queue "exp16_sparsecam_aug|--script train.py --model ./models/Qwen3-VL-2B-Instruct --aug-mult 2 --lr 0.0001 --epochs 1 --max-hours 0 --prompt v1_list --aug-weights ./outputs/aug_weights_exp16.csv --snapshot-steps 150"
# ⚠️ 시작 전: Stop-Process -Name ollama (VRAM 경쟁, 1차 OOM 공범 의심) + 전원 연결·덮개 열기

# 평가 단독 실행 (큐가 못 이었을 때)
.\.venv\Scripts\python.exe scripts/eval_zero_shot.py --model ./models/Qwen3-VL-2B-Instruct --adapter ./outputs/runs/exp16_sparsecam_aug/adapter --prompt v1_list

# 제출 생성 (판정 승자만; scratchpad make_exp14_submission.py 패턴, prompt_lab.make_submission)
```

## 6. 운영 함정 요약 (상세: HANDOVER §7, 1~18번)

- **절전이 모든 것을 죽인다** (누적 4회): 독립 프로세스 런처(Start-Process) + 절전 차단 + 덮개 열기·전원 연결
- **Windows ollama가 트레이에 상주하며 VRAM 탈취** — 학습 전 종료 (런처에 워치독 있음)
- 같은 스텝 반복 실패 = 데이터/재현성 요인, 외부 요인 아님 (OOM 오진 전례)
- eval_zero_shot.py에 팀원의 힌트 주입 코드가 병합돼 있으나 v1_list는 `needs_hint=False`라 안 탐 — **v6/v7 계열 프롬프트로 평가하지 말 것** (누수 경로)
- PowerShell cp949: 커밋 메시지·print에 em dash(—) 등 특수문자 금지, 멀티라인 here-string 파싱 실패 잦음 → `-m` 여러 개로
