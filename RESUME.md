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

## 1. ⚠️ 확정된 제약 (7/20, 사용자 결정) — 모든 설계의 전제

**추론 입력 = test 문장 + 이미지 4장만. 파생 정보(gemma 라벨·CLIP 피처·OWL-ViT 좌표) 주입 일절 불가.**

- 학습 프롬프트에 힌트를 넣고 추론에서 빼는 것도 불가 (입력 분포 불일치) → **힌트 주입 트랙 전면 폐기** (실측도 2연패: mini_hint 9.9%, mini_gemma_cot 3.2% vs 기준 15.5%)
- gemma/CLIP/OWL-ViT 자산의 허용 용도: **train 증강 선별, 데이터 정제, 오답 분석, 본선 보고서** — 추론 파이프라인에는 절대 안 들어감
- 최종 제출물 = Qwen + LoRA 어댑터 1개, 규정 논란 여지 없는 구성

## 2. ⚠️ 팀원 코드의 정답 누수 (공유 필요, 미해결)

7/19 pull된 OWL-ViT 파이프라인(`scripts/extract_owlvit_features.py` + `structure_features.build_comprehensive_hints` + eval_zero_shot 힌트 주입부)이 **train 좌표를 Answer로 시간순 정렬해 저장**하고 평가 시 정답에서 역산한 순서로 힌트를 만든다 → 정답 유출. 이 경로로 나온 성능 수치는 전부 무효이며 test에선 KeyError.

- 수정 방향: 좌표를 **제시 순서(Input_1~4)** 로 저장 + 증강 변형은 재매핑 (CLIP `remap_pairs` 패턴)
- §1 제약 확정으로 힌트 주입 용도는 어차피 폐기 — 추출기는 **약점 식별(증강 축)·분석용**으로 전환하라고 팀에 전달할 것
- 그 코드의 `torch.hub`/`from_pretrained` 원격 로드도 검증 환경(인터넷 차단)에서 실패함
- 팀원 계획서 PDF 평가와 적용 계획은 이 대화 기록 및 아래 §4 참조

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
