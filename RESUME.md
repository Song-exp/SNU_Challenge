# 즉시 인수인계 — 2026-07-19 14:00 (재부팅 직전 작성)

> 새 세션 에이전트가 이 문서만 읽고 이어받을 수 있게 쓴 **실행용** 문서.
> 전체 이력은 `HANDOVER.md`, 큰 그림은 `VISION_final_modeling.md`, 유형 기준은 `REPORT_typing_criteria.md`.
> **마감 D-5 (예선 7/24).**

## 0. 지금 당장 할 일 — exp16 3차 시작

재부팅 직후 GPU가 비어 있는지 확인하고(`nvidia-smi`, 500MB 미만이어야 정상) 아래를 실행:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\송정현\AppData\Local\Temp\claude\C--Users-----Documents-Projects-SNU-AI-Challenge\996fccbf-cd4e-4e84-ac9d-6ad6512faaf6\scratchpad\run_exp16.ps1"
```

이 런처에는 **절전 차단 + ollama 워치독**이 내장돼 있다. (scratchpad가 지워졌으면 §5의 커맨드로 직접 실행)

- 진행 확인: `outputs/runs/exp16_sparsecam_aug/train_log.csv` (마지막 행 = opt_step/1488), 큐 로그 `outputs/train_queue.log`
- 학습 종료 시 큐가 holdout 평가까지 자동 실행 → `outputs/experiments.csv`에 누적
- 소요: 낮 3.3초/항목, 밤 1.1초/항목 → **시작이 14시면 익일 새벽 완료**

## 1. exp16이 무엇이고 왜 중요한가

**exp16_sparsecam_aug = 유형 기반 타깃 증강의 첫 검증.** exp07(현 주력, holdout shuffled 48.41%, Public 0.766)과 **`--aug-weights` 하나만 다르고** 나머지(모델·LoRA·lr·1에폭·v1_list·fp16)는 동일 → 차이 = 순수 증강 효과.

- 증강: `outputs/aug_weights_exp16.csv` — **sparse_camX 2,672개 x4** / 나머지 6,563개 x2 = 23,814항목
- 근거: 4유형 중 sparse_camX가 holdout **21.6%**로 최약점이면서 train의 29% (`REPORT_typing_criteria.md`)
- **판정**: vs exp07 48.41% ±2%p + **sparse_camX 세그먼트가 실제로 올랐는지** 확인 (총점 동률에 구성만 다른 exp14 전례 있음)
- 세그먼트 분해는 `Structure_Pipeline.ipynb` ⑨ 또는 `Structure_Typing_EDA.ipynb` 방식으로

## 2. 오늘(7/19) 실패 이력 — 3차에서 반복하지 말 것

| 시각 | 사건 |
|---|---|
| 02:15 | 1차 실패. **CUDA OOM @ 540스텝**. 당시 Windows ollama 탓으로 오진 |
| 05:06 | 2차 시작 (ollama 종료 + 워치독) |
| 12:59 | 2차 실패. **또 CUDA OOM @ 540스텝** — 시드 고정이라 같은 샘플에서 재현 |

**진단**: 특정 샘플(8,640번째 항목 근처)이 VRAM 스파이크를 일으킨다. 평소 peak 6.47GB인데 가용 7.3GB라 여유가 얇음.
**조치 (적용 완료, train.py)**: ① OOM 샘플 자동 스킵 + `empty_cache` 후 계속 ② `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
→ 3차에서는 "[OOM 스킵 N]" 로그가 몇 개 찍히며 통과해야 정상. **또 죽으면** `--max-pixels`를 640*480→512*384로 낮추는 게 다음 카드 (단 평가도 같은 값으로 맞춰야 비교 유효).

## 3. 규정 미결 — 이게 다음 실험의 성립 조건

`PROJECT_SETUP.md` §4.3: "평가 데이터셋 정보를 학습에 활용하는 행위 금지(수작업 라벨링, 평가 데이터 특성 분석 후 학습 등). 위반 시 실격."

| 등급 | 내용 | 판단 |
|---|---|---|
| 안전 | train 라벨로 증강 선별 (= **exp16**) | test 미접촉 |
| **미결** | train에서 확정한 파이프라인을 test에 동일 적용 (gemma Story note, CLIP 힌트) | 표준 전처리 관행이나 조항 해석 여지 — **주최측 문의 필요** |
| 금지 | test 분포 분석 후 학습 설계 변경, test에 정답 라벨 부여 | 확정 |

- 사용자는 보수적 해석(주입 불가) 쪽으로 기울었고, 그 경우 **v9(세 기법 조합)는 폐기**된다
- ⚠️ 같은 논리면 **팀원의 `src/features/video_feature_extractor.py`(추론 시 CLIP 실시간 주입)도 사용 불가** — 팀 합의 필요
- 참고: 그 추출기는 `torch.hub.load`로 CLIP을 받으므로 **인터넷 차단 검증 환경에서 실패**한다. 쓰게 되면 로컬 가중치 로드로 수정 필수

## 4. 검증 완료된 실험 결과 (재시도 금지)

| 실험 | 결과 | 결론 |
|---|---|---|
| mini_hint_aug2 (v6, CLIP 유사쌍 입력) | 9.9% vs 기준 15.5% | 기각 — identity 답변 28→46%로 지름길 증폭 |
| mini_gemma_cot_aug2 (v7, gemma events 타깃) | 3.2%, identity 81% | 기각 — 분해 품질을 고쳐도 CoT SFT 무익 |
| exp12 (v4 CoT full) | 37.7% vs exp06 42.06% | 기각 (7/16) |
| exp14 (v5_reorder full) | 48.02% ≈ exp07 | 동률, 주력 교체 없음 |

**총평**: gemma/CLIP 정보를 **모델에 주입**하는 경로는 입력·출력 모두 실패. 이 자산의 가치는 **증강 선별·분석·보고서**에 있다.

## 5. 실행 커맨드 모음

```powershell
# exp16 (런처 없이 직접)
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe scripts/train.py --run-name exp16_sparsecam_aug `
  --model ./models/Qwen3-VL-2B-Instruct --aug-mult 2 --lr 1e-4 --epochs 1 --max-hours 0 `
  --prompt v1_list --aug-weights ./outputs/aug_weights_exp16.csv --snapshot-steps 150

# 평가 (학습 후, 프롬프트는 학습과 반드시 동일)
.\.venv\Scripts\python.exe scripts/eval_zero_shot.py --model ./models/Qwen3-VL-2B-Instruct `
  --adapter ./outputs/runs/exp16_sparsecam_aug/adapter --prompt v1_list

# 증강 가중 재생성 (조건·배수 변경 시) — Mini_Hint_Experiments.ipynb ⑩ 또는 structure_features.make_aug_weights
# 유형 EDA: Structure_Typing_EDA.ipynb (전 셀 CPU, 학습과 병행 가능)
```

## 6. exp16 판정 후 (D-4~D-1 계획)

1. **exp16 승리(+2%p↑)** → 7/20 제출 1회로 검증 → 승자를 주력 교체
2. **동률/패배** → 남은 카드:
   - **무표지 축 증강** (미사용 최강 카드: 표지 0개 17.2% vs 있음 57.7%, −40.5%p) — `n_markers==0` 조건으로 가중 CSV만 새로 만들면 즉시 실행 가능
   - **4B 스케일업** (4bit, 스모크에서 VRAM 확인 필수) — 슬롯 1회 필요, 7/21까지 시작해야 완주
   - v9(세 기법 조합)는 §3 규정 답변이 허용일 때만
3. **최종 제출은 단일 모델** — 어댑터 2개 라우팅은 앙상블로 간주되어 실격. 폴백은 exp07(검증된 Public 0.766)
4. 제출은 **팀 전체 1일 2회** — holdout으로 검증하고 확신될 때만 사용

## 7. 운영 함정 (오늘 다 겪은 것)

- **절전으로 3회 작업 사망** (WSL 라벨링 2회, 학습 1회). 세션(하네스) 백그라운드 태스크는 절전을 못 넘김 → **`Start-Process`로 독립 프로세스 + 절전 차단 내장 런처** 사용. 덮개 열기 + 전원 연결
- **Windows ollama가 상주하며 VRAM 경쟁** — 학습 전 `Stop-Process -Name ollama` (런처에 워치독 포함)
- **train_queue 자기 감지 데드락** — `--script` 인자 문자열이 자기 커맨드라인에 잡힘 (수정 완료)
- 인코딩: PowerShell이 cp949라 한글·em dash(—) 출력 시 UnicodeEncodeError 가능 → 스크립트 print에 특수문자 자제
- 전체 함정 목록은 `HANDOVER.md` §7 (1~18번)

## 8. 데이터 자산 현황 (전부 확보 완료)

- `outputs/gemma_labels/parts/*.jsonl` — train 9,535 문장 구조 라벨 (실패 0)
- `outputs/gemma_labels/test_parts/*.jsonl` + `test_features.csv`/`test_hints.csv`/`test_types.csv` — test 819 (⚠️ 규정 §3 미결이라 **추론 주입 보류**, 분석 전용)
- `outputs/gemma_labels/train_types.csv` — 4유형 + 태그 (생성은 `Structure_Typing_EDA.ipynb` ⑤)
- `snu_clip_features.csv` — train CLIP 거리 (팀원 산출물)
- `outputs/aug_weights_exp16.csv` — 현재 실험의 증강 가중
