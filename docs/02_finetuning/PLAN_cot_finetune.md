# CoT 파인튜닝 설계 (exp12_v4cot_aug1) — 2026-07-15

팀 고도화 프롬프트(`v4_story_cot`)를 **학습에 적용**하는 실험. 추론만으로는 CoT가 작동하지
않음이 실측됐고(7/15: 형식은 따르나 취약 세그먼트 22%→12% 붕괴), "추론 능력은 학습으로만"이
결론이었다 (`PLAN_prompt_experiments.md` 실측 결과 참고).

## 1. 가설·비교 설계

- **가설**: CoT 형식의 응답(이벤트 분해 → 시간 매핑 → 답)을 지도하면, 모델이 문장 구조를
  명시적으로 활용하는 법을 배워 특히 취약 세그먼트(Type-1/모호 문장)가 개선된다.
- **비교 대상**: `exp06_aug1_full` (같은 1배 증강, v1_list 즉답, shuffled **42.06%**)
  — 증강 배수가 같아 "CoT 응답 학습" 효과만 격리됨. exp07(48.4%, 증강2)과의 비교는 오염.
- **판정**: exp06 대비 acc_shuffled **+4%p 이상**이면 CoT 승 → 증강2 버전(exp13)으로 확전.
  exp07(48.4%)까지 넘으면 주력 교체 후보.

## 2. 파일 구조 — 기존 파이프라인 무변경 원칙

| 파일 | 상태 | 내용 |
|---|---|---|
| `scripts/train_cot.py` | **신규** | CoT 학습 전용. train.py의 순수 헬퍼(build_messages 등)만 import — **train.py는 한 줄도 수정 안 함**. 학습 루프·LoRA·안전장치(max-hours/스냅샷/절전차단)는 train.py와 동일 설계 |
| `scripts/eval_zero_shot.py` | 보강 | 파서에 `<ANSWER>` 태그 우선순위 추가 (태그 없으면 기존 동작 100% 동일 — 단위테스트 6/6 통과). 기존 실험 재채점에 영향 없음 |
| `scripts/prompts.py` | 기등록 | `v4_story_cot` (팀 원문) — 학습·평가가 같은 이름 공유 |

## 3. CoT target 생성 레시피 — "헛소리 없는 기계 생성"

원칙: **문장·정답에서 역산 가능한 사실만** target에 넣는다. 이미지를 안 보고 쓴 추론문은
모델에게 그럴듯한 헛소리를 가르치므로 배제.

```
[Story Analysis]
- Event 1: A hand lowers a tool onto a towel        <- spacy 절 분해 (문장에서만 유도)
- Event 2: moves to carefully work on the nails
- Event 3: shifts away
[Chronological Mapping]
- 1st: Image 3                                       <- 정답에서 역산 (사실)
- 2nd: Image 1                                       <- "because ..."는 넣지 않음
...
[Final Answer]
<ANSWER>[3, 1, 2, 4]</ANSWER>                        <- 파서가 태그 우선 파싱
```

- **이벤트 분해기** (`EventSplitter`): 절 머리동사(ROOT/conj/advcl/ccomp) 왼쪽 경계 +
  시간 부사(then/finally/...) + 전치사 뒤 동명사(before cutting...)를 자름점으로.
  접두/접미 접속어 트림, 한 단어 이벤트 제거, 최대 5개. flag_detector의 3유형 분류와
  같은 의존관계 기준 → 문장 유형 파이프라인과 일관.
- **[Visual Evidence]는 1차 버전에서 제외** — 이미지를 안 보고 쓸 수 없는 유일한 섹션.
  프롬프트는 4단계를 지시하지만 target이 이 섹션을 건너뛰므로 모델도 건너뛰는 응답을 학습
  (형식 불일치는 무해 — 응답 분포는 target이 결정). **2차 옵션**: CLIP 유사쌍 한 줄로 채우기
  (train 전체 CLIP 피처 보유, 사실 기반) — 1차 결과 본 뒤 변수 하나씩.
- **알려진 한계**: 분해 노이즈 잔존 ("We see" / "A person is" 류 껍데기 이벤트 소수) —
  사실 기반이라 헛소리 학습 위험은 없음. 미리보기로 표본 검사: `--preview N`.

## 4. 실행 커맨드 (순서대로)

```bash
# ① target 미리보기 (GPU 불필요, 형식 눈검사)
python scripts/train_cot.py --run-name preview --preview 5

# ② 스모크 (~3분) — peak VRAM 반드시 확인 (8GB 한계)
python scripts/train_cot.py --run-name exp12_v4cot_aug1_smoke --max-samples 12 --grad-accum 4 --max-steps 5

# ③ (권장) 미니 스크리닝 (~2h) — CoT 응답이 파싱 가능하게 수렴하는지 조기 확인
python scripts/train_cot.py --run-name exp12_v4cot_mini --max-samples 1000 --max-steps 300
python scripts/eval_zero_shot.py --model ./models/Qwen3-VL-2B-Instruct --adapter ./outputs/runs/exp12_v4cot_mini/adapter --prompt v4_story_cot --max-new-tokens 512

# ④ 본학습 (1배 증강 완주, 예상 11~13h — 밤 배치)
python scripts/train_cot.py --run-name exp12_v4cot_aug1 --aug-mult 1 --snapshot-steps 150

# ⑤ 평가 (~40-75분: 생성이 길어짐)
python scripts/eval_zero_shot.py --model ./models/Qwen3-VL-2B-Instruct --adapter ./outputs/runs/exp12_v4cot_aug1/adapter --prompt v4_story_cot --max-new-tokens 512
```

## 5. 비용·리스크

| 항목 | 예상 | 근거·대응 |
|---|---|---|
| 학습 시간 | 11~13h (exp06 9.6h 대비 +15~30%) | target 12→~100+토큰. max-hours 없이 완주 설계, 스냅샷 150스텝 |
| peak VRAM | 6.99GB보다 상승 가능 | **스모크에서 확인 필수**. 초과 시 이벤트 수 축소(5→3)로 target 압축 |
| 평가 시간 | 40~75분 (300개 × 8~15초) | `--max-new-tokens 512` 필수 — 32로 돌리면 전부 파싱 실패 |
| 제출 추론 | 819개 × ~8-15초 ≈ 2~3.5h | 규정 한도 24h 내 여유. 채택 시에만 감수 |
| 파싱 붕괴 | 낮음 | 태그 파싱 단위테스트 6/6 + exp07이 zero-shot으로도 태그 형식 준수했음(parse_fail 0) |
| 과최적화 위험 | Story Analysis가 문장 베끼기로 퇴화 | loss 곡선 + 평가 raw에서 응답이 이벤트 분해를 실제로 수행하는지 질적 확인 |

## 6. 진행 체크리스트

- [x] `train_cot.py` 작성 (train.py 무변경) — 7/15
- [x] `<ANSWER>` 파서 보강 + 단위테스트 6/6 — 7/15
- [x] target 미리보기 품질 검사 (분해기 1차 개선 완료) — 7/15
- [ ] 스모크 (제출 추론 종료 후) → peak VRAM 기록
- [ ] 미니 스크리닝 → 응답 형식 수렴 확인
- [ ] 밤 배치 exp12_v4cot_aug1 완주
- [ ] holdout 평가 → exp06(42.06%)과 세그먼트 분해 비교 → 확전/보류 판정
- [ ] 결과를 EXPERIMENTS.md·PLAN_prompt_experiments.md에 기록
