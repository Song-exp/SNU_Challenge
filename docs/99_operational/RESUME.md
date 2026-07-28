# 즉시 인수인계 — 2026-07-22 02:40 갱신

## 🎯 7/22 새벽 — 목표 0.904(본선 커트라인) 도전, 2트랙 전개

**현황: 최고 Public 0.857 (exp17). 커트라인 0.904까지 +4.7%p 필요 → "도약 아니면 탈락".**

### v10 판정 (scene_cuts 이미지 힌트) — 사실상 무이득
- **v10 = Public 0.85514, exp17(0.85689)보다 -0.00175** (거의 동률, 못 넘음). holdout 평가는 제출용으로 중단(사용자 지시)하고 test 직접 제출로 판정
- **이미지 힌트 주입 = 풀 스케일에서도 무효 확정** (미니 2연패 + 풀 1무). scene_cuts는 커버리지 100%·holdout 37.8%p 격차의 강신호였는데도 안 됨 → "모델에 텍스트로 떠먹이기" 자체가 안 통함. 힌트 주입 계열 완전 종결
- 부산물: **CLIP 동적 scene_cuts 추론 파이프라인 완성** (`scripts/clip_scene_cuts.py`, 정합성 99% 검증, 로컬 CLIP `models/clip-vit-base-patch32`). 힌트는 폐기지만 본선 재현 가능 구조로 남김

### 우도 스코어링 (추론 레버, 학습 슬롯 0) — 구현·검증 완료
- `scripts/score_permutations.py` + `scripts/test_perm_coords.py`(좌표 단위테스트). **KV캐시 공유 M-RoPE 구현 성공** (7.5초/샘플, 캐시=비캐시 우도 일치 검증). 좌표 규약 = answer와 모델출력이 **역순열** 관계(단위테스트로 확정)
- **holdout K=1 = acc_shuffled 52.38% vs exp17 greedy 51.98% (+0.4%p)** — 유효하나 작음. prior 테이블 `outputs/prior_exp17.csv` 저장됨. 2·3단계(prior차감·K=4 TTA)는 중단, 최종 승자에 재적용 가능
- ⚠️ 어댑터 유무로 rope_deltas 경로 다름: PeftModel=`m.base_model.model.model`, 无=`m.model` (하니스는 자동 탐색)

### 8B 모델 — 로컬 불가, Kaggle 트랙 개설
- **로컬 8GB에서 8B 학습 물리적 불가**: 4bit 로드 5.96GB + kbit fp32 업캐스트 2.3GB → OOM. 업캐스트 생략해도 학습스텝 peak 8.03GB로 23/24 OOM. zero-shot 우도도 67초/샘플+정확도 0%(미파인튜닝). **4B가 이 GPU 학습 상한**
- **핵심 인사이트(사용자 지적)**: 규정 채점환경 = RTX 3090 **24GB**. 8B는 채점상 문제없음(공개일 2025-10, 단일모델, 추론 24h 내). "내 로컬에서 학습만 못 할 뿐" → **Kaggle 무료 GPU(T4×2 16GB, 주30h)에서 8B QLoRA 학습 가능** 확인
- **Kaggle 패키지 완성·git 푸시** (`kaggle/`): `KAGGLE_ALL_IN_ONE.py`(학습+추론, 경로 자동탐색, 체크포인트 재개), `사용법_초보용.md`, aux CSV 3개. 레시피 = exp17(8B+v5+타깃증강+어려운셔플+max_pixels 512×384)+우도. 12h 세션 관리가 관건 → 사용자가 실행 예정

### 로컬 진행 (안전 폴백 강화)
- **🔄 exp20_4b_hardshuffle (02:31~, ~15.5h)**: 4B + v5 + **어려운 셔플**(CLIP 유사쌍 순서 뒤집기, 8446/9535 적용) + exp16증강. 주 오답(쌍교환) 공략. 낮 완주·자동평가. 판정 vs exp17 51.98%
- train.py 신규 옵션: `--hard-shuffle` `--loss-weights` `--scene-cut-hints` `--owlvit-hints` `--mem-fraction` `--skip-kbit-upcast`. 손실가중 CSV `outputs/loss_weights_exp19.csv`(876개 w0.3) 준비됨

### 다음 우선순위 (마감 7/24, D-2)
1. **Kaggle 8B** (사용자 실행) — 유일한 체급 도약 = 0.904 실질 경로. exp17 넘으면 최종제출
2. **exp20 판정** (낮) → 이기면 어려운셔플 채택
3. **우도 K=4** — 최종 승자 어댑터에 얹기 (학습 슬롯 0)
4. exp19(손실가중) — 여유 슬롯 시
- ⚠️ 미커밋 대량: 우도 하니스·CLIP 파이프라인·train.py 옵션들 (kaggle/만 커밋함). 정리 필요
- 8B 다운로드 17GB `models/Qwen3-VL-8B-Instruct/` 로컬 학습엔 무용(지워도 됨), Kaggle이 자체 다운

## 🏆 7/21 대약진 — 4B 스케일업으로 Public 0.857 (상세: CHANGELOG_0720-0721.md)

- **exp17 (4B + v5_reorder + exp16증강) = Public 0.85689** — 종전 최고 exp16(0.784) 대비 **+7.3%p**. holdout 51.98%(+3.96%p)보다 Public 격차가 더 큼 → 4B+v5가 test 분포에서 특히 강함. **새 주력·폴백 = exp17** (순수 Qwen+LoRA, 규정 논란 없음)
- 4B 미니 게이트 +13.5%p(사상 최대) → 풀에서 검증 완료. 4B(4bit) VRAM 6.2~6.9GB, 2.23초/항목
- **CoT 4연패 종결**: v8(구조요약+손실가중)도 3.2%/identity 92% — "그래디언트 희석" 가설까지 기각. 재시도 금지
- **OWL-ViT 좌표 힌트 폐기**: 검출률 28.5%/4프레임전부 9.3% (작은물체 안잡힘·배경은 안움직임). 재료 부실
- **🔄 가동 중: v10_scenecut_4b (07:22~, ~15.5h)** — 4B + v5 + **scene_cuts 힌트**(CLIP, 커버리지 100%, holdout 37.8%p 격차) + exp16증강. **프로젝트 최초 이미지 힌트 풀 학습**. exp17에 힌트 한 줄만 추가 → 판정 vs 51.98%. 내일 오전 완주
- 다음 카드(순차, exp17 베이스): exp20(--hard-shuffle, 쌍교환 직격) → exp19(--loss-weights, 학습불가군 876개 w=0.3). 조합 금지(해석 불가+간섭)
- 미결: 앙상블 조항(CLIP/OWL 추론 포함 가부) 주최측 문의 → 폴백 exp17. gemma4 공개일 6/3(커트라인 초과)

## ⚡ 7/20 새벽 결과 속보 (상세는 아래 원문 + HANDOVER)

- **v8 구조 CoT (mini_struct_cot_aug1) 기각 — CoT SFT 4연패 확정**: gemma 구조 요약 타깃 + 분석 구간 손실 가중(×0.3, `--analysis-loss-weight` 신규 구현)까지 적용했으나 shuffled **3.2% vs 공정 기준점 mini_v1_aug1 12.3%** (동일 aug1·1000샘플), identity 92%. "그래디언트 희석" 가설도 실측 기각 — 분석문 선행 생성 구조 자체가 identity 지름길 유발. **CoT 트랙 영구 종료** (v8·손실가중 포함 재시도 금지 목록에 추가)
- 신규 인프라 (재사용 가능): `prompts.py` v8_struct_cot, `train_cot.py` build_struct_target/load_gemma_structs/--analysis-loss-weight (구간별 토큰 손실 가중 — 다른 실험에서 쓸 일 있으면 참조)
- **exp16 3차 큐로 재시작됨 (03:26~)**: 미니 쌍 → exp16 순서의 3연전 큐 (01:09 가동, 독립 프로세스). 80스텝 통과, peak 6.45GB 안정. 540스텝 관문 오전 중 도달 예상 → 통과 시 판정은 저녁
- **🔧 exp16 크롤·540 OOM의 진짜 사인 규명 + 수정 (7/20 새벽, 3~5차 시도)**: reserved 16.37GB는 로깅 버그가 아니라 **실제값 = 전용 8.15 + 공유 8.2GB의 절대 한도**. torch 할당자가 예약을 공유(시스템 RAM) GPU 메모리까지 부풀림 → PCIe 스래싱(클럭 352MHz, 3~13초/항목 크롤) → 한도 소진 시 hard OOM(=540스텝 사망의 정체). dwm/창 정리로는 안 풀림 (4차: 깨끗한 VRAM 시작에도 20스텝 만에 재발). **수정 3종 (train.py)**: ① `set_per_process_memory_fraction(0.85)` 하드 캡 — 스필 원천 봉쇄, 초과분은 OOM 스킵으로 처리 ② expandable_segments(Windows 미지원) → `garbage_collection_threshold:0.8,max_split_size_mb:256` ③ reserved 로깅 max→현재값. **5차(06:36~) 실측: 0.92초/항목 유지, reserved 5~6.3GB 캡 안 — 완치 확인.** 완주 ~12:45, 판정 13시대 예상
- 신규 유틸: `scripts/watch_log.py` — GPU 무점유 학습 감시 (파일명에 train 금지 이유 포함)
- **⚠️ 규정 공개일 확인 완료 (7/20, 웹 공식 소스)**: Qwen3-VL-2B = 2025-10-21 ✅ 안전 | **gemma4:12b = 2026-06-03 → 커트라인(5/31) 3일 초과**. 제출물(Qwen+LoRA)엔 gemma 미포함이라 직접 위반은 아니고, §4.3이 전처리용 외부 상용 API를 명시 허용하므로 공개일 조항은 제출 시스템 구성 모델 대상으로 해석함이 일관적. 대응: ① 주최측에 전처리 도구 적용 여부 문의 권장 ② 폴백 = gemma3:12b(2025-03-10 공개, 허용)로 재라벨링 ~24h — 예선 후 슬롯에 재현 실험, camera 축은 이미 정규식 이식 완료

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
