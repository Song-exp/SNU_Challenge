# SNU AI Challenge — Text-guided Video Frame Ordering

**최종 성적**: Private 0.86991 (1위) / Public 0.88307 (팀 최고 Public 0.88830)  
**최종 시스템**: Qwen3-VL-8B (NF4 4bit QLoRA) + v5_reorder + 문장 4유형 차등 증강 + 하드 셔플 + 우도 K=4 순열 TTA

---

## 1. 환경 (Environment)
* **OS / GPU**: (개발) Windows 11 · RTX 5060 8GB, (학습) Kaggle T4×2, (검증/실행 대상) RTX 3090 24GB · CUDA 12.4
* **Python / 라이브러리**: Python 3.10+ / `requirements.txt` (transformers==5.13.0 등 버전 고정)
* **환경 설치**:
  ```bash
  pip install -r requirements.txt
  ```
  *(※ 추론 및 평가 코드는 오프라인 환경에서도 실행 가능하며 런타임 pip install이 없습니다.)*

---

## 2. 데이터 배치 (Data Layout)
기본 실행을 위해 데이터는 프로젝트 루트의 `./data/` 경로에 배치합니다:
```
./data/
├── train.csv
├── test.csv
├── train/
└── test/
```

---

## 3. 모델 가중치 (Model Weights)
* **LoRA 어댑터 (최종)**: `weights/adapter_8b_final/` (레포지토리에 직접 포함)
* **베이스 모델 (`Qwen3-VL-8B-Instruct`)**:
  자동 다운로드 스크립트를 사용하여 다운로드:
  ```bash
  python final_code/download_weights.py --out-dir ./models/Qwen3-VL-8B-Instruct
  ```
  또는 HuggingFace Hub 링크: [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
* **오프라인 실행 시**: 가중치 다운로드 후 `HF_HUB_OFFLINE=1` 환경 변수를 설정하여 인터넷 연결 없이 추론 가능합니다.

---

## 4. 추론 재현 (Inference Reproduction)
단일 RTX 3090 (24GB VRAM) 기준, 제출용 추론 실행:
```bash
python final_code/INFER_ONLY_K4.py --data ./data --adapter ./weights/adapter_8b_final --out ./outputs/submission.csv
```
* **소요 시간**: test 819건 기준 약 1.5~2시간 (K=4 우도 TTA 연산 포함)
* **Greedy 모드 (빠른 검증)**:
  ```bash
  python final_code/INFER_ONLY_K4.py --no-likelihood
  ```

---

## 5. 학습 재현 (Training Reproduction)
1. **전처리 준비**: Gemma 라벨 및 aux 데이터세트를 기반으로 `aug_weights` 준비 (산출물 `./aux/` 동봉)
2. **학습 실행**:
   ```bash
   python final_code/FINAL_8B_v2.py --data ./data --aux ./aux --out ./weights/adapter_8b_final --ckpt ./outputs/ckpt
   ```
   *(Kaggle T4×2 기준 약 11시간, 1,488 steps / step 977 체크포인트 최적)*

---

## 6. 디렉토리 구조 (Directory Map)
* `final_code/`: 제출 파이프라인 진입점 및 상대 경로 실행 스크립트
* `weights/`: 최종 LoRA 어댑터 가중치 (`adapter_8b_final`)
* `aux/`: 학습 및 증강 보조 자료 (`aug_weights_exp16.csv`, `snu_clip_features.csv`, `holdout_300.csv`)
* `outputs_evidence/`: 보고서 서술 및 수치 재현 근거 원장 CSV
* `scripts/`: 실험 및 정밀 분석 인프라 스크립트
* `eda/`: 탐색적 데이터 분석(EDA) 아카이브
* `kaggle/`: Kaggle 세션 실행 원본 파일
* `docs/`: 프로젝트 보고서 및 팀 설계 문서

---

## 7. 보고서 수치 근거 (Evidence Files)
`outputs_evidence/` 디렉터리 내 수치 검증 근거 파일:
* `experiments.csv`: 전 실험 이력 및 리더보드/Holdout 측정 원장
* `submission_diff_by_type.csv`: 977 vs 1488 step 및 유형별 분석 수치
* `all_untrained_items_raw.csv`: 미학습 피처 및 타입 분류 결과
