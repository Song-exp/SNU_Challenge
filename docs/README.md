# 문서 인덱스 (본선 보고서용 정리)

root에 흩어져 있던 md 문서를 **보고서 파트별**로 정리한 폴더입니다.
최종 파이프라인: **Qwen3-VL 4B/8B + LoRA + v5_reorder + 타깃증강(sparse_camX) + 무작위셔플 → 추론 우도 K4 TTA**

> ⚠️ `eda/`, `kaggle/`, `outputs/`, `검수기프로그램/` 하위의 md는 해당 코드·데이터와 붙어 있어 이동하지 않았습니다.

---

## 00_overview — 보고서 백본
| 문서 | 역할 |
|---|---|
| FINAL_MODEL_EDA_RATIONALE.md | 최종 모델 설정 ↔ EDA/실험 근거 매핑 (목차 뼈대) |
| VISION_final_modeling.md | "구조 인지형 순서 예측" 큰 그림 (방법론 개요) |
| PROJECT_SETUP.md | 과제 정의·규정·평가지표 (배경 절) |
| project_plan.md | 초기 프로젝트 추진 계획 |

## 01_data_preprocessing — 데이터 전처리
| 문서 | 내용 | 상태 |
|---|---|---|
| REPORT_gemma_labels.md | gemma4 문장 구조 라벨링 (train 9,535 전량) | 채택 |
| REPORT_typing_criteria.md | 문장 4유형 기준 (sparse/dense × camO/X), holdout 실측 | 채택 |
| eda_문장모호성.md | 문장 모호성 EDA (3단계 파티션 + 직교 플래그) | 채택 |
| pipeline_data_mapping.md | 전처리 설계 ↔ 스크립트/데이터 매핑 | 채택 |
| 문법성분_추출_보고서.md | SpaCy 경량 주어/서술어 추출 (발표용) | 채택 |
| sentence_ambiguity_literature_review.md | 모호성 분류 학술 타당성 검토 | 보조 |
| eda_신규유형발굴.md | 선행연구 기반 신규 유형 발굴 | 보조 |
| notion_ambiguity_vlm_strategy.md | 모호성 정량화 + VLM 추론 전략 | 보조 |

### _rejected_image_physics — 시도 후 기각 (이미지 물리분석 계열)
보고서에서는 "탐색했으나 실효 없어 제외"로 서술 (엄밀성 근거).
| 문서 | 기각 사유 |
|---|---|
| vision_advanced_methodology.md | Depth/YOLO 궤적 — 힌트 무효(v10), OWL-ViT 검출률 낮음 |
| IMAGE_INTEGRATION_PLAN.md | 이미지-문장 물리 인과 연동 계획 (동일 계열) |
| eda_image_generalization_report.md / _v2.md | 물리 힌트 일반화 검증 → 미채택 |
| 일반화_물리법칙_검증_보완보고서.md | 물리 법칙 정합성 보완 검증 → 미채택 |

## 02_finetuning — 파인튜닝
| 문서 | 내용 | 상태 |
|---|---|---|
| EXPERIMENTS.md | 실험 원칙·세팅·로드맵 (holdout 고정, acc_shuffled) | 채택(코어) |
| PLAN_post_labeling.md | 타깃 증강 트랙 (camera/n_events 약점 증강) | 채택 |
| CHANGELOG_0720-0721.md | 4B 스케일업 대약진 (exp16→exp17, +7.3%p) | 채택 |
| LOCAL_FINAL_SPEC.md | exp17 최종 어댑터 재현 스펙 + 우도 K4 | 채택 |
| PLAN_cot_finetune.md | CoT SFT 설계 — 4연패로 종료 | 기각 |

> **우도 K4 TTA**(holdout +4.76%p)는 학습이 아닌 **추론 최적화** 기법.
> 보고서에서는 파인튜닝 뒤 별도 "추론 전략" 소절로 분리 권장 (규정 §4.3 TTA 허용).
> 구현: `scripts/score_permutations.py`, 스펙: LOCAL_FINAL_SPEC.md

## 03_prompt_engineering — 프롬프트 엔지니어링
| 문서 | 내용 | 상태 |
|---|---|---|
| PLAN_prompt_experiments.md | 프롬프트 추론 실험 상세 (민감도·라우팅, v5_reorder 확정) | 채택(v5) |
| PLAN_prompt_and_preprocessing.md | 전처리 검토 + 프롬프트 실험 계획 (교차) | 채택 |

## 99_operational — 운영/현황 (보고서 파트 아님)
| 문서 | 내용 |
|---|---|
| HANDOVER.md, HANDOVER_0723_EVENING.md, HANDOVER_KAGGLE_LOCAL.md | 인수인계 이력 |
| RESUME.md | 마감 직전 즉시 인수인계 |
| notion_summary.md | 팀원 포트폴리오 요약 (외부 공유용) |
| 클로드_질문용_컨텍스트.md | 자문용 종합 컨텍스트 |

---

## ⚠️ 정리 시 참고
- 이동 후 문서 간 **상대경로 상호참조**(예: "관련 문서: pipeline_data_mapping.md")는 폴더가 달라져 깨질 수 있음 — 보고서 인용 시 내용만 발췌.
- 일부 팀원 문서에 다른 사용자 경로(`C:/Users/bella/...`)가 박혀 있음 — 그대로 넣지 말 것.
