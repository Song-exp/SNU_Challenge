# 📑 [제안서] CLIP/MSE 하이브리드 수치 및 구문 힌트의 VLM 프롬프트 주입 설계안

본 제안서는 **연속형 수치 힌트(CLIP/MSE) 주입 설계**와 **문법성분(고유 주어/서술어) 및 통사 구조 분류 정보**를 결합하여, VLM(예: Qwen2-VL)의 시각-언어 시간축 정렬 성능을 극대화하기 위한 프롬프트 힌트 설계 사양을 정의합니다.

---

## 🎯 1. 핵심 설계 철학 (Design Philosophy)

> [!IMPORTANT]
> **왜 하드 라벨("장면전환 N회") 대신 소프트 메트릭(수치 Z-Score)을 제공해야 하는가?**
> 1. **오류 파급(Error Cascade) 차단**: 장면 전환 임계치(0.20) 경계면에 있는 애매한 샘플에 대해 기계가 틀린 하드 힌트("0회 전환")를 던져주면 VLM이 오독을 맹신하여 완전히 틀리게 됩니다. 수치(Z-score)를 던져주면 VLM이 유연하게 확률론적 추론을 수행합니다.
> 2. **2차원 공간 변화율 학습**: 의미론적 변화(CLIP)와 물리적 픽셀 변화(MSE)를 동시에 주면, VLM은 "의미는 안 바뀌었는데 픽셀 변화가 크다 = 카메라 움직임 또는 한 대상의 큰 움직임"과 같은 미세한 차이를 스스로 매핑할 수 있게 됩니다.

---

## 🛠️ 2. 프롬프트 주입 정보 구조 (Data Schema)

모델 입력 시 텍스트 프롬프트의 최상단 또는 인스트럭션 바로 위에 아래의 구조화된 힌트 텍스트 블록(Information Block)을 생성하여 주입합니다.

```text
[Grammar & Transition Clues]
- Sentence Structure Type: [Type-1 / Type-2 / Type-3]
- Target Subjects: [추출된 고유 주어 리스트] (Total Count: N)
- Key Action Verbs: [추출된 고유 동사 리스트] (Total Count: M)

[Visual Frame-to-Frame Transition Metrics]
- CLIP Semantic Distance (Z-score Max): {Max_clip_scaled:.2f}
- MSE Physical Pixel Difference (Z-score Max): {Max_mse_scaled:.2f}
```

---

## 📄 3. 3대 통사 구조(Type-1, 2, 3)별 프롬프트 템플릿 제안

VLM의 어텐션(Attention)을 각 유형별 문제 해결 방식에 집중시키기 위해, 분류 유형에 따라 지시문(Instruction)의 형태를 미세하게 다르게 조절합니다.

### 📌 Type-1: 단일 절 구조 (Single-Clause) ➔ "비주얼 물리 변화 집중형"
* **특징**: 문장 내 시간 힌트가 없어 VLM이 전적으로 이미지 변화량에 의존해야 하는 유형입니다.
* **프롬프트 템플릿**:
  ```text
  [Context Clues]
  - Sentence Type: Type-1 (Single-Clause)
  - Target Actors: {Unique_Subject_Words} (Total: {Unique_Subj_Count})
  - Target Actions: {Unique_Predicate_Words} (Total: {Unique_Pred_Count})

  [Visual Transition Metrics]
  - CLIP Semantic Distance (Z-score Max): {Max_clip_scaled:.2f}
  - MSE Pixel Difference (Z-score Max): {Max_mse_scaled:.2f}

  Instruction: The video description has a single-clause structure. There is no explicit temporal sequence in the text. Rely primarily on the provided visual transition metrics (CLIP and MSE Z-scores) to determine how the physical action of the subject progresses, and arrange the 4 shuffled frames in the correct chronological order.
  ```

### 📌 Type-2: 복합 종속 구조 (Complex-Subordinate) ➔ "카메라/보조행동 매핑형"
* **특징**: zooms out, showing... 처럼 주행동과 보조행동, 혹은 카메라 줌이 섞인 구조입니다.
* **프롬프트 템플릿**:
  ```text
  [Context Clues]
  - Sentence Type: Type-2 (Complex-Subordinate)
  - Target Actors: {Unique_Subject_Words} (Total: {Unique_Subj_Count})
  - Target Actions: {Unique_Predicate_Words} (Total: {Unique_Pred_Count})

  [Visual Transition Metrics]
  - CLIP Semantic Distance (Z-score Max): {Max_clip_scaled:.2f}
  - MSE Pixel Difference (Z-score Max): {Max_mse_scaled:.2f}

  Instruction: The video description contains a main clause and a subordinate clause or participle phrase (e.g., camera zoom or secondary actions). Match the timing of these described transitions with the provided CLIP and MSE Z-score metrics to determine which frames represent the main action and which represent the subordinate detail, then sequence the 4 frames chronologically.
  ```

### 📌 Type-3: 대등 병렬 구조 (Parallel-Coordinated) ➔ "동사 어순-타임라인 매칭형"
* **특징**: chops onions and mixes them 처럼 어순과 시간 흐름이 1:1로 정확하게 맞아떨어지는 구조입니다.
* **프롬프트 템플릿**:
  ```text
  [Context Clues]
  - Sentence Type: Type-3 (Parallel-Coordinated)
  - Target Actors: {Unique_Subject_Words} (Total: {Unique_Subj_Count})
  - Target Actions: {Unique_Predicate_Words} (Total: {Unique_Pred_Count})

  [Visual Transition Metrics]
  - CLIP Semantic Distance (Z-score Max): {Max_clip_scaled:.2f}
  - MSE Pixel Difference (Z-score Max): {Max_mse_scaled:.2f}

  Instruction: The video description lists sequential actions connected by 'and' or commas. Map the chronological sequence of the extracted actions ({Unique_Predicate_Words}) to the visual transition scores (CLIP and MSE Z-scores) to order the 4 shuffled frames correctly.
  ```

---

## 💻 4. 파이토치 데이터로더(Dataset) 연동 코드 예시

학습 스크립트의 `Dataset` 클래스 내부에서 문자열을 포맷팅하여 최종 프롬프트로 병합하는 실제 파이썬 구현 예시입니다.

```python
def generate_vlm_prompt(row):
    """
    row: snu_clip_features.csv 와 train_검토_최종_완료_수정본.csv가 머지된 DataFrame의 일부분
    """
    # 1. 널값 예외 처리 및 수치 포맷팅
    max_clip_z = row['Max_clip_scaled'] if not pd.isna(row['Max_clip_scaled']) else 0.0
    max_mse_z = row['Max_mse_scaled'] if not pd.isna(row['Max_mse_scaled']) else 0.0
    
    # 2. 문법 및 문장 성분 정보 로드 (수정본 우선 채택)
    partition = row['수정된 Partition'] if not pd.isna(row['수정된 Partition']) else row['Partition']
    subj_words = row['고유 주어'] if not pd.isna(row['고유 주어']) else "[unspecified subject]"
    pred_words = row['서술어'] if not pd.isna(row['서술어']) else ""
    subj_count = int(row['수정된 고유 주어 개수']) if not pd.isna(row['수정된 고유 주어 개수']) else int(row['고유 주어 개수'])
    pred_count = int(row['수정된 서술어 개수']) if not pd.isna(row['수정된 서술어 개수']) else int(row['서술어 개수'])
    
    # 3. 유형별 동적 인스트럭션 생성
    if partition == "Type-1":
        instruction = (
            "The video description has a single-clause structure. There is no explicit temporal sequence in the text. "
            "Rely primarily on the provided visual transition metrics (CLIP and MSE Z-scores) to determine how the physical "
            "action of the subject progresses, and arrange the 4 shuffled frames in the correct chronological order."
        )
    elif partition == "Type-2":
        instruction = (
            "The video description contains a main clause and a subordinate clause or participle phrase (e.g., camera zoom or secondary actions). "
            "Match the timing of these described transitions with the provided CLIP and MSE Z-score metrics to determine "
            "which frames represent the main action and which represent the subordinate detail, then sequence the 4 frames chronologically."
        )
    else:  # Type-3
        instruction = (
            f"The video description lists sequential actions connected by 'and' or commas. Map the chronological sequence of "
            f"the extracted actions ({pred_words}) to the visual transition scores (CLIP and MSE Z-scores) to order the 4 shuffled frames correctly."
        )
        
    # 4. 최종 통합 텍스트 템플릿 조립
    prompt = (
        f"[Context Clues]\n"
        f"- Sentence Type: {partition}\n"
        f"- Target Actors: {subj_words} (Total: {subj_count})\n"
        f"- Target Actions: {pred_words} (Total: {pred_count})\n\n"
        f"[Visual Transition Metrics]\n"
        f"- CLIP Semantic Distance (Z-score Max): {max_clip_z:.3f}\n"
        f"- MSE Pixel Difference (Z-score Max): {max_mse_z:.3f}\n\n"
        f"Description: {row['Sentence']}\n"
        f"Instruction: {instruction}"
    )
    return prompt
```
