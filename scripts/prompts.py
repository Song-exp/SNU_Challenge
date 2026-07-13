# -*- coding: utf-8 -*-
"""프롬프트 레지스트리 — train.py와 eval_zero_shot.py가 공유한다.

규칙:
- 프롬프트 실험은 학습·평가에 **같은 이름**을 써야 유효하다 (양쪽 다 --prompt <이름>)
- 새 후보는 PROMPTS에 추가하고 이름은 바꾸지 않는다 (experiments.csv 기록과 대응되므로)
- {sentence} 자리에 캡션이 들어간다. 이미지 4장 + "Image N" 라벨은 스크립트가 앞에 붙인다
- 출력 형식 자체(리스트 -> 순열 토큰 등)를 바꾸는 실험은 target_text 생성과 파서도
  같이 바꿔야 하므로 여기만으로는 불가 — train.py/eval의 파싱부 수정 필요
"""

PROMPTS = {
    # v1: 베이스라인 원본. zero-shot 비교(2026-07-13)와 exp01 기준점이 이 프롬프트
    "v1_list": (
        'Thinking about the sentence: "{sentence}"\n'
        "Look at the 4 images above labeled Image 1 to Image 4. "
        "Determine the correct chronological order of these images to match the sentence. "
        "Provide the answer ONLY as a Python list of integers. "
        "Example: [1, 2, 3, 4]"
    ),
    # v2: 시간 키워드 주목 유도 (텍스트 EDA 키워드 분석 연계 후보 — 자유롭게 수정)
    "v2_temporal": (
        'Read the sentence carefully, paying attention to temporal cues such as '
        '"first", "then", "after", "next", and "finally": "{sentence}"\n'
        "The 4 images above (Image 1 to Image 4) are video frames in shuffled order. "
        "Match each event in the sentence to one image, then determine the chronological order. "
        "Provide the answer ONLY as a Python list of integers. "
        "Example: [1, 2, 3, 4]"
    ),
}


def build_user_text(prompt_name, sentence):
    return PROMPTS[prompt_name].format(sentence=sentence)
