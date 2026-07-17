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
    # v3: CoT — 묘사→매칭→순서 도출 단계를 거친 뒤 마지막에 리스트 출력.
    # ⚠️ 평가 시 --max-new-tokens 256 필수 (기본 32면 추론 과정에서 잘려 전부 파싱 실패)
    "v3_cot": (
        'Sentence: "{sentence}"\n'
        "Step 1: Briefly describe what happens in each of the 4 images above (Image 1 to Image 4).\n"
        "Step 2: Match each image to a part of the sentence.\n"
        "Step 3: Determine the chronological order of the images.\n"
        "End your answer with ONLY a Python list of integers on the last line. "
        "Example: [1, 2, 3, 4]"
    ),
    # v4: 팀 제안(7/15) 축약판 — 역할 부여 + 스토리라인 프레이밍 + 시각 단서 가이드,
    # 출력은 리스트 즉답(target 변경 불필요 → 미니 학습 스크리닝 바로 가능)
    "v4_story": (
        "You are an expert visual storyteller and video editor. Your task is to "
        "reconstruct the correct chronological order of 4 shuffled video frames "
        "based on a provided storyline.\n"
        'Storyline: "{sentence}"\n'
        "Look at the 4 images above labeled Image 1 to Image 4. Identify key visual "
        "cues in each image (object states, character actions, background changes) "
        "that match the events in the storyline, and determine the correct "
        "chronological order. Provide the answer ONLY as a Python list of integers. "
        "Example: [1, 2, 3, 4]"
    ),
    # v4_cot: 팀 제안 풀버전 — 단계별 출력 + <ANSWER> 태그.
    # ⚠️ 학습에 쓰려면 train.py target_text(단계 응답 생성) + eval 파서(<ANSWER> 우선) 동시 수정 필요.
    # ⚠️ 평가 시 --max-new-tokens 512 필수. exp11 후보 (PLAN_prompt_and_preprocessing.md §2.5 기법②)
    "v4_story_cot": (
        "You are an expert visual storyteller and video editor. Your task is to "
        "reconstruct the correct chronological order of 4 shuffled video frames "
        "based on a provided storyline.\n\n"
        'Storyline: "{sentence}"\n\n'
        "Look at the 4 images provided, labeled Image 1, Image 2, Image 3, and Image 4.\n"
        "Please determine the correct chronological order by following these steps carefully:\n\n"
        "1. [Story Analysis]: Break down the storyline into sequential events or stages.\n"
        "2. [Visual Evidence]: Examine Image 1 to 4. Identify key visual cues in each "
        "image (e.g., object states, character actions, background changes) that match "
        "the events in the storyline.\n"
        "3. [Chronological Mapping]: Map each image to its corresponding stage in the "
        "storyline to determine the correct flow of time.\n"
        "4. [Final Answer]: Provide the final ordered list of image numbers.\n\n"
        "You MUST enclose your final python list of integers within <ANSWER> tags "
        "so it can be parsed programmatically.\n\n"
        "Output Format Example:\n"
        "[Story Analysis]\n- Event 1: ...\n- Event 2: ...\n"
        "[Visual Evidence]\n- Image 1 shows ...\n- Image 2 shows ...\n"
        "[Chronological Mapping]\n- 1st event is Image 3 because ...\n"
        "- 2nd event is Image 1 because ...\n"
        "[Final Answer]\n<ANSWER>[3, 1, 4, 2]</ANSWER>"
    ),
    # v5: 구조 변형 — 문장을 지시 뒤로 (Prompt_Experiments r1_reorder 원문, 7/15 추론 실측 +6%p
    # 유일 양성). 출력은 리스트 즉답이라 target·파서 변경 불필요 → 미니 학습 스크리닝 대상
    "v5_reorder": (
        "Look at the 4 images above labeled Image 1 to Image 4. Determine the correct "
        "chronological order of these images to match the sentence below.\n"
        'Sentence: "{sentence}"\n'
        "Provide the answer ONLY as a Python list of integers. Example: [1, 2, 3, 4]"
    ),
}


def build_user_text(prompt_name, sentence):
    return PROMPTS[prompt_name].format(sentence=sentence)
