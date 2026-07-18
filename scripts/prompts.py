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
    # ---- v6/v7: CLIP 유사쌍 힌트 트랙 (7/17 설계 합의, VISION 트랙 2) --------------------
    # {hint}는 structure_features.hint_text()가 채움 — 관측 사실(유사쌍)만.
    # 체인(정밀도 34.5%)·전환횟수(미검증)는 기각, 유사쌍은 인접 정밀도 82% 실측.
    # ⚠️ 학습 시 증강 변형마다 쌍 번호가 제시 순서로 재매핑됨 (train.py/train_cot.py가 처리).
    # v6: 직답 + 힌트 — v1_list와 {hint} 한 줄 차이 (미니 효과 = 순수 힌트 효과)
    "v6_hint": (
        'Thinking about the sentence: "{sentence}"\n'
        "{hint}"
        "Look at the 4 images above labeled Image 1 to Image 4. "
        "Determine the correct chronological order of these images to match the sentence. "
        "Provide the answer ONLY as a Python list of integers. "
        "Example: [1, 2, 3, 4]"
    ),
    # v7: CoT (타깃 = gemma events 기계생성, train_cot.py --events-from gemma)
    # 섹션 구성이 타깃과 정확히 일치 (exp12의 프롬프트-타깃 불일치 제거).
    # ⚠️ 평가 --max-new-tokens 512 필수
    "v7_cot": (
        "[Role]: You are an expert video-language understanding AI.\n"
        "[Task]: Your core function is to reconstruct the correct chronological order "
        "of 4 shuffled video frames to accurately match the given storyline.\n\n"
        'Sentence: "{sentence}"\n\n'
        "Look at the 4 images above labeled Image 1 to Image 4.\n"
        "Execute your task by following these exact steps:\n\n"
        "1. [Story Analysis]: List the distinct events of the sentence in narrated order.\n"
        "2. [Chronological Mapping]: Assign each position in time to an image. "
        "Apply this logic internally and output ONLY the mapping result without "
        "explaining the reasons.\n"
        "3. [Final Answer]: Give the final list.\n\n"
        "You MUST enclose your final python list of integers within <ANSWER> tags.\n\n"
        "Output format:\n"
        "[Story Analysis]\n- Event 1: ...\n"
        "[Chronological Mapping]\n- 1st: Image 3\n- 2nd: Image 1\n- 3rd: Image 4\n- 4th: Image 2\n"
        "[Final Answer]\n<ANSWER>[3, 1, 4, 2]</ANSWER>"
    ),
    # v7 + 힌트: 사용자 제안 템플릿(7/17) + Mapping Rule 연성판 (MUST -> strong guidance,
    # 문장 우선 조항 추가 — 힌트 정밀도 77~82%라 경성 규칙은 오답 강제 위험)
    "v7_cot_hint": (
        "[Role]: You are an expert video-language understanding AI.\n"
        "[Task]: Your core function is to reconstruct the correct chronological order "
        "of 4 shuffled video frames to accurately match the given storyline.\n\n"
        'Sentence: "{sentence}"\n'
        "{hint}\n"
        "Look at the 4 images above labeled Image 1 to Image 4.\n"
        "Execute your task by following these exact steps:\n\n"
        "1. [Story Analysis]: List the distinct events of the sentence in narrated order.\n"
        "2. [Chronological Mapping]: Assign each position in time to an image.\n"
        "   * Mapping Rule: Use the visual note above as strong guidance — images noted "
        "as similar usually belong to continuous or adjacent moments in time, and "
        "clearly different images usually belong to different stages of the story. "
        "If the sentence clearly contradicts the note, trust the sentence. "
        "Apply this logic internally and output ONLY the mapping result without "
        "explaining the reasons.\n"
        "3. [Final Answer]: Give the final list.\n\n"
        "You MUST enclose your final python list of integers within <ANSWER> tags.\n\n"
        "Output format:\n"
        "[Story Analysis]\n- Event 1: ...\n"
        "[Chronological Mapping]\n- 1st: Image 3\n- 2nd: Image 1\n- 3rd: Image 4\n- 4th: Image 2\n"
        "[Final Answer]\n<ANSWER>[3, 1, 4, 2]</ANSWER>"
    ),
}


def build_user_text(prompt_name, sentence, hint=""):
    return PROMPTS[prompt_name].format(sentence=sentence, hint=hint)


def needs_hint(prompt_name):
    """이 프롬프트가 샘플별 힌트 주입을 요구하는가 (train/eval이 CLIP 유사쌍 로드 필요)."""
    return "{hint}" in PROMPTS[prompt_name]
