# -*- coding: utf-8 -*-
"""OWL-ViT 궤적 피처 추출 — 누수 없는 재작성판 (7/20).

원본(extract_owlvit_features.py)과의 차이:
- **좌표를 항상 제시 순서(Input_1~4)로 저장** — 원본은 train만 Answer 시간순으로
  재배열해 저장(정답 누수). 증강 변형·시간순 해석은 힌트 생성 단계에서 perm으로 처리한다.
- 이어받기: JSONL append, 기존 Id 스킵 (절전·중단 대응)
- 로컬 가중치 필수 (`models/owlvit-base-patch32`) — 검증 환경(인터넷 차단) 호환
- 쿼리 선택 결정적: gemma subjects 1순위 (set() 비결정 순서 버그 제거)
- CPU 전용 기본 (GPU는 학습이 점유 중)

사용:
    python scripts/extract_owlvit_clean.py --mini          # 미니 풀 1000 + holdout 300
    python scripts/extract_owlvit_clean.py --split train   # train 전량
    python scripts/extract_owlvit_clean.py --split test    # test 819
출력: outputs/owlvit_clean/{split}.jsonl  (Id, query, 프레임별 status/x/y/area — 제시 순서)
"""
import argparse
import json
import os
import re
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

OUT_DIR = "./outputs/owlvit_clean"
MODEL_DIR = "./models/owlvit-base-patch32"

STOPWORDS = {
    "the", "a", "an", "and", "in", "on", "at", "with", "is", "are", "of", "to",
    "holding", "sits", "gets", "using", "front", "behind", "next",
    "standing", "sitting", "walking", "camera", "scene", "video", "frame",
}
# 대명사·2인칭 등 OWL-ViT 쿼리로 무의미한 토큰 (7/21 재추출: "he/she/they"가
# 쿼리 상위권 177건 — gemma subjects가 원문 대명사를 그대로 넘긴 사인 확인)
PRONOUNS = {"he", "she", "they", "it", "him", "her", "them", "his", "its", "their",
           "i", "you", "we", "who", "someone", "something"}
# 화면 전체를 덮어 bbox 신뢰도가 낮게 나오는 범용어 — 후보에서 배제하지 않고
# "최후 후보"로 순위만 맨 뒤로 미룬다 (아예 빼면 폴백이 없어질 수 있음)
GENERIC_LOW_PRIORITY = {"person", "man", "woman", "people", "group", "child",
                        "hand", "hands", "he", "she"}


def _clean(s):
    s = str(s).strip().lower()
    s = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", s)
    s = re.sub(r"[^a-z ]", "", s).strip()
    return s


_NLP = None  # lazy load — 명사 후보 추출 전용 (품사 태깅으로 동사·전치사류 배제)


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


def _sentence_nouns(sentence):
    """spacy POS로 NOUN/PROPN만 추출 — 정규식 방식은 동사·전치사(crawls/away/from)를
    걸러내지 못해 폐기 (7/21 재추출 시도 중 발견)."""
    doc = _get_nlp()(sentence)
    return [t.lower_ for t in doc if t.pos_ in ("NOUN", "PROPN") and t.lower_ not in STOPWORDS]


def build_candidates(sentence, subjects, max_n=4):
    """구체적 쿼리를 우선하고 범용어를 뒤로 미루는 후보 리스트 (중복 제거, 대명사 배제).

    우선순위: subject 중 구체어 > 문장 명사(POS 필터) > subject 중 범용어(person 등).
    문장 fallback은 subject 구체어가 부족할 때만 채우는 역할 — 우선순위가 항상 밀림.
    최종 채택은 이 리스트를 OWL-ViT에 실제로 돌려 신뢰도 비교로 결정한다
    (§2-A 설계 의도 — 이전 버전은 이 비교 없이 1순위를 맹목 채택했었음)."""
    specific, generic = [], []
    for s in subjects or []:
        s = _clean(s)
        if not s or s in PRONOUNS or len(s) < 2:
            continue
        s = s.split()[-1] if " " in s else s  # "yellow slide" -> "slide"
        (generic if s in GENERIC_LOW_PRIORITY else specific).append(s)

    for w in _sentence_nouns(sentence):
        if w in PRONOUNS or len(w) < 2:
            continue
        (generic if w in GENERIC_LOW_PRIORITY else specific).append(w)

    seen, ordered = set(), []
    for w in specific + generic:  # 구체어 먼저, 범용어는 뒤로 밀려 최후 후보가 됨
        if w not in seen:
            seen.add(w)
            ordered.append(w)
    return ordered[:max_n] or ["object"]


def load_done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["Id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--mini", action="store_true",
                    help="train 중 미니 풀(시드42 1000, train.py와 동일 선정) + holdout 300만")
    ap.add_argument("--threshold", type=float, default=0.08)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    assert os.path.exists(MODEL_DIR), f"{MODEL_DIR} 없음 — scripts/download_owlvit.sh 먼저"
    os.makedirs(OUT_DIR, exist_ok=True)

    split = "train" if args.mini else args.split
    df = pd.read_csv(f"./snuaichallenge_data/{split}.csv")

    if args.mini:
        import sys
        sys.path.insert(0, "scripts")
        from train import load_excluded_ids
        excluded = load_excluded_ids()
        pool = df[~df["Id"].isin(excluded)].reset_index(drop=True)
        mini = pool.sample(n=1000, random_state=42)
        hold = df[df["Id"].isin(set(pd.read_csv("./splits/holdout_300.csv")["Id"]))]
        df = pd.concat([mini, hold]).drop_duplicates("Id").reset_index(drop=True)
        out_path = os.path.join(OUT_DIR, "train.jsonl")  # train 전량 확장 시 이어서 사용
    else:
        out_path = os.path.join(OUT_DIR, f"{split}.jsonl")

    # gemma subjects (쿼리 소스)
    subjects_by_id = {}
    import sys
    sys.path.insert(0, "scripts")
    from structure_features import load_gemma_labels
    g = load_gemma_labels()
    subjects_by_id = {r.Id: list(r.subjects) for r in g.itertuples()}

    done = load_done_ids(out_path)
    todo = df[~df["Id"].isin(done)]
    print(f"{split}: 총 {len(df)} | 완료 {len(done)} | 남음 {len(todo)}", flush=True)
    if not len(todo):
        return

    processor = OwlViTProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_DIR, local_files_only=True).to(args.device)
    model.eval()

    img_root = os.path.join("./snuaichallenge_data", split)
    t0 = time.time()
    n_done = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for _, row in tqdm(todo.iterrows(), total=len(todo)):
            sid = str(row["Id"])
            files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]  # 제시 순서 고정
            paths = [os.path.join(img_root, sid, f) for f in files]
            candidates = build_candidates(row["Sentence"], subjects_by_id.get(sid))

            rec = {"Id": sid, "candidates": candidates, "query": None, "frames": []}
            if not all(os.path.exists(p) for p in paths):
                rec["frames"] = [{"status": "file_not_found"}] * 4
            else:
                images = [Image.open(p).convert("RGB") for p in paths]
                # 후보 전체를 한 번에 배치 추론 (§2-A: 신뢰도 비교로 최종 쿼리 자동 채택)
                inputs = processor(text=[candidates] * 4, images=images, padding=True,
                                   return_tensors="pt").to(args.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                # 후보별 4프레임 합산 신뢰도(각 프레임 최고점의 합) -> 최고 후보 채택
                per_cand_conf = [0.0] * len(candidates)
                per_frame_scores = []  # [(scores_tensor, boxes_tensor)] x4
                for k in range(4):
                    scores = torch.sigmoid(outputs.logits[k])  # (num_boxes, num_candidates)
                    per_frame_scores.append((scores, outputs.pred_boxes[k]))
                    for c in range(len(candidates)):
                        per_cand_conf[c] += float(scores[:, c].max()) if len(scores) else 0.0
                best_c = max(range(len(candidates)), key=lambda c: per_cand_conf[c])
                rec["query"] = candidates[best_c]
                rec["candidate_conf"] = [round(x, 3) for x in per_cand_conf]

                for k in range(4):
                    scores, boxes_all = per_frame_scores[k]
                    col = scores[:, best_c]
                    keep = col >= args.threshold
                    if not keep.any():
                        rec["frames"].append({"status": "no_detection"})
                        continue
                    boxes = boxes_all[keep].cpu().numpy()  # cx,cy,w,h (0~1)
                    areas = boxes[:, 2] * boxes[:, 3]
                    b = boxes[areas.argmax()]
                    rec["frames"].append({
                        "status": "ok",
                        "x": round(float(b[0]), 3), "y": round(float(b[1]), 3),
                        "area": round(float(areas.max()), 4),
                    })
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n_done += 1
            if n_done % 50 == 0:
                rate = (time.time() - t0) / n_done
                print(f"  {n_done}/{len(todo)} | {rate:.1f}초/샘플 | "
                      f"ETA {rate * (len(todo) - n_done) / 60:.0f}분", flush=True)

    print(f"완료: {out_path} (+{n_done})", flush=True)


if __name__ == "__main__":
    main()
