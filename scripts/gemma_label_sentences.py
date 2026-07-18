# -*- coding: utf-8 -*-
"""문장 라벨 추출기 — gemma4 (WSL ollama :11435)로 문장당 5필드 JSON 라벨 생성.

산출물:
  - outputs/gemma_labels/labels.jsonl        (7/17 이전 분 902건 — 그대로 보존, 읽기 전용)
  - outputs/gemma_labels/parts/part_NNN.jsonl (신규분, 300건 채우면 다음 파트로 롤오버)
한 줄 = 한 문장, 즉시 기록. 재실행 시 이미 성공한 Id는 건너뛰고 이어서 진행 (이어받기 안전).
진행 상황: outputs/gemma_labels/progress.txt (갱신형 한 줄)

대상 순서 (7/17 오후 확장: train 전체):
  1) holdout 300 전량 — 라벨 품질 검증 + 정확도 분석용
  2) train 미니 풀(시드 42, 1000개) — train.py 미니 학습과 같은 시드 (CoT 미니 재료가 먼저 완성)
  3) 나머지 train 전체(시드 43 셔플) — 중간에 끊어도 표본이 무작위라 부분 커버리지도 사용 가능

사용:
    python scripts/gemma_label_sentences.py                # 전체 (holdout 300 + train 전량)
    python scripts/gemma_label_sentences.py --limit 300    # 이번 실행은 300건만
    python scripts/gemma_label_sentences.py --train-count 1000   # train은 미니 풀까지만
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from train import load_excluded_ids

OLLAMA_URL = "http://localhost:11435/api/generate"
OUT_DIR = "./outputs/gemma_labels"
OUT_PATH = os.path.join(OUT_DIR, "labels.jsonl")   # 7/17 이전 분 (보존)
PARTS_DIR = os.path.join(OUT_DIR, "parts")          # 신규분 300건 단위
PART_SIZE = 300
PROGRESS_PATH = os.path.join(OUT_DIR, "progress.txt")

FIELDS = ["subjects", "events", "camera_language", "viewer_language", "temporal_markers"]

PROMPT = '''You are an expert NLP and computer vision data annotator specializing in video caption analysis.
Your task is to analyze a caption that describes a sequence of video frames in narrative order and extract specific narrative and structural elements into a single JSON object.

Sentence: "{sentence}"

Extract the following as a single JSON object with the exact keys below:
{{
  "subjects": [array of distinct actors or entities that perform actions, as short noun phrases. Include the camera only if camera_language is true],
  "events": [array of distinct actions/events in the chronological order they are narrated, as short self-contained phrases. Do not split one action into fragments],
  "camera_language": boolean (true ONLY for filming/editing references like 'camera', 'pans', 'zooms', 'cuts to', 'scene shifts', 'close-up', 'angle', 'framing' — NOT physical actions like 'cuts through water'),
  "viewer_language": boolean (true for viewer-perspective phrasing like 'we see', 'is shown', 'the video shows', 'revealing'),
  "temporal_markers": [array of explicit order words appearing in the sentence, e.g., 'then', 'after', 'finally', 'begins', 'as', 'before']
}}

Rules:
1. Accuracy: Base your extraction strictly on the provided sentence. Do not hallucinate entities or events that are not explicitly mentioned.
2. Event Independence: Ensure each event phrase is grammatically self-contained including the subject (e.g., "a man chops onions", not just "chops onions").
3. Empty Fields: If no temporal markers, subjects, or events are found, output an empty array [].
4. Output Format: Output ONLY the raw valid JSON object. Do not include any explanations, and do not wrap the output in markdown blocks (e.g., no ```json ... ```).

Examples:

Sentence: "A man chops onions, then the camera zooms in as we see him crying."
Output:
{{
  "subjects": ["a man", "the camera"],
  "events": ["a man chops onions", "the camera zooms in", "the man is crying"],
  "camera_language": true,
  "viewer_language": true,
  "temporal_markers": ["then", "as"]
}}

Sentence: "The dog runs across the yard and catches the frisbee."
Output:
{{
  "subjects": ["the dog"],
  "events": ["the dog runs across the yard", "the dog catches the frisbee"],
  "camera_language": false,
  "viewer_language": false,
  "temporal_markers": []
}}'''


def ask_gemma(sentence, model, retries=2):
    """1문장 추출. (라벨 dict 또는 None, 원문 응답, 소요 초)"""
    body = json.dumps({
        "model": model, "prompt": PROMPT.format(sentence=sentence.strip()),
        "stream": False, "think": False, "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 400},
    }).encode()
    last_err = ""
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=300).read())["response"]
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = f"연결/타임아웃: {e}"
            time.sleep(30 * (attempt + 1))  # 서버 재기동 대기
            continue
        dt = time.time() - t0
        m = re.search(r"\{.*\}", resp, re.S)
        if m:
            try:
                j = json.loads(m.group(0))
                if all(k in j for k in FIELDS):
                    return {k: j[k] for k in FIELDS}, resp, dt
            except json.JSONDecodeError:
                pass
        last_err = f"파싱 실패: {resp.strip()[:100]}"
    return None, last_err, 0.0


def build_targets_test():
    """test 819 전량 — 추론 전처리·제출 후 분석 전용.
    ⚠️ 규정(PROJECT_SETUP §4.3): 이 라벨을 학습 설계(증강 가중·실험 선정)에 쓰면 실격."""
    import pandas as pd
    test_df = pd.read_csv("./snuaichallenge_data/test.csv")
    return [("test", r["Id"], r["Sentence"]) for _, r in test_df.iterrows()]


def derive_test_outputs(parts_dir):
    """test 라벨 -> 파생 2종: 분류용 test_features.csv + 추론용 test_hints.csv."""
    import glob as g
    import pandas as pd
    rows = []
    for path in sorted(g.glob(os.path.join(parts_dir, "part_*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ok"):
                    rows.append(r)
    if not rows:
        print("파생 생성 건너뜀: 성공 라벨 없음", flush=True)
        return
    df = pd.DataFrame(rows).drop_duplicates("Id", keep="last")
    feat = pd.DataFrame({
        "Id": df["Id"],
        "camera": df["camera_language"].astype(bool),
        "viewer": df["viewer_language"].astype(bool),
        "n_events": df["events"].map(len),
        "n_subj": df["subjects"].map(len),
        "n_markers": df["temporal_markers"].map(len),
    })
    feat_path = os.path.join(OUT_DIR, "test_features.csv")
    feat.to_csv(feat_path, index=False)
    hints = pd.DataFrame({
        "Id": df["Id"],
        "events": df["events"].map(lambda e: " -> ".join(map(str, e))),
        "subjects": df["subjects"].map(lambda s: ", ".join(map(str, s))),
        "markers": df["temporal_markers"].map(lambda m: ", ".join(map(str, m))),
        "camera": df["camera_language"].astype(bool),
        "viewer": df["viewer_language"].astype(bool),
    })
    hints_path = os.path.join(OUT_DIR, "test_hints.csv")
    hints.to_csv(hints_path, index=False)
    print(f"파생 생성: {feat_path} (분류용) + {hints_path} (추론용) — {len(df)}행", flush=True)


def build_targets(train_count):
    """(split, Id, sentence) 목록 — holdout 300 + 미니 풀 1000(시드42) + 나머지 train(시드43 셔플).

    미니 풀을 항상 맨 앞에 둬서 기존 라벨(902건)과의 순서 일관성을 보존한다.
    train_count: 양수면 train을 앞에서 N개까지만, 0이면 전체.
    """
    import pandas as pd
    targets = []
    hold = pd.read_csv("./splits/holdout_300.csv")
    for _, r in hold.iterrows():
        targets.append(("holdout", r["Id"], r["Sentence"]))
    train_df = pd.read_csv("./snuaichallenge_data/train.csv")
    train_df = train_df[~train_df["Id"].isin(load_excluded_ids())].reset_index(drop=True)
    pool = train_df.sample(n=1000, random_state=42).reset_index(drop=True)  # train.py 미니와 동일 시드
    rest = train_df[~train_df["Id"].isin(pool["Id"])].sample(frac=1, random_state=43).reset_index(drop=True)
    ordered = pd.concat([pool, rest], ignore_index=True)
    if train_count:
        ordered = ordered.head(train_count)
    for _, r in ordered.iterrows():
        targets.append(("train", r["Id"], r["Sentence"]))
    return targets


def load_done_ids(parts_dir=PARTS_DIR, include_legacy=True):
    """labels.jsonl(옵션) + parts/*.jsonl에서 성공한 Id 집합을 회수."""
    import glob
    done = set()
    paths = ([OUT_PATH] if include_legacy and os.path.exists(OUT_PATH) else []) \
        + sorted(glob.glob(os.path.join(parts_dir, "part_*.jsonl")))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("ok"):
                        done.add(row["Id"])
                except json.JSONDecodeError:
                    continue
    return done


def open_part_writer(parts_dir=PARTS_DIR):
    """이어쓸 파트 파일 경로와 현재 줄 수를 반환 — 300줄 차면 다음 파트로."""
    import glob
    os.makedirs(parts_dir, exist_ok=True)
    parts = sorted(glob.glob(os.path.join(parts_dir, "part_*.jsonl")))
    if parts:
        last = parts[-1]
        with open(last, encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        if n_lines < PART_SIZE:
            return last, n_lines
        next_no = int(os.path.basename(last)[5:8]) + 1
    else:
        next_no = 1
    return os.path.join(parts_dir, f"part_{next_no:03d}.jsonl"), 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--train-count", type=int, default=0, help="train 앞 N개까지만 (0=전체)")
    parser.add_argument("--limit", type=int, default=0, help="이번 실행에서 처리할 최대 건수 (0=전부, 스모크용)")
    parser.add_argument("--dataset", default="train", choices=["train", "test"],
                        help="test = test.csv 819 전량 (별도 test_parts/, 추론 전처리 전용 — 학습 설계 사용 금지)")
    parser.add_argument("--derive-only", action="store_true",
                        help="(test 전용) 추출 없이 파생 파일(features/hints CSV)만 재생성")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.dataset == "test":
        parts_dir = os.path.join(OUT_DIR, "test_parts")
        if args.derive_only:
            derive_test_outputs(parts_dir)
            return
        done = load_done_ids(parts_dir, include_legacy=False)
        targets = build_targets_test()
    else:
        parts_dir = PARTS_DIR
        done = load_done_ids()
        targets = build_targets(args.train_count)

    todo = [t for t in targets if t[1] not in done]
    print(f"[{args.dataset}] 전체 {len(targets)} | 완료 {len(targets) - len(todo)} | 남음 {len(todo)}", flush=True)
    if args.limit:
        todo = todo[:args.limit]
        print(f"--limit {args.limit}: 이번 실행은 {len(todo)}건만 처리", flush=True)

    part_path, part_lines = open_part_writer(parts_dir)
    t_start, n_ok, n_fail = time.time(), 0, 0
    for i, (split, sid, sentence) in enumerate(todo):
        labels, raw, dt = ask_gemma(sentence, args.model)
        row = {"Id": sid, "split": split, "sentence": sentence, "ok": labels is not None,
               "sec": round(dt, 1), "model": args.model}
        if labels:
            row.update(labels)
            n_ok += 1
        else:
            row["error"] = raw[:200]
            n_fail += 1
        if part_lines >= PART_SIZE:
            part_path, part_lines = open_part_writer(parts_dir)
        with open(part_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        part_lines += 1
        if part_lines == PART_SIZE:
            print(f"    ** {os.path.basename(part_path)} 300건 완성 **", flush=True)

        elapsed = time.time() - t_start
        rate = elapsed / (i + 1)
        eta = time.strftime("%H:%M", time.localtime(time.time() + rate * (len(todo) - i - 1)))
        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {i + 1}/{len(todo)} (성공 {n_ok}, 실패 {n_fail}) "
                    f"| {rate:.1f}초/문장 | 완료 예상 {eta}\n")
        status = "OK  " if labels else "FAIL"
        print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(todo)} {status} {split}/{sid} "
              f"{dt:.1f}s | 평균 {rate:.1f}초/문장 | ETA {eta}", flush=True)
        print(f"    원문: {sentence.strip()}", flush=True)
        if labels:
            subj = ", ".join(map(str, labels["subjects"])) or "-"
            mk = ", ".join(map(str, labels["temporal_markers"])) or "-"
            ev = " -> ".join(map(str, labels["events"])) or "-"
            print(f"    cam={'O' if labels['camera_language'] else 'X'} "
                  f"view={'O' if labels['viewer_language'] else 'X'} | 주체: {subj} | 표지: {mk}", flush=True)
            print(f"    events({len(labels['events'])}): {ev}", flush=True)
        else:
            print(f"    └ {raw[:120]}", flush=True)

    print(f"종료: 성공 {n_ok}, 실패 {n_fail}, {(time.time() - t_start) / 3600:.1f}시간 "
          f"-> {parts_dir}", flush=True)
    if args.dataset == "test":
        derive_test_outputs(parts_dir)


if __name__ == "__main__":
    main()
