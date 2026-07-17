# -*- coding: utf-8 -*-
"""구조 피처 단일 진실 (VISION_final_modeling.md 1단계의 데이터 층).

세 소스를 하나의 피처 테이블로 통합:
  1) 정규식 축 — camera (holdout 실측 +26.9%p, gemma와 일치율 92.1%) : 전량 커버, 라벨 불요
  2) gemma 라벨 — outputs/gemma_labels/labels.jsonl + parts/*.jsonl (5필드 + 파생 수치)
  3) CLIP 피처 — snu_clip_features.csv 형식 (팀원 산출물, Id + predicted_scene_cuts + dist_*)

사용처: Structure_Pipeline.ipynb (분석·증강 가중 생성), make_hint_data.py (힌트 학습셋)
"""
import glob
import json
import os
import re

# 카메라/촬영 표현 정규식 — 7/17 검증본 (scratchpad gemma_vs_regex.py에서 이식)
# "cuts through water" 같은 물리 동작 오탐은 gemma 대비 7.9%의 불일치로 남아 있음 — 전량 커버가 우선
CAM_RE = re.compile(
    r"\b(camera|pans?|panning|zooms?|zooming|cuts?|cutting|the (scene|view|shot|frame|screen)"
    r"|view (shifts?|changes?)|close-?up|focus(es)? on|angle|footage|transitions?|frame)\b", re.I)

GEMMA_DIR = "./outputs/gemma_labels"
CLIP_TRAIN_PATH = "./snu_clip_features.csv"        # 팀원 산출물 (train 9,535행)
CLIP_TEST_PATH = "./snu_clip_features_test.csv"    # 팀원 산출물 예정 (test 819행, 같은 스키마)
CLIP_REQUIRED_COLS = ["Id", "predicted_scene_cuts",
                      "dist_12", "dist_13", "dist_14", "dist_23", "dist_24", "dist_34"]


def camera_regex(sentence):
    """문장에 카메라/촬영 표현이 있는가 (정규식 축)."""
    return bool(CAM_RE.search(str(sentence)))


def load_gemma_labels():
    """labels.jsonl + parts/*.jsonl 병합 -> 성공 라벨 DataFrame (Id 중복은 마지막 성공 우선).

    열: Id, split, sentence, camera, viewer, n_events, n_subj, n_markers,
        events(list), subjects(list), markers(list)
    """
    import pandas as pd
    paths = []
    legacy = os.path.join(GEMMA_DIR, "labels.jsonl")
    if os.path.exists(legacy):
        paths.append(legacy)
    paths += sorted(glob.glob(os.path.join(GEMMA_DIR, "parts", "part_*.jsonl")))

    rows = {}
    n_fail = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("ok"):
                    n_fail += 1
                    continue
                rows[r["Id"]] = {
                    "Id": r["Id"], "split": r["split"], "sentence": r["sentence"],
                    "camera": bool(r["camera_language"]), "viewer": bool(r["viewer_language"]),
                    "n_events": len(r["events"]), "n_subj": len(r["subjects"]),
                    "n_markers": len(r["temporal_markers"]),
                    "events": r["events"], "subjects": r["subjects"],
                    "markers": r["temporal_markers"],
                }
    df = pd.DataFrame(list(rows.values()))
    print(f"gemma 라벨: {len(df)}개 (holdout {int((df.split == 'holdout').sum()) if len(df) else 0} / "
          f"train {int((df.split == 'train').sum()) if len(df) else 0}) | 실패 기록 {n_fail}건은 제외", flush=True)
    return df


def load_clip_features(path=CLIP_TRAIN_PATH):
    """CLIP 피처 로드 + 스키마 검증 (팀원 산출물 수령 인터페이스)."""
    import pandas as pd
    if not os.path.exists(path):
        raise FileNotFoundError(f"CLIP 피처 없음: {path} — 팀원 산출물 대기 중이면 이 셀은 건너뛸 것")
    df = pd.read_csv(path)
    missing = [c for c in CLIP_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CLIP 피처 스키마 불일치: {missing} 열 없음 (기대: {CLIP_REQUIRED_COLS})")
    if df["Id"].duplicated().any():
        raise ValueError("CLIP 피처에 중복 Id 존재")
    print(f"CLIP 피처: {path} ({len(df)}행)", flush=True)
    return df


def build_feature_table(sentences_df):
    """(Id, Sentence) DataFrame -> 피처 테이블.

    camera_re는 전 행 정규식으로 채우고, gemma 라벨이 있는 행은 gemma 축(camera/viewer/개수)을 병합.
    has_gemma로 소스 구분 — 분석 시 gemma 축은 has_gemma 행에서만 사용할 것.
    """
    import pandas as pd
    out = sentences_df[["Id", "Sentence"]].copy()
    out["camera_re"] = out["Sentence"].map(camera_regex)
    gemma = load_gemma_labels()
    if len(gemma):
        cols = ["Id", "camera", "viewer", "n_events", "n_subj", "n_markers"]
        out = out.merge(gemma[cols], on="Id", how="left")
        out["has_gemma"] = out["camera"].notna()
    else:
        out["has_gemma"] = False
    print(f"피처 테이블: {len(out)}행 | camera_re {out.camera_re.mean():.1%} | "
          f"gemma 커버 {out.has_gemma.mean():.1%}", flush=True)
    return out


def make_aug_weights(feature_df, rules, default_mult, out_path):
    """조건 -> 배수 규칙으로 train.py --aug-weights 입력 CSV 생성.

    rules: [(조건함수(row) -> bool, 배수)] — 위에서부터 첫 매치 적용, 미매치는 default_mult.
    반환: 저장 경로. 예 (exp15):
        make_aug_weights(ft, [(lambda r: not r.camera_re, 4)], 2, "./outputs/aug_weights_exp15.csv")
    """
    import pandas as pd
    mults = []
    for _, row in feature_df.iterrows():
        for cond, mult in rules:
            if cond(row):
                mults.append(mult)
                break
        else:
            mults.append(default_mult)
    wdf = pd.DataFrame({"Id": feature_df["Id"], "aug_mult": mults})
    wdf.to_csv(out_path, index=False)
    dist = wdf["aug_mult"].value_counts().sort_index()
    total = int((wdf["aug_mult"]).sum())
    print(f"증강 가중 저장: {out_path} | " + ", ".join(f"x{m}: {n}개" for m, n in dist.items())
          + f" | 총 학습 항목 {total}개 상당", flush=True)
    return out_path
