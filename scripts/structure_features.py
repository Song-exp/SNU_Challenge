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
# 카메라 계열 주어/사건 판별 (구절 서두 기준 — "moves closer to the camera" 같은 목적어 언급은 제외)
CAM_PHRASE_STARTS = ("the camera", "camera", "the scene", "the view", "the screen",
                     "the shot", "the frame", "the video", "the footage")


def is_cam_phrase(phrase):
    return str(phrase).strip().lower().startswith(CAM_PHRASE_STARTS)
CLIP_TRAIN_PATH = "./snu_clip_features.csv"        # 팀원 산출물 (train 9,535행)
CLIP_TEST_PATH = "./snu_clip_features_test.csv"    # 팀원 산출물 예정 (test 819행, 같은 스키마)
CLIP_REQUIRED_COLS = ["Id", "predicted_scene_cuts",
                      "dist_12", "dist_13", "dist_14", "dist_23", "dist_24", "dist_34"]
PAIR_COLS = {(1, 2): "dist_12", (1, 3): "dist_13", (1, 4): "dist_14",
             (2, 3): "dist_23", (2, 4): "dist_24", (3, 4): "dist_34"}
CLIP_SIM_THRESHOLD = 0.20   # 유사쌍 판정 임계 (팀 non-strict 매핑과 동일 기준)


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
                    # 카메라 주어/사건 제외 카운트 — camO는 gemma가 camera를 주어·사건에
                    # 포함하므로(프롬프트 규칙) 원본 카운트는 camera 축과 교란됨 (7/18 실측)
                    "n_events_noncam": sum(not is_cam_phrase(e) for e in r["events"]),
                    "n_subj_noncam": sum(not is_cam_phrase(s) for s in r["subjects"]),
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


def load_clip_pairs(path=CLIP_TRAIN_PATH, threshold=CLIP_SIM_THRESHOLD):
    """{Id: [(a, b), ...]} — CLIP 거리 < threshold 인 유사쌍 (원본 제시 순서 기준 1-based)."""
    df = load_clip_features(path)
    out = {}
    for row in df.itertuples():
        out[row.Id] = [p for p, col in PAIR_COLS.items() if getattr(row, col) < threshold]
    return out


def remap_pairs(pairs, perm):
    """유사쌍 번호를 증강 변형의 제시 순서로 변환.

    perm: train.py build_training_items의 변형 규약 — perm[j] = 이번 변형에서
    Image j+1로 제시되는 원본 이미지의 0-based 인덱스.
    """
    pos = {orig + 1: j + 1 for j, orig in enumerate(perm)}   # 원본 번호 -> 새 제시 번호
    return sorted(tuple(sorted((pos[a], pos[b]))) for a, b in pairs)


def hint_text(pairs):
    """유사쌍 목록 -> 프롬프트 힌트 한 줄 (개행 포함). 관측 사실만, 순서 주장 없음."""
    if not pairs:
        return "Visual note: all 4 images look clearly different from each other.\n"
    if len(pairs) >= 5:
        return "Visual note: all 4 images look very similar to each other.\n"
    body = " ".join(f"Image {a} and Image {b} look visually similar." for a, b in pairs)
    return f"Visual note: {body}\n"


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
        cols = ["Id", "camera", "viewer", "n_events", "n_subj", "n_markers",
                "n_events_noncam", "n_subj_noncam"]
        out = out.merge(gemma[cols], on="Id", how="left")
        out["has_gemma"] = out["camera"].notna()
    else:
        out["has_gemma"] = False
    if os.path.exists(CLIP_TRAIN_PATH):   # CLIP 축: scene_cuts + 유사쌍 개수
        clip = load_clip_features()
        dcols = list(PAIR_COLS.values())
        clip["n_similar"] = (clip[dcols] < CLIP_SIM_THRESHOLD).sum(axis=1)
        out = out.merge(clip[["Id", "predicted_scene_cuts", "n_similar"]].rename(
            columns={"predicted_scene_cuts": "scene_cuts"}), on="Id", how="left")
    print(f"피처 테이블: {len(out)}행 | camera_re {out.camera_re.mean():.1%} | "
          f"gemma 커버 {out.has_gemma.mean():.1%}"
          + (f" | scene_cuts 커버 {out.scene_cuts.notna().mean():.1%}" if "scene_cuts" in out else ""),
          flush=True)
    return out


def assign_types(feature_df, events_cuts=(2,), density_col="n_events_noncam"):
    """구조 수치 기반 유형 부여 — 1차 유형(서사 밀도 x camera) + 보조 태그.

    7/18 개정: 밀도 축 기본 = **비카메라 사건 수** (camO의 카운트 부풀림 교란 제거,
    "카메라 표현이 빈약 서사를 구제한다" 발견의 근거). 기본 컷 (2,) = sparse(<=2)/dense(>=3)
    2구간 x camera = 4유형 — holdout 분리 21.6/48.7/~53/~74%, 셀 n=39~74.
    events_cuts에 (lo, hi)를 주면 sparse/mid/rich 3구간.
    train/test에 같은 함수를 적용해야 유형 정의가 어긋나지 않는다 (단일 진실).
    """
    import pandas as pd
    df = feature_df.copy()
    if len(events_cuts) == 1:
        bins, labels = [-1, events_cuts[0], 99], ["sparse", "dense"]
    else:
        bins, labels = [-1, events_cuts[0], events_cuts[1], 99], ["sparse", "mid", "rich"]
    density = pd.cut(df[density_col], bins, labels=labels)
    cam = df["camera"].map({True: "camO", False: "camX"})
    df["stype"] = density.astype(str) + "_" + cam
    df.loc[density.isna() | cam.isna(), "stype"] = None
    df["tag_multi_subj"] = df["n_subj_noncam"] >= 2   # 보조 태그 (유형 셀은 쪼개지 않음)
    df["tag_no_marker"] = df["n_markers"] == 0
    df["tag_viewer"] = df["viewer"].fillna(False)
    return df


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


# ---------------------------------------------------------------- OWL-ViT 기능 연동 추가
OWLVIT_TRAIN_PATH = "./snu_owlvit_features_train.csv"
OWLVIT_TEST_PATH = "./snu_owlvit_features_test.csv"

def load_owlvit_features(path=OWLVIT_TRAIN_PATH):
    """OWL-ViT 피처를 로드합니다. 파일이 없으면 None을 리턴하여 부드러운 스킵을 보장합니다."""
    import pandas as pd
    if not os.path.exists(path):
        print(f"[WARNING] OWL-ViT 피처 파일 없음: {path} (가용한 경우에만 힌트가 주입됩니다)", flush=True)
        return None
    df = pd.read_csv(path).set_index("Id")
    print(f"OWL-ViT 피처 로드 완료: {path} ({len(df)}행)", flush=True)
    return df


def build_comprehensive_hints(clip_pairs, owlvit_features, sid, chrono, perm):
    """CLIP 유사쌍 정보와 OWL-ViT 사물 궤적 및 면적 정보를 하나로 병합하여 최종 힌트 텍스트를 만듭니다."""
    hints_lines = []
    
    # 1. CLIP 유사쌍 힌트 추가
    if clip_pairs is not None:
        hints_lines.append(hint_text(remap_pairs(clip_pairs.get(sid, []), perm)))
        
    # 2. OWL-ViT 궤적/크기 힌트 추가
    if owlvit_features is not None and sid in owlvit_features.index:
        row = owlvit_features.loc[sid]
        query = row["query"]
        if str(query).lower() != "none":
            # Map original image number -> chronological step index (1-based)
            orig_to_chrono_step = {orig: step + 1 for step, orig in enumerate(chrono)}
            
            # Shuffled presentation images 1 to 4
            traj_hints = []
            for k in range(4):
                orig_num = perm[k] + 1
                chrono_step = orig_to_chrono_step[orig_num]
                coord_val = row[f"coord_{chrono_step}"]
                traj_hints.append(f"- Image {k+1}: {coord_val}")
                
            hints_lines.append(f"[Visual Object Trajectory Hints]\nTarget query: '{query}'\n" + "\n".join(traj_hints) + "\n")
            
    return "\n".join(hints_lines)

