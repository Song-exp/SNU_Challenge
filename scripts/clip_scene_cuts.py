# -*- coding: utf-8 -*-
"""CLIP 동적 scene_cuts 추론 파이프라인 (폐쇄망 재현 대비, 로컬 로드).

팀원 eda/clip_labeling_model.py의 scene_cuts 로직을 이식:
  4프레임 → CLIP 임베딩 → 6쌍 코사인 거리 → (거리<0.20 유사쌍 개수) → cuts 매핑.
차이: torch.hub 원격 로드 대신 transformers 로컬 CLIP (models/clip-vit-base-patch32).

⚠️ 정합성: 팀원은 openai/CLIP(torch.hub) ViT-B/32, 본 스크립트는 transformers ViT-B/32.
   같은 백본이나 전처리 구현이 미세하게 다를 수 있어, --verify로 train scene_cuts와
   분포·일치율을 반드시 대조한 뒤 사용할 것 (임계값 0.20은 팀원 튜닝값).

사용:
    # 로컬 CLIP 확인 + train 재현으로 정합성 검증
    python scripts/clip_scene_cuts.py --split train --verify
    # test scene_cuts 생성 (제출/추론용)
    python scripts/clip_scene_cuts.py --split test --out snu_clip_features_test.csv
"""
import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

CLIP_DIR = "./models/clip-vit-base-patch32"
CLIP_THRESHOLD = 0.20     # 팀원 튜닝값 — 변경 금지 (train scene_cuts가 이 값 기준)
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PAIR_COLS = ["dist_12", "dist_13", "dist_14", "dist_23", "dist_24", "dist_34"]


def map_similar_pairs_to_cuts(similar_pairs):
    """팀원 로직 그대로: 유사쌍 개수 -> 장면 전환 수(0~3)."""
    if similar_pairs >= 5:
        return 0
    if 2 <= similar_pairs <= 4:
        return 1
    if similar_pairs == 1:
        return 2
    return 3


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    assert os.path.exists(CLIP_DIR), f"{CLIP_DIR} 없음 — scripts/download_clip.sh 먼저"
    model = CLIPModel.from_pretrained(CLIP_DIR, local_files_only=True).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
    return model, proc


def compute(df, image_dir, device):
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    model, proc = load_clip(device)

    # 고유 이미지 임베딩 캐시 (배치)
    paths = []
    for _, r in df.iterrows():
        for f in [r["Input_1"], r["Input_2"], r["Input_3"], r["Input_4"]]:
            paths.append(os.path.join(image_dir, str(r["Id"]), f))
    uniq = sorted(set(paths))
    cache = {}
    bs = 128
    for i in tqdm(range(0, len(uniq), bs), desc="CLIP 임베딩"):
        chunk = uniq[i:i + bs]
        imgs, valid = [], []
        for p in chunk:
            if os.path.exists(p):
                try:
                    imgs.append(Image.open(p).convert("RGB")); valid.append(p)
                except Exception:
                    pass
        if not imgs:
            continue
        inp = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            # get_image_features가 이 transformers 버전에서 projection 전 출력을 반환 →
            # vision_model + visual_projection을 직접 태워 표준 image_embeds(512d) 획득
            vout = model.vision_model(**inp)
            feat = model.visual_projection(vout.pooler_output)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        for p, f in zip(valid, feat.cpu().numpy()):
            cache[p] = f

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="scene_cuts"):
        sid = str(r["Id"])
        fp = [os.path.join(image_dir, sid, r[c]) for c in ["Input_1", "Input_2", "Input_3", "Input_4"]]
        feats = [cache.get(p) for p in fp]
        dists = []
        for a, b in PAIRS:
            if feats[a] is not None and feats[b] is not None:
                dists.append(float(1.0 - np.dot(feats[a], feats[b])))
            else:
                dists.append(0.0)
        sim = sum(1 for d in dists if d < CLIP_THRESHOLD)
        rec = {"Id": sid, "predicted_scene_cuts": map_similar_pairs_to_cuts(sim)}
        rec.update({c: d for c, d in zip(PAIR_COLS, dists)})
        rows.append(rec)
    import pandas as pd
    return pd.DataFrame(rows)


def main():
    import pandas as pd
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", default="")
    ap.add_argument("--verify", action="store_true",
                    help="train일 때 기존 snu_clip_features.csv와 scene_cuts 일치율 대조")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    df = pd.read_csv(f"./snuaichallenge_data/{args.split}.csv")
    image_dir = os.path.join("./snuaichallenge_data", args.split)
    out = compute(df, image_dir, args.device)

    if args.verify and args.split == "train":
        ref = pd.read_csv("./snu_clip_features.csv")[["Id", "predicted_scene_cuts"]]
        m = out.merge(ref, on="Id", suffixes=("_new", "_ref"))
        agree = (m.predicted_scene_cuts_new == m.predicted_scene_cuts_ref).mean()
        print(f"\n=== 정합성 검증 (n={len(m)}) ===")
        print(f"scene_cuts 완전일치율: {agree * 100:.1f}%")
        print("신규 분포:", out.predicted_scene_cuts.value_counts().sort_index().to_dict())
        print("기존 분포:", ref.predicted_scene_cuts.value_counts().sort_index().to_dict())
        if agree < 0.85:
            print("⚠️ 일치율 낮음 — transformers/torch.hub CLIP 전처리 차이. "
                  "임계값 재튜닝 또는 팀원 CSV 사용 검토")

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"저장: {args.out} ({len(out)}행)")


if __name__ == "__main__":
    main()
