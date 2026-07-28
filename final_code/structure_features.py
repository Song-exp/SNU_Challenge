# ================================================================================
# SNU AI Challenge — Compiling Feature Tables and Generating Augment weights
# ================================================================================

import os
import re
import pandas as pd

CAM_RE = re.compile(
    r"\b(camera|pans?|panning|zooms?|zooming|cuts?|cutting|the (scene|view|shot|frame|screen)"
    r"|view (shifts?|changes?)|close-?up|focus(es)? on|angle|footage|transitions?|frame)\b", 
    re.IGNORECASE
)

CLIP_TRAIN_PATH = "snu_clip_features.csv"
CLIP_SIM_THRESHOLD = 0.20

def camera_regex(sentence):
    """
    Checks if camera or discourse cues are present in the text description.
    """
    return bool(CAM_RE.search(str(sentence)))

def load_clip_features(path=CLIP_TRAIN_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CLIP features not found at: {path}")
    return pd.read_csv(path)

def build_feature_table(sentences_df):
    """
    Compiles text properties and scene cuts into a unified feature table.
    """
    out = sentences_df[["Id", "Sentence"]].copy()
    out["camera_re"] = out["Sentence"].map(camera_regex)
    
    if os.path.exists(CLIP_TRAIN_PATH):
        clip = load_clip_features()
        dcols = ["dist_12", "dist_13", "dist_14", "dist_23", "dist_24", "dist_34"]
        clip["n_similar"] = (clip[dcols] < CLIP_SIM_THRESHOLD).sum(axis=1)
        out = out.merge(
            clip[["Id", "Max_clip_scaled", "n_similar"]].rename(
                columns={"Max_clip_scaled": "clip_max"}
            ), 
            on="Id", 
            how="left"
        )
    return out

def make_aug_weights(feature_df, rules, default_mult, out_path):
    """
    Applies logic rules to generate aug_weights_exp16.csv containing sample-level weights.
    """
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
    print(f"Augment weights saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    train_df = pd.read_csv("train.csv")
    ft = build_feature_table(train_df)
    
    # Define rules: samples with camera details are augmented to ensure coverage
    # and offset position bias in static scenes.
    rules = [
        (lambda r: not r["camera_re"], 4),
    ]
    make_aug_weights(ft, rules, 2, "aug_weights_exp16.csv")
