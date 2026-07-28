# -*- coding: utf-8 -*-
"""
gemma 성분축별 학습 사용 횟수(copy_0 ~ copy_4) 교차표 생성 스크립트.
"""
import os
import sys
import re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import structure_features as sf

SEQ_MARKERS = ("then", "after", "before", "next", "finally", "first", "later", "once", "subsequently")
SIM_MARKERS = ("while", "as", "during", "simultaneously", "meanwhile")

def bucket_events(n):
    if n <= 1: return "ev_0-1"
    if n == 2: return "ev_2"
    if n == 3: return "ev_3"
    return "ev_4+"

def bucket_subj(n):
    return {0: "subj_0", 1: "subj_1", 2: "subj_2"}.get(n, "subj_3+")

def bucket_mark(n):
    return {0: "mark_0", 1: "mark_1"}.get(n, "mark_2+")

def bucket_wc(n):
    if n <= 10: return "wc_<=10"
    if n <= 18: return "wc_11-18"
    if n <= 28: return "wc_19-28"
    return "wc_29+"

def marker_kind(markers_val):
    if markers_val is None or (isinstance(markers_val, float) and np.isnan(markers_val)):
        return "mk_none"
    txt = str(markers_val).lower()
    if txt == "[]" or txt == "nan" or not txt:
        return "mk_none"
    has_seq = any(k in txt for k in SEQ_MARKERS)
    has_sim = any(k in txt for k in SIM_MARKERS)
    if has_seq and has_sim: return "mk_mixed"
    if has_seq: return "mk_seq"
    if has_sim: return "mk_sim"
    return "mk_none"

def main():
    train = pd.read_csv("snuaichallenge_data/train.csv")
    hold = set(pd.read_csv("splits/holdout_300.csv")["Id"])
    universe = train[~train["Id"].isin(hold)].copy()
    
    augw = pd.read_csv("outputs/aug_weights_exp16.csv")
    augw_map = dict(zip(augw["Id"], augw["aug_mult"].astype(int)))
    universe["aug_mult"] = universe["Id"].map(lambda i: augw_map.get(i, 2))
    universe["total_copies"] = universe["aug_mult"]
    
    un = pd.read_csv("all_untrained_items_raw.csv")
    un_counts = un["id"].value_counts()
    universe["untrained_copies"] = universe["Id"].map(lambda i: int(un_counts.get(i, 0)))
    universe["untrained_copies"] = universe[["untrained_copies", "total_copies"]].min(axis=1)
    universe["trained_copies"] = universe["total_copies"] - universe["untrained_copies"]
    
    gem = sf.load_gemma_labels()
    types = sf.assign_types(gem)
    gem_u = universe.merge(gem, on="Id", how="left").merge(types[["Id", "stype"]], on="Id", how="left")
    
    gem_u["ev_bucket"] = gem_u["n_events_noncam"].map(bucket_events)
    gem_u["subj_bucket"] = gem_u["n_subj_noncam"].map(bucket_subj)
    gem_u["mark_bucket"] = gem_u["n_markers"].map(bucket_mark)
    gem_u["wc_bucket"] = gem_u["sentence"].map(lambda s: bucket_wc(len(str(s).split())))
    gem_u["axis_viewer"] = gem_u["viewer"].map({True: "viewerY", False: "viewerN"})
    gem_u["marker_kind"] = gem_u["markers"].map(marker_kind)

    axes = [
        ("4유형 (stype)", "stype"),
        ("사건 개수 (n_events)", "ev_bucket"),
        ("주어 개수 (n_subj)", "subj_bucket"),
        ("표지 개수 (n_markers)", "mark_bucket"),
        ("표지 종류 (marker_kind)", "marker_kind"),
        ("문장 길이 (word_count)", "wc_bucket")
    ]
    
    for title, col in axes:
        print(f"=== {title} ===")
        ct = pd.crosstab(gem_u[col], gem_u["trained_copies"], margins=True)
        print(ct)
        print()

if __name__ == "__main__":
    main()
