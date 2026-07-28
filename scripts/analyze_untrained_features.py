# -*- coding: utf-8 -*-
"""
학습 안 된 데이터(완전미학습 837개 Id & 미학습 사본 8,182개)와 학습 완료된 데이터 간의
Gemma 문장성분적/어휘적 특성 차이 정밀 분석 스크립트.
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

def extract_first_verb(sent):
    if pd.isna(sent): return "noevent"
    toks = re.findall(r"[a-z]+", str(sent).lower())
    for t in toks[1:5]:
        if t not in ("the", "a", "an", "of", "in", "on", "to", "and", "is", "are", "was", "were"):
            return t
    return toks[0] if toks else "noevent"

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
    universe["is_fully_untrained"] = (universe["trained_copies"] == 0)
    
    gem = sf.load_gemma_labels()
    gem_u = universe.merge(gem, on="Id", how="left")
    
    gem_u["ev_bucket"] = gem_u["n_events_noncam"].map(bucket_events)
    gem_u["subj_bucket"] = gem_u["n_subj_noncam"].map(bucket_subj)
    gem_u["mark_bucket"] = gem_u["n_markers"].map(bucket_mark)
    gem_u["wc_bucket"] = gem_u["sentence"].map(lambda s: bucket_wc(len(str(s).split())))
    gem_u["axis_viewer"] = gem_u["viewer"].map({True: "viewerY", False: "viewerN"})
    gem_u["marker_kind"] = gem_u["markers"].map(marker_kind)
    gem_u["first_verb"] = gem_u["sentence"].map(extract_first_verb)

    fu = gem_u[gem_u["is_fully_untrained"]]
    tr = gem_u[~gem_u["is_fully_untrained"]]
    
    print("=== 완전미학습(837개 Id) vs 학습완료(8398개 Id) Gemma 성분 특성 비교 ===\n")
    
    axes = [
        ("시간표지 개수 (mark_bucket)", "mark_bucket"),
        ("시간표지 종류 (marker_kind)", "marker_kind"),
        ("사건 개수 (ev_bucket)", "ev_bucket"),
        ("주어 개수 (subj_bucket)", "subj_bucket"),
        ("문장 길이 (wc_bucket)", "wc_bucket"),
        ("카메라 시점 (axis_viewer)", "axis_viewer")
    ]
    
    for label, col in axes:
        fu_vc = fu[col].value_counts(normalize=True).mul(100).round(2)
        tr_vc = tr[col].value_counts(normalize=True).mul(100).round(2)
        all_vc = gem_u[col].value_counts(normalize=True).mul(100).round(2)
        
        df_comp = pd.DataFrame({
            "Fully_Untrained_% (837개)": fu_vc,
            "Trained_% (8398개)": tr_vc,
            "All_Universe_% (9235개)": all_vc
        }).fillna(0)
        df_comp["Diff_%p (FU - Tr)"] = (df_comp["Fully_Untrained_% (837개)"] - df_comp["Trained_% (8398개)"]).round(2)
        print(f"[{label}]")
        print(df_comp.to_string())
        print()

    print("=== [첫 동사 (first_verb) 상위 15개 특성 대조] ===")
    fv_fu = fu["first_verb"].value_counts().head(15)
    fv_tr = tr["first_verb"].value_counts(normalize=True).mul(100)
    fv_comp = pd.DataFrame({
        "FU_count (837개중)": fv_fu,
        "FU_%": (fv_fu / len(fu) * 100).round(2),
        "Tr_% (8398개중)": fv_tr[fv_fu.index].round(2)
    })
    fv_comp["Diff_%p"] = (fv_comp["FU_%"] - fv_comp["Tr_% (8398개중)"]).round(2)
    print(fv_comp.to_string())

if __name__ == "__main__":
    main()
