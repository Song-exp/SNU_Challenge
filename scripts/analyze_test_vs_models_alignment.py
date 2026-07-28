# -*- coding: utf-8 -*-
"""
Test set (819개) Gemma 성분축과 제출 977(중간 모델) vs 1488(풀학습 모델) 간의
일치(안정적 학습 성공 영역) / 불일치(977 시점 놓쳤다가 1488 완주로 보정된 영역) 정밀 대조 분석 스크립트.
"""
import os
import sys
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
    # 1. Test 제출 비교 (977 vs 1488)
    s977 = pd.read_csv("outputs/submissions/submission_exp16_sparsecam_0720_1315.csv")
    s1488 = pd.read_csv("outputs/submissions/submission_exp17_4b_reorder_0721_0708.csv")
    m_sub = s977.merge(s1488, on="Id", suffixes=("_977", "_1488"))
    m_sub["is_agreed"] = (m_sub["Answer_977"] == m_sub["Answer_1488"]).astype(int)
    m_sub["is_changed"] = (m_sub["Answer_977"] != m_sub["Answer_1488"]).astype(int)

    # 2. Test Gemma Types & Features & Hints
    tt = pd.read_csv("outputs/gemma_labels/test_types.csv")
    test_raw = pd.read_csv("snuaichallenge_data/test.csv")[["Id", "Sentence"]]
    test_hints = pd.read_csv("outputs/gemma_labels/test_hints.csv")[["Id", "markers"]]
    
    test_df = m_sub.merge(tt, on="Id", how="left").merge(test_raw, on="Id", how="left").merge(test_hints, on="Id", how="left")
    
    test_df["ev_bucket"] = test_df["n_events_noncam"].map(bucket_events)
    test_df["subj_bucket"] = test_df["n_subj_noncam"].map(bucket_subj)
    test_df["mark_bucket"] = test_df["n_markers"].map(bucket_mark)
    test_df["wc_bucket"] = test_df["Sentence"].map(lambda s: bucket_wc(len(str(s).split())))
    test_df["marker_kind"] = test_df["markers"].map(marker_kind)
    
    print("=== Test Set 819개 전체 제출 비교 요약 ===")
    tot = len(test_df)
    agreed = test_df["is_agreed"].sum()
    changed = test_df["is_changed"].sum()
    print(f"총 Test 문장: {tot}개")
    print(f"977 & 1488 동의(일치, 977 시점 이미 안정 학습): {agreed}개 ({agreed/tot*100:.2f}%)")
    print(f"977 -> 1488 불일치(변경, 977 시점 놓쳤다가 완주 보정): {changed}개 ({changed/tot*100:.2f}%)\n")

    axes = [
        ("4유형 (stype)", "stype"),
        ("사건 개수 (n_events)", "ev_bucket"),
        ("시간표지 개수 (n_markers)", "mark_bucket"),
        ("시간표지 종류 (marker_kind)", "marker_kind"),
        ("주어 개수 (n_subj)", "subj_bucket"),
        ("문장 길이 (word_count)", "wc_bucket")
    ]

    for label, col in axes:
        g_summary = test_df.groupby(col).agg(
            test_cnt=("Id", "count"),
            agreed_cnt=("is_agreed", "sum"),
            changed_cnt=("is_changed", "sum")
        ).reset_index()
        
        g_summary["agreed_%"] = (g_summary["agreed_cnt"] / g_summary["test_cnt"] * 100).round(2)
        g_summary["changed_%"] = (g_summary["changed_cnt"] / g_summary["test_cnt"] * 100).round(2)
        g_summary["test_share_%"] = (g_summary["test_cnt"] / tot * 100).round(2)
        
        g_summary = g_summary.sort_values("agreed_%", ascending=False)
        print(f"--- [{label}] ---")
        print(g_summary[["col" if "col" in g_summary else col, "test_cnt", "test_share_%", "agreed_cnt", "agreed_%", "changed_cnt", "changed_%"]].to_string(index=False))
        print()

if __name__ == "__main__":
    main()
