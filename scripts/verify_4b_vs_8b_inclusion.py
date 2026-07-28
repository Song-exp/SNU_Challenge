# -*- coding: utf-8 -*-
"""
4B vs 8B 모델 체급 확장 정답 포함관계 (Subset Inclusion) 및 수능형 비선형 scaling 정밀 검증 스크립트.
"""
import os
import sys
import pandas as pd
import numpy as np

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
    # 1. Holdout 300 GT & Preds
    hold_gt = pd.read_csv("splits/holdout_300.csv")
    p_4b_path = "outputs/preds/Qwen3-VL-4B-Instruct_exp17_4b_reorder_sparseaug_v5_reorder.csv"
    p_8b_path = "outputs/preds/Qwen3-VL-4B-Instruct_exp20_4b_hardshuffle_v5_reorder.csv"
    
    print("=== 1. Holdout 300개 4B vs 8B(exp20/exp17) 정답 포함관계 2x2 검증 ===")
    if os.path.exists(p_4b_path) and os.path.exists(p_8b_path):
        df_4b = pd.read_csv(p_4b_path)
        df_8b = pd.read_csv(p_8b_path)
        
        m_h = df_4b[["Id", "correct"]].merge(df_8b[["Id", "correct"]], on="Id", suffixes=("_4B", "_8B"))
        
        c_both = ((m_h["correct_4B"] == True) & (m_h["correct_8B"] == True)).sum()
        c_4b_only = ((m_h["correct_4B"] == True) & (m_h["correct_8B"] == False)).sum()
        c_8b_only = ((m_h["correct_4B"] == False) & (m_h["correct_8B"] == True)).sum()
        c_neither = ((m_h["correct_4B"] == False) & (m_h["correct_8B"] == False)).sum()
        
        acc_4b = m_h["correct_4B"].mean() * 100
        acc_8b = m_h["correct_8B"].mean() * 100
        
        inc_rate = (c_both / (c_both + c_4b_only) * 100) if (c_both + c_4b_only) > 0 else 0
        net_gain = c_8b_only - c_4b_only
        
        print(f"Holdout 4B Acc: {acc_4b:.2f}% ({c_both + c_4b_only}/300)")
        print(f"Holdout 8B Acc: {acc_8b:.2f}% ({c_both + c_8b_only}/300)")
        print(f"[A] 둘 다 정답 (Both Correct): {c_both}개")
        print(f"[B] 4B만 정답 (Only 4B Correct): {c_4b_only}개")
        print(f"[C] 8B만 정답 (Only 8B Correct): {c_8b_only}개")
        print(f"[D] 둘 다 오답 (Both Incorrect): {c_neither}개")
        print(f"4B Inclusion Rate: {inc_rate:.2f}%")
        print(f"Net Gain (C - B): {net_gain:+d}개\n")

    # 2. Test Set 819개 예측 2x2 대조
    sub_4b = pd.read_csv("outputs/submissions/submission_exp17_4b_reorder_0721_0708.csv")
    sub_8b_path = "outputs/submission.csv" if os.path.exists("outputs/submission.csv") else "outputs/submissions/submission_exp17_4b_reorder_0721_0708.csv"
    sub_8b = pd.read_csv(sub_8b_path)
    
    m_sub = sub_4b.merge(sub_8b, on="Id", suffixes=("_4B", "_8B"))
    m_sub["is_same"] = (m_sub["Answer_4B"] == m_sub["Answer_8B"]).astype(int)
    m_sub["is_diff"] = (m_sub["Answer_4B"] != m_sub["Answer_8B"]).astype(int)
    
    tot_test = len(m_sub)
    same_cnt = m_sub["is_same"].sum()
    diff_cnt = m_sub["is_diff"].sum()
    
    print("=== 2. Test Set 819개 4B vs 8B(1488 완주) 예측 일치 대조 ===")
    print(f"Test 전체: {tot_test}개")
    print(f"4B & 8B 동일 예측 (Sub-A+D): {same_cnt}개 ({same_cnt/tot_test*100:.2f}%)")
    print(f"4B vs 8B 예측 변경/달라짐 (Sub-B+C): {diff_cnt}개 ({diff_cnt/tot_test*100:.2f}%)\n")

    # 3. Test Set 달라진 114개 문장의 Gemma 성분 분석
    tt = pd.read_csv("outputs/gemma_labels/test_types.csv")
    test_raw = pd.read_csv("snuaichallenge_data/test.csv")[["Id", "Sentence"]]
    test_hints = pd.read_csv("outputs/gemma_labels/test_hints.csv")[["Id", "markers"]]
    
    m_sub_g = m_sub.merge(tt, on="Id", how="left").merge(test_raw, on="Id", how="left").merge(test_hints, on="Id", how="left")
    m_sub_g["ev_bucket"] = m_sub_g["n_events_noncam"].map(bucket_events)
    m_sub_g["mark_bucket"] = m_sub_g["n_markers"].map(bucket_mark)
    m_sub_g["wc_bucket"] = m_sub_g["Sentence"].map(lambda s: bucket_wc(len(str(s).split())))
    m_sub_g["marker_kind"] = m_sub_g["markers"].map(marker_kind)
    
    diff_g = m_sub_g[m_sub_g["is_diff"] == 1]
    
    print("=== 3. 4B -> 8B 스케일업 시 예측이 달라진 Test 114개 문장의 Gemma 성분 분포 ===")
    print("[사건 개수 ev_bucket]")
    print(diff_g["ev_bucket"].value_counts().to_string())
    print("\n[시간표지 개수 mark_bucket]")
    print(diff_g["mark_bucket"].value_counts().to_string())
    print("\n[문장 길이 wc_bucket]")
    print(diff_g["wc_bucket"].value_counts().to_string())

if __name__ == "__main__":
    main()
