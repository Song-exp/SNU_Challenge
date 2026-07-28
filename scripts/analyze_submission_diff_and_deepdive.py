# -*- coding: utf-8 -*-
"""
제출 비교 (977 vs 1488) 및 미학습 유형 Deep Dive 분석 스크립트.

1. 제출 977 (부분/중간) vs 1488 (완주) test 예측 불일치 분석:
   - 819개 test 문장에 대해 gemma 성분축별 불일치율(disagreement rate) 및 z-score 집계.
   - outputs/submission_diff_by_type.csv 저장.

2. 미학습 데이터 Deep Dive:
   - all_untrained_items_raw.csv (8,182개 미학습 사본, 837개 완전미학습 Id) 원인 및 특징 분석.
   - 증강 multiplier (x2 vs x4) 및 문장성분 복잡도(dense/sparse, 표지종류, 첫동사) 영향 규명.
   - 제출 불일치 유형과 미학습 데이터 커버리지의 인과 사슬 검증.
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

TEST_TYPES_CSV = "outputs/gemma_labels/test_types.csv"
TEST_FEATS_CSV = "outputs/gemma_labels/test_features.csv"
UNTRAINED_RAW = "all_untrained_items_raw.csv"
TRAIN_CSV = "snuaichallenge_data/train.csv"
HOLDOUT = "splits/holdout_300.csv"
AUGW = "outputs/aug_weights_exp16.csv"

# 제출 파일 경로 지정 (977: exp16/exp14 계열, 1488: exp17/exp20 완주 계열)
SUB_977_PATH = "outputs/submissions/submission_exp16_sparsecam_0720_1315.csv"
SUB_1488_PATH = "outputs/submissions/submission_exp17_4b_reorder_0721_0708.csv"

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
    if pd.isna(markers_val) or not markers_val:
        return "mk_none"
    txt = str(markers_val).lower()
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
    print("=== PART 1: 제출 977 vs 1488 Test 예측 불일치 성분축 분석 ===")
    s977 = pd.read_csv(SUB_977_PATH)
    s1488 = pd.read_csv(SUB_1488_PATH)
    
    m_sub = s977.merge(s1488, on="Id", suffixes=("_977", "_1488"))
    m_sub["changed"] = (m_sub["Answer_977"] != m_sub["Answer_1488"]).astype(int)
    
    total_test = len(m_sub)
    total_changed = m_sub["changed"].sum()
    p_global = total_changed / total_test
    print(f"전체 test ({total_test}개) 불일치: {total_changed}개 ({p_global*100:.2f}%)\n")

    # test dataset & gemma test features 병합
    tt = pd.read_csv(TEST_TYPES_CSV)
    test_raw = pd.read_csv("snuaichallenge_data/test.csv")[["Id", "Sentence"]]
    test_df = m_sub.merge(tt, on="Id", how="left").merge(test_raw, on="Id", how="left")
    
    gem_labels = sf.load_gemma_labels()
    test_df = test_df.merge(gem_labels[["Id", "markers"]], on="Id", how="left")
    
    test_df["ev_bucket"] = test_df["n_events_noncam"].map(bucket_events)
    test_df["subj_bucket"] = test_df["n_subj_noncam"].map(bucket_subj)
    test_df["mark_bucket"] = test_df["n_markers"].map(bucket_mark)
    test_df["wc_bucket"] = test_df["Sentence"].map(lambda s: bucket_wc(len(str(s).split())))
    test_df["axis_camera"] = test_df["camera"].map({True: "camO", False: "camX"})
    test_df["axis_viewer"] = test_df["viewer"].map({True: "viewerY", False: "viewerN"})
    test_df["axis_multisubj"] = (test_df["n_subj_noncam"] >= 2).map({True: "multi", False: "single"})
    test_df["marker_kind"] = test_df["markers"].map(marker_kind)
    test_df["first_verb"] = test_df["Sentence"].map(extract_first_verb)

    diff_rows = []

    def calc_axis_diff(axis_name, col):
        for val, g in test_df.groupby(col):
            n = len(g)
            ch = g["changed"].sum()
            rate = ch / n * 100
            exp = n * p_global
            std = np.sqrt(n * p_global * (1 - p_global))
            z = (ch - exp) / std if std > 0 else 0.0
            diff_rows.append({
                "axis": axis_name,
                "bucket": val,
                "n_test": n,
                "changed_cnt": ch,
                "disagreement_pct": round(rate, 2),
                "z_score": round(z, 2)
            })

    axes = [
        ("stype", "stype"),
        ("camera", "axis_camera"),
        ("viewer", "axis_viewer"),
        ("n_events", "ev_bucket"),
        ("n_subj", "subj_bucket"),
        ("n_markers", "mark_bucket"),
        ("marker_kind", "marker_kind"),
        ("multisubj", "axis_multisubj"),
        ("word_count", "wc_bucket")
    ]

    for name, col in axes:
        calc_axis_diff(name, col)

    df_diff = pd.DataFrame(diff_rows).sort_values("z_score", key=abs, ascending=False)
    out_diff_path = "outputs/submission_diff_by_type.csv"
    df_diff.to_csv(out_diff_path, index=False, encoding="utf-8-sig")
    print(f"제출 불일치 성분축 분석 결과 저장: {out_diff_path}\n")
    print("--- [상위 불일치 편차 성분축] ---")
    print(df_diff.head(15).to_string(index=False))

    print("\n\n=== PART 2: 미학습 데이터 (8,182개 사본 & 837개 완전미학습 Id) Deep Dive ===")
    train = pd.read_csv(TRAIN_CSV)
    hold = set(pd.read_csv(HOLDOUT)["Id"])
    universe = train[~train["Id"].isin(hold)].copy()
    
    gem = sf.load_gemma_labels()
    gem_u = universe[["Id"]].merge(gem, on="Id", how="left")
    
    augw = pd.read_csv(AUGW)
    augw_map = dict(zip(augw["Id"], augw["aug_mult"].astype(int)))
    gem_u["total_copies"] = gem_u["Id"].map(lambda i: augw_map.get(i, 2))

    un = pd.read_csv(UNTRAINED_RAW)
    un_counts = un["id"].value_counts()
    gem_u["untrained_copies"] = gem_u["Id"].map(lambda i: int(un_counts.get(i, 0)))
    gem_u["untrained_copies"] = gem_u[["untrained_copies", "total_copies"]].min(axis=1)
    gem_u["trained_copies"] = gem_u["total_copies"] - gem_u["untrained_copies"]
    gem_u["is_fully_untrained"] = gem_u["trained_copies"] == 0

    types = sf.assign_types(gem)
    gem_u = gem_u.merge(types[["Id", "stype"]], on="Id", how="left")

    print("\n--- 1. 유형별 증강 Multiplier(x2 vs x4) 및 완전미학습 비율 분석 ---")
    mult_summary = gem_u.groupby(["stype", "total_copies"]).agg(
        n_ids=("Id", "count"),
        fully_untrained_ids=("is_fully_untrained", "sum"),
        tot_copies=("total_copies", "sum"),
        untr_copies=("untrained_copies", "sum")
    ).reset_index()
    mult_summary["fully_untr_pct"] = (mult_summary["fully_untrained_ids"] / mult_summary["n_ids"] * 100).round(2)
    mult_summary["untr_copy_pct"] = (mult_summary["untr_copies"] / mult_summary["tot_copies"] * 100).round(2)
    print(mult_summary.to_string(index=False))

    print("\n--- 2. 완전미학습(837개) 문장성분 특성 (vs 전체 학습 유니버스) ---")
    fu_df = gem_u[gem_u["is_fully_untrained"]]
    print(f"완전미학습 총 Id 수: {len(fu_df)}개")
    print("완전미학습 stype 분포:")
    print(fu_df["stype"].value_counts(normalize=True).mul(100).round(2).to_string())

    print("\n--- 3. 인과 사슬 검증: 미학습 편중(dense) vs Test 불일치(977->1488 변경) ---")
    stype_diff = test_df.groupby("stype")["changed"].agg(["count", "sum", "mean"]).reset_index()
    stype_diff.columns = ["stype", "test_cnt", "changed_cnt", "changed_rate"]
    stype_diff["changed_rate"] = (stype_diff["changed_rate"] * 100).round(2)
    
    stype_untr = gem_u.groupby("stype").agg(
        total_ids=("Id", "count"),
        fully_untr_ids=("is_fully_untrained", "sum"),
        tot_copies=("total_copies", "sum"),
        untr_copies=("untrained_copies", "sum")
    ).reset_index()
    stype_untr["fully_untr_pct"] = (stype_untr["fully_untr_ids"] / stype_untr["total_ids"] * 100).round(2)
    
    causal_df = stype_diff.merge(stype_untr, on="stype")
    print(causal_df[["stype", "test_cnt", "changed_cnt", "changed_rate", "total_ids", "fully_untr_ids", "fully_untr_pct"]].to_string(index=False))

if __name__ == "__main__":
    main()
