# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import structure_features as sf

def main():
    # 1. Train Universe (train.csv - holdout_300.csv)
    train = pd.read_csv("snuaichallenge_data/train.csv")
    hold = set(pd.read_csv("splits/holdout_300.csv")["Id"])
    universe = train[~train["Id"].isin(hold)].copy()
    
    # 2. Gemma Types (assign_types: dense_camO, dense_camX, sparse_camO, sparse_camX)
    gem = sf.load_gemma_labels()
    types = sf.assign_types(gem)
    universe = universe.merge(types[["Id", "stype"]], on="Id", how="left")
    
    # 3. Augment Multipliers (outputs/aug_weights_exp16.csv)
    augw = pd.read_csv("outputs/aug_weights_exp16.csv")
    augw_map = dict(zip(augw["Id"], augw["aug_mult"].astype(int)))
    universe["aug_mult"] = universe["Id"].map(lambda i: augw_map.get(i, 2))
    universe["total_copies"] = universe["aug_mult"]
    
    # 4. Untrained items from all_untrained_items_raw.csv
    un = pd.read_csv("all_untrained_items_raw.csv")
    un_counts = un["id"].value_counts()
    
    universe["untrained_copies"] = universe["Id"].map(lambda i: int(un_counts.get(i, 0)))
    universe["untrained_copies"] = universe[["untrained_copies", "total_copies"]].min(axis=1)
    universe["trained_copies"] = universe["total_copies"] - universe["untrained_copies"]
    universe["is_fully_untrained"] = (universe["trained_copies"] == 0)
    
    # 5. Test Dataset (819) & Submissions (977 vs 1488)
    test_types = pd.read_csv("outputs/gemma_labels/test_types.csv")
    s977 = pd.read_csv("outputs/submissions/submission_exp16_sparsecam_0720_1315.csv")
    s1488 = pd.read_csv("outputs/submissions/submission_exp17_4b_reorder_0721_0708.csv")
    test_sub = s977.merge(s1488, on="Id", suffixes=("_977", "_1488")).merge(test_types[["Id", "stype"]], on="Id", how="left")
    test_sub["changed"] = (test_sub["Answer_977"] != test_sub["Answer_1488"]).astype(int)
    
    stypes = ["dense_camO", "dense_camX", "sparse_camO", "sparse_camX"]
    res = []
    
    tot_fu_ids = universe["is_fully_untrained"].sum()
    
    for st in stypes:
        g = universe[universe["stype"] == st]
        t_g = test_sub[test_sub["stype"] == st]
        
        n_ids = len(g)
        mult = g["aug_mult"].iloc[0] if len(g) > 0 else 2
        tot_copies = g["total_copies"].sum()
        untr_copies = g["untrained_copies"].sum()
        tr_copies = g["trained_copies"].sum()
        
        fu_ids = g["is_fully_untrained"].sum()
        fu_pct_within_type = (fu_ids / n_ids * 100) if n_ids > 0 else 0
        fu_share_of_837 = (fu_ids / tot_fu_ids * 100) if tot_fu_ids > 0 else 0
        
        c0 = (g["trained_copies"] == 0).sum()
        c1 = (g["trained_copies"] == 1).sum()
        c2 = (g["trained_copies"] == 2).sum()
        c3 = (g["trained_copies"] == 3).sum()
        c4 = (g["trained_copies"] == 4).sum()
        
        n_test = len(t_g)
        ch_test = t_g["changed"].sum()
        disag_pct = (ch_test / n_test * 100) if n_test > 0 else 0
        
        res.append({
            "stype": st,
            "aug_mult": mult,
            "n_ids": n_ids,
            "tot_copies": tot_copies,
            "untr_copies": untr_copies,
            "untr_copy_pct": round(untr_copies / tot_copies * 100, 2),
            "tr_copies": tr_copies,
            "fu_ids": fu_ids,
            "fu_pct_within_type": round(fu_pct_within_type, 2),
            "fu_share_of_837_pct": round(fu_share_of_837, 2),
            "copy_0": c0,
            "copy_1": c1,
            "copy_2": c2,
            "copy_3": c3,
            "copy_4": c4,
            "test_cnt": n_test,
            "test_changed": ch_test,
            "test_disag_pct": round(disag_pct, 2)
        })
    
    df_res = pd.DataFrame(res)
    print("=== 4개 stype 유형별 엄밀 수치 종합 검증표 ===")
    print(df_res.to_string(index=False))
    
    print("\n=== Totals Check ===")
    print(f"Train Universe Total IDs: {df_res['n_ids'].sum()} (Expected: 9235)")
    print(f"Total Copies: {df_res['tot_copies'].sum()}")
    print(f"Untrained Copies: {df_res['untr_copies'].sum()} (Raw file rows: {len(un)})")
    print(f"Fully Untrained IDs: {df_res['fu_ids'].sum()} (Expected: 837)")
    print(f"Share of 837 % Sum: {df_res['fu_share_of_837_pct'].sum():.2f}% (Expected: 100.00%)")
    print(f"Test Total: {df_res['test_cnt'].sum()} (Expected: 819)")
    print(f"Test Total Changed: {df_res['test_changed'].sum()} (Expected: 177)")

if __name__ == "__main__":
    main()
