# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def main():
    sub_2b = pd.read_csv("outputs/submissions/submission_exp16_sparsecam_0720_1315.csv")
    sub_4b = pd.read_csv("outputs/submissions/submission_exp17_4b_reorder_0721_0708.csv")
    sub_8b_path = "outputs/submission.csv" if os.path.exists("outputs/submission.csv") else "outputs/submissions/submission_exp17_4b_reorder_0721_0708.csv"
    sub_8b = pd.read_csv(sub_8b_path)
    
    m = sub_2b.merge(sub_4b, on="Id", suffixes=("_2B", "_4B")).merge(sub_8b, on="Id")
    m.rename(columns={"Answer": "Answer_8B"}, inplace=True)
    
    m["same_2B_4B"] = (m["Answer_2B"] == m["Answer_4B"])
    m["same_4B_8B"] = (m["Answer_4B"] == m["Answer_8B"])
    m["same_2B_8B"] = (m["Answer_2B"] == m["Answer_8B"])
    m["same_all"] = (m["same_2B_4B"] & m["same_4B_8B"])
    
    tot = len(m)
    print("=== Test Set 819개 2B vs 4B vs 8B 예측 일치율 대조 ===")
    print(f"2B vs 4B 일치: {m['same_2B_4B'].sum()} / {tot} ({m['same_2B_4B'].mean()*100:.2f}%)")
    print(f"4B vs 8B 일치: {m['same_4B_8B'].sum()} / {tot} ({m['same_4B_8B'].mean()*100:.2f}%)")
    print(f"2B vs 8B 일치: {m['same_2B_8B'].sum()} / {tot} ({m['same_2B_8B'].mean()*100:.2f}%)")
    print(f"2B & 4B & 8B 3개 모두 일치: {m['same_all'].sum()} / {tot} ({m['same_all'].mean()*100:.2f}%)\n")
    
    # Holdout 300개 GT가 있는 경우 4B vs 8B 정답 누적 및 포함관계(Subset) 검증
    p_4b_h = "outputs/preds/Qwen3-VL-4B-Instruct_exp17_4b_reorder_sparseaug_v5_reorder.csv"
    if os.path.exists(p_4b_h):
        df_4b_h = pd.read_csv(p_4b_h)
        print("=== Holdout 300개 4B 모델 정답 검증 ===")
        print(f"4B acc_shuffled: {df_4b_h['correct'].mean()*100:.2f}% ({df_4b_h['correct'].sum()}/300)")

if __name__ == "__main__":
    main()
