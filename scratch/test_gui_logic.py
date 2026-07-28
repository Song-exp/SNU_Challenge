import os
import ast
import numpy as np
import pandas as pd

CSV_PATH = "./train_검토_최종_완료_수정본.csv"
FEATURES_CSV = "./snu_clip_features.csv"

try:
    print("1. Loading CSV...")
    train_df = pd.read_csv(CSV_PATH, encoding='cp949')
    print(f"   Train CSV loaded, shape: {train_df.shape}")
    
    print("2. Merging features...")
    feat_df = pd.read_csv(FEATURES_CSV)
    max_col = 'Max_clip' if 'Max_clip' in feat_df.columns else 'Max'
    mean_col = 'Mean_clip' if 'Mean_clip' in feat_df.columns else 'Mean'
    temp_df = feat_df[['Id', max_col, mean_col]].copy()
    temp_df.rename(columns={max_col: 'Max_clip', mean_col: 'Mean_clip'}, inplace=True)
    merged_df = pd.merge(train_df, temp_df, on='Id', how='left')
    print(f"   Merged shape: {merged_df.shape}")
    
    print("3. Initializing columns...")
    for col in ['수정된 장면 전환 횟수', '수정된 고유 주어 개수', '수정된 서술어 개수', '수정된 Partition']:
        if col not in merged_df.columns:
            merged_df[col] = np.nan
    if '모호_여부' not in merged_df.columns:
        merged_df['모호_여부'] = False
    if '모호_이유' not in merged_df.columns:
        merged_df['모호_이유'] = ""
    if '검수_자' not in merged_df.columns:
        merged_df['검수_자'] = ""
    if '검수_완료' not in merged_df.columns:
        merged_df['검수_완료'] = False
        
    print("4. Mimicking edit on index 0...")
    current_idx = 0
    new_cuts = 2
    new_subjs = 1
    new_preds = 3
    new_part = "Type-2"
    new_ambig = False
    new_reason = "Test Reason"
    
    # Test setting
    merged_df.at[current_idx, '수정된 장면 전환 횟수'] = new_cuts
    merged_df.at[current_idx, '수정된 고유 주어 개수'] = new_subjs
    merged_df.at[current_idx, '수정된 서술어 개수'] = new_preds
    merged_df.at[current_idx, '수정된 Partition'] = new_part
    merged_df.at[current_idx, '모호_여부'] = new_ambig
    merged_df.at[current_idx, '모호_이유'] = new_reason
    
    merged_df.at[current_idx, '검수_자'] = "병철"
    merged_df.at[current_idx, '검수_완료'] = True
    
    print("5. Syncing to train_df...")
    for col in ['수정된 장면 전환 횟수', '수정된 고유 주어 개수', '수정된 서술어 개수', '수정된 Partition', '모호_여부', '모호_이유', '검수_자', '검수_완료']:
        train_df.at[current_idx, col] = merged_df.at[current_idx, col]
        
    print("6. Saving to file...")
    save_cols = ['Id', 'Input_1', 'Input_2', 'Input_3', 'Input_4', 'Sentence', 'Answer', 
                 'No_ordering', 'Partition', '서술어 개수', '서술어', '장면 전환 횟수', 
                 'Unique_Subj_Count', '고유 주어 개수', '고유 주어',
                 '수정된 장면 전환 횟수', '수정된 고유 주어 개수', '수정된 서술어 개수', '수정된 Partition',
                 '모호_여부', '모호_이유', '검수_자', '검수_완료']
                 
    train_df[save_cols].to_csv("./train_검토_최종_완료_수정본_test.csv", index=False, encoding='cp949')
    print("🟢 All logic completed successfully without error!")

except Exception as e:
    import traceback
    print("🚨 ERROR OCCURRED:")
    traceback.print_exc()
