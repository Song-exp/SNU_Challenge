import os
import pandas as pd
import numpy as np

WORKSPACE_DIR = "C:/Users/user/Desktop/서울대"
TRAIN_CSV = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/train.csv")
FEATURES_CSV = os.path.join(WORKSPACE_DIR, "snu_clip_features.csv")
OUTPUT_CSV = os.path.join(WORKSPACE_DIR, "eda/raw_train_scene_cuts.csv")

def main():
    print("=== Extracting with the EXACT ORIGINAL 0.20 Counting Formula ===")
    df_train = pd.read_csv(TRAIN_CSV)
    df_feat = pd.read_csv(FEATURES_CSV)
    
    merged = pd.merge(df_train, df_feat, on='Id', how='left')
    
    predicted_cuts_list = []
    
    for idx, row in merged.iterrows():
        if pd.isna(row['dist_12']):
            predicted_cuts_list.append(np.nan)
            continue
            
        # Extract the 6 CLIP distances
        clip_arr = np.array([
            float(row['dist_12']),
            float(row['dist_13']),
            float(row['dist_14']),
            float(row['dist_23']),
            float(row['dist_24']),
            float(row['dist_34'])
        ])
        
        # Exact original counting formula from video_feature_extractor.py
        similar_clip_pairs = sum([1 for c in clip_arr if c < 0.20])
        if similar_clip_pairs >= 5:
            predicted_scene_cuts = 0
        elif 2 <= similar_clip_pairs <= 4:
            predicted_scene_cuts = 1
        elif similar_clip_pairs == 1:
            predicted_scene_cuts = 2
        else:
            predicted_scene_cuts = 3
            
        predicted_cuts_list.append(predicted_scene_cuts)
        
    df_train['predicted_scene_cuts'] = predicted_cuts_list
    
    # Save output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_train.to_csv(OUTPUT_CSV, index=False)
    print(f"Original cuts prediction saved successfully to {OUTPUT_CSV}")
    
    # Verify match with train_검토_최종_완료.csv
    orig_db_path = os.path.join(WORKSPACE_DIR, "train_검토_최종_완료.csv")
    if os.path.exists(orig_db_path):
        df_orig = pd.read_csv(orig_db_path, encoding='cp949')
        # Rename or find column
        col_name = [c for c in df_orig.columns if '장면' in c and '전환' in c and '횟수' in c]
        if col_name:
            c_name = col_name[0]
            merged_verify = pd.merge(df_train, df_orig[['Id', c_name]], on='Id', how='left')
            mismatches = merged_verify[merged_verify['predicted_scene_cuts'] != merged_verify[c_name]]
            print(f"Verification: Number of mismatches with train_검토_최종_완료.csv: {len(mismatches)}")
            
    # Stats
    print("\n=== Predicted Scene Transition Statistics (Exact Original) ===")
    stats = df_train['predicted_scene_cuts'].value_counts().sort_index()
    for cuts, count in stats.items():
        percentage = count / len(df_train) * 100
        print(f"장면 전환 {int(cuts)}회: {count}개 ({percentage:.2f}%)")

if __name__ == "__main__":
    main()
