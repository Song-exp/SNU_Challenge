import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Path configurations
WORKSPACE_DIR = "C:/Users/user/Desktop/서울대"
FEATURES_CSV = os.path.join(WORKSPACE_DIR, "snu_clip_features.csv")
TRAIN_CSV = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/train.csv")
OUTPUT_CSV = os.path.join(WORKSPACE_DIR, "eda/raw_train_scene_cuts.csv")

def main():
    print("Loading CLIP features dataset...")
    df_feat = pd.read_csv(FEATURES_CSV)
    
    # Calculate similarity metrics (s_ij = 1 - d_ij)
    dist_cols = ['dist_12', 'dist_13', 'dist_14', 'dist_23', 'dist_24', 'dist_34']
    s_df = 1.0 - df_feat[dist_cols]
    
    mean_s = s_df.mean(axis=1).values
    max_s = s_df.max(axis=1).values
    
    # Calculate level (mean similarity normalized by the stable anchor 0.985)
    levels = mean_s / 0.985
    
    # Calculate max gaps in similarity
    max_gaps = []
    for idx, row in s_df.iterrows():
        sorted_vals = sorted(row.values)
        diffs = np.diff(sorted_vals)
        max_gaps.append(max(diffs))
    max_gaps = np.array(max_gaps)
    
    print("\n=== Raw Statistical Distribution of Dataset (N=9,535) ===")
    print("--- Similarity Level (mean_s / 0.985) ---")
    levels_series = pd.Series(levels)
    print(levels_series.describe())
    
    print("\n--- Maximum Gap (Max Gap between adjacent similarities) ---")
    gaps_series = pd.Series(max_gaps)
    print(gaps_series.describe())
    
    # Unsupervised Thresholding using Statistical Percentiles (75th percentile of levels & median of gaps)
    level_threshold = 0.86
    gap_threshold = 0.08
    print("\n=== Unsupervised Threshold Calibration (Dataset Percentiles) ===")
    print(f"-> Selected Level Threshold (75th percentile): {level_threshold:.2f}")
    print(f"-> Selected Max Gap Threshold (50th percentile/Median): {gap_threshold:.2f}")
    
    # Load original train.csv (N=9,538)
    print("\nLoading original train.csv...")
    df_train = pd.read_csv(TRAIN_CSV)
    
    # Merge train and features
    merged = pd.merge(df_train, df_feat, on='Id', how='left')
    
    predicted_cuts_list = []
    groups_list = []
    
    # Apply the calibrated thresholds to classify all samples
    print("Applying calibrated unsupervised grouping to all train samples...")
    for idx, row in merged.iterrows():
        sample_id = str(row['Id'])
        if pd.isna(row['dist_12']):
            predicted_cuts_list.append(np.nan)
            groups_list.append("")
            continue
            
        ans = eval(row['Answer'])
        
        # Get similarities
        s = {
            (0, 1): 1.0 - float(row['dist_12']),
            (0, 2): 1.0 - float(row['dist_13']),
            (0, 3): 1.0 - float(row['dist_14']),
            (1, 2): 1.0 - float(row['dist_23']),
            (1, 3): 1.0 - float(row['dist_24']),
            (2, 3): 1.0 - float(row['dist_34'])
        }
        
        s_arr = np.array(list(s.values()))
        mean_s_val = np.mean(s_arr)
        level_val = mean_s_val / 0.985
        max_s_val = np.max(s_arr)
        
        # Calculate max gap
        sorted_s = sorted(s.values(), reverse=True)
        max_gap_val = -1.0
        split_idx = -1
        for k in range(len(sorted_s) - 1):
            gap = sorted_s[k] - sorted_s[k+1]
            if gap > max_gap_val:
                max_gap_val = gap
                split_idx = k
                
        # 1. Level check (using calibrated level threshold)
        if level_val >= level_threshold:
            groups = [[0, 1, 2, 3]]
            pred_cuts = 0
        # 2. Check for 3 cuts (if even the most similar frames are very different)
        # We can set the threshold for 3 cuts at centers_level[0] (low similarity cluster center)
        elif max_s_val < 0.73:
            groups = [[0], [1], [2], [3]]
            pred_cuts = 3
        # 3. Check for 0 cuts due to uniform noise (max gap is below calibrated gap threshold)
        elif max_gap_val < gap_threshold:
            groups = [[0, 1, 2, 3]]
            pred_cuts = 0
        # 4. Otherwise: relative Gap split + Union-Find
        else:
            same_scene_pairs = [pair for pair, sim_val in sorted(s.items(), key=lambda x: x[1], reverse=True)[:split_idx + 1]]
            
            parent = list(range(4))
            def find(x):
                while parent[x] != x:
                    x = parent[x]
                return x
            def union(x, y):
                root_x = find(x)
                root_y = find(y)
                if root_x != root_y:
                    parent[root_y] = root_x
                    
            for (i, j) in same_scene_pairs:
                union(i, j)
                
            groups_dict = {}
            for idx_node in range(4):
                root = find(idx_node)
                if root not in groups_dict:
                    groups_dict[root] = []
                groups_dict[root].append(idx_node)
                
            groups = list(groups_dict.values())
            pred_cuts = len(groups) - 1
            
        # Map groups to chronological frame order (1 to 4)
        chrono_groups = []
        for g in groups:
            chrono_g = sorted([ans[idx_node] for idx_node in g])
            chrono_groups.append(chrono_g)
        chrono_groups.sort(key=lambda x: x[0])
        
        groups_str = " | ".join([f"{{{', '.join(map(str, cg))}}}" for cg in chrono_groups])
        
        predicted_cuts_list.append(pred_cuts)
        groups_list.append(groups_str)
        
    df_train['predicted_scene_cuts'] = predicted_cuts_list
    df_train['predicted_scene_groups'] = groups_list
    
    # Save output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_train.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPredictions saved to {OUTPUT_CSV}")
    
    # Scene transition statistics analysis
    print("\n=== Predicted Scene Transition Statistics on Raw train.csv ===")
    stats = df_train['predicted_scene_cuts'].value_counts().sort_index()
    for cuts, count in stats.items():
        percentage = count / len(df_train) * 100
        print(f"장면 전환 {int(cuts)}회: {count}개 ({percentage:.2f}%)")

if __name__ == "__main__":
    main()
