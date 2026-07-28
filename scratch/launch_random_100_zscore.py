import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np

WORKSPACE_DIR = "C:/Users/user/Desktop/서울대"
TRAIN_CSV = os.path.join(WORKSPACE_DIR, "train_검토_최종_완료.csv")
MODIFIED_CSV = os.path.join(WORKSPACE_DIR, "train_검토_최종_완료_수정본.csv")
FEATURES_CSV = os.path.join(WORKSPACE_DIR, "snu_clip_features.csv")
UNSUPERVISED_CSV = os.path.join(WORKSPACE_DIR, "eda/unsupervised_gui_results.csv")
LAUNCHER_BAT = os.path.join(WORKSPACE_DIR, "검수기프로그램/run_gui_inspector_100.bat")

def main():
    print("=== Z-Score based Union-Find Random 100 Validation ===")
    
    # 1. Back up original files
    print("Backing up original CSV files...")
    if os.path.exists(MODIFIED_CSV):
        shutil.copy(MODIFIED_CSV, MODIFIED_CSV + ".bak")
    if os.path.exists(UNSUPERVISED_CSV):
        shutil.copy(UNSUPERVISED_CSV, UNSUPERVISED_CSV + ".bak")
        
    try:
        # 2. Load dataset and select 100 random samples (excluding first 100 rows)
        print("Selecting 100 random samples (seed=100)...")
        df_all = pd.read_csv(TRAIN_CSV, encoding='cp949')
        
        # Exclude the first 100 rows
        df_pool = df_all.iloc[100:].copy()
        
        # Sample 100 rows
        df_sampled = df_pool.sample(n=100, random_state=100).copy()
        # Sort by original index to keep things orderly
        df_sampled = df_sampled.sort_index()
        
        # 3. Load CLIP features and merge
        df_feat = pd.read_csv(FEATURES_CSV)
        merged = pd.merge(df_sampled, df_feat, on='Id', how='left')
        
        # 4. Run Z-score based Union-Find scene grouping
        pairs = [('dist_12', (0, 1)), ('dist_13', (0, 2)), ('dist_14', (0, 3)),
                 ('dist_23', (1, 2)), ('dist_24', (1, 3)), ('dist_34', (2, 3))]
                 
        pred_cuts_list = []
        groups_list = []
        
        for idx, row in merged.iterrows():
            dists = [float(row[col]) for col, _ in pairs]
            mean_d = np.mean(dists)
            std_d = np.std(dists)
            
            # 1. Absolute Rule: if even the maximum distance is small, it's 0 cuts
            if max(dists) < 0.20:
                groups = [[0, 1, 2, 3]]
                pred_cuts = 0
            else:
                # 2. Otherwise: relative Z-score based Union-Find
                # Connect if Z-score < -0.75
                connected = []
                for col, nodes in pairs:
                    d = float(row[col])
                    z = (d - mean_d) / std_d if std_d > 1e-9 else 0.0
                    if z < -0.75:
                        connected.append(nodes)
                        
                # Union-Find
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
                        
                for u, v in connected:
                    union(u, v)
                    
                groups_dict = {}
                for i in range(4):
                    root = find(i)
                    if root not in groups_dict:
                        groups_dict[root] = []
                    groups_dict[root].append(i)
                    
                groups = list(groups_dict.values())
                pred_cuts = len(groups) - 1
            
            # Map groups to chronological order (1 to 4)
            ans = eval(row['Answer'])
            chrono_groups = []
            for g in groups:
                chrono_g = sorted([ans[i] for i in g])
                chrono_groups.append(chrono_g)
            chrono_groups.sort(key=lambda x: x[0])
            
            groups_str = " | ".join([f"{{{', '.join(map(str, cg))}}}" for cg in chrono_groups])
            
            pred_cuts_list.append(pred_cuts)
            groups_list.append(groups_str)
            
        # 5. Save the 100 sampled rows to train_검토_최종_완료_수정본.csv
        df_sampled.to_csv(MODIFIED_CSV, index=False, encoding='cp949')
        
        # 6. Save the 100 unsupervised results to unsupervised_gui_results.csv
        unsup_res = pd.DataFrame({
            'Id': df_sampled['Id'].astype(str),
            'unsupervised_cuts': pred_cuts_list,
            'unsupervised_groups': groups_list
        })
        os.makedirs(os.path.dirname(UNSUPERVISED_CSV), exist_ok=True)
        unsup_res.to_csv(UNSUPERVISED_CSV, index=False)
        
        # 7. Print stats
        print("\n=== Predicted Scene Transition Stats (Z-Score Union-Find) ===")
        val_counts = unsup_res['unsupervised_cuts'].value_counts().sort_index()
        for cuts, count in val_counts.items():
            print(f"장면 전환 {int(cuts)}회: {count}개")
            
        print("\nLaunching GUI inspector (100 random samples)...")
        # Run the batch file launcher synchronously
        subprocess.run([LAUNCHER_BAT], shell=True, cwd=os.path.join(WORKSPACE_DIR, "검수기프로그램"))
        
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # 8. Restore original files
        print("\nRestoring original CSV files from backup...")
        if os.path.exists(MODIFIED_CSV + ".bak"):
            if os.path.exists(MODIFIED_CSV):
                os.remove(MODIFIED_CSV)
            os.rename(MODIFIED_CSV + ".bak", MODIFIED_CSV)
        if os.path.exists(UNSUPERVISED_CSV + ".bak"):
            if os.path.exists(UNSUPERVISED_CSV):
                os.remove(UNSUPERVISED_CSV)
            os.rename(UNSUPERVISED_CSV + ".bak", UNSUPERVISED_CSV)
        print("Restore completed successfully.")

if __name__ == "__main__":
    main()
