import pandas as pd
import os

WORKSPACE_DIR = "C:/Users/user/Desktop/서울대"
HOLDOUT_CSV = os.path.join(WORKSPACE_DIR, "splits/holdout_300.csv")
FEATURES_CSV = os.path.join(WORKSPACE_DIR, "snu_clip_features.csv")

def main():
    if os.path.exists(HOLDOUT_CSV):
        df_hold = pd.read_csv(HOLDOUT_CSV)
        h9 = df_hold[df_hold['Id'] == 'H9PMiH']
        print("=== Holdout entry for H9PMiH ===")
        print(h9)
    
    if os.path.exists(FEATURES_CSV):
        df_feat = pd.read_csv(FEATURES_CSV)
        h9_feat = df_feat[df_feat['Id'] == 'H9PMiH']
        print("\n=== Features entry for H9PMiH ===")
        print(h9_feat)

if __name__ == "__main__":
    main()
